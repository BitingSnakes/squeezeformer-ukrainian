from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import torch

from squeezeformer_pytorch.asr import SqueezeformerCTC
from squeezeformer_pytorch.model import (
    FP8_SHAPE_ALIGNMENT,
    squeezeformer_variant,
    transformer_engine_available,
)
from squeezeformer_pytorch.runtime_types import DecodeStrategy, DTypeChoice, OptimizerChoice

try:
    import transformer_engine.pytorch as te
except (ImportError, OSError):
    te = None

DTYPE_CHOICES = ("auto",) + tuple(choice.value for choice in DTypeChoice)
OPTIMIZER_CHOICES = tuple(choice.value for choice in OptimizerChoice)
TOKENIZER_CHOICES = ("sentencepiece", "character")
DECODE_STRATEGY_CHOICES = tuple(choice.value for choice in DecodeStrategy)
VARIANT_CHOICES = ("xs", "s", "sm", "m", "ml", "l")


@dataclass(frozen=True)
class DeviceProfile:
    device: str
    device_type: str
    memory_gb: float | None
    cpu_count: int


@dataclass(frozen=True)
class TrainingEstimate:
    variant: str
    distributed: bool
    world_size: int
    batch_size: int
    max_batch_frames: int
    gradient_accumulation_steps: int
    num_workers: int
    metadata_workers: int
    prefetch_factor: int
    beam_size: int
    pin_memory: bool
    persistent_workers: bool
    model_parameters: int
    model_parameters_millions: float
    reference_model_parameters: int
    parameter_scale: float
    available_memory_gb: float | None
    target_effective_frames: int
    estimated_effective_frames: int
    estimated_per_rank_effective_frames: int
    resolved_dtype: str
    resolved_compile: bool
    fp8_supported: bool
    fp8_support_reason: str | None
    compile_support_reason: str | None


def _validate_device_argument(device: str) -> str:
    try:
        torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"Invalid device '{device}': {error}") from error
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate hardware-sensitive train.py hyperparameters and emit a ready command."
    )
    parser.add_argument("--dataset-repo", default="speech-uk/cv22")
    parser.add_argument("--variant", default="sm", choices=VARIANT_CHOICES)
    parser.add_argument("--optimizer", default="muon", choices=OPTIMIZER_CHOICES)
    parser.add_argument("--tokenizer", default="sentencepiece", choices=TOKENIZER_CHOICES)
    parser.add_argument("--spm-vocab-size", type=int, default=4096)
    parser.add_argument("--device", type=_validate_device_argument, required=True)
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--dtype", default="auto", choices=DTYPE_CHOICES)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--speed-perturb-prob", type=float, default=0.0)
    parser.add_argument("--noise-prob", type=float, default=0.0)
    parser.add_argument("--reverb-prob", type=float, default=0.0)
    parser.add_argument(
        "--decode-strategy",
        default="greedy",
        choices=DECODE_STRATEGY_CHOICES,
    )
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--output-dir", default="artifacts/squeezeformer-cv22")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--base-batch-size", type=int, default=8)
    parser.add_argument("--base-max-batch-frames", type=int, default=12000)
    parser.add_argument("--base-gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--avg-frames-per-sample", type=int, default=1500)
    parser.add_argument(
        "--emit-format",
        default="both",
        choices=("shell", "json", "both"),
    )
    return parser.parse_args()


def _dtype_size_bytes(dtype: str) -> int:
    if dtype == "float32":
        return 4
    if dtype in {"float16", "bfloat16"}:
        return 2
    if dtype == "fp8":
        return 1
    raise ValueError(f"Unsupported dtype: {dtype}")


def _cpu_memory_gb() -> float | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(page_count, int):
        return None
    return (page_size * page_count) / float(1024**3)


def probe_device(device: str) -> DeviceProfile:
    cpu_count = os.cpu_count() or 1
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
        index = 0 if resolved.index is None else resolved.index
        props = torch.cuda.get_device_properties(index)
        memory_gb = props.total_memory / float(1024**3)
        return DeviceProfile(
            device=str(resolved),
            device_type="cuda",
            memory_gb=memory_gb,
            cpu_count=cpu_count,
        )
    if resolved.type == "cpu":
        return DeviceProfile(
            device=str(resolved),
            device_type="cpu",
            memory_gb=_cpu_memory_gb(),
            cpu_count=cpu_count,
        )
    if resolved.type == "mps":
        return DeviceProfile(
            device=str(resolved),
            device_type="mps",
            memory_gb=_cpu_memory_gb(),
            cpu_count=cpu_count,
        )
    raise ValueError(f"Unsupported device type for tuning: {resolved.type}")


def count_model_parameters(variant: str, vocab_size: int) -> int:
    model = SqueezeformerCTC(
        encoder_config=squeezeformer_variant(variant),
        vocab_size=vocab_size,
    )
    return sum(parameter.numel() for parameter in model.parameters())


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def _fp8_support_status(device: str, variant: str) -> tuple[bool, str | None]:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return False, "FP8 requires a CUDA device."
    if not torch.cuda.is_available():
        return False, "CUDA was requested, but torch.cuda.is_available() is false."
    if not transformer_engine_available() or te is None:
        return False, "transformer-engine is unavailable in this runtime."
    config = squeezeformer_variant(variant)
    if config.d_model % FP8_SHAPE_ALIGNMENT != 0:
        return (
            False,
            f"Variant {variant} is not FP8-compatible; d_model must be divisible by {FP8_SHAPE_ALIGNMENT}.",
        )
    if hasattr(te, "is_fp8_available"):
        availability = te.is_fp8_available()
        if isinstance(availability, tuple):
            is_available, reason = availability
        else:
            is_available, reason = bool(availability), None
        if not is_available:
            return False, reason or "Transformer Engine reports FP8 is unavailable."
    return True, None


def _compile_support_status(device: str) -> tuple[bool, str | None]:
    return True, None


def _resolve_training_dtype(args: argparse.Namespace) -> tuple[str, bool, str | None]:
    fp8_supported, fp8_reason = _fp8_support_status(args.device, args.variant)
    requested_dtype = args.dtype
    if requested_dtype == "auto":
        if fp8_supported:
            return "fp8", fp8_supported, fp8_reason
        if torch.device(args.device).type == "cuda":
            if torch.cuda.is_bf16_supported():
                return "bfloat16", fp8_supported, fp8_reason
            return "float16", fp8_supported, fp8_reason
        return "bfloat16", fp8_supported, fp8_reason
    if requested_dtype == "fp8" and not fp8_supported:
        reason = fp8_reason or "FP8 is unavailable on this runtime."
        raise ValueError(f"Requested dtype fp8 is not supported. {reason}")
    return requested_dtype, fp8_supported, fp8_reason


def estimate_training_hparams(args: argparse.Namespace) -> TrainingEstimate:
    if args.distributed and args.nproc_per_node < 2:
        raise ValueError("--distributed requires --nproc-per-node >= 2.")
    profile = probe_device(args.device)
    resolved_dtype, fp8_supported, fp8_support_reason = _resolve_training_dtype(args)
    compile_supported, compile_support_reason = _compile_support_status(args.device)
    world_size = args.nproc_per_node if args.distributed else 1
    resolved_compile = args.compile and compile_supported and not args.distributed
    vocab_size = args.spm_vocab_size if args.tokenizer == "sentencepiece" else 256
    model_parameters = count_model_parameters(args.variant, vocab_size=vocab_size)
    model_parameters_millions = model_parameters / 1_000_000.0

    reference_model_parameters = count_model_parameters("sm", vocab_size=128)
    parameter_scale = max(model_parameters, 1) / max(reference_model_parameters, 1)
    variant_factor = math.sqrt(parameter_scale)
    dtype_factor = 4.0 / _dtype_size_bytes(resolved_dtype)
    augmentation_factor = 1.0 - min(
        0.35,
        0.20 * args.speed_perturb_prob + 0.08 * args.noise_prob + 0.06 * args.reverb_prob,
    )
    decode_factor = (
        0.75 if args.decode_strategy == "beam" and profile.device_type != "cuda" else 1.0
    )

    memory_gb = profile.memory_gb or (16.0 if profile.device_type == "cpu" else 8.0)
    if profile.device_type == "cpu":
        memory_factor = math.sqrt(max(memory_gb, 4.0) / 16.0)
        frame_budget = (
            args.base_max_batch_frames
            * memory_factor
            * dtype_factor
            * augmentation_factor
            * decode_factor
            / variant_factor
        )
        max_batch_frames = min(32000, max(4000, _round_to_multiple(frame_budget, 500)))
        target_effective_frames = max(
            args.base_max_batch_frames * args.base_gradient_accumulation_steps,
            48000,
        )
        batch_cap = 16
        num_workers = min(8, max(1, profile.cpu_count // 2))
        metadata_workers = min(8, max(1, profile.cpu_count))
        pin_memory = False
    elif profile.device_type == "cuda":
        memory_factor = max(memory_gb, 4.0) / 8.0
        frame_budget = (
            args.base_max_batch_frames
            * memory_factor
            * dtype_factor
            * augmentation_factor
            / variant_factor
        )
        max_batch_frames = min(128000, max(6000, _round_to_multiple(frame_budget, 1000)))
        target_effective_frames = max(
            args.base_max_batch_frames * args.base_gradient_accumulation_steps * 2,
            96000,
        )
        batch_cap = 48
        num_workers = min(8, max(2, profile.cpu_count // 2))
        metadata_workers = min(8, max(2, profile.cpu_count // 2))
        pin_memory = True
    else:
        memory_factor = math.sqrt(max(memory_gb, 8.0) / 16.0)
        frame_budget = (
            args.base_max_batch_frames
            * 0.8
            * memory_factor
            * dtype_factor
            * augmentation_factor
            / variant_factor
        )
        max_batch_frames = min(48000, max(4000, _round_to_multiple(frame_budget, 500)))
        target_effective_frames = max(
            args.base_max_batch_frames * args.base_gradient_accumulation_steps,
            64000,
        )
        batch_cap = 24
        num_workers = min(6, max(1, profile.cpu_count // 2))
        metadata_workers = min(6, max(1, profile.cpu_count // 2))
        pin_memory = False

    batch_size = max(1, min(batch_cap, max_batch_frames // max(1, args.avg_frames_per_sample)))
    per_rank_effective_frames = max_batch_frames
    gradient_accumulation_steps = max(
        1,
        math.ceil(target_effective_frames / max(1, per_rank_effective_frames * world_size)),
    )
    prefetch_factor = 2 if num_workers > 0 else 0
    persistent_workers = num_workers > 0

    if args.decode_strategy == "beam":
        if profile.device_type == "cpu":
            beam_size = min(args.beam_size, 4 if args.variant in {"sm", "m", "ml", "l"} else 6)
        else:
            beam_size = args.beam_size
    else:
        beam_size = args.beam_size

    return TrainingEstimate(
        variant=args.variant,
        distributed=args.distributed,
        world_size=world_size,
        batch_size=batch_size,
        max_batch_frames=max_batch_frames,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_workers=num_workers,
        metadata_workers=metadata_workers,
        prefetch_factor=prefetch_factor,
        beam_size=beam_size,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        model_parameters=model_parameters,
        model_parameters_millions=model_parameters_millions,
        reference_model_parameters=reference_model_parameters,
        parameter_scale=parameter_scale,
        available_memory_gb=profile.memory_gb,
        target_effective_frames=target_effective_frames,
        estimated_effective_frames=per_rank_effective_frames
        * gradient_accumulation_steps
        * world_size,
        estimated_per_rank_effective_frames=per_rank_effective_frames * gradient_accumulation_steps,
        resolved_dtype=resolved_dtype,
        resolved_compile=resolved_compile,
        fp8_supported=fp8_supported,
        fp8_support_reason=fp8_support_reason,
        compile_support_reason=compile_support_reason,
    )


def build_train_command(args: argparse.Namespace, estimate: TrainingEstimate) -> str:
    dataset_repo = getattr(args, "dataset_repo", "speech-uk/cv22")
    command = [
        (
            f"uv run torchrun --nproc_per_node={estimate.world_size} train.py"
            if estimate.distributed
            else "uv run python train.py"
        ),
    ]
    if estimate.distributed:
        command.append("--distributed")
    command.extend(
        [
            f"--dataset-repo {dataset_repo}",
            f"--variant {args.variant}",
            f"--optimizer {args.optimizer}",
            f"--tokenizer {args.tokenizer}",
        ]
    )
    if args.tokenizer == "sentencepiece":
        command.append(f"--spm-vocab-size {args.spm_vocab_size}")
    if args.feature_cache_dir is not None:
        command.append(f"--feature-cache-dir {args.feature_cache_dir}")
    command.extend(
        [
            f"--device {args.device}",
            f"--dtype {estimate.resolved_dtype}",
            "--compile" if estimate.resolved_compile else "--no-compile",
            f"--gradient-accumulation-steps {estimate.gradient_accumulation_steps}",
            f"--max-batch-frames {estimate.max_batch_frames}",
            f"--speed-perturb-prob {args.speed_perturb_prob}",
            f"--noise-prob {args.noise_prob}",
            f"--reverb-prob {args.reverb_prob}",
            f"--decode-strategy {args.decode_strategy}",
            f"--beam-size {estimate.beam_size}",
            f"--output-dir {args.output_dir}",
            f"--batch-size {estimate.batch_size}",
            f"--epochs {args.epochs}",
            f"--num-workers {estimate.num_workers}",
            f"--metadata-workers {estimate.metadata_workers}",
            f"--prefetch-factor {estimate.prefetch_factor}",
            "--pin-memory" if estimate.pin_memory else "--no-pin-memory",
            "--persistent-workers" if estimate.persistent_workers else "--no-persistent-workers",
        ]
    )
    return " \\\n  ".join(command)


def main() -> None:
    args = parse_args()
    estimate = estimate_training_hparams(args)
    payload = {
        "inputs": {
            "variant": args.variant,
            "optimizer": args.optimizer,
            "tokenizer": args.tokenizer,
            "spm_vocab_size": args.spm_vocab_size,
            "device": args.device,
            "distributed": args.distributed,
            "nproc_per_node": args.nproc_per_node,
            "dtype": args.dtype,
            "resolved_dtype": estimate.resolved_dtype,
            "compile": args.compile,
            "resolved_compile": estimate.resolved_compile,
            "decode_strategy": args.decode_strategy,
            "requested_beam_size": args.beam_size,
            "output_dir": args.output_dir,
            "feature_cache_dir": args.feature_cache_dir,
            "epochs": args.epochs,
        },
        "estimate": asdict(estimate),
        "command": build_train_command(args, estimate),
    }

    if args.emit_format in {"shell", "both"}:
        print(payload["command"])
    if args.emit_format == "both":
        print()
    if args.emit_format in {"json", "both"}:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
