from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import NamedTuple

import torch
import torch.distributed as dist
import trackio
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from squeezeformer_pytorch.asr import (
    CharacterTokenizer,
    SentencePieceTokenizer,
    SqueezeformerCTC,
    Tokenizer,
    ctc_prefix_beam_search,
    load_lm_scorer,
    tokenizer_from_dict,
)
from squeezeformer_pytorch.checkpoints import save_checkpoint
from squeezeformer_pytorch.data import (
    AudioFeaturizer,
    CV22ASRDataset,
    SpecAugment,
    WaveformAugment,
    create_dataloader,
    download_cv22_dataset,
    load_cv22_records,
    prevalidate_records,
)
from squeezeformer_pytorch.lm import NGramLanguageModel
from squeezeformer_pytorch.metrics import char_error_rate, word_error_rate
from squeezeformer_pytorch.model import (
    FP8_SHAPE_ALIGNMENT,
    SqueezeformerConfig,
    squeezeformer_variant,
    transformer_engine_available,
)
from squeezeformer_pytorch.runtime_types import (
    AdaptiveBatchUnit,
    DecodeStrategy,
    DTypeChoice,
    OptimizerChoice,
)

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
except ImportError:
    te = None
    DelayedScaling = None
    Format = None


def _checkpoint_name(epoch: int, val_wer: float) -> str:
    return f"checkpoint_epoch={epoch:04d}_valwer={val_wer:.6f}.pt"


def _safetensors_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".safetensors")


def _inference_checkpoint_payload(checkpoint: dict[str, object]) -> dict[str, object]:
    return {
        "model_state_dict": checkpoint["model_state_dict"],
        "encoder_config": checkpoint["encoder_config"],
        "tokenizer": checkpoint["tokenizer"],
        "featurizer_config": checkpoint.get("featurizer_config", {}),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "best_val_wer": checkpoint.get("best_val_wer"),
        "metrics": checkpoint.get("metrics"),
        "training_args": checkpoint.get("training_args", {}),
        "averaged_from": checkpoint.get("averaged_from"),
    }


def _export_inference_checkpoint(checkpoint: dict[str, object], checkpoint_path: Path) -> Path:
    safetensors_path = _safetensors_path(checkpoint_path)
    save_checkpoint(_inference_checkpoint_payload(checkpoint), safetensors_path)
    return safetensors_path


class SchedulerDefaults(NamedTuple):
    peak_lr: float
    num_time_masks: int


def _configure_console_logger(rank: int, is_main_process: bool) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO if is_main_process else logging.WARNING)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | rank=%(rank)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logging.LoggerAdapter(logger, {"rank": rank})


def _parse_xla_device_argument(device: str) -> int | None:
    normalized = device.strip().lower()
    if normalized in {"xla", "tpu"}:
        return None
    for prefix in ("xla:", "tpu:"):
        if normalized.startswith(prefix):
            suffix = normalized[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
            break
    raise argparse.ArgumentTypeError(
        f"Invalid device '{device}': TPU devices must be 'xla', 'xla:N', 'tpu', or 'tpu:N'."
    )


def _is_xla_device_argument(device: str) -> bool:
    normalized = device.strip().lower()
    return normalized == "xla" or normalized == "tpu" or normalized.startswith(("xla:", "tpu:"))


def _validate_device_argument(device: str) -> str:
    if _is_xla_device_argument(device):
        _parse_xla_device_argument(device)
        return device
    try:
        torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"Invalid device '{device}': {error}") from error
    return device


def _require_xla_runtime() -> None:
    if xm is None:
        raise RuntimeError(
            "TPU/XLA support requires torch_xla. Install a matching torch_xla build and rerun."
        )


def resolve_device(device: str) -> torch.device:
    if _is_xla_device_argument(device):
        _require_xla_runtime()
        index = _parse_xla_device_argument(device)
        return xm.xla_device(n=index)
    return torch.device(device)


def _validate_device_ready(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested with --device, but torch.cuda.is_available() is false."
        )
    if device.type == "xla":
        _require_xla_runtime()


def _is_xla_device(device: torch.device) -> bool:
    return device.type == "xla"


def _variant_defaults(variant: str) -> SchedulerDefaults:
    if variant in {"xs", "s", "sm"}:
        return SchedulerDefaults(peak_lr=2e-3, num_time_masks=5)
    if variant == "m":
        return SchedulerDefaults(peak_lr=1.5e-3, num_time_masks=7)
    return SchedulerDefaults(peak_lr=1e-3, num_time_masks=10)


def build_paper_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    warmup_epochs: int = 20,
    hold_epochs: int = 160,
    decay_exponent: float = 1.0,
):
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    hold_steps = max(0, hold_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        current_step = step + 1
        if current_step < warmup_steps:
            return current_step / warmup_steps
        if current_step < warmup_steps + hold_steps:
            return 1.0
        decay_step = max(1, current_step - hold_steps)
        return (warmup_steps**decay_exponent) / (decay_step**decay_exponent)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_optimizer(
    model: SqueezeformerCTC,
    optimizer_name: OptimizerChoice,
    muon_lr: float,
    adamw_lr: float,
    muon_weight_decay: float,
    adamw_weight_decay: float,
) -> tuple[list[torch.optim.Optimizer], list[str]]:
    if optimizer_name == OptimizerChoice.ADAMW:
        decay_params = []
        no_decay_params = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim < 2 or any(
                token in name.lower() for token in ("bias", "norm", "scale")
            ):
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)
        return [
            torch.optim.AdamW(
                [
                    {"params": decay_params, "weight_decay": adamw_weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=adamw_lr,
            )
        ], ["adamw"]

    muon_params = []
    adamw_decay_params = []
    adamw_no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder.") and parameter.ndim == 2:
            muon_params.append(parameter)
        elif parameter.ndim < 2 or any(
            token in name.lower() for token in ("bias", "norm", "scale")
        ):
            adamw_no_decay_params.append(parameter)
        else:
            adamw_decay_params.append(parameter)

    optimizers: list[torch.optim.Optimizer] = []
    optimizer_names: list[str] = []
    if muon_params:
        optimizers.append(
            torch.optim.Muon(
                muon_params,
                lr=muon_lr,
                weight_decay=muon_weight_decay,
                adjust_lr_fn="match_rms_adamw",
            )
        )
        optimizer_names.append("muon")
    if adamw_decay_params or adamw_no_decay_params:
        optimizers.append(
            torch.optim.AdamW(
                [
                    {"params": adamw_decay_params, "weight_decay": adamw_weight_decay},
                    {"params": adamw_no_decay_params, "weight_decay": 0.0},
                ],
                lr=adamw_lr,
            )
        )
        optimizer_names.append("adamw_aux")
    return optimizers, optimizer_names


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float, warmup_steps: int = 0) -> None:
        self.target_decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def _shadow_for(self, name: str, parameter: Tensor) -> Tensor:
        shadow = self.shadow[name]
        if shadow.device != parameter.device or shadow.dtype != parameter.dtype:
            shadow = shadow.to(device=parameter.device, dtype=parameter.dtype)
            self.shadow[name] = shadow
        return shadow

    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = self.current_decay()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    shadow = self._shadow_for(name, parameter)
                    shadow.mul_(decay).add_(parameter.detach(), alpha=1 - decay)

    def current_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.target_decay
        progress = min(1.0, self.num_updates / self.warmup_steps)
        return self.target_decay * progress

    def state_dict(self) -> dict[str, object]:
        return {
            "target_decay": self.target_decay,
            "warmup_steps": self.warmup_steps,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.target_decay = float(state_dict.get("target_decay", state_dict.get("decay", 0.0)))
        self.warmup_steps = int(state_dict.get("warmup_steps", 0))
        self.num_updates = int(state_dict.get("num_updates", 0))
        self.shadow = {name: tensor.clone() for name, tensor in state_dict["shadow"].items()}

    def apply_to(self, model: nn.Module) -> dict[str, Tensor]:
        backup: dict[str, Tensor] = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                backup[name] = parameter.detach().clone()
                parameter.data.copy_(self._shadow_for(name, parameter))
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: dict[str, Tensor]) -> None:
        for name, parameter in model.named_parameters():
            if name in backup:
                parameter.data.copy_(
                    backup[name].to(device=parameter.device, dtype=parameter.dtype)
                )


def _resolve_autocast_dtype(dtype: DTypeChoice) -> torch.dtype | None:
    if dtype == DTypeChoice.FLOAT32:
        return None
    if dtype == DTypeChoice.FLOAT16:
        return torch.float16
    if dtype == DTypeChoice.BFLOAT16:
        return torch.bfloat16
    if dtype == DTypeChoice.FP8:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _resolve_fp8_format(name: str):
    if Format is None:
        raise RuntimeError("transformer-engine is required for FP8 training.")
    normalized = name.strip().lower()
    if normalized == "hybrid":
        return Format.HYBRID
    if normalized == "e4m3":
        return Format.E4M3
    raise ValueError(f"Unsupported FP8 format: {name}")


def _build_fp8_recipe(args: argparse.Namespace):
    if args.dtype != DTypeChoice.FP8:
        return None
    if DelayedScaling is None:
        raise RuntimeError("transformer-engine is required for FP8 training.")
    return DelayedScaling(
        fp8_format=_resolve_fp8_format(args.fp8_format),
        amax_history_len=args.fp8_amax_history_len,
        amax_compute_algo=args.fp8_amax_compute_algo,
    )


def _validate_fp8_runtime(
    device: torch.device,
    encoder_config: SqueezeformerConfig,
) -> None:
    if device.type != "cuda":
        raise ValueError("FP8 training requires a CUDA device.")
    if not transformer_engine_available() or te is None:
        raise RuntimeError(
            "FP8 training requires transformer-engine. Install the package and CUDA extension."
        )
    if encoder_config.d_model % FP8_SHAPE_ALIGNMENT != 0:
        raise ValueError(
            "FP8 training requires d_model to be divisible by "
            f"{FP8_SHAPE_ALIGNMENT}; choose variant xs, sm, ml, or l."
        )
    if hasattr(te, "is_fp8_available"):
        availability = te.is_fp8_available()
        if isinstance(availability, tuple):
            is_available, reason = availability
        else:
            is_available, reason = bool(availability), None
        if not is_available:
            suffix = f" {reason}" if reason else ""
            raise RuntimeError(
                f"Transformer Engine reports FP8 is unavailable on this runtime.{suffix}"
            )


def _autocast_context(
    device: torch.device,
    dtype: DTypeChoice,
    *,
    fp8_recipe=None,
):
    if dtype == DTypeChoice.FP8:
        if device.type != "cuda":
            raise ValueError("FP8 autocast is only supported on CUDA.")
        if te is None:
            raise RuntimeError("transformer-engine is required for FP8 autocast.")
        stack = ExitStack()
        stack.enter_context(torch.autocast(device_type=device.type, dtype=torch.bfloat16))
        stack.enter_context(te.autocast(enabled=True, recipe=fp8_recipe))
        return stack
    autocast_dtype = _resolve_autocast_dtype(dtype)
    if autocast_dtype is None:
        return nullcontext()
    if device.type == "cpu" and autocast_dtype == torch.float16:
        raise ValueError("float16 autocast is not supported on CPU. Use bfloat16 or float32.")
    if device.type == "xla" and autocast_dtype == torch.float16:
        raise ValueError("float16 autocast is not supported on TPU/XLA. Use bfloat16 or float32.")
    return torch.autocast(device_type=device.type, dtype=autocast_dtype)


def _mark_xla_step(device: torch.device) -> None:
    if _is_xla_device(device):
        _require_xla_runtime()
        xm.mark_step()


def _update_top_checkpoints(
    output_dir: Path,
    checkpoint: dict[str, object],
    epoch: int,
    val_wer: float,
    keep_top_k: int,
) -> None:
    topk_dir = output_dir / "checkpoints_topk"
    topk_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = topk_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = []

    filename = _checkpoint_name(epoch=epoch, val_wer=val_wer)
    checkpoint_path = topk_dir / filename
    save_checkpoint(checkpoint, checkpoint_path)

    metadata.append(
        {
            "epoch": epoch,
            "val_wer": val_wer,
            "path": str(checkpoint_path.name),
        }
    )
    metadata.sort(key=lambda item: (float(item["val_wer"]), int(item["epoch"])))

    removed = metadata[keep_top_k:]
    metadata = metadata[:keep_top_k]
    for item in removed:
        stale_path = topk_dir / str(item["path"])
        if stale_path.exists():
            stale_path.unlink()

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _average_topk_checkpoints(output_dir: Path) -> Path | None:
    metadata_path = output_dir / "checkpoints_topk" / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata:
        return None

    averaged_state: dict[str, Tensor] | None = None
    template_checkpoint: dict[str, object] | None = None
    template_shapes: dict[str, tuple[int, ...]] | None = None
    included_metadata: list[dict[str, object]] = []
    for item in metadata:
        checkpoint_path = output_dir / "checkpoints_topk" / str(item["path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        if averaged_state is None:
            averaged_state = {
                key: value.detach().clone().to(dtype=torch.float32)
                for key, value in state_dict.items()
            }
            template_checkpoint = checkpoint
            template_shapes = {key: tuple(value.shape) for key, value in state_dict.items()}
            included_metadata.append(item)
        else:
            current_shapes = {key: tuple(value.shape) for key, value in state_dict.items()}
            if current_shapes != template_shapes:
                logging.getLogger("train").warning(
                    "Skipping checkpoint %s during top-k averaging because parameter shapes do not match the template checkpoint.",
                    checkpoint_path,
                )
                continue
            for key, value in state_dict.items():
                averaged_state[key].add_(value.detach().to(dtype=torch.float32))
            included_metadata.append(item)

    assert averaged_state is not None
    assert template_checkpoint is not None
    factor = 1.0 / len(included_metadata)
    model_state_dict = {}
    for key, value in template_checkpoint["model_state_dict"].items():
        averaged = averaged_state[key] * factor
        model_state_dict[key] = averaged.to(dtype=value.dtype)
    averaged_checkpoint = dict(template_checkpoint)
    averaged_checkpoint["model_state_dict"] = model_state_dict
    averaged_checkpoint["averaged_from"] = included_metadata
    averaged_path = output_dir / "checkpoint_topk_avg.pt"
    save_checkpoint(averaged_checkpoint, averaged_path)
    _export_inference_checkpoint(averaged_checkpoint, averaged_path)
    return averaged_path


def _build_split_audit(split_records: dict[str, list]) -> dict[str, object]:
    speaker_sets = {
        split_name: {
            record.speaker_id for record in records if record.has_speaker_id and record.speaker_id
        }
        for split_name, records in split_records.items()
    }
    counts = {
        split_name: {
            "samples": len(records),
            "speakers": len(speaker_sets[split_name]),
            "records_with_speaker_id": sum(int(record.has_speaker_id) for record in records),
        }
        for split_name, records in split_records.items()
    }
    overlaps = {}
    split_names = list(split_records)
    for index, split_name in enumerate(split_names):
        for other_name in split_names[index + 1 :]:
            key = f"{split_name}_vs_{other_name}"
            overlaps[key] = len(speaker_sets[split_name] & speaker_sets[other_name])
    speaker_counts = [item["speakers"] for item in counts.values() if item["speakers"] > 0]
    balance_ratio = (
        max(speaker_counts) / max(1, min(speaker_counts)) if len(speaker_counts) >= 2 else 1.0
    )
    return {
        "counts": counts,
        "speaker_overlaps": overlaps,
        "speaker_balance_ratio": balance_ratio,
        "speaker_id_available": all(
            item["records_with_speaker_id"] == item["samples"] for item in counts.values()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Squeezeformer CTC on speech-uk/cv22.")
    parser.add_argument("--dataset-repo", default="speech-uk/cv22")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--variant", default="sm", choices=["xs", "s", "sm", "m", "ml", "l"])
    parser.add_argument("--output-dir", default="artifacts/squeezeformer-cv22")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--muon-learning-rate", type=float, default=None)
    parser.add_argument("--adamw-learning-rate", type=float, default=None)
    parser.add_argument("--muon-weight-decay", type=float, default=None)
    parser.add_argument("--adamw-weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--min-transcript-chars", type=int, default=1)
    parser.add_argument("--max-transcript-chars", type=int, default=400)
    parser.add_argument("--max-symbol-ratio", type=float, default=0.5)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--max-batch-frames", type=int, default=None)
    parser.add_argument(
        "--adaptive-batch-unit",
        type=AdaptiveBatchUnit,
        choices=list(AdaptiveBatchUnit),
        default=None,
    )
    parser.add_argument("--adaptive-batch-budget", type=int, default=None)
    parser.add_argument(
        "--bucket-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--metadata-workers", type=int, default=4)
    parser.add_argument(
        "--prevalidate-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prevalidate-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=_validate_device_argument,
        required=True,
        help="Execution device, for example 'cpu', 'cuda', or 'cuda:0'.",
    )
    parser.add_argument(
        "--optimizer",
        type=OptimizerChoice,
        choices=list(OptimizerChoice),
        default=OptimizerChoice.MUON,
    )
    parser.add_argument(
        "--dtype",
        type=DTypeChoice,
        choices=list(DTypeChoice),
        default=DTypeChoice.BFLOAT16,
    )
    parser.add_argument(
        "--fp8-format",
        default="hybrid",
        choices=["hybrid", "e4m3"],
    )
    parser.add_argument("--fp8-amax-history-len", type=int, default=16)
    parser.add_argument(
        "--fp8-amax-compute-algo",
        default="max",
        choices=["max", "most_recent"],
    )
    parser.add_argument("--trackio-project", default="squeezeformer-cv22")
    parser.add_argument("--trackio-space-id", default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--keep-top-k", type=int, default=5)
    parser.add_argument(
        "--tokenizer",
        default="sentencepiece",
        choices=["character", "sentencepiece"],
    )
    parser.add_argument("--spm-vocab-size", type=int, default=128)
    parser.add_argument(
        "--spm-model-type",
        default="unigram",
        choices=["unigram", "bpe", "char", "word"],
    )
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--hold-epochs", type=int, default=160)
    parser.add_argument("--decay-exponent", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-warmup-steps", type=int, default=0)
    parser.add_argument("--muon-warmup-epochs", type=int, default=None)
    parser.add_argument("--muon-hold-epochs", type=int, default=None)
    parser.add_argument("--muon-decay-exponent", type=float, default=None)
    parser.add_argument("--adamw-warmup-epochs", type=int, default=None)
    parser.add_argument("--adamw-hold-epochs", type=int, default=None)
    parser.add_argument("--adamw-decay-exponent", type=float, default=None)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--attention-backend",
        default="flash",
        choices=["relative", "flash"],
        help=(
            "Attention implementation. 'flash' uses PyTorch scaled_dot_product_attention, "
            "which dispatches to FlashAttention kernels on supported CUDA setups."
        ),
    )
    parser.add_argument("--block-pattern", default="M,s,C,s")
    parser.add_argument(
        "--frontend-backend",
        default="audioflux",
        choices=["torchaudio", "audioflux"],
    )
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--preemphasis", type=float, default=0.97)
    parser.add_argument(
        "--normalize-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize-feature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize-per-frame",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--num-freq-masks", type=int, default=2)
    parser.add_argument("--freq-mask-param", type=int, default=27)
    parser.add_argument("--num-time-masks", type=int, default=None)
    parser.add_argument("--time-mask-max-ratio", type=float, default=0.05)
    parser.add_argument("--speed-perturb-prob", type=float, default=0.0)
    parser.add_argument("--speed-factors", default="0.9,1.0,1.1")
    parser.add_argument("--noise-prob", type=float, default=0.0)
    parser.add_argument("--noise-snr-db-min", type=float, default=10.0)
    parser.add_argument("--noise-snr-db-max", type=float, default=30.0)
    parser.add_argument("--reverb-prob", type=float, default=0.0)
    parser.add_argument("--reverb-decay-min", type=float, default=0.15)
    parser.add_argument("--reverb-decay-max", type=float, default=0.5)
    parser.add_argument("--reverb-delay-ms-min", type=float, default=8.0)
    parser.add_argument("--reverb-delay-ms-max", type=float, default=35.0)
    parser.add_argument(
        "--decode-strategy",
        type=DecodeStrategy,
        choices=list(DecodeStrategy),
        default=DecodeStrategy.GREEDY,
    )
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--lm-scorer", default=None)
    parser.add_argument("--lm-weight", type=float, default=0.0)
    parser.add_argument(
        "--fit-shallow-fusion-lm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--shallow-fusion-lm-order", type=int, default=3)
    parser.add_argument("--shallow-fusion-lm-alpha", type=float, default=0.1)
    parser.add_argument("--example-limit", type=int, default=5)
    return parser.parse_args()


def greedy_decode(log_probs: torch.Tensor, tokenizer: Tokenizer) -> list[str]:
    token_ids = log_probs.argmax(dim=-1).cpu().tolist()
    return [tokenizer.decode_ctc(sequence) for sequence in token_ids]


def decode_batch(
    log_probs: torch.Tensor,
    tokenizer: Tokenizer,
    strategy: DecodeStrategy,
    beam_size: int = 8,
    lm_scorer=None,
    lm_weight: float = 0.0,
) -> list[str]:
    if strategy == DecodeStrategy.GREEDY:
        return greedy_decode(log_probs, tokenizer)
    return [
        ctc_prefix_beam_search(
            sequence.cpu(),
            tokenizer=tokenizer,
            beam_size=beam_size,
            lm_scorer=lm_scorer,
            lm_weight=lm_weight,
        )
        for sequence in log_probs
    ]


def _bucket_name(reference: str) -> str:
    word_count = len(reference.split())
    if word_count <= 5:
        return "short"
    if word_count <= 15:
        return "medium"
    return "long"


def length_bucket_metrics(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, float]:
    grouped_refs: dict[str, list[str]] = defaultdict(list)
    grouped_hyps: dict[str, list[str]] = defaultdict(list)
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        bucket = _bucket_name(reference)
        grouped_refs[bucket].append(reference)
        grouped_hyps[bucket].append(hypothesis)

    metrics: dict[str, float] = {}
    for bucket in ("short", "medium", "long"):
        metrics[f"samples_{bucket}"] = float(len(grouped_refs[bucket]))
        if grouped_refs[bucket]:
            metrics[f"wer_{bucket}"] = word_error_rate(grouped_refs[bucket], grouped_hyps[bucket])
            metrics[f"cer_{bucket}"] = char_error_rate(grouped_refs[bucket], grouped_hyps[bucket])
    return metrics


def collect_examples(
    utterance_ids: list[str],
    speaker_ids: list[str | None],
    references: list[str],
    hypotheses: list[str],
    limit: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    hardest_examples = []
    random_examples = []
    pairs = list(zip(utterance_ids, speaker_ids, references, hypotheses, strict=True))
    ranked_pairs = sorted(
        pairs,
        key=lambda item: (item[2] == item[3], abs(len(item[2]) - len(item[3]))),
    )
    for utterance_id, speaker_id, reference, hypothesis in ranked_pairs[:limit]:
        hardest_examples.append(
            {
                "utterance_id": utterance_id,
                "speaker_id": speaker_id or "",
                "reference": reference,
                "hypothesis": hypothesis,
            }
        )
    shuffled_pairs = list(pairs)
    random.Random(13).shuffle(shuffled_pairs)
    for utterance_id, speaker_id, reference, hypothesis in shuffled_pairs[:limit]:
        random_examples.append(
            {
                "utterance_id": utterance_id,
                "speaker_id": speaker_id or "",
                "reference": reference,
                "hypothesis": hypothesis,
            }
        )
    return hardest_examples, random_examples


def speaker_level_metrics(
    speaker_ids: list[str | None],
    has_speaker_ids: list[bool],
    references: list[str],
    hypotheses: list[str],
) -> dict[str, object]:
    grouped_refs: dict[str, list[str]] = defaultdict(list)
    grouped_hyps: dict[str, list[str]] = defaultdict(list)
    missing_count = 0
    for speaker_id, has_speaker_id, reference, hypothesis in zip(
        speaker_ids, has_speaker_ids, references, hypotheses, strict=True
    ):
        if not has_speaker_id or not speaker_id:
            missing_count += 1
            continue
        grouped_refs[speaker_id].append(reference)
        grouped_hyps[speaker_id].append(hypothesis)
    per_speaker = {
        speaker_id: {
            "samples": len(grouped_refs[speaker_id]),
            "wer": word_error_rate(grouped_refs[speaker_id], grouped_hyps[speaker_id]),
            "cer": char_error_rate(grouped_refs[speaker_id], grouped_hyps[speaker_id]),
        }
        for speaker_id in grouped_refs
    }
    macro_wer = (
        sum(item["wer"] for item in per_speaker.values()) / len(per_speaker) if per_speaker else 0.0
    )
    return {
        "speaker_count": len(per_speaker),
        "speaker_macro_wer": macro_wer,
        "speaker_id_available": missing_count == 0,
        "missing_speaker_id_samples": missing_count,
        "per_speaker": per_speaker,
    }


def evaluate(
    model: SqueezeformerCTC,
    dataloader,
    criterion: nn.CTCLoss,
    tokenizer: Tokenizer,
    device: torch.device,
    dtype: DTypeChoice,
    fp8_recipe=None,
    decode_strategy: DecodeStrategy = DecodeStrategy.GREEDY,
    beam_size: int = 8,
    lm_scorer=None,
    lm_weight: float = 0.0,
    example_limit: int = 5,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    references: list[str] = []
    hypotheses: list[str] = []
    utterance_ids: list[str] = []
    speaker_ids: list[str | None] = []
    has_speaker_ids: list[bool] = []
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            with _autocast_context(device, dtype, fp8_recipe=fp8_recipe):
                log_probs, output_lengths = model.log_probs(features, feature_lengths)
                loss = criterion(log_probs.transpose(0, 1), targets, output_lengths, target_lengths)
            total_loss += float(loss.item())
            total_batches += 1
            references.extend(batch["transcripts"])
            utterance_ids.extend(batch["utterance_ids"])
            speaker_ids.extend(batch["speaker_ids"])
            has_speaker_ids.extend(batch["has_speaker_ids"])
            hypotheses.extend(
                decode_batch(
                    log_probs,
                    tokenizer=tokenizer,
                    strategy=decode_strategy,
                    beam_size=beam_size,
                    lm_scorer=lm_scorer,
                    lm_weight=lm_weight,
                )
            )
            _mark_xla_step(device)

    metrics = {
        "loss": total_loss / max(1, total_batches),
        "cer": char_error_rate(references, hypotheses),
        "wer": word_error_rate(references, hypotheses),
    }
    metrics.update(length_bucket_metrics(references, hypotheses))
    speaker_metrics = speaker_level_metrics(speaker_ids, has_speaker_ids, references, hypotheses)
    metrics["speaker_count"] = float(speaker_metrics["speaker_count"])
    metrics["speaker_macro_wer"] = float(speaker_metrics["speaker_macro_wer"])
    metrics["speaker_id_available"] = float(speaker_metrics["speaker_id_available"])
    metrics["missing_speaker_id_samples"] = float(speaker_metrics["missing_speaker_id_samples"])
    hardest_examples, random_examples = collect_examples(
        utterance_ids,
        speaker_ids,
        references,
        hypotheses,
        limit=example_limit,
    )
    return {
        "metrics": metrics,
        "hardest_examples": hardest_examples,
        "random_examples": random_examples,
        "speaker_metrics": speaker_metrics,
    }


def _resolve_block_pattern(block_pattern: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in block_pattern.split(",") if token.strip())
    if not tokens or any(token not in {"M", "C", "s"} for token in tokens):
        raise ValueError("block pattern must be a comma-separated sequence drawn from M,C,s")
    return tokens


def _resolve_float_tuple(values: str) -> tuple[float, ...]:
    parsed = tuple(float(value.strip()) for value in values.split(",") if value.strip())
    if not parsed:
        raise ValueError("expected at least one float value")
    return parsed


def _resolve_scheduler_kwargs(args: argparse.Namespace, optimizer_name: str) -> dict[str, float]:
    if optimizer_name == "muon":
        return {
            "warmup_epochs": (
                args.muon_warmup_epochs
                if args.muon_warmup_epochs is not None
                else args.warmup_epochs
            ),
            "hold_epochs": (
                args.muon_hold_epochs if args.muon_hold_epochs is not None else args.hold_epochs
            ),
            "decay_exponent": (
                args.muon_decay_exponent
                if args.muon_decay_exponent is not None
                else args.decay_exponent
            ),
        }
    return {
        "warmup_epochs": (
            args.adamw_warmup_epochs if args.adamw_warmup_epochs is not None else args.warmup_epochs
        ),
        "hold_epochs": (
            args.adamw_hold_epochs if args.adamw_hold_epochs is not None else args.hold_epochs
        ),
        "decay_exponent": (
            args.adamw_decay_exponent
            if args.adamw_decay_exponent is not None
            else args.decay_exponent
        ),
    }


def _flatten_examples(prefix: str, examples: list[dict[str, str]]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for index, example in enumerate(examples):
        payload[f"{prefix}_example_{index}_id"] = example["utterance_id"]
        payload[f"{prefix}_example_{index}_speaker"] = example["speaker_id"]
        payload[f"{prefix}_example_{index}_ref"] = example["reference"]
        payload[f"{prefix}_example_{index}_hyp"] = example["hypothesis"]
    return payload


def _build_checkpoint(
    model: SqueezeformerCTC,
    encoder_config: SqueezeformerConfig,
    tokenizer: Tokenizer,
    featurizer: AudioFeaturizer,
    epoch: int,
    global_step: int,
    best_val_wer: float,
    metrics: dict[str, float],
    optimizers: list[torch.optim.Optimizer],
    optimizer_names: list[str],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    scaler: torch.amp.GradScaler,
    ema: ExponentialMovingAverage | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "encoder_config": asdict(encoder_config),
        "tokenizer": tokenizer.to_dict(),
        "featurizer_config": featurizer.config_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_wer": best_val_wer,
        "metrics": metrics,
        "optimizer_names": optimizer_names,
        "optimizer_state_dicts": [optimizer.state_dict() for optimizer in optimizers],
        "scheduler_state_dicts": [scheduler.state_dict() for scheduler in schedulers],
        "scaler_state_dict": scaler.state_dict(),
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "training_args": vars(args),
    }


def main() -> None:
    args = parse_args()
    if (args.adaptive_batch_unit is None) != (args.adaptive_batch_budget is None):
        raise ValueError("--adaptive-batch-unit and --adaptive-batch-budget must be set together.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if args.distributed and not distributed:
        raise ValueError("--distributed expects a torchrun-style environment with WORLD_SIZE > 1.")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested_device = resolve_device(args.device)
    _validate_device_ready(requested_device)
    if distributed and _is_xla_device(requested_device):
        raise ValueError("TPU/XLA training does not support torchrun-style distributed mode here.")
    is_main_process = rank == 0
    logger = _configure_console_logger(rank=rank, is_main_process=is_main_process)
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
    if distributed and args.compile:
        raise ValueError("--compile is not currently supported together with distributed training.")
    if _is_xla_device(requested_device) and args.compile:
        raise ValueError("--compile is not currently supported on TPU/XLA.")
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "starting training variant=%s device=%s distributed=%s world_size=%s output_dir=%s",
        args.variant,
        requested_device,
        distributed,
        world_size,
        output_dir,
    )
    variant_defaults = _variant_defaults(args.variant)
    lm_scorer = load_lm_scorer(args.lm_scorer)

    dataset_root = download_cv22_dataset(
        repo_id=args.dataset_repo,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    train_records = load_cv22_records(
        dataset_root=dataset_root,
        split="train",
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        max_samples=args.max_train_samples,
        min_transcript_chars=args.min_transcript_chars,
        max_transcript_chars=args.max_transcript_chars,
        max_symbol_ratio=args.max_symbol_ratio,
    )
    val_records = load_cv22_records(
        dataset_root=dataset_root,
        split="validation",
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        max_samples=args.max_val_samples,
        min_transcript_chars=args.min_transcript_chars,
        max_transcript_chars=args.max_transcript_chars,
        max_symbol_ratio=args.max_symbol_ratio,
    )
    if args.prevalidate_audio:
        train_records = prevalidate_records(train_records, num_workers=args.prevalidate_workers)
        val_records = prevalidate_records(val_records, num_workers=args.prevalidate_workers)
        if not train_records or not val_records:
            raise RuntimeError("Audio prevalidation removed every sample from train or validation.")
    split_audit = _build_split_audit({"train": train_records, "validation": val_records})
    if is_main_process:
        (output_dir / "split_audit.json").write_text(
            json.dumps(split_audit, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "loaded dataset train_samples=%s val_samples=%s speaker_balance_ratio=%.3f",
            len(train_records),
            len(val_records),
            float(split_audit["speaker_balance_ratio"]),
        )

    checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    )
    if checkpoint is not None:
        tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    elif args.tokenizer == "sentencepiece":
        tokenizer = SentencePieceTokenizer.train(
            (record.transcript for record in train_records),
            model_prefix=output_dir / "tokenizer",
            vocab_size=args.spm_vocab_size,
            model_type=args.spm_model_type,
        )
        tokenizer.save(output_dir / "tokenizer.model")
    else:
        tokenizer = CharacterTokenizer.build(record.transcript for record in train_records)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    if args.fit_shallow_fusion_lm:
        shallow_fusion_lm = NGramLanguageModel.train(
            (record.transcript for record in train_records),
            order=args.shallow_fusion_lm_order,
            alpha=args.shallow_fusion_lm_alpha,
        )
        shallow_fusion_lm_path = output_dir / "shallow_fusion_lm.json"
        shallow_fusion_lm.save(shallow_fusion_lm_path)
        if lm_scorer is None:
            lm_scorer = shallow_fusion_lm.score_extension

    featurizer = AudioFeaturizer(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        backend=args.frontend_backend,
        preemphasis=args.preemphasis,
        normalize_signal=args.normalize_signal,
        normalize_feature=args.normalize_feature,
        normalize_per_frame=args.normalize_per_frame,
    )
    specaugment = SpecAugment(
        num_freq_masks=args.num_freq_masks,
        freq_mask_param=args.freq_mask_param,
        num_time_masks=args.num_time_masks or variant_defaults.num_time_masks,
        time_mask_max_ratio=args.time_mask_max_ratio,
    )
    waveform_augment = WaveformAugment(
        speed_perturb_prob=args.speed_perturb_prob,
        speed_factors=_resolve_float_tuple(args.speed_factors),
        noise_prob=args.noise_prob,
        noise_snr_db_range=(args.noise_snr_db_min, args.noise_snr_db_max),
        reverb_prob=args.reverb_prob,
        reverb_decay_range=(args.reverb_decay_min, args.reverb_decay_max),
        reverb_delay_ms_range=(args.reverb_delay_ms_min, args.reverb_delay_ms_max),
    )
    train_feature_cache_dir = (
        Path(args.feature_cache_dir) / "train" if args.feature_cache_dir is not None else None
    )
    val_feature_cache_dir = (
        Path(args.feature_cache_dir) / "validation" if args.feature_cache_dir is not None else None
    )
    local_train_records = (
        [record for index, record in enumerate(train_records) if index % world_size == rank]
        if distributed
        else train_records
    )
    train_dataset = CV22ASRDataset(
        local_train_records,
        tokenizer=tokenizer,
        featurizer=featurizer,
        specaugment=specaugment,
        waveform_augment=waveform_augment,
        feature_cache_dir=train_feature_cache_dir,
    )
    val_dataset = CV22ASRDataset(
        val_records,
        tokenizer=tokenizer,
        featurizer=featurizer,
        feature_cache_dir=val_feature_cache_dir,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        bucket_by_length=args.bucket_by_length,
        max_batch_frames=args.max_batch_frames,
        adaptive_batch_unit=args.adaptive_batch_unit,
        adaptive_batch_budget=args.adaptive_batch_budget,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        metadata_workers=args.metadata_workers,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        bucket_by_length=args.bucket_by_length,
        max_batch_frames=args.max_batch_frames,
        adaptive_batch_unit=args.adaptive_batch_unit,
        adaptive_batch_budget=args.adaptive_batch_budget,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        metadata_workers=args.metadata_workers,
    )

    encoder_config = (
        SqueezeformerConfig(**checkpoint["encoder_config"])
        if checkpoint is not None
        else squeezeformer_variant(args.variant)
    )
    if checkpoint is None:
        encoder_config = replace(
            deepcopy(encoder_config),
            block_pattern=_resolve_block_pattern(args.block_pattern),
            activation_checkpointing=args.activation_checkpointing,
            attention_backend=args.attention_backend,
        )
    fp8_recipe = _build_fp8_recipe(args)
    if distributed and requested_device.type == "cuda":
        if requested_device.index not in {None, local_rank}:
            raise ValueError(
                f"--device {args.device} conflicts with LOCAL_RANK={local_rank}. "
                "Use --device cuda or the matching cuda:<local_rank>."
            )
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = requested_device
    if args.dtype == DTypeChoice.FP8:
        _validate_fp8_runtime(device, encoder_config)
    model = SqueezeformerCTC(
        encoder_config=encoder_config,
        vocab_size=tokenizer.vocab_size,
        use_transformer_engine=args.dtype == DTypeChoice.FP8,
    )
    model.to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if distributed:
        forward_model: nn.Module = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )
    else:
        forward_model = torch.compile(model) if args.compile else model
    use_grad_scaler = device.type == "cuda" and args.dtype == DTypeChoice.FLOAT16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    peak_lr = args.learning_rate if args.learning_rate is not None else variant_defaults.peak_lr
    muon_lr = args.muon_learning_rate if args.muon_learning_rate is not None else peak_lr
    adamw_lr = args.adamw_learning_rate if args.adamw_learning_rate is not None else peak_lr
    muon_weight_decay = (
        args.muon_weight_decay if args.muon_weight_decay is not None else args.weight_decay
    )
    adamw_weight_decay = (
        args.adamw_weight_decay if args.adamw_weight_decay is not None else args.weight_decay
    )
    optimizer_steps_per_epoch = max(
        1,
        (len(train_loader) + args.gradient_accumulation_steps - 1)
        // args.gradient_accumulation_steps,
    )
    optimizers, optimizer_names = build_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        muon_weight_decay=muon_weight_decay,
        adamw_weight_decay=adamw_weight_decay,
    )
    schedulers = []
    for optimizer, optimizer_group_name in zip(optimizers, optimizer_names, strict=True):
        schedulers.append(
            build_paper_scheduler(
                optimizer,
                steps_per_epoch=optimizer_steps_per_epoch,
                **_resolve_scheduler_kwargs(args, optimizer_group_name),
            )
        )
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    ema = (
        ExponentialMovingAverage(
            model,
            decay=args.ema_decay,
            warmup_steps=args.ema_warmup_steps,
        )
        if args.ema_decay > 0
        else None
    )
    start_epoch = 1
    global_step = 0
    best_val_wer = float("inf")
    if checkpoint is not None:
        optimizer_states = checkpoint.get("optimizer_state_dicts", [])
        scheduler_states = checkpoint.get("scheduler_state_dicts", [])
        if len(optimizer_states) != len(optimizers) or len(scheduler_states) != len(schedulers):
            raise RuntimeError(
                "Resume checkpoint optimizer/scheduler layout does not match current setup."
            )
        for optimizer, state_dict in zip(optimizers, optimizer_states, strict=True):
            optimizer.load_state_dict(state_dict)
        for scheduler, state_dict in zip(schedulers, scheduler_states, strict=True):
            scheduler.load_state_dict(state_dict)
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        if ema is not None and checkpoint.get("ema_state_dict") is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_wer = float(checkpoint.get("best_val_wer", float("inf")))
        logger.info(
            "resumed from %s starting_epoch=%s global_step=%s best_val_wer=%.4f",
            args.resume,
            start_epoch,
            global_step,
            best_val_wer,
        )

    if is_main_process:
        trackio.init(
            project=args.trackio_project,
            space_id=args.trackio_space_id,
            config={
                **vars(args),
                "encoder_config": asdict(encoder_config),
                "train_samples": len(train_records),
                "val_samples": len(val_records),
                "active_optimizers": optimizer_names,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "featurizer_config": featurizer.config_dict(),
                "split_audit": split_audit,
                "distributed": distributed,
                "world_size": world_size,
            },
        )

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info("epoch %s/%s started", epoch, args.epochs)
        forward_model.train()
        running_loss = 0.0
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            with _autocast_context(device, args.dtype, fp8_recipe=fp8_recipe):
                log_probs, output_lengths = forward_model.log_probs(features, feature_lengths)
                loss = criterion(log_probs.transpose(0, 1), targets, output_lengths, target_lengths)
            running_loss += float(loss.item())
            scaler.scale(loss / args.gradient_accumulation_steps).backward()

            should_step = batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(
                train_loader
            )
            if should_step:
                if args.grad_clip_norm > 0:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    )
                else:
                    grad_norm = 0.0
                if _is_xla_device(device):
                    for optimizer in optimizers:
                        xm.optimizer_step(optimizer, barrier=False)
                else:
                    for optimizer in optimizers:
                        scaler.step(optimizer)
                scaler.update()
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in schedulers:
                    scheduler.step()
                global_step += 1
                if ema is not None:
                    ema.update(model)
                _mark_xla_step(device)

                if is_main_process and global_step % args.log_every == 0:
                    learning_rates = {
                        f"learning_rate_{name}": optimizer.param_groups[0]["lr"]
                        for name, optimizer in zip(optimizer_names, optimizers, strict=True)
                    }
                    logger.info(
                        "epoch=%s step=%s/%s global_step=%s train_loss=%.4f grad_norm=%.4f %s",
                        epoch,
                        batch_index,
                        len(train_loader),
                        global_step,
                        float(loss.item()),
                        grad_norm,
                        " ".join(f"{name}={value:.6g}" for name, value in learning_rates.items()),
                    )
                    trackio.log(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "train_loss_step": float(loss.item()),
                            "grad_norm": grad_norm,
                            "ema_decay": ema.current_decay() if ema is not None else 0.0,
                            **learning_rates,
                        }
                    )

        train_loss = running_loss / max(1, len(train_loader))
        if is_main_process:
            ema_backup = ema.apply_to(model) if ema is not None else None
            logger.info(
                "epoch %s training complete train_loss=%.4f, starting validation", epoch, train_loss
            )
            validation = evaluate(
                model,
                val_loader,
                criterion,
                tokenizer,
                device,
                args.dtype,
                fp8_recipe=fp8_recipe,
                decode_strategy=args.decode_strategy,
                beam_size=args.beam_size,
                lm_scorer=lm_scorer,
                lm_weight=args.lm_weight,
                example_limit=args.example_limit,
            )
            if ema_backup is not None:
                ExponentialMovingAverage.restore(model, ema_backup)
            val_metrics = validation["metrics"]
            log_payload = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_cer": val_metrics["cer"],
                "val_wer": val_metrics["wer"],
            }
            for key, value in val_metrics.items():
                if key not in {"loss", "cer", "wer"}:
                    log_payload[f"val_{key}"] = value
            log_payload.update(_flatten_examples("val_hardest", validation["hardest_examples"]))
            log_payload.update(_flatten_examples("val_random", validation["random_examples"]))
            trackio.log(log_payload)
            report = {
                "epoch": epoch,
                "global_step": global_step,
                "metrics": val_metrics,
                "hardest_examples": validation["hardest_examples"],
                "random_examples": validation["random_examples"],
                "speaker_metrics": validation["speaker_metrics"],
                "split_audit": split_audit,
            }
            reports_dir = output_dir / "eval_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / f"epoch_{epoch:04d}.json"
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            checkpoint = _build_checkpoint(
                model=model,
                encoder_config=encoder_config,
                tokenizer=tokenizer,
                featurizer=featurizer,
                epoch=epoch,
                global_step=global_step,
                best_val_wer=min(best_val_wer, val_metrics["wer"]),
                metrics=log_payload,
                optimizers=optimizers,
                optimizer_names=optimizer_names,
                schedulers=schedulers,
                scaler=scaler,
                ema=ema,
                args=args,
            )
            latest_path = output_dir / "checkpoint_last.pt"
            save_checkpoint(checkpoint, latest_path)
            latest_safetensors_path = _export_inference_checkpoint(checkpoint, latest_path)
            if val_metrics["wer"] < best_val_wer:
                best_val_wer = val_metrics["wer"]
                best_path = output_dir / "checkpoint_best.pt"
                save_checkpoint(checkpoint, best_path)
                best_safetensors_path = _export_inference_checkpoint(checkpoint, best_path)
            else:
                best_safetensors_path = _safetensors_path(output_dir / "checkpoint_best.pt")
            _update_top_checkpoints(
                output_dir=output_dir,
                checkpoint=checkpoint,
                epoch=epoch,
                val_wer=val_metrics["wer"],
                keep_top_k=args.keep_top_k,
            )
            averaged_path = _average_topk_checkpoints(output_dir)
            logger.info(
                (
                    "epoch %s complete train_loss=%.4f val_loss=%.4f val_cer=%.4f "
                    "val_wer=%.4f best_val_wer=%.4f report=%s latest=%s best=%s "
                    "averaged=%s"
                ),
                epoch,
                train_loss,
                float(val_metrics["loss"]),
                float(val_metrics["cer"]),
                float(val_metrics["wer"]),
                best_val_wer,
                report_path,
                latest_path,
                output_dir / "checkpoint_best.pt",
                averaged_path if averaged_path is not None else "n/a",
            )
            logger.info(
                "exported inference artifacts latest_safe=%s best_safe=%s averaged_safe=%s",
                latest_safetensors_path,
                best_safetensors_path if best_safetensors_path.exists() else "n/a",
                _safetensors_path(averaged_path).as_posix() if averaged_path is not None else "n/a",
            )
        if distributed:
            dist.barrier()

    if is_main_process:
        (output_dir / "train_summary.json").write_text(
            json.dumps(
                {
                    "best_val_wer": best_val_wer,
                    "variant": args.variant,
                    "keep_top_k": args.keep_top_k,
                    "decode_strategy": args.decode_strategy,
                    "distributed": distributed,
                    "world_size": world_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        trackio.finish()
        logger.info(
            "training finished epochs=%s best_val_wer=%.4f summary=%s",
            args.epochs,
            best_val_wer,
            output_dir / "train_summary.json",
        )
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
