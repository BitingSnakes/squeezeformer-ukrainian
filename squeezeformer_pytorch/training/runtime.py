from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import NamedTuple

import torch
import trackio
from torch import Tensor, nn
from torch.nn import functional as F

from squeezeformer_pytorch.asr import SqueezeformerCTC
from squeezeformer_pytorch.checkpoints import save_checkpoint
from squeezeformer_pytorch.model import (
    FP8_SHAPE_ALIGNMENT,
    SqueezeformerConfig,
    transformer_engine_available,
)
from squeezeformer_pytorch.runtime_types import DTypeChoice, OptimizerChoice

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
except (ImportError, OSError):
    te = None
    DelayedScaling = None
    Format = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None

try:
    from pytorch_optimizer import Muon as ExternalMuon
except ImportError:
    ExternalMuon = None


def _checkpoint_name(epoch: int, global_step: int, val_wer: float) -> str:
    return f"checkpoint_epoch={epoch:04d}_step={global_step:08d}_valwer={val_wer:.6f}.pt"


def _safetensors_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".safetensors")


def _inference_checkpoint_payload(checkpoint: dict[str, object]) -> dict[str, object]:
    state_dict = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if not key.startswith("aed_decoder.") and not key.startswith("liberta_projection.")
    }
    training_args = dict(checkpoint.get("training_args", {}))
    training_args["aed_decoder"] = False
    training_args["liberta_distill"] = False
    return {
        "model_state_dict": state_dict,
        "encoder_config": checkpoint["encoder_config"],
        "tokenizer": checkpoint["tokenizer"],
        "featurizer_config": checkpoint.get("featurizer_config", {}),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "best_val_wer": checkpoint.get("best_val_wer"),
        "metrics": checkpoint.get("metrics"),
        "training_args": training_args,
        "averaged_from": checkpoint.get("averaged_from"),
    }


def _export_inference_checkpoint(checkpoint: dict[str, object], checkpoint_path: Path) -> Path:
    safetensors_path = _safetensors_path(checkpoint_path)
    save_checkpoint(_inference_checkpoint_payload(checkpoint), safetensors_path)
    return safetensors_path


class SchedulerDefaults(NamedTuple):
    peak_lr: float
    num_time_masks: int


class FrozenLibertaTeacher:
    def __init__(
        self,
        model_source: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_length: int = 256,
    ) -> None:
        if AutoModel is None or AutoTokenizer is None:
            raise RuntimeError(
                "LiBERTa distillation requires transformers. Install the package and rerun."
            )
        self.device = device
        self.max_length = max_length
        tokenizer_kwargs = {"trust_remote_code": True}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
        except ValueError:
            # Some Ukrainian checkpoints expose only custom slow tokenizer code.
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                use_fast=False,
                **tokenizer_kwargs,
            )
        self.model = AutoModel.from_pretrained(
            model_source,
            trust_remote_code=True,
            dtype=dtype,
        )
        self.model.to(device=device, dtype=dtype)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)

    def encode(self, texts: list[str]) -> Tensor:
        return self._encode_recursive(texts)

    def _encode_recursive(self, texts: list[str]) -> Tensor:
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
        try:
            with torch.inference_mode():
                outputs = self.model(**tokenized)
        except torch.OutOfMemoryError:
            if self.device.type != "cuda" or len(texts) <= 1:
                raise
            torch.cuda.empty_cache()
            midpoint = max(1, len(texts) // 2)
            first_half = self._encode_recursive(texts[:midpoint])
            second_half = self._encode_recursive(texts[midpoint:])
            return torch.cat((first_half, second_half), dim=0)
        hidden = outputs.last_hidden_state
        mask = tokenized["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (hidden * mask).sum(dim=1) / denom
        return pooled


def _configure_console_logger(
    rank: int,
    is_main_process: bool,
    *,
    log_path: Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO if is_main_process else logging.WARNING)

    class _ColorFormatter(logging.Formatter):
        _RESET = "\033[0m"
        _LEVEL_COLORS = {
            logging.DEBUG: "\033[36m",
            logging.INFO: "\033[32m",
            logging.WARNING: "\033[33m",
            logging.ERROR: "\033[31m",
            logging.CRITICAL: "\033[35m",
        }

        def __init__(self, fmt: str, *, use_color: bool) -> None:
            super().__init__(fmt)
            self.use_color = use_color

        def format(self, record: logging.LogRecord) -> str:
            original_levelname = record.levelname
            if self.use_color:
                color = self._LEVEL_COLORS.get(record.levelno)
                if color is not None:
                    record.levelname = f"{color}{record.levelname}{self._RESET}"
            try:
                return super().format(record)
            finally:
                record.levelname = original_levelname

    class _RankFilter(logging.Filter):
        def __init__(self, default_rank: int) -> None:
            super().__init__()
            self.default_rank = default_rank

        def filter(self, record: logging.LogRecord) -> bool:
            if not hasattr(record, "rank"):
                record.rank = self.default_rank
            return True

    formatter = _ColorFormatter(
        "%(asctime)s | %(levelname)s | rank=%(rank)s | %(message)s",
        use_color=sys.stdout.isatty(),
    )
    plain_formatter = logging.Formatter("%(asctime)s | %(levelname)s | rank=%(rank)s | %(message)s")
    if not any(getattr(handler, "_train_console_handler", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler._train_console_handler = True  # type: ignore[attr-defined]
        handler.addFilter(_RankFilter(rank))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if (
        is_main_process
        and log_path is not None
        and not any(
            getattr(handler, "_train_file_path", None) == str(log_path)
            for handler in logger.handlers
        )
    ):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler._train_file_path = str(log_path)  # type: ignore[attr-defined]
        file_handler.addFilter(_RankFilter(rank))
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logging.LoggerAdapter(logger, {"rank": rank})


def _format_elapsed_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:.1f}s"


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _read_proc_status_memory_bytes(field_name: str) -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith(field_name):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def _peak_process_memory_bytes() -> int | None:
    try:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError, ValueError):
        return None
    if sys.platform == "darwin":
        return int(peak_rss)
    return int(peak_rss) * 1024


def _format_memory_snapshot(device: torch.device) -> str:
    memory_parts: list[str] = []
    current_rss = _read_proc_status_memory_bytes("VmRSS:")
    peak_rss = _read_proc_status_memory_bytes("VmHWM:")
    if peak_rss is None:
        peak_rss = _peak_process_memory_bytes()
    if current_rss is not None:
        memory_parts.append(f"cpu_rss={_format_bytes(current_rss)}")
    if peak_rss is not None:
        memory_parts.append(f"cpu_peak_rss={_format_bytes(peak_rss)}")
    if device.type == "cuda" and torch.cuda.is_available():
        memory_parts.extend(
            (
                f"cuda_allocated={_format_bytes(torch.cuda.memory_allocated(device))}",
                f"cuda_reserved={_format_bytes(torch.cuda.memory_reserved(device))}",
                f"cuda_peak_allocated={_format_bytes(torch.cuda.max_memory_allocated(device))}",
                f"cuda_peak_reserved={_format_bytes(torch.cuda.max_memory_reserved(device))}",
            )
        )
    return " ".join(memory_parts) if memory_parts else "memory=unavailable"


def _log_batch_autotune_snapshot(
    logger: logging.Logger,
    *,
    epoch: int,
    global_step: int,
    batch_index: int,
    total_batches: int,
    optimizer_step_index: int,
    effective_frames: int,
    padded_frames: int,
    target_tokens: int,
    device: torch.device,
) -> None:
    padding_waste = max(0, padded_frames - effective_frames)
    padding_waste_ratio = (padding_waste / padded_frames) if padded_frames > 0 else 0.0
    parts = [
        f"memory_tune epoch={epoch}",
        f"optimizer_step={optimizer_step_index}",
        f"batch={batch_index}/{total_batches}",
        f"global_step={global_step}",
        f"effective_frames={effective_frames}",
        f"padded_frames={padded_frames}",
        f"padding_waste_frames={padding_waste}",
        f"padding_waste_ratio={padding_waste_ratio:.3f}",
        f"target_tokens={target_tokens}",
    ]
    if device.type == "cuda" and torch.cuda.is_available():
        parts.extend(
            (
                f"cuda_allocated={_format_bytes(torch.cuda.memory_allocated(device))}",
                f"cuda_reserved={_format_bytes(torch.cuda.memory_reserved(device))}",
                f"cuda_peak_allocated={_format_bytes(torch.cuda.max_memory_allocated(device))}",
                f"cuda_peak_reserved={_format_bytes(torch.cuda.max_memory_reserved(device))}",
            )
        )
    logger.info(" ".join(parts))


def _validate_resume_checkpoint_payload(
    checkpoint: dict[str, object],
    *,
    checkpoint_path: Path,
) -> None:
    required_keys = (
        "model_state_dict",
        "encoder_config",
        "tokenizer",
        "optimizer_state_dicts",
        "scheduler_state_dicts",
        "epoch",
        "global_step",
    )
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise RuntimeError(
            f"Resume checkpoint '{checkpoint_path}' is missing required field(s): {', '.join(missing)}."
        )


def _configure_trackio_storage(output_dir: Path) -> Path:
    trackio_dir = output_dir / "trackio"
    trackio_dir.mkdir(parents=True, exist_ok=True)
    media_dir = trackio_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_DIR"] = str(trackio_dir)
    trackio.utils.TRACKIO_DIR = trackio_dir
    trackio.utils.MEDIA_DIR = media_dir
    trackio.TRACKIO_DIR = trackio_dir
    import trackio.sqlite_storage as trackio_sqlite_storage

    trackio_sqlite_storage.TRACKIO_DIR = trackio_dir
    trackio_sqlite_storage.MEDIA_DIR = media_dir
    return trackio_dir


def _resolve_resume_checkpoint_path(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    logger: logging.Logger,
) -> Path | None:
    if args.resume and args.auto_resume:
        raise ValueError("Use only one of --resume or --auto-resume.")
    if args.resume:
        return Path(args.resume)
    if not args.auto_resume:
        return None

    checkpoint_path = output_dir / "checkpoint_last.pt"
    if not checkpoint_path.exists():
        logger.info(
            "auto-resume requested but no checkpoint found at %s; starting a fresh run",
            checkpoint_path,
        )
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load auto-resume checkpoint '{checkpoint_path}': {error}"
        ) from error
    _validate_resume_checkpoint_payload(checkpoint, checkpoint_path=checkpoint_path)
    logger.info(
        "auto-resume validated checkpoint path=%s epoch=%s global_step=%s",
        checkpoint_path,
        checkpoint.get("epoch"),
        checkpoint.get("global_step"),
    )
    return checkpoint_path


def resolve_device(device: str) -> torch.device:
    return torch.device(device)


def _validate_device_ready(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested with --device, but torch.cuda.is_available() is false."
        )


def _variant_defaults(variant: str) -> SchedulerDefaults:
    if variant in {"xs", "s", "sm"}:
        return SchedulerDefaults(peak_lr=2e-3, num_time_masks=5)
    if variant == "m":
        return SchedulerDefaults(peak_lr=1.5e-3, num_time_masks=7)
    return SchedulerDefaults(peak_lr=1e-3, num_time_masks=10)


def _parse_intermediate_ctc_layers(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return ()
        return tuple(int(part.strip()) for part in normalized.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    raise TypeError(f"Unsupported intermediate CTC layer specification: {type(value)!r}")


def _dedupe_sorted_layers(layers: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(set(layers)))


def _default_intermediate_ctc_layers(encoder_config: SqueezeformerConfig) -> tuple[int, ...]:
    if encoder_config.num_layers < 4:
        return (max(0, encoder_config.num_layers - 2),)
    return _dedupe_sorted_layers(
        (
            max(0, (encoder_config.num_layers // 3) - 1),
            max(0, ((2 * encoder_config.num_layers) // 3) - 1),
        )
    )


def _resolve_intermediate_ctc_settings(
    args: argparse.Namespace,
    encoder_config: SqueezeformerConfig,
    checkpoint: dict[str, object] | None,
) -> tuple[tuple[int, ...], float]:
    checkpoint_args = checkpoint.get("training_args", {}) if checkpoint is not None else {}
    checkpoint_enabled = checkpoint_args.get("intermediate_ctc_enabled")
    checkpoint_weight = checkpoint_args.get("intermediate_ctc_weight")
    checkpoint_layers = checkpoint_args.get("intermediate_ctc_layers")
    checkpoint_layer = checkpoint_args.get("intermediate_ctc_layer")

    if args.intermediate_ctc is False:
        return (), 0.0
    if args.intermediate_ctc is None and checkpoint is not None and checkpoint_enabled is False:
        return (), 0.0

    if checkpoint_weight is not None:
        weight = float(checkpoint_weight)
        if checkpoint_layers is not None:
            layers = _parse_intermediate_ctc_layers(checkpoint_layers)
        else:
            layers = _parse_intermediate_ctc_layers(checkpoint_layer)
    else:
        weight = float(args.intermediate_ctc_weight)
        if args.intermediate_ctc_layers is not None:
            layers = _parse_intermediate_ctc_layers(args.intermediate_ctc_layers)
        else:
            layers = _parse_intermediate_ctc_layers(args.intermediate_ctc_layer)

    if weight <= 0.0:
        return (), 0.0
    if not layers:
        layers = _default_intermediate_ctc_layers(encoder_config)
    layers = _dedupe_sorted_layers(layers)
    invalid_layers = [layer for layer in layers if not 0 <= layer < encoder_config.num_layers]
    if invalid_layers:
        raise ValueError(
            "--intermediate-ctc-layers must be within encoder block range "
            f"[0, {encoder_config.num_layers - 1}], got {invalid_layers}."
        )
    return layers, weight


def _resolve_blank_pruning_settings(
    args: argparse.Namespace,
    encoder_config: SqueezeformerConfig,
    checkpoint: dict[str, object] | None,
) -> tuple[int | None, float, int]:
    checkpoint_args = checkpoint.get("training_args", {}) if checkpoint is not None else {}
    checkpoint_enabled = checkpoint_args.get("blank_prune_enabled")
    if args.blank_prune is False:
        return None, 0.0, max(1, int(args.blank_prune_min_keep_frames))
    if args.blank_prune is None and checkpoint is not None and checkpoint_enabled is False:
        return None, 0.0, max(1, int(checkpoint_args.get("blank_prune_min_keep_frames", 1)))
    if "blank_prune_threshold" in checkpoint_args:
        threshold = float(checkpoint_args.get("blank_prune_threshold", 0.0))
        layer = checkpoint_args.get("blank_prune_layer")
        min_keep_frames = int(checkpoint_args.get("blank_prune_min_keep_frames", 1))
    else:
        threshold = float(args.blank_prune_threshold)
        layer = args.blank_prune_layer
        min_keep_frames = int(args.blank_prune_min_keep_frames)

    if threshold <= 0.0:
        return None, 0.0, max(1, min_keep_frames)
    if layer is None:
        raise ValueError("--blank-prune-layer is required when --blank-prune-threshold > 0.")
    layer = int(layer)
    if not 0 <= layer < encoder_config.num_layers:
        raise ValueError(
            "--blank-prune-layer must be within encoder block range "
            f"[0, {encoder_config.num_layers - 1}], got {layer}."
        )
    if min_keep_frames < 1:
        raise ValueError("--blank-prune-min-keep-frames must be at least 1.")
    return layer, threshold, min_keep_frames


def _resolve_aed_settings(
    args: argparse.Namespace,
    checkpoint: dict[str, object] | None,
) -> tuple[bool, int, int, float, float]:
    checkpoint_args = checkpoint.get("training_args", {}) if checkpoint is not None else {}
    checkpoint_enabled = checkpoint_args.get("aed_decoder")

    if args.aed_decoder is False:
        return (
            False,
            int(args.aed_decoder_layers),
            int(args.aed_decoder_heads),
            float(args.aed_decoder_dropout),
            float(args.aed_loss_weight),
        )
    if args.aed_decoder is None and checkpoint is not None and checkpoint_enabled is not None:
        enabled = bool(checkpoint_enabled)
        return (
            enabled,
            int(checkpoint_args.get("aed_decoder_layers", args.aed_decoder_layers)),
            int(checkpoint_args.get("aed_decoder_heads", args.aed_decoder_heads)),
            float(checkpoint_args.get("aed_decoder_dropout", args.aed_decoder_dropout)),
            float(checkpoint_args.get("aed_loss_weight", args.aed_loss_weight)),
        )
    enabled = bool(args.aed_decoder) if args.aed_decoder is not None else False
    return (
        enabled,
        int(args.aed_decoder_layers),
        int(args.aed_decoder_heads),
        float(args.aed_decoder_dropout),
        float(args.aed_loss_weight),
    )


def _resolve_liberta_settings(
    args: argparse.Namespace,
    checkpoint: dict[str, object] | None,
    *,
    aed_enabled: bool,
) -> tuple[bool, str, str | None, float, int]:
    checkpoint_args = checkpoint.get("training_args", {}) if checkpoint is not None else {}
    checkpoint_enabled = checkpoint_args.get("liberta_distill")

    if args.liberta_distill is False:
        return (
            False,
            args.liberta_model_name,
            args.liberta_model_path,
            float(args.liberta_distill_weight),
            int(args.liberta_max_length),
        )
    if args.liberta_distill is None and checkpoint is not None and checkpoint_enabled is not None:
        enabled = bool(checkpoint_enabled)
        model_name = str(checkpoint_args.get("liberta_model_name", args.liberta_model_name))
        checkpoint_model_path = checkpoint_args.get("liberta_model_path", args.liberta_model_path)
        model_path = str(checkpoint_model_path) if checkpoint_model_path is not None else None
        weight = float(checkpoint_args.get("liberta_distill_weight", args.liberta_distill_weight))
        max_length = int(checkpoint_args.get("liberta_max_length", args.liberta_max_length))
    else:
        enabled = bool(args.liberta_distill) if args.liberta_distill is not None else False
        model_name = args.liberta_model_name
        model_path = args.liberta_model_path
        weight = float(args.liberta_distill_weight)
        max_length = int(args.liberta_max_length)

    if enabled and not aed_enabled:
        raise ValueError("LiBERTa distillation requires --aed-decoder.")
    if not enabled:
        return False, model_name, model_path, weight, max_length
    if weight <= 0.0:
        raise ValueError(
            "--liberta-distill-weight must be > 0 when LiBERTa distillation is enabled."
        )
    return True, model_name, model_path, weight, max_length


def _build_aed_targets(
    targets: Tensor,
    target_lengths: Tensor,
    *,
    bos_id: int,
    eos_id: int,
    token_offset: int,
    pad_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    sequences: list[Tensor] = []
    offset = 0
    for length in target_lengths.tolist():
        sequence = targets[offset : offset + length] + token_offset
        offset += length
        sequences.append(sequence)

    max_target_length = max((sequence.numel() for sequence in sequences), default=0) + 1
    batch_size = len(sequences)
    decoder_inputs = targets.new_full((batch_size, max_target_length), pad_id)
    decoder_targets = targets.new_full((batch_size, max_target_length), pad_id)
    decoder_target_lengths = target_lengths.new_empty(batch_size)

    for index, sequence in enumerate(sequences):
        target_sequence = torch.cat([sequence, sequence.new_tensor([eos_id])], dim=0)
        input_sequence = torch.cat([sequence.new_tensor([bos_id]), sequence], dim=0)
        decoder_inputs[index, : input_sequence.numel()] = input_sequence
        decoder_targets[index, : target_sequence.numel()] = target_sequence
        decoder_target_lengths[index] = target_sequence.numel()

    return decoder_inputs, decoder_targets, decoder_target_lengths


def _aed_cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    pad_id: int,
) -> Tensor:
    if logits.dim() != 3:
        raise ValueError(
            f"Expected AED logits with shape [batch, time, vocab], got {tuple(logits.shape)}"
        )
    if targets.dim() != 2:
        raise ValueError(
            f"Expected AED targets with shape [batch, time], got {tuple(targets.shape)}"
        )
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "AED logits and targets must agree on batch/time dimensions, got "
            f"{tuple(logits.shape[:2])} vs {tuple(targets.shape)}"
        )
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
        reduction="sum",
    )
    normalizer = targets.ne(pad_id).sum().clamp_min(1)
    return loss / normalizer


def _compute_grad_norm(parameters, norm_type: float = 2.0) -> Tensor:
    grads = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    device = grads[0].device
    per_param_norms = torch.stack(
        [torch.norm(grad, p=norm_type).to(device=device) for grad in grads]
    )
    return torch.norm(per_param_norms, p=norm_type)


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
        muon_cls = getattr(torch.optim, "Muon", None) or ExternalMuon
        if muon_cls is None:
            raise RuntimeError(
                "Muon optimizer requires `torch.optim.Muon` or the `pytorch-optimizer` package. "
                "Install the training dependencies with `pytorch-optimizer>=3.10.0`."
            )
        muon_param_group: dict[str, object] = {"params": muon_params}
        if muon_cls is ExternalMuon:
            muon_param_group["use_muon"] = True
        optimizers.append(
            muon_cls(
                [muon_param_group],
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


def _resolve_model_load_dtype(dtype: DTypeChoice) -> torch.dtype:
    autocast_dtype = _resolve_autocast_dtype(dtype)
    if autocast_dtype is not None:
        return autocast_dtype
    return torch.float32


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
    return torch.autocast(device_type=device.type, dtype=autocast_dtype)


def _state_dict_shape_map(state_dict: dict[str, Tensor]) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in state_dict.items()}


def _checkpoint_compatibility_signature(model_state_dict: dict[str, Tensor]) -> str:
    shape_items = sorted(
        (key, list(shape)) for key, shape in _state_dict_shape_map(model_state_dict).items()
    )
    payload = json.dumps(shape_items, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_topk_metadata(metadata_path: Path) -> tuple[str | None, list[dict[str, object]]]:
    if not metadata_path.exists():
        return None, []
    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(raw_metadata, list):
        return None, raw_metadata
    if not isinstance(raw_metadata, dict):
        raise ValueError(f"Unexpected top-k metadata format in {metadata_path}.")
    items = raw_metadata.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"Unexpected top-k metadata items format in {metadata_path}.")
    compatibility_signature = raw_metadata.get("compatibility_signature")
    if compatibility_signature is not None and not isinstance(compatibility_signature, str):
        raise ValueError(f"Unexpected top-k compatibility signature format in {metadata_path}.")
    return compatibility_signature, items


def _write_topk_metadata(
    metadata_path: Path,
    compatibility_signature: str,
    items: list[dict[str, object]],
) -> None:
    metadata_path.write_text(
        json.dumps(
            {
                "compatibility_signature": compatibility_signature,
                "items": items,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _update_top_checkpoints(
    output_dir: Path,
    checkpoint: dict[str, object],
    epoch: int,
    global_step: int,
    val_wer: float,
    keep_top_k: int,
) -> None:
    topk_dir = output_dir / "checkpoints_topk"
    topk_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = topk_dir / "metadata.json"
    compatibility_signature, metadata = _load_topk_metadata(metadata_path)

    current_signature = _checkpoint_compatibility_signature(checkpoint["model_state_dict"])
    logger = logging.getLogger("train")
    if compatibility_signature is not None and compatibility_signature != current_signature:
        for item in metadata:
            checkpoint_path = topk_dir / str(item["path"])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        metadata = []
    compatible_metadata: list[dict[str, object]] = []
    removed_incompatible = 0
    for item in metadata:
        checkpoint_path = topk_dir / str(item["path"])
        if not checkpoint_path.exists():
            continue
        saved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_signature = _checkpoint_compatibility_signature(saved_checkpoint["model_state_dict"])
        if saved_signature != current_signature:
            checkpoint_path.unlink()
            removed_incompatible += 1
            continue
        compatible_metadata.append(item)
    if removed_incompatible:
        logger.info(
            "Removed %s incompatible top-k checkpoint artifact(s) from %s.",
            removed_incompatible,
            topk_dir,
            extra={"rank": 0},
        )
    metadata = compatible_metadata

    filename = _checkpoint_name(epoch=epoch, global_step=global_step, val_wer=val_wer)
    checkpoint_path = topk_dir / filename
    save_checkpoint(checkpoint, checkpoint_path)

    metadata.append(
        {
            "epoch": epoch,
            "global_step": global_step,
            "val_wer": val_wer,
            "path": str(checkpoint_path.name),
        }
    )
    metadata.sort(
        key=lambda item: (
            float(item["val_wer"]),
            int(item.get("global_step", 0)),
            int(item["epoch"]),
        )
    )

    removed = metadata[keep_top_k:]
    metadata = metadata[:keep_top_k]
    for item in removed:
        stale_path = topk_dir / str(item["path"])
        if stale_path.exists():
            stale_path.unlink()

    _write_topk_metadata(metadata_path, current_signature, metadata)


def _average_topk_checkpoints(output_dir: Path) -> Path | None:
    metadata_path = output_dir / "checkpoints_topk" / "metadata.json"
    if not metadata_path.exists():
        return None
    _, metadata = _load_topk_metadata(metadata_path)
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
            template_shapes = _state_dict_shape_map(state_dict)
            included_metadata.append(item)
        else:
            current_shapes = _state_dict_shape_map(state_dict)
            if current_shapes != template_shapes:
                logging.getLogger("train").warning(
                    "Skipping checkpoint %s during top-k averaging because parameter shapes do not match the template checkpoint.",
                    checkpoint_path,
                    extra={"rank": 0},
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
