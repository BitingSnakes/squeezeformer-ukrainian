from __future__ import annotations

import argparse
import array
import base64
import hashlib
import json
import logging
import mmap
import os
import random
import resource
import struct
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple
import torch
import torch.distributed as dist
import trackio
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from squeezeformer_pytorch.asr import (
    CharacterTokenizer,
    SentencePieceTokenizer,
    SqueezeformerCTC,
    Tokenizer,
    ctc_prefix_beam_search,
    load_lm_scorer,
    load_tokenizer,
    tokenizer_from_dict,
)
from squeezeformer_pytorch.checkpoints import save_checkpoint
from squeezeformer_pytorch.data import (
    AudioFeaturizer,
    CV22ASRDataset,
    CVRecord,
    SpecAugment,
    WaveformAugment,
    create_dataloader,
    download_cv22_dataset,
    iter_cv22_records,
    iter_cv22_records_from_source,
    load_audio,
    prevalidate_records,
    probe_audio_metadata,
    read_binary_source,
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


def _checkpoint_name(epoch: int, val_wer: float) -> str:
    return f"checkpoint_epoch={epoch:04d}_valwer={val_wer:.6f}.pt"


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
        model_name: str,
        *,
        device: torch.device,
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
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        except ValueError:
            # Some Ukrainian checkpoints expose only custom slow tokenizer code.
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                **tokenizer_kwargs,
            )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)

    def encode(self, texts: list[str]) -> Tensor:
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
        with torch.no_grad():
            outputs = self.model(**tokenized)
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


def _validate_device_argument(device: str) -> str:
    try:
        torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"Invalid device '{device}': {error}") from error
    return device


def _is_explicit_local_path(value: str) -> bool:
    return value.startswith(("/", "./", "../", "~"))


def _validate_existing_local_path_argument(
    argument_name: str,
    raw_value: str,
    *,
    expected: str = "any",
) -> None:
    path = Path(raw_value).expanduser()
    if path.exists():
        if expected == "file" and not path.is_file():
            raise ValueError(f"{argument_name} must point to a file, got '{raw_value}'.")
        if expected == "dir" and not path.is_dir():
            raise ValueError(f"{argument_name} must point to a directory, got '{raw_value}'.")
        return
    if _is_explicit_local_path(raw_value):
        raise ValueError(f"{argument_name} points to a missing local path: '{raw_value}'.")


def _validate_creatable_directory(argument_name: str, raw_value: str | None) -> None:
    if raw_value is None:
        return
    path = Path(raw_value).expanduser()
    if path.exists() and not path.is_dir():
        raise ValueError(f"{argument_name} must be a directory path, got '{raw_value}'.")
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        raise ValueError(f"{argument_name} has no existing parent directory: '{raw_value}'.")
    if not probe.is_dir():
        raise ValueError(f"{argument_name} parent is not a directory: '{probe}'.")
    if not os.access(probe, os.W_OK):
        raise ValueError(f"{argument_name} is not writable via parent directory '{probe}'.")


def _validate_startup_args(args: argparse.Namespace, *, world_size: int) -> None:
    if (args.adaptive_batch_unit is None) != (args.adaptive_batch_budget is None):
        raise ValueError("--adaptive-batch-unit and --adaptive-batch-budget must be set together.")
    if args.distributed and world_size <= 1:
        raise ValueError("--distributed expects a torchrun-style environment with WORLD_SIZE > 1.")
    if world_size > 1 and args.compile:
        raise ValueError("--compile is not currently supported together with distributed training.")

    requested_device = resolve_device(args.device)
    _validate_device_ready(requested_device)

    positive_int_arguments = {
        "--batch-size": args.batch_size,
        "--epochs": args.epochs,
        "--gradient-accumulation-steps": args.gradient_accumulation_steps,
        "--metadata-workers": args.metadata_workers,
        "--prevalidate-workers": args.prevalidate_workers,
        "--log-every": args.log_every,
        "--keep-top-k": args.keep_top_k,
        "--beam-size": args.beam_size,
        "--spm-vocab-size": args.spm_vocab_size,
        "--warmup-epochs": args.warmup_epochs,
        "--hold-epochs": args.hold_epochs,
        "--blank-prune-min-keep-frames": args.blank_prune_min_keep_frames,
        "--aed-decoder-layers": args.aed_decoder_layers,
        "--aed-decoder-heads": args.aed_decoder_heads,
        "--liberta-max-length": args.liberta_max_length,
        "--fp8-amax-history-len": args.fp8_amax_history_len,
        "--example-limit": args.example_limit,
        "--n-fft": args.n_fft,
        "--hop-length": args.hop_length,
        "--n-mels": args.n_mels,
        "--num-freq-masks": args.num_freq_masks,
        "--freq-mask-param": args.freq_mask_param,
    }
    if args.prefetch_factor is not None and args.num_workers > 0:
        positive_int_arguments["--prefetch-factor"] = args.prefetch_factor
    if args.num_time_masks is not None:
        positive_int_arguments["--num-time-masks"] = args.num_time_masks
    if args.max_train_samples is not None:
        positive_int_arguments["--max-train-samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        positive_int_arguments["--max-val-samples"] = args.max_val_samples
    if args.max_batch_frames is not None:
        positive_int_arguments["--max-batch-frames"] = args.max_batch_frames
    if args.adaptive_batch_budget is not None:
        positive_int_arguments["--adaptive-batch-budget"] = args.adaptive_batch_budget
    for name, value in positive_int_arguments.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}.")

    nonnegative_int_arguments = {
        "--num-workers": args.num_workers,
        "--seed": args.seed,
        "--intermediate-ctc-layer": args.intermediate_ctc_layer,
        "--blank-prune-layer": args.blank_prune_layer,
        "--ema-warmup-steps": args.ema_warmup_steps,
    }
    for name, value in nonnegative_int_arguments.items():
        if value is not None and value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}.")

    nonnegative_float_arguments = {
        "--weight-decay": args.weight_decay,
        "--grad-clip-norm": args.grad_clip_norm,
        "--muon-weight-decay": args.muon_weight_decay,
        "--adamw-weight-decay": args.adamw_weight_decay,
        "--learning-rate": args.learning_rate,
        "--muon-learning-rate": args.muon_learning_rate,
        "--adamw-learning-rate": args.adamw_learning_rate,
        "--preemphasis": args.preemphasis,
        "--lm-weight": args.lm_weight,
    }
    for name, value in nonnegative_float_arguments.items():
        if value is not None and value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}.")

    probability_arguments = {
        "--val-fraction": args.val_fraction,
        "--test-fraction": args.test_fraction,
        "--max-symbol-ratio": args.max_symbol_ratio,
        "--speed-perturb-prob": args.speed_perturb_prob,
        "--noise-prob": args.noise_prob,
        "--reverb-prob": args.reverb_prob,
        "--time-mask-max-ratio": args.time_mask_max_ratio,
        "--intermediate-ctc-weight": args.intermediate_ctc_weight,
        "--aed-decoder-dropout": args.aed_decoder_dropout,
        "--aed-loss-weight": args.aed_loss_weight,
        "--liberta-distill-weight": args.liberta_distill_weight,
        "--blank-prune-threshold": args.blank_prune_threshold,
        "--ema-decay": args.ema_decay,
    }
    for name, value in probability_arguments.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0, 1], got {value}.")
    if args.val_fraction + args.test_fraction >= 1.0:
        raise ValueError("--val-fraction + --test-fraction must be < 1.")

    if args.min_transcript_chars < 1:
        raise ValueError(f"--min-transcript-chars must be >= 1, got {args.min_transcript_chars}.")
    if args.max_transcript_chars < args.min_transcript_chars:
        raise ValueError(
            "--max-transcript-chars must be >= --min-transcript-chars, got "
            f"{args.max_transcript_chars} < {args.min_transcript_chars}."
        )
    if args.min_audio_duration_sec <= 0:
        raise ValueError(
            f"--min-audio-duration-sec must be > 0, got {args.min_audio_duration_sec}."
        )
    if args.max_audio_duration_sec < args.min_audio_duration_sec:
        raise ValueError(
            "--max-audio-duration-sec must be >= --min-audio-duration-sec, got "
            f"{args.max_audio_duration_sec} < {args.min_audio_duration_sec}."
        )
    if args.noise_snr_db_max < args.noise_snr_db_min:
        raise ValueError(
            "--noise-snr-db-max must be >= --noise-snr-db-min, got "
            f"{args.noise_snr_db_max} < {args.noise_snr_db_min}."
        )
    if args.reverb_decay_max < args.reverb_decay_min:
        raise ValueError(
            "--reverb-decay-max must be >= --reverb-decay-min, got "
            f"{args.reverb_decay_max} < {args.reverb_decay_min}."
        )
    if args.reverb_delay_ms_max < args.reverb_delay_ms_min:
        raise ValueError(
            "--reverb-delay-ms-max must be >= --reverb-delay-ms-min, got "
            f"{args.reverb_delay_ms_max} < {args.reverb_delay_ms_min}."
        )

    speed_factors = _resolve_float_tuple(args.speed_factors)
    if any(value <= 0 for value in speed_factors):
        raise ValueError(
            f"--speed-factors must contain only positive values, got {args.speed_factors}."
        )
    _resolve_block_pattern(args.block_pattern)

    _validate_creatable_directory("--output-dir", args.output_dir)
    _validate_creatable_directory("--feature-cache-dir", args.feature_cache_dir)
    _validate_creatable_directory("--record-cache-dir", args.record_cache_dir)
    if args.resume is not None:
        _validate_existing_local_path_argument("--resume", args.resume, expected="file")
    if args.tokenizer_path is not None:
        _validate_existing_local_path_argument(
            "--tokenizer-path",
            args.tokenizer_path,
            expected="file",
        )

    for source in args.dataset_source or []:
        _validate_existing_local_path_argument("--dataset-source", source)
    for source in args.validation_dataset_source or []:
        _validate_existing_local_path_argument("--validation-dataset-source", source)


def _resolve_dataset_roots(args: argparse.Namespace) -> list[Path]:
    sources = list(args.dataset_source or [])
    if not sources:
        sources = [args.dataset_repo]

    dataset_roots: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        dataset_root = download_cv22_dataset(
            repo_id=source,
            token=args.hf_token,
            cache_dir=args.cache_dir,
        ).resolve()
        if dataset_root in seen:
            continue
        seen.add(dataset_root)
        dataset_roots.append(dataset_root)
    return dataset_roots


def _resolve_sources(
    raw_sources: list[str] | None, *, fallback: str | None = None
) -> list[str | Path]:
    sources = list(raw_sources or [])
    if not sources:
        if fallback is None:
            return []
        sources = [fallback]

    resolved_sources: list[str | Path] = []
    seen: set[str] = set()
    for source in sources:
        source_path = Path(source).expanduser()
        resolved_source: str | Path = source_path.resolve() if source_path.exists() else source
        dedupe_key = str(resolved_source)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        resolved_sources.append(resolved_source)
    return resolved_sources


def _resolve_dataset_sources(args: argparse.Namespace) -> list[str | Path]:
    return _resolve_sources(args.dataset_source, fallback=args.dataset_repo)


def _resolve_validation_dataset_sources(args: argparse.Namespace) -> list[str | Path]:
    return _resolve_sources(args.validation_dataset_source)


class DiskBackedRecordStore:
    def __init__(
        self,
        records_path: Path,
        offsets: array.array,
        estimated_frames: array.array,
        *,
        start: int = 0,
        step: int = 1,
    ) -> None:
        self.records_path = records_path
        self.offsets = offsets
        self.estimated_frames = estimated_frames
        self.start = start
        self.step = step
        self._handle = None
        self._handle_pid: int | None = None

    def _global_index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.start + (index * self.step)

    def _open_handle(self):
        current_pid = os.getpid()
        if self._handle_pid != current_pid and self._handle is not None and not self._handle.closed:
            self._handle.close()
            self._handle = None
        if self._handle is None or self._handle.closed:
            self._handle = self.records_path.open("rb")
            self._handle_pid = current_pid
        return self._handle

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        self._handle = None
        self._handle_pid = None

    def __len__(self) -> int:
        total = len(self.offsets)
        if self.start >= total:
            return 0
        return ((total - self.start - 1) // self.step) + 1

    def __getitem__(self, index: int) -> CVRecord:
        global_index = self._global_index(index)
        handle = self._open_handle()
        handle.seek(self.offsets[global_index])
        payload = json.loads(handle.readline().decode("utf-8"))
        return CVRecord(
            audio_path=payload["audio_path"],
            audio_bytes=_load_cached_audio_bytes(payload, records_path=self.records_path),
            transcript=payload["transcript"],
            utterance_id=payload["utterance_id"],
            estimated_frames=int(self.estimated_frames[global_index]),
            speaker_id=payload["speaker_id"],
            has_speaker_id=bool(payload["has_speaker_id"]),
        )

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def shard(self, rank: int, world_size: int) -> "DiskBackedRecordStore":
        return DiskBackedRecordStore(
            self.records_path,
            self.offsets,
            self.estimated_frames,
            start=self.start + rank,
            step=self.step * world_size,
        )

    def populate_metadata(self, hop_length: int, num_workers: int = 4) -> None:
        global_indices = [
            self._global_index(index)
            for index in range(len(self))
            if int(self.estimated_frames[self._global_index(index)]) <= 0
        ]
        if not global_indices:
            return

        def populate(global_index: int) -> tuple[int, int]:
            handle = self.records_path.open("rb")
            try:
                handle.seek(self.offsets[global_index])
                payload = json.loads(handle.readline().decode("utf-8"))
            finally:
                handle.close()
            audio_bytes = _load_cached_audio_bytes(payload, records_path=self.records_path)
            num_samples, sample_rate = probe_audio_metadata(payload["audio_path"], audio_bytes)
            frames = (
                max(1, int(num_samples / hop_length)) if num_samples > 0 and sample_rate > 0 else 0
            )
            return global_index, frames

        if num_workers <= 1:
            populated = [populate(global_index) for global_index in global_indices]
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                populated = list(executor.map(populate, global_indices))
        for global_index, frames in populated:
            self.estimated_frames[global_index] = max(0, int(frames))


class _BinaryIndexView:
    def __init__(
        self,
        path: Path,
        *,
        item_size: int,
        fmt: str,
        writable: bool = False,
    ) -> None:
        self.path = path
        self.item_size = item_size
        self.fmt = fmt
        self.writable = writable
        self._length = path.stat().st_size // item_size if path.exists() else 0
        self._handle = None
        self._mmap = None
        self._pid: int | None = None

    def _ensure_open(self):
        current_pid = os.getpid()
        if self._pid != current_pid:
            self.close()
        if self._mmap is None:
            mode = "r+b" if self.writable else "rb"
            self._handle = self.path.open(mode)
            access = mmap.ACCESS_WRITE if self.writable else mmap.ACCESS_READ
            self._mmap = mmap.mmap(self._handle.fileno(), length=0, access=access)
            self._pid = current_pid
        return self._mmap

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        self._mmap = None
        self._handle = None
        self._pid = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> int:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        return int(struct.unpack_from(self.fmt, self._ensure_open(), index * self.item_size)[0])

    def __setitem__(self, index: int, value: int) -> None:
        if not self.writable:
            raise TypeError("Index view is read-only.")
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        struct.pack_into(self.fmt, self._ensure_open(), index * self.item_size, int(value))

    def __iter__(self):
        for index in range(self._length):
            yield self[index]


def _record_index_path(records_path: Path, suffix: str) -> Path:
    return records_path.with_suffix(records_path.suffix + suffix)


def _build_disk_backed_record_store(
    dataset_sources: list[str | Path],
    *,
    split: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    max_samples: int | None,
    min_transcript_chars: int,
    max_transcript_chars: int,
    max_symbol_ratio: float,
    lowercase_transcripts: bool,
    records_path: Path,
    min_audio_duration_sec: float = 0.01,
    max_audio_duration_sec: float = 30.0,
    hf_token: str | None = None,
    cache_dir: str | None = None,
) -> DiskBackedRecordStore:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    audio_blob_dir = records_path.parent / f"{records_path.stem}_audio_blobs"
    offsets_path = _record_index_path(records_path, ".offsets.u64")
    estimated_frames_path = _record_index_path(records_path, ".estimated_frames.u32")
    written = 0
    with (
        records_path.open("wb") as handle,
        offsets_path.open("wb") as offsets_handle,
        estimated_frames_path.open("wb") as estimated_frames_handle,
    ):
        for dataset_source in dataset_sources:
            remaining_samples = None
            if max_samples is not None:
                remaining_samples = max_samples - written
                if remaining_samples <= 0:
                    break
            record_iterator = (
                iter_cv22_records(
                    dataset_root=dataset_source,
                    split=split,
                    seed=seed,
                    val_fraction=val_fraction,
                    test_fraction=test_fraction,
                    max_samples=remaining_samples,
                    min_transcript_chars=min_transcript_chars,
                    max_transcript_chars=max_transcript_chars,
                    max_symbol_ratio=max_symbol_ratio,
                    min_audio_duration_sec=min_audio_duration_sec,
                    max_audio_duration_sec=max_audio_duration_sec,
                    lowercase_transcripts=lowercase_transcripts,
                )
                if isinstance(dataset_source, Path)
                and dataset_source.exists()
                and dataset_source.is_dir()
                else iter_cv22_records_from_source(
                    dataset_source,
                    split=split,
                    seed=seed,
                    val_fraction=val_fraction,
                    test_fraction=test_fraction,
                    max_samples=remaining_samples,
                    min_transcript_chars=min_transcript_chars,
                    max_transcript_chars=max_transcript_chars,
                    max_symbol_ratio=max_symbol_ratio,
                    min_audio_duration_sec=min_audio_duration_sec,
                    max_audio_duration_sec=max_audio_duration_sec,
                    lowercase_transcripts=lowercase_transcripts,
                    hf_token=hf_token,
                    cache_dir=cache_dir,
                )
            )
            for record in record_iterator:
                audio_bytes = record.audio_bytes
                if audio_bytes is None and record.audio_path is not None:
                    try:
                        audio_bytes = read_binary_source(
                            record.audio_path,
                            token=hf_token,
                            cache_dir=cache_dir,
                        )
                    except Exception:
                        audio_bytes = None
                preserve_audio_bytes = audio_bytes is not None and not (
                    record.audio_path is not None and Path(record.audio_path).exists()
                )
                audio_blob_path: str | None = None
                if preserve_audio_bytes:
                    audio_blob_dir.mkdir(parents=True, exist_ok=True)
                    blob_name = hashlib.sha256(audio_bytes).hexdigest() + ".bin"
                    blob_path = audio_blob_dir / blob_name
                    if not blob_path.exists():
                        blob_path.write_bytes(audio_bytes)
                    audio_blob_path = str(blob_path.relative_to(records_path.parent))
                offsets_handle.write(struct.pack("<Q", handle.tell()))
                payload = json.dumps(
                    {
                        "audio_path": record.audio_path,
                        "audio_blob_path": audio_blob_path,
                        "transcript": record.transcript,
                        "utterance_id": record.utterance_id,
                        "speaker_id": record.speaker_id,
                        "has_speaker_id": record.has_speaker_id,
                    },
                    ensure_ascii=False,
                )
                handle.write(payload.encode("utf-8"))
                handle.write(b"\n")
                estimated_frames_handle.write(
                    struct.pack("<I", max(0, int(record.estimated_frames)))
                )
                written += 1
    if written == 0:
        raise RuntimeError(
            f"Split '{split}' is empty after applying the current split fractions across "
            "all dataset sources."
        )
    return DiskBackedRecordStore(
        records_path,
        _BinaryIndexView(offsets_path, item_size=8, fmt="<Q"),
        _BinaryIndexView(
            estimated_frames_path,
            item_size=4,
            fmt="<I",
            writable=True,
        ),
    )


def _load_cached_audio_bytes(payload: dict[str, object], *, records_path: Path) -> bytes | None:
    audio_blob_path = payload.get("audio_blob_path")
    if isinstance(audio_blob_path, str) and audio_blob_path:
        blob_path = Path(audio_blob_path)
        if not blob_path.is_absolute():
            blob_path = records_path.parent / blob_path
        return blob_path.read_bytes()
    audio_bytes_payload = payload.get("audio_bytes")
    if isinstance(audio_bytes_payload, str) and audio_bytes_payload:
        return base64.b64decode(audio_bytes_payload)
    return None


_OPUS_HEADER_SCAN_BYTES = 64 * 1024


def _audio_header_looks_like_opus(header: bytes) -> bool:
    return b"OpusHead" in header[:_OPUS_HEADER_SCAN_BYTES]


def _read_header_from_path(audio_path: str | None) -> bytes:
    if not audio_path:
        return b""
    try:
        path = Path(audio_path)
    except OSError:
        return b""
    if not path.exists():
        return b""
    try:
        with path.open("rb") as handle:
            return handle.read(_OPUS_HEADER_SCAN_BYTES)
    except OSError:
        return b""


def _path_declares_opus(audio_path: str | None) -> bool:
    if not audio_path:
        return False
    return Path(urlparse(audio_path).path).suffix.lower() == ".opus"


def _record_looks_like_opus(record: CVRecord) -> bool:
    if _path_declares_opus(record.audio_path):
        return True
    if record.audio_bytes is not None and _audio_header_looks_like_opus(record.audio_bytes):
        return True
    return _audio_header_looks_like_opus(_read_header_from_path(record.audio_path))


def _find_opus_probe_record(
    records: list[CVRecord] | DiskBackedRecordStore,
) -> tuple[str | None, bytes | None, str] | None:
    if isinstance(records, DiskBackedRecordStore):
        with records.records_path.open("rb") as handle:
            for global_index in range(records.start, len(records.offsets), records.step):
                handle.seek(records.offsets[global_index])
                payload = json.loads(handle.readline().decode("utf-8"))
                audio_path = payload.get("audio_path")
                if not isinstance(audio_path, str):
                    audio_path = None
                audio_blob_path = payload.get("audio_blob_path")
                header = b""
                if isinstance(audio_blob_path, str) and audio_blob_path:
                    blob_path = Path(audio_blob_path)
                    if not blob_path.is_absolute():
                        blob_path = records.records_path.parent / blob_path
                    header = _read_header_from_path(str(blob_path))
                elif audio_path is not None:
                    header = _read_header_from_path(audio_path)
                if _path_declares_opus(audio_path) or _audio_header_looks_like_opus(header):
                    return (
                        audio_path,
                        _load_cached_audio_bytes(payload, records_path=records.records_path),
                        audio_path or f"blob:{audio_blob_path}",
                    )
        return None

    for record in records:
        if _record_looks_like_opus(record):
            return record.audio_path, record.audio_bytes, record.audio_path or record.utterance_id
    return None


def _ensure_opus_decode_support(
    records: list[CVRecord] | DiskBackedRecordStore,
    *,
    split: str,
) -> None:
    probe = _find_opus_probe_record(records)
    if probe is None:
        return
    audio_path, audio_bytes, description = probe
    try:
        load_audio(audio_path, audio_bytes)
    except Exception as error:
        raise RuntimeError(
            f"Split '{split}' contains Opus audio ({description}), but this runtime cannot decode "
            "it with torchaudio. Install torchaudio with Opus/FFmpeg support or convert those "
            "files to WAV before training."
        ) from error


def _load_records_from_dataset_roots(
    dataset_sources: list[str | Path],
    *,
    split: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    max_samples: int | None,
    min_transcript_chars: int,
    max_transcript_chars: int,
    max_symbol_ratio: float,
    lowercase_transcripts: bool,
    min_audio_duration_sec: float = 0.01,
    max_audio_duration_sec: float = 30.0,
    hf_token: str | None = None,
    cache_dir: str | None = None,
) -> list:
    records = []
    for dataset_source in dataset_sources:
        remaining_samples = None
        if max_samples is not None:
            remaining_samples = max_samples - len(records)
            if remaining_samples <= 0:
                break
        iterator = (
            iter_cv22_records(
                dataset_root=dataset_source,
                split=split,
                seed=seed,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                max_samples=remaining_samples,
                min_transcript_chars=min_transcript_chars,
                max_transcript_chars=max_transcript_chars,
                max_symbol_ratio=max_symbol_ratio,
                min_audio_duration_sec=min_audio_duration_sec,
                max_audio_duration_sec=max_audio_duration_sec,
                lowercase_transcripts=lowercase_transcripts,
            )
            if isinstance(dataset_source, Path)
            and dataset_source.exists()
            and dataset_source.is_dir()
            else iter_cv22_records_from_source(
                dataset_source,
                split=split,
                seed=seed,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                max_samples=remaining_samples,
                min_transcript_chars=min_transcript_chars,
                max_transcript_chars=max_transcript_chars,
                max_symbol_ratio=max_symbol_ratio,
                min_audio_duration_sec=min_audio_duration_sec,
                max_audio_duration_sec=max_audio_duration_sec,
                lowercase_transcripts=lowercase_transcripts,
                hf_token=hf_token,
                cache_dir=cache_dir,
            )
        )
        records.extend(iterator)
    if not records:
        raise RuntimeError(
            f"Split '{split}' is empty after applying the current split fractions across "
            "all dataset sources."
        )
    return records


def _prevalidate_loaded_records(
    records: list[CVRecord],
    *,
    split: str,
    num_workers: int,
) -> list[CVRecord]:
    validated_records = prevalidate_records(records, num_workers=num_workers)
    if not validated_records:
        raise RuntimeError(f"Split '{split}' is empty after audio prevalidation.")
    return validated_records


def _load_train_val_records(
    args: argparse.Namespace,
    train_dataset_sources: list[str | Path],
    validation_dataset_sources: list[str | Path] | None = None,
    *,
    lowercase_transcripts: bool,
    output_dir: Path,
) -> tuple[list[CVRecord] | DiskBackedRecordStore, list[CVRecord] | DiskBackedRecordStore]:
    validation_dataset_sources = validation_dataset_sources or []
    use_external_validation = bool(validation_dataset_sources)
    train_val_fraction = 0.0 if use_external_validation else args.val_fraction
    train_test_fraction = 0.0 if use_external_validation else args.test_fraction
    validation_split = "train" if use_external_validation else "validation"
    validation_val_fraction = 0.0 if use_external_validation else args.val_fraction
    validation_test_fraction = 0.0 if use_external_validation else args.test_fraction

    if args.record_cache:
        common_record_store_kwargs = {
            "seed": args.seed,
            "min_transcript_chars": args.min_transcript_chars,
            "max_transcript_chars": args.max_transcript_chars,
            "max_symbol_ratio": args.max_symbol_ratio,
            "lowercase_transcripts": lowercase_transcripts,
            "hf_token": args.hf_token,
            "cache_dir": args.cache_dir,
        }
        if hasattr(args, "min_audio_duration_sec"):
            common_record_store_kwargs["min_audio_duration_sec"] = args.min_audio_duration_sec
        if hasattr(args, "max_audio_duration_sec"):
            common_record_store_kwargs["max_audio_duration_sec"] = args.max_audio_duration_sec
        record_store_dir = (
            Path(args.record_cache_dir)
            if args.record_cache_dir is not None
            else output_dir / "record_cache"
        )
        train_records = _build_disk_backed_record_store(
            train_dataset_sources,
            split="train",
            val_fraction=train_val_fraction,
            test_fraction=train_test_fraction,
            max_samples=args.max_train_samples,
            records_path=record_store_dir / "train.jsonl",
            **common_record_store_kwargs,
        )
        val_records = _build_disk_backed_record_store(
            validation_dataset_sources or train_dataset_sources,
            split=validation_split,
            val_fraction=validation_val_fraction,
            test_fraction=validation_test_fraction,
            max_samples=args.max_val_samples,
            records_path=record_store_dir / "validation.jsonl",
            **common_record_store_kwargs,
        )
        if args.prevalidate_audio:
            raise ValueError(
                "--prevalidate-audio is not supported with the disk-backed training record store. "
                "Leave it disabled for large multi-source training runs or use --no-record-cache."
            )
        return train_records, val_records

    common_load_kwargs = {
        "seed": args.seed,
        "min_transcript_chars": args.min_transcript_chars,
        "max_transcript_chars": args.max_transcript_chars,
        "max_symbol_ratio": args.max_symbol_ratio,
        "lowercase_transcripts": lowercase_transcripts,
        "hf_token": args.hf_token,
        "cache_dir": args.cache_dir,
    }
    if hasattr(args, "min_audio_duration_sec"):
        common_load_kwargs["min_audio_duration_sec"] = args.min_audio_duration_sec
    if hasattr(args, "max_audio_duration_sec"):
        common_load_kwargs["max_audio_duration_sec"] = args.max_audio_duration_sec

    train_records = _load_records_from_dataset_roots(
        train_dataset_sources,
        split="train",
        val_fraction=train_val_fraction,
        test_fraction=train_test_fraction,
        max_samples=args.max_train_samples,
        **common_load_kwargs,
    )
    val_records = _load_records_from_dataset_roots(
        validation_dataset_sources or train_dataset_sources,
        split=validation_split,
        val_fraction=validation_val_fraction,
        test_fraction=validation_test_fraction,
        max_samples=args.max_val_samples,
        **common_load_kwargs,
    )
    if args.prevalidate_audio:
        train_records = _prevalidate_loaded_records(
            train_records,
            split="train",
            num_workers=args.prevalidate_workers,
        )
        val_records = _prevalidate_loaded_records(
            val_records,
            split="validation",
            num_workers=args.prevalidate_workers,
        )
    return train_records, val_records


def _shard_records_for_rank(
    records: list[CVRecord] | DiskBackedRecordStore,
    *,
    rank: int,
    world_size: int,
) -> list[CVRecord] | DiskBackedRecordStore:
    if world_size <= 1:
        return records
    if hasattr(records, "shard"):
        return records.shard(rank, world_size)
    return records[rank::world_size]


def _record_store_duration_hours(
    records: list[CVRecord] | DiskBackedRecordStore,
    *,
    hop_length: int,
    sample_rate: int = 16000,
) -> float:
    if sample_rate <= 0:
        return 0.0
    total_frames = 0
    if hasattr(records, "estimated_frames"):
        total_frames = sum(int(value) for value in records.estimated_frames)
    else:
        total_frames = sum(max(0, int(record.estimated_frames)) for record in records)
    total_seconds = (float(total_frames) * float(hop_length)) / float(sample_rate)
    return total_seconds / 3600.0


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
) -> tuple[bool, str, float, int]:
    checkpoint_args = checkpoint.get("training_args", {}) if checkpoint is not None else {}
    checkpoint_enabled = checkpoint_args.get("liberta_distill")

    if args.liberta_distill is False:
        return (
            False,
            args.liberta_model_name,
            float(args.liberta_distill_weight),
            int(args.liberta_max_length),
        )
    if args.liberta_distill is None and checkpoint is not None and checkpoint_enabled is not None:
        enabled = bool(checkpoint_enabled)
        model_name = str(checkpoint_args.get("liberta_model_name", args.liberta_model_name))
        weight = float(checkpoint_args.get("liberta_distill_weight", args.liberta_distill_weight))
        max_length = int(checkpoint_args.get("liberta_max_length", args.liberta_max_length))
    else:
        enabled = bool(args.liberta_distill) if args.liberta_distill is not None else False
        model_name = args.liberta_model_name
        weight = float(args.liberta_distill_weight)
        max_length = int(args.liberta_max_length)

    if enabled and not aed_enabled:
        raise ValueError("LiBERTa distillation requires --aed-decoder.")
    if not enabled:
        return False, model_name, weight, max_length
    if weight <= 0.0:
        raise ValueError(
            "--liberta-distill-weight must be > 0 when LiBERTa distillation is enabled."
        )
    return True, model_name, weight, max_length


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
    loss = F.cross_entropy(
        logits.transpose(1, 2),
        targets,
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
        optimizers.append(
            muon_cls(
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


def _build_split_audit(
    split_records: dict[str, list[CVRecord] | DiskBackedRecordStore],
    *,
    hop_length: int,
) -> dict[str, object]:
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
            "hours": _record_store_duration_hours(records, hop_length=hop_length),
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
        "hours": {
            split_name: item["hours"] for split_name, item in counts.items()
        },
        "total_hours": sum(float(item["hours"]) for item in counts.values()),
        "speaker_overlaps": overlaps,
        "speaker_balance_ratio": balance_ratio,
        "speaker_id_available": all(
            item["records_with_speaker_id"] == item["samples"] for item in counts.values()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Squeezeformer CTC on speech-uk/cv22.")
    parser.add_argument("--dataset-repo", default="speech-uk/cv22")
    parser.add_argument(
        "--dataset-source",
        action="append",
        default=None,
        help=(
            "Dataset source to load. Repeat to combine multiple sources. Each source may be a "
            "Hugging Face dataset repo, a direct TSV/Parquet file path or URL, or a local "
            "directory with Common Voice-style TSV or Parquet manifests plus audio files. "
            "If omitted, --dataset-repo is used."
        ),
    )
    parser.add_argument(
        "--validation-dataset-source",
        action="append",
        default=None,
        help=(
            "Validation-only dataset source. Repeat to combine multiple sources. Each source may "
            "be a Hugging Face dataset repo, a direct TSV/Parquet file path or URL, or a local "
            "directory with Common Voice-style TSV or Parquet manifests plus audio files. When "
            "provided, the full set of records from these sources is used for validation and "
            "--dataset-source is consumed in full for training without train/validation/test "
            "splitting."
        ),
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--record-cache-dir",
        default=None,
        help=(
            "Directory for disk-backed train/validation record indexes. Defaults to "
            "OUTPUT_DIR/record_cache."
        ),
    )
    parser.add_argument(
        "--record-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to build disk-backed train/validation record indexes. Disable with "
            "--no-record-cache to keep records only in memory."
        ),
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help=(
            "Resume from OUTPUT_DIR/checkpoint_last.pt when it exists and passes validation. "
            "Starts a fresh run when no last checkpoint is available."
        ),
    )
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
    parser.add_argument("--min-audio-duration-sec", type=float, default=0.01)
    parser.add_argument("--max-audio-duration-sec", type=float, default=30.0)
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
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "Path to an existing tokenizer artifact to reuse instead of building one from the "
            "training transcripts. Supports tokenizer JSON files and SentencePiece .model files."
        ),
    )
    parser.add_argument("--spm-vocab-size", type=int, default=4096)
    parser.add_argument(
        "--spm-model-type",
        default="unigram",
        choices=["unigram", "bpe", "char", "word"],
    )
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--hold-epochs", type=int, default=160)
    parser.add_argument("--decay-exponent", type=float, default=1.0)
    parser.add_argument("--intermediate-ctc-layer", type=int, default=None)
    parser.add_argument("--intermediate-ctc-layers", default=None)
    parser.add_argument(
        "--intermediate-ctc",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--intermediate-ctc-weight", type=float, default=0.3)
    parser.add_argument(
        "--aed-decoder",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--aed-decoder-layers", type=int, default=1)
    parser.add_argument("--aed-decoder-heads", type=int, default=4)
    parser.add_argument("--aed-decoder-dropout", type=float, default=0.1)
    parser.add_argument("--aed-loss-weight", type=float, default=0.3)
    parser.add_argument(
        "--liberta-distill",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--liberta-model-name", default="Goader/liberta-large-v2")
    parser.add_argument("--liberta-distill-weight", type=float, default=0.05)
    parser.add_argument("--liberta-max-length", type=int, default=256)
    parser.add_argument("--blank-prune-layer", type=int, default=None)
    parser.add_argument("--blank-prune-threshold", type=float, default=0.0)
    parser.add_argument("--blank-prune-min-keep-frames", type=int, default=1)
    parser.add_argument(
        "--blank-prune",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
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
            "Attention implementation. 'flash' prefers kernels-community/flash-attn2 on "
            "supported CUDA setups and falls back to PyTorch scaled_dot_product_attention."
        ),
    )
    parser.add_argument("--block-pattern", default="M,s,C,s")
    parser.add_argument(
        "--frontend-backend",
        default="torchaudio",
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
    parser.add_argument(
        "--no-data-augmentation",
        action="store_true",
        help=(
            "Disable SpecAugment and waveform augmentation regardless of the configured "
            "augmentation parameters."
        ),
    )
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
    intermediate_ctc_weight: float = 0.0,
    aed_loss_weight: float = 0.0,
    liberta_teacher: FrozenLibertaTeacher | None = None,
    liberta_distill_weight: float = 0.0,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_main_ctc_loss = 0.0
    total_intermediate_ctc_loss = 0.0
    total_combined_ctc_loss = 0.0
    total_aed_loss = 0.0
    total_liberta_distill_loss = 0.0
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
            if model.aed_decoder is not None:
                decoder_inputs, decoder_targets, decoder_target_lengths = _build_aed_targets(
                    targets,
                    target_lengths,
                    bos_id=model.aed_decoder.bos_id,
                    eos_id=model.aed_decoder.eos_id,
                    token_offset=model.aed_decoder.token_offset,
                    pad_id=model.aed_decoder.pad_id,
                )
                decoder_inputs = decoder_inputs.to(device)
                decoder_targets = decoder_targets.to(device)
                decoder_target_lengths = decoder_target_lengths.to(device)
            else:
                decoder_inputs = None
                decoder_targets = None
                decoder_target_lengths = None

            with _autocast_context(device, dtype, fp8_recipe=fp8_recipe):
                log_probs, output_lengths, intermediate_log_probs, intermediate_output_lengths = (
                    model.log_probs_with_intermediate(features, feature_lengths)
                )
                main_ctc_loss = criterion(
                    log_probs.transpose(0, 1),
                    targets,
                    output_lengths,
                    target_lengths,
                )
                if intermediate_log_probs and intermediate_output_lengths:
                    intermediate_ctc_losses = [
                        criterion(
                            intermediate_log_probs[layer_index].transpose(0, 1),
                            targets,
                            intermediate_output_lengths[layer_index],
                            target_lengths,
                        )
                        for layer_index in model.intermediate_ctc_layers
                    ]
                    intermediate_ctc_loss = torch.stack(intermediate_ctc_losses).mean()
                    combined_ctc_loss = (
                        1.0 - intermediate_ctc_weight
                    ) * main_ctc_loss + intermediate_ctc_weight * intermediate_ctc_loss
                else:
                    intermediate_ctc_loss = None
                    combined_ctc_loss = main_ctc_loss
                if decoder_inputs is not None and decoder_targets is not None:
                    aed_logits, _, aed_hidden = model.aed_forward(
                        features,
                        feature_lengths,
                        decoder_inputs,
                    )
                    aed_loss = _aed_cross_entropy_loss(
                        aed_logits,
                        decoder_targets,
                        pad_id=model.aed_decoder.pad_id,
                    )
                    loss = (1.0 - aed_loss_weight) * combined_ctc_loss + aed_loss_weight * aed_loss
                else:
                    aed_loss = None
                    aed_hidden = None
                    loss = combined_ctc_loss
            if (
                liberta_teacher is not None
                and aed_hidden is not None
                and decoder_target_lengths is not None
            ):
                teacher_embeddings = liberta_teacher.encode(batch["transcripts"])
                student_embeddings = model.project_aed_hidden_for_liberta(
                    aed_hidden,
                    decoder_target_lengths,
                )
                liberta_distill_loss = F.mse_loss(
                    F.normalize(student_embeddings, dim=-1),
                    F.normalize(teacher_embeddings, dim=-1),
                )
                loss = loss + (liberta_distill_weight * liberta_distill_loss)
            else:
                liberta_distill_loss = None
            total_loss += float(loss.item())
            total_main_ctc_loss += float(main_ctc_loss.item())
            total_intermediate_ctc_loss += float(
                intermediate_ctc_loss.item() if intermediate_ctc_loss is not None else 0.0
            )
            total_combined_ctc_loss += float(combined_ctc_loss.item())
            total_aed_loss += float(aed_loss.item() if aed_loss is not None else 0.0)
            total_liberta_distill_loss += float(
                liberta_distill_loss.item() if liberta_distill_loss is not None else 0.0
            )
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
    metrics = {
        "loss": total_loss / max(1, total_batches),
        "main_ctc_loss": total_main_ctc_loss / max(1, total_batches),
        "intermediate_ctc_loss": total_intermediate_ctc_loss / max(1, total_batches),
        "combined_ctc_loss": total_combined_ctc_loss / max(1, total_batches),
        "aed_loss": total_aed_loss / max(1, total_batches),
        "liberta_distill_loss": total_liberta_distill_loss / max(1, total_batches),
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
    process_start_time = time.perf_counter()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _validate_startup_args(args, world_size=world_size)
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested_device = resolve_device(args.device)
    is_main_process = rank == 0
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = output_dir / (
        f"training_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = _configure_console_logger(
        rank=rank,
        is_main_process=is_main_process,
        log_path=training_log_path,
    )
    trackio_dir = _configure_trackio_storage(output_dir)
    resume_path = _resolve_resume_checkpoint_path(args, output_dir=output_dir, logger=logger)
    logger.info(
        "starting training variant=%s device=%s distributed=%s world_size=%s output_dir=%s",
        args.variant,
        requested_device,
        distributed,
        world_size,
        output_dir,
    )
    variant_defaults = _variant_defaults(args.variant)
    stage_start_time = time.perf_counter()
    logger.info("loading LM scorer source=%s", args.lm_scorer or "disabled")
    lm_scorer = load_lm_scorer(args.lm_scorer)
    logger.info(
        "LM scorer ready source=%s elapsed=%s",
        args.lm_scorer or "disabled",
        _format_elapsed_seconds(time.perf_counter() - stage_start_time),
    )

    stage_start_time = time.perf_counter()
    logger.info("resolving dataset sources")
    train_dataset_sources = _resolve_dataset_sources(args)
    validation_dataset_sources = _resolve_validation_dataset_sources(args)
    lowercase_transcripts = args.tokenizer != "sentencepiece"
    logger.info(
        "dataset sources resolved train_count=%s validation_count=%s elapsed=%s",
        len(train_dataset_sources),
        len(validation_dataset_sources),
        _format_elapsed_seconds(time.perf_counter() - stage_start_time),
    )
    stage_start_time = time.perf_counter()
    logger.info(
        "loading dataset records train_sources=%s validation_sources=%s validation_mode=%s record_cache=%s prevalidate_audio=%s",
        [str(dataset_source) for dataset_source in train_dataset_sources],
        [str(dataset_source) for dataset_source in validation_dataset_sources]
        if validation_dataset_sources
        else [str(dataset_source) for dataset_source in train_dataset_sources],
        "external" if validation_dataset_sources else "split",
        args.record_cache,
        args.prevalidate_audio,
    )
    train_records, val_records = _load_train_val_records(
        args,
        train_dataset_sources,
        validation_dataset_sources,
        lowercase_transcripts=lowercase_transcripts,
        output_dir=output_dir,
    )
    _ensure_opus_decode_support(train_records, split="train")
    _ensure_opus_decode_support(val_records, split="validation")
    split_audit = _build_split_audit(
        {"train": train_records, "validation": val_records},
        hop_length=args.hop_length,
    )
    if is_main_process:
        (output_dir / "split_audit.json").write_text(
            json.dumps(split_audit, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "loaded datasets train_sources=%s validation_sources=%s train_samples=%s val_samples=%s speaker_balance_ratio=%.3f elapsed=%s",
            [str(dataset_source) for dataset_source in train_dataset_sources],
            [str(dataset_source) for dataset_source in validation_dataset_sources]
            if validation_dataset_sources
            else [str(dataset_source) for dataset_source in train_dataset_sources],
            len(train_records),
            len(val_records),
            float(split_audit["speaker_balance_ratio"]),
            _format_elapsed_seconds(time.perf_counter() - stage_start_time),
        )

    stage_start_time = time.perf_counter()
    checkpoint = (
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_path is not None
        else None
    )
    if resume_path is not None:
        _validate_resume_checkpoint_payload(checkpoint, checkpoint_path=resume_path)
        logger.info(
            "resume checkpoint loaded path=%s elapsed=%s",
            resume_path,
            _format_elapsed_seconds(time.perf_counter() - stage_start_time),
        )
    stage_start_time = time.perf_counter()
    logger.info(
        "preparing tokenizer mode=%s resume=%s tokenizer_path=%s",
        args.tokenizer,
        checkpoint is not None,
        args.tokenizer_path or "auto",
    )
    if checkpoint is not None:
        tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    elif args.tokenizer_path is not None:
        tokenizer = load_tokenizer(args.tokenizer_path)
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
    logger.info(
        "tokenizer ready vocab_size=%s elapsed=%s",
        tokenizer.vocab_size,
        _format_elapsed_seconds(time.perf_counter() - stage_start_time),
    )
    if args.fit_shallow_fusion_lm:
        stage_start_time = time.perf_counter()
        logger.info(
            "training shallow-fusion LM order=%s alpha=%.3f",
            args.shallow_fusion_lm_order,
            args.shallow_fusion_lm_alpha,
        )
        shallow_fusion_lm = NGramLanguageModel.train(
            (record.transcript for record in train_records),
            order=args.shallow_fusion_lm_order,
            alpha=args.shallow_fusion_lm_alpha,
        )
        shallow_fusion_lm_path = output_dir / "shallow_fusion_lm.json"
        shallow_fusion_lm.save(shallow_fusion_lm_path)
        if lm_scorer is None:
            lm_scorer = shallow_fusion_lm.score_extension
        logger.info(
            "shallow-fusion LM ready path=%s elapsed=%s",
            shallow_fusion_lm_path,
            _format_elapsed_seconds(time.perf_counter() - stage_start_time),
        )
    if hasattr(train_records, "close"):
        train_records.close()
    if hasattr(val_records, "close"):
        val_records.close()

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
    specaugment = None
    waveform_augment = None
    if not args.no_data_augmentation:
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
    local_train_records = _shard_records_for_rank(
        train_records,
        rank=rank,
        world_size=world_size,
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
    stage_start_time = time.perf_counter()
    logger.info(
        "building dataloaders train_shard_samples=%s val_samples=%s train_hours=%.2f val_hours=%.2f num_workers=%s metadata_workers=%s persistent_workers=%s prefetch_factor=%s",
        len(local_train_records),
        len(val_records),
        _record_store_duration_hours(local_train_records, hop_length=args.hop_length),
        _record_store_duration_hours(val_records, hop_length=args.hop_length),
        args.num_workers,
        args.metadata_workers,
        args.persistent_workers,
        args.prefetch_factor,
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
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    logger.info(
        "dataloaders ready train_batches=%s val_batches=%s elapsed=%s",
        train_batches,
        val_batches,
        _format_elapsed_seconds(time.perf_counter() - stage_start_time),
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
    intermediate_ctc_layers, intermediate_ctc_weight = _resolve_intermediate_ctc_settings(
        args,
        encoder_config,
        checkpoint,
    )
    (
        aed_decoder_enabled,
        aed_decoder_layers,
        aed_decoder_heads,
        aed_decoder_dropout,
        aed_loss_weight,
    ) = _resolve_aed_settings(args, checkpoint)
    liberta_distill_enabled, liberta_model_name, liberta_distill_weight, liberta_max_length = (
        _resolve_liberta_settings(
            args,
            checkpoint,
            aed_enabled=aed_decoder_enabled,
        )
    )
    blank_prune_layer, blank_prune_threshold, blank_prune_min_keep_frames = (
        _resolve_blank_pruning_settings(args, encoder_config, checkpoint)
    )
    args.intermediate_ctc_layers = list(intermediate_ctc_layers)
    args.intermediate_ctc_layer = (
        intermediate_ctc_layers[0] if len(intermediate_ctc_layers) == 1 else None
    )
    args.intermediate_ctc = bool(intermediate_ctc_layers) and intermediate_ctc_weight > 0.0
    args.intermediate_ctc_weight = intermediate_ctc_weight
    args.aed_decoder = aed_decoder_enabled
    args.aed_decoder_layers = aed_decoder_layers
    args.aed_decoder_heads = aed_decoder_heads
    args.aed_decoder_dropout = aed_decoder_dropout
    args.aed_loss_weight = aed_loss_weight
    args.liberta_distill = liberta_distill_enabled
    args.liberta_model_name = liberta_model_name
    args.liberta_distill_weight = liberta_distill_weight
    args.liberta_max_length = liberta_max_length
    args.blank_prune = blank_prune_layer is not None and blank_prune_threshold > 0.0
    args.blank_prune_layer = blank_prune_layer
    args.blank_prune_threshold = blank_prune_threshold
    args.blank_prune_min_keep_frames = blank_prune_min_keep_frames
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
    stage_start_time = time.perf_counter()
    logger.info(
        "building model variant=%s dtype=%s compile=%s intermediate_ctc_layers=%s aed=%s liberta=%s blank_prune_layer=%s",
        args.variant,
        args.dtype,
        args.compile,
        list(intermediate_ctc_layers),
        aed_decoder_enabled,
        liberta_distill_enabled,
        blank_prune_layer,
    )
    model = SqueezeformerCTC(
        encoder_config=encoder_config,
        vocab_size=tokenizer.vocab_size,
        intermediate_ctc_layers=intermediate_ctc_layers,
        blank_prune_layer=blank_prune_layer,
        blank_prune_threshold=blank_prune_threshold,
        blank_prune_min_keep_frames=blank_prune_min_keep_frames,
        aed_decoder_enabled=aed_decoder_enabled,
        aed_decoder_layers=aed_decoder_layers,
        aed_decoder_heads=aed_decoder_heads,
        aed_decoder_dropout=aed_decoder_dropout,
        liberta_distill_enabled=liberta_distill_enabled,
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
    liberta_teacher = (
        FrozenLibertaTeacher(
            liberta_model_name,
            device=device,
            max_length=liberta_max_length,
        )
        if liberta_distill_enabled
        else None
    )
    logger.info(
        "model and auxiliaries ready params=%s elapsed=%s",
        sum(parameter.numel() for parameter in model.parameters()),
        _format_elapsed_seconds(time.perf_counter() - stage_start_time),
    )
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
            "resumed from %s starting_epoch=%s global_step=%s best_val_wer=%.4f intermediate_ctc_layers=%s intermediate_ctc_weight=%.3f blank_prune_layer=%s blank_prune_threshold=%.3f",
            resume_path,
            start_epoch,
            global_step,
            best_val_wer,
            list(intermediate_ctc_layers),
            intermediate_ctc_weight,
            blank_prune_layer,
            blank_prune_threshold,
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
                "intermediate_ctc_layers": list(intermediate_ctc_layers),
                "intermediate_ctc_layer": (
                    intermediate_ctc_layers[0] if len(intermediate_ctc_layers) == 1 else None
                ),
                "intermediate_ctc_weight": intermediate_ctc_weight,
                "blank_prune_layer": blank_prune_layer,
                "blank_prune_threshold": blank_prune_threshold,
                "blank_prune_min_keep_frames": blank_prune_min_keep_frames,
                "split_audit": split_audit,
                "distributed": distributed,
                "world_size": world_size,
            },
        )
        logger.info(
            "trackio initialized project=%s trackio_dir=%s elapsed_since_start=%s",
            args.trackio_project,
            trackio_dir,
            _format_elapsed_seconds(time.perf_counter() - process_start_time),
        )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        logger.info(
            "epoch %s/%s started train_batches=%s val_batches=%s grad_accumulation=%s",
            epoch,
            args.epochs,
            train_batches,
            val_batches,
            args.gradient_accumulation_steps,
        )
        forward_model.train()
        running_loss = 0.0
        running_main_ctc_loss = 0.0
        running_intermediate_ctc_loss = 0.0
        running_aed_loss = 0.0
        running_liberta_distill_loss = 0.0
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            if is_main_process and batch_index == 1:
                logger.info(
                    "epoch=%s first_train_batch_ready elapsed=%s batch_size=%s max_feature_frames=%s target_tokens=%s",
                    epoch,
                    _format_elapsed_seconds(time.perf_counter() - epoch_start_time),
                    int(features.size(0)),
                    int(feature_lengths.max().item()),
                    int(target_lengths.sum().item()),
                )
            if aed_decoder_enabled:
                if model.aed_decoder is None:
                    raise RuntimeError("AED decoder was enabled but not constructed on the model.")
                decoder_inputs, decoder_targets, decoder_target_lengths = _build_aed_targets(
                    targets,
                    target_lengths,
                    bos_id=model.aed_decoder.bos_id,
                    eos_id=model.aed_decoder.eos_id,
                    token_offset=model.aed_decoder.token_offset,
                    pad_id=model.aed_decoder.pad_id,
                )
                decoder_inputs = decoder_inputs.to(device)
                decoder_targets = decoder_targets.to(device)
                decoder_target_lengths = decoder_target_lengths.to(device)
            else:
                decoder_inputs = None
                decoder_targets = None
                decoder_target_lengths = None

            with _autocast_context(device, args.dtype, fp8_recipe=fp8_recipe):
                log_probs, output_lengths, intermediate_log_probs, intermediate_output_lengths = (
                    forward_model.log_probs_with_intermediate(features, feature_lengths)
                )
                main_ctc_loss = criterion(
                    log_probs.transpose(0, 1),
                    targets,
                    output_lengths,
                    target_lengths,
                )
                if intermediate_log_probs and intermediate_output_lengths:
                    intermediate_ctc_losses = [
                        criterion(
                            intermediate_log_probs[layer_index].transpose(0, 1),
                            targets,
                            intermediate_output_lengths[layer_index],
                            target_lengths,
                        )
                        for layer_index in intermediate_ctc_layers
                    ]
                    intermediate_ctc_loss = torch.stack(intermediate_ctc_losses).mean()
                    loss = (
                        1.0 - intermediate_ctc_weight
                    ) * main_ctc_loss + intermediate_ctc_weight * intermediate_ctc_loss
                else:
                    intermediate_ctc_loss = None
                    loss = main_ctc_loss
                if (
                    aed_decoder_enabled
                    and decoder_inputs is not None
                    and decoder_targets is not None
                ):
                    aed_logits, _, aed_hidden = forward_model.aed_forward(
                        features,
                        feature_lengths,
                        decoder_inputs,
                    )
                    aed_loss = _aed_cross_entropy_loss(
                        aed_logits,
                        decoder_targets,
                        pad_id=model.aed_decoder.pad_id,
                    )
                    loss = (1.0 - args.aed_loss_weight) * loss + args.aed_loss_weight * aed_loss
                else:
                    aed_loss = None
                    aed_hidden = None
            if (
                liberta_teacher is not None
                and aed_hidden is not None
                and decoder_target_lengths is not None
            ):
                teacher_embeddings = liberta_teacher.encode(batch["transcripts"])
                student_embeddings = forward_model.project_aed_hidden_for_liberta(
                    aed_hidden,
                    decoder_target_lengths,
                )
                liberta_distill_loss = F.mse_loss(
                    F.normalize(student_embeddings, dim=-1),
                    F.normalize(teacher_embeddings, dim=-1),
                )
                loss = loss + (args.liberta_distill_weight * liberta_distill_loss)
            else:
                liberta_distill_loss = None
            running_loss += float(loss.item())
            running_main_ctc_loss += float(main_ctc_loss.item())
            running_intermediate_ctc_loss += float(
                intermediate_ctc_loss.item() if intermediate_ctc_loss is not None else 0.0
            )
            running_aed_loss += float(aed_loss.item() if aed_loss is not None else 0.0)
            running_liberta_distill_loss += float(
                liberta_distill_loss.item() if liberta_distill_loss is not None else 0.0
            )
            scaler.scale(loss / args.gradient_accumulation_steps).backward()

            should_step = batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(
                train_loader
            )
            if should_step:
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)
                grad_norm = float(_compute_grad_norm(model.parameters()).item())
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
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

                if is_main_process and global_step % args.log_every == 0:
                    learning_rates = {
                        f"learning_rate_{name}": optimizer.param_groups[0]["lr"]
                        for name, optimizer in zip(optimizer_names, optimizers, strict=True)
                    }
                    memory_snapshot = _format_memory_snapshot(device)
                    logger.info(
                        (
                            "epoch=%s step=%s/%s global_step=%s train_loss=%.4f "
                            "train_main_ctc_loss=%.4f train_intermediate_ctc_loss=%.4f "
                            "train_aed_loss=%.4f train_liberta_distill_loss=%.4f "
                            "grad_norm=%.4f %s %s"
                        ),
                        epoch,
                        batch_index,
                        len(train_loader),
                        global_step,
                        float(loss.item()),
                        float(main_ctc_loss.item()),
                        float(
                            intermediate_ctc_loss.item()
                            if intermediate_ctc_loss is not None
                            else 0.0
                        ),
                        float(aed_loss.item() if aed_loss is not None else 0.0),
                        float(
                            liberta_distill_loss.item() if liberta_distill_loss is not None else 0.0
                        ),
                        grad_norm,
                        memory_snapshot,
                        " ".join(f"{name}={value:.6g}" for name, value in learning_rates.items()),
                    )
                    trackio.log(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "train_loss_step": float(loss.item()),
                            "train_main_ctc_loss_step": float(main_ctc_loss.item()),
                            "train_intermediate_ctc_loss_step": float(
                                intermediate_ctc_loss.item()
                                if intermediate_ctc_loss is not None
                                else 0.0
                            ),
                            "train_aed_loss_step": float(
                                aed_loss.item() if aed_loss is not None else 0.0
                            ),
                            "train_liberta_distill_loss_step": float(
                                liberta_distill_loss.item()
                                if liberta_distill_loss is not None
                                else 0.0
                            ),
                            "grad_norm": grad_norm,
                            "ema_decay": ema.current_decay() if ema is not None else 0.0,
                            "cpu_rss_bytes": _read_proc_status_memory_bytes("VmRSS:") or 0,
                            "cpu_peak_rss_bytes": (
                                _read_proc_status_memory_bytes("VmHWM:")
                                or _peak_process_memory_bytes()
                                or 0
                            ),
                            "cuda_allocated_bytes": (
                                torch.cuda.memory_allocated(device)
                                if device.type == "cuda" and torch.cuda.is_available()
                                else 0
                            ),
                            "cuda_reserved_bytes": (
                                torch.cuda.memory_reserved(device)
                                if device.type == "cuda" and torch.cuda.is_available()
                                else 0
                            ),
                            "cuda_peak_allocated_bytes": (
                                torch.cuda.max_memory_allocated(device)
                                if device.type == "cuda" and torch.cuda.is_available()
                                else 0
                            ),
                            "cuda_peak_reserved_bytes": (
                                torch.cuda.max_memory_reserved(device)
                                if device.type == "cuda" and torch.cuda.is_available()
                                else 0
                            ),
                            **learning_rates,
                        }
                    )

        train_loss = running_loss / max(1, len(train_loader))
        train_main_ctc_loss = running_main_ctc_loss / max(1, len(train_loader))
        train_intermediate_ctc_loss = running_intermediate_ctc_loss / max(1, len(train_loader))
        train_aed_loss = running_aed_loss / max(1, len(train_loader))
        train_liberta_distill_loss = running_liberta_distill_loss / max(1, len(train_loader))
        if is_main_process:
            ema_backup = ema.apply_to(model) if ema is not None else None
            logger.info(
                "epoch %s training complete train_loss=%.4f train_main_ctc_loss=%.4f train_intermediate_ctc_loss=%.4f train_aed_loss=%.4f train_liberta_distill_loss=%.4f elapsed=%s, starting validation",
                epoch,
                train_loss,
                train_main_ctc_loss,
                train_intermediate_ctc_loss,
                train_aed_loss,
                train_liberta_distill_loss,
                _format_elapsed_seconds(time.perf_counter() - epoch_start_time),
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
                intermediate_ctc_weight=intermediate_ctc_weight,
                aed_loss_weight=args.aed_loss_weight,
                liberta_teacher=liberta_teacher,
                liberta_distill_weight=args.liberta_distill_weight,
            )
            if ema_backup is not None:
                ExponentialMovingAverage.restore(model, ema_backup)
            val_metrics = validation["metrics"]
            log_payload = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "train_main_ctc_loss": train_main_ctc_loss,
                "train_intermediate_ctc_loss": train_intermediate_ctc_loss,
                "train_aed_loss": train_aed_loss,
                "train_liberta_distill_loss": train_liberta_distill_loss,
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
                    "epoch %s complete train_loss=%.4f val_loss=%.4f "
                    "val_main_ctc_loss=%.4f val_intermediate_ctc_loss=%.4f "
                    "val_combined_ctc_loss=%.4f val_aed_loss=%.4f "
                    "val_liberta_distill_loss=%.4f val_cer=%.4f val_wer=%.4f "
                    "best_val_wer=%.4f report=%s latest=%s best=%s averaged=%s"
                ),
                epoch,
                train_loss,
                float(val_metrics["loss"]),
                float(val_metrics["main_ctc_loss"]),
                float(val_metrics["intermediate_ctc_loss"]),
                float(val_metrics["combined_ctc_loss"]),
                float(val_metrics["aed_loss"]),
                float(val_metrics["liberta_distill_loss"]),
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
