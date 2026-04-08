from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from squeezeformer_pytorch.runtime_types import (
    AdaptiveBatchUnit,
    DecodeStrategy,
    DTypeChoice,
    OptimizerChoice,
)
from squeezeformer_pytorch.training.runtime import _validate_device_ready, resolve_device


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
    if args.validate_every_steps is not None and args.validate_every_steps > 0:
        positive_int_arguments["--validate-every-steps"] = args.validate_every_steps
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
    if args.liberta_model_path is not None:
        _validate_existing_local_path_argument(
            "--liberta-model-path",
            args.liberta_model_path,
            expected="dir",
        )

    for source in args.dataset_source or []:
        _validate_existing_local_path_argument("--dataset-source", source)
    for source in args.validation_dataset_source or []:
        _validate_existing_local_path_argument("--validation-dataset-source", source)


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
        "--longest-batches-first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When using length-aware batching, emit the heaviest batches first so oversized "
            "frame budgets fail early."
        ),
    )
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
    parser.add_argument(
        "--memory-tune-steps",
        type=int,
        default=0,
        help=(
            "For the first N optimizer steps, log effective/padded batch frames and per-step "
            "peak CUDA memory to help tune adaptive_batch/max_batch_frames."
        ),
    )
    parser.add_argument(
        "--validate-every-steps",
        type=int,
        default=0,
        help=(
            "Run validation and write step-based checkpoints every N optimizer steps. "
            "Set to 0 to validate only at epoch end."
        ),
    )
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
    parser.add_argument(
        "--liberta-model-path",
        default=None,
        help=(
            "Local directory containing a LiBERTa model/tokenizer to load with "
            "transformers.from_pretrained(). Overrides --liberta-model-name when set."
        ),
    )
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
