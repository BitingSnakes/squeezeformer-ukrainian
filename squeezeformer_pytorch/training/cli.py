from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from squeezeformer_pytorch.runtime_types import (
    AdaptiveBatchUnit,
    DecodeStrategy,
    DTypeChoice,
    FeatureCacheFormat,
    OptimizerChoice,
    ValidationModelSource,
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


def _validate_startup_args(
    args: argparse.Namespace,
    *,
    world_size: int,
    explicit_batch_size: bool = False,
) -> None:
    if (args.adaptive_batch_unit is None) != (args.adaptive_batch_budget is None):
        raise ValueError("--adaptive-batch-unit and --adaptive-batch-budget must be set together.")
    dynamic_batching_modes = [
        flag
        for flag, enabled in (
            ("--max-batch-duration-sec", args.max_batch_duration_sec is not None),
            ("--max-batch-frames", args.max_batch_frames is not None),
            ("--adaptive-batch-unit/--adaptive-batch-budget", args.adaptive_batch_unit is not None),
        )
        if enabled
    ]
    if len(dynamic_batching_modes) > 1:
        raise ValueError(
            "Batching controls are mutually exclusive; choose only one of "
            "--max-batch-duration-sec, --max-batch-frames, or the adaptive batch options."
        )
    if explicit_batch_size and dynamic_batching_modes:
        raise ValueError(
            f"--batch-size cannot be combined with {dynamic_batching_modes[0]} because that "
            "batching mode ignores sample-count batching."
        )
    if args.distributed and world_size <= 1:
        raise ValueError("--distributed expects a torchrun-style environment with WORLD_SIZE > 1.")
    if world_size > 1 and args.compile:
        raise ValueError("--compile is not currently supported together with distributed training.")
    if args.zipformer:
        if args.aed_decoder:
            raise ValueError("--zipformer does not support the AED decoder.")
        if args.liberta_distill:
            raise ValueError("--zipformer does not support LiBERTa distillation.")
        if args.audio_teacher:
            raise ValueError("--zipformer does not support audio-teacher distillation.")
    if args.w2v_bert:
        if args.zipformer:
            raise ValueError("--w2v-bert cannot be combined with --zipformer.")
        if args.zipformer_transducer:
            raise ValueError("--w2v-bert does not support the Zipformer transducer objective.")
        if args.aed_decoder:
            raise ValueError("--w2v-bert does not support the AED decoder.")
        if args.liberta_distill:
            raise ValueError("--w2v-bert does not support LiBERTa distillation.")
        if args.audio_teacher:
            raise ValueError("--w2v-bert does not support audio-teacher distillation.")
    if args.zipformer_transducer and not args.zipformer:
        raise ValueError("--zipformer-transducer requires --zipformer.")

    requested_device = resolve_device(args.device)
    _validate_device_ready(requested_device)
    liberta_device = resolve_device(args.liberta_device)
    _validate_device_ready(liberta_device)
    audio_teacher_device = resolve_device(args.audio_teacher_device)
    _validate_device_ready(audio_teacher_device)

    positive_int_arguments = {
        "--batch-size": args.batch_size,
        "--epochs": args.epochs,
        "--gradient-accumulation-steps": args.gradient_accumulation_steps,
        "--metadata-workers": args.metadata_workers,
        "--prevalidate-workers": args.prevalidate_workers,
        "--log-every": args.log_every,
        "--keep-top-k": args.keep_top_k,
        "--beam-size": args.beam_size,
        "--zipformer-transducer-context-size": args.zipformer_transducer_context_size,
        "--zipformer-transducer-decoder-dim": args.zipformer_transducer_decoder_dim,
        "--zipformer-transducer-joiner-dim": args.zipformer_transducer_joiner_dim,
        "--zipformer-transducer-prune-range": args.zipformer_transducer_prune_range,
        "--zipformer-transducer-joiner-chunk-size": args.zipformer_transducer_joiner_chunk_size,
        "--spm-vocab-size": args.spm_vocab_size,
        "--warmup-epochs": args.warmup_epochs,
        "--hold-epochs": args.hold_epochs,
        "--aed-decoder-layers": args.aed_decoder_layers,
        "--aed-decoder-heads": args.aed_decoder_heads,
        "--liberta-max-length": args.liberta_max_length,
        "--audio-teacher-sample-rate": args.audio_teacher_sample_rate,
        "--fp8-amax-history-len": args.fp8_amax_history_len,
        "--example-limit": args.example_limit,
        "--n-fft": args.n_fft,
        "--hop-length": args.hop_length,
        "--n-mels": args.n_mels,
        "--num-freq-masks": args.num_freq_masks,
        "--freq-mask-param": args.freq_mask_param,
    }
    if args.prefetch_factor is not None:
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
        "--ema-warmup-steps": args.ema_warmup_steps,
        "--save-audio-preview-samples": args.save_audio_preview_samples,
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
        "--beam-length-bonus": args.beam_length_bonus,
    }
    for name, value in nonnegative_float_arguments.items():
        if value is not None and value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}.")
    if args.max_batch_duration_sec is not None and args.max_batch_duration_sec <= 0:
        raise ValueError(
            f"--max-batch-duration-sec must be > 0, got {args.max_batch_duration_sec}."
        )

    probability_arguments = {
        "--val-fraction": args.val_fraction,
        "--test-fraction": args.test_fraction,
        "--max-symbol-ratio": args.max_symbol_ratio,
        "--speed-perturb-prob": args.speed_perturb_prob,
        "--noise-prob": args.noise_prob,
        "--reverb-prob": args.reverb_prob,
        "--time-mask-max-ratio": args.time_mask_max_ratio,
        "--aed-decoder-dropout": args.aed_decoder_dropout,
        "--aed-loss-weight": args.aed_loss_weight,
        "--liberta-distill-weight": args.liberta_distill_weight,
        "--ema-decay": args.ema_decay,
        "--audio-teacher-weight": args.audio_teacher_weight,
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
    if args.max_chars_per_second < args.min_chars_per_second:
        raise ValueError(
            "--max-chars-per-second must be >= --min-chars-per-second, got "
            f"{args.max_chars_per_second} < {args.min_chars_per_second}."
        )
    if args.max_words_per_second < args.min_words_per_second:
        raise ValueError(
            "--max-words-per-second must be >= --min-words-per-second, got "
            f"{args.max_words_per_second} < {args.min_words_per_second}."
        )
    if args.max_duration_per_char < args.min_duration_per_char:
        raise ValueError(
            "--max-duration-per-char must be >= --min-duration-per-char, got "
            f"{args.max_duration_per_char} < {args.min_duration_per_char}."
        )
    if args.max_duration_per_word < args.min_duration_per_word:
        raise ValueError(
            "--max-duration-per-word must be >= --min-duration-per-word, got "
            f"{args.max_duration_per_word} < {args.min_duration_per_word}."
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
    if args.w2v_bert and _is_explicit_local_path(args.w2v_bert_model_name):
        _validate_existing_local_path_argument(
            "--w2v-bert-model-name",
            args.w2v_bert_model_name,
            expected="dir",
        )
    if args.w2v_bert_model_path is not None:
        _validate_existing_local_path_argument(
            "--w2v-bert-model-path",
            args.w2v_bert_model_path,
            expected="dir",
        )
    if args.liberta_model_path is not None:
        _validate_existing_local_path_argument(
            "--liberta-model-path",
            args.liberta_model_path,
            expected="dir",
        )
    if args.audio_teacher_model_path is not None:
        _validate_existing_local_path_argument(
            "--audio-teacher-model-path",
            args.audio_teacher_model_path,
            expected="dir",
        )
    if args.audio_teacher_max_seconds <= 0:
        raise ValueError(
            f"--audio-teacher-max-seconds must be > 0, got {args.audio_teacher_max_seconds}."
        )
    if getattr(args, "hf_upload_checkpoints", False) and not getattr(
        args, "hf_upload_repo_id", None
    ):
        raise ValueError("--hf-upload-repo-id is required when --hf-upload-checkpoints is set.")

    for source in args.dataset_source or []:
        _validate_existing_local_path_argument("--dataset-source", source)
    for source in args.validation_dataset_source or []:
        _validate_existing_local_path_argument("--validation-dataset-source", source)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
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
        "--hf-upload-token",
        default=os.environ.get("HF_UPLOAD_TOKEN"),
        help=(
            "Hugging Face token used for checkpoint uploads. Defaults to HF_UPLOAD_TOKEN, "
            "then falls back to --hf-token/HF_TOKEN when omitted."
        ),
    )
    parser.add_argument(
        "--hf-upload-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Upload checkpoint artifacts from OUTPUT_DIR to Hugging Face after checkpoint saves. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--hf-upload-repo-id",
        default=None,
        help="Destination Hugging Face repo ID for --hf-upload-checkpoints.",
    )
    parser.add_argument(
        "--hf-upload-repo-type",
        choices=["model", "dataset", "space"],
        default="model",
        help="Destination Hugging Face repo type.",
    )
    parser.add_argument(
        "--hf-upload-path-in-repo",
        default=None,
        help="Optional destination folder inside the Hugging Face repo.",
    )
    parser.add_argument(
        "--hf-upload-revision",
        default=None,
        help="Optional branch or revision to upload checkpoint artifacts to.",
    )
    parser.add_argument(
        "--hf-upload-checkpoint-format",
        choices=["pt", "safetensors", "all"],
        default="all",
        help=(
            "Checkpoint artifact format to upload. 'safetensors' keeps required JSON metadata "
            "sidecars; 'all' preserves the current behavior."
        ),
    )
    parser.add_argument(
        "--hf-upload-private",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Privacy setting used when creating the destination repo. If omitted, Hugging Face "
            "keeps its default for new repos and leaves existing repos unchanged."
        ),
    )
    parser.add_argument(
        "--hf-upload-create-repo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the destination Hugging Face repo if it does not exist.",
    )
    parser.add_argument(
        "--hf-upload-fail-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Abort training when a Hugging Face checkpoint upload fails.",
    )
    parser.add_argument(
        "--hf-upload-allow-pattern",
        action="append",
        default=None,
        help=(
            "Optional Hugging Face upload allow pattern. Repeat to include multiple patterns. "
            "When omitted, all non-ignored files under OUTPUT_DIR are considered."
        ),
    )
    parser.add_argument(
        "--hf-upload-ignore-pattern",
        action="append",
        default=None,
        help=(
            "Additional Hugging Face upload ignore pattern. Repeat to ignore multiple patterns. "
            "Local caches and Trackio artifacts are ignored automatically."
        ),
    )
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
    parser.add_argument(
        "--zipformer",
        action="store_true",
        help="Train with Zipformer blocks from zipformer_pytorch instead of Squeezeformer.",
    )
    parser.add_argument(
        "--zipformer-transducer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the Zipformer transducer objective and beam-search decoder instead of the "
            "Zipformer CTC head."
        ),
    )
    parser.add_argument(
        "--w2v-bert",
        action="store_true",
        help="Fine-tune facebook/w2v-bert-2.0 with the current CTC training pipeline.",
    )
    parser.add_argument(
        "--w2v-bert-model-name",
        default="facebook/w2v-bert-2.0",
        help="Hugging Face model ID to use for --w2v-bert.",
    )
    parser.add_argument(
        "--w2v-bert-model-path",
        default=None,
        help=(
            "Local directory containing a W2V-BERT model/feature extractor to load with "
            "transformers.from_pretrained(). Overrides --w2v-bert-model-name when set."
        ),
    )
    parser.add_argument("--variant", default="sm", choices=["xs", "s", "sm", "m", "ml", "l"])
    parser.add_argument("--zipformer-transducer-context-size", type=int, default=2)
    parser.add_argument("--zipformer-transducer-decoder-dim", type=int, default=512)
    parser.add_argument("--zipformer-transducer-joiner-dim", type=int, default=512)
    parser.add_argument("--zipformer-transducer-prune-range", type=int, default=5)
    parser.add_argument("--zipformer-transducer-joiner-chunk-size", type=int, default=32)
    parser.add_argument("--output-dir", default="artifacts/squeezeformer-cv22")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
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
    parser.add_argument("--min-chars-per-second", type=float, default=0.0)
    parser.add_argument("--max-chars-per-second", type=float, default=float("inf"))
    parser.add_argument(
        "--min-words-per-second",
        "--min-tokens-per-second",
        dest="min_words_per_second",
        type=float,
        default=0.0,
        help=(
            "Minimum whitespace-delimited word rate accepted by alignment filtering. "
            "The legacy --min-tokens-per-second spelling remains accepted as an alias."
        ),
    )
    parser.add_argument(
        "--max-words-per-second",
        "--max-tokens-per-second",
        dest="max_words_per_second",
        type=float,
        default=float("inf"),
        help=(
            "Maximum whitespace-delimited word rate accepted by alignment filtering. "
            "The legacy --max-tokens-per-second spelling remains accepted as an alias."
        ),
    )
    parser.add_argument("--min-duration-per-char", type=float, default=0.0)
    parser.add_argument("--max-duration-per-char", type=float, default=float("inf"))
    parser.add_argument(
        "--min-duration-per-word",
        "--min-duration-per-token",
        dest="min_duration_per_word",
        type=float,
        default=0.0,
        help=(
            "Minimum seconds per whitespace-delimited word accepted by alignment filtering. "
            "The legacy --min-duration-per-token spelling remains accepted as an alias."
        ),
    )
    parser.add_argument(
        "--max-duration-per-word",
        "--max-duration-per-token",
        dest="max_duration_per_word",
        type=float,
        default=float("inf"),
        help=(
            "Maximum seconds per whitespace-delimited word accepted by alignment filtering. "
            "The legacy --max-duration-per-token spelling remains accepted as an alias."
        ),
    )
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument(
        "--feature-cache-format",
        type=FeatureCacheFormat,
        choices=list(FeatureCacheFormat),
        default=FeatureCacheFormat.FILE,
        help=(
            "Feature cache backend used when --feature-cache-dir is set. "
            "'file' writes one .pt file per utterance; 'parquet' writes sharded Parquet files."
        ),
    )
    parser.add_argument(
        "--save-audio-preview-samples",
        type=int,
        default=0,
        help=(
            "Decode and save this many training audio examples to "
            "OUTPUT_DIR/audio_previews before training starts. Writes WAV files, "
            "transcripts, and a manifest for listening/debugging."
        ),
    )
    parser.add_argument(
        "--max-batch-duration-sec",
        type=float,
        default=None,
        help=(
            "Cap each length-aware batch by summed audio duration in seconds. "
            "Works with --longest-batches-first."
        ),
    )
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
    parser.add_argument(
        "--dataloader-mp-context",
        choices=["auto", "fork", "forkserver", "spawn"],
        default="auto",
        help=(
            "Multiprocessing start method for DataLoader workers. 'auto' keeps the current "
            "platform-aware default and uses 'spawn' when distributed training is initialized."
        ),
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--dataloader-in-order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Yield training DataLoader batches in sampler order. Disable this to let PyTorch "
            "return whichever worker finishes first, reducing head-of-line stalls during cold "
            "audio decode or feature-cache warmup. Validation remains ordered."
        ),
    )
    parser.add_argument("--metadata-workers", type=int, default=4)
    parser.add_argument(
        "--force-audio-metadata-probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Probe decoded audio metadata before constructing length-aware samplers, even when "
            "manifest durations already supplied sample counts. This makes dynamic batching "
            "match codecs such as Opus that decode at 48 kHz, but adds startup I/O."
        ),
    )
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
        "--liberta-device",
        type=_validate_device_argument,
        default="cpu",
        help="Execution device for the optional LiBERTa teacher, for example 'cpu' or 'cuda:1'.",
    )
    parser.add_argument(
        "--optimizer",
        type=OptimizerChoice,
        choices=list(OptimizerChoice),
        default=OptimizerChoice.ADAMW,
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
    parser.add_argument(
        "--run-trackio-ui",
        action="store_true",
        help=(
            "Launch a detached Trackio UI process with TRACKIO_DIR pointed at the current "
            "training run's output directory."
        ),
    )
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
    parser.add_argument("--spm-vocab-size", type=int, default=128)
    parser.add_argument(
        "--spm-model-type",
        default="unigram",
        choices=["unigram", "bpe", "char", "word"],
    )
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--hold-epochs", type=int, default=160)
    parser.add_argument("--decay-exponent", type=float, default=1.0)
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
    parser.add_argument(
        "--audio-teacher",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--audio-teacher-model-name", default="facebook/wav2vec2-bert-2.0")
    parser.add_argument(
        "--audio-teacher-model-path",
        default=None,
        help=(
            "Local directory containing a wav2vec2-bert teacher to load with "
            "transformers.from_pretrained(). Overrides --audio-teacher-model-name when set."
        ),
    )
    parser.add_argument(
        "--audio-teacher-device",
        default="cpu",
        help="Execution device for the optional audio teacher, for example 'cpu' or 'cuda:1'.",
    )
    parser.add_argument("--audio-teacher-weight", type=float, default=0.05)
    parser.add_argument(
        "--audio-teacher-objective",
        default="hidden_mse",
        choices=["hidden_mse", "hidden_cosine", "ctc_kl"],
    )
    parser.add_argument(
        "--audio-teacher-target",
        default="encoder",
        choices=["encoder", "intermediate", "ctc_logits"],
    )
    parser.add_argument("--audio-teacher-layer", type=int, default=-1)
    parser.add_argument("--audio-teacher-sample-rate", type=int, default=16_000)
    parser.add_argument("--audio-teacher-max-seconds", type=float, default=30.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--ema-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--validation-model-source",
        type=ValidationModelSource,
        choices=list(ValidationModelSource),
        default=ValidationModelSource.RAW,
        help="Which model weights to use for validation and validation-selected checkpoints.",
    )
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
    parser.add_argument("--beam-length-bonus", type=float, default=0.1)
    parser.add_argument(
        "--fit-shallow-fusion-lm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--shallow-fusion-lm-order", type=int, default=3)
    parser.add_argument("--shallow-fusion-lm-alpha", type=float, default=0.1)
    parser.add_argument("--example-limit", type=int, default=5)
    args = parser.parse_args(argv)
    _validate_startup_args(
        args,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        explicit_batch_size="--batch-size" in argv,
    )
    return args


def _resolve_float_tuple(values: str) -> tuple[float, ...]:
    parsed = tuple(float(value.strip()) for value in values.split(",") if value.strip())
    if not parsed:
        raise ValueError("expected at least one float value")
    return parsed


def _resolve_scheduler_kwargs(args: argparse.Namespace, optimizer_name: str) -> dict[str, float]:
    if optimizer_name == "scaledadam":
        return {
            "schedule_type": "eden",
            "eden_step_scale": 5000.0,
            "eden_epoch_scale": 6.0,
            "eden_warmup_steps": 500,
            "eden_start_scale": 0.5,
        }
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
