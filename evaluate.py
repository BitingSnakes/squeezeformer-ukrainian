from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import trackio
from torch import nn

from squeezeformer_pytorch.asr import (
    SqueezeformerCTC,
    load_lm_scorer,
    tokenizer_from_dict,
)
from squeezeformer_pytorch.checkpoints import (
    load_checkpoint,
    should_use_transformer_engine_for_checkpoint,
)
from squeezeformer_pytorch.data import (
    ASRDataset,
    AudioFeaturizer,
    create_dataloader,
    prevalidate_records,
)
from squeezeformer_pytorch.evaluation_runtime import (
    build_evaluation_payload,
    resolve_evaluation_checkpoint_settings,
    resolve_lowercase_transcripts,
)
from squeezeformer_pytorch.model import SqueezeformerConfig
from squeezeformer_pytorch.runtime_types import DecodeStrategy, DTypeChoice, ValidationModelSource
from squeezeformer_pytorch.training.cli import (
    _validate_device_argument,
    _validate_existing_local_path_argument,
)
from squeezeformer_pytorch.training.data_loading import (
    _load_records_from_dataset_roots,
    _resolve_sources,
)
from squeezeformer_pytorch.training.evaluation import evaluate
from squeezeformer_pytorch.training.runtime import (
    FrozenLibertaTeacher,
    _validate_device_ready,
    resolve_device,
)
from train import _build_trackio_run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Squeezeformer CTC model.")
    parser.add_argument("--checkpoint", required=True)
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
            "provided together with --split validation, evaluation uses the full set of records "
            "from these sources."
        ),
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument(
        "--lowercase-transcripts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Normalize evaluation references to lowercase before loading the dataset. "
            "Defaults to the checkpoint tokenizer convention."
        ),
    )
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
        "--dtype",
        type=DTypeChoice,
        choices=list(DTypeChoice),
        default=DTypeChoice.BFLOAT16,
    )
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
    parser.add_argument("--example-limit", type=int, default=5)
    parser.add_argument(
        "--validation-model-source",
        type=ValidationModelSource,
        choices=list(ValidationModelSource),
        default=None,
        help=(
            "Which checkpoint weights to evaluate when both exported validation weights and "
            "resume-time raw weights are available. Defaults to the checkpoint's recorded "
            "validation source, or raw when that metadata is missing."
        ),
    )
    parser.add_argument("--report-path", default=None)
    parser.add_argument(
        "--trackio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log evaluation metrics to Trackio. Disabled by default for standalone evaluation.",
    )
    parser.add_argument("--trackio-project", default="squeezeformer-cv22")
    parser.add_argument("--trackio-space-id", default=None)
    args = parser.parse_args()
    for source in args.dataset_source or []:
        _validate_existing_local_path_argument("--dataset-source", source)
    for source in args.validation_dataset_source or []:
        _validate_existing_local_path_argument("--validation-dataset-source", source)
    if args.beam_length_bonus < 0:
        raise ValueError(f"--beam-length-bonus must be >= 0, got {args.beam_length_bonus}.")
    return args


def main() -> None:
    args = parse_args()
    beam_length_bonus = getattr(args, "beam_length_bonus", 0.1)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    checkpoint_tokenizer_type = str(checkpoint["tokenizer"].get("type", ""))
    encoder_config = SqueezeformerConfig(**checkpoint["encoder_config"])
    checkpoint_settings = resolve_evaluation_checkpoint_settings(checkpoint)
    model = SqueezeformerCTC(
        encoder_config=encoder_config,
        vocab_size=tokenizer.vocab_size,
        intermediate_ctc_layers=checkpoint_settings["resolved_intermediate_ctc_layers"],
        blank_prune_layer=checkpoint_settings["blank_prune_layer"],
        blank_prune_threshold=checkpoint_settings["blank_prune_threshold"],
        blank_prune_min_keep_frames=checkpoint_settings["blank_prune_min_keep_frames"],
        aed_decoder_enabled=checkpoint_settings["aed_decoder_enabled"],
        aed_decoder_layers=checkpoint_settings["aed_decoder_layers"],
        aed_decoder_heads=checkpoint_settings["aed_decoder_heads"],
        aed_decoder_dropout=checkpoint_settings["aed_decoder_dropout"],
        liberta_distill_enabled=checkpoint_settings["liberta_distill_enabled"],
        use_transformer_engine=should_use_transformer_engine_for_checkpoint(
            checkpoint,
            requested_dtype=args.dtype,
        ),
    )
    selected_validation_model_source = (
        args.validation_model_source
        if args.validation_model_source is not None
        else ValidationModelSource(
            checkpoint.get("validation_model_source", ValidationModelSource.RAW)
        )
    )
    if (
        selected_validation_model_source == ValidationModelSource.RAW
        and checkpoint.get("resume_model_state_dict") is not None
    ):
        state_dict = checkpoint["resume_model_state_dict"]
    else:
        state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    device = resolve_device(args.device)
    _validate_device_ready(device)
    model.to(device)
    model.eval()
    lowercase_transcripts = resolve_lowercase_transcripts(
        args.lowercase_transcripts,
        checkpoint_tokenizer_type=checkpoint_tokenizer_type,
    )

    dataset_sources = _resolve_sources(args.dataset_source, fallback=args.dataset_repo)
    validation_dataset_sources = _resolve_sources(args.validation_dataset_source)
    selected_sources = dataset_sources
    selected_split = args.split
    selected_val_fraction = args.val_fraction
    selected_test_fraction = args.test_fraction
    if validation_dataset_sources and not args.dataset_source:
        selected_sources = validation_dataset_sources
    if args.split == "train" and args.dataset_source:
        selected_split = "train"
        selected_val_fraction = 0.0
        selected_test_fraction = 0.0
    elif args.split == "validation" and validation_dataset_sources:
        selected_sources = validation_dataset_sources
        selected_split = "train"
        selected_val_fraction = 0.0
        selected_test_fraction = 0.0

    records = _load_records_from_dataset_roots(
        selected_sources,
        split=selected_split,
        seed=args.seed,
        val_fraction=selected_val_fraction,
        test_fraction=selected_test_fraction,
        max_samples=args.max_samples,
        min_transcript_chars=1,
        max_transcript_chars=5000,
        max_symbol_ratio=1.0,
        lowercase_transcripts=lowercase_transcripts,
        hf_token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    if args.prevalidate_audio:
        records = prevalidate_records(records, num_workers=args.prevalidate_workers)
        if not records:
            raise RuntimeError(
                "Audio prevalidation removed every sample from the evaluation split."
            )
    featurizer = AudioFeaturizer(**checkpoint.get("featurizer_config", {}))
    feature_cache_dir = (
        Path(args.feature_cache_dir) / args.split if args.feature_cache_dir else None
    )
    dataset = ASRDataset(
        records,
        tokenizer=tokenizer,
        featurizer=featurizer,
        feature_cache_dir=feature_cache_dir,
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        bucket_by_length=args.bucket_by_length,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        multiprocessing_context=getattr(args, "dataloader_mp_context", "auto"),
    )
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    lm_scorer = load_lm_scorer(args.lm_scorer)
    liberta_teacher = (
        FrozenLibertaTeacher(
            checkpoint_settings["liberta_model_name"],
            device=device,
            max_length=checkpoint_settings["liberta_max_length"],
        )
        if checkpoint_settings["liberta_distill_enabled"]
        else None
    )
    result = evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        tokenizer=tokenizer,
        device=device,
        dtype=args.dtype,
        decode_strategy=args.decode_strategy,
        beam_size=args.beam_size,
        lm_scorer=lm_scorer,
        lm_weight=args.lm_weight,
        beam_length_bonus=beam_length_bonus,
        example_limit=args.example_limit,
        intermediate_ctc_weight=checkpoint_settings["intermediate_ctc_weight"],
        aed_loss_weight=checkpoint_settings["aed_loss_weight"],
        liberta_teacher=liberta_teacher,
        liberta_distill_weight=checkpoint_settings["liberta_distill_weight"],
    )
    payload = build_evaluation_payload(
        metrics=result["metrics"],
        result=result,
        split=args.split,
        samples=len(records),
        decode_strategy=args.decode_strategy,
    )
    metrics = payload["metrics"]
    print(
        "losses:",
        " ".join(
            f"{name}={float(metrics[name]):.6f}"
            for name in (
                "loss",
                "main_ctc_loss",
                "intermediate_ctc_loss",
                "combined_ctc_loss",
                "aed_loss",
                "liberta_distill_loss",
            )
            if name in metrics
        ),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.trackio:
        trackio_run_name = _build_trackio_run_name(
            trackio_project=args.trackio_project,
            output_dir=Path(args.checkpoint).expanduser().resolve().parent,
            start_epoch=1,
            global_step=0,
            process_start_timestamp=time.time(),
        )
        trackio_config = {
            "evaluation_split": args.split,
            "checkpoint": str(args.checkpoint),
            "decode_strategy": args.decode_strategy,
        }
        trackio.init(
            project=args.trackio_project,
            name=trackio_run_name,
            space_id=args.trackio_space_id,
            config=trackio_config,
        )
        trackio.log(metrics)
        trackio.finish()


if __name__ == "__main__":
    main()
