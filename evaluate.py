from __future__ import annotations

import argparse
import json
import os
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
    AudioFeaturizer,
    CV22ASRDataset,
    create_dataloader,
    download_cv22_dataset,
    load_cv22_records,
    prevalidate_records,
)
from squeezeformer_pytorch.model import SqueezeformerConfig
from squeezeformer_pytorch.runtime_types import DecodeStrategy, DTypeChoice
from train import (
    FrozenLibertaTeacher,
    _validate_device_argument,
    _validate_device_ready,
    evaluate,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Squeezeformer CTC model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-repo", default="speech-uk/cv22")
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
    parser.add_argument("--example-limit", type=int, default=5)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--trackio-project", default="squeezeformer-cv22")
    parser.add_argument("--trackio-space-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    checkpoint_tokenizer_type = str(checkpoint["tokenizer"].get("type", ""))
    encoder_config = SqueezeformerConfig(**checkpoint["encoder_config"])
    training_args = checkpoint.get("training_args", {})
    intermediate_ctc_weight = float(training_args.get("intermediate_ctc_weight", 0.0))
    intermediate_ctc_layers = training_args.get("intermediate_ctc_layers")
    intermediate_ctc_layer = training_args.get("intermediate_ctc_layer")
    blank_prune_threshold = float(training_args.get("blank_prune_threshold", 0.0))
    blank_prune_layer = training_args.get("blank_prune_layer")
    blank_prune_min_keep_frames = int(training_args.get("blank_prune_min_keep_frames", 1))
    aed_decoder_enabled = bool(training_args.get("aed_decoder", False))
    aed_decoder_layers = int(training_args.get("aed_decoder_layers", 1))
    aed_decoder_heads = int(training_args.get("aed_decoder_heads", 4))
    aed_decoder_dropout = float(training_args.get("aed_decoder_dropout", 0.1))
    aed_loss_weight = float(training_args.get("aed_loss_weight", 0.3))
    liberta_distill_enabled = bool(training_args.get("liberta_distill", False))
    liberta_model_name = str(training_args.get("liberta_model_name", "Yehor/Liberta"))
    liberta_distill_weight = float(training_args.get("liberta_distill_weight", 0.05))
    liberta_max_length = int(training_args.get("liberta_max_length", 256))
    if intermediate_ctc_weight > 0.0:
        if intermediate_ctc_layers is not None:
            resolved_intermediate_ctc_layers = tuple(
                int(layer) for layer in intermediate_ctc_layers
            )
        elif intermediate_ctc_layer is not None:
            resolved_intermediate_ctc_layers = (int(intermediate_ctc_layer),)
        else:
            resolved_intermediate_ctc_layers = ()
    else:
        resolved_intermediate_ctc_layers = ()
    model = SqueezeformerCTC(
        encoder_config=encoder_config,
        vocab_size=tokenizer.vocab_size,
        intermediate_ctc_layers=resolved_intermediate_ctc_layers,
        blank_prune_layer=(
            int(blank_prune_layer)
            if blank_prune_threshold > 0.0 and blank_prune_layer is not None
            else None
        ),
        blank_prune_threshold=blank_prune_threshold,
        blank_prune_min_keep_frames=blank_prune_min_keep_frames,
        aed_decoder_enabled=aed_decoder_enabled,
        aed_decoder_layers=aed_decoder_layers,
        aed_decoder_heads=aed_decoder_heads,
        aed_decoder_dropout=aed_decoder_dropout,
        liberta_distill_enabled=liberta_distill_enabled,
        use_transformer_engine=should_use_transformer_engine_for_checkpoint(
            checkpoint,
            requested_dtype=args.dtype,
        ),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_device(args.device)
    _validate_device_ready(device)
    model.to(device)
    model.eval()
    lowercase_transcripts = (
        args.lowercase_transcripts
        if args.lowercase_transcripts is not None
        else checkpoint_tokenizer_type != "sentencepiece"
    )

    dataset_root = download_cv22_dataset(
        repo_id=args.dataset_repo,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    records = load_cv22_records(
        dataset_root=dataset_root,
        split=args.split,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        max_samples=args.max_samples,
        lowercase_transcripts=lowercase_transcripts,
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
    dataset = CV22ASRDataset(
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
    )
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    lm_scorer = load_lm_scorer(args.lm_scorer)
    liberta_teacher = (
        FrozenLibertaTeacher(
            liberta_model_name,
            device=device,
            max_length=liberta_max_length,
        )
        if liberta_distill_enabled
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
        example_limit=args.example_limit,
        intermediate_ctc_weight=intermediate_ctc_weight,
        aed_loss_weight=aed_loss_weight,
        liberta_teacher=liberta_teacher,
        liberta_distill_weight=liberta_distill_weight,
    )
    metrics = result["metrics"] | {
        "split": args.split,
        "samples": len(records),
        "decode_strategy": args.decode_strategy,
    }
    payload = {
        "metrics": metrics,
        "hardest_examples": result["hardest_examples"],
        "random_examples": result["random_examples"],
        "speaker_metrics": result["speaker_metrics"],
    }
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
        ),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    trackio_config = {
        "evaluation_split": args.split,
        "checkpoint": str(args.checkpoint),
        "decode_strategy": args.decode_strategy,
    }
    trackio.init(
        project=args.trackio_project,
        space_id=args.trackio_space_id,
        config=trackio_config,
    )
    trackio.log(metrics)
    trackio.finish()


if __name__ == "__main__":
    main()
