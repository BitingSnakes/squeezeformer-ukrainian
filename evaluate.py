from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import trackio
from torch import nn

from squeezeformer_pytorch.asr import (
    SqueezeformerCTC,
    load_lm_scorer,
    tokenizer_from_dict,
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
from train import DecodeStrategy, DTypeChoice, _validate_device_argument, evaluate


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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    encoder_config = SqueezeformerConfig(**checkpoint["encoder_config"])
    model = SqueezeformerCTC(encoder_config=encoder_config, vocab_size=tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested with --device, but torch.cuda.is_available() is false."
        )
    model.to(device)
    model.eval()

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
