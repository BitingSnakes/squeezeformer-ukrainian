from __future__ import annotations

import argparse
import json
import time

import torch

from squeezeformer_pytorch import SqueezeformerCTC, squeezeformer_variant, tokenizer_from_dict
from train import (
    DecodeStrategy,
    DTypeChoice,
    _autocast_context,
    _validate_device_argument,
    decode_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Squeezeformer forward and decode speed."
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--variant", default="sm", choices=["xs", "s", "sm", "m", "ml", "l"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--time-steps", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=80)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument(
        "--decode-strategy",
        type=DecodeStrategy,
        choices=list(DecodeStrategy),
        default=DecodeStrategy.GREEDY,
    )
    parser.add_argument(
        "--dtype",
        type=DTypeChoice,
        choices=list(DTypeChoice),
        default=DTypeChoice.BFLOAT16,
    )
    parser.add_argument(
        "--device",
        type=_validate_device_argument,
        required=True,
        help="Execution device, for example 'cpu', 'cuda', or 'cuda:0'.",
    )
    return parser.parse_args()


class _DummyTokenizer:
    blank_id = 0

    def decode_ctc(self, token_ids):
        return "".join(chr(97 + (token_id % 26)) for token_id in token_ids if token_id != 0)

    def decode(self, token_ids):
        return "".join(chr(97 + (token_id % 26)) for token_id in token_ids if token_id != 0)


def _load_model(args: argparse.Namespace) -> tuple[SqueezeformerCTC, object]:
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
        config = squeezeformer_variant(args.variant)
        if "encoder_config" in checkpoint:
            from squeezeformer_pytorch.model import SqueezeformerConfig

            config = SqueezeformerConfig(**checkpoint["encoder_config"])
        model = SqueezeformerCTC(encoder_config=config, vocab_size=tokenizer.vocab_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, tokenizer
    config = squeezeformer_variant(args.variant)
    model = SqueezeformerCTC(encoder_config=config, vocab_size=128)
    return model, _DummyTokenizer()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested with --device, but torch.cuda.is_available() is false."
        )
    model, tokenizer = _load_model(args)
    model.to(device)
    model.eval()

    features = torch.randn(args.batch_size, args.time_steps, args.feature_dim, device=device)
    lengths = torch.full((args.batch_size,), args.time_steps, dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(args.warmup_iters):
        with torch.no_grad():
            with _autocast_context(device, args.dtype):
                log_probs, _ = model.log_probs(features, lengths)
                if args.decode_strategy == DecodeStrategy.BEAM:
                    decode_batch(
                        log_probs,
                        tokenizer,
                        strategy=args.decode_strategy,
                        beam_size=args.beam_size,
                    )
                else:
                    decode_batch(log_probs, tokenizer, strategy=args.decode_strategy)
        _synchronize(device)

    start = time.perf_counter()
    decode_time = 0.0
    total_frames = 0
    with torch.no_grad():
        for _ in range(args.iters):
            with _autocast_context(device, args.dtype):
                log_probs, output_lengths = model.log_probs(features, lengths)
            _synchronize(device)
            forward_end = time.perf_counter()
            decode_batch(
                log_probs,
                tokenizer,
                strategy=args.decode_strategy,
                beam_size=args.beam_size,
            )
            _synchronize(device)
            iter_end = time.perf_counter()
            decode_time += iter_end - forward_end
            total_frames += int(output_lengths.sum().item())
    elapsed = time.perf_counter() - start
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    )
    report = {
        "device": str(device),
        "dtype": args.dtype,
        "decode_strategy": args.decode_strategy,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "time_steps": args.time_steps,
        "frames_per_second": total_frames / elapsed,
        "avg_iteration_ms": 1000.0 * elapsed / args.iters,
        "avg_decode_ms": 1000.0 * decode_time / args.iters,
        "peak_memory_mb": peak_memory_mb,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
