from __future__ import annotations

import argparse

import torch
import torchaudio

from squeezeformer_pytorch.asr import SqueezeformerCTC, tokenizer_from_dict
from squeezeformer_pytorch.checkpoints import load_checkpoint
from squeezeformer_pytorch.data import AudioFeaturizer
from squeezeformer_pytorch.model import SqueezeformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASR inference for one audio file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a training checkpoint.")
    parser.add_argument("--audio", required=True, help="Path to an audio file to transcribe.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device, for example 'cpu' or 'cuda:0'.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    encoder_config = SqueezeformerConfig(**checkpoint["encoder_config"])
    training_args = checkpoint.get("training_args", {})
    checkpoint_dtype = str(training_args.get("dtype", ""))

    model = SqueezeformerCTC(
        encoder_config=encoder_config,
        vocab_size=tokenizer.vocab_size,
        use_transformer_engine=checkpoint_dtype == "fp8",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    featurizer = AudioFeaturizer(**checkpoint.get("featurizer_config", {}))
    waveform, sample_rate = torchaudio.load(args.audio)
    features = featurizer(waveform, sample_rate).unsqueeze(0).to(device)
    feature_lengths = torch.tensor([features.size(1)], device=device)

    with torch.inference_mode():
        log_probs, _ = model.log_probs(features, feature_lengths)

    token_ids = log_probs.argmax(dim=-1)[0].cpu().tolist()
    print(tokenizer.decode_ctc(token_ids))


if __name__ == "__main__":
    main()
