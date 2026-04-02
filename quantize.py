from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from squeezeformer_pytorch.asr import SqueezeformerCTC, tokenizer_from_dict
from squeezeformer_pytorch.checkpoints import load_checkpoint
from squeezeformer_pytorch.model import SqueezeformerConfig

try:
    import torchao
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
except ImportError:
    torchao = None
    Int8WeightOnlyConfig = None
    quantize_ = None

DEFAULT_CHECKPOINT = (
    "https://huggingface.co/speech-uk/squeezeformer-sm/resolve/main/checkpoint_best.pt"
)


def resolve_checkpoint_path(checkpoint: str) -> str:
    if checkpoint == DEFAULT_CHECKPOINT:
        return hf_hub_download(repo_id="speech-uk/squeezeformer-sm", filename="checkpoint_best.pt")
    if checkpoint.startswith("https://huggingface.co/"):
        parts = checkpoint.removeprefix("https://huggingface.co/").split("/")
        if len(parts) >= 5 and parts[2] == "resolve":
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[4:])
            return hf_hub_download(repo_id=repo_id, filename=filename)
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a Squeezeformer checkpoint with TorchAO."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path or Hugging Face URL.",
    )
    parser.add_argument(
        "--output",
        help="Destination .pt path for the quantized checkpoint. Defaults next to the source.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run quantization on, for example 'cpu' or 'cuda:0'.",
    )
    return parser.parse_args()


def resolve_output_path(checkpoint_path: str, output: str | None) -> Path:
    if output:
        output_path = Path(output)
    else:
        source = Path(checkpoint_path)
        suffix = "".join(source.suffixes)
        if suffix:
            output_path = source.with_name(f"{source.name.removesuffix(suffix)}.torchao-int8.pt")
        else:
            output_path = source.with_name(f"{source.name}.torchao-int8.pt")
    if output_path.suffix == ".safetensors":
        raise ValueError("TorchAO quantized checkpoints must be saved as .pt files.")
    return output_path


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def build_model(checkpoint_data: dict[str, Any]) -> SqueezeformerCTC:
    tokenizer = tokenizer_from_dict(checkpoint_data["tokenizer"])
    encoder_config = SqueezeformerConfig(**checkpoint_data["encoder_config"])
    training_args = checkpoint_data.get("training_args", {})
    intermediate_ctc_weight = float(training_args.get("intermediate_ctc_weight", 0.0))
    intermediate_ctc_layers = training_args.get("intermediate_ctc_layers")
    intermediate_ctc_layer = training_args.get("intermediate_ctc_layer")
    blank_prune_threshold = float(training_args.get("blank_prune_threshold", 0.0))
    blank_prune_layer = training_args.get("blank_prune_layer")
    blank_prune_min_keep_frames = int(training_args.get("blank_prune_min_keep_frames", 1))

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
        use_transformer_engine=False,
    )
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    return model


def build_quantized_checkpoint_payload(
    checkpoint_data: dict[str, Any], model: SqueezeformerCTC
) -> dict[str, Any]:
    payload = dict(checkpoint_data)
    payload["model_state_dict"] = model.state_dict()
    payload["quantization"] = {
        "backend": "torchao",
        "config": "Int8WeightOnlyConfig",
        "torchao_version": getattr(torchao, "__version__", "unknown"),
    }
    return payload


def main() -> None:
    args = parse_args()
    if quantize_ is None or Int8WeightOnlyConfig is None or torchao is None:
        raise RuntimeError("torchao is required. Install it with `pip install torchao`.")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    output_path = resolve_output_path(checkpoint_path, args.output)
    device = resolve_device(args.device)

    checkpoint_data = load_checkpoint(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint_data.get("quantization"), dict):
        raise ValueError("Checkpoint already contains quantization metadata.")

    model = build_model(checkpoint_data)
    quantize_(model, Int8WeightOnlyConfig(), device=device)

    payload = build_quantized_checkpoint_payload(checkpoint_data, model)
    torch.save(payload, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
