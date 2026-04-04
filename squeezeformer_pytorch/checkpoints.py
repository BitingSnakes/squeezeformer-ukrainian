from __future__ import annotations

import json
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from .runtime_types import AdaptiveBatchUnit, DecodeStrategy, DTypeChoice, OptimizerChoice
from .secrets import sanitize_for_serialization


def _register_legacy_main_aliases() -> None:
    """Expose historical enum symbols on __main__ for older torch checkpoints."""
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return

    for cls in (AdaptiveBatchUnit, DTypeChoice, DecodeStrategy, OptimizerChoice):
        if not hasattr(main_module, cls.__name__):
            setattr(main_module, cls.__name__, cls)


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    _register_legacy_main_aliases()
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix != ".safetensors":
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    metadata_path = checkpoint_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata sidecar for safetensors checkpoint: {metadata_path}"
        )

    checkpoint = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "encoder_config" in checkpoint:
        from .model import SqueezeformerConfig

        checkpoint["encoder_config"] = asdict(SqueezeformerConfig(**checkpoint["encoder_config"]))
    checkpoint["model_state_dict"] = load_file(str(checkpoint_path), device=str(map_location))
    return checkpoint


def is_torchao_quantized_checkpoint(checkpoint: Mapping[str, Any]) -> bool:
    quantization = checkpoint.get("quantization")
    return isinstance(quantization, Mapping) and quantization.get("backend") == "torchao"


def should_use_transformer_engine_for_checkpoint(
    checkpoint: Mapping[str, Any],
    requested_dtype: DTypeChoice | None = None,
) -> bool:
    if is_torchao_quantized_checkpoint(checkpoint):
        return False
    training_args = checkpoint.get("training_args")
    checkpoint_dtype = ""
    if isinstance(training_args, Mapping):
        checkpoint_dtype = str(training_args.get("dtype", ""))
    return checkpoint_dtype == DTypeChoice.FP8.value or requested_dtype == DTypeChoice.FP8


def save_checkpoint(checkpoint: dict[str, Any], checkpoint_path: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix != ".safetensors":
        torch.save(checkpoint, checkpoint_path)
        return

    if "model_state_dict" not in checkpoint:
        raise KeyError("Safetensors checkpoints require a 'model_state_dict' entry.")

    state_dict = {
        key: value.detach().cpu().contiguous()
        for key, value in checkpoint["model_state_dict"].items()
    }
    save_file(state_dict, str(checkpoint_path), metadata={"format": "squeezeformer-pytorch"})

    metadata = sanitize_for_serialization(
        {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    )
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
