from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def _register_legacy_main_aliases() -> None:
    """Expose train.py enums on the current __main__ module for legacy checkpoints."""
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return

    from train import AdaptiveBatchUnit, DTypeChoice, DecodeStrategy, OptimizerChoice

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
    checkpoint["model_state_dict"] = load_file(str(checkpoint_path), device=str(map_location))
    return checkpoint


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

    metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
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
