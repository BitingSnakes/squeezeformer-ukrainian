from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from squeezeformer_pytorch.checkpoints import load_checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a checkpoint and report whether it is a full training checkpoint or an exported inference checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt or .safetensors checkpoint.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of plain text.",
    )
    return parser.parse_args()


def _checkpoint_kind(path: Path) -> str:
    if path.suffix == ".safetensors":
        return "inference"
    return "training"


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    training_args = checkpoint.get("training_args", {})
    state_dict = checkpoint.get("model_state_dict", {})
    state_keys = set(state_dict.keys())

    has_aed_weights = any(key.startswith("aed_decoder.") for key in state_keys)
    has_liberta_projection = any(key.startswith("liberta_projection.") for key in state_keys)
    has_intermediate_ctc_weights = any(
        key.startswith("intermediate_classifiers.") for key in state_keys
    )

    return {
        "path": str(checkpoint_path),
        "kind": _checkpoint_kind(checkpoint_path),
        "file_suffix": checkpoint_path.suffix,
        "has_metadata_sidecar": checkpoint_path.with_suffix(".json").exists()
        if checkpoint_path.suffix == ".safetensors"
        else False,
        "aed_decoder_flag": training_args.get("aed_decoder"),
        "liberta_distill_flag": training_args.get("liberta_distill"),
        "blank_prune_flag": training_args.get("blank_prune"),
        "intermediate_ctc_flag": training_args.get("intermediate_ctc"),
        "has_aed_decoder_weights": has_aed_weights,
        "has_liberta_projection_weights": has_liberta_projection,
        "has_intermediate_ctc_weights": has_intermediate_ctc_weights,
        "inference_ready": _checkpoint_kind(checkpoint_path) == "inference"
        and not has_aed_weights
        and not has_liberta_projection,
    }


def _render_text(report: dict[str, Any]) -> str:
    lines = [
        f"path: {report['path']}",
        f"kind: {report['kind']}",
        f"file_suffix: {report['file_suffix']}",
        f"inference_ready: {report['inference_ready']}",
        f"has_metadata_sidecar: {report['has_metadata_sidecar']}",
        f"aed_decoder_flag: {report['aed_decoder_flag']}",
        f"liberta_distill_flag: {report['liberta_distill_flag']}",
        f"blank_prune_flag: {report['blank_prune_flag']}",
        f"intermediate_ctc_flag: {report['intermediate_ctc_flag']}",
        f"has_aed_decoder_weights: {report['has_aed_decoder_weights']}",
        f"has_liberta_projection_weights: {report['has_liberta_projection_weights']}",
        f"has_intermediate_ctc_weights: {report['has_intermediate_ctc_weights']}",
    ]
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    report = inspect_checkpoint(args.checkpoint)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    print(_render_text(report))


if __name__ == "__main__":
    main()
