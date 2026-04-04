from __future__ import annotations

from pathlib import Path

import quantize


def test_resolve_output_path_defaults_to_safetensors_for_pt_source() -> None:
    resolved = quantize.resolve_output_path("/tmp/checkpoint_best.pt", None)

    assert resolved == Path("/tmp/checkpoint_best.torchao-int8.safetensors")


def test_resolve_output_path_defaults_to_safetensors_for_safetensors_source() -> None:
    resolved = quantize.resolve_output_path("/tmp/checkpoint_best.safetensors", None)

    assert resolved == Path("/tmp/checkpoint_best.torchao-int8.safetensors")


def test_resolve_output_path_preserves_explicit_output() -> None:
    resolved = quantize.resolve_output_path(
        "/tmp/checkpoint_best.pt", "/tmp/custom-quantized.safetensors"
    )

    assert resolved == Path("/tmp/custom-quantized.safetensors")
