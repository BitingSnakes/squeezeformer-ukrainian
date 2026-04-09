from __future__ import annotations

from argparse import Namespace

import pytest

from squeezeformer_pytorch.model import SqueezeformerConfig, VARIANT_CONFIGS
from squeezeformer_pytorch.training.cli import parse_args
from squeezeformer_pytorch.training.runtime import (
    _default_intermediate_ctc_layers,
    _resolve_intermediate_ctc_settings,
)


def test_parse_args_rejects_explicit_batch_size_with_duration_batching() -> None:
    with pytest.raises(ValueError) as error:
        parse_args(
            [
                "--device",
                "cpu",
                "--batch-size",
                "30",
                "--max-batch-duration-sec",
                "1200",
            ]
        )

    assert (
        str(error.value)
        == "--batch-size cannot be combined with --max-batch-duration-sec because that batching "
        "mode ignores sample-count batching."
    )


def test_parse_args_allows_duration_batching_with_default_batch_size() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--max-batch-duration-sec",
            "1200",
        ]
    )

    assert args.max_batch_duration_sec == 1200
    assert args.batch_size == 8


def test_parse_args_rejects_multiple_dynamic_batching_modes() -> None:
    with pytest.raises(ValueError) as error:
        parse_args(
            [
                "--device",
                "cpu",
                "--max-batch-duration-sec",
                "1200",
                "--max-batch-frames",
                "16000",
            ]
        )

    assert (
        str(error.value) == "Batching controls are mutually exclusive; choose only one of "
        "--max-batch-duration-sec, --max-batch-frames, or the adaptive batch options."
    )


def test_parse_args_accepts_no_intermediate_ctc_layers_flag() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--no-intermediate-ctc-layers",
        ]
    )

    assert args.no_intermediate_ctc_layers is True


def test_parse_args_accepts_disable_flash_attn2_kernels_flag() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--disable-flash-attn2-kernels",
        ]
    )

    assert args.disable_flash_attn2_kernels is True


def test_parse_args_accepts_disable_flash_attention_flag() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--disable-flash-attention",
        ]
    )

    assert args.disable_flash_attention is True


def test_parse_args_accepts_run_trackio_ui_flag() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--run-trackio-ui",
        ]
    )

    assert args.run_trackio_ui is True


def test_no_intermediate_ctc_layers_overrides_checkpoint_settings() -> None:
    args = Namespace(
        intermediate_ctc=None,
        intermediate_ctc_weight=0.3,
        intermediate_ctc_layers=None,
        intermediate_ctc_layer=None,
        no_intermediate_ctc_layers=True,
    )
    encoder_config = SqueezeformerConfig(num_layers=12)
    checkpoint = {
        "training_args": {
            "intermediate_ctc_enabled": True,
            "intermediate_ctc_weight": 0.3,
            "intermediate_ctc_layers": [3, 7],
        }
    }

    layers, weight = _resolve_intermediate_ctc_settings(args, encoder_config, checkpoint)

    assert layers == ()
    assert weight == 0.0


def test_default_intermediate_ctc_layers_avoid_reduced_time_segments_for_all_variants() -> None:
    for config in VARIANT_CONFIGS.values():
        reduced_layers: set[int] = set()
        for reduce_idx, recover_idx in zip(
            sorted(config.time_reduce_idx),
            sorted(config.time_recover_idx),
            strict=False,
        ):
            reduced_layers.update(range(reduce_idx, recover_idx))

        layers = _default_intermediate_ctc_layers(config)

        assert all(layer not in reduced_layers for layer in layers)
        assert all(layer < config.num_layers - 1 for layer in layers)
