from __future__ import annotations

import pytest

from squeezeformer_pytorch.training.cli import parse_args


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
