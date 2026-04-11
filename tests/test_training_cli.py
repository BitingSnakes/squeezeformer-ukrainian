from __future__ import annotations

import pytest

from squeezeformer_pytorch.runtime_types import OptimizerChoice, ValidationModelSource
from squeezeformer_pytorch.training.cli import parse_args
from squeezeformer_pytorch.training.runtime import (
    _variant_defaults,
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


def test_parse_args_accepts_beam_length_bonus() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--beam-length-bonus",
            "0.25",
        ]
    )

    assert args.beam_length_bonus == 0.25


def test_parse_args_defaults_match_paper_recipe() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
        ]
    )

    assert args.epochs == 500
    assert args.optimizer == OptimizerChoice.ADAMW
    assert args.weight_decay == 5e-4
    assert args.spm_vocab_size == 128
    assert args.warmup_epochs == 20
    assert args.hold_epochs == 160
    assert args.ema_decay == 0.0
    assert args.validation_model_source == ValidationModelSource.RAW


def test_variant_scheduler_defaults_match_paper_recipe() -> None:
    assert _variant_defaults("sm").peak_lr == 2e-3
    assert _variant_defaults("m").peak_lr == 1.5e-3
    assert _variant_defaults("ml").peak_lr == 1e-3
    assert _variant_defaults("l").peak_lr == 1e-3


def test_parse_args_accepts_run_trackio_ui_flag() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--run-trackio-ui",
        ]
    )

    assert args.run_trackio_ui is True


def test_parse_args_accepts_zipformer_flag() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--zipformer",
        ]
    )

    assert args.zipformer is True


def test_parse_args_accepts_force_audio_metadata_probe_flag() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--force-audio-metadata-probe",
        ]
    )

    assert args.force_audio_metadata_probe is True


def test_parse_args_accepts_audio_preview_sample_count() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--save-audio-preview-samples",
            "3",
        ]
    )

    assert args.save_audio_preview_samples == 3


def test_parse_args_keeps_hf_checkpoint_upload_disabled_by_default() -> None:
    args = parse_args(["--device", "cpu"])

    assert args.hf_upload_checkpoints is False
    assert args.hf_upload_repo_id is None
    assert args.hf_upload_checkpoint_format == "all"


def test_parse_args_rejects_hf_checkpoint_upload_without_repo() -> None:
    with pytest.raises(ValueError) as error:
        parse_args(["--device", "cpu", "--hf-upload-checkpoints"])

    assert str(error.value) == "--hf-upload-repo-id is required when --hf-upload-checkpoints is set."


def test_parse_args_accepts_hf_checkpoint_upload_options() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--hf-upload-checkpoints",
            "--hf-upload-repo-id",
            "speech-uk/checkpoints",
            "--hf-upload-repo-type",
            "model",
            "--hf-upload-path-in-repo",
            "runs/demo",
            "--hf-upload-revision",
            "main",
            "--hf-upload-checkpoint-format",
            "safetensors",
            "--hf-upload-ignore-pattern",
            "*.tmp",
            "--hf-upload-allow-pattern",
            "checkpoint*",
        ]
    )

    assert args.hf_upload_checkpoints is True
    assert args.hf_upload_repo_id == "speech-uk/checkpoints"
    assert args.hf_upload_repo_type == "model"
    assert args.hf_upload_path_in_repo == "runs/demo"
    assert args.hf_upload_revision == "main"
    assert args.hf_upload_checkpoint_format == "safetensors"
    assert args.hf_upload_ignore_pattern == ["*.tmp"]
    assert args.hf_upload_allow_pattern == ["checkpoint*"]
