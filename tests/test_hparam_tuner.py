from __future__ import annotations

from argparse import Namespace

import hparam_tuner
from hparam_tuner import build_train_command, estimate_training_hparams


def _base_args(**overrides: object) -> Namespace:
    values = dict(
        variant="sm",
        optimizer="muon",
        tokenizer="sentencepiece",
        spm_vocab_size=128,
        device="cpu",
        dtype="bfloat16",
        feature_cache_dir="artifacts/feature_cache",
        compile=True,
        speed_perturb_prob=0.5,
        noise_prob=0.2,
        reverb_prob=0.1,
        decode_strategy="beam",
        beam_size=8,
        output_dir="artifacts/cv22-sm",
        epochs=10,
        base_batch_size=8,
        base_max_batch_frames=12000,
        base_gradient_accumulation_steps=4,
        avg_frames_per_sample=1500,
        emit_format="json",
    )
    values.update(overrides)
    return Namespace(**values)


def test_estimate_training_hparams_cpu_smoke() -> None:
    args = _base_args()

    estimate = estimate_training_hparams(args)

    assert estimate.batch_size >= 1
    assert estimate.max_batch_frames >= 4000
    assert estimate.gradient_accumulation_steps >= 1
    assert estimate.num_workers >= 1
    assert estimate.metadata_workers >= 1
    assert estimate.prefetch_factor == 2
    assert estimate.beam_size <= 8
    assert estimate.estimated_effective_frames >= estimate.max_batch_frames
    assert estimate.variant == "sm"
    assert estimate.resolved_dtype == "bfloat16"
    assert estimate.parameter_scale > 0.0


def test_build_train_command_includes_estimated_knobs() -> None:
    args = _base_args(emit_format="shell")
    estimate = estimate_training_hparams(args)

    command = build_train_command(args, estimate)

    assert "--device cpu" in command
    assert f"--batch-size {estimate.batch_size}" in command
    assert f"--max-batch-frames {estimate.max_batch_frames}" in command
    assert f"--gradient-accumulation-steps {estimate.gradient_accumulation_steps}" in command
    assert f"--dtype {estimate.resolved_dtype}" in command
    assert "--compile" in command


def test_estimate_training_hparams_xla_smoke() -> None:
    args = _base_args(device="xla")

    estimate = estimate_training_hparams(args)

    assert estimate.batch_size >= 1
    assert estimate.max_batch_frames >= 12000
    assert estimate.gradient_accumulation_steps >= 1
    assert estimate.num_workers >= 2
    assert estimate.metadata_workers >= 2
    assert estimate.prefetch_factor == 2
    assert estimate.beam_size <= 6
    assert estimate.estimated_effective_frames >= estimate.max_batch_frames


def test_larger_variant_reduces_frame_budget(monkeypatch) -> None:
    args_sm = _base_args(variant="sm")
    args_l = _base_args(variant="l")

    monkeypatch.setattr(
        hparam_tuner,
        "count_model_parameters",
        lambda variant, vocab_size: {"sm": 10_000_000, "l": 40_000_000}[variant],
    )

    estimate_sm = estimate_training_hparams(args_sm)
    estimate_l = estimate_training_hparams(args_l)

    assert estimate_sm.parameter_scale == 1.0
    assert estimate_l.parameter_scale == 4.0
    assert estimate_l.max_batch_frames < estimate_sm.max_batch_frames


def test_auto_dtype_prefers_fp8_when_supported(monkeypatch) -> None:
    args = _base_args(device="cuda:0", dtype="auto")

    monkeypatch.setattr(
        hparam_tuner,
        "probe_device",
        lambda device: hparam_tuner.DeviceProfile(device, "cuda", 24.0, 16),
    )
    monkeypatch.setattr(hparam_tuner, "_fp8_support_status", lambda device, variant: (True, None))

    estimate = estimate_training_hparams(args)

    assert estimate.resolved_dtype == "fp8"
    assert estimate.fp8_supported is True


def test_auto_dtype_falls_back_when_fp8_is_unavailable(monkeypatch) -> None:
    args = _base_args(device="cpu", dtype="auto")

    monkeypatch.setattr(
        hparam_tuner,
        "_fp8_support_status",
        lambda device, variant: (False, "FP8 requires a CUDA device."),
    )

    estimate = estimate_training_hparams(args)

    assert estimate.resolved_dtype == "bfloat16"
    assert estimate.fp8_supported is False
    assert estimate.fp8_support_reason == "FP8 requires a CUDA device."
