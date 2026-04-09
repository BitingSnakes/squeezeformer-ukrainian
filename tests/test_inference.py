from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from huggingface_hub.errors import EntryNotFoundError

import inference
from squeezeformer_pytorch import squeezeformer_variant
from squeezeformer_pytorch.checkpoints import (
    is_torchao_quantized_checkpoint,
    should_use_transformer_engine_for_checkpoint,
)
from squeezeformer_pytorch.inference_runtime import resolve_inference_checkpoint_settings
from squeezeformer_pytorch.runtime_types import DTypeChoice
from squeezeformer_pytorch.training.runtime import _inference_checkpoint_payload


def test_resolve_checkpoint_path_supports_hf_repo_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_download(*, repo_id: str, filename: str) -> str:
        captured["repo_id"] = repo_id
        captured["filename"] = filename
        return "/tmp/downloaded-checkpoint.pt"

    monkeypatch.setattr(inference, "hf_hub_download", fake_download)

    resolved = inference.resolve_checkpoint_path("speech-uk/squeezeformer-bf16-lm-sm-moredata")

    assert resolved == "/tmp/downloaded-checkpoint.pt"
    assert captured == {
        "repo_id": "speech-uk/squeezeformer-bf16-lm-sm-moredata",
        "filename": "checkpoint_best.pt",
    }


def test_resolve_checkpoint_path_falls_back_to_hf_safetensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_download(*, repo_id: str, filename: str) -> str:
        calls.append((repo_id, filename))
        if filename == "checkpoint_best.pt":
            raise EntryNotFoundError("missing")
        return f"/tmp/{filename}"

    monkeypatch.setattr(inference, "hf_hub_download", fake_download)

    resolved = inference.resolve_checkpoint_path("speech-uk/squeezeformer-bf16-lm-sm-moredata")

    assert resolved == "/tmp/checkpoint_best.safetensors"
    assert calls == [
        ("speech-uk/squeezeformer-bf16-lm-sm-moredata", "checkpoint_best.pt"),
        ("speech-uk/squeezeformer-bf16-lm-sm-moredata", "checkpoint_best.safetensors"),
        ("speech-uk/squeezeformer-bf16-lm-sm-moredata", "checkpoint_best.json"),
    ]


def test_resolve_checkpoint_path_supports_hf_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_download(*, repo_id: str, filename: str) -> str:
        captured["repo_id"] = repo_id
        captured["filename"] = filename
        return "/tmp/downloaded-checkpoint.pt"

    monkeypatch.setattr(inference, "hf_hub_download", fake_download)

    resolved = inference.resolve_checkpoint_path(
        "https://huggingface.co/speech-uk/squeezeformer-sm/resolve/main/checkpoint_best.pt"
    )

    assert resolved == "/tmp/downloaded-checkpoint.pt"
    assert captured == {
        "repo_id": "speech-uk/squeezeformer-sm",
        "filename": "checkpoint_best.pt",
    }


def test_resolve_checkpoint_path_downloads_safetensors_sidecar_for_hf_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_download(*, repo_id: str, filename: str) -> str:
        calls.append((repo_id, filename))
        return f"/tmp/{filename}"

    monkeypatch.setattr(inference, "hf_hub_download", fake_download)

    resolved = inference.resolve_checkpoint_path(
        "https://huggingface.co/speech-uk/squeezeformer-sm/resolve/main/checkpoint_best.safetensors"
    )

    assert resolved == "/tmp/checkpoint_best.safetensors"
    assert calls == [
        ("speech-uk/squeezeformer-sm", "checkpoint_best.safetensors"),
        ("speech-uk/squeezeformer-sm", "checkpoint_best.json"),
    ]


def test_asr_session_passes_checkpoint_metadata_to_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        vocab_size = 32

    class DummyFeaturizer:
        def __init__(self, **_: object) -> None:
            pass

    class DummyModel:
        def __init__(self, **kwargs: object) -> None:
            captured["use_transformer_engine"] = kwargs["use_transformer_engine"]

        def load_state_dict(
            self,
            state_dict: dict[str, torch.Tensor],
            strict: bool = True,
            **kwargs,
        ):
            captured["load_kwargs"] = kwargs
            captured["strict"] = strict
            captured["state_dict"] = state_dict

        def to(self, device: torch.device):
            captured["device"] = device
            return self

        def eval(self):
            captured["eval_called"] = True
            return self

    checkpoint_data = {
        "tokenizer": {"type": "character", "symbols": ["а"]},
        "encoder_config": asdict(squeezeformer_variant("xs")),
        "training_args": {},
        "featurizer_config": {},
        "model_state_dict": {"weight": torch.zeros(1)},
    }

    def fake_load_checkpoint(
        checkpoint_path: str,
        *,
        map_location: str | torch.device,
        metadata_path: str | None = None,
    ) -> dict[str, object]:
        captured["checkpoint_path"] = checkpoint_path
        captured["map_location"] = map_location
        captured["metadata_path"] = metadata_path
        return checkpoint_data

    monkeypatch.setattr(inference, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        inference,
        "tokenizer_from_dict",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )
    monkeypatch.setattr(inference, "AudioFeaturizer", DummyFeaturizer)
    monkeypatch.setattr(inference, "SqueezeformerCTC", DummyModel)

    inference.ASRInferenceSession(
        "checkpoint_best.safetensors",
        torch.device("cpu"),
        DTypeChoice.FLOAT32,
        checkpoint_metadata="custom-checkpoint.json",
    )

    assert captured["checkpoint_path"] == "checkpoint_best.safetensors"
    assert captured["map_location"] == "cpu"
    assert captured["metadata_path"] == "custom-checkpoint.json"


def test_detects_torchao_quantized_checkpoint() -> None:
    assert is_torchao_quantized_checkpoint({"quantization": {"backend": "torchao"}})
    assert not is_torchao_quantized_checkpoint({})
    assert not is_torchao_quantized_checkpoint({"quantization": {"backend": "other"}})


def test_should_use_transformer_engine_for_fp8_checkpoint() -> None:
    checkpoint = {"training_args": {"dtype": "fp8"}}

    assert should_use_transformer_engine_for_checkpoint(
        checkpoint,
        requested_dtype=DTypeChoice.FLOAT32,
    )


def test_should_not_use_transformer_engine_for_torchao_checkpoint() -> None:
    checkpoint = {
        "training_args": {"dtype": "fp8"},
        "quantization": {"backend": "torchao", "config": "Int8WeightOnlyConfig"},
    }

    assert not should_use_transformer_engine_for_checkpoint(
        checkpoint,
        requested_dtype=DTypeChoice.FP8,
    )


def test_asr_session_uses_assign_for_torchao_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        vocab_size = 32

    class DummyFeaturizer:
        def __init__(self, **_: object) -> None:
            pass

    class DummyModel:
        def __init__(self, **kwargs: object) -> None:
            captured["use_transformer_engine"] = kwargs["use_transformer_engine"]

        def load_state_dict(
            self,
            state_dict: dict[str, torch.Tensor],
            strict: bool = True,
            **kwargs,
        ):
            captured["load_kwargs"] = kwargs
            captured["strict"] = strict
            captured["state_dict"] = state_dict

        def to(self, device: torch.device):
            captured["device"] = device
            return self

        def eval(self):
            captured["eval_called"] = True
            return self

    checkpoint_data = {
        "tokenizer": {"type": "character", "symbols": ["а"]},
        "encoder_config": asdict(squeezeformer_variant("xs")),
        "training_args": {"dtype": "fp8"},
        "featurizer_config": {},
        "model_state_dict": {"weight": torch.zeros(1)},
        "quantization": {"backend": "torchao", "config": "Int8WeightOnlyConfig"},
    }

    monkeypatch.setattr(inference, "load_checkpoint", lambda *_args, **_kwargs: checkpoint_data)
    monkeypatch.setattr(
        inference,
        "tokenizer_from_dict",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )
    monkeypatch.setattr(inference, "AudioFeaturizer", DummyFeaturizer)
    monkeypatch.setattr(inference, "SqueezeformerCTC", DummyModel)

    inference.ASRInferenceSession("checkpoint.pt", torch.device("cpu"), DTypeChoice.FLOAT32)

    assert captured["use_transformer_engine"] is False
    assert captured["load_kwargs"] == {"assign": True}
    assert captured["strict"] is True
    assert captured["device"] == torch.device("cpu")
    assert captured["eval_called"] is True


def test_inference_runtime_resolves_audio_teacher_metadata() -> None:
    settings = resolve_inference_checkpoint_settings(
        {
            "training_args": {
                "audio_teacher": True,
                "initial_ctc_blank_bias": -0.5,
                "identical_initial_ctc_heads": True,
                "audio_teacher_target": "encoder",
            }
        }
    )

    assert settings["initial_ctc_blank_bias"] == -0.5
    assert settings["identical_initial_ctc_heads"] is True
    assert settings["audio_teacher_enabled"] is True
    assert settings["audio_teacher_target"] == "encoder"


def test_inference_checkpoint_payload_strips_audio_teacher_projection() -> None:
    payload = _inference_checkpoint_payload(
        {
            "model_state_dict": {
                "encoder.weight": torch.zeros(1),
                "audio_teacher_projection.weight": torch.zeros(1, 1),
                "audio_teacher_projection.bias": torch.zeros(1),
            },
            "encoder_config": asdict(squeezeformer_variant("xs")),
            "tokenizer": {"type": "character", "symbols": ["а"]},
            "training_args": {"audio_teacher": True},
        }
    )

    assert "audio_teacher_projection.weight" not in payload["model_state_dict"]
    assert "audio_teacher_projection.bias" not in payload["model_state_dict"]
    assert payload["training_args"]["audio_teacher"] is False


def test_asr_session_uses_transformer_engine_for_fp8_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        vocab_size = 32

    class DummyFeaturizer:
        def __init__(self, **_: object) -> None:
            pass

    class DummyModel:
        def __init__(self, **kwargs: object) -> None:
            captured["use_transformer_engine"] = kwargs["use_transformer_engine"]

        def load_state_dict(
            self,
            state_dict: dict[str, torch.Tensor],
            strict: bool = True,
            **kwargs,
        ):
            captured["load_kwargs"] = kwargs
            captured["strict"] = strict
            captured["state_dict"] = state_dict

        def to(self, device: torch.device):
            captured["device"] = device
            return self

        def eval(self):
            captured["eval_called"] = True
            return self

    checkpoint_data = {
        "tokenizer": {"type": "character", "symbols": ["а"]},
        "encoder_config": asdict(squeezeformer_variant("xs")),
        "training_args": {"dtype": "fp8"},
        "featurizer_config": {},
        "model_state_dict": {"weight": torch.zeros(1)},
    }

    monkeypatch.setattr(inference, "load_checkpoint", lambda *_args, **_kwargs: checkpoint_data)
    monkeypatch.setattr(
        inference,
        "tokenizer_from_dict",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )
    monkeypatch.setattr(inference, "AudioFeaturizer", DummyFeaturizer)
    monkeypatch.setattr(inference, "SqueezeformerCTC", DummyModel)

    inference.ASRInferenceSession("checkpoint.pt", torch.device("cuda"), DTypeChoice.FLOAT32)

    assert captured["use_transformer_engine"] is True
    assert captured["load_kwargs"] == {}
    assert captured["strict"] is True
    assert captured["device"] == torch.device("cuda")
    assert captured["eval_called"] is True


def test_asr_session_rejects_fp8_for_torchao_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_data = {
        "tokenizer": {"type": "character", "symbols": ["а"]},
        "encoder_config": asdict(squeezeformer_variant("xs")),
        "training_args": {},
        "featurizer_config": {},
        "model_state_dict": {},
        "quantization": {"backend": "torchao", "config": "Int8WeightOnlyConfig"},
    }

    monkeypatch.setattr(inference, "load_checkpoint", lambda *_args, **_kwargs: checkpoint_data)
    monkeypatch.setattr(
        inference,
        "tokenizer_from_dict",
        lambda *_args, **_kwargs: type("DummyTokenizer", (), {"vocab_size": 32})(),
    )

    with pytest.raises(ValueError, match="do not support FP8 inference"):
        inference.ASRInferenceSession("checkpoint.pt", torch.device("cpu"), DTypeChoice.FP8)
