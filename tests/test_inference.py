from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

import inference
from squeezeformer_pytorch import squeezeformer_variant
from squeezeformer_pytorch.checkpoints import (
    is_torchao_quantized_checkpoint,
    should_use_transformer_engine_for_checkpoint,
)
from squeezeformer_pytorch.runtime_types import DTypeChoice


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
