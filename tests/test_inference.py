from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

import inference
from squeezeformer_pytorch import squeezeformer_variant
from squeezeformer_pytorch.runtime_types import DTypeChoice


def test_detects_torchao_quantized_checkpoint() -> None:
    assert inference._is_torchao_quantized_checkpoint({"quantization": {"backend": "torchao"}})
    assert not inference._is_torchao_quantized_checkpoint({})
    assert not inference._is_torchao_quantized_checkpoint({"quantization": {"backend": "other"}})


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
