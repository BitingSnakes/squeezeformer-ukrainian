from __future__ import annotations

import torch
from transformers import Wav2Vec2BertConfig as HFWav2Vec2BertConfig

import squeezeformer_pytorch.model as squeezeformer_model
import w2v_bert.asr as w2v_bert_asr
from w2v_bert.asr import W2VBertConfig, W2VBertCTC, W2VBertFeatureExtractor


def _tiny_hf_config() -> HFWav2Vec2BertConfig:
    return HFWav2Vec2BertConfig(
        feature_projection_input_dim=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        conv_depthwise_kernel_size=3,
        num_conv_pos_embedding_groups=1,
        num_conv_pos_embeddings=5,
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
        apply_spec_augment=False,
    )


def _tiny_w2v_bert_config() -> W2VBertConfig:
    return W2VBertConfig(
        model_name="tiny-w2v-bert",
        hidden_size=16,
        feature_dim=8,
        sample_rate=16_000,
        model_config=_tiny_hf_config().to_dict(),
    )


def test_w2v_bert_ctc_returns_training_outputs_and_backward_runs() -> None:
    model = W2VBertCTC(
        encoder_config=_tiny_w2v_bert_config(),
        vocab_size=6,
        load_pretrained=False,
    )
    features = torch.randn(2, 12, 8)
    feature_lengths = torch.tensor([12, 10], dtype=torch.long)
    targets = torch.tensor([1, 2, 3], dtype=torch.long)
    target_lengths = torch.tensor([2, 1], dtype=torch.long)

    outputs = model(
        features,
        feature_lengths,
        return_training_outputs=True,
        targets=targets,
        target_lengths=target_lengths,
        blank_id=0,
        return_main_log_probs=True,
    )

    assert outputs["encoded"].shape == (2, 12, 16)
    assert outputs["output_lengths"].tolist() == [12, 10]
    assert outputs["main_log_probs"].shape == (2, 12, 6)
    assert torch.isfinite(outputs["main_ctc_loss"])
    outputs["main_ctc_loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_w2v_bert_uses_transformer_engine_linears_when_fp8_enabled(monkeypatch) -> None:
    class _FakeLinear(torch.nn.Linear):
        pass

    class _FakeTE:
        Linear = _FakeLinear

    monkeypatch.setattr(squeezeformer_model, "te", _FakeTE)

    model = W2VBertCTC(
        encoder_config=_tiny_w2v_bert_config(),
        vocab_size=16,
        load_pretrained=False,
        use_transformer_engine=True,
    )

    assert any(isinstance(module, _FakeLinear) for module in model.modules())

    outputs = model(
        torch.randn(2, 12, 8),
        torch.tensor([12, 10], dtype=torch.long),
        return_training_outputs=True,
        targets=torch.tensor([1, 2, 3], dtype=torch.long),
        target_lengths=torch.tensor([2, 1], dtype=torch.long),
        blank_id=0,
    )

    assert torch.isfinite(outputs["main_ctc_loss"])


def test_w2v_bert_feature_extractor_matches_asr_dataset_contract() -> None:
    class _FakeFeatureExtractor:
        sampling_rate = 16_000
        feature_size = 4
        stride = 2
        padding_value = 1.0

        def __call__(self, samples, *, sampling_rate, return_tensors, padding):
            assert len(samples) == 1
            assert sampling_rate == 16_000
            assert return_tensors == "pt"
            assert padding is True
            return {"input_features": torch.ones(1, 3, 8)}

    featurizer = W2VBertFeatureExtractor(
        model_source="fake",
        feature_extractor=_FakeFeatureExtractor(),
    )

    features = featurizer(torch.randn(16_000), 16_000)

    assert features.shape == (3, 8)
    assert featurizer.n_mels == 8
    assert featurizer.padding_value == 1.0
    assert featurizer.config_dict()["type"] == "w2v_bert"


def test_w2v_bert_feature_extractor_config_does_not_override_hf_defaults(monkeypatch) -> None:
    class _FakeFeatureExtractor:
        sampling_rate = 48_000
        feature_size = 6
        stride = 3
        padding_value = 7.0

        def __call__(self, samples, *, sampling_rate, return_tensors, padding):
            return {"input_features": torch.zeros(1, 2, 18)}

    class _FakeAutoFeatureExtractor:
        @staticmethod
        def from_pretrained(model_source, **kwargs):
            assert model_source == "custom/w2v-bert"
            assert kwargs == {"trust_remote_code": True}
            return _FakeFeatureExtractor()

    monkeypatch.setattr(w2v_bert_asr, "AutoFeatureExtractor", _FakeAutoFeatureExtractor)

    featurizer = W2VBertFeatureExtractor.from_config(
        {"type": "w2v_bert", "model_source": "custom/w2v-bert"}
    )

    assert featurizer.sample_rate == 48_000
    assert featurizer.n_mels == 18
    assert featurizer.hop_length == 480
    assert featurizer.padding_value == 7.0
