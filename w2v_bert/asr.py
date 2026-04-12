from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torchaudio
from torch import Tensor, nn
from torch.nn import functional as F

import squeezeformer_pytorch.model as _squeezeformer_model
from squeezeformer_pytorch.model import apply_linear_with_fp8_padding, make_linear

try:
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        Wav2Vec2BertModel,
    )
    from transformers import (
        Wav2Vec2BertConfig as HFWav2Vec2BertConfig,
    )
except ImportError:
    AutoConfig = None
    AutoFeatureExtractor = None
    HFWav2Vec2BertConfig = None
    Wav2Vec2BertModel = None


DEFAULT_W2V_BERT_MODEL = "facebook/w2v-bert-2.0"


def _require_transformers() -> None:
    if (
        AutoConfig is None
        or AutoFeatureExtractor is None
        or HFWav2Vec2BertConfig is None
        or Wav2Vec2BertModel is None
    ):
        raise RuntimeError(
            "W2V-BERT training requires transformers. Install the training dependencies and rerun."
        )


def _disable_hf_internal_specaugment(config: Any) -> Any:
    # The repository already owns SpecAugment in the data pipeline. Disabling the
    # Hugging Face masking path also avoids short-utterance failures from the
    # pretraining mask length.
    setattr(config, "apply_spec_augment", False)
    return config


def _is_fp8_compatible_linear(linear: nn.Linear) -> bool:
    return (
        linear.in_features % _squeezeformer_model.FP8_SHAPE_ALIGNMENT == 0
        and linear.out_features % _squeezeformer_model.FP8_SHAPE_ALIGNMENT == 0
    )


def _linear_to_transformer_engine(linear: nn.Linear) -> nn.Module | None:
    te = _squeezeformer_model.te
    if te is None or not _is_fp8_compatible_linear(linear):
        return None

    replacement = te.Linear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
    )
    replacement.to(device=linear.weight.device, dtype=linear.weight.dtype)
    with torch.no_grad():
        replacement.weight.copy_(linear.weight)
        replacement.weight.requires_grad_(linear.weight.requires_grad)
        if linear.bias is not None:
            replacement.bias.copy_(linear.bias)
            replacement.bias.requires_grad_(linear.bias.requires_grad)
    return replacement


def _convert_linear_modules_to_transformer_engine(module: nn.Module) -> int:
    converted = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            replacement = _linear_to_transformer_engine(child)
            if replacement is not None:
                setattr(module, child_name, replacement)
                converted += 1
                continue
        converted += _convert_linear_modules_to_transformer_engine(child)
    return converted


def _attention_mask_from_lengths(lengths: Tensor, max_length: int) -> Tensor:
    return (
        torch.arange(max_length, device=lengths.device).unsqueeze(0)
        < lengths.to(dtype=torch.long).unsqueeze(1)
    ).to(dtype=torch.long)


def _ctc_log_softmax(logits: Tensor) -> Tensor:
    return F.log_softmax(logits, dim=-1, dtype=torch.float32)


def _ctc_loss(
    log_probs: Tensor,
    output_lengths: Tensor,
    targets: Tensor,
    target_lengths: Tensor,
    *,
    blank_id: int,
) -> Tensor:
    per_sample_losses = F.ctc_loss(
        log_probs.transpose(0, 1),
        targets,
        output_lengths,
        target_lengths,
        blank=blank_id,
        reduction="none",
        zero_infinity=True,
    )
    return (per_sample_losses / target_lengths.clamp_min(1)).mean()


@dataclass(frozen=True)
class W2VBertConfig:
    architecture: str = "w2v_bert"
    model_name: str = DEFAULT_W2V_BERT_MODEL
    hidden_size: int = 1024
    feature_dim: int = 160
    sample_rate: int = 16_000
    model_config: dict[str, Any] = field(default_factory=dict)

    @property
    def model_dim(self) -> int:
        return self.hidden_size

    @classmethod
    def from_model_source(
        cls,
        model_source: str = DEFAULT_W2V_BERT_MODEL,
        *,
        sample_rate: int = 16_000,
        feature_dim: int = 160,
    ) -> "W2VBertConfig":
        _require_transformers()
        hf_config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
        hf_config = _disable_hf_internal_specaugment(hf_config)
        hidden_size = int(getattr(hf_config, "hidden_size", 1024))
        resolved_feature_dim = int(
            getattr(hf_config, "feature_projection_input_dim", feature_dim)
        )
        return cls(
            model_name=model_source,
            hidden_size=hidden_size,
            feature_dim=resolved_feature_dim,
            sample_rate=int(sample_rate),
            model_config=hf_config.to_dict(),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "W2VBertConfig":
        model_config = values.get("model_config", {})
        if not isinstance(model_config, Mapping):
            model_config = {}
        return cls(
            architecture=str(values.get("architecture", "w2v_bert")),
            model_name=str(values.get("model_name", DEFAULT_W2V_BERT_MODEL)),
            hidden_size=int(values.get("hidden_size", 1024)),
            feature_dim=int(values.get("feature_dim", 160)),
            sample_rate=int(values.get("sample_rate", 16_000)),
            model_config=dict(model_config),
        )

    def to_hf_config(self) -> Any:
        _require_transformers()
        if self.model_config:
            hf_config = HFWav2Vec2BertConfig.from_dict(dict(self.model_config))
        else:
            hf_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        return _disable_hf_internal_specaugment(hf_config)


def w2v_bert_featurizer_config(
    model_source: str = DEFAULT_W2V_BERT_MODEL,
) -> dict[str, object]:
    return {
        "type": "w2v_bert",
        "model_source": model_source,
    }


class W2VBertFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_source: str = DEFAULT_W2V_BERT_MODEL,
        *,
        feature_extractor: Any | None = None,
        sample_rate: int | None = None,
        feature_dim: int | None = None,
        padding_value: float | None = None,
    ) -> None:
        super().__init__()
        _require_transformers()
        self.model_source = model_source
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else AutoFeatureExtractor.from_pretrained(model_source, trust_remote_code=True)
        )
        self.sample_rate = int(
            sample_rate
            if sample_rate is not None
            else getattr(self.feature_extractor, "sampling_rate", 16_000)
        )
        feature_size = int(getattr(self.feature_extractor, "feature_size", 80))
        stride = int(getattr(self.feature_extractor, "stride", 2))
        self.feature_size = feature_size
        self.stride = max(1, stride)
        self.n_mels = int(feature_dim if feature_dim is not None else feature_size * self.stride)
        self.hop_length = 160 * self.stride
        self.padding_value = float(
            padding_value
            if padding_value is not None
            else getattr(self.feature_extractor, "padding_value", 0.0)
        )

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "W2VBertFeatureExtractor":
        return cls(
            model_source=str(config.get("model_source", DEFAULT_W2V_BERT_MODEL)),
            sample_rate=int(config.get("sample_rate", 16_000)),
            feature_dim=int(config.get("feature_dim", 160)),
            padding_value=float(config.get("padding_value", 1.0)),
        )

    def forward(self, waveform: Tensor, sample_rate: int) -> Tensor:
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        elif waveform.dim() != 1:
            raise ValueError(f"Expected waveform with shape [time] or [channels, time], got {tuple(waveform.shape)}.")
        if int(sample_rate) != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                int(sample_rate),
                self.sample_rate,
            ).squeeze(0)
        waveform = waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()
        processed = self.feature_extractor(
            [waveform.numpy()],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        features = processed["input_features"][0].to(dtype=torch.float32).contiguous()
        if features.dim() != 2:
            raise RuntimeError(
                "W2V-BERT feature extractor returned unexpected input_features shape "
                f"{tuple(features.shape)}."
            )
        return features

    def estimate_num_frames(self, num_samples: int, sample_rate: int) -> int:
        if num_samples <= 0 or sample_rate <= 0:
            return 0
        effective_samples = int(num_samples)
        if int(sample_rate) != self.sample_rate:
            effective_samples = max(
                1,
                int(math.ceil((effective_samples * self.sample_rate) / int(sample_rate))),
            )
        return max(1, int(math.ceil(effective_samples / max(1, self.hop_length))))

    def config_dict(self) -> dict[str, object]:
        return {
            "type": "w2v_bert",
            "model_source": self.model_source,
            "sample_rate": self.sample_rate,
            "feature_size": self.feature_size,
            "stride": self.stride,
            "feature_dim": self.n_mels,
            "padding_value": self.padding_value,
        }


class W2VBertCTC(nn.Module):
    def __init__(
        self,
        encoder_config: W2VBertConfig,
        vocab_size: int,
        *,
        pretrained_model_name_or_path: str | None = None,
        load_pretrained: bool = False,
        use_transformer_engine: bool = False,
    ) -> None:
        super().__init__()
        _require_transformers()
        self.encoder_config = encoder_config
        self.aed_decoder = None
        self.use_transformer_engine = use_transformer_engine
        hf_config = encoder_config.to_hf_config()
        if load_pretrained:
            self.encoder = Wav2Vec2BertModel.from_pretrained(
                pretrained_model_name_or_path or encoder_config.model_name,
                config=hf_config,
                trust_remote_code=True,
            )
        else:
            self.encoder = Wav2Vec2BertModel(hf_config)
        if use_transformer_engine:
            _convert_linear_modules_to_transformer_engine(self.encoder)
        self.classifier = make_linear(
            encoder_config.hidden_size,
            vocab_size,
            use_transformer_engine=use_transformer_engine,
        )
        self._initialize_ctc_head()

    def _initialize_ctc_head(self) -> None:
        bias = getattr(self.classifier, "bias", None)
        if bias is None:
            return
        with torch.no_grad():
            bias.zero_()

    def _output_lengths(self, feature_lengths: Tensor, encoded_length: int) -> Tensor:
        lengths = feature_lengths.to(dtype=torch.long)
        if bool(getattr(self.encoder.config, "add_adapter", False)):
            stride = max(1, int(getattr(self.encoder.config, "adapter_stride", 1)))
            num_layers = max(1, int(getattr(self.encoder.config, "num_adapter_layers", 1)))
            for _ in range(num_layers):
                lengths = torch.div(lengths + stride - 1, stride, rounding_mode="floor")
        return lengths.clamp(min=1, max=encoded_length)

    def _encode(self, features: Tensor, feature_lengths: Tensor) -> tuple[Tensor, Tensor]:
        if features.size(-1) != self.encoder_config.feature_dim:
            raise ValueError(
                "W2V-BERT expected feature dimension "
                f"{self.encoder_config.feature_dim}, got {features.size(-1)}."
            )
        feature_lengths = feature_lengths.to(device=features.device, dtype=torch.long).clamp(
            min=1,
            max=features.size(1),
        )
        attention_mask = _attention_mask_from_lengths(feature_lengths, features.size(1))
        outputs = self.encoder(
            input_features=features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoded = outputs.last_hidden_state
        output_lengths = self._output_lengths(feature_lengths, encoded.size(1))
        return encoded, output_lengths

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
        *,
        return_training_outputs: bool = False,
        targets: Tensor | None = None,
        target_lengths: Tensor | None = None,
        blank_id: int | None = None,
        return_main_log_probs: bool = False,
        decoder_inputs: Tensor | None = None,
        liberta_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | dict[str, Any]:
        del liberta_lengths
        if decoder_inputs is not None:
            raise RuntimeError("AED decoder is not supported by the W2V-BERT training path.")

        encoded, output_lengths = self._encode(features, feature_lengths)
        logits = apply_linear_with_fp8_padding(self.classifier, encoded)
        if not return_training_outputs:
            return logits, output_lengths

        log_probs = _ctc_log_softmax(logits)
        main_ctc_loss = None
        if targets is not None and target_lengths is not None and blank_id is not None:
            main_ctc_loss = _ctc_loss(
                log_probs,
                output_lengths,
                targets,
                target_lengths,
                blank_id=blank_id,
            )
        output: dict[str, Any] = {
            "encoded": encoded,
            "output_lengths": output_lengths,
            "main_ctc_loss": main_ctc_loss,
        }
        if return_main_log_probs:
            output["main_logits"] = logits
            output["main_log_probs"] = log_probs
        return output

    def log_probs(self, features: Tensor, feature_lengths: Tensor) -> tuple[Tensor, Tensor]:
        logits, output_lengths = self(features, feature_lengths)
        return _ctc_log_softmax(logits), output_lengths

    def to_config_dict(self) -> dict[str, object]:
        return {
            "architecture": self.encoder_config.architecture,
            "model_name": self.encoder_config.model_name,
            "hidden_size": self.encoder_config.hidden_size,
            "feature_dim": self.encoder_config.feature_dim,
            "sample_rate": self.encoder_config.sample_rate,
            "model_config": dict(self.encoder_config.model_config),
        }
