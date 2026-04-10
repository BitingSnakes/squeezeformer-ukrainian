from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from zipformer_pytorch.zipformer import Zipformer


_DEFAULT_DOWNSAMPLING = (1, 2, 4, 8, 4, 2)
_DEFAULT_ENCODER_DIM = (192, 256, 384, 512, 384, 256)
_DEFAULT_NUM_LAYERS = (2, 2, 3, 4, 3, 2)
_DEFAULT_NUM_HEADS = (4, 4, 4, 8, 4, 4)
_DEFAULT_FEEDFORWARD_DIM = (512, 768, 1024, 1536, 1024, 768)
_DEFAULT_CNN_KERNELS = (31, 31, 15, 15, 15, 31)


def _expand_stack_value(name: str, value: tuple[int, ...], num_stacks: int) -> tuple[int, ...]:
    if len(value) == num_stacks:
        return value
    if len(value) == 1:
        return value * num_stacks
    raise ValueError(
        f"ZipformerConfig.{name} must have length 1 or {num_stacks}, got {len(value)}."
    )


@dataclass(frozen=True)
class ZipformerConfig:
    architecture: str = "zipformer"
    input_dim: int = 80
    output_downsampling_factor: int = 2
    downsampling_factor: tuple[int, ...] = _DEFAULT_DOWNSAMPLING
    encoder_dim: tuple[int, ...] = _DEFAULT_ENCODER_DIM
    num_encoder_layers: tuple[int, ...] = _DEFAULT_NUM_LAYERS
    num_heads: tuple[int, ...] = _DEFAULT_NUM_HEADS
    query_head_dim: tuple[int, ...] = (32,)
    value_head_dim: tuple[int, ...] = (12,)
    pos_head_dim: tuple[int, ...] = (4,)
    feedforward_dim: tuple[int, ...] = _DEFAULT_FEEDFORWARD_DIM
    cnn_module_kernel: tuple[int, ...] = _DEFAULT_CNN_KERNELS
    pos_dim: int = 48
    dropout: float = 0.1

    @property
    def num_stacks(self) -> int:
        return len(self.downsampling_factor)

    @property
    def resolved_query_head_dim(self) -> tuple[int, ...]:
        return _expand_stack_value("query_head_dim", self.query_head_dim, self.num_stacks)

    @property
    def resolved_value_head_dim(self) -> tuple[int, ...]:
        return _expand_stack_value("value_head_dim", self.value_head_dim, self.num_stacks)

    @property
    def resolved_pos_head_dim(self) -> tuple[int, ...]:
        return _expand_stack_value("pos_head_dim", self.pos_head_dim, self.num_stacks)

    @property
    def model_dim(self) -> int:
        return max(self.encoder_dim)

    @property
    def num_layers(self) -> int:
        return sum(self.num_encoder_layers)


ZIPFORMER_VARIANTS = {
    "xs": ZipformerConfig(
        encoder_dim=(64, 96, 128, 160, 128, 96),
        num_encoder_layers=(1, 1, 1, 1, 1, 1),
        num_heads=(4, 4, 4, 4, 4, 4),
        query_head_dim=(16,),
        value_head_dim=(8,),
        feedforward_dim=(192, 256, 384, 512, 384, 256),
        pos_dim=24,
    ),
    # Paper S/M/L ladder from the Zipformer paper.
    "s": ZipformerConfig(
        encoder_dim=(192, 256, 256, 256, 256, 256),
        num_encoder_layers=(2, 2, 2, 2, 2, 2),
        num_heads=(4, 4, 4, 4, 4, 4),
        query_head_dim=(32,),
        value_head_dim=(12,),
        feedforward_dim=(512, 768, 768, 768, 768, 768),
        pos_dim=48,
    ),
    "m": ZipformerConfig(),
    "l": ZipformerConfig(
        encoder_dim=(192, 256, 512, 768, 512, 256),
        num_encoder_layers=(2, 2, 4, 5, 4, 2),
        num_heads=(4, 4, 8, 8, 8, 4),
        query_head_dim=(32,),
        value_head_dim=(12,),
        feedforward_dim=(512, 768, 1536, 2048, 1536, 768),
        pos_dim=48,
    ),
}

_ZIPFORMER_VARIANT_ALIASES = {
    # Backward-compatible legacy names.
    "sm": "m",
    "ml": "l",
}


def zipformer_variant(name: str) -> ZipformerConfig:
    name = _ZIPFORMER_VARIANT_ALIASES.get(name, name)
    try:
        return ZIPFORMER_VARIANTS[name]
    except KeyError as error:
        raise KeyError(f"Unknown Zipformer variant: {name}") from error


def _make_padding_mask(lengths: Tensor, *, max_length: int) -> Tensor:
    return torch.arange(max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


class ZipformerEncoder(nn.Module):
    def __init__(self, config: ZipformerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Zipformer(
            input_dim=config.input_dim,
            output_downsampling_factor=config.output_downsampling_factor,
            downsampling_factor=config.downsampling_factor,
            encoder_dim=config.encoder_dim,
            num_encoder_layers=config.num_encoder_layers,
            num_heads=config.num_heads,
            query_head_dim=config.resolved_query_head_dim,
            pos_head_dim=config.resolved_pos_head_dim,
            value_head_dim=config.resolved_value_head_dim,
            feedforward_dim=config.feedforward_dim,
            cnn_module_kernel=config.cnn_module_kernel,
            pos_dim=config.pos_dim,
            dropout=config.dropout,
        )

    def set_batch_count(self, batch_count: int) -> None:
        self.encoder.set_batch_count(batch_count)

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if features.size(-1) != self.config.input_dim:
            raise ValueError(
                "Zipformer encoder expected feature dimension "
                f"{self.config.input_dim}, got {features.size(-1)}."
            )
        output_lengths = feature_lengths.to(dtype=torch.int64).clamp_(1, features.size(1))
        encoded, output_lengths = self.encoder(features, output_lengths)
        return encoded, output_lengths


class ZipformerCTC(nn.Module):
    def __init__(
        self,
        encoder_config: ZipformerConfig,
        vocab_size: int,
        *,
        audio_teacher_enabled: bool = False,
        audio_teacher_hidden_size: int = 1024,
        audio_teacher_target: str = "encoder",
        initial_ctc_blank_bias: float = 0.0,
        blank_logit_offset: float = 0.0,
        blank_logit_regularization_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.intermediate_ctc_layers: tuple[int, ...] = ()
        self.blank_prune_layer = None
        self.blank_prune_threshold = 0.0
        self.blank_prune_min_keep_frames = 1
        self.aed_decoder = None
        self.audio_teacher_target = audio_teacher_target
        self.initial_ctc_blank_bias = float(initial_ctc_blank_bias)
        self.blank_logit_offset = float(blank_logit_offset)
        self.blank_logit_regularization_weight = float(blank_logit_regularization_weight)

        self.encoder = ZipformerEncoder(encoder_config)
        self.classifier = nn.Linear(encoder_config.model_dim, vocab_size)
        self.audio_teacher_projection = (
            nn.Linear(encoder_config.model_dim, audio_teacher_hidden_size)
            if audio_teacher_enabled and audio_teacher_target == "encoder"
            else None
        )
        self._initialize_ctc_head(blank_bias=self.initial_ctc_blank_bias)

    def set_batch_count(self, batch_count: int) -> None:
        self.encoder.set_batch_count(batch_count)

    def _initialize_ctc_head(self, *, blank_bias: float) -> None:
        with torch.no_grad():
            self.classifier.bias.zero_()
            self.classifier.bias[0] = float(blank_bias)

    def _apply_training_blank_logit_offset(self, logits: Tensor) -> Tensor:
        if not self.training or self.blank_logit_offset <= 0.0:
            return logits
        adjusted_logits = logits.clone()
        adjusted_logits[..., 0] = adjusted_logits[..., 0] - self.blank_logit_offset
        return adjusted_logits

    @staticmethod
    def _ctc_log_softmax(logits: Tensor) -> Tensor:
        return F.log_softmax(logits, dim=-1, dtype=torch.float32)

    def _blank_logit_regularization_from_logits(
        self,
        logits: Tensor,
        output_lengths: Tensor,
        *,
        blank_id: int,
    ) -> Tensor:
        if self.blank_logit_regularization_weight <= 0.0:
            return logits.new_zeros((), dtype=torch.float32)
        valid_mask = torch.arange(logits.size(1), device=output_lengths.device).unsqueeze(0) < output_lengths.unsqueeze(1)
        if not bool(valid_mask.any()):
            return logits.new_zeros((), dtype=torch.float32)
        blank_logits = logits[..., blank_id]
        nonblank_logits = logits.clone()
        nonblank_logits[..., blank_id] = float("-inf")
        best_nonblank_logits = nonblank_logits.max(dim=-1).values
        positive_margin = (blank_logits - best_nonblank_logits).masked_select(valid_mask).relu()
        if positive_margin.numel() == 0:
            return logits.new_zeros((), dtype=torch.float32)
        return positive_margin.float().mean()

    def _ctc_loss(
        self,
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

    def project_encoder_for_audio_teacher(self, hidden: Tensor, lengths: Tensor) -> Tensor:
        if self.audio_teacher_projection is None:
            raise RuntimeError("Audio teacher projection head is disabled for this model.")
        mask = _make_padding_mask(lengths, max_length=hidden.size(1)).unsqueeze(-1)
        pooled = hidden.masked_fill(~mask, 0.0).sum(dim=1)
        pooled = pooled / lengths.clamp_min(1).to(device=hidden.device, dtype=hidden.dtype).unsqueeze(1)
        return self.audio_teacher_projection(pooled)

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
            raise RuntimeError("AED decoder is not supported by the Zipformer training path.")

        encoded, output_lengths = self.encoder(features, feature_lengths)
        logits = self.classifier(encoded)
        if not return_training_outputs:
            return logits, output_lengths

        output: dict[str, Any] = {
            "encoded": encoded,
            "output_lengths": output_lengths,
            "main_ctc_loss": None,
            "blank_logit_regularization_loss": encoded.new_zeros((), dtype=torch.float32),
            "intermediate_ctc_losses": {},
            "intermediate_ctc_diagnostics": {},
        }
        adjusted_logits = self._apply_training_blank_logit_offset(logits)
        main_log_probs = self._ctc_log_softmax(adjusted_logits)
        if targets is not None and target_lengths is not None and blank_id is not None:
            output["main_ctc_loss"] = self._ctc_loss(
                main_log_probs,
                output_lengths,
                targets,
                target_lengths,
                blank_id=blank_id,
            )
            output["blank_logit_regularization_loss"] = self._blank_logit_regularization_from_logits(
                logits,
                output_lengths,
                blank_id=blank_id,
            )
        if return_main_log_probs:
            output["main_logits"] = logits
            output["main_log_probs"] = main_log_probs
        if self.audio_teacher_projection is not None:
            output["audio_teacher_student_states"] = self.project_encoder_for_audio_teacher(
                encoded,
                output_lengths,
            )
        return output

    def log_probs(self, features: Tensor, feature_lengths: Tensor) -> tuple[Tensor, Tensor]:
        logits, output_lengths = self(features, feature_lengths)
        return self._ctc_log_softmax(logits), output_lengths

    def to_config_dict(self) -> dict[str, object]:
        return asdict(self.encoder_config)
