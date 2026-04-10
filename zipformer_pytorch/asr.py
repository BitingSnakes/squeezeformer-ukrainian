from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from zipformer_pytorch.zipformer import ZipformerBlock


@dataclass(frozen=True)
class ZipformerConfig:
    architecture: str = "zipformer"
    input_dim: int = 80
    model_dim: int = 256
    num_layers: int = 6
    num_heads: int = 4
    ff_mult: int = 4
    input_dropout: float = 0.1


ZIPFORMER_VARIANTS = {
    "xs": ZipformerConfig(model_dim=128, num_layers=4, num_heads=4),
    "s": ZipformerConfig(model_dim=192, num_layers=5, num_heads=4),
    "sm": ZipformerConfig(model_dim=256, num_layers=6, num_heads=4),
    "m": ZipformerConfig(model_dim=320, num_layers=8, num_heads=4),
    "ml": ZipformerConfig(model_dim=384, num_layers=10, num_heads=6),
    "l": ZipformerConfig(model_dim=512, num_layers=12, num_heads=8),
}


def zipformer_variant(name: str) -> ZipformerConfig:
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
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_projection = nn.Linear(config.input_dim, config.model_dim)
        self.input_dropout = nn.Dropout(config.input_dropout)
        self.blocks = nn.ModuleList(
            [
                ZipformerBlock(
                    config.model_dim,
                    heads=config.num_heads,
                    mult=config.ff_mult,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.model_dim)

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
        padding_mask = _make_padding_mask(output_lengths, max_length=features.size(1))

        x = self.input_projection(self.input_norm(features))
        x = self.input_dropout(x)
        x = x.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            x = block(x, mask=padding_mask)
            x = x.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        x = self.output_norm(x)
        x = x.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        return x, output_lengths


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
        valid_mask = torch.arange(logits.size(1), device=output_lengths.device).unsqueeze(
            0
        ) < output_lengths.unsqueeze(1)
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
        pooled = pooled / lengths.clamp_min(1).to(
            device=hidden.device, dtype=hidden.dtype
        ).unsqueeze(1)
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
            output["blank_logit_regularization_loss"] = (
                self._blank_logit_regularization_from_logits(
                    logits,
                    output_lengths,
                    blank_id=blank_id,
                )
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
