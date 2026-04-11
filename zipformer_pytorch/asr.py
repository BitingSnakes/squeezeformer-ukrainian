from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from zipformer_pytorch.transducer import (
    StatelessDecoder,
    TransducerJoiner,
    batched_rnnt_loss,
    linear_prune_ranges,
    make_decoder_inputs,
    targets_to_padded,
    transducer_greedy_search_batch,
    transducer_modified_beam_search_batch,
)
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
        num_heads=(4, 4, 4, 8, 4, 4),
        query_head_dim=(32,),
        value_head_dim=(12,),
        feedforward_dim=(512, 768, 768, 768, 768, 768),
        pos_dim=48,
    ),
    "m": ZipformerConfig(),
    "l": ZipformerConfig(
        encoder_dim=(192, 256, 512, 768, 512, 256),
        num_encoder_layers=(2, 2, 4, 5, 4, 2),
        num_heads=(4, 4, 4, 8, 4, 4),
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
        output_lengths = feature_lengths.to(dtype=torch.int64).clamp(1, features.size(1))
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
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.aed_decoder = None
        self.audio_teacher_target = audio_teacher_target

        self.encoder = ZipformerEncoder(encoder_config)
        self.classifier = nn.Linear(encoder_config.model_dim, vocab_size)
        self.audio_teacher_projection = (
            nn.Linear(encoder_config.model_dim, audio_teacher_hidden_size)
            if audio_teacher_enabled and audio_teacher_target == "encoder"
            else None
        )
        self._initialize_ctc_head()

    def set_batch_count(self, batch_count: int) -> None:
        self.encoder.set_batch_count(batch_count)

    def _initialize_ctc_head(self) -> None:
        with torch.no_grad():
            self.classifier.bias.zero_()

    @staticmethod
    def _ctc_log_softmax(logits: Tensor) -> Tensor:
        return F.log_softmax(logits, dim=-1, dtype=torch.float32)

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
        }
        main_log_probs = self._ctc_log_softmax(logits)
        if targets is not None and target_lengths is not None and blank_id is not None:
            output["main_ctc_loss"] = self._ctc_loss(
                main_log_probs,
                output_lengths,
                targets,
                target_lengths,
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


class ZipformerTransducer(nn.Module):
    is_transducer = True

    def __init__(
        self,
        encoder_config: ZipformerConfig,
        vocab_size: int,
        *,
        blank_id: int = 0,
        decoder_dim: int = 512,
        joiner_dim: int = 512,
        context_size: int = 2,
        prune_range: int = 5,
        joiner_chunk_size: int = 32,
        audio_teacher_enabled: bool = False,
        audio_teacher_hidden_size: int = 1024,
        audio_teacher_target: str = "encoder",
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.blank_id = int(blank_id)
        self.decoder_dim = int(decoder_dim)
        self.joiner_dim = int(joiner_dim)
        self.context_size = int(context_size)
        self.prune_range = int(prune_range)
        self.joiner_chunk_size = max(1, int(joiner_chunk_size))
        self.audio_teacher_target = audio_teacher_target
        self.aed_decoder = None

        self.encoder = ZipformerEncoder(encoder_config)
        self.decoder = StatelessDecoder(
            vocab_size=vocab_size,
            decoder_dim=self.decoder_dim,
            blank_id=self.blank_id,
            context_size=self.context_size,
        )
        self.joiner = TransducerJoiner(
            encoder_dim=encoder_config.model_dim,
            decoder_dim=self.decoder_dim,
            joiner_dim=self.joiner_dim,
            vocab_size=vocab_size,
        )
        self.audio_teacher_projection = (
            nn.Linear(encoder_config.model_dim, audio_teacher_hidden_size)
            if audio_teacher_enabled and audio_teacher_target == "encoder"
            else None
        )

    def set_batch_count(self, batch_count: int) -> None:
        self.encoder.set_batch_count(batch_count)

    def project_encoder_for_audio_teacher(self, hidden: Tensor, lengths: Tensor) -> Tensor:
        if self.audio_teacher_projection is None:
            raise RuntimeError("Audio teacher projection head is disabled for this model.")
        mask = _make_padding_mask(lengths, max_length=hidden.size(1)).unsqueeze(-1)
        pooled = hidden.masked_fill(~mask, 0.0).sum(dim=1)
        pooled = pooled / lengths.clamp_min(1).to(device=hidden.device, dtype=hidden.dtype).unsqueeze(1)
        return self.audio_teacher_projection(pooled)

    def encode(self, features: Tensor, feature_lengths: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder(features, feature_lengths)

    def _banded_log_probs(
        self,
        encoded: Tensor,
        decoder_out: Tensor,
        targets_padded: Tensor,
        target_lengths: Tensor,
        output_lengths: Tensor,
        ranges: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, max_time, _ = encoded.shape
        max_target_length = int(targets_padded.size(1))
        u_plus_one = max_target_length + 1
        blank_chunks: list[Tensor] = []
        label_chunks: list[Tensor] = []
        projected_encoder = self.joiner.encoder_proj(encoded)
        projected_decoder = self.joiner.decoder_proj(decoder_out)
        target_lengths_expanded = target_lengths.view(batch_size, 1, 1)

        for start in range(0, max_time, self.joiner_chunk_size):
            stop = min(max_time, start + self.joiner_chunk_size)
            current_ranges = ranges[:, start:stop]
            current_encoder = projected_encoder[:, start:stop].unsqueeze(2)
            gather_index = current_ranges.unsqueeze(-1).expand(
                -1,
                -1,
                -1,
                projected_decoder.size(-1),
            )
            current_decoder = torch.gather(
                projected_decoder.unsqueeze(1).expand(-1, stop - start, -1, -1),
                dim=2,
                index=gather_index,
            )
            logits = self.joiner(
                current_encoder,
                current_decoder,
                project_input=False,
            )
            log_probs = F.log_softmax(logits.float(), dim=-1)
            blank_chunk = encoded.new_full(
                (batch_size, stop - start, u_plus_one),
                float("-inf"),
                dtype=torch.float32,
            ).scatter_reduce(
                2,
                current_ranges,
                log_probs[..., self.blank_id],
                reduce="amax",
                include_self=True,
            )
            if max_target_length == 0:
                blank_chunks.append(blank_chunk)
                label_chunks.append(
                    encoded.new_full(
                        (batch_size, stop - start, 0),
                        float("-inf"),
                        dtype=torch.float32,
                    )
                )
                continue
            label_indices = current_ranges.clamp_max(max_target_length - 1)
            target_band = torch.gather(
                targets_padded.unsqueeze(1).expand(-1, stop - start, -1),
                dim=2,
                index=label_indices,
            )
            label_scores = torch.gather(
                log_probs,
                dim=-1,
                index=target_band.unsqueeze(-1),
            ).squeeze(-1)
            valid_emit = current_ranges < target_lengths_expanded
            label_scores = torch.where(
                valid_emit,
                label_scores,
                label_scores.new_full(label_scores.shape, float("-inf")),
            )
            label_chunk = encoded.new_full(
                (batch_size, stop - start, max_target_length),
                float("-inf"),
                dtype=torch.float32,
            ).scatter_reduce(
                2,
                label_indices,
                label_scores,
                reduce="amax",
                include_self=True,
            )
            blank_chunks.append(blank_chunk)
            label_chunks.append(label_chunk)

        blank_log_probs = torch.cat(blank_chunks, dim=1)
        label_log_probs = (
            torch.cat(label_chunks, dim=1)
            if label_chunks
            else encoded.new_full((batch_size, max_time, 0), float("-inf"), dtype=torch.float32)
        )

        valid_time = torch.arange(max_time, device=output_lengths.device).view(1, -1, 1) < output_lengths.view(
            batch_size,
            1,
            1,
        )
        valid_blank_u = torch.arange(u_plus_one, device=target_lengths.device).view(1, 1, -1) <= target_lengths.view(
            batch_size,
            1,
            1,
        )
        blank_log_probs = torch.where(
            valid_time & valid_blank_u,
            blank_log_probs,
            blank_log_probs.new_full(blank_log_probs.shape, float("-inf")),
        )
        if max_target_length > 0:
            valid_label_u = torch.arange(max_target_length, device=target_lengths.device).view(
                1,
                1,
                -1,
            ) < target_lengths.view(batch_size, 1, 1)
            label_log_probs = torch.where(
                valid_time & valid_label_u,
                label_log_probs,
                label_log_probs.new_full(label_log_probs.shape, float("-inf")),
            )
        return blank_log_probs, label_log_probs

    def _pruned_transducer_loss(
        self,
        encoded: Tensor,
        output_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        targets_padded = targets_to_padded(targets, target_lengths)
        decoder_inputs = make_decoder_inputs(targets_padded, blank_id=self.blank_id)
        decoder_out = self.decoder(decoder_inputs, need_pad=True)
        ranges = linear_prune_ranges(
            output_lengths,
            target_lengths,
            prune_range=self.prune_range,
        )
        blank_log_probs, label_log_probs = self._banded_log_probs(
            encoded,
            decoder_out,
            targets_padded,
            target_lengths,
            output_lengths,
            ranges,
        )
        return batched_rnnt_loss(
            blank_log_probs,
            label_log_probs,
            output_lengths.to(torch.long),
            target_lengths.to(torch.long),
        )

    def decode_token_ids(
        self,
        encoded: Tensor,
        output_lengths: Tensor,
        *,
        strategy: str,
        beam_size: int = 4,
    ) -> list[list[int]]:
        if strategy == "greedy":
            return transducer_greedy_search_batch(self, encoded, output_lengths)
        return transducer_modified_beam_search_batch(
            self,
            encoded,
            output_lengths,
            beam_size=beam_size,
        )

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
        del decoder_inputs, liberta_lengths, return_main_log_probs
        if blank_id is not None and int(blank_id) != self.blank_id:
            raise ValueError(
                f"ZipformerTransducer was built with blank_id={self.blank_id}, got {blank_id}."
            )
        encoded, output_lengths = self.encoder(features, feature_lengths)
        if not return_training_outputs:
            return encoded, output_lengths

        output: dict[str, Any] = {
            "encoded": encoded,
            "output_lengths": output_lengths,
            "main_ctc_loss": None,
            "main_logits": None,
            "main_log_probs": None,
            "pruned_transducer_loss": None,
        }
        if targets is not None and target_lengths is not None:
            pruned_loss = self._pruned_transducer_loss(
                encoded,
                output_lengths,
                targets,
                target_lengths,
            )
            output["main_ctc_loss"] = pruned_loss
            output["pruned_transducer_loss"] = pruned_loss
        if self.audio_teacher_projection is not None:
            output["audio_teacher_student_states"] = self.project_encoder_for_audio_teacher(
                encoded,
                output_lengths,
            )
        return output

    def to_config_dict(self) -> dict[str, object]:
        return asdict(self.encoder_config)
