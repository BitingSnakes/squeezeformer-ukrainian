from __future__ import annotations

import base64
import importlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import sentencepiece as spm
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .masking import make_padding_mask, make_sequence_mask
from .model import (
    SqueezeformerConfig,
    SqueezeformerEncoder,
    apply_linear_with_fp8_padding,
    make_linear,
)

_BLANK_PRUNE_TARGET_BYTES = 128 * 1024 * 1024
DEFAULT_INITIAL_CTC_BLANK_BIAS = 0.0
DEFAULT_CTC_BEAM_LENGTH_BONUS = 0.1


class TrainingOutputs(dict[str, Any]):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del cls, types, kwargs
        if func is torch.isfinite:
            if not args:
                return torch.tensor(True)
            instance = args[0]
            finite_values: list[Tensor] = []
            for value in instance.values():
                if isinstance(value, Tensor):
                    finite_values.append(torch.isfinite(value).all())
                elif isinstance(value, dict):
                    for nested_value in value.values():
                        if isinstance(nested_value, Tensor):
                            finite_values.append(torch.isfinite(nested_value).all())
            if not finite_values:
                return torch.tensor(True)
            return torch.stack(finite_values).all()
        return NotImplemented


def mean_pool_sequence(x: Tensor, lengths: Tensor) -> Tensor:
    mask = make_sequence_mask(lengths, x.size(1))
    masked = x * mask.unsqueeze(-1).to(dtype=x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=x.dtype)
    return masked.sum(dim=1) / denom


class Tokenizer:
    blank_id: int
    pad_id: int

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: Iterable[int]) -> str:
        raise NotImplementedError

    def decode_ctc(self, token_ids: Iterable[int]) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict[str, object]:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError


class CharacterTokenizer(Tokenizer):
    def __init__(self, symbols: list[str]) -> None:
        if "<blank>" in symbols:
            raise ValueError("Do not include the reserved <blank> token in symbols.")
        self.blank_id = 0
        self.pad_id = 0
        self.id_to_token = ["<blank>"] + symbols
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}

    @classmethod
    def build(cls, texts: Iterable[str], min_frequency: int = 1) -> "CharacterTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text)
        symbols = sorted(token for token, count in counter.items() if count >= min_frequency)
        return cls(symbols=symbols)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        unknown_chars = sorted({char for char in text if char not in self.token_to_id})
        if unknown_chars:
            formatted = ", ".join(repr(char) for char in unknown_chars)
            raise ValueError(
                f"CharacterTokenizer encountered character(s) outside the vocabulary: {formatted}"
            )
        return [self.token_to_id[char] for char in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(self.id_to_token[index] for index in token_ids if index != self.blank_id)

    def decode_ctc(self, token_ids: Iterable[int]) -> str:
        result: list[str] = []
        previous = self.blank_id
        for token_id in token_ids:
            if token_id != self.blank_id and token_id != previous:
                result.append(self.id_to_token[token_id])
            previous = token_id
        return "".join(result)

    def to_dict(self) -> dict[str, object]:
        return {"type": "character", "symbols": self.id_to_token[1:]}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CharacterTokenizer":
        symbols = payload.get("symbols")
        if not isinstance(symbols, list) or not all(isinstance(item, str) for item in symbols):
            raise ValueError("Tokenizer payload must contain a string list under 'symbols'.")
        return cls(symbols=symbols)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, processor: spm.SentencePieceProcessor, model_proto: bytes) -> None:
        self.processor = processor
        self.model_proto = model_proto
        self.blank_id = 0
        self.pad_id = 0
        self._validate_nonblank_pieces_nonempty()

    def _validate_nonblank_pieces_nonempty(self) -> None:
        invalid_tokens: list[str] = []
        id_to_piece = getattr(self.processor, "id_to_piece", None)
        for token_id in range(1, self.vocab_size):
            piece = "<unknown>"
            if callable(id_to_piece):
                try:
                    piece = str(id_to_piece(token_id))
                except Exception:
                    piece = "<unknown>"
            if piece:
                continue
            invalid_tokens.append(f"{token_id}:{piece!r}")
            if len(invalid_tokens) >= 8:
                break
        if invalid_tokens:
            raise ValueError(
                "SentencePiece tokenizer contains non-blank token(s) with empty pieces, "
                "which is incompatible with CTC decoding. Invalid ids/pieces: "
                + ", ".join(invalid_tokens)
            )

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        model_prefix: str | Path,
        vocab_size: int = 256,
        model_type: str = "unigram",
        character_coverage: float = 1.0,
    ) -> "SentencePieceTokenizer":
        model_prefix = str(model_prefix)
        text_path = f"{model_prefix}.training.txt"
        with open(text_path, "w", encoding="utf-8") as handle:
            for text in texts:
                handle.write(text.replace("\n", " ").strip())
                handle.write("\n")

        spm.SentencePieceTrainer.train(
            input=text_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=0,
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
        )
        model_path = f"{model_prefix}.model"
        model_proto = Path(model_path).read_bytes()
        processor = spm.SentencePieceProcessor(model_file=model_path)
        return cls(processor=processor, model_proto=model_proto)

    @classmethod
    def from_model_bytes(cls, model_proto: bytes) -> "SentencePieceTokenizer":
        processor = spm.SentencePieceProcessor(model_proto=model_proto)
        return cls(processor=processor, model_proto=model_proto)

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    def encode(self, text: str) -> list[int]:
        return list(self.processor.encode(text, out_type=int))

    def decode(self, token_ids: Iterable[int]) -> str:
        filtered = [token_id for token_id in token_ids if token_id != self.blank_id]
        return self.processor.decode(filtered)

    def decode_ctc(self, token_ids: Iterable[int]) -> str:
        result: list[int] = []
        previous = self.blank_id
        for token_id in token_ids:
            if token_id != self.blank_id and token_id != previous:
                result.append(token_id)
            previous = token_id
        return self.processor.decode(result)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "sentencepiece",
            "model_proto_b64": base64.b64encode(self.model_proto).decode("ascii"),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix == ".model":
            path.write_bytes(self.model_proto)
            return
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def tokenizer_from_dict(payload: dict[str, object]) -> Tokenizer:
    tokenizer_type = payload.get("type", "character")
    if tokenizer_type == "character":
        return CharacterTokenizer.from_dict(payload)
    if tokenizer_type == "sentencepiece":
        model_b64 = payload.get("model_proto_b64")
        if not isinstance(model_b64, str):
            raise ValueError("SentencePiece tokenizer payload must contain 'model_proto_b64'.")
        return SentencePieceTokenizer.from_model_bytes(base64.b64decode(model_b64))
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def load_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)
    if path.suffix == ".model":
        return SentencePieceTokenizer.from_model_bytes(path.read_bytes())
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tokenizer_from_dict(payload)


def prune_encoder_frames_by_blank_probability(
    x: Tensor,
    lengths: Tensor,
    blank_probabilities: Tensor,
    *,
    threshold: float,
    min_keep_frames: int = 1,
    minimum_required_lengths: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    batch_size, max_time, hidden_dim = x.shape
    valid_mask = torch.arange(max_time, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    threshold_keep_mask = (blank_probabilities < threshold) & valid_mask
    pruned_lengths = threshold_keep_mask.sum(dim=1)

    required_keep = lengths.clamp(min=0, max=max(1, min_keep_frames))
    if minimum_required_lengths is not None:
        required_keep = torch.minimum(
            lengths,
            torch.maximum(
                required_keep,
                minimum_required_lengths.to(device=lengths.device, dtype=lengths.dtype),
            ),
        )
    needs_fallback = pruned_lengths < min_keep_frames
    if minimum_required_lengths is not None:
        needs_fallback = pruned_lengths < required_keep
    if needs_fallback.any():
        # Preserve threshold-selected frames when possible; if none survived the threshold,
        # fall back to the lowest-blank frames instead.
        rows_needing_padding = needs_fallback & pruned_lengths.gt(0)
        if rows_needing_padding.any():
            prefix_positions = torch.arange(max_time, device=lengths.device).unsqueeze(0)
            prefix_candidates = valid_mask & ~threshold_keep_mask
            prefix_rank = torch.where(
                prefix_candidates,
                prefix_positions,
                torch.full_like(prefix_positions, max_time),
            )
            prefix_indices = torch.argsort(prefix_rank, dim=1)
            additional_keep = (required_keep - pruned_lengths).clamp_min(0)
            additional_rank_mask = torch.arange(max_time, device=lengths.device).unsqueeze(
                0
            ) < additional_keep.unsqueeze(1)
            padding_keep_mask = torch.zeros_like(valid_mask)
            padding_keep_mask.scatter_(1, prefix_indices, additional_rank_mask)
            threshold_keep_mask = torch.where(
                rows_needing_padding.unsqueeze(1),
                threshold_keep_mask | padding_keep_mask,
                threshold_keep_mask,
            )
        rows_with_no_kept_frames = needs_fallback & pruned_lengths.eq(0)
        if rows_with_no_kept_frames.any():
            masked_blank_probabilities = blank_probabilities.masked_fill(~valid_mask, float("inf"))
            global_topk = min(max_time, max(1, min_keep_frames))
            topk_indices = torch.topk(
                masked_blank_probabilities,
                k=global_topk,
                dim=1,
                largest=False,
            ).indices
            topk_rank_mask = torch.arange(global_topk, device=lengths.device).unsqueeze(
                0
            ) < required_keep.unsqueeze(1)
            fallback_keep_mask = torch.zeros_like(valid_mask)
            fallback_keep_mask.scatter_(1, topk_indices, topk_rank_mask)
            threshold_keep_mask = torch.where(
                rows_with_no_kept_frames.unsqueeze(1),
                fallback_keep_mask,
                threshold_keep_mask,
            )
        pruned_lengths = torch.where(needs_fallback, required_keep, pruned_lengths)

    max_pruned_length = int(pruned_lengths.max().item()) if batch_size > 0 else 0
    if max_pruned_length == 0:
        return x.new_zeros((batch_size, 0, hidden_dim)), pruned_lengths.to(dtype=lengths.dtype)

    time_indices = torch.arange(max_time, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    sort_keys = torch.where(
        threshold_keep_mask,
        time_indices,
        torch.full_like(time_indices, max_time),
    )
    sorted_time_indices = torch.argsort(sort_keys, dim=1)
    pruned_batch = torch.gather(
        x,
        dim=1,
        index=sorted_time_indices[:, :max_pruned_length].unsqueeze(-1).expand(-1, -1, hidden_dim),
    )
    output_mask = torch.arange(max_pruned_length, device=lengths.device).unsqueeze(
        0
    ) < pruned_lengths.unsqueeze(1)
    pruned_batch = pruned_batch * output_mask.unsqueeze(-1).to(dtype=x.dtype)
    return pruned_batch, pruned_lengths.to(dtype=lengths.dtype)


class TrainingOnlyAEDDecoder(nn.Module):
    def __init__(
        self,
        *,
        acoustic_vocab_size: int,
        model_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_target_length: int = 512,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.token_offset = 3
        self.decoder_vocab_size = acoustic_vocab_size + self.token_offset
        self.embedding = nn.Embedding(
            self.decoder_vocab_size,
            model_dim,
            padding_idx=self.pad_id,
        )
        self.position_embedding = nn.Embedding(max_target_length, model_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(model_dim, self.decoder_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.activation_checkpointing = activation_checkpointing
        self.max_target_length = max_target_length
        self.register_buffer(
            "_causal_mask",
            torch.triu(
                torch.ones(max_target_length, max_target_length, dtype=torch.bool), diagonal=1
            ),
            persistent=False,
        )

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        if seq_len > self._causal_mask.size(0):
            raise ValueError(
                f"decoder_inputs length {seq_len} exceeds max_target_length={self.max_target_length}"
            )
        return self._causal_mask[:seq_len, :seq_len].to(device=device)

    def forward(
        self,
        memory: Tensor,
        memory_lengths: Tensor,
        decoder_inputs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        seq_len = decoder_inputs.size(1)
        positions = torch.arange(seq_len, device=decoder_inputs.device)
        x = self.embedding(decoder_inputs) + self.position_embedding(positions).unsqueeze(0)
        x = self.dropout(x)
        causal_mask = self._get_causal_mask(seq_len, decoder_inputs.device)
        target_padding_mask = decoder_inputs.eq(self.pad_id)
        memory_padding_mask = make_padding_mask(memory_lengths, memory.size(1))
        if self.activation_checkpointing and self.training:
            hidden = activation_checkpoint(
                lambda a, b, c, d, e: self.decoder(
                    a,
                    b,
                    tgt_mask=c,
                    tgt_key_padding_mask=d,
                    memory_key_padding_mask=e,
                ),
                x,
                memory,
                causal_mask,
                target_padding_mask,
                memory_padding_mask,
                use_reentrant=False,
            )
        else:
            hidden = self.decoder(
                x,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=target_padding_mask,
                memory_key_padding_mask=memory_padding_mask,
            )
        return self.output_projection(hidden), hidden


class SqueezeformerCTC(nn.Module):
    def __init__(
        self,
        encoder_config: SqueezeformerConfig,
        vocab_size: int,
        intermediate_ctc_layers: tuple[int, ...] = (),
        blank_prune_layer: int | None = None,
        blank_prune_threshold: float = 0.0,
        blank_prune_min_keep_frames: int = 1,
        aed_decoder_enabled: bool = False,
        aed_decoder_layers: int = 1,
        aed_decoder_heads: int = 4,
        aed_decoder_dropout: float = 0.1,
        liberta_distill_enabled: bool = False,
        liberta_hidden_size: int = 1024,
        audio_teacher_enabled: bool = False,
        audio_teacher_hidden_size: int = 1024,
        audio_teacher_target: str = "encoder",
        initial_ctc_blank_bias: float = DEFAULT_INITIAL_CTC_BLANK_BIAS,
        identical_initial_ctc_heads: bool = False,
        blank_logit_offset: float = 0.0,
        blank_logit_regularization_weight: float = 0.0,
        use_transformer_engine: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.intermediate_ctc_layers = tuple(intermediate_ctc_layers)
        self.blank_prune_layer = blank_prune_layer
        self.blank_prune_threshold = blank_prune_threshold
        self.blank_prune_min_keep_frames = blank_prune_min_keep_frames
        self.aed_decoder_enabled = aed_decoder_enabled
        self.liberta_distill_enabled = liberta_distill_enabled
        self.audio_teacher_enabled = audio_teacher_enabled
        self.audio_teacher_target = audio_teacher_target
        self.initial_ctc_blank_bias = float(initial_ctc_blank_bias)
        self.identical_initial_ctc_heads = bool(identical_initial_ctc_heads)
        self.blank_logit_offset = float(blank_logit_offset)
        self.blank_logit_regularization_weight = float(blank_logit_regularization_weight)
        self.use_transformer_engine = use_transformer_engine
        self.encoder = SqueezeformerEncoder(
            encoder_config,
            use_transformer_engine=use_transformer_engine,
        )
        self.classifier = make_linear(
            encoder_config.d_model,
            vocab_size,
            use_transformer_engine=use_transformer_engine,
        )
        ctc_head_layers = set(self.intermediate_ctc_layers)
        if self.blank_prune_layer is not None and self.blank_prune_threshold > 0.0:
            ctc_head_layers.add(self.blank_prune_layer)
        self.intermediate_classifiers = nn.ModuleDict(
            {
                str(layer_index): make_linear(
                    encoder_config.d_model,
                    vocab_size,
                    use_transformer_engine=use_transformer_engine,
                )
                for layer_index in sorted(ctc_head_layers)
            }
        )
        self.aed_decoder = (
            TrainingOnlyAEDDecoder(
                acoustic_vocab_size=vocab_size,
                model_dim=encoder_config.d_model,
                num_layers=aed_decoder_layers,
                num_heads=aed_decoder_heads,
                dropout=aed_decoder_dropout,
                activation_checkpointing=encoder_config.activation_checkpointing,
            )
            if aed_decoder_enabled
            else None
        )
        self.liberta_projection = (
            nn.Linear(encoder_config.d_model, liberta_hidden_size)
            if liberta_distill_enabled
            else None
        )
        self.audio_teacher_projection = (
            nn.Linear(encoder_config.d_model, audio_teacher_hidden_size)
            if audio_teacher_enabled and audio_teacher_target == "encoder"
            else None
        )
        self._initialize_ctc_head(self.classifier, blank_bias=self.initial_ctc_blank_bias)
        for classifier in self.intermediate_classifiers.values():
            self._initialize_ctc_head(classifier, blank_bias=self.initial_ctc_blank_bias)
        if self.identical_initial_ctc_heads:
            self._copy_main_ctc_head_to_intermediates()

    @staticmethod
    def _initialize_ctc_head(classifier: nn.Module, *, blank_bias: float) -> None:
        bias = getattr(classifier, "bias", None)
        if bias is None:
            return
        with torch.no_grad():
            bias.zero_()
            bias[0] = float(blank_bias)

    def _copy_main_ctc_head_to_intermediates(self) -> None:
        if not self.intermediate_classifiers:
            return
        main_state_dict = self.classifier.state_dict()
        for classifier in self.intermediate_classifiers.values():
            classifier.load_state_dict(main_state_dict)

    @staticmethod
    def _ctc_length_diagnostics(
        output_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> dict[str, float]:
        sample_count = max(1, int(output_lengths.numel()))
        minimum_lengths = SqueezeformerCTC._ctc_minimum_alignment_lengths(
            targets,
            target_lengths,
        )
        impossible = int(output_lengths.lt(minimum_lengths).sum().item())
        tight = int(output_lengths.le(minimum_lengths).sum().item())
        return {
            "sample_count": float(sample_count),
            "impossible_sample_count": float(impossible),
            "tight_sample_count": float(tight),
        }

    @staticmethod
    def _ctc_minimum_alignment_lengths(targets: Tensor, target_lengths: Tensor) -> Tensor:
        minimum_lengths = target_lengths.to(dtype=torch.int64).clone()
        offset = 0
        for sample_index, target_length in enumerate(target_lengths.tolist()):
            length = int(target_length)
            if length > 1:
                sample_targets = targets[offset : offset + length]
                minimum_lengths[sample_index] += sample_targets[:-1].eq(sample_targets[1:]).sum()
            offset += length
        return minimum_lengths

    @staticmethod
    def _ctc_batch_diagnostics_from_log_probs(
        log_probs: Tensor,
        output_lengths: Tensor,
        *,
        blank_id: int,
        targets: Tensor | None = None,
        target_lengths: Tensor | None = None,
    ) -> dict[str, float]:
        valid_mask = torch.arange(log_probs.size(1), device=output_lengths.device).unsqueeze(
            0
        ) < output_lengths.unsqueeze(1)
        valid_frames = int(valid_mask.sum().item())
        blank_probabilities = log_probs[..., blank_id].exp()
        argmax_tokens = log_probs.argmax(dim=-1)
        argmax_blank_frames = int(((argmax_tokens == blank_id) & valid_mask).sum().item())

        nonblank_log_probs = log_probs.clone()
        nonblank_log_probs[..., blank_id] = float("-inf")
        top_nonblank_probabilities = nonblank_log_probs.max(dim=-1).values.exp()

        diagnostics = {
            "blank_probability_sum": float(
                blank_probabilities.masked_select(valid_mask).sum().item()
            ),
            "decoded_frames": float(valid_frames),
            "argmax_blank_frames": float(argmax_blank_frames),
            "top_nonblank_probability_sum": float(
                top_nonblank_probabilities.masked_select(valid_mask).sum().item()
            ),
            "sample_count": float(output_lengths.numel()),
            "output_frames_sum": float(output_lengths.sum().item()),
        }
        if target_lengths is not None:
            diagnostics["target_tokens_sum"] = float(target_lengths.sum().item())
        if targets is not None and target_lengths is not None:
            minimum_lengths = SqueezeformerCTC._ctc_minimum_alignment_lengths(
                targets,
                target_lengths,
            )
            diagnostics["impossible_sample_count"] = float(
                output_lengths.lt(minimum_lengths).sum().item()
            )
            diagnostics["tight_sample_count"] = float(
                output_lengths.le(minimum_lengths).sum().item()
            )
        return diagnostics

    @staticmethod
    def _ctc_logit_diagnostics_from_logits(
        logits: Tensor,
        output_lengths: Tensor,
        *,
        blank_id: int,
    ) -> dict[str, float]:
        valid_mask = torch.arange(logits.size(1), device=output_lengths.device).unsqueeze(
            0
        ) < output_lengths.unsqueeze(1)
        valid_frames = int(valid_mask.sum().item())
        if valid_frames <= 0:
            return {
                "decoded_frames": 0.0,
                "blank_logit_sum": 0.0,
                "top_logit_sum": 0.0,
                "top2_margin_sum": 0.0,
                "blank_nonblank_margin_sum": 0.0,
                "entropy_sum": 0.0,
            }

        top2 = torch.topk(logits, k=min(2, logits.size(-1)), dim=-1).values
        top_logits = top2[..., 0]
        top2_margins = (
            top2[..., 0] - top2[..., 1] if logits.size(-1) > 1 else torch.zeros_like(top_logits)
        )
        blank_logits = logits[..., blank_id]
        nonblank_logits = logits.clone()
        nonblank_logits[..., blank_id] = float("-inf")
        best_nonblank_logits = nonblank_logits.max(dim=-1).values
        blank_nonblank_margins = blank_logits - best_nonblank_logits
        probabilities = torch.softmax(logits, dim=-1)
        entropy = -(probabilities * torch.log_softmax(logits, dim=-1)).sum(dim=-1)

        return {
            "decoded_frames": float(valid_frames),
            "blank_logit_sum": float(blank_logits.masked_select(valid_mask).sum().item()),
            "top_logit_sum": float(top_logits.masked_select(valid_mask).sum().item()),
            "top2_margin_sum": float(top2_margins.masked_select(valid_mask).sum().item()),
            "blank_nonblank_margin_sum": float(
                blank_nonblank_margins.masked_select(valid_mask).sum().item()
            ),
            "entropy_sum": float(entropy.masked_select(valid_mask).sum().item()),
        }

    def _apply_training_blank_logit_offset(self, logits: Tensor) -> Tensor:
        if not self.training or self.blank_logit_offset <= 0.0:
            return logits
        adjusted_logits = logits.clone()
        adjusted_logits[..., 0] = adjusted_logits[..., 0] - self.blank_logit_offset
        return adjusted_logits

    @staticmethod
    def _ctc_log_softmax(logits: Tensor) -> Tensor:
        # CTC is numerically sensitive under mixed precision. Compute the
        # normalization in float32 instead of autocast bf16/fp16, mirroring the
        # explicit float32 path already used for AED cross-entropy.
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

    def encode_with_intermediates(
        self,
        features: Tensor,
        feature_lengths: Tensor,
    ) -> tuple[Tensor, Tensor, dict[int, Tensor], dict[int, Tensor]]:
        if not self.intermediate_classifiers and (
            self.blank_prune_layer is None or self.blank_prune_threshold <= 0.0
        ):
            encoded, output_lengths = self.encoder(features, feature_lengths)
            return encoded, output_lengths, {}, {}
        return self.encoder.forward_with_intermediates(
            features,
            feature_lengths,
            intermediate_layer_indices=self.intermediate_ctc_layers,
            post_block_transforms=self._post_block_transforms(),
        )

    def encode_with_online_intermediate_ctc_losses(
        self,
        features: Tensor,
        feature_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
        *,
        blank_id: int,
    ) -> tuple[Tensor, Tensor, dict[int, Tensor], dict[int, dict[str, float]]]:
        intermediate_ctc_losses: dict[int, Tensor] = {}
        intermediate_ctc_diagnostics: dict[int, dict[str, float]] = {}
        if not self.intermediate_classifiers and (
            self.blank_prune_layer is None or self.blank_prune_threshold <= 0.0
        ):
            encoded, output_lengths = self.encoder(features, feature_lengths)
            return encoded, output_lengths, intermediate_ctc_losses, intermediate_ctc_diagnostics

        def accumulate_intermediate_ctc_loss(layer_index: int, x: Tensor, lengths: Tensor) -> None:
            (
                intermediate_ctc_losses[layer_index],
                intermediate_ctc_diagnostics[layer_index],
            ) = self._chunked_ctc_loss_and_diagnostics_from_classifier(
                self.intermediate_classifiers[str(layer_index)],
                x,
                lengths,
                targets,
                target_lengths,
                blank_id=blank_id,
            )

        encoded, output_lengths = self.encoder.forward_with_intermediate_callback(
            features,
            feature_lengths,
            intermediate_layer_indices=self.intermediate_ctc_layers,
            intermediate_layer_callback=accumulate_intermediate_ctc_loss,
            post_block_transforms=self._post_block_transforms(
                minimum_required_lengths=target_lengths,
            ),
        )
        return encoded, output_lengths, intermediate_ctc_losses, intermediate_ctc_diagnostics

    def ctc_log_probs_from_encoded(
        self,
        encoded: Tensor,
        intermediate_encoded: dict[int, Tensor],
    ) -> tuple[Tensor, dict[int, Tensor]]:
        intermediate_log_probs = {
            layer_index: self._chunked_log_probs_from_classifier(
                self.intermediate_classifiers[str(layer_index)],
                intermediate_encoded[layer_index],
            )
            for layer_index in self.intermediate_ctc_layers
        }
        return self._chunked_log_probs_from_classifier(
            self.classifier, encoded
        ), intermediate_log_probs

    def _chunked_log_probs_from_classifier(self, classifier: nn.Module, x: Tensor) -> Tensor:
        _logits, log_probs = self._chunked_logits_and_log_probs_from_classifier(classifier, x)
        return log_probs

    def _chunked_logits_and_log_probs_from_classifier(
        self,
        classifier: nn.Module,
        x: Tensor,
        *,
        apply_blank_offset: bool = False,
    ) -> tuple[Tensor, Tensor]:
        batch, time, _ = x.shape
        vocab_size = getattr(classifier, "out_features", None)
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            logits = apply_linear_with_fp8_padding(classifier, x)
            logits_for_log_probs = (
                self._apply_training_blank_logit_offset(logits) if apply_blank_offset else logits
            )
            return logits, self._ctc_log_softmax(logits_for_log_probs)

        bytes_per_element = max(1, x.element_size())
        target_elements = max(1, _BLANK_PRUNE_TARGET_BYTES // bytes_per_element)
        chunk_batch = max(1, target_elements // max(1, time * vocab_size))
        if chunk_batch >= batch:
            logits = apply_linear_with_fp8_padding(classifier, x)
            logits_for_log_probs = (
                self._apply_training_blank_logit_offset(logits) if apply_blank_offset else logits
            )
            return logits, self._ctc_log_softmax(logits_for_log_probs)

        logits = x.new_empty((batch, time, vocab_size))
        log_probs = x.new_empty((batch, time, vocab_size), dtype=torch.float32)
        start = 0
        for chunk in x.split(chunk_batch, dim=0):
            stop = start + chunk.size(0)
            chunk_logits = apply_linear_with_fp8_padding(classifier, chunk)
            chunk_logits_for_log_probs = (
                self._apply_training_blank_logit_offset(chunk_logits)
                if apply_blank_offset
                else chunk_logits
            )
            logits[start:stop] = chunk_logits
            log_probs[start:stop] = self._ctc_log_softmax(chunk_logits_for_log_probs)
            start = stop
        return logits, log_probs

    def ctc_loss_from_encoded(
        self,
        encoded: Tensor,
        output_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
        *,
        blank_id: int,
        intermediate_encoded: dict[int, Tensor] | None = None,
        intermediate_output_lengths: dict[int, Tensor] | None = None,
    ) -> tuple[Tensor, dict[int, Tensor]]:
        main_ctc_loss = self._chunked_ctc_loss_from_classifier(
            self.classifier,
            encoded,
            output_lengths,
            targets,
            target_lengths,
            blank_id=blank_id,
        )
        intermediate_ctc_losses: dict[int, Tensor] = {}
        if intermediate_encoded is None or intermediate_output_lengths is None:
            return main_ctc_loss, intermediate_ctc_losses
        for layer_index in self.intermediate_ctc_layers:
            if (
                layer_index not in intermediate_encoded
                or layer_index not in intermediate_output_lengths
            ):
                continue
            intermediate_ctc_losses[layer_index] = self._chunked_ctc_loss_from_classifier(
                self.intermediate_classifiers[str(layer_index)],
                intermediate_encoded[layer_index],
                intermediate_output_lengths[layer_index],
                targets,
                target_lengths,
                blank_id=blank_id,
            )
        return main_ctc_loss, intermediate_ctc_losses

    def _chunked_ctc_loss_and_diagnostics_from_classifier(
        self,
        classifier: nn.Module,
        x: Tensor,
        output_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
        *,
        blank_id: int,
    ) -> tuple[Tensor, dict[str, float]]:
        batch, time, _ = x.shape
        vocab_size = getattr(classifier, "out_features", None)

        def summarize(logits: Tensor, log_probs: Tensor) -> dict[str, float]:
            diagnostics = self._ctc_batch_diagnostics_from_log_probs(
                log_probs,
                output_lengths,
                blank_id=blank_id,
                targets=targets,
                target_lengths=target_lengths,
            )
            diagnostics.update(
                self._ctc_logit_diagnostics_from_logits(
                    logits,
                    output_lengths,
                    blank_id=blank_id,
                )
            )
            return diagnostics

        if not isinstance(vocab_size, int) or vocab_size <= 0:
            logits = apply_linear_with_fp8_padding(classifier, x)
            logits_for_log_probs = self._apply_training_blank_logit_offset(logits)
            log_probs = self._ctc_log_softmax(logits_for_log_probs)
            per_sample_losses = F.ctc_loss(
                log_probs.transpose(0, 1),
                targets,
                output_lengths,
                target_lengths,
                blank=blank_id,
                reduction="none",
                zero_infinity=True,
            )
            return (per_sample_losses / target_lengths.clamp_min(1)).mean(), summarize(
                logits,
                log_probs,
            )

        bytes_per_element = max(1, x.element_size())
        target_elements = max(1, _BLANK_PRUNE_TARGET_BYTES // bytes_per_element)
        chunk_batch = max(1, target_elements // max(1, time * vocab_size))
        if chunk_batch >= batch:
            logits = apply_linear_with_fp8_padding(classifier, x)
            logits_for_log_probs = self._apply_training_blank_logit_offset(logits)
            log_probs = self._ctc_log_softmax(logits_for_log_probs)
            per_sample_losses = F.ctc_loss(
                log_probs.transpose(0, 1),
                targets,
                output_lengths,
                target_lengths,
                blank=blank_id,
                reduction="none",
                zero_infinity=True,
            )
            return (per_sample_losses / target_lengths.clamp_min(1)).mean(), summarize(
                logits,
                log_probs,
            )

        target_lengths_list = [int(length) for length in target_lengths.tolist()]
        target_offsets = [0]
        for length in target_lengths_list:
            target_offsets.append(target_offsets[-1] + length)

        weighted_loss_sum = x.new_zeros((), dtype=torch.float32)
        diagnostics: dict[str, float] = {}
        for start in range(0, batch, chunk_batch):
            stop = min(batch, start + chunk_batch)
            chunk_logits = apply_linear_with_fp8_padding(classifier, x[start:stop])
            chunk_logits_for_log_probs = self._apply_training_blank_logit_offset(chunk_logits)
            chunk_log_probs = self._ctc_log_softmax(chunk_logits_for_log_probs)
            chunk_targets = targets[target_offsets[start] : target_offsets[stop]]
            chunk_output_lengths = output_lengths[start:stop]
            chunk_target_lengths = target_lengths[start:stop]
            per_sample_losses = F.ctc_loss(
                chunk_log_probs.transpose(0, 1),
                chunk_targets,
                chunk_output_lengths,
                chunk_target_lengths,
                blank=blank_id,
                reduction="none",
                zero_infinity=True,
            )
            weighted_loss_sum = (
                weighted_loss_sum + (per_sample_losses / chunk_target_lengths.clamp_min(1)).sum()
            )
            chunk_diagnostics = self._ctc_batch_diagnostics_from_log_probs(
                chunk_log_probs,
                chunk_output_lengths,
                blank_id=blank_id,
                targets=chunk_targets,
                target_lengths=chunk_target_lengths,
            )
            chunk_diagnostics.update(
                self._ctc_logit_diagnostics_from_logits(
                    chunk_logits,
                    chunk_output_lengths,
                    blank_id=blank_id,
                )
            )
            for key, value in chunk_diagnostics.items():
                diagnostics[key] = diagnostics.get(key, 0.0) + float(value)
        return weighted_loss_sum / max(1, batch), diagnostics

    def _chunked_ctc_loss_from_classifier(
        self,
        classifier: nn.Module,
        x: Tensor,
        output_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
        *,
        blank_id: int,
    ) -> Tensor:
        batch, time, _ = x.shape
        vocab_size = getattr(classifier, "out_features", None)
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            logits = apply_linear_with_fp8_padding(classifier, x)
            logits_for_log_probs = self._apply_training_blank_logit_offset(logits)
            log_probs = self._ctc_log_softmax(logits_for_log_probs)
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

        bytes_per_element = max(1, x.element_size())
        target_elements = max(1, _BLANK_PRUNE_TARGET_BYTES // bytes_per_element)
        chunk_batch = max(1, target_elements // max(1, time * vocab_size))
        if chunk_batch >= batch:
            logits = apply_linear_with_fp8_padding(classifier, x)
            logits_for_log_probs = self._apply_training_blank_logit_offset(logits)
            log_probs = self._ctc_log_softmax(logits_for_log_probs)
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

        target_lengths_list = [int(length) for length in target_lengths.tolist()]
        target_offsets = [0]
        for length in target_lengths_list:
            target_offsets.append(target_offsets[-1] + length)

        weighted_loss_sum = x.new_zeros((), dtype=torch.float32)
        for start in range(0, batch, chunk_batch):
            stop = min(batch, start + chunk_batch)
            chunk_logits = apply_linear_with_fp8_padding(classifier, x[start:stop])
            chunk_logits_for_log_probs = self._apply_training_blank_logit_offset(chunk_logits)
            chunk_log_probs = self._ctc_log_softmax(chunk_logits_for_log_probs)
            chunk_targets = targets[target_offsets[start] : target_offsets[stop]]
            chunk_output_lengths = output_lengths[start:stop]
            chunk_target_lengths = target_lengths[start:stop]
            per_sample_losses = F.ctc_loss(
                chunk_log_probs.transpose(0, 1),
                chunk_targets,
                chunk_output_lengths,
                chunk_target_lengths,
                blank=blank_id,
                reduction="none",
                zero_infinity=True,
            )
            weighted_loss_sum = (
                weighted_loss_sum + (per_sample_losses / chunk_target_lengths.clamp_min(1)).sum()
            )
        return weighted_loss_sum / max(1, batch)

    def _post_block_transforms(
        self,
        *,
        minimum_required_lengths: Tensor | None = None,
    ) -> dict[int, Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]]:
        post_block_transforms: dict[int, Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]] = {}
        if self.blank_prune_layer is None or self.blank_prune_threshold <= 0.0:
            return post_block_transforms

        prune_layer_key = str(self.blank_prune_layer)
        if prune_layer_key not in self.intermediate_classifiers:
            raise RuntimeError(
                f"Missing CTC head for blank pruning layer {self.blank_prune_layer}."
            )

        prune_classifier = self.intermediate_classifiers[prune_layer_key]

        def blank_prune_transform(x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
            with torch.no_grad():
                blank_probabilities = self._blank_probabilities_for_pruning(x, prune_classifier)
            return prune_encoder_frames_by_blank_probability(
                x,
                lengths,
                blank_probabilities,
                threshold=self.blank_prune_threshold,
                min_keep_frames=self.blank_prune_min_keep_frames,
                minimum_required_lengths=minimum_required_lengths,
            )

        post_block_transforms[self.blank_prune_layer] = blank_prune_transform
        return post_block_transforms

    def _blank_probabilities_for_pruning(self, x: Tensor, classifier: nn.Module) -> Tensor:
        batch, time, _ = x.shape
        vocab_size = getattr(classifier, "out_features", None)
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            logits = apply_linear_with_fp8_padding(classifier, x)
            return (logits[..., 0] - torch.logsumexp(logits, dim=-1)).exp()
        bytes_per_element = max(1, x.element_size())
        target_elements = max(1, _BLANK_PRUNE_TARGET_BYTES // bytes_per_element)
        chunk_batch = max(1, target_elements // max(1, time * vocab_size))
        if chunk_batch >= batch:
            logits = apply_linear_with_fp8_padding(classifier, x)
            return (logits[..., 0] - torch.logsumexp(logits, dim=-1)).exp()
        blank_probability_chunks = []
        for chunk in x.split(chunk_batch, dim=0):
            logits = apply_linear_with_fp8_padding(classifier, chunk)
            blank_probability_chunks.append(
                (logits[..., 0] - torch.logsumexp(logits, dim=-1)).exp()
            )
        return torch.cat(blank_probability_chunks, dim=0)

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
        if return_training_outputs or decoder_inputs is not None:
            if targets is not None and target_lengths is not None and blank_id is not None:
                encoded, output_lengths, intermediate_ctc_losses, intermediate_ctc_diagnostics = (
                    self.encode_with_online_intermediate_ctc_losses(
                        features,
                        feature_lengths,
                        targets,
                        target_lengths,
                        blank_id=blank_id,
                    )
                )
                main_ctc_loss = self._chunked_ctc_loss_from_classifier(
                    self.classifier,
                    encoded,
                    output_lengths,
                    targets,
                    target_lengths,
                    blank_id=blank_id,
                )
            else:
                encoded, output_lengths, intermediate_encoded, intermediate_output_lengths = (
                    self.encode_with_intermediates(features, feature_lengths)
                )
                intermediate_ctc_losses = {}
                intermediate_ctc_diagnostics = {}
                main_ctc_loss = None
            output = TrainingOutputs(
                {
                    "encoded": encoded,
                    "output_lengths": output_lengths,
                    "main_ctc_loss": main_ctc_loss,
                    "blank_logit_regularization_loss": encoded.new_zeros((), dtype=torch.float32),
                    "intermediate_ctc_losses": intermediate_ctc_losses,
                    "intermediate_ctc_diagnostics": intermediate_ctc_diagnostics,
                }
            )
            if (
                targets is not None
                and target_lengths is not None
                and blank_id is not None
                and self.blank_logit_regularization_weight > 0.0
            ):
                blank_reg_logits, _ = self._chunked_logits_and_log_probs_from_classifier(
                    self.classifier,
                    encoded,
                    apply_blank_offset=False,
                )
                output["blank_logit_regularization_loss"] = (
                    self._blank_logit_regularization_from_logits(
                        blank_reg_logits,
                        output_lengths,
                        blank_id=blank_id,
                    )
                )
            if return_main_log_probs:
                main_logits, main_log_probs = self._chunked_logits_and_log_probs_from_classifier(
                    self.classifier,
                    encoded,
                    apply_blank_offset=True,
                )
                output["main_logits"] = main_logits
                output["main_log_probs"] = main_log_probs
            if self.audio_teacher_projection is not None:
                output["audio_teacher_student_states"] = self.project_encoder_for_audio_teacher(
                    encoded,
                    output_lengths,
                )
            if decoder_inputs is not None:
                if self.aed_decoder is None:
                    raise RuntimeError("AED decoder is disabled for this model.")
                aed_logits, aed_hidden = self.aed_decoder(encoded, output_lengths, decoder_inputs)
                output["aed_logits"] = aed_logits
                output["aed_hidden"] = aed_hidden
                if liberta_lengths is not None and self.liberta_projection is not None:
                    output["liberta_student_embeddings"] = self.project_aed_hidden_for_liberta(
                        aed_hidden,
                        liberta_lengths,
                    )
            return output
        if self.blank_prune_layer is not None and self.blank_prune_threshold > 0.0:
            encoded, output_lengths, _, _ = self.encoder.forward_with_intermediates(
                features,
                feature_lengths,
                intermediate_layer_indices=(),
                post_block_transforms=self._post_block_transforms(),
            )
        else:
            encoded, output_lengths = self.encoder(features, feature_lengths)
        logits = apply_linear_with_fp8_padding(self.classifier, encoded)
        return logits, output_lengths

    def log_probs(self, features: Tensor, feature_lengths: Tensor) -> tuple[Tensor, Tensor]:
        logits, output_lengths = self(features, feature_lengths)
        return self._ctc_log_softmax(logits), output_lengths

    def log_probs_with_intermediate(
        self,
        features: Tensor,
        feature_lengths: Tensor,
    ) -> tuple[Tensor, Tensor, dict[int, Tensor], dict[int, Tensor]]:
        encoded, output_lengths, intermediate_encoded, intermediate_lengths = (
            self.encode_with_intermediates(features, feature_lengths)
        )
        log_probs, intermediate_log_probs = self.ctc_log_probs_from_encoded(
            encoded,
            intermediate_encoded,
        )
        return (
            log_probs,
            output_lengths,
            intermediate_log_probs,
            intermediate_lengths,
        )

    def to_config_dict(self) -> dict[str, object]:
        return asdict(self.encoder_config)

    def aed_forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
        decoder_inputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.aed_decoder is None:
            raise RuntimeError("AED decoder is disabled for this model.")
        encoded, output_lengths, _, _ = self.encode_with_intermediates(features, feature_lengths)
        logits, hidden = self.aed_decoder(encoded, output_lengths, decoder_inputs)
        return logits, output_lengths, hidden

    def aed_from_encoded(
        self,
        encoded: Tensor,
        output_lengths: Tensor,
        decoder_inputs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.aed_decoder is None:
            raise RuntimeError("AED decoder is disabled for this model.")
        logits, hidden = self.aed_decoder(encoded, output_lengths, decoder_inputs)
        return logits, hidden

    def project_aed_hidden_for_liberta(self, hidden: Tensor, lengths: Tensor) -> Tensor:
        if self.liberta_projection is None:
            raise RuntimeError("LiBERTa distillation head is disabled for this model.")
        pooled = mean_pool_sequence(hidden, lengths)
        return self.liberta_projection(pooled)

    def project_encoder_for_audio_teacher(self, hidden: Tensor, lengths: Tensor) -> Tensor:
        if self.audio_teacher_projection is None:
            raise RuntimeError("Audio teacher projection head is disabled for this model.")
        pooled = mean_pool_sequence(hidden, lengths)
        return self.audio_teacher_projection(pooled)

    def load_state_dict(self, state_dict: dict[str, object], strict: bool = True):
        # Backward-compatibility for checkpoints created with a single
        # `intermediate_classifier.*` head before multi-head support landed.
        remapped_state_dict = dict(state_dict)
        if (
            "intermediate_classifier.weight" in remapped_state_dict
            and len(self.intermediate_ctc_layers) == 1
            and f"intermediate_classifiers.{self.intermediate_ctc_layers[0]}.weight"
            not in remapped_state_dict
        ):
            layer_key = str(self.intermediate_ctc_layers[0])
            for suffix in ("weight", "bias"):
                old_key = f"intermediate_classifier.{suffix}"
                if old_key in remapped_state_dict:
                    remapped_state_dict[f"intermediate_classifiers.{layer_key}.{suffix}"] = (
                        remapped_state_dict.pop(old_key)
                    )
        if self.aed_decoder is None:
            remapped_state_dict = {
                key: value
                for key, value in remapped_state_dict.items()
                if not key.startswith("aed_decoder.")
            }
        if self.liberta_projection is None:
            remapped_state_dict = {
                key: value
                for key, value in remapped_state_dict.items()
                if not key.startswith("liberta_projection.")
            }
        if self.audio_teacher_projection is None:
            remapped_state_dict = {
                key: value
                for key, value in remapped_state_dict.items()
                if not key.startswith("audio_teacher_projection.")
            }
        return super().load_state_dict(remapped_state_dict, strict=strict)


def load_lm_scorer(spec: str | None) -> Callable[[str], float] | None:
    if spec is None:
        return None
    module_name, function_name, *factory_args = spec.split(":", maxsplit=2)
    module = importlib.import_module(module_name)
    scorer = getattr(module, function_name)
    if not callable(scorer):
        raise TypeError(f"LM scorer '{spec}' is not callable.")
    if factory_args:
        scorer = scorer(factory_args[0])
        if not callable(scorer):
            raise TypeError(f"LM scorer factory '{spec}' did not return a callable scorer.")
    return scorer


def _logaddexp(left: float, right: float) -> float:
    if left == float("-inf"):
        return right
    if right == float("-inf"):
        return left
    if left > right:
        return left + math.log1p(math.exp(right - left))
    return right + math.log1p(math.exp(left - right))


def _beam_total_score(
    blank_score: float,
    nonblank_score: float,
    *,
    prefix_length: int,
    length_bonus: float,
) -> float:
    return _logaddexp(blank_score, nonblank_score) + (length_bonus * prefix_length)


def ctc_prefix_beam_search(
    log_probs: Tensor,
    tokenizer: Tokenizer,
    beam_size: int = 8,
    lm_scorer: Callable[[str], float] | None = None,
    lm_weight: float = 0.0,
    length_bonus: float = DEFAULT_CTC_BEAM_LENGTH_BONUS,
) -> str:
    blank_id = tokenizer.blank_id
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, float("-inf"))}
    decoded_prefix_cache: dict[tuple[int, ...], str] = {(): ""}
    lm_bonus_cache: dict[tuple[int, ...], float] = {}

    for timestep in range(log_probs.size(0)):
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}
        top_values, top_indices = torch.topk(
            log_probs[timestep], k=min(beam_size * 4, log_probs.size(-1))
        )
        beam_items = [
            (
                prefix,
                blank_score,
                nonblank_score,
                _logaddexp(blank_score, nonblank_score),
                prefix[-1] if prefix else None,
            )
            for prefix, (blank_score, nonblank_score) in beams.items()
        ]
        top_scores = top_values.tolist()
        top_token_ids = top_indices.tolist()
        for score, token_id in zip(top_scores, top_token_ids, strict=True):
            for prefix, blank_score, nonblank_score, total_prefix_score, end_token in beam_items:
                existing_blank, existing_nonblank = next_beams.get(
                    prefix, (float("-inf"), float("-inf"))
                )
                if token_id == blank_id:
                    next_beams[prefix] = (
                        _logaddexp(existing_blank, total_prefix_score + score),
                        existing_nonblank,
                    )
                    continue

                new_prefix = prefix + (token_id,)
                lm_bonus = 0.0
                if lm_scorer is not None:
                    cached_bonus = lm_bonus_cache.get(new_prefix)
                    if cached_bonus is None:
                        decoded_prefix = decoded_prefix_cache.get(new_prefix)
                        if decoded_prefix is None:
                            decoded_prefix = tokenizer.decode(new_prefix)
                            decoded_prefix_cache[new_prefix] = decoded_prefix
                        cached_bonus = lm_weight * float(lm_scorer(decoded_prefix))
                        lm_bonus_cache[new_prefix] = cached_bonus
                    lm_bonus = cached_bonus

                if token_id == end_token:
                    same_blank, same_nonblank = next_beams.get(
                        prefix, (float("-inf"), float("-inf"))
                    )
                    next_beams[prefix] = (
                        same_blank,
                        _logaddexp(same_nonblank, nonblank_score + score),
                    )
                    new_blank, new_nonblank = next_beams.get(
                        new_prefix, (float("-inf"), float("-inf"))
                    )
                    next_beams[new_prefix] = (
                        new_blank,
                        _logaddexp(new_nonblank, blank_score + score + lm_bonus),
                    )
                else:
                    new_blank, new_nonblank = next_beams.get(
                        new_prefix, (float("-inf"), float("-inf"))
                    )
                    next_beams[new_prefix] = (
                        new_blank,
                        _logaddexp(new_nonblank, total_prefix_score + score + lm_bonus),
                    )

        ranked = sorted(
            next_beams.items(),
            key=lambda item: _beam_total_score(
                item[1][0],
                item[1][1],
                prefix_length=len(item[0]),
                length_bonus=length_bonus,
            ),
            reverse=True,
        )[:beam_size]
        beams = dict(ranked)

    best_prefix = max(
        beams.items(),
        key=lambda item: _beam_total_score(
            item[1][0],
            item[1][1],
            prefix_length=len(item[0]),
            length_bonus=length_bonus,
        ),
    )[0]
    return tokenizer.decode(best_prefix)
