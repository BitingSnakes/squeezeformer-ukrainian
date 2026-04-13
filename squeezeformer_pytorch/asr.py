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

_CTC_HEAD_TARGET_BYTES = 128 * 1024 * 1024
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
        aed_decoder_enabled: bool = False,
        aed_decoder_layers: int = 1,
        aed_decoder_heads: int = 4,
        aed_decoder_dropout: float = 0.1,
        liberta_distill_enabled: bool = False,
        liberta_hidden_size: int = 1024,
        audio_teacher_enabled: bool = False,
        audio_teacher_hidden_size: int = 1024,
        audio_teacher_target: str = "encoder",
        use_transformer_engine: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.aed_decoder_enabled = aed_decoder_enabled
        self.liberta_distill_enabled = liberta_distill_enabled
        self.audio_teacher_enabled = audio_teacher_enabled
        self.audio_teacher_target = audio_teacher_target
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
        self._initialize_ctc_head(self.classifier)

    @staticmethod
    def _initialize_ctc_head(classifier: nn.Module) -> None:
        bias = getattr(classifier, "bias", None)
        if bias is None:
            return
        with torch.no_grad():
            bias.zero_()

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

    @staticmethod
    def _ctc_log_softmax(logits: Tensor) -> Tensor:
        # CTC is numerically sensitive under mixed precision. Compute the
        # normalization in float32 instead of autocast bf16/fp16, mirroring the
        # explicit float32 path already used for AED cross-entropy.
        return F.log_softmax(logits, dim=-1, dtype=torch.float32)

    def encode_with_intermediates(
        self,
        features: Tensor,
        feature_lengths: Tensor,
    ) -> tuple[Tensor, Tensor, dict[int, Tensor], dict[int, Tensor]]:
        encoded, output_lengths = self.encoder(features, feature_lengths)
        return encoded, output_lengths, {}, {}

    def _chunked_log_probs_from_classifier(self, classifier: nn.Module, x: Tensor) -> Tensor:
        _logits, log_probs = self._chunked_logits_and_log_probs_from_classifier(classifier, x)
        return log_probs

    def _chunked_logits_and_log_probs_from_classifier(
        self,
        classifier: nn.Module,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, time, _ = x.shape
        vocab_size = getattr(classifier, "out_features", None)
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            logits = apply_linear_with_fp8_padding(classifier, x)
            return logits, self._ctc_log_softmax(logits)

        bytes_per_element = max(1, x.element_size())
        target_elements = max(1, _CTC_HEAD_TARGET_BYTES // bytes_per_element)
        chunk_batch = max(1, target_elements // max(1, time * vocab_size))
        if chunk_batch >= batch:
            logits = apply_linear_with_fp8_padding(classifier, x)
            return logits, self._ctc_log_softmax(logits)

        logits = x.new_empty((batch, time, vocab_size))
        log_probs = x.new_empty((batch, time, vocab_size), dtype=torch.float32)
        start = 0
        for chunk in x.split(chunk_batch, dim=0):
            stop = start + chunk.size(0)
            chunk_logits = apply_linear_with_fp8_padding(classifier, chunk)
            logits[start:stop] = chunk_logits
            log_probs[start:stop] = self._ctc_log_softmax(chunk_logits)
            start = stop
        return logits, log_probs

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
            log_probs = self._ctc_log_softmax(logits)
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
        target_elements = max(1, _CTC_HEAD_TARGET_BYTES // bytes_per_element)
        chunk_batch = max(1, target_elements // max(1, time * vocab_size))
        if chunk_batch >= batch:
            logits = apply_linear_with_fp8_padding(classifier, x)
            log_probs = self._ctc_log_softmax(logits)
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
            chunk_log_probs = self._ctc_log_softmax(chunk_logits)
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
            encoded, output_lengths, _, _ = self.encode_with_intermediates(
                features,
                feature_lengths,
            )
            if targets is not None and target_lengths is not None and blank_id is not None:
                main_ctc_loss = self._chunked_ctc_loss_from_classifier(
                    self.classifier,
                    encoded,
                    output_lengths,
                    targets,
                    target_lengths,
                    blank_id=blank_id,
                )
            else:
                main_ctc_loss = None
            output = TrainingOutputs(
                {
                    "encoded": encoded,
                    "output_lengths": output_lengths,
                    "main_ctc_loss": main_ctc_loss,
                }
            )
            if return_main_log_probs:
                main_logits, main_log_probs = self._chunked_logits_and_log_probs_from_classifier(
                    self.classifier,
                    encoded,
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
        encoded, output_lengths = self.encoder(features, feature_lengths)
        logits = apply_linear_with_fp8_padding(self.classifier, encoded)
        return logits, output_lengths

    def log_probs(self, features: Tensor, feature_lengths: Tensor) -> tuple[Tensor, Tensor]:
        logits, output_lengths = self(features, feature_lengths)
        return self._ctc_log_softmax(logits), output_lengths

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
        remapped_state_dict = dict(state_dict)
        remapped_state_dict = {
            key: value
            for key, value in remapped_state_dict.items()
            if not key.startswith("intermediate_classifier.")
            and not key.startswith("intermediate_classifiers.")
        }
        if getattr(self, "aed_decoder", None) is None:
            remapped_state_dict = {
                key: value
                for key, value in remapped_state_dict.items()
                if not key.startswith("aed_decoder.")
            }
        if getattr(self, "liberta_projection", None) is None:
            remapped_state_dict = {
                key: value
                for key, value in remapped_state_dict.items()
                if not key.startswith("liberta_projection.")
            }
        if getattr(self, "audio_teacher_projection", None) is None:
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
