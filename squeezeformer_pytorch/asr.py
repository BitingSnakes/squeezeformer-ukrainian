from __future__ import annotations

import base64
import importlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import sentencepiece as spm
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .masking import make_padding_mask, make_sequence_mask
from .model import (
    SqueezeformerConfig,
    SqueezeformerEncoder,
    apply_linear_with_fp8_padding,
    make_linear,
)


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
        return [self.token_to_id[char] for char in text if char in self.token_to_id]

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
) -> tuple[Tensor, Tensor]:
    batch_size, max_time, hidden_dim = x.shape
    valid_mask = torch.arange(max_time, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    threshold_keep_mask = (blank_probabilities < threshold) & valid_mask
    pruned_lengths = threshold_keep_mask.sum(dim=1)

    required_keep = lengths.clamp(min=0, max=max(1, min_keep_frames))
    needs_fallback = pruned_lengths < min_keep_frames
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
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=decoder_inputs.device, dtype=torch.bool),
            diagonal=1,
        )
        target_padding_mask = decoder_inputs.eq(self.pad_id)
        memory_padding_mask = make_padding_mask(memory_lengths, memory.size(1))
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
            )
            if aed_decoder_enabled
            else None
        )
        self.liberta_projection = (
            nn.Linear(encoder_config.d_model, liberta_hidden_size)
            if liberta_distill_enabled
            else None
        )

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

    def ctc_log_probs_from_encoded(
        self,
        encoded: Tensor,
        intermediate_encoded: dict[int, Tensor],
    ) -> tuple[Tensor, dict[int, Tensor]]:
        logits = apply_linear_with_fp8_padding(self.classifier, encoded)
        intermediate_log_probs = {
            layer_index: F.log_softmax(
                apply_linear_with_fp8_padding(
                    self.intermediate_classifiers[str(layer_index)],
                    intermediate_encoded[layer_index],
                ),
                dim=-1,
            )
            for layer_index in self.intermediate_ctc_layers
        }
        return F.log_softmax(logits, dim=-1), intermediate_log_probs

    def _post_block_transforms(
        self,
    ) -> dict[int, Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]]:
        post_block_transforms: dict[int, Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]] = {}
        if self.blank_prune_layer is None or self.blank_prune_threshold <= 0.0:
            return post_block_transforms

        prune_layer_key = str(self.blank_prune_layer)
        if prune_layer_key not in self.intermediate_classifiers:
            raise RuntimeError(
                f"Missing CTC head for blank pruning layer {self.blank_prune_layer}."
            )

        def blank_prune_transform(x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
            prune_logits = apply_linear_with_fp8_padding(
                self.intermediate_classifiers[prune_layer_key],
                x,
            )
            prune_log_probs = F.log_softmax(prune_logits, dim=-1)
            blank_probabilities = prune_log_probs[..., 0].exp()
            return prune_encoder_frames_by_blank_probability(
                x,
                lengths,
                blank_probabilities,
                threshold=self.blank_prune_threshold,
                min_keep_frames=self.blank_prune_min_keep_frames,
            )

        post_block_transforms[self.blank_prune_layer] = blank_prune_transform
        return post_block_transforms

    def forward(self, features: Tensor, feature_lengths: Tensor) -> tuple[Tensor, Tensor]:
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
        return F.log_softmax(logits, dim=-1), output_lengths

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


def ctc_prefix_beam_search(
    log_probs: Tensor,
    tokenizer: Tokenizer,
    beam_size: int = 8,
    lm_scorer: Callable[[str], float] | None = None,
    lm_weight: float = 0.0,
) -> str:
    blank_id = tokenizer.blank_id
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, float("-inf"))}

    for timestep in range(log_probs.size(0)):
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}
        top_values, top_indices = torch.topk(
            log_probs[timestep], k=min(beam_size * 4, log_probs.size(-1))
        )
        for score, token_id in zip(top_values.tolist(), top_indices.tolist(), strict=True):
            for prefix, (blank_score, nonblank_score) in beams.items():
                total_prefix_score = torch.logsumexp(
                    torch.tensor([blank_score, nonblank_score]),
                    dim=0,
                ).item()
                existing_blank, existing_nonblank = next_beams.get(
                    prefix, (float("-inf"), float("-inf"))
                )
                if token_id == blank_id:
                    next_beams[prefix] = (
                        torch.logsumexp(
                            torch.tensor([existing_blank, total_prefix_score + score]), dim=0
                        ).item(),
                        existing_nonblank,
                    )
                    continue

                end_token = prefix[-1] if prefix else None
                new_prefix = prefix + (token_id,)
                lm_bonus = 0.0
                if lm_scorer is not None:
                    lm_bonus = lm_weight * float(lm_scorer(tokenizer.decode(new_prefix)))

                if token_id == end_token:
                    same_blank, same_nonblank = next_beams.get(
                        prefix, (float("-inf"), float("-inf"))
                    )
                    next_beams[prefix] = (
                        same_blank,
                        torch.logsumexp(
                            torch.tensor([same_nonblank, nonblank_score + score]), dim=0
                        ).item(),
                    )
                    new_blank, new_nonblank = next_beams.get(
                        new_prefix, (float("-inf"), float("-inf"))
                    )
                    next_beams[new_prefix] = (
                        new_blank,
                        torch.logsumexp(
                            torch.tensor([new_nonblank, blank_score + score + lm_bonus]), dim=0
                        ).item(),
                    )
                else:
                    new_blank, new_nonblank = next_beams.get(
                        new_prefix, (float("-inf"), float("-inf"))
                    )
                    next_beams[new_prefix] = (
                        new_blank,
                        torch.logsumexp(
                            torch.tensor([new_nonblank, total_prefix_score + score + lm_bonus]),
                            dim=0,
                        ).item(),
                    )

        ranked = sorted(
            next_beams.items(),
            key=lambda item: torch.logsumexp(torch.tensor(item[1]), dim=0).item(),
            reverse=True,
        )[:beam_size]
        beams = dict(ranked)

    best_prefix = max(
        beams.items(),
        key=lambda item: torch.logsumexp(torch.tensor(item[1]), dim=0).item(),
    )[0]
    return tokenizer.decode(best_prefix)
