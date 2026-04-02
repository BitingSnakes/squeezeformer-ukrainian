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

from .model import (
    SqueezeformerConfig,
    SqueezeformerEncoder,
    apply_linear_with_fp8_padding,
    make_linear,
)


def _lengths_to_padding_mask(lengths: Tensor, max_length: int) -> Tensor:
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) >= lengths.unsqueeze(1)


def mean_pool_sequence(x: Tensor, lengths: Tensor) -> Tensor:
    mask = ~_lengths_to_padding_mask(lengths, x.size(1))
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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tokenizer_from_dict(payload)


def prune_encoder_frames_by_blank_probability(
    x: Tensor,
    lengths: Tensor,
    blank_probabilities: Tensor,
    *,
    threshold: float,
    min_keep_frames: int = 1,
) -> tuple[Tensor, Tensor]:
    batch_size, _, hidden_dim = x.shape
    pruned_sequences: list[Tensor] = []
    pruned_lengths: list[int] = []

    for batch_index in range(batch_size):
        valid_length = int(lengths[batch_index].item())
        valid_states = x[batch_index, :valid_length]
        valid_blank_probs = blank_probabilities[batch_index, :valid_length]
        keep_mask = valid_blank_probs < threshold
        keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)

        if keep_indices.numel() < min_keep_frames:
            topk = min(valid_length, max(1, min_keep_frames))
            keep_indices = torch.topk(
                -valid_blank_probs,
                k=topk,
                largest=True,
            ).indices.sort().values

        pruned_sequences.append(valid_states.index_select(0, keep_indices))
        pruned_lengths.append(int(keep_indices.numel()))

    max_pruned_length = max(pruned_lengths)
    pruned_batch = x.new_zeros((batch_size, max_pruned_length, hidden_dim))
    for batch_index, sequence in enumerate(pruned_sequences):
        pruned_batch[batch_index, : sequence.size(0)] = sequence
    return pruned_batch, lengths.new_tensor(pruned_lengths)


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
        memory_padding_mask = _lengths_to_padding_mask(memory_lengths, memory.size(1))
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

    def _post_block_transforms(self) -> dict[int, Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]]:
        post_block_transforms: dict[int, Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]] = {}
        if self.blank_prune_layer is None or self.blank_prune_threshold <= 0.0:
            return post_block_transforms

        prune_layer_key = str(self.blank_prune_layer)
        if prune_layer_key not in self.intermediate_classifiers:
            raise RuntimeError(f"Missing CTC head for blank pruning layer {self.blank_prune_layer}.")

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
        if (
            not self.intermediate_classifiers
            and (self.blank_prune_layer is None or self.blank_prune_threshold <= 0.0)
        ):
            log_probs, output_lengths = self.log_probs(features, feature_lengths)
            return log_probs, output_lengths, {}, {}

        encoded, output_lengths, intermediate_encoded, intermediate_lengths = (
            self.encoder.forward_with_intermediates(
                features,
                feature_lengths,
                intermediate_layer_indices=self.intermediate_ctc_layers,
                post_block_transforms=self._post_block_transforms(),
            )
        )
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
        return (
            F.log_softmax(logits, dim=-1),
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
        if self.blank_prune_layer is not None and self.blank_prune_threshold > 0.0:
            encoded, output_lengths, _, _ = self.encoder.forward_with_intermediates(
                features,
                feature_lengths,
                intermediate_layer_indices=(),
                post_block_transforms=self._post_block_transforms(),
            )
        else:
            encoded, output_lengths = self.encoder(features, feature_lengths)
        logits, hidden = self.aed_decoder(encoded, output_lengths, decoder_inputs)
        return logits, output_lengths, hidden

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
