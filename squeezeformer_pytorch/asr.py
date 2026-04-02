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


class SqueezeformerCTC(nn.Module):
    def __init__(
        self,
        encoder_config: SqueezeformerConfig,
        vocab_size: int,
        intermediate_ctc_layer: int | None = None,
        use_transformer_engine: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.intermediate_ctc_layer = intermediate_ctc_layer
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
        self.intermediate_classifier = (
            make_linear(
                encoder_config.d_model,
                vocab_size,
                use_transformer_engine=use_transformer_engine,
            )
            if intermediate_ctc_layer is not None
            else None
        )

    def forward(self, features: Tensor, feature_lengths: Tensor) -> tuple[Tensor, Tensor]:
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
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if self.intermediate_classifier is None or self.intermediate_ctc_layer is None:
            log_probs, output_lengths = self.log_probs(features, feature_lengths)
            return log_probs, output_lengths, None, None

        encoded, output_lengths, intermediate_encoded, intermediate_lengths = (
            self.encoder.forward_with_intermediate(
                features,
                feature_lengths,
                intermediate_layer_index=self.intermediate_ctc_layer,
            )
        )
        logits = apply_linear_with_fp8_padding(self.classifier, encoded)
        intermediate_logits = apply_linear_with_fp8_padding(
            self.intermediate_classifier,
            intermediate_encoded,
        )
        return (
            F.log_softmax(logits, dim=-1),
            output_lengths,
            F.log_softmax(intermediate_logits, dim=-1),
            intermediate_lengths,
        )

    def to_config_dict(self) -> dict[str, object]:
        return asdict(self.encoder_config)


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
