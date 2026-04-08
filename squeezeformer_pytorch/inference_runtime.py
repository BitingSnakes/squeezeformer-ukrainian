from __future__ import annotations

import re
from typing import Any


def resolve_inference_checkpoint_settings(checkpoint_data: dict[str, Any]) -> dict[str, Any]:
    training_args = checkpoint_data.get("training_args", {})
    intermediate_ctc_weight = float(training_args.get("intermediate_ctc_weight", 0.0))
    intermediate_ctc_layers = training_args.get("intermediate_ctc_layers")
    intermediate_ctc_layer = training_args.get("intermediate_ctc_layer")

    if intermediate_ctc_weight > 0.0:
        if intermediate_ctc_layers is not None:
            resolved_intermediate_ctc_layers = tuple(int(layer) for layer in intermediate_ctc_layers)
        elif intermediate_ctc_layer is not None:
            resolved_intermediate_ctc_layers = (int(intermediate_ctc_layer),)
        else:
            resolved_intermediate_ctc_layers = ()
    else:
        resolved_intermediate_ctc_layers = ()

    blank_prune_threshold = float(training_args.get("blank_prune_threshold", 0.0))
    blank_prune_layer = training_args.get("blank_prune_layer")

    return {
        "resolved_intermediate_ctc_layers": resolved_intermediate_ctc_layers,
        "blank_prune_threshold": blank_prune_threshold,
        "blank_prune_layer": (
            int(blank_prune_layer)
            if blank_prune_threshold > 0.0 and blank_prune_layer is not None
            else None
        ),
        "blank_prune_min_keep_frames": int(training_args.get("blank_prune_min_keep_frames", 1)),
        "aed_decoder_enabled": bool(training_args.get("aed_decoder", False)),
        "aed_decoder_layers": int(training_args.get("aed_decoder_layers", 1)),
        "aed_decoder_heads": int(training_args.get("aed_decoder_heads", 4)),
        "aed_decoder_dropout": float(training_args.get("aed_decoder_dropout", 0.1)),
        "liberta_distill_enabled": bool(training_args.get("liberta_distill", False)),
    }


def normalize_transcript_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def join_transcripts(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left.endswith((" ", "\n")) or right.startswith((" ", "\n", ",", ".", "!", "?", ":", ";")):
        return f"{left}{right}"
    return f"{left} {right}"


def merge_chunk_transcript(existing: str, new: str) -> str:
    existing = normalize_transcript_whitespace(existing)
    new = normalize_transcript_whitespace(new)
    if not existing:
        return new
    if not new:
        return existing
    if new in existing:
        return existing
    if existing in new:
        return new

    existing_words = existing.split(" ")
    new_words = new.split(" ")
    max_word_overlap = min(len(existing_words), len(new_words), 20)
    for overlap in range(max_word_overlap, 0, -1):
        if existing_words[-overlap:] == new_words[:overlap]:
            return " ".join(existing_words + new_words[overlap:])

    max_char_overlap = min(len(existing), len(new), 120)
    for overlap in range(max_char_overlap, 8, -1):
        if existing[-overlap:] == new[:overlap]:
            return f"{existing}{new[overlap:]}"

    return join_transcripts(existing, new)
