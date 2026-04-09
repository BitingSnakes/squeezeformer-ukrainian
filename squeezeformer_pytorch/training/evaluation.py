from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import trackio
from torch import nn
from torch.nn import functional as F

from squeezeformer_pytorch.asr import SqueezeformerCTC, Tokenizer, ctc_prefix_beam_search
from squeezeformer_pytorch.checkpoints import save_checkpoint
from squeezeformer_pytorch.data import AudioFeaturizer
from squeezeformer_pytorch.metrics import char_error_rate, word_error_rate
from squeezeformer_pytorch.model import SqueezeformerConfig
from squeezeformer_pytorch.runtime_types import DecodeStrategy, DTypeChoice, ValidationModelSource
from squeezeformer_pytorch.training.runtime import (
    ExponentialMovingAverage,
    FrozenAudioTeacher,
    FrozenLibertaTeacher,
    _aed_cross_entropy_loss,
    _autocast_context,
    _average_topk_checkpoints,
    _build_aed_targets,
    _build_trackio_grouped_metrics,
    _export_inference_checkpoint,
    _safetensors_path,
    _update_top_checkpoints,
)

logger = logging.getLogger("train")


def _clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _load_cloned_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    module.load_state_dict(
        {
            key: value.to(device=tensor.device, dtype=tensor.dtype)
            for key, tensor in module.state_dict().items()
            for value in (state_dict[key],)
        },
        strict=True,
    )


def _token_display_text(tokenizer: Tokenizer, token_id: int) -> str:
    if token_id == tokenizer.blank_id:
        return "<blank>"

    decoded = tokenizer.decode([token_id]).replace("\n", "\\n")
    if decoded:
        return decoded

    processor = getattr(tokenizer, "processor", None)
    if processor is not None:
        id_to_piece = getattr(processor, "id_to_piece", None)
        if callable(id_to_piece):
            try:
                piece = str(id_to_piece(token_id)).replace("\n", "\\n")
            except Exception:
                piece = ""
            if piece:
                return piece

    return f"<id:{token_id}>"


def greedy_decode(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    tokenizer: Tokenizer,
) -> list[str]:
    token_ids = log_probs.argmax(dim=-1).cpu().tolist()
    lengths = output_lengths.cpu().tolist()
    return [
        tokenizer.decode_ctc(sequence[: int(length)])
        for sequence, length in zip(token_ids, lengths, strict=True)
    ]


def decode_batch(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    tokenizer: Tokenizer,
    strategy: DecodeStrategy,
    beam_size: int = 8,
    lm_scorer=None,
    lm_weight: float = 0.0,
    beam_length_bonus: float = 0.1,
) -> list[str]:
    if strategy == DecodeStrategy.GREEDY:
        return greedy_decode(log_probs, output_lengths, tokenizer)
    return [
        ctc_prefix_beam_search(
            sequence[: int(length)].cpu(),
            tokenizer=tokenizer,
            beam_size=beam_size,
            lm_scorer=lm_scorer,
            lm_weight=lm_weight,
            length_bonus=beam_length_bonus,
        )
        for sequence, length in zip(log_probs, output_lengths.cpu().tolist(), strict=True)
    ]


def ctc_batch_diagnostics(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    tokenizer: Tokenizer,
    *,
    target_lengths: torch.Tensor | None = None,
) -> dict[str, float]:
    valid_mask = torch.arange(log_probs.size(1), device=output_lengths.device).unsqueeze(
        0
    ) < output_lengths.unsqueeze(1)
    valid_frames = int(valid_mask.sum().item())
    blank_probabilities = log_probs[..., tokenizer.blank_id].exp()
    argmax_tokens = log_probs.argmax(dim=-1)
    argmax_blank_frames = int(((argmax_tokens == tokenizer.blank_id) & valid_mask).sum().item())

    nonblank_log_probs = log_probs.clone()
    nonblank_log_probs[..., tokenizer.blank_id] = float("-inf")
    top_nonblank_probabilities = nonblank_log_probs.max(dim=-1).values.exp()

    diagnostics = {
        "blank_probability_sum": float(blank_probabilities.masked_select(valid_mask).sum().item()),
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
        diagnostics["impossible_sample_count"] = float(
            output_lengths.lt(target_lengths).sum().item()
        )
        diagnostics["tight_sample_count"] = float(output_lengths.le(target_lengths).sum().item())
    return diagnostics


def ctc_logit_diagnostics(
    logits: torch.Tensor,
    output_lengths: torch.Tensor,
    tokenizer: Tokenizer,
) -> dict[str, float]:
    valid_mask = torch.arange(logits.size(1), device=output_lengths.device).unsqueeze(0) < output_lengths.unsqueeze(1)
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
    top2_margins = top2[..., 0] - top2[..., 1] if logits.size(-1) > 1 else torch.zeros_like(top_logits)
    blank_logits = logits[..., tokenizer.blank_id]
    nonblank_logits = logits.clone()
    nonblank_logits[..., tokenizer.blank_id] = float("-inf")
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


def summarize_ctc_logit_diagnostics(diagnostics: dict[str, float]) -> dict[str, float]:
    decoded_frames = max(1.0, float(diagnostics.get("decoded_frames", 0.0)))
    return {
        "avg_blank_logit": float(diagnostics.get("blank_logit_sum", 0.0)) / decoded_frames,
        "avg_top_logit": float(diagnostics.get("top_logit_sum", 0.0)) / decoded_frames,
        "avg_top2_margin": float(diagnostics.get("top2_margin_sum", 0.0)) / decoded_frames,
        "avg_blank_nonblank_margin": float(
            diagnostics.get("blank_nonblank_margin_sum", 0.0)
        )
        / decoded_frames,
        "avg_entropy": float(diagnostics.get("entropy_sum", 0.0)) / decoded_frames,
    }


def encoder_output_diagnostics(
    encoded: torch.Tensor,
    output_lengths: torch.Tensor,
) -> dict[str, float]:
    valid_mask = (
        torch.arange(encoded.size(1), device=output_lengths.device).unsqueeze(0)
        < output_lengths.unsqueeze(1)
    )
    valid_frames = int(valid_mask.sum().item())
    if valid_frames <= 0:
        return {
            "decoded_frames": 0.0,
            "value_sum": 0.0,
            "value_sq_sum": 0.0,
            "token_norm_sum": 0.0,
            "feature_count": 0.0,
        }

    valid_values = encoded.masked_select(valid_mask.unsqueeze(-1))
    token_norms = encoded.float().norm(dim=-1).masked_select(valid_mask)
    return {
        "decoded_frames": float(valid_frames),
        "value_sum": float(valid_values.float().sum().item()),
        "value_sq_sum": float(valid_values.float().square().sum().item()),
        "token_norm_sum": float(token_norms.sum().item()),
        "feature_count": float(valid_values.numel()),
    }


def summarize_encoder_output_diagnostics(diagnostics: dict[str, float]) -> dict[str, float]:
    decoded_frames = max(1.0, float(diagnostics.get("decoded_frames", 0.0)))
    feature_count = max(1.0, float(diagnostics.get("feature_count", 0.0)))
    mean_value = float(diagnostics.get("value_sum", 0.0)) / feature_count
    mean_square = float(diagnostics.get("value_sq_sum", 0.0)) / feature_count
    variance = max(0.0, mean_square - (mean_value * mean_value))
    return {
        "avg_mean": mean_value,
        "avg_std": variance ** 0.5,
        "avg_token_l2_norm": float(diagnostics.get("token_norm_sum", 0.0)) / decoded_frames,
    }


def top_emitted_token_histogram(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    tokenizer: Tokenizer,
    *,
    top_k: int = 5,
) -> list[tuple[int, float, str]]:
    valid_mask = (
        torch.arange(log_probs.size(1), device=output_lengths.device).unsqueeze(0)
        < output_lengths.unsqueeze(1)
    )
    valid_frames = int(valid_mask.sum().item())
    if valid_frames <= 0:
        return []

    top_token_ids = log_probs.argmax(dim=-1).masked_select(valid_mask)
    counts = torch.bincount(top_token_ids, minlength=log_probs.size(-1)).float()
    if counts.numel() == 0:
        return []
    top_count = min(max(1, top_k), int((counts > 0).sum().item()))
    top_values, top_indices = torch.topk(counts, k=top_count)

    histogram: list[tuple[int, float, str]] = []
    for token_id, count in zip(top_indices.tolist(), top_values.tolist(), strict=True):
        token_text = _token_display_text(tokenizer, int(token_id))
        histogram.append((int(token_id), float(count) / float(valid_frames), token_text))
    return histogram


def summarize_ctc_batch_diagnostics(diagnostics: dict[str, float]) -> dict[str, float]:
    decoded_frames = max(1.0, float(diagnostics.get("decoded_frames", 0.0)))
    sample_count = max(1.0, float(diagnostics.get("sample_count", 0.0)))
    target_tokens_sum = float(diagnostics.get("target_tokens_sum", 0.0))
    output_frames_sum = float(diagnostics.get("output_frames_sum", 0.0))
    return {
        "avg_blank_probability": float(diagnostics.get("blank_probability_sum", 0.0))
        / decoded_frames,
        "argmax_blank_fraction": float(diagnostics.get("argmax_blank_frames", 0.0))
        / decoded_frames,
        "avg_top_nonblank_probability": float(diagnostics.get("top_nonblank_probability_sum", 0.0))
        / decoded_frames,
        "avg_output_frames": output_frames_sum / sample_count,
        "avg_target_tokens": target_tokens_sum / sample_count,
        "target_tokens_per_frame": target_tokens_sum / decoded_frames,
        "impossible_sample_fraction": float(diagnostics.get("impossible_sample_count", 0.0))
        / sample_count,
        "tight_sample_fraction": float(diagnostics.get("tight_sample_count", 0.0)) / sample_count,
    }


def _bucket_name(reference: str) -> str:
    word_count = len(reference.split())
    if word_count <= 5:
        return "short"
    if word_count <= 15:
        return "medium"
    return "long"


def length_bucket_metrics(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, float]:
    grouped_refs: dict[str, list[str]] = defaultdict(list)
    grouped_hyps: dict[str, list[str]] = defaultdict(list)
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        bucket = _bucket_name(reference)
        grouped_refs[bucket].append(reference)
        grouped_hyps[bucket].append(hypothesis)

    metrics: dict[str, float] = {}
    for bucket in ("short", "medium", "long"):
        metrics[f"samples_{bucket}"] = float(len(grouped_refs[bucket]))
        if grouped_refs[bucket]:
            metrics[f"wer_{bucket}"] = word_error_rate(grouped_refs[bucket], grouped_hyps[bucket])
            metrics[f"cer_{bucket}"] = char_error_rate(grouped_refs[bucket], grouped_hyps[bucket])
    return metrics


def collect_examples(
    utterance_ids: list[str],
    speaker_ids: list[str | None],
    references: list[str],
    hypotheses: list[str],
    limit: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    hardest_examples = []
    random_examples = []
    pairs = list(zip(utterance_ids, speaker_ids, references, hypotheses, strict=True))
    ranked_pairs = sorted(
        pairs,
        key=lambda item: (item[2] == item[3], abs(len(item[2]) - len(item[3]))),
    )
    for utterance_id, speaker_id, reference, hypothesis in ranked_pairs[:limit]:
        hardest_examples.append(
            {
                "utterance_id": utterance_id,
                "speaker_id": speaker_id or "",
                "reference": reference,
                "hypothesis": hypothesis,
            }
        )
    shuffled_pairs = list(pairs)
    random.Random(13).shuffle(shuffled_pairs)
    for utterance_id, speaker_id, reference, hypothesis in shuffled_pairs[:limit]:
        random_examples.append(
            {
                "utterance_id": utterance_id,
                "speaker_id": speaker_id or "",
                "reference": reference,
                "hypothesis": hypothesis,
            }
        )
    return hardest_examples, random_examples


def decoding_debug_metrics(hypotheses: list[str]) -> dict[str, float]:
    if not hypotheses:
        return {
            "decoded_empty_fraction": 0.0,
            "decoded_avg_char_length": 0.0,
            "decoded_avg_word_length": 0.0,
        }
    empty_count = sum(1 for hypothesis in hypotheses if not hypothesis.strip())
    return {
        "decoded_empty_fraction": empty_count / len(hypotheses),
        "decoded_avg_char_length": sum(len(hypothesis) for hypothesis in hypotheses)
        / len(hypotheses),
        "decoded_avg_word_length": (
            sum(len(hypothesis.split()) for hypothesis in hypotheses) / len(hypotheses)
        ),
    }


def speaker_level_metrics(
    speaker_ids: list[str | None],
    has_speaker_ids: list[bool],
    references: list[str],
    hypotheses: list[str],
) -> dict[str, object]:
    grouped_refs: dict[str, list[str]] = defaultdict(list)
    grouped_hyps: dict[str, list[str]] = defaultdict(list)
    missing_count = 0
    for speaker_id, has_speaker_id, reference, hypothesis in zip(
        speaker_ids, has_speaker_ids, references, hypotheses, strict=True
    ):
        if not has_speaker_id or not speaker_id:
            missing_count += 1
            continue
        grouped_refs[speaker_id].append(reference)
        grouped_hyps[speaker_id].append(hypothesis)
    per_speaker = {
        speaker_id: {
            "samples": len(grouped_refs[speaker_id]),
            "wer": word_error_rate(grouped_refs[speaker_id], grouped_hyps[speaker_id]),
            "cer": char_error_rate(grouped_refs[speaker_id], grouped_hyps[speaker_id]),
        }
        for speaker_id in grouped_refs
    }
    macro_wer = (
        sum(item["wer"] for item in per_speaker.values()) / len(per_speaker) if per_speaker else 0.0
    )
    return {
        "speaker_count": len(per_speaker),
        "speaker_macro_wer": macro_wer,
        "speaker_id_available": missing_count == 0,
        "missing_speaker_id_samples": missing_count,
        "per_speaker": per_speaker,
    }


def _merge_evaluation_shards(
    shards: list[dict[str, object]],
    *,
    example_limit: int,
) -> dict[str, object]:
    total_loss = 0.0
    total_main_ctc_loss = 0.0
    total_intermediate_ctc_loss = 0.0
    total_combined_ctc_loss = 0.0
    total_aed_loss = 0.0
    total_liberta_distill_loss = 0.0
    total_audio_teacher_loss = 0.0
    total_batches = 0
    total_forward_seconds = 0.0
    total_teacher_seconds = 0.0
    total_decode_seconds = 0.0
    total_blank_probability = 0.0
    total_decoded_frames = 0
    total_argmax_blank_frames = 0.0
    total_top_nonblank_probability = 0.0
    total_target_tokens = 0.0
    total_samples = 0.0
    intermediate_ctc_diagnostics_totals: dict[int, dict[str, float]] = {}
    intermediate_ctc_diagnostics_totals: dict[int, dict[str, float]] = {}
    references: list[str] = []
    hypotheses: list[str] = []
    utterance_ids: list[str] = []
    speaker_ids: list[str | None] = []
    has_speaker_ids: list[bool] = []
    for shard in shards:
        total_loss += float(shard["total_loss"])
        total_main_ctc_loss += float(shard["total_main_ctc_loss"])
        total_intermediate_ctc_loss += float(shard["total_intermediate_ctc_loss"])
        total_combined_ctc_loss += float(shard["total_combined_ctc_loss"])
        total_aed_loss += float(shard["total_aed_loss"])
        total_liberta_distill_loss += float(shard["total_liberta_distill_loss"])
        total_audio_teacher_loss += float(shard.get("total_audio_teacher_loss", 0.0))
        total_batches += int(shard["total_batches"])
        total_forward_seconds += float(shard.get("total_forward_seconds", 0.0))
        total_teacher_seconds += float(shard.get("total_teacher_seconds", 0.0))
        total_decode_seconds += float(shard.get("total_decode_seconds", 0.0))
        total_blank_probability += float(shard.get("total_blank_probability", 0.0))
        total_decoded_frames += int(shard.get("total_decoded_frames", 0))
        total_argmax_blank_frames += float(shard.get("total_argmax_blank_frames", 0.0))
        total_top_nonblank_probability += float(shard.get("total_top_nonblank_probability", 0.0))
        total_target_tokens += float(shard.get("total_target_tokens", 0.0))
        total_samples += float(shard.get("total_samples", 0.0))
        for layer_key, diagnostics in shard.get("intermediate_ctc_diagnostics_totals", {}).items():
            layer_index = int(layer_key)
            current = intermediate_ctc_diagnostics_totals.setdefault(
                layer_index,
                {
                    "sample_count": 0.0,
                    "impossible_sample_count": 0.0,
                    "tight_sample_count": 0.0,
                },
            )
            current["sample_count"] += float(diagnostics.get("sample_count", 0.0))
            current["impossible_sample_count"] += float(
                diagnostics.get("impossible_sample_count", 0.0)
            )
            current["tight_sample_count"] += float(diagnostics.get("tight_sample_count", 0.0))
        references.extend(shard["references"])
        hypotheses.extend(shard["hypotheses"])
        utterance_ids.extend(shard["utterance_ids"])
        speaker_ids.extend(shard["speaker_ids"])
        has_speaker_ids.extend(shard["has_speaker_ids"])

    metrics = {
        "loss": total_loss / max(1, total_batches),
        "main_ctc_loss": total_main_ctc_loss / max(1, total_batches),
        "intermediate_ctc_loss": total_intermediate_ctc_loss / max(1, total_batches),
        "combined_ctc_loss": total_combined_ctc_loss / max(1, total_batches),
        "aed_loss": total_aed_loss / max(1, total_batches),
        "liberta_distill_loss": total_liberta_distill_loss / max(1, total_batches),
        "audio_teacher_loss": total_audio_teacher_loss / max(1, total_batches),
        "cer": char_error_rate(references, hypotheses),
        "wer": word_error_rate(references, hypotheses),
    }
    metrics.update(
        summarize_ctc_batch_diagnostics(
            {
                "blank_probability_sum": total_blank_probability,
                "decoded_frames": float(total_decoded_frames),
                "argmax_blank_frames": total_argmax_blank_frames,
                "top_nonblank_probability_sum": total_top_nonblank_probability,
                "sample_count": total_samples,
                "output_frames_sum": float(total_decoded_frames),
                "target_tokens_sum": total_target_tokens,
            }
        )
    )
    metrics.update(length_bucket_metrics(references, hypotheses))
    metrics.update(decoding_debug_metrics(hypotheses))
    for layer_index, diagnostics in sorted(intermediate_ctc_diagnostics_totals.items()):
        sample_count = max(1.0, float(diagnostics.get("sample_count", 0.0)))
        metrics[f"layer{layer_index}_ctc_impossible_fraction"] = float(
            diagnostics.get("impossible_sample_count", 0.0)
        ) / sample_count
        metrics[f"layer{layer_index}_ctc_tight_fraction"] = float(
            diagnostics.get("tight_sample_count", 0.0)
        ) / sample_count
    speaker_metrics = speaker_level_metrics(speaker_ids, has_speaker_ids, references, hypotheses)
    metrics["speaker_count"] = float(speaker_metrics["speaker_count"])
    metrics["speaker_macro_wer"] = float(speaker_metrics["speaker_macro_wer"])
    metrics["speaker_id_available"] = float(speaker_metrics["speaker_id_available"])
    metrics["missing_speaker_id_samples"] = float(speaker_metrics["missing_speaker_id_samples"])
    hardest_examples, random_examples = collect_examples(
        utterance_ids,
        speaker_ids,
        references,
        hypotheses,
        limit=example_limit,
    )
    return {
        "metrics": metrics,
        "timings": {
            "forward_seconds": total_forward_seconds,
            "teacher_seconds": total_teacher_seconds,
            "decode_seconds": total_decode_seconds,
        },
        "intermediate_ctc_diagnostics_totals": intermediate_ctc_diagnostics_totals,
        "hardest_examples": hardest_examples,
        "random_examples": random_examples,
        "speaker_metrics": speaker_metrics,
    }


def evaluate(
    model: SqueezeformerCTC,
    dataloader,
    criterion: nn.CTCLoss,
    tokenizer: Tokenizer,
    device: torch.device,
    dtype: DTypeChoice,
    fp8_recipe=None,
    decode_strategy: DecodeStrategy = DecodeStrategy.GREEDY,
    beam_size: int = 8,
    lm_scorer=None,
    lm_weight: float = 0.0,
    beam_length_bonus: float = 0.1,
    example_limit: int = 5,
    intermediate_ctc_weight: float = 0.0,
    aed_loss_weight: float = 0.0,
    liberta_teacher: FrozenLibertaTeacher | None = None,
    liberta_distill_weight: float = 0.0,
    audio_teacher: FrozenAudioTeacher | None = None,
    audio_teacher_weight: float = 0.0,
    audio_teacher_objective: str = "hidden_mse",
    distributed: bool = False,
    is_main_process: bool = True,
) -> dict[str, object] | None:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_main_ctc_loss = 0.0
    total_intermediate_ctc_loss = 0.0
    total_combined_ctc_loss = 0.0
    total_aed_loss = 0.0
    total_liberta_distill_loss = 0.0
    total_audio_teacher_loss = 0.0
    total_batches = 0
    total_forward_seconds = 0.0
    total_teacher_seconds = 0.0
    total_decode_seconds = 0.0
    total_blank_probability = 0.0
    total_decoded_frames = 0
    total_argmax_blank_frames = 0.0
    total_top_nonblank_probability = 0.0
    total_target_tokens = 0.0
    total_samples = 0.0
    intermediate_ctc_diagnostics_totals: dict[int, dict[str, float]] = {}
    references: list[str] = []
    hypotheses: list[str] = []
    utterance_ids: list[str] = []
    speaker_ids: list[str | None] = []
    has_speaker_ids: list[bool] = []
    try:
        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    logger.warning("skipping empty validation batch after dataset filtering")
                    continue
                features = batch["features"].to(device)
                feature_lengths = batch["feature_lengths"].to(device)
                targets = batch["targets"].to(device)
                target_lengths = batch["target_lengths"].to(device)
                if model.aed_decoder is not None:
                    decoder_inputs, decoder_targets, decoder_target_lengths = _build_aed_targets(
                        targets,
                        target_lengths,
                        bos_id=model.aed_decoder.bos_id,
                        eos_id=model.aed_decoder.eos_id,
                        token_offset=model.aed_decoder.token_offset,
                        pad_id=model.aed_decoder.pad_id,
                    )
                    decoder_inputs = decoder_inputs.to(device)
                    decoder_targets = decoder_targets.to(device)
                    decoder_target_lengths = decoder_target_lengths.to(device)
                else:
                    decoder_inputs = None
                    decoder_targets = None
                    decoder_target_lengths = None

                with _autocast_context(device, dtype, fp8_recipe=fp8_recipe):
                    forward_start_time = time.perf_counter()
                    forward_outputs = model(
                        features,
                        feature_lengths,
                        return_training_outputs=True,
                        targets=targets,
                        target_lengths=target_lengths,
                        blank_id=tokenizer.blank_id,
                        return_main_log_probs=True,
                        decoder_inputs=decoder_inputs,
                        liberta_lengths=decoder_target_lengths,
                    )
                    output_lengths = forward_outputs["output_lengths"]
                    log_probs = forward_outputs["main_log_probs"]
                    main_ctc_loss = forward_outputs["main_ctc_loss"]
                    intermediate_ctc_losses_map = forward_outputs["intermediate_ctc_losses"]
                    intermediate_ctc_diagnostics_map = forward_outputs.get(
                        "intermediate_ctc_diagnostics",
                        {},
                    )
                    if intermediate_ctc_losses_map:
                        intermediate_ctc_loss = torch.stack(
                            [
                                intermediate_ctc_losses_map[layer_index]
                                for layer_index in model.intermediate_ctc_layers
                            ]
                        ).mean()
                        combined_ctc_loss = (
                            1.0 - intermediate_ctc_weight
                        ) * main_ctc_loss + intermediate_ctc_weight * intermediate_ctc_loss
                    else:
                        intermediate_ctc_loss = None
                        combined_ctc_loss = main_ctc_loss
                    aed_logits = forward_outputs.get("aed_logits")
                    liberta_student_embeddings = forward_outputs.get("liberta_student_embeddings")
                    audio_teacher_student_states = forward_outputs.get(
                        "audio_teacher_student_states"
                    )
                    if aed_logits is not None and decoder_targets is not None:
                        aed_loss = _aed_cross_entropy_loss(
                            aed_logits,
                            decoder_targets,
                            pad_id=model.aed_decoder.pad_id,
                        )
                        loss = (
                            1.0 - aed_loss_weight
                        ) * combined_ctc_loss + aed_loss_weight * aed_loss
                    else:
                        aed_loss = None
                        loss = combined_ctc_loss
                total_forward_seconds += time.perf_counter() - forward_start_time
                if liberta_teacher is not None and liberta_student_embeddings is not None:
                    teacher_start_time = time.perf_counter()
                    teacher_embeddings = liberta_teacher.encode(batch["transcripts"]).to(
                        device=liberta_student_embeddings.device,
                        dtype=liberta_student_embeddings.dtype,
                    )
                    liberta_distill_loss = F.mse_loss(
                        F.normalize(liberta_student_embeddings.float(), dim=-1),
                        F.normalize(teacher_embeddings.float(), dim=-1),
                    )
                    loss = loss + (liberta_distill_weight * liberta_distill_loss)
                    total_teacher_seconds += time.perf_counter() - teacher_start_time
                else:
                    liberta_distill_loss = None
                if audio_teacher is not None and audio_teacher_student_states is not None:
                    teacher_start_time = time.perf_counter()
                    teacher_outputs = audio_teacher.encode_waveforms(
                        batch["waveforms"],
                        batch["waveform_lengths"],
                        sample_rates=batch.get("sample_rates"),
                    )
                    teacher_embeddings = teacher_outputs["pooled_hidden"].to(
                        device=audio_teacher_student_states.device,
                        dtype=audio_teacher_student_states.dtype,
                    )
                    if audio_teacher_objective == "hidden_cosine":
                        audio_teacher_loss = (
                            1.0
                            - F.cosine_similarity(
                                audio_teacher_student_states.float(),
                                teacher_embeddings.float(),
                                dim=-1,
                            )
                        ).mean()
                    else:
                        audio_teacher_loss = F.mse_loss(
                            F.normalize(audio_teacher_student_states.float(), dim=-1),
                            F.normalize(teacher_embeddings.float(), dim=-1),
                        )
                    loss = loss + (audio_teacher_weight * audio_teacher_loss)
                    total_teacher_seconds += time.perf_counter() - teacher_start_time
                else:
                    audio_teacher_loss = None
                total_loss += float(loss.item())
                total_main_ctc_loss += float(main_ctc_loss.item())
                total_intermediate_ctc_loss += float(
                    intermediate_ctc_loss.item() if intermediate_ctc_loss is not None else 0.0
                )
                total_combined_ctc_loss += float(combined_ctc_loss.item())
                total_aed_loss += float(aed_loss.item() if aed_loss is not None else 0.0)
                total_liberta_distill_loss += float(
                    liberta_distill_loss.item() if liberta_distill_loss is not None else 0.0
                )
                total_audio_teacher_loss += float(
                    audio_teacher_loss.item() if audio_teacher_loss is not None else 0.0
                )
                total_batches += 1
                references.extend(batch["transcripts"])
                utterance_ids.extend(batch["utterance_ids"])
                speaker_ids.extend(batch["speaker_ids"])
                has_speaker_ids.extend(batch["has_speaker_ids"])
                diagnostics = ctc_batch_diagnostics(
                    log_probs,
                    output_lengths,
                    tokenizer,
                    target_lengths=target_lengths,
                )
                total_blank_probability += diagnostics["blank_probability_sum"]
                total_decoded_frames += int(diagnostics["decoded_frames"])
                total_argmax_blank_frames += diagnostics["argmax_blank_frames"]
                total_top_nonblank_probability += diagnostics["top_nonblank_probability_sum"]
                total_target_tokens += diagnostics.get("target_tokens_sum", 0.0)
                total_samples += diagnostics["sample_count"]
                for layer_index, layer_diagnostics in intermediate_ctc_diagnostics_map.items():
                    current = intermediate_ctc_diagnostics_totals.setdefault(
                        int(layer_index),
                        {
                            "sample_count": 0.0,
                            "impossible_sample_count": 0.0,
                            "tight_sample_count": 0.0,
                        },
                    )
                    current["sample_count"] += float(layer_diagnostics.get("sample_count", 0.0))
                    current["impossible_sample_count"] += float(
                        layer_diagnostics.get("impossible_sample_count", 0.0)
                    )
                    current["tight_sample_count"] += float(
                        layer_diagnostics.get("tight_sample_count", 0.0)
                    )
                decode_start_time = time.perf_counter()
                decoded_hypotheses = decode_batch(
                    log_probs,
                    output_lengths,
                    tokenizer=tokenizer,
                    strategy=decode_strategy,
                    beam_size=beam_size,
                    lm_scorer=lm_scorer,
                    lm_weight=lm_weight,
                    beam_length_bonus=beam_length_bonus,
                )
                total_decode_seconds += time.perf_counter() - decode_start_time
                hypotheses.extend(decoded_hypotheses)
    finally:
        if was_training:
            model.train()
    local_results = {
        "total_loss": total_loss,
        "total_main_ctc_loss": total_main_ctc_loss,
        "total_intermediate_ctc_loss": total_intermediate_ctc_loss,
        "total_combined_ctc_loss": total_combined_ctc_loss,
        "total_aed_loss": total_aed_loss,
        "total_liberta_distill_loss": total_liberta_distill_loss,
        "total_audio_teacher_loss": total_audio_teacher_loss,
        "total_batches": total_batches,
        "total_forward_seconds": total_forward_seconds,
        "total_teacher_seconds": total_teacher_seconds,
        "total_decode_seconds": total_decode_seconds,
        "total_blank_probability": total_blank_probability,
        "total_decoded_frames": total_decoded_frames,
        "total_argmax_blank_frames": total_argmax_blank_frames,
        "total_top_nonblank_probability": total_top_nonblank_probability,
        "total_target_tokens": total_target_tokens,
        "total_samples": total_samples,
        "intermediate_ctc_diagnostics_totals": intermediate_ctc_diagnostics_totals,
        "references": references,
        "hypotheses": hypotheses,
        "utterance_ids": utterance_ids,
        "speaker_ids": speaker_ids,
        "has_speaker_ids": has_speaker_ids,
    }
    if distributed and dist.is_initialized():
        gathered_results = [None] * dist.get_world_size() if is_main_process else None
        gather_start_time = time.perf_counter()
        dist.gather_object(
            local_results,
            object_gather_list=gathered_results,
            dst=0,
        )
        gather_seconds = time.perf_counter() - gather_start_time
        if not is_main_process:
            return None
        merged_results = _merge_evaluation_shards(gathered_results, example_limit=example_limit)
        merged_results["timings"]["gather_seconds"] = gather_seconds
        return merged_results
    merged_results = _merge_evaluation_shards([local_results], example_limit=example_limit)
    merged_results["timings"]["gather_seconds"] = 0.0
    return merged_results


def _resolve_block_pattern(block_pattern: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in block_pattern.split(",") if token.strip())
    if not tokens or any(token not in {"M", "C", "s"} for token in tokens):
        raise ValueError("block pattern must be a comma-separated sequence drawn from M,C,s")
    return tokens


def _resolve_float_tuple(values: str) -> tuple[float, ...]:
    parsed = tuple(float(value.strip()) for value in values.split(",") if value.strip())
    if not parsed:
        raise ValueError("expected at least one float value")
    return parsed


def _resolve_scheduler_kwargs(args: argparse.Namespace, optimizer_name: str) -> dict[str, float]:
    if optimizer_name == "muon":
        return {
            "warmup_epochs": (
                args.muon_warmup_epochs
                if args.muon_warmup_epochs is not None
                else args.warmup_epochs
            ),
            "hold_epochs": (
                args.muon_hold_epochs if args.muon_hold_epochs is not None else args.hold_epochs
            ),
            "decay_exponent": (
                args.muon_decay_exponent
                if args.muon_decay_exponent is not None
                else args.decay_exponent
            ),
        }
    return {
        "warmup_epochs": (
            args.adamw_warmup_epochs if args.adamw_warmup_epochs is not None else args.warmup_epochs
        ),
        "hold_epochs": (
            args.adamw_hold_epochs if args.adamw_hold_epochs is not None else args.hold_epochs
        ),
        "decay_exponent": (
            args.adamw_decay_exponent
            if args.adamw_decay_exponent is not None
            else args.decay_exponent
        ),
    }


def _flatten_examples(prefix: str, examples: list[dict[str, str]]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for index, example in enumerate(examples):
        payload[f"{prefix}_example_{index}_id"] = example["utterance_id"]
        payload[f"{prefix}_example_{index}_speaker"] = example["speaker_id"]
        payload[f"{prefix}_example_{index}_ref"] = example["reference"]
        payload[f"{prefix}_example_{index}_hyp"] = example["hypothesis"]
    return payload


def _examples_table(
    examples: list[dict[str, str]],
    *,
    split: str,
    category: str,
    epoch: int,
    global_step: int,
) -> trackio.Table | None:
    if not examples:
        return None
    rows = [
        {
            "split": split,
            "category": category,
            "rank": index,
            "epoch": epoch,
            "global_step": global_step,
            "utterance_id": example["utterance_id"],
            "speaker_id": example["speaker_id"],
            "reference": example["reference"],
            "hypothesis": example["hypothesis"],
        }
        for index, example in enumerate(examples, start=1)
    ]
    return trackio.Table(data=rows)


def _checkpoint_safe_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in metrics.items() if not isinstance(value, trackio.Table)}


def _evaluate_and_checkpoint(
    *,
    model: SqueezeformerCTC,
    val_loader,
    criterion: nn.CTCLoss,
    tokenizer: Tokenizer,
    device: torch.device,
    dtype: DTypeChoice,
    fp8_recipe,
    decode_strategy: DecodeStrategy,
    beam_size: int,
    lm_scorer,
    lm_weight: float,
    beam_length_bonus: float,
    example_limit: int,
    intermediate_ctc_weight: float,
    aed_loss_weight: float,
    liberta_teacher: FrozenLibertaTeacher | None,
    liberta_distill_weight: float,
    audio_teacher: FrozenAudioTeacher | None,
    audio_teacher_weight: float,
    audio_teacher_objective: str,
    ema: ExponentialMovingAverage | None,
    validation_model_source: ValidationModelSource,
    train_metrics: dict[str, float],
    epoch: int,
    global_step: int,
    output_dir: Path,
    encoder_config: SqueezeformerConfig,
    featurizer: AudioFeaturizer,
    optimizers: list[torch.optim.Optimizer],
    optimizer_names: list[str],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    best_val_wer: float,
    split_audit: dict[str, object],
    logger: logging.Logger,
    save_last_checkpoint: bool,
    report_stem: str,
    distributed: bool,
    is_main_process: bool,
) -> float:
    raw_state_dict = _clone_state_dict(model)
    selected_source = validation_model_source
    if selected_source == ValidationModelSource.EMA:
        if ema is None:
            raise RuntimeError("validation_model_source='ema' requires EMA to be enabled.")
        ema.apply_to(model)

    try:
        validation = evaluate(
            model,
            val_loader,
            criterion,
            tokenizer,
            device,
            dtype,
            fp8_recipe=fp8_recipe,
            decode_strategy=decode_strategy,
            beam_size=beam_size,
            lm_scorer=lm_scorer,
            lm_weight=lm_weight,
            beam_length_bonus=beam_length_bonus,
            example_limit=example_limit,
            intermediate_ctc_weight=intermediate_ctc_weight,
            aed_loss_weight=aed_loss_weight,
            liberta_teacher=liberta_teacher,
            liberta_distill_weight=liberta_distill_weight,
            audio_teacher=audio_teacher,
            audio_teacher_weight=audio_teacher_weight,
            audio_teacher_objective=audio_teacher_objective,
            distributed=distributed,
            is_main_process=is_main_process,
        )
        if not is_main_process:
            return best_val_wer

        val_metrics = validation["metrics"]
        validation_timings = validation.get("timings", {})
        log_payload = {
            "epoch": epoch,
            "global_step": global_step,
            **train_metrics,
            "val_loss": val_metrics["loss"],
            "val_cer": val_metrics["cer"],
            "val_wer": val_metrics["wer"],
            "val_forward_seconds": float(validation_timings.get("forward_seconds", 0.0)),
            "val_teacher_seconds": float(validation_timings.get("teacher_seconds", 0.0)),
            "val_decode_seconds": float(validation_timings.get("decode_seconds", 0.0)),
            "val_gather_seconds": float(validation_timings.get("gather_seconds", 0.0)),
            "val_model_source": selected_source,
        }
        for key, value in val_metrics.items():
            if key not in {"loss", "cer", "wer"}:
                log_payload[f"val_{key}"] = value
        log_payload.update(_flatten_examples("val_hardest", validation["hardest_examples"]))
        log_payload.update(_flatten_examples("val_random", validation["random_examples"]))
        hardest_examples_table = _examples_table(
            validation["hardest_examples"],
            split="validation",
            category="hardest",
            epoch=epoch,
            global_step=global_step,
        )
        if hardest_examples_table is not None:
            log_payload["val_hardest_samples"] = hardest_examples_table
        random_examples_table = _examples_table(
            validation["random_examples"],
            split="validation",
            category="random",
            epoch=epoch,
            global_step=global_step,
        )
        if random_examples_table is not None:
            log_payload["val_random_samples"] = random_examples_table
        log_payload.update(
            _build_trackio_grouped_metrics(
                groups={
                    "val": {
                        "loss": val_metrics["loss"],
                        "cer": val_metrics["cer"],
                        "wer": val_metrics["wer"],
                        **{
                            key: value
                            for key, value in val_metrics.items()
                            if key not in {"loss", "cer", "wer"}
                        },
                        "forward_seconds": float(validation_timings.get("forward_seconds", 0.0)),
                        "teacher_seconds": float(validation_timings.get("teacher_seconds", 0.0)),
                        "decode_seconds": float(validation_timings.get("decode_seconds", 0.0)),
                        "gather_seconds": float(validation_timings.get("gather_seconds", 0.0)),
                    },
                    "train": {
                        key.removeprefix("train_"): value for key, value in train_metrics.items()
                    },
                },
            )
        )
        trackio.log(log_payload)

        report = {
            "epoch": epoch,
            "global_step": global_step,
            "model_source": selected_source,
            "metrics": val_metrics,
            "hardest_examples": validation["hardest_examples"],
            "random_examples": validation["random_examples"],
            "speaker_metrics": validation["speaker_metrics"],
            "split_audit": split_audit,
        }
        reports_dir = output_dir / "eval_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{report_stem}.json"
        report_start_time = time.perf_counter()
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        report_seconds = time.perf_counter() - report_start_time

        updated_best_val_wer = min(best_val_wer, float(val_metrics["wer"]))
        checkpoint_build_start_time = time.perf_counter()
        checkpoint = _build_checkpoint(
            model=model,
            encoder_config=encoder_config,
            tokenizer=tokenizer,
            featurizer=featurizer,
            epoch=epoch,
            global_step=global_step,
            best_val_wer=updated_best_val_wer,
            metrics=_checkpoint_safe_metrics(log_payload),
            optimizers=optimizers,
            optimizer_names=optimizer_names,
            schedulers=schedulers,
            scaler=scaler,
            ema=ema,
            args=args,
            resume_model_state_dict=(
                raw_state_dict if selected_source == ValidationModelSource.EMA else None
            ),
            validation_model_source=selected_source,
        )
        checkpoint_build_seconds = time.perf_counter() - checkpoint_build_start_time

        if save_last_checkpoint:
            latest_path = output_dir / "checkpoint_last.pt"
        else:
            latest_path = output_dir / "checkpoint_step_last.pt"
        latest_export_start_time = time.perf_counter()
        save_checkpoint(checkpoint, latest_path)
        latest_safetensors_path = _export_inference_checkpoint(checkpoint, latest_path)
        latest_export_seconds = time.perf_counter() - latest_export_start_time

        if float(val_metrics["wer"]) < best_val_wer:
            updated_best_val_wer = float(val_metrics["wer"])
            best_path = output_dir / "checkpoint_best.pt"
            best_export_start_time = time.perf_counter()
            save_checkpoint(checkpoint, best_path)
            best_safetensors_path = _export_inference_checkpoint(checkpoint, best_path)
            best_export_seconds = time.perf_counter() - best_export_start_time
        else:
            best_safetensors_path = _safetensors_path(output_dir / "checkpoint_best.pt")
            best_export_seconds = 0.0

        topk_update_start_time = time.perf_counter()
        _update_top_checkpoints(
            output_dir=output_dir,
            checkpoint=checkpoint,
            epoch=epoch,
            global_step=global_step,
            val_wer=float(val_metrics["wer"]),
            keep_top_k=args.keep_top_k,
        )
        topk_update_seconds = time.perf_counter() - topk_update_start_time
        topk_average_start_time = time.perf_counter()
        averaged_path = _average_topk_checkpoints(output_dir)
        topk_average_seconds = time.perf_counter() - topk_average_start_time
        checkpoint_export_seconds = (
            report_seconds
            + checkpoint_build_seconds
            + latest_export_seconds
            + best_export_seconds
            + topk_update_seconds
            + topk_average_seconds
        )
        argmax_blank_fraction = float(val_metrics.get("argmax_blank_fraction", 0.0))
        avg_top_nonblank_probability = float(
            val_metrics.get("avg_top_nonblank_probability", 0.0)
        )
        target_tokens_per_frame = float(val_metrics.get("target_tokens_per_frame", 0.0))
        impossible_sample_fraction = float(val_metrics.get("impossible_sample_fraction", 0.0))
        tight_sample_fraction = float(val_metrics.get("tight_sample_fraction", 0.0))
        logger.info(
            (
                "%s complete train_loss=%.4f val_loss=%.4f "
                "val_main_ctc_loss=%.4f val_intermediate_ctc_loss=%.4f "
                "val_combined_ctc_loss=%.4f val_aed_loss=%.4f "
                "val_liberta_distill_loss=%.4f val_audio_teacher_loss=%.4f val_cer=%.4f val_wer=%.4f "
                "val_avg_blank_prob=%.4f val_argmax_blank_frac=%.4f "
                "val_avg_top_nonblank_prob=%.4f val_target_tokens_per_frame=%.4f "
                "val_ctc_impossible_frac=%.4f val_ctc_tight_frac=%.4f "
                "val_empty_hyp_frac=%.4f "
                "val_avg_hyp_chars=%.2f val_avg_hyp_words=%.2f "
                "val_model_source=%s best_val_wer=%.4f timing_forward=%.2fs timing_teacher=%.2fs "
                "timing_decode=%.2fs timing_gather=%.2fs timing_checkpoint_export=%.2fs "
                "report=%s latest=%s best=%s averaged=%s"
            ),
            report_stem,
            float(train_metrics["train_loss"]),
            float(val_metrics["loss"]),
            float(val_metrics["main_ctc_loss"]),
            float(val_metrics["intermediate_ctc_loss"]),
            float(val_metrics["combined_ctc_loss"]),
            float(val_metrics["aed_loss"]),
            float(val_metrics["liberta_distill_loss"]),
            float(val_metrics["audio_teacher_loss"]),
            float(val_metrics["cer"]),
            float(val_metrics["wer"]),
            float(val_metrics["avg_blank_probability"]),
            argmax_blank_fraction,
            avg_top_nonblank_probability,
            target_tokens_per_frame,
            impossible_sample_fraction,
            tight_sample_fraction,
            float(val_metrics["decoded_empty_fraction"]),
            float(val_metrics["decoded_avg_char_length"]),
            float(val_metrics["decoded_avg_word_length"]),
            selected_source,
            updated_best_val_wer,
            float(validation_timings.get("forward_seconds", 0.0)),
            float(validation_timings.get("teacher_seconds", 0.0)),
            float(validation_timings.get("decode_seconds", 0.0)),
            float(validation_timings.get("gather_seconds", 0.0)),
            checkpoint_export_seconds,
            report_path,
            latest_path,
            output_dir / "checkpoint_best.pt",
            averaged_path if averaged_path is not None else "n/a",
        )
        intermediate_layers = tuple(getattr(model, "intermediate_ctc_layers", ()))
        intermediate_length_detail = " ".join(
            (
                f"layer{layer_index}_ctc_impossible="
                f"{float(val_metrics[f'layer{layer_index}_ctc_impossible_fraction']):.4f} "
                f"layer{layer_index}_ctc_tight="
                f"{float(val_metrics[f'layer{layer_index}_ctc_tight_fraction']):.4f}"
            )
            for layer_index in intermediate_layers
            if f"layer{layer_index}_ctc_impossible_fraction" in val_metrics
        )
        if intermediate_length_detail:
            logger.info("%s intermediate CTC feasibility %s", report_stem, intermediate_length_detail)
        logger.info(
            (
                "%s timing detail report_write=%.2fs checkpoint_build=%.2fs "
                "latest_export=%.2fs best_export=%.2fs topk_update=%.2fs topk_average=%.2fs"
            ),
            report_stem,
            report_seconds,
            checkpoint_build_seconds,
            latest_export_seconds,
            best_export_seconds,
            topk_update_seconds,
            topk_average_seconds,
        )
        logger.info(
            "exported inference artifacts latest_safe=%s best_safe=%s averaged_safe=%s",
            latest_safetensors_path,
            best_safetensors_path if best_safetensors_path.exists() else "n/a",
            _safetensors_path(averaged_path).as_posix() if averaged_path is not None else "n/a",
        )
        if validation["hardest_examples"]:
            hardest = validation["hardest_examples"][0]
            logger.info(
                "validation preview hardest_ref=%r hardest_hyp=%r",
                hardest["reference"][:120],
                hardest["hypothesis"][:120],
            )
        return updated_best_val_wer
    finally:
        _load_cloned_state_dict(model, raw_state_dict)


def _build_checkpoint(
    model: SqueezeformerCTC,
    encoder_config: SqueezeformerConfig,
    tokenizer: Tokenizer,
    featurizer: AudioFeaturizer,
    epoch: int,
    global_step: int,
    best_val_wer: float,
    metrics: dict[str, float],
    optimizers: list[torch.optim.Optimizer],
    optimizer_names: list[str],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    scaler: torch.amp.GradScaler,
    ema: ExponentialMovingAverage | None,
    args: argparse.Namespace,
    resume_model_state_dict: dict[str, torch.Tensor] | None = None,
    validation_model_source: ValidationModelSource = ValidationModelSource.RAW,
) -> dict[str, object]:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "encoder_config": asdict(encoder_config),
        "tokenizer": tokenizer.to_dict(),
        "featurizer_config": featurizer.config_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_wer": best_val_wer,
        "metrics": metrics,
        "optimizer_names": optimizer_names,
        "optimizer_state_dicts": [optimizer.state_dict() for optimizer in optimizers],
        "scheduler_state_dicts": [scheduler.state_dict() for scheduler in schedulers],
        "scaler_state_dict": scaler.state_dict(),
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "training_args": vars(args),
        "validation_model_source": validation_model_source,
    }
    if resume_model_state_dict is not None:
        checkpoint["resume_model_state_dict"] = resume_model_state_dict
    return checkpoint
