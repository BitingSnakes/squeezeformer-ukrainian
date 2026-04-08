from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
import trackio
from torch import Tensor, nn
from torch.nn import functional as F

from squeezeformer_pytorch.asr import Tokenizer, SqueezeformerCTC, ctc_prefix_beam_search
from squeezeformer_pytorch.checkpoints import save_checkpoint
from squeezeformer_pytorch.data import AudioFeaturizer
from squeezeformer_pytorch.metrics import char_error_rate, word_error_rate
from squeezeformer_pytorch.model import SqueezeformerConfig
from squeezeformer_pytorch.runtime_types import DecodeStrategy, DTypeChoice
from squeezeformer_pytorch.training.runtime import (
    ExponentialMovingAverage,
    FrozenLibertaTeacher,
    _autocast_context,
    _average_topk_checkpoints,
    _export_inference_checkpoint,
    _safetensors_path,
    _update_top_checkpoints,
)

def greedy_decode(log_probs: torch.Tensor, tokenizer: Tokenizer) -> list[str]:
    token_ids = log_probs.argmax(dim=-1).cpu().tolist()
    return [tokenizer.decode_ctc(sequence) for sequence in token_ids]


def decode_batch(
    log_probs: torch.Tensor,
    tokenizer: Tokenizer,
    strategy: DecodeStrategy,
    beam_size: int = 8,
    lm_scorer=None,
    lm_weight: float = 0.0,
) -> list[str]:
    if strategy == DecodeStrategy.GREEDY:
        return greedy_decode(log_probs, tokenizer)
    return [
        ctc_prefix_beam_search(
            sequence.cpu(),
            tokenizer=tokenizer,
            beam_size=beam_size,
            lm_scorer=lm_scorer,
            lm_weight=lm_weight,
        )
        for sequence in log_probs
    ]


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
    example_limit: int = 5,
    intermediate_ctc_weight: float = 0.0,
    aed_loss_weight: float = 0.0,
    liberta_teacher: FrozenLibertaTeacher | None = None,
    liberta_distill_weight: float = 0.0,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_main_ctc_loss = 0.0
    total_intermediate_ctc_loss = 0.0
    total_combined_ctc_loss = 0.0
    total_aed_loss = 0.0
    total_liberta_distill_loss = 0.0
    total_batches = 0
    references: list[str] = []
    hypotheses: list[str] = []
    utterance_ids: list[str] = []
    speaker_ids: list[str | None] = []
    has_speaker_ids: list[bool] = []
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
                encoded, output_lengths, intermediate_encoded, intermediate_output_lengths = (
                    model.encode_with_intermediates(features, feature_lengths)
                )
                log_probs, intermediate_log_probs = model.ctc_log_probs_from_encoded(
                    encoded,
                    intermediate_encoded,
                )
                main_ctc_loss = criterion(
                    log_probs.float().transpose(0, 1),
                    targets,
                    output_lengths,
                    target_lengths,
                )
                if intermediate_log_probs and intermediate_output_lengths:
                    intermediate_ctc_losses = [
                        criterion(
                            intermediate_log_probs[layer_index].float().transpose(0, 1),
                            targets,
                            intermediate_output_lengths[layer_index],
                            target_lengths,
                        )
                        for layer_index in model.intermediate_ctc_layers
                    ]
                    intermediate_ctc_loss = torch.stack(intermediate_ctc_losses).mean()
                    combined_ctc_loss = (
                        1.0 - intermediate_ctc_weight
                    ) * main_ctc_loss + intermediate_ctc_weight * intermediate_ctc_loss
                else:
                    intermediate_ctc_loss = None
                    combined_ctc_loss = main_ctc_loss
                if decoder_inputs is not None and decoder_targets is not None:
                    aed_logits, aed_hidden = model.aed_from_encoded(
                        encoded,
                        output_lengths,
                        decoder_inputs,
                    )
                    aed_loss = _aed_cross_entropy_loss(
                        aed_logits,
                        decoder_targets,
                        pad_id=model.aed_decoder.pad_id,
                    )
                    loss = (1.0 - aed_loss_weight) * combined_ctc_loss + aed_loss_weight * aed_loss
                else:
                    aed_loss = None
                    aed_hidden = None
                    loss = combined_ctc_loss
            if (
                liberta_teacher is not None
                and aed_hidden is not None
                and decoder_target_lengths is not None
            ):
                teacher_embeddings = liberta_teacher.encode(batch["transcripts"])
                student_embeddings = model.project_aed_hidden_for_liberta(
                    aed_hidden,
                    decoder_target_lengths,
                )
                liberta_distill_loss = F.mse_loss(
                    F.normalize(student_embeddings.float(), dim=-1),
                    F.normalize(teacher_embeddings.float(), dim=-1),
                )
                loss = loss + (liberta_distill_weight * liberta_distill_loss)
            else:
                liberta_distill_loss = None
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
            total_batches += 1
            references.extend(batch["transcripts"])
            utterance_ids.extend(batch["utterance_ids"])
            speaker_ids.extend(batch["speaker_ids"])
            has_speaker_ids.extend(batch["has_speaker_ids"])
            hypotheses.extend(
                decode_batch(
                    log_probs,
                    tokenizer=tokenizer,
                    strategy=decode_strategy,
                    beam_size=beam_size,
                    lm_scorer=lm_scorer,
                    lm_weight=lm_weight,
                )
            )
    metrics = {
        "loss": total_loss / max(1, total_batches),
        "main_ctc_loss": total_main_ctc_loss / max(1, total_batches),
        "intermediate_ctc_loss": total_intermediate_ctc_loss / max(1, total_batches),
        "combined_ctc_loss": total_combined_ctc_loss / max(1, total_batches),
        "aed_loss": total_aed_loss / max(1, total_batches),
        "liberta_distill_loss": total_liberta_distill_loss / max(1, total_batches),
        "cer": char_error_rate(references, hypotheses),
        "wer": word_error_rate(references, hypotheses),
    }
    metrics.update(length_bucket_metrics(references, hypotheses))
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
        "hardest_examples": hardest_examples,
        "random_examples": random_examples,
        "speaker_metrics": speaker_metrics,
    }


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
    example_limit: int,
    intermediate_ctc_weight: float,
    aed_loss_weight: float,
    liberta_teacher: FrozenLibertaTeacher | None,
    liberta_distill_weight: float,
    ema: ExponentialMovingAverage | None,
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
) -> float:
    ema_backup = ema.apply_to(model) if ema is not None else None
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
        example_limit=example_limit,
        intermediate_ctc_weight=intermediate_ctc_weight,
        aed_loss_weight=aed_loss_weight,
        liberta_teacher=liberta_teacher,
        liberta_distill_weight=liberta_distill_weight,
    )
    if ema_backup is not None:
        ExponentialMovingAverage.restore(model, ema_backup)

    val_metrics = validation["metrics"]
    log_payload = {
        "epoch": epoch,
        "global_step": global_step,
        **train_metrics,
        "val_loss": val_metrics["loss"],
        "val_cer": val_metrics["cer"],
        "val_wer": val_metrics["wer"],
    }
    for key, value in val_metrics.items():
        if key not in {"loss", "cer", "wer"}:
            log_payload[f"val_{key}"] = value
    log_payload.update(_flatten_examples("val_hardest", validation["hardest_examples"]))
    log_payload.update(_flatten_examples("val_random", validation["random_examples"]))
    trackio.log(log_payload)

    report = {
        "epoch": epoch,
        "global_step": global_step,
        "metrics": val_metrics,
        "hardest_examples": validation["hardest_examples"],
        "random_examples": validation["random_examples"],
        "speaker_metrics": validation["speaker_metrics"],
        "split_audit": split_audit,
    }
    reports_dir = output_dir / "eval_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{report_stem}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    updated_best_val_wer = min(best_val_wer, float(val_metrics["wer"]))
    checkpoint = _build_checkpoint(
        model=model,
        encoder_config=encoder_config,
        tokenizer=tokenizer,
        featurizer=featurizer,
        epoch=epoch,
        global_step=global_step,
        best_val_wer=updated_best_val_wer,
        metrics=log_payload,
        optimizers=optimizers,
        optimizer_names=optimizer_names,
        schedulers=schedulers,
        scaler=scaler,
        ema=ema,
        args=args,
    )

    if save_last_checkpoint:
        latest_path = output_dir / "checkpoint_last.pt"
    else:
        latest_path = output_dir / "checkpoint_step_last.pt"
    save_checkpoint(checkpoint, latest_path)
    latest_safetensors_path = _export_inference_checkpoint(checkpoint, latest_path)

    if float(val_metrics["wer"]) < best_val_wer:
        updated_best_val_wer = float(val_metrics["wer"])
        best_path = output_dir / "checkpoint_best.pt"
        save_checkpoint(checkpoint, best_path)
        best_safetensors_path = _export_inference_checkpoint(checkpoint, best_path)
    else:
        best_safetensors_path = _safetensors_path(output_dir / "checkpoint_best.pt")

    _update_top_checkpoints(
        output_dir=output_dir,
        checkpoint=checkpoint,
        epoch=epoch,
        global_step=global_step,
        val_wer=float(val_metrics["wer"]),
        keep_top_k=args.keep_top_k,
    )
    averaged_path = _average_topk_checkpoints(output_dir)
    logger.info(
        (
            "%s complete train_loss=%.4f val_loss=%.4f "
            "val_main_ctc_loss=%.4f val_intermediate_ctc_loss=%.4f "
            "val_combined_ctc_loss=%.4f val_aed_loss=%.4f "
            "val_liberta_distill_loss=%.4f val_cer=%.4f val_wer=%.4f "
            "best_val_wer=%.4f report=%s latest=%s best=%s averaged=%s"
        ),
        report_stem,
        float(train_metrics["train_loss"]),
        float(val_metrics["loss"]),
        float(val_metrics["main_ctc_loss"]),
        float(val_metrics["intermediate_ctc_loss"]),
        float(val_metrics["combined_ctc_loss"]),
        float(val_metrics["aed_loss"]),
        float(val_metrics["liberta_distill_loss"]),
        float(val_metrics["cer"]),
        float(val_metrics["wer"]),
        updated_best_val_wer,
        report_path,
        latest_path,
        output_dir / "checkpoint_best.pt",
        averaged_path if averaged_path is not None else "n/a",
    )
    logger.info(
        "exported inference artifacts latest_safe=%s best_safe=%s averaged_safe=%s",
        latest_safetensors_path,
        best_safetensors_path if best_safetensors_path.exists() else "n/a",
        _safetensors_path(averaged_path).as_posix() if averaged_path is not None else "n/a",
    )
    return updated_best_val_wer


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
) -> dict[str, object]:
    return {
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
    }


