from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

try:
    import trackio
except ImportError:
    trackio = None

from squeezeformer_pytorch.asr import (
    CharacterTokenizer,
    SentencePieceTokenizer,
    Tokenizer,
    tokenizer_from_dict,
)
from squeezeformer_pytorch.data import (
    AudioFeaturizer,
    CV22ASRDataset,
    SpecAugment,
    WaveformAugment,
    create_dataloader,
    download_cv22_dataset,
    load_cv22_records,
    prevalidate_records,
)
from squeezeformer_pytorch.metrics import char_error_rate, word_error_rate
from squeezeformer_pytorch.runtime_types import AdaptiveBatchUnit
from squeezeformer_pytorch.secrets import sanitize_for_serialization

from .model import SqueezeformerConfig, SqueezeformerCTC, squeezeformer_variant
from .training import (
    TrainState,
    create_train_state,
    replicate_state,
    shard_batch,
    train_step,
    unreplicate_state,
)


def _configure_console_logger() -> logging.Logger:
    logger = logging.getLogger("train_jax")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def _resolve_dataloader_settings(
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    logger: logging.Logger,
) -> tuple[int, bool, bool]:
    resolved_num_workers = 0
    resolved_pin_memory = False
    resolved_persistent_workers = False
    if num_workers != 0:
        logger.warning(
            "forcing --num-workers=0 for the JAX path because PyTorch DataLoader "
            "worker forking can deadlock with JAX's multithreaded runtime"
        )
    if pin_memory:
        logger.warning(
            "forcing --pin-memory=false for the JAX path because batches are copied "
            "into JAX arrays rather than moved through PyTorch accelerator memory"
        )
    if persistent_workers:
        logger.warning(
            "forcing --persistent-workers=false because JAX training uses single-process "
            "data loading in this script"
        )
    return resolved_num_workers, resolved_pin_memory, resolved_persistent_workers


def _variant_peak_lr(variant: str) -> float:
    if variant in {"xs", "s", "sm"}:
        return 2e-3
    if variant == "m":
        return 1.5e-3
    return 1e-3


def _variant_num_time_masks(variant: str) -> int:
    if variant in {"xs", "s", "sm"}:
        return 5
    if variant == "m":
        return 7
    return 10


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


def _flatten_examples(prefix: str, examples: list[dict[str, str]]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for index, example in enumerate(examples):
        payload[f"{prefix}_example_{index}_id"] = example["utterance_id"]
        payload[f"{prefix}_example_{index}_speaker"] = example["speaker_id"]
        payload[f"{prefix}_example_{index}_ref"] = example["reference"]
        payload[f"{prefix}_example_{index}_hyp"] = example["hypothesis"]
    return payload


def _bucket_name(reference: str) -> str:
    word_count = len(reference.split())
    if word_count <= 5:
        return "short"
    if word_count <= 15:
        return "medium"
    return "long"


def _length_bucket_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for bucket in ("short", "medium", "long"):
        bucket_refs = [reference for reference in references if _bucket_name(reference) == bucket]
        bucket_hyps = [
            hypothesis
            for reference, hypothesis in zip(references, hypotheses, strict=True)
            if _bucket_name(reference) == bucket
        ]
        metrics[f"samples_{bucket}"] = float(len(bucket_refs))
        if bucket_refs:
            metrics[f"wer_{bucket}"] = word_error_rate(bucket_refs, bucket_hyps)
            metrics[f"cer_{bucket}"] = char_error_rate(bucket_refs, bucket_hyps)
    return metrics


def _collect_examples(
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


def _paper_schedule(
    *,
    steps_per_epoch: int,
    peak_lr: float,
    warmup_epochs: int,
    hold_epochs: int,
    decay_exponent: float,
) -> optax.Schedule:
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    hold_steps = max(0, hold_epochs * steps_per_epoch)

    def schedule(step: jax.Array) -> jax.Array:
        current_step = step + 1
        warmup = current_step / warmup_steps
        decay_step = jnp.maximum(1, current_step - hold_steps)
        decay = (warmup_steps**decay_exponent) / (decay_step**decay_exponent)
        multiplier = jnp.where(
            current_step < warmup_steps,
            warmup,
            jnp.where(current_step < warmup_steps + hold_steps, 1.0, decay),
        )
        return peak_lr * multiplier

    return schedule


def _create_optimizer(
    *,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    steps_per_epoch: int,
    warmup_epochs: int,
    hold_epochs: int,
    decay_exponent: float,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    schedule = _paper_schedule(
        steps_per_epoch=steps_per_epoch,
        peak_lr=learning_rate,
        warmup_epochs=warmup_epochs,
        hold_epochs=hold_epochs,
        decay_exponent=decay_exponent,
    )
    transforms: list[optax.GradientTransformation] = []
    if grad_clip_norm > 0:
        transforms.append(optax.clip_by_global_norm(grad_clip_norm))
    transforms.append(optax.adamw(learning_rate=schedule, weight_decay=weight_decay))
    return optax.chain(*transforms), schedule


def _flatten_targets(targets: torch.Tensor, target_lengths: torch.Tensor) -> jnp.ndarray:
    labels = []
    offset = 0
    max_length = int(target_lengths.max().item())
    for length in target_lengths.tolist():
        current = targets[offset : offset + length]
        offset += length
        if current.numel() < max_length:
            current = torch.nn.functional.pad(current, (0, max_length - current.numel()))
        labels.append(current)
    return jnp.asarray(torch.stack(labels, dim=0).cpu().numpy(), dtype=jnp.int32)


def _to_jax_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "features": jnp.asarray(batch["features"].cpu().numpy(), dtype=jnp.float32),
        "feature_lengths": jnp.asarray(batch["feature_lengths"].cpu().numpy(), dtype=jnp.int32),
        "labels": _flatten_targets(batch["targets"], batch["target_lengths"]),
        "label_lengths": jnp.asarray(batch["target_lengths"].cpu().numpy(), dtype=jnp.int32),
        "transcripts": batch["transcripts"],
        "utterance_ids": batch["utterance_ids"],
        "speaker_ids": batch["speaker_ids"],
    }


def _pad_batch_for_devices(batch: dict[str, Any], device_count: int) -> tuple[dict[str, Any], int]:
    batch_size = int(batch["features"].shape[0])
    remainder = batch_size % device_count
    if remainder == 0:
        return batch, batch_size
    pad = device_count - remainder

    def _pad_array(x: jnp.ndarray) -> jnp.ndarray:
        tail = jnp.repeat(x[-1:], repeats=pad, axis=0)
        return jnp.concatenate([x, tail], axis=0)

    padded = dict(batch)
    for key in ("features", "feature_lengths", "labels", "label_lengths"):
        padded[key] = _pad_array(batch[key])
    for key in ("transcripts", "utterance_ids", "speaker_ids"):
        padded[key] = list(batch[key]) + [batch[key][-1]] * pad
    return padded, batch_size


def _save_checkpoint(path: Path, checkpoint: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(checkpoint, handle)
    metadata = sanitize_for_serialization(
        {
            key: value
            for key, value in checkpoint.items()
            if key not in {"params", "batch_stats", "opt_state"}
        }
    )
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_checkpoint(path: str | Path) -> dict[str, object]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _tree_to_numpy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value)), tree)


def _greedy_decode(
    logits: jnp.ndarray,
    output_lengths: jnp.ndarray,
    tokenizer: Tokenizer,
) -> list[str]:
    token_ids = np.asarray(jnp.argmax(logits, axis=-1))
    lengths = np.asarray(output_lengths)
    return [
        tokenizer.decode_ctc(sequence[:length].tolist())
        for sequence, length in zip(token_ids, lengths, strict=True)
    ]


def _evaluate(
    state: TrainState,
    model: SqueezeformerCTC,
    dataloader,
    tokenizer: Tokenizer,
    *,
    blank_id: int,
    example_limit: int,
) -> dict[str, object]:
    total_loss = 0.0
    total_batches = 0
    references: list[str] = []
    hypotheses: list[str] = []
    utterance_ids: list[str] = []
    speaker_ids: list[str | None] = []

    infer_state = (
        unreplicate_state(state)
        if hasattr(state, "params") and jax.local_device_count() > 1
        else state
    )

    for batch in dataloader:
        jax_batch = _to_jax_batch(batch)
        logits, output_lengths = model.apply(
            {"params": infer_state.params, "batch_stats": infer_state.batch_stats},
            jax_batch["features"],
            jax_batch["feature_lengths"],
            deterministic=True,
        )
        logit_paddings = (jnp.arange(logits.shape[1])[None, :] >= output_lengths[:, None]).astype(
            jnp.float32
        )
        label_paddings = (
            jnp.arange(jax_batch["labels"].shape[1])[None, :] >= jax_batch["label_lengths"][:, None]
        ).astype(jnp.float32)
        loss = optax.ctc_loss(
            logits=logits,
            logit_paddings=logit_paddings,
            labels=jax_batch["labels"],
            label_paddings=label_paddings,
            blank_id=blank_id,
        ).mean()
        total_loss += float(jax.device_get(loss))
        total_batches += 1
        batch_hypotheses = _greedy_decode(logits, output_lengths, tokenizer)
        references.extend(jax_batch["transcripts"])
        hypotheses.extend(batch_hypotheses)
        utterance_ids.extend(jax_batch["utterance_ids"])
        speaker_ids.extend(jax_batch["speaker_ids"])

    metrics = {
        "loss": total_loss / max(1, total_batches),
        "cer": char_error_rate(references, hypotheses),
        "wer": word_error_rate(references, hypotheses),
    }
    metrics.update(_length_bucket_metrics(references, hypotheses))
    hardest_examples, random_examples = _collect_examples(
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JAX Squeezeformer CTC on speech-uk/cv22.")
    parser.add_argument("--dataset-repo", default="speech-uk/cv22")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--variant", default="sm", choices=["xs", "s", "sm", "m", "ml", "l"])
    parser.add_argument("--output-dir", default="artifacts/squeezeformer-cv22-jax")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--min-transcript-chars", type=int, default=1)
    parser.add_argument("--max-transcript-chars", type=int, default=400)
    parser.add_argument("--max-symbol-ratio", type=float, default=0.5)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--max-batch-frames", type=int, default=None)
    parser.add_argument(
        "--adaptive-batch-unit",
        type=AdaptiveBatchUnit,
        choices=list(AdaptiveBatchUnit),
        default=None,
    )
    parser.add_argument("--adaptive-batch-budget", type=int, default=None)
    parser.add_argument(
        "--bucket-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--metadata-workers", type=int, default=4)
    parser.add_argument(
        "--prevalidate-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prevalidate-workers", type=int, default=4)
    parser.add_argument("--trackio-project", default="squeezeformer-cv22-jax")
    parser.add_argument("--trackio-space-id", default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--tokenizer",
        default="sentencepiece",
        choices=["character", "sentencepiece"],
    )
    parser.add_argument("--spm-vocab-size", type=int, default=4096)
    parser.add_argument(
        "--spm-model-type",
        default="unigram",
        choices=["unigram", "bpe", "char", "word"],
    )
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--hold-epochs", type=int, default=160)
    parser.add_argument("--decay-exponent", type=float, default=1.0)
    parser.add_argument(
        "--intermediate-ctc",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--blank-prune",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--block-pattern", default="M,s,C,s")
    parser.add_argument(
        "--attention-backend",
        default="relative",
        choices=["relative", "flash"],
    )
    parser.add_argument(
        "--frontend-backend",
        default="audioflux",
        choices=["torchaudio", "audioflux"],
    )
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--preemphasis", type=float, default=0.97)
    parser.add_argument(
        "--normalize-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize-feature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize-per-frame",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--num-freq-masks", type=int, default=2)
    parser.add_argument("--freq-mask-param", type=int, default=27)
    parser.add_argument("--num-time-masks", type=int, default=None)
    parser.add_argument("--time-mask-max-ratio", type=float, default=0.05)
    parser.add_argument("--speed-perturb-prob", type=float, default=0.0)
    parser.add_argument("--speed-factors", default="0.9,1.0,1.1")
    parser.add_argument("--noise-prob", type=float, default=0.0)
    parser.add_argument("--noise-snr-db-min", type=float, default=10.0)
    parser.add_argument("--noise-snr-db-max", type=float, default=30.0)
    parser.add_argument("--reverb-prob", type=float, default=0.0)
    parser.add_argument("--reverb-decay-min", type=float, default=0.15)
    parser.add_argument("--reverb-decay-max", type=float, default=0.5)
    parser.add_argument("--reverb-delay-ms-min", type=float, default=8.0)
    parser.add_argument("--reverb-delay-ms-max", type=float, default=35.0)
    parser.add_argument("--example-limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.adaptive_batch_unit is None) != (args.adaptive_batch_budget is None):
        raise ValueError("--adaptive-batch-unit and --adaptive-batch-budget must be set together.")
    if args.intermediate_ctc:
        raise ValueError(
            "The JAX trainer does not implement intermediate CTC yet. "
            "Use --no-intermediate-ctc or the PyTorch trainer."
        )
    if args.blank_prune:
        raise ValueError(
            "The JAX trainer does not implement blank-driven frame pruning yet. "
            "Use --no-blank-prune or the PyTorch trainer."
        )

    logger = _configure_console_logger()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "starting jax training variant=%s devices=%s backend=%s output_dir=%s",
        args.variant,
        jax.local_device_count(),
        jax.default_backend(),
        output_dir,
    )
    args.num_workers, args.pin_memory, args.persistent_workers = _resolve_dataloader_settings(
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        logger=logger,
    )

    dataset_root = download_cv22_dataset(
        repo_id=args.dataset_repo,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    lowercase_transcripts = args.tokenizer != "sentencepiece"
    train_records = load_cv22_records(
        dataset_root=dataset_root,
        split="train",
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        max_samples=args.max_train_samples,
        min_transcript_chars=args.min_transcript_chars,
        max_transcript_chars=args.max_transcript_chars,
        max_symbol_ratio=args.max_symbol_ratio,
        lowercase_transcripts=lowercase_transcripts,
    )
    val_records = load_cv22_records(
        dataset_root=dataset_root,
        split="validation",
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        max_samples=args.max_val_samples,
        min_transcript_chars=args.min_transcript_chars,
        max_transcript_chars=args.max_transcript_chars,
        max_symbol_ratio=args.max_symbol_ratio,
        lowercase_transcripts=lowercase_transcripts,
    )
    if args.prevalidate_audio:
        train_records = prevalidate_records(train_records, num_workers=args.prevalidate_workers)
        val_records = prevalidate_records(val_records, num_workers=args.prevalidate_workers)
        if not train_records or not val_records:
            raise RuntimeError("Audio prevalidation removed every sample from train or validation.")

    checkpoint = _load_checkpoint(args.resume) if args.resume else None
    if checkpoint is not None:
        tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    elif args.tokenizer == "sentencepiece":
        tokenizer = SentencePieceTokenizer.train(
            (record.transcript for record in train_records),
            model_prefix=output_dir / "tokenizer",
            vocab_size=args.spm_vocab_size,
            model_type=args.spm_model_type,
        )
        tokenizer.save(output_dir / "tokenizer.model")
    else:
        tokenizer = CharacterTokenizer.build(record.transcript for record in train_records)
    tokenizer.save(output_dir / "tokenizer.json")

    featurizer = AudioFeaturizer(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        backend=args.frontend_backend,
        preemphasis=args.preemphasis,
        normalize_signal=args.normalize_signal,
        normalize_feature=args.normalize_feature,
        normalize_per_frame=args.normalize_per_frame,
    )
    specaugment = SpecAugment(
        num_freq_masks=args.num_freq_masks,
        freq_mask_param=args.freq_mask_param,
        num_time_masks=args.num_time_masks or _variant_num_time_masks(args.variant),
        time_mask_max_ratio=args.time_mask_max_ratio,
    )
    waveform_augment = WaveformAugment(
        speed_perturb_prob=args.speed_perturb_prob,
        speed_factors=_resolve_float_tuple(args.speed_factors),
        noise_prob=args.noise_prob,
        noise_snr_db_range=(args.noise_snr_db_min, args.noise_snr_db_max),
        reverb_prob=args.reverb_prob,
        reverb_decay_range=(args.reverb_decay_min, args.reverb_decay_max),
        reverb_delay_ms_range=(args.reverb_delay_ms_min, args.reverb_delay_ms_max),
    )
    train_feature_cache_dir = (
        Path(args.feature_cache_dir) / "train" if args.feature_cache_dir is not None else None
    )
    val_feature_cache_dir = (
        Path(args.feature_cache_dir) / "validation" if args.feature_cache_dir is not None else None
    )
    train_dataset = CV22ASRDataset(
        train_records,
        tokenizer=tokenizer,
        featurizer=featurizer,
        specaugment=specaugment,
        waveform_augment=waveform_augment,
        feature_cache_dir=train_feature_cache_dir,
    )
    val_dataset = CV22ASRDataset(
        val_records,
        tokenizer=tokenizer,
        featurizer=featurizer,
        feature_cache_dir=val_feature_cache_dir,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        bucket_by_length=args.bucket_by_length,
        max_batch_frames=args.max_batch_frames,
        adaptive_batch_unit=args.adaptive_batch_unit,
        adaptive_batch_budget=args.adaptive_batch_budget,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        metadata_workers=args.metadata_workers,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        bucket_by_length=args.bucket_by_length,
        max_batch_frames=args.max_batch_frames,
        adaptive_batch_unit=args.adaptive_batch_unit,
        adaptive_batch_budget=args.adaptive_batch_budget,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        metadata_workers=args.metadata_workers,
    )

    encoder_config = (
        SqueezeformerConfig(**checkpoint["encoder_config"])
        if checkpoint is not None
        else replace(
            squeezeformer_variant(args.variant),
            block_pattern=_resolve_block_pattern(args.block_pattern),
            attention_backend=args.attention_backend,
        )
    )
    model = SqueezeformerCTC(
        encoder_config=encoder_config,
        vocab_size=tokenizer.vocab_size,
    )
    steps_per_epoch = max(1, len(train_loader))
    learning_rate = (
        args.learning_rate if args.learning_rate is not None else _variant_peak_lr(args.variant)
    )
    tx, lr_schedule = _create_optimizer(
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        hold_epochs=args.hold_epochs,
        decay_exponent=args.decay_exponent,
    )

    example_batch = next(iter(train_loader))
    example_features = jnp.asarray(example_batch["features"].cpu().numpy(), dtype=jnp.float32)
    example_feature_lengths = jnp.asarray(
        example_batch["feature_lengths"].cpu().numpy(),
        dtype=jnp.int32,
    )
    rng = jax.random.PRNGKey(args.seed)
    state = create_train_state(
        rng,
        model,
        example_features=example_features,
        example_feature_lengths=example_feature_lengths,
        tx=tx,
    )
    best_val_wer = float("inf")
    start_epoch = 1
    global_step = 0

    if checkpoint is not None:
        state = state.replace(
            params=checkpoint["params"],
            batch_stats=checkpoint.get("batch_stats", state.batch_stats),
            opt_state=checkpoint["opt_state"],
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_wer = float(checkpoint.get("best_val_wer", best_val_wer))
        logger.info(
            "resumed from %s starting_epoch=%s global_step=%s best_val_wer=%.4f",
            args.resume,
            start_epoch,
            global_step,
            best_val_wer,
        )

    device_count = jax.local_device_count()
    use_pmap = device_count > 1
    if use_pmap:
        state = replicate_state(state)
        compiled_train_step = jax.pmap(
            partial(train_step, blank_id=tokenizer.blank_id, axis_name="batch"),
            axis_name="batch",
        )
    else:
        compiled_train_step = jax.jit(partial(train_step, blank_id=tokenizer.blank_id))

    if trackio is not None:
        trackio.init(
            project=args.trackio_project,
            space_id=args.trackio_space_id,
            config={
                **vars(args),
                "backend": jax.default_backend(),
                "local_device_count": device_count,
                "encoder_config": asdict(encoder_config),
                "train_samples": len(train_records),
                "val_samples": len(val_records),
                "featurizer_config": featurizer.config_dict(),
            },
        )

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info("epoch %s/%s started", epoch, args.epochs)
        running_loss = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            host_batch = _to_jax_batch(batch)
            if use_pmap:
                host_batch, original_size = _pad_batch_for_devices(host_batch, device_count)
                sharded = {
                    key: value
                    for key, value in host_batch.items()
                    if key in {"features", "feature_lengths", "labels", "label_lengths"}
                }
                sharded = shard_batch(sharded)
                rng, step_rng = jax.random.split(rng)
                dropout_rng = jax.random.split(step_rng, device_count)
                state, metrics = compiled_train_step(state, sharded, dropout_rng=dropout_rng)
                del original_size
                metrics = jax.tree_util.tree_map(
                    lambda value: float(np.asarray(jax.device_get(value)).mean()),
                    metrics,
                )
            else:
                single_batch = {
                    key: value
                    for key, value in host_batch.items()
                    if key in {"features", "feature_lengths", "labels", "label_lengths"}
                }
                rng, step_rng = jax.random.split(rng)
                state, metrics = compiled_train_step(state, single_batch, dropout_rng=step_rng)
                metrics = {
                    key: float(np.asarray(jax.device_get(value))) for key, value in metrics.items()
                }

            running_loss += metrics["loss"]
            global_step += 1
            if global_step % args.log_every == 0:
                lr = float(jax.device_get(lr_schedule(jnp.asarray(max(0, global_step - 1)))))
                logger.info(
                    "epoch=%s step=%s/%s global_step=%s train_loss=%.4f lr=%.6g",
                    epoch,
                    batch_index,
                    len(train_loader),
                    global_step,
                    metrics["loss"],
                    lr,
                )
                if trackio is not None:
                    trackio.log(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "train_loss_step": metrics["loss"],
                            "learning_rate": lr,
                        }
                    )

        train_loss = running_loss / max(1, len(train_loader))
        validation = _evaluate(
            state,
            model,
            val_loader,
            tokenizer,
            blank_id=tokenizer.blank_id,
            example_limit=args.example_limit,
        )
        val_metrics = validation["metrics"]
        report = {
            "epoch": epoch,
            "global_step": global_step,
            "metrics": val_metrics,
            "hardest_examples": validation["hardest_examples"],
            "random_examples": validation["random_examples"],
        }
        reports_dir = output_dir / "eval_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"epoch_{epoch:04d}.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        checkpoint_state = unreplicate_state(state) if use_pmap else state
        checkpoint_payload = {
            "params": _tree_to_numpy(checkpoint_state.params),
            "batch_stats": _tree_to_numpy(checkpoint_state.batch_stats),
            "opt_state": _tree_to_numpy(checkpoint_state.opt_state),
            "encoder_config": asdict(encoder_config),
            "tokenizer": tokenizer.to_dict(),
            "featurizer_config": featurizer.config_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_wer": min(best_val_wer, float(val_metrics["wer"])),
            "metrics": {
                "train_loss": train_loss,
                "val_loss": float(val_metrics["loss"]),
                "val_cer": float(val_metrics["cer"]),
                "val_wer": float(val_metrics["wer"]),
            },
            "training_args": vars(args),
        }
        latest_path = output_dir / "checkpoint_last.pkl"
        _save_checkpoint(latest_path, checkpoint_payload)
        if val_metrics["wer"] < best_val_wer:
            best_val_wer = float(val_metrics["wer"])
            _save_checkpoint(output_dir / "checkpoint_best.pkl", checkpoint_payload)

        logger.info(
            (
                "epoch %s complete train_loss=%.4f val_loss=%.4f val_cer=%.4f "
                "val_wer=%.4f best_val_wer=%.4f report=%s"
            ),
            epoch,
            train_loss,
            float(val_metrics["loss"]),
            float(val_metrics["cer"]),
            float(val_metrics["wer"]),
            best_val_wer,
            report_path,
        )

        if trackio is not None:
            log_payload = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": float(val_metrics["loss"]),
                "val_cer": float(val_metrics["cer"]),
                "val_wer": float(val_metrics["wer"]),
            }
            for key, value in val_metrics.items():
                if key not in {"loss", "cer", "wer"}:
                    log_payload[f"val_{key}"] = value
            log_payload.update(_flatten_examples("val_hardest", validation["hardest_examples"]))
            log_payload.update(_flatten_examples("val_random", validation["random_examples"]))
            trackio.log(log_payload)

    (output_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "best_val_wer": best_val_wer,
                "variant": args.variant,
                "backend": jax.default_backend(),
                "local_device_count": device_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if trackio is not None:
        trackio.finish()
    logger.info(
        "training finished epochs=%s best_val_wer=%.4f summary=%s",
        args.epochs,
        best_val_wer,
        output_dir / "train_summary.json",
    )


if __name__ == "__main__":
    main()
