from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from squeezeformer_pytorch.data import (
    AudioRecord,
    ShardedParquetFeatureCache,
    feature_cache_path,
    feature_tensor_is_plausible,
    load_audio,
)
from squeezeformer_pytorch.frontend import (
    build_featurizer_from_config,
    zipformer_paper_featurizer_config,
)
from squeezeformer_pytorch.runtime_types import FeatureCacheFormat
from squeezeformer_pytorch.training.cli import parse_args as parse_training_args
from squeezeformer_pytorch.training.data_loading import (
    DiskBackedRecordStore,
    _ensure_opus_decode_support,
    _load_train_val_records,
    _record_store_duration_hours,
    _resolve_dataset_sources,
    _resolve_validation_dataset_sources,
)
from squeezeformer_pytorch.training.runtime import _format_elapsed_seconds
from w2v_bert.asr import DEFAULT_W2V_BERT_MODEL, w2v_bert_featurizer_config

logger = logging.getLogger(__name__)


class FeatureCacheWarmDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[AudioRecord] | DiskBackedRecordStore,
        *,
        featurizer,
        feature_cache_dir: str | Path,
        feature_cache_format: str | FeatureCacheFormat,
        overwrite: bool = False,
        validate_existing: bool = False,
        write_cache: bool = True,
    ) -> None:
        self.records = records
        self.featurizer = featurizer
        self.feature_cache_dir = Path(feature_cache_dir)
        self.feature_cache_format = str(feature_cache_format)
        if self.feature_cache_format not in {"file", "parquet"}:
            raise ValueError(
                "feature_cache_format must be either 'file' or 'parquet', "
                f"got {feature_cache_format!r}"
            )
        self.feature_cache = (
            ShardedParquetFeatureCache(self.feature_cache_dir, commit_every=1)
            if self.feature_cache_format == "parquet"
            else None
        )
        self.overwrite = overwrite
        self.validate_existing = validate_existing
        self.write_cache = write_cache

    def __len__(self) -> int:
        return len(self.records)

    def close(self) -> None:
        if self.feature_cache is not None:
            self.feature_cache.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _cache_hit(self, record: AudioRecord) -> bool:
        if self.feature_cache is not None:
            features = self.feature_cache.load(record.utterance_id, self.featurizer)
            if features is None:
                return False
            if not self.validate_existing:
                return True
            if feature_tensor_is_plausible(
                record,
                features,
                expected_feature_bins=self.featurizer.n_mels,
            ):
                return True
            self.feature_cache.delete(record.utterance_id, self.featurizer)
            return False

        path = feature_cache_path(self.feature_cache_dir, record.utterance_id, self.featurizer)
        if path is None or not path.exists():
            return False
        if not self.validate_existing:
            return True
        features = torch.load(path, map_location="cpu")
        if feature_tensor_is_plausible(
            record, features, expected_feature_bins=self.featurizer.n_mels
        ):
            return True
        path.unlink(missing_ok=True)
        return False

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        started_at = time.perf_counter()
        try:
            if not self.overwrite and self._cache_hit(record):
                return {
                    "status": "hit",
                    "utterance_id": record.utterance_id,
                    "frames": int(record.estimated_frames),
                    "elapsed": time.perf_counter() - started_at,
                }

            waveform, sample_rate = load_audio(record.audio_path, record.audio_bytes)
            features = self.featurizer(waveform, sample_rate)
            if not feature_tensor_is_plausible(
                record,
                features,
                expected_feature_bins=self.featurizer.n_mels,
            ):
                return {
                    "status": "invalid",
                    "utterance_id": record.utterance_id,
                    "frames": int(getattr(features, "shape", [0])[0]),
                    "elapsed": time.perf_counter() - started_at,
                }

            if self.feature_cache is not None and self.write_cache:
                self.feature_cache.store(record.utterance_id, self.featurizer, features)
            elif self.feature_cache is not None:
                return {
                    "status": "written",
                    "utterance_id": record.utterance_id,
                    "features": features,
                    "frames": int(features.size(0)),
                    "elapsed": time.perf_counter() - started_at,
                }
            else:
                path = feature_cache_path(
                    self.feature_cache_dir, record.utterance_id, self.featurizer
                )
                if path is None:
                    raise RuntimeError("feature_cache_dir unexpectedly resolved to None")
                torch.save(features, path)
            return {
                "status": "written",
                "utterance_id": record.utterance_id,
                "frames": int(features.size(0)),
                "elapsed": time.perf_counter() - started_at,
            }
        except Exception as error:
            return {
                "status": "failed",
                "utterance_id": record.utterance_id,
                "error": str(error),
                "frames": 0,
                "elapsed": time.perf_counter() - started_at,
            }


def _collate_statuses(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return items


def _resolve_w2v_bert_model_source(args: argparse.Namespace) -> str:
    model_path = getattr(args, "w2v_bert_model_path", None)
    if model_path is not None:
        return str(Path(model_path).expanduser().resolve())
    return str(getattr(args, "w2v_bert_model_name", DEFAULT_W2V_BERT_MODEL))


def _resolve_featurizer_config(args: argparse.Namespace) -> dict[str, object]:
    if args.w2v_bert:
        return w2v_bert_featurizer_config(_resolve_w2v_bert_model_source(args))
    if args.zipformer:
        return zipformer_paper_featurizer_config()
    return {
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "backend": args.frontend_backend,
        "preemphasis": args.preemphasis,
        "normalize_signal": args.normalize_signal,
        "normalize_feature": args.normalize_feature,
        "normalize_per_frame": args.normalize_per_frame,
    }


def _add_warmer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-warm-split",
        default="train",
        help=(
            "Dataset split(s) to warm. Accepts 'train', 'validation', 'both', or a "
            "comma-separated list such as 'train,validation'."
        ),
    )
    parser.add_argument(
        "--cache-warm-workers",
        type=int,
        default=None,
        help="DataLoader workers used for offline cache warming. Defaults to --num-workers.",
    )
    parser.add_argument(
        "--cache-warm-batch-size",
        type=int,
        default=1,
        help="Number of utterances assigned per DataLoader batch during warming.",
    )
    parser.add_argument(
        "--cache-warm-log-interval",
        type=int,
        default=1000,
        help="Log cache warming progress every N records.",
    )
    parser.add_argument(
        "--cache-warm-wait-log-after",
        type=float,
        default=10.0,
        help="Log when waiting this many seconds for the next warmed item.",
    )
    parser.add_argument(
        "--cache-warm-wait-log-every",
        type=float,
        default=30.0,
        help="Repeat wait logs every N seconds while warming is blocked.",
    )
    parser.add_argument(
        "--cache-warm-timeout",
        type=float,
        default=0.0,
        help=(
            "DataLoader timeout in seconds while waiting for worker output. "
            "The default 0 disables timeout and only logs waits."
        ),
    )
    parser.add_argument(
        "--cache-warm-overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute and overwrite existing feature cache entries.",
    )
    parser.add_argument(
        "--cache-warm-validate-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load existing cache entries and validate their shape before skipping them.",
    )
    parser.add_argument(
        "--cache-warm-fail-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop after warming if any record failed.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute training feature cache entries. Accepts the same dataset/frontend/cache "
            "flags as train.py plus cache-warm-specific controls."
        )
    )
    _add_warmer_args(parser)
    warmer_args, training_argv = parser.parse_known_args(argv)
    training_args = parse_training_args(training_argv)
    for key, value in vars(warmer_args).items():
        setattr(training_args, key, value)
    if training_args.cache_warm_workers is not None and training_args.cache_warm_workers < 0:
        raise ValueError(
            f"--cache-warm-workers must be >= 0, got {training_args.cache_warm_workers}."
        )
    if training_args.cache_warm_batch_size <= 0:
        raise ValueError(
            f"--cache-warm-batch-size must be > 0, got {training_args.cache_warm_batch_size}."
        )
    if training_args.cache_warm_log_interval <= 0:
        raise ValueError(
            f"--cache-warm-log-interval must be > 0, got {training_args.cache_warm_log_interval}."
        )
    if training_args.cache_warm_wait_log_after < 0:
        raise ValueError(
            "--cache-warm-wait-log-after must be >= 0, "
            f"got {training_args.cache_warm_wait_log_after}."
        )
    if training_args.cache_warm_wait_log_every <= 0:
        raise ValueError(
            "--cache-warm-wait-log-every must be > 0, "
            f"got {training_args.cache_warm_wait_log_every}."
        )
    if training_args.cache_warm_timeout < 0:
        raise ValueError(
            f"--cache-warm-timeout must be >= 0, got {training_args.cache_warm_timeout}."
        )
    return training_args


def _resolve_cache_warm_splits(raw_value: str) -> set[str]:
    values = {value.strip().lower() for value in str(raw_value).split(",") if value.strip()}
    if not values:
        raise ValueError("--cache-warm-split must include at least one split.")
    if "both" in values:
        values.remove("both")
        values.update({"train", "validation"})
    invalid = values - {"train", "validation"}
    if invalid:
        raise ValueError(
            "--cache-warm-split must contain only 'train', 'validation', or 'both'; "
            f"got {', '.join(sorted(invalid))}."
        )
    return values


def _log_progress(
    *,
    split: str,
    processed: int,
    total: int,
    counts: dict[str, int],
    frames: int,
    started_at: float,
) -> None:
    elapsed = time.perf_counter() - started_at
    rate = processed / max(elapsed, 1e-9)
    percent = (processed / total * 100.0) if total else 100.0
    logger.info(
        "%s feature cache warm progress=%s/%s %.1f%% rate=%.1f/s written=%s hit=%s "
        "invalid=%s failed=%s frames=%s elapsed=%s",
        split,
        processed,
        total,
        percent,
        rate,
        counts.get("written", 0),
        counts.get("hit", 0),
        counts.get("invalid", 0),
        counts.get("failed", 0),
        frames,
        _format_elapsed_seconds(elapsed),
    )


def _record_wait_label(records, index: int) -> str:
    if not 0 <= index < len(records):
        return f"index={index}"
    try:
        record = records[index]
    except Exception:
        return f"index={index}"
    audio_path = str(record.audio_path) if record.audio_path is not None else "<bytes>"
    return f"index={index} utterance_id={record.utterance_id} audio={audio_path}"


def _next_with_wait_logging(
    iterator,
    *,
    description: str,
    log_after_seconds: float,
    log_every_seconds: float,
):
    started_at = time.perf_counter()
    if log_after_seconds <= 0:
        item = next(iterator)
        return item, time.perf_counter() - started_at

    done = threading.Event()

    def watchdog() -> None:
        if done.wait(log_after_seconds):
            return
        while not done.is_set():
            logger.info(
                "%s still waiting elapsed=%s",
                description,
                _format_elapsed_seconds(time.perf_counter() - started_at),
            )
            if done.wait(log_every_seconds):
                return

    thread = threading.Thread(target=watchdog, name="feature-cache-warm-watchdog", daemon=True)
    thread.start()
    try:
        item = next(iterator)
        return item, time.perf_counter() - started_at
    finally:
        done.set()


def _warm_split(
    split: str,
    records,
    *,
    args: argparse.Namespace,
    featurizer,
    cache_dir: Path,
) -> dict[str, int]:
    workers = args.num_workers if args.cache_warm_workers is None else args.cache_warm_workers
    main_feature_cache = (
        ShardedParquetFeatureCache(cache_dir)
        if workers > 0 and str(args.feature_cache_format) == "parquet"
        else None
    )
    dataset = FeatureCacheWarmDataset(
        records,
        featurizer=featurizer,
        feature_cache_dir=cache_dir,
        feature_cache_format=args.feature_cache_format,
        overwrite=args.cache_warm_overwrite,
        validate_existing=args.cache_warm_validate_existing,
        write_cache=main_feature_cache is None,
    )
    dataloader_kwargs: dict[str, object] = {}
    if workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
        dataloader_kwargs["persistent_workers"] = False
        dataloader_kwargs["in_order"] = args.dataloader_in_order
        if args.cache_warm_timeout > 0:
            dataloader_kwargs["timeout"] = args.cache_warm_timeout
        if args.dataloader_mp_context != "auto":
            dataloader_kwargs["multiprocessing_context"] = args.dataloader_mp_context
    loader = DataLoader(
        dataset,
        batch_size=args.cache_warm_batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=_collate_statuses,
        **dataloader_kwargs,
    )
    logger.info(
        "%s feature cache warm started records=%s hours=%.2f cache_dir=%s format=%s "
        "workers=%s batch_size=%s prefetch_factor=%s in_order=%s timeout=%s "
        "overwrite=%s validate_existing=%s",
        split,
        len(records),
        _record_store_duration_hours(records, hop_length=featurizer.hop_length),
        cache_dir,
        args.feature_cache_format,
        workers,
        args.cache_warm_batch_size,
        args.prefetch_factor if workers > 0 else "none",
        args.dataloader_in_order if workers > 0 else "none",
        args.cache_warm_timeout if workers > 0 else "none",
        args.cache_warm_overwrite,
        args.cache_warm_validate_existing,
    )
    started_at = time.perf_counter()
    counts: dict[str, int] = {"written": 0, "hit": 0, "invalid": 0, "failed": 0}
    frames = 0
    processed = 0
    log_interval = max(1, int(args.cache_warm_log_interval))
    iterator = iter(loader)
    while processed < len(records):
        next_index = processed
        description = (
            f"{split} feature cache warm waiting for item {next_index + 1}/{len(records)} "
            f"workers={workers} prefetch_factor={args.prefetch_factor if workers > 0 else 'none'} "
            f"in_order={args.dataloader_in_order if workers > 0 else 'none'} "
            f"next={_record_wait_label(records, next_index)}"
        )
        try:
            batch, wait_seconds = _next_with_wait_logging(
                iterator,
                description=description,
                log_after_seconds=args.cache_warm_wait_log_after,
                log_every_seconds=args.cache_warm_wait_log_every,
            )
        except StopIteration:
            break
        if wait_seconds >= args.cache_warm_wait_log_after > 0:
            logger.info(
                "%s feature cache warm item ready item=%s/%s wait=%s",
                split,
                next_index + 1,
                len(records),
                _format_elapsed_seconds(wait_seconds),
            )
        for item in batch:
            status = str(item["status"])
            counts[status] = counts.get(status, 0) + 1
            frames += int(item.get("frames", 0))
            processed += 1
            if status == "written" and main_feature_cache is not None:
                main_feature_cache.store(
                    str(item["utterance_id"]),
                    featurizer,
                    item["features"],
                )
            if status == "failed":
                logger.warning(
                    "%s feature cache warm failed utterance_id=%s error=%s",
                    split,
                    item.get("utterance_id"),
                    item.get("error"),
                )
        if processed % log_interval == 0 or processed >= len(records):
            _log_progress(
                split=split,
                processed=processed,
                total=len(records),
                counts=counts,
                frames=frames,
                started_at=started_at,
            )
    dataset.close()
    if main_feature_cache is not None:
        main_feature_cache.close()
    logger.info(
        "%s feature cache warm complete records=%s written=%s hit=%s invalid=%s failed=%s "
        "elapsed=%s",
        split,
        processed,
        counts.get("written", 0),
        counts.get("hit", 0),
        counts.get("invalid", 0),
        counts.get("failed", 0),
        _format_elapsed_seconds(time.perf_counter() - started_at),
    )
    return counts


def _accumulate_counts(total_counts: dict[str, int], counts: dict[str, int]) -> None:
    for key, value in counts.items():
        total_counts[key] = total_counts.get(key, 0) + value


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args(argv)
    if args.feature_cache_dir is None:
        raise SystemExit("--feature-cache-dir is required for offline cache warming.")
    cache_warm_splits = _resolve_cache_warm_splits(args.cache_warm_split)
    ordered_splits = [split for split in ("train", "validation") if split in cache_warm_splits]

    train_sources = _resolve_dataset_sources(args)
    val_sources = _resolve_validation_dataset_sources(args)
    lowercase_transcripts = args.tokenizer != "sentencepiece"
    output_dir = Path(args.output_dir)
    args.require_readable_audio = True
    logger.info(
        "loading records train_sources=%s validation_sources=%s record_cache=%s",
        train_sources,
        val_sources,
        args.record_cache,
    )
    train_records, val_records = _load_train_val_records(
        args,
        train_sources,
        val_sources,
        lowercase_transcripts=lowercase_transcripts,
        output_dir=output_dir,
    )
    if "train" in cache_warm_splits:
        _ensure_opus_decode_support(train_records, split="train")
    if "validation" in cache_warm_splits:
        _ensure_opus_decode_support(val_records, split="validation")

    featurizer = build_featurizer_from_config(
        _resolve_featurizer_config(args),
        use_zipformer=args.zipformer,
        use_w2v_bert=args.w2v_bert,
    )
    cache_root = Path(args.feature_cache_dir)
    total_counts: dict[str, int] = {"written": 0, "hit": 0, "invalid": 0, "failed": 0}
    logger.info(
        "feature cache warm plan splits=%s total_splits=%s cache_root=%s format=%s",
        ",".join(ordered_splits),
        len(ordered_splits),
        cache_root,
        args.feature_cache_format,
    )
    for split_index, split in enumerate(ordered_splits, start=1):
        split_records = train_records if split == "train" else val_records
        logger.info(
            "feature cache warm split %s/%s starting split=%s records=%s",
            split_index,
            len(ordered_splits),
            split,
            len(split_records),
        )
        counts = _warm_split(
            split,
            split_records,
            args=args,
            featurizer=featurizer,
            cache_dir=cache_root / split,
        )
        _accumulate_counts(total_counts, counts)
        logger.info(
            "feature cache warm split %s/%s complete split=%s written=%s hit=%s invalid=%s failed=%s",
            split_index,
            len(ordered_splits),
            split,
            counts.get("written", 0),
            counts.get("hit", 0),
            counts.get("invalid", 0),
            counts.get("failed", 0),
        )
    logger.info(
        "feature cache warm all requested splits complete splits=%s written=%s hit=%s invalid=%s failed=%s",
        ",".join(ordered_splits),
        total_counts.get("written", 0),
        total_counts.get("hit", 0),
        total_counts.get("invalid", 0),
        total_counts.get("failed", 0),
    )
    if args.cache_warm_fail_on_error and total_counts.get("failed", 0) > 0:
        raise SystemExit(f"Feature cache warming failed for {total_counts['failed']} record(s).")


if __name__ == "__main__":
    main(sys.argv[1:])
