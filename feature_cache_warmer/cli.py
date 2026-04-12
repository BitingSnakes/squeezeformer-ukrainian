from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
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
        record_timeout_seconds: float = 0.0,
        skipped_audio_sources: set[str] | None = None,
        indices: list[int] | None = None,
        ffprobe_timeout_seconds: float = 0.0,
        ffmpeg_timeout_seconds: float = 0.0,
    ) -> None:
        self.records = records
        self.indices = indices
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
        self.record_timeout_seconds = max(0.0, float(record_timeout_seconds))
        self.skipped_audio_sources = skipped_audio_sources or set()
        self.ffprobe_timeout_seconds = max(0.0, float(ffprobe_timeout_seconds))
        self.ffmpeg_timeout_seconds = max(0.0, float(ffmpeg_timeout_seconds))

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.records)

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
        record_index = self.indices[index] if self.indices is not None else index
        record = self.records[record_index]
        started_at = time.perf_counter()
        try:
            if _record_matches_skip_list(record, self.skipped_audio_sources):
                return {
                    "status": "skipped",
                    "index": record_index,
                    "utterance_id": record.utterance_id,
                    "audio_path": record.audio_path,
                    "audio_source": _record_audio_source(record),
                    "frames": int(record.estimated_frames),
                    "elapsed": time.perf_counter() - started_at,
                }
            with _record_timeout(self.record_timeout_seconds):
                if not self.overwrite and self._cache_hit(record):
                    return {
                        "status": "hit",
                        "index": record_index,
                        "utterance_id": record.utterance_id,
                        "audio_path": record.audio_path,
                        "audio_source": _record_audio_source(record),
                        "frames": int(record.estimated_frames),
                        "elapsed": time.perf_counter() - started_at,
                    }

                ffprobe_error = _ffprobe_audio_source(record, self.ffprobe_timeout_seconds)
                if ffprobe_error is not None:
                    return {
                        "status": "failed",
                        "index": record_index,
                        "utterance_id": record.utterance_id,
                        "audio_path": record.audio_path,
                        "audio_source": _record_audio_source(record),
                        "error": ffprobe_error,
                        "frames": 0,
                        "elapsed": time.perf_counter() - started_at,
                    }

                ffmpeg_error = _ffmpeg_decode_audio_source(
                    record,
                    self.ffmpeg_timeout_seconds,
                )
                if ffmpeg_error is not None:
                    return {
                        "status": "failed",
                        "index": record_index,
                        "utterance_id": record.utterance_id,
                        "audio_path": record.audio_path,
                        "audio_source": _record_audio_source(record),
                        "error": ffmpeg_error,
                        "frames": 0,
                        "elapsed": time.perf_counter() - started_at,
                    }

                waveform, sample_rate = load_audio(
                    None if record.audio_bytes is not None else record.audio_path,
                    record.audio_bytes,
                )
                features = self.featurizer(waveform, sample_rate)
                if not feature_tensor_is_plausible(
                    record,
                    features,
                    expected_feature_bins=self.featurizer.n_mels,
                ):
                    return {
                        "status": "invalid",
                        "index": record_index,
                        "utterance_id": record.utterance_id,
                        "audio_path": record.audio_path,
                        "audio_source": _record_audio_source(record),
                        "frames": int(getattr(features, "shape", [0])[0]),
                        "elapsed": time.perf_counter() - started_at,
                    }

                if self.feature_cache is not None and self.write_cache:
                    self.feature_cache.store(record.utterance_id, self.featurizer, features)
                elif self.feature_cache is not None:
                    return {
                        "status": "written",
                        "index": record_index,
                        "utterance_id": record.utterance_id,
                        "audio_path": record.audio_path,
                        "audio_source": _record_audio_source(record),
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
                    "index": record_index,
                    "utterance_id": record.utterance_id,
                    "audio_path": record.audio_path,
                    "audio_source": _record_audio_source(record),
                    "frames": int(features.size(0)),
                    "elapsed": time.perf_counter() - started_at,
                }
        except Exception as error:
            return {
                "status": "failed",
                "index": record_index,
                "utterance_id": record.utterance_id,
                "audio_path": record.audio_path,
                "audio_source": _record_audio_source(record),
                "error": str(error),
                "frames": 0,
                "elapsed": time.perf_counter() - started_at,
            }


def _collate_statuses(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return items


def _record_skip_keys(record: AudioRecord) -> set[str]:
    keys = {str(record.utterance_id)}
    if record.audio_path:
        audio_path = str(record.audio_path)
        keys.add(audio_path)
        keys.add(Path(audio_path).name)
    return keys


def _record_audio_source(record: AudioRecord) -> str:
    if record.audio_bytes is not None:
        return f"bytes:{len(record.audio_bytes)}"
    if record.audio_path:
        return f"path:{record.audio_path}"
    return "none"


def _record_matches_skip_list(record: AudioRecord, skipped_audio_sources: set[str]) -> bool:
    return bool(_record_skip_keys(record) & skipped_audio_sources)


def _load_skip_list(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    skip_path = Path(path)
    if not skip_path.exists():
        raise FileNotFoundError(f"cache warm skip list does not exist: {skip_path}")
    values: set[str] = set()
    for line in skip_path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        values.add(value)
        values.add(Path(value).name)
    return values


def _append_failed_record(path: str | Path | None, item: dict[str, Any]) -> None:
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path = str(item.get("audio_path") or "")
    audio_source = str(item.get("audio_source") or "")
    utterance_id = str(item.get("utterance_id") or "")
    error = str(item.get("error") or "")
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{audio_path}\t{audio_source}\t{utterance_id}\t"
            f"{error.replace(chr(9), ' ').replace(chr(10), ' ')}\n"
        )


def _append_skipped_record(path: str | Path | None, record: AudioRecord, error: str) -> None:
    _append_failed_record(
        path,
        {
            "audio_path": record.audio_path,
            "audio_source": _record_audio_source(record),
            "utterance_id": record.utterance_id,
            "error": error,
        },
    )


def _ffprobe_audio_source(record: AudioRecord, timeout_seconds: float) -> str | None:
    if timeout_seconds <= 0:
        return None
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
    ]
    try:
        if record.audio_bytes is not None:
            result = subprocess.run(
                [*command, "pipe:0"],
                input=record.audio_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        elif record.audio_path is not None:
            result = subprocess.run(
                [*command, str(record.audio_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        else:
            return "ffprobe skipped record without audio path or embedded bytes"
    except subprocess.TimeoutExpired:
        return f"ffprobe timed out after {timeout_seconds:g}s"
    except FileNotFoundError:
        return "ffprobe executable was not found"
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        return f"ffprobe failed with exit code {result.returncode}: {stderr}"
    if b"audio" not in result.stdout:
        return "ffprobe did not find an audio stream"
    return None


def _ffmpeg_decode_audio_source(record: AudioRecord, timeout_seconds: float) -> str | None:
    if timeout_seconds <= 0:
        return None
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
    ]
    try:
        if record.audio_bytes is not None:
            result = subprocess.run(
                [
                    *command,
                    "-i",
                    "pipe:0",
                    "-map",
                    "0:a:0",
                    "-f",
                    "null",
                    "-",
                ],
                input=record.audio_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        elif record.audio_path is not None:
            result = subprocess.run(
                [
                    *command,
                    "-i",
                    str(record.audio_path),
                    "-map",
                    "0:a:0",
                    "-f",
                    "null",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        else:
            return "ffmpeg skipped record without audio path or embedded bytes"
    except subprocess.TimeoutExpired:
        return f"ffmpeg decode validation timed out after {timeout_seconds:g}s"
    except FileNotFoundError:
        return "ffmpeg executable was not found"
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        return f"ffmpeg decode validation failed with exit code {result.returncode}: {stderr}"
    return None


@contextmanager
def _record_timeout(seconds: float):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise_timeout(signum, frame) -> None:
        raise TimeoutError(f"feature cache warm record timed out after {seconds:g}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _raise_timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            remaining = max(0.0, previous_timer[0] - seconds)
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


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
            "When --cache-warm-skip-on-timeout is enabled, 0 uses an automatic timeout; "
            "otherwise 0 disables timeout and only logs waits."
        ),
    )
    parser.add_argument(
        "--cache-warm-skip-on-timeout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When the DataLoader times out, mark the next pending record as skipped, rebuild "
            "the loader, and continue. With --no-dataloader-in-order the skipped record is a "
            "best-effort candidate because workers can finish out of order."
        ),
    )
    parser.add_argument(
        "--cache-warm-record-timeout",
        type=float,
        default=30.0,
        help=(
            "Per-record timeout in seconds inside each warmer worker. Timed-out records are "
            "logged as failed and skipped. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--cache-warm-ffprobe-timeout",
        type=float,
        default=0.0,
        help=(
            "Optional fast ffprobe validation timeout in seconds before Python decode. "
            "Records rejected or timed out by ffprobe are logged as failed and skipped. "
            "Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--cache-warm-ffmpeg-timeout",
        type=float,
        default=0.0,
        help=(
            "Optional ffmpeg decode-validation timeout in seconds before Python decode. "
            "This feeds the sample to ffmpeg and decodes to null; rejected or timed-out "
            "records are logged as failed and skipped. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--cache-warm-skip-list",
        default=None,
        help=(
            "Optional text file of audio paths, basenames, or utterance ids to skip before "
            "audio decode. One entry per line; lines starting with # are ignored."
        ),
    )
    parser.add_argument(
        "--cache-warm-failed-list",
        default=None,
        help=(
            "Optional TSV file to append failed warm records to. Its first column can be reused "
            "as --cache-warm-skip-list on later runs."
        ),
    )
    parser.add_argument(
        "--cache-warm-audio-source",
        choices=["any", "bytes"],
        default="any",
        help=(
            "Which audio records to warm. 'any' accepts embedded bytes or readable paths. "
            "'bytes' warms only records whose manifest row provides embedded audio bytes."
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
    if 0 < training_args.cache_warm_timeout < 1:
        raise ValueError(
            "--cache-warm-timeout must be either 0 to disable DataLoader timeout or at least "
            f"1 second, got {training_args.cache_warm_timeout}."
        )
    if training_args.cache_warm_record_timeout < 0:
        raise ValueError(
            "--cache-warm-record-timeout must be >= 0, "
            f"got {training_args.cache_warm_record_timeout}."
        )
    if 0 < training_args.cache_warm_record_timeout < 1:
        raise ValueError(
            "--cache-warm-record-timeout must be either 0 to disable per-record timeout or at "
            f"least 1 second, got {training_args.cache_warm_record_timeout}."
        )
    if training_args.cache_warm_ffprobe_timeout < 0:
        raise ValueError(
            "--cache-warm-ffprobe-timeout must be >= 0, "
            f"got {training_args.cache_warm_ffprobe_timeout}."
        )
    if training_args.cache_warm_ffmpeg_timeout < 0:
        raise ValueError(
            "--cache-warm-ffmpeg-timeout must be >= 0, "
            f"got {training_args.cache_warm_ffmpeg_timeout}."
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


def _resolve_cache_warm_dataloader_timeout(args: argparse.Namespace, workers: int) -> float:
    if workers <= 0:
        return 0.0
    if args.cache_warm_timeout > 0:
        return float(args.cache_warm_timeout)
    if not args.cache_warm_skip_on_timeout:
        return 0.0

    validation_timeout = 0.0
    if args.cache_warm_ffprobe_timeout > 0:
        validation_timeout += float(args.cache_warm_ffprobe_timeout)
    if args.cache_warm_ffmpeg_timeout > 0:
        validation_timeout += float(args.cache_warm_ffmpeg_timeout)
    if validation_timeout > 0:
        return max(5.0, validation_timeout + 2.0)

    if args.cache_warm_record_timeout > 0:
        return float(args.cache_warm_record_timeout)
    return max(30.0, float(args.cache_warm_wait_log_after))


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
        "invalid=%s skipped=%s failed=%s frames=%s elapsed=%s",
        split,
        processed,
        total,
        percent,
        rate,
        counts.get("written", 0),
        counts.get("hit", 0),
        counts.get("invalid", 0),
        counts.get("skipped", 0),
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
    return f"index={index} utterance_id={record.utterance_id} source={_record_audio_source(record)}"


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
    skipped_audio_sources: set[str],
) -> dict[str, int]:
    workers = args.num_workers if args.cache_warm_workers is None else args.cache_warm_workers
    dataloader_timeout = _resolve_cache_warm_dataloader_timeout(args, workers)
    main_feature_cache = (
        ShardedParquetFeatureCache(cache_dir)
        if workers > 0 and str(args.feature_cache_format) == "parquet"
        else None
    )
    dataloader_kwargs: dict[str, object] = {}
    if workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
        dataloader_kwargs["persistent_workers"] = False
        dataloader_kwargs["in_order"] = args.dataloader_in_order
        if dataloader_timeout > 0:
            dataloader_kwargs["timeout"] = dataloader_timeout
        if args.dataloader_mp_context != "auto":
            dataloader_kwargs["multiprocessing_context"] = args.dataloader_mp_context

    pending_indices = list(range(len(records)))
    dataset: FeatureCacheWarmDataset | None = None
    loader: DataLoader | None = None
    iterator = None

    def rebuild_loader() -> None:
        nonlocal dataset, loader, iterator
        if dataset is not None:
            dataset.close()
        dataset = FeatureCacheWarmDataset(
            records,
            featurizer=featurizer,
            feature_cache_dir=cache_dir,
            feature_cache_format=args.feature_cache_format,
            overwrite=args.cache_warm_overwrite,
            validate_existing=args.cache_warm_validate_existing,
            write_cache=main_feature_cache is None,
            record_timeout_seconds=args.cache_warm_record_timeout,
            skipped_audio_sources=skipped_audio_sources,
            indices=pending_indices,
            ffprobe_timeout_seconds=args.cache_warm_ffprobe_timeout,
            ffmpeg_timeout_seconds=args.cache_warm_ffmpeg_timeout,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.cache_warm_batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=_collate_statuses,
            **dataloader_kwargs,
        )
        iterator = iter(loader)

    rebuild_loader()
    logger.info(
        "%s feature cache warm started records=%s hours=%.2f cache_dir=%s format=%s "
        "workers=%s batch_size=%s prefetch_factor=%s in_order=%s dataloader_timeout=%s "
        "requested_timeout=%s "
        "record_timeout=%s ffprobe_timeout=%s ffmpeg_timeout=%s overwrite=%s "
        "validate_existing=%s",
        split,
        len(records),
        _record_store_duration_hours(records, hop_length=featurizer.hop_length),
        cache_dir,
        args.feature_cache_format,
        workers,
        args.cache_warm_batch_size,
        args.prefetch_factor if workers > 0 else "none",
        args.dataloader_in_order if workers > 0 else "none",
        dataloader_timeout if workers > 0 else "none",
        args.cache_warm_timeout if workers > 0 else "none",
        args.cache_warm_record_timeout,
        args.cache_warm_ffprobe_timeout,
        args.cache_warm_ffmpeg_timeout,
        args.cache_warm_overwrite,
        args.cache_warm_validate_existing,
    )
    started_at = time.perf_counter()
    counts: dict[str, int] = {
        "written": 0,
        "hit": 0,
        "invalid": 0,
        "skipped": 0,
        "failed": 0,
    }
    frames = 0
    processed = 0
    log_interval = max(1, int(args.cache_warm_log_interval))
    while pending_indices:
        next_index = pending_indices[0]
        description = (
            f"{split} feature cache warm waiting for item {processed + 1}/{len(records)} "
            f"workers={workers} prefetch_factor={args.prefetch_factor if workers > 0 else 'none'} "
            f"in_order={args.dataloader_in_order if workers > 0 else 'none'} "
            f"dataloader_timeout={dataloader_timeout if workers > 0 else 'none'} "
            f"next={_record_wait_label(records, next_index)}"
        )
        try:
            batch, wait_seconds = _next_with_wait_logging(
                iterator,
                description=description,
                log_after_seconds=args.cache_warm_wait_log_after,
                log_every_seconds=args.cache_warm_wait_log_every,
            )
        except RuntimeError as error:
            if (
                not args.cache_warm_skip_on_timeout
                or dataloader_timeout <= 0
                or "DataLoader timed out" not in str(error)
            ):
                raise
            skipped_index = pending_indices.pop(0)
            skipped_record = records[skipped_index]
            counts["skipped"] = counts.get("skipped", 0) + 1
            processed += 1
            reason = f"DataLoader timed out after {dataloader_timeout:g}s"
            logger.warning(
                "%s feature cache warm skipped after dataloader timeout index=%s "
                "utterance_id=%s source=%s in_order=%s reason=%s",
                split,
                skipped_index,
                skipped_record.utterance_id,
                _record_audio_source(skipped_record),
                args.dataloader_in_order if workers > 0 else "none",
                reason,
            )
            _append_skipped_record(args.cache_warm_failed_list, skipped_record, reason)
            rebuild_loader()
            if processed % log_interval == 0 or processed >= len(records):
                _log_progress(
                    split=split,
                    processed=processed,
                    total=len(records),
                    counts=counts,
                    frames=frames,
                    started_at=started_at,
                )
            continue
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
            record_index = int(item.get("index", pending_indices[0] if pending_indices else -1))
            try:
                pending_indices.remove(record_index)
            except ValueError:
                pass
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
            if status == "skipped":
                logger.info(
                    "%s feature cache warm skipped utterance_id=%s source=%s",
                    split,
                    item.get("utterance_id"),
                    item.get("audio_source") or item.get("audio_path"),
                )
            if status == "failed":
                logger.warning(
                    "%s feature cache warm failed utterance_id=%s source=%s error=%s",
                    split,
                    item.get("utterance_id"),
                    item.get("audio_source") or item.get("audio_path"),
                    item.get("error"),
                )
                _append_failed_record(args.cache_warm_failed_list, item)
        if processed % log_interval == 0 or processed >= len(records):
            _log_progress(
                split=split,
                processed=processed,
                total=len(records),
                counts=counts,
                frames=frames,
                started_at=started_at,
            )
    if dataset is not None:
        dataset.close()
    if main_feature_cache is not None:
        main_feature_cache.close()
    logger.info(
        "%s feature cache warm complete records=%s written=%s hit=%s invalid=%s skipped=%s "
        "failed=%s elapsed=%s",
        split,
        processed,
        counts.get("written", 0),
        counts.get("hit", 0),
        counts.get("invalid", 0),
        counts.get("skipped", 0),
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
    args.require_audio_bytes = args.cache_warm_audio_source == "bytes"
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
    skipped_audio_sources = _load_skip_list(args.cache_warm_skip_list)
    total_counts: dict[str, int] = {
        "written": 0,
        "hit": 0,
        "invalid": 0,
        "skipped": 0,
        "failed": 0,
    }
    logger.info(
        "feature cache warm plan splits=%s total_splits=%s cache_root=%s format=%s skip_list=%s",
        ",".join(ordered_splits),
        len(ordered_splits),
        cache_root,
        args.feature_cache_format,
        len(skipped_audio_sources),
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
            skipped_audio_sources=skipped_audio_sources,
        )
        _accumulate_counts(total_counts, counts)
        logger.info(
            "feature cache warm split %s/%s complete split=%s written=%s hit=%s "
            "invalid=%s skipped=%s failed=%s",
            split_index,
            len(ordered_splits),
            split,
            counts.get("written", 0),
            counts.get("hit", 0),
            counts.get("invalid", 0),
            counts.get("skipped", 0),
            counts.get("failed", 0),
        )
    logger.info(
        "feature cache warm all requested splits complete splits=%s written=%s hit=%s "
        "invalid=%s skipped=%s failed=%s",
        ",".join(ordered_splits),
        total_counts.get("written", 0),
        total_counts.get("hit", 0),
        total_counts.get("invalid", 0),
        total_counts.get("skipped", 0),
        total_counts.get("failed", 0),
    )
    if args.cache_warm_fail_on_error and total_counts.get("failed", 0) > 0:
        raise SystemExit(f"Feature cache warming failed for {total_counts['failed']} record(s).")


if __name__ == "__main__":
    main(sys.argv[1:])
