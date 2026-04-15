from __future__ import annotations

import atexit
import functools
import hashlib
import io
import logging
import math
import multiprocessing as mp
import os
import re
import struct
import sys
import threading
import time
import wave
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import polars as pl
import torch
import torchaudio
from huggingface_hub import hf_hub_url, list_repo_files, snapshot_download
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler

from .asr import Tokenizer
from .frontend import (
    AudioFeaturizer,
    SpecAugment,
    WaveformAugment,
    estimate_num_feature_frames,
)

TRANSCRIPT_COLUMNS = ("sentence", "transcript", "transcription", "text", "normalized_text")
AUDIO_COLUMNS = ("path", "audio")


logger = logging.getLogger("train")

_DEFAULT_FEATURE_SAMPLE_RATE = 16_000
_DEFAULT_FEATURE_N_FFT = 400
_DEFAULT_FEATURE_WIN_LENGTH = 400
_DEFAULT_FEATURE_HOP_LENGTH = 160
_DEFAULT_FEATURE_BACKEND = "torchaudio"


def _epoch_shuffled_order(length: int, *, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch))
    return torch.randperm(length, generator=generator).tolist()


def _dataloader_multiprocessing_context(
    num_workers: int,
    multiprocessing_context: str | None = None,
):
    if num_workers <= 0 or not sys.platform.startswith("linux"):
        return None
    if multiprocessing_context is not None and multiprocessing_context != "auto":
        return mp.get_context(multiprocessing_context)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # Avoid forking a process that already owns distributed/CUDA runtime state.
        return mp.get_context("spawn")
    return mp.get_context("fork")


def _limit_dataloader_worker_threads(worker_id: int, *, num_threads: int) -> None:
    del worker_id
    worker_threads = max(1, int(num_threads))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[variable] = str(worker_threads)
    try:
        torch.set_num_threads(worker_threads)
    except Exception:
        logger.debug("failed to set torch DataLoader worker threads", exc_info=True)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        logger.debug("failed to set torch DataLoader worker interop threads", exc_info=True)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    try:
        threadpool_limits(limits=worker_threads)
    except Exception:
        logger.debug("failed to limit native DataLoader worker thread pools", exc_info=True)


def _progress_log_interval(total: int) -> int:
    if total <= 0:
        return 1
    return max(1_000, min(50_000, total // 20 or 1))


def _log_progress(
    progress_logger: logging.Logger | None,
    message: str,
    completed: int,
    total: int,
    start_time: float,
) -> None:
    if progress_logger is None:
        return
    elapsed = max(0.0, time.perf_counter() - start_time)
    rate = completed / elapsed if elapsed > 0 else 0.0
    percent = (completed / total * 100.0) if total > 0 else 0.0
    progress_logger.info(
        "%s progress=%s/%s %.1f%% rate=%.1f/s elapsed=%.1fs",
        message,
        completed,
        total,
        percent,
        rate,
        elapsed,
    )


def _should_log_progress(completed: int, total: int) -> bool:
    return completed == total or completed % _progress_log_interval(total) == 0


def _record_estimated_frames(records, index: int) -> int:
    if hasattr(records, "estimated_frames_at"):
        return int(records.estimated_frames_at(index))
    return int(records[index].estimated_frames)


def _record_duration_seconds(records, index: int) -> float:
    if hasattr(records, "duration_seconds_at"):
        return float(records.duration_seconds_at(index))
    record = records[index]
    if record.num_samples > 0 and record.sample_rate > 0:
        return max(0.0, record.num_samples / record.sample_rate)
    return 0.0


def _record_transcript_length(records, index: int) -> int:
    if hasattr(records, "transcript_length_at"):
        return int(records.transcript_length_at(index))
    return len(records[index].transcript)


def _record_token_length(records, index: int) -> int:
    if hasattr(records, "token_length_at"):
        return int(records.token_length_at(index))
    return _record_transcript_length(records, index)


def _sorted_indices_with_progress(
    records: list[AudioRecord],
    key_fn: Callable[[int], object],
    *,
    progress_logger: logging.Logger | None,
    progress_label: str,
    phase: str,
) -> list[int]:
    total = len(records)
    start_time = time.perf_counter()
    if progress_logger is not None:
        progress_logger.info("%s %s computing sort keys records=%s", progress_label, phase, total)
    keyed_indices: list[tuple[object, int]] = []
    for completed, index in enumerate(range(total), start=1):
        keyed_indices.append((key_fn(index), index))
        if progress_logger is not None and _should_log_progress(completed, total):
            _log_progress(
                progress_logger,
                f"{progress_label} {phase} computed sort keys",
                completed,
                total,
                start_time,
            )
    sort_start_time = time.perf_counter()
    if progress_logger is not None:
        progress_logger.info("%s %s sorting keyed records total=%s", progress_label, phase, total)
    keyed_indices.sort(key=lambda item: item[0])
    if progress_logger is not None:
        progress_logger.info(
            "%s %s sorted records total=%s key_elapsed=%.1fs sort_elapsed=%.1fs",
            progress_label,
            phase,
            total,
            sort_start_time - start_time,
            time.perf_counter() - sort_start_time,
        )
    return [index for _, index in keyed_indices]


@dataclass
class LoaderSummary:
    scanned: int = 0
    selected: int = 0
    skipped_missing_transcript: int = 0
    skipped_missing_audio: int = 0
    skipped_missing_duration: int = 0
    skipped_too_short: int = 0
    skipped_too_long: int = 0
    skipped_symbol_ratio: int = 0
    skipped_no_alnum: int = 0
    skipped_audio_too_short: int = 0
    skipped_audio_too_long: int = 0
    skipped_chars_per_second_too_low: int = 0
    skipped_chars_per_second_too_high: int = 0
    skipped_words_per_second_too_low: int = 0
    skipped_words_per_second_too_high: int = 0
    skipped_duration_per_char_too_low: int = 0
    skipped_duration_per_char_too_high: int = 0
    skipped_duration_per_word_too_low: int = 0
    skipped_duration_per_word_too_high: int = 0
    skipped_split: int = 0


def _source_summary_label(source: str | Path) -> str:
    return str(source)


def _transcript_rejection_reason(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    max_symbol_ratio: float,
) -> str | None:
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if transcript_symbol_ratio(text) > max_symbol_ratio:
        return "symbol_ratio"
    if not any(char.isalnum() for char in text):
        return "no_alnum"
    return None


def _log_loader_summary(
    *,
    source: str | Path,
    split: str,
    summary: LoaderSummary,
    max_samples: int | None,
) -> None:
    logger.info(
        "loader summary source=%s split=%s scanned=%s selected=%s skipped_missing_transcript=%s "
        "skipped_missing_audio=%s skipped_missing_duration=%s skipped_too_short=%s "
        "skipped_too_long=%s skipped_symbol_ratio=%s skipped_no_alnum=%s "
        "skipped_audio_too_short=%s skipped_audio_too_long=%s "
        "skipped_chars_per_second_too_low=%s skipped_chars_per_second_too_high=%s "
        "skipped_words_per_second_too_low=%s skipped_words_per_second_too_high=%s "
        "skipped_duration_per_char_too_low=%s skipped_duration_per_char_too_high=%s "
        "skipped_duration_per_word_too_low=%s skipped_duration_per_word_too_high=%s "
        "skipped_split=%s max_samples=%s",
        _source_summary_label(source),
        split,
        summary.scanned,
        summary.selected,
        summary.skipped_missing_transcript,
        summary.skipped_missing_audio,
        summary.skipped_missing_duration,
        summary.skipped_too_short,
        summary.skipped_too_long,
        summary.skipped_symbol_ratio,
        summary.skipped_no_alnum,
        summary.skipped_audio_too_short,
        summary.skipped_audio_too_long,
        summary.skipped_chars_per_second_too_low,
        summary.skipped_chars_per_second_too_high,
        summary.skipped_words_per_second_too_low,
        summary.skipped_words_per_second_too_high,
        summary.skipped_duration_per_char_too_low,
        summary.skipped_duration_per_char_too_high,
        summary.skipped_duration_per_word_too_low,
        summary.skipped_duration_per_word_too_high,
        summary.skipped_split,
        max_samples if max_samples is not None else "none",
    )


def _audio_duration_rejection_reason(
    duration_seconds: float,
    *,
    min_duration_seconds: float,
    max_duration_seconds: float,
) -> str | None:
    if duration_seconds < min_duration_seconds:
        return "too_short"
    if duration_seconds > max_duration_seconds:
        return "too_long"
    return None


def _alignment_rejection_reason(
    text: str,
    duration_seconds: float,
    *,
    min_chars_per_second: float,
    max_chars_per_second: float,
    min_words_per_second: float,
    max_words_per_second: float,
    min_duration_per_char: float,
    max_duration_per_char: float,
    min_duration_per_word: float,
    max_duration_per_word: float,
) -> str | None:
    if duration_seconds <= 0.0:
        return None
    char_count = sum(1 for char in text if not char.isspace())
    word_count = len(text.split())
    if char_count <= 0 or word_count <= 0:
        return None

    chars_per_second = char_count / duration_seconds
    if chars_per_second < min_chars_per_second:
        return "chars_per_second_too_low"
    if chars_per_second > max_chars_per_second:
        return "chars_per_second_too_high"

    words_per_second = word_count / duration_seconds
    if words_per_second < min_words_per_second:
        return "words_per_second_too_low"
    if words_per_second > max_words_per_second:
        return "words_per_second_too_high"

    duration_per_char = duration_seconds / char_count
    if duration_per_char < min_duration_per_char:
        return "duration_per_char_too_low"
    if duration_per_char > max_duration_per_char:
        return "duration_per_char_too_high"

    duration_per_word = duration_seconds / word_count
    if duration_per_word < min_duration_per_word:
        return "duration_per_word_too_low"
    if duration_per_word > max_duration_per_word:
        return "duration_per_word_too_high"

    return None


def _increment_summary_field(
    summary: LoaderSummary,
    reason: str,
    *,
    prefix: str = "skipped_",
) -> None:
    setattr(summary, f"{prefix}{reason}", getattr(summary, f"{prefix}{reason}") + 1)


@dataclass(frozen=True)
class AudioRecord:
    audio_path: str | None
    audio_bytes: bytes | None
    transcript: str
    utterance_id: str
    estimated_frames: int
    speaker_id: str | None = None
    has_speaker_id: bool = False
    num_samples: int = 0
    sample_rate: int = 0


def estimate_feature_frames_from_metadata(
    num_samples: int,
    sample_rate: int,
    *,
    hop_length: int = _DEFAULT_FEATURE_HOP_LENGTH,
    featurizer: AudioFeaturizer | None = None,
) -> int:
    if num_samples <= 0 or sample_rate <= 0:
        return 0
    if featurizer is not None:
        return featurizer.estimate_num_frames(num_samples, sample_rate)
    return estimate_num_feature_frames(
        num_samples,
        sample_rate=sample_rate,
        target_sample_rate=_DEFAULT_FEATURE_SAMPLE_RATE,
        n_fft=_DEFAULT_FEATURE_N_FFT,
        win_length=_DEFAULT_FEATURE_WIN_LENGTH,
        hop_length=hop_length,
        backend=_DEFAULT_FEATURE_BACKEND,
    )


def normalize_transcript(text: str, lowercase: bool = True) -> str:
    normalized = text.strip()
    if lowercase:
        normalized = normalized.lower()
    normalized = normalized.replace("’", "'").replace("`", "'").replace("ʼ", "'")
    normalized = re.sub(r"[“”«»]", '"', normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    return normalized


def transcript_symbol_ratio(text: str) -> float:
    if not text:
        return 1.0
    noisy = sum(1 for char in text if not (char.isalnum() or char.isspace() or char in {"'", "-"}))
    return noisy / len(text)


def transcript_is_usable(
    text: str,
    min_chars: int = 1,
    max_chars: int = 400,
    max_symbol_ratio: float = 0.5,
) -> bool:
    return (
        min_chars <= len(text) <= max_chars
        and transcript_symbol_ratio(text) <= max_symbol_ratio
        and any(char.isalnum() for char in text)
    )


def _hash_to_unit_interval(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _collect_manifest_frames(dataset_root: Path) -> list[pl.LazyFrame]:
    manifest_frames: list[pl.LazyFrame] = []
    tsv_files = sorted(dataset_root.rglob("*.tsv"))
    if tsv_files:
        for path in tsv_files:
            manifest_frames.append(pl.scan_csv(path, separator="\t", infer_schema_length=1000))
        return manifest_frames

    parquet_files = sorted(dataset_root.rglob("*.parquet"))
    for path in parquet_files:
        manifest_frames.append(pl.scan_parquet(path))
    return manifest_frames


def _hf_storage_options(token: str | None) -> dict[str, str]:
    return {"token": token} if token else {}


def _select_transcript_column(columns: Iterable[str]) -> str | None:
    available = set(columns)
    for column in TRANSCRIPT_COLUMNS:
        if column in available:
            return column
    return None


def _iter_manifest_paths(dataset_root: Path) -> Iterable[Path]:
    tsv_files = sorted(dataset_root.rglob("*.tsv"))
    if tsv_files:
        return tsv_files
    return sorted(dataset_root.rglob("*.parquet"))


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _is_probable_hf_dataset_repo(value: str) -> bool:
    if _is_url(value):
        return False
    if value.startswith(("/", "./", "../", "~/")):
        return False
    if value.endswith((".parquet", ".tsv")):
        return False
    return "/" in value


def _auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _remote_source_cache_root(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser() / "remote_sources"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    base_cache_dir = Path(xdg_cache_home).expanduser() if xdg_cache_home else Path.home() / ".cache"
    return base_cache_dir / "squeezeformer_pytorch" / "remote_sources"


def _remote_source_cache_path(url: str, cache_dir: str | Path | None = None) -> Path:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    if not suffix:
        suffix = ".bin"
    cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    host_dir = parsed.netloc.replace(":", "_") or "remote"
    return _remote_source_cache_root(cache_dir) / host_dir / f"{cache_key}{suffix}"


def _read_remote_bytes(
    url: str,
    token: str | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> bytes:
    cache_path = _remote_source_cache_path(url, cache_dir)
    if cache_path.exists():
        return cache_path.read_bytes()
    request = Request(url, headers=_auth_headers(token))
    with urlopen(request) as response:
        payload = response.read()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temp_path.write_bytes(payload)
    temp_path.replace(cache_path)
    return payload


def read_binary_source(
    path_or_url: str,
    token: str | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> bytes:
    if _is_url(path_or_url):
        return _read_remote_bytes(path_or_url, token=token, cache_dir=cache_dir)
    return Path(path_or_url).read_bytes()


def _iter_remote_rows(
    source_url: str,
    *,
    token: str | None,
    batch_size: int,
    cache_dir: str | Path | None = None,
) -> Iterable[dict[str, Any]]:
    suffix = Path(urlparse(source_url).path).suffix.lower()
    payload = io.BytesIO(_read_remote_bytes(source_url, token=token, cache_dir=cache_dir))
    if suffix == ".tsv":
        reader = pl.scan_csv(
            payload,
            separator="\t",
            infer_schema_length=1000,
        )
        for batch in reader.collect_batches(chunk_size=batch_size):
            yield from batch.iter_rows(named=True)
        return
    if suffix == ".parquet":
        logger.info("loading remote parquet manifest %s", source_url)
        yield from pl.read_parquet(payload).iter_rows(named=True)
        return
    raise ValueError(f"Unsupported remote manifest format for {source_url}.")


def _iter_manifest_file_rows(path: Path, *, batch_size: int) -> Iterable[dict[str, Any]]:
    if path.suffix == ".tsv":
        reader = pl.scan_csv(
            path,
            separator="\t",
            infer_schema_length=1000,
        )
        for batch in reader.collect_batches(chunk_size=batch_size):
            yield from batch.iter_rows(named=True)
        return
    if path.suffix == ".parquet":
        logger.info("loading parquet manifest %s", path)
        yield from pl.read_parquet(path).iter_rows(named=True)
        return
    raise ValueError(f"Unsupported manifest file: {path}")


def _iter_repo_manifest_urls(repo_id: str, token: str | None) -> Iterable[str]:
    repo_files = sorted(list_repo_files(repo_id, repo_type="dataset", token=token))
    tsv_files = [repo_path for repo_path in repo_files if repo_path.endswith(".tsv")]
    parquet_files = [repo_path for repo_path in repo_files if repo_path.endswith(".parquet")]
    manifest_files = tsv_files or parquet_files
    if not manifest_files:
        raise FileNotFoundError(
            f"No TSV or Parquet manifest files found in dataset repo {repo_id}."
        )
    if parquet_files and not tsv_files:
        logger.info(
            "discovered %s parquet manifest file(s) in dataset repo %s",
            len(parquet_files),
            repo_id,
        )
    for repo_path in manifest_files:
        yield hf_hub_url(repo_id=repo_id, filename=repo_path, repo_type="dataset")


def iter_manifest_rows(dataset_root: Path, batch_size: int = 8192) -> Iterable[dict[str, Any]]:
    manifest_paths = list(_iter_manifest_paths(dataset_root))
    if not manifest_paths:
        raise FileNotFoundError(f"No TSV or Parquet manifest files found under {dataset_root}.")

    parquet_paths = [path for path in manifest_paths if path.suffix == ".parquet"]
    if parquet_paths:
        logger.info(
            "discovered %s parquet manifest file(s) under %s",
            len(parquet_paths),
            dataset_root,
        )

    for path in manifest_paths:
        yield from _iter_manifest_file_rows(path, batch_size=batch_size)


def iter_manifest_rows_from_source(
    source: str | Path,
    *,
    hf_token: str | None = None,
    batch_size: int = 8192,
    cache_dir: str | Path | None = None,
) -> Iterable[dict[str, Any]]:
    source_path = Path(source).expanduser()
    if source_path.exists():
        if source_path.is_dir():
            yield from iter_manifest_rows(source_path, batch_size=batch_size)
            return
        yield from _iter_manifest_file_rows(source_path, batch_size=batch_size)
        return

    source_text = str(source)
    if _is_url(source_text):
        yield from _iter_remote_rows(
            source_text,
            token=hf_token,
            batch_size=batch_size,
            cache_dir=cache_dir,
        )
        return
    if _is_probable_hf_dataset_repo(source_text):
        for manifest_url in _iter_repo_manifest_urls(source_text, token=hf_token):
            yield from _iter_remote_rows(
                manifest_url,
                token=hf_token,
                batch_size=batch_size,
                cache_dir=cache_dir,
            )
        return
    raise FileNotFoundError(f"Dataset source does not exist or is unsupported: {source_text}")


def _extract_transcript(row: dict[str, Any], lowercase: bool = True) -> str:
    for column in TRANSCRIPT_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return normalize_transcript(value, lowercase=lowercase)
    raise KeyError(f"No transcript column found in row. Tried {TRANSCRIPT_COLUMNS}.")


def _resolve_audio(row: dict[str, Any], dataset_root: Path) -> tuple[str | None, bytes | None]:
    path_value = row.get("path")
    if isinstance(path_value, str) and path_value:
        path = Path(path_value)
        if not path.is_absolute():
            path = dataset_root / path
        return str(path), None

    audio_value = row.get("audio")
    if isinstance(audio_value, dict):
        audio_bytes = audio_value.get("bytes")
        audio_path = audio_value.get("path")
        resolved_path: str | None = None
        if isinstance(audio_path, str) and audio_path:
            path = Path(audio_path)
            if not path.is_absolute():
                path = dataset_root / audio_path
            resolved_path = str(path)
        if isinstance(audio_bytes, (bytes, bytearray)):
            return resolved_path, bytes(audio_bytes)
        if resolved_path is not None:
            return resolved_path, None

    raise KeyError(f"No audio source found in row. Tried {AUDIO_COLUMNS}.")


def resolve_source_base(source: str | Path) -> Path | str:
    source_path = Path(source).expanduser()
    if source_path.exists():
        return source_path if source_path.is_dir() else source_path.parent
    source_text = str(source)
    if _is_url(source_text):
        return source_text.rsplit("/", 1)[0] + "/"
    if _is_probable_hf_dataset_repo(source_text):
        return source_text
    return source_path.parent


def resolve_audio_from_source(
    row: dict[str, Any],
    *,
    source: str | Path,
) -> tuple[str | None, bytes | None]:
    source_base = resolve_source_base(source)

    def resolve_path(path_value: str) -> str:
        if _is_url(path_value):
            return path_value
        if isinstance(source_base, Path):
            path = Path(path_value)
            if not path.is_absolute():
                path = source_base / path
            return str(path)
        if _is_probable_hf_dataset_repo(source_base):
            return hf_hub_url(repo_id=source_base, filename=path_value, repo_type="dataset")
        return urljoin(source_base, path_value)

    path_value = row.get("path")
    if isinstance(path_value, str) and path_value:
        return resolve_path(path_value), None

    audio_value = row.get("audio")
    if isinstance(audio_value, dict):
        audio_bytes = audio_value.get("bytes")
        audio_path = audio_value.get("path")
        resolved_path: str | None = None
        if isinstance(audio_path, str) and audio_path:
            resolved_path = resolve_path(audio_path)
        if isinstance(audio_bytes, (bytes, bytearray)):
            return resolved_path, bytes(audio_bytes)
        if resolved_path is not None:
            return resolved_path, None

    raise KeyError(f"No audio source found in row. Tried {AUDIO_COLUMNS}.")


def _record_split_matches(
    record: AudioRecord,
    *,
    split: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> bool:
    train_cutoff = max(0.0, 1.0 - val_fraction - test_fraction)
    split_key = record.speaker_id or record.utterance_id
    score = _hash_to_unit_interval(split_key, seed=seed)
    if split == "train":
        return score < train_cutoff
    if split == "validation":
        return train_cutoff <= score < train_cutoff + val_fraction
    if split == "test":
        return score >= train_cutoff + val_fraction
    raise ValueError(f"Unsupported split: {split}")


def _build_cv_record(
    row: dict[str, Any],
    *,
    resolve_audio: Callable[[dict[str, Any]], tuple[str | None, bytes | None]],
    scanned_rows: int,
    summary: LoaderSummary,
    min_transcript_chars: int,
    max_transcript_chars: int,
    max_symbol_ratio: float,
    min_audio_duration_sec: float,
    max_audio_duration_sec: float,
    min_chars_per_second: float,
    max_chars_per_second: float,
    min_words_per_second: float,
    max_words_per_second: float,
    min_duration_per_char: float,
    max_duration_per_char: float,
    min_duration_per_word: float,
    max_duration_per_word: float,
    lowercase_transcripts: bool,
) -> AudioRecord | None:
    try:
        transcript = _extract_transcript(row, lowercase=lowercase_transcripts)
    except KeyError:
        summary.skipped_missing_transcript += 1
        return None
    try:
        audio_path, audio_bytes = resolve_audio(row)
    except KeyError:
        summary.skipped_missing_audio += 1
        return None

    duration_seconds = (
        row.get("duration") or row.get("duration_seconds") or row.get("audio_duration")
    )
    if not isinstance(duration_seconds, (float, int)):
        summary.skipped_missing_duration += 1
        return None
    duration_seconds = float(duration_seconds)

    duration_rejection_reason = _audio_duration_rejection_reason(
        duration_seconds,
        min_duration_seconds=min_audio_duration_sec,
        max_duration_seconds=max_audio_duration_sec,
    )
    if duration_rejection_reason is not None:
        _increment_summary_field(summary, duration_rejection_reason, prefix="skipped_audio_")
        return None

    transcript_rejection_reason = _transcript_rejection_reason(
        transcript,
        min_chars=min_transcript_chars,
        max_chars=max_transcript_chars,
        max_symbol_ratio=max_symbol_ratio,
    )
    if transcript_rejection_reason is not None:
        _increment_summary_field(summary, transcript_rejection_reason)
        return None

    alignment_rejection_reason = _alignment_rejection_reason(
        transcript,
        duration_seconds,
        min_chars_per_second=min_chars_per_second,
        max_chars_per_second=max_chars_per_second,
        min_words_per_second=min_words_per_second,
        max_words_per_second=max_words_per_second,
        min_duration_per_char=min_duration_per_char,
        max_duration_per_char=max_duration_per_char,
        min_duration_per_word=min_duration_per_word,
        max_duration_per_word=max_duration_per_word,
    )
    if alignment_rejection_reason is not None:
        _increment_summary_field(summary, alignment_rejection_reason)
        return None

    utterance_id = str(row.get("id") or audio_path or scanned_rows)
    raw_speaker_id = row.get("client_id") or row.get("speaker_id") or row.get("speaker")
    speaker_id = str(raw_speaker_id) if raw_speaker_id not in {None, ""} else None
    num_samples = max(1, int(round(duration_seconds * _DEFAULT_FEATURE_SAMPLE_RATE)))
    estimated_frames = estimate_feature_frames_from_metadata(
        num_samples,
        _DEFAULT_FEATURE_SAMPLE_RATE,
        hop_length=_DEFAULT_FEATURE_HOP_LENGTH,
    )
    return AudioRecord(
        audio_path=audio_path,
        audio_bytes=audio_bytes,
        transcript=transcript,
        utterance_id=utterance_id,
        speaker_id=speaker_id,
        has_speaker_id=speaker_id is not None,
        estimated_frames=estimated_frames,
        num_samples=num_samples,
        sample_rate=_DEFAULT_FEATURE_SAMPLE_RATE,
    )


def _load_wave_from_stdlib(source: str | io.BytesIO) -> tuple[Tensor, int]:
    with wave.open(str(source) if isinstance(source, str) else source, "rb") as handle:
        sample_rate = handle.getframerate()
        num_channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if sample_width == 1:
        tensor = torch.frombuffer(bytearray(frames), dtype=torch.uint8).to(torch.float32)
        tensor = (tensor - 128.0) / 128.0
    elif sample_width == 2:
        tensor = torch.frombuffer(bytearray(frames), dtype=torch.int16).to(torch.float32) / 32768.0
    elif sample_width == 4:
        tensor = (
            torch.frombuffer(bytearray(frames), dtype=torch.int32).to(torch.float32) / 2147483648.0
        )
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    waveform = tensor.view(-1, num_channels).transpose(0, 1).contiguous()
    return waveform, sample_rate


def probe_audio_metadata(audio_path: str | None, audio_bytes: bytes | None) -> tuple[int, int]:
    try:
        if audio_path is not None and Path(audio_path).exists():
            info = torchaudio.info(audio_path)
            return int(info.num_frames), int(info.sample_rate)
        if audio_bytes is not None:
            info = torchaudio.info(io.BytesIO(audio_bytes))
            return int(info.num_frames), int(info.sample_rate)
    except ImportError:
        pass
    except Exception:
        pass

    try:
        waveform, sample_rate = load_audio(audio_path, audio_bytes)
    except Exception:
        return 0, 0
    return int(waveform.size(-1)), int(sample_rate)


def load_audio(audio_path: str | None, audio_bytes: bytes | None) -> tuple[Tensor, int]:
    try:
        if audio_bytes is not None:
            return torchaudio.load(io.BytesIO(audio_bytes))
        if audio_path is not None and Path(audio_path).exists():
            return torchaudio.load(audio_path)
    except ImportError:
        pass

    if audio_bytes is not None:
        return _load_wave_from_stdlib(io.BytesIO(audio_bytes))
    if audio_path is not None and Path(audio_path).exists():
        return _load_wave_from_stdlib(audio_path)
    raise FileNotFoundError("Audio source is not available.")


def download_dataset(
    repo_id: str,
    token: str | None,
    cache_dir: str | None = None,
    force_download: bool = False,
    allow_patterns: list[str] | None = None,
) -> Path:
    local_path = Path(repo_id)
    if local_path.exists():
        return local_path.resolve()
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
    )
    return Path(local_path)


def load_records(
    dataset_root: Path,
    split: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    max_samples: int | None = None,
    min_transcript_chars: int = 1,
    max_transcript_chars: int = 400,
    max_symbol_ratio: float = 0.5,
    min_audio_duration_sec: float = 0.01,
    max_audio_duration_sec: float = 30.0,
    min_chars_per_second: float = 0.0,
    max_chars_per_second: float = float("inf"),
    min_words_per_second: float = 0.0,
    max_words_per_second: float = float("inf"),
    min_duration_per_char: float = 0.0,
    max_duration_per_char: float = float("inf"),
    min_duration_per_word: float = 0.0,
    max_duration_per_word: float = float("inf"),
    lowercase_transcripts: bool = True,
) -> list[AudioRecord]:
    records = list(
        iter_records(
            dataset_root=dataset_root,
            split=split,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            max_samples=max_samples,
            min_transcript_chars=min_transcript_chars,
            max_transcript_chars=max_transcript_chars,
            max_symbol_ratio=max_symbol_ratio,
            min_audio_duration_sec=min_audio_duration_sec,
            max_audio_duration_sec=max_audio_duration_sec,
            min_chars_per_second=min_chars_per_second,
            max_chars_per_second=max_chars_per_second,
            min_words_per_second=min_words_per_second,
            max_words_per_second=max_words_per_second,
            min_duration_per_char=min_duration_per_char,
            max_duration_per_char=max_duration_per_char,
            min_duration_per_word=min_duration_per_word,
            max_duration_per_word=max_duration_per_word,
            lowercase_transcripts=lowercase_transcripts,
        )
    )
    if not records:
        raise RuntimeError(f"Split '{split}' is empty after applying the current split fractions.")
    return records


def iter_records(
    dataset_root: Path,
    split: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    max_samples: int | None = None,
    min_transcript_chars: int = 1,
    max_transcript_chars: int = 400,
    max_symbol_ratio: float = 0.5,
    min_audio_duration_sec: float = 0.01,
    max_audio_duration_sec: float = 30.0,
    min_chars_per_second: float = 0.0,
    max_chars_per_second: float = float("inf"),
    min_words_per_second: float = 0.0,
    max_words_per_second: float = float("inf"),
    min_duration_per_char: float = 0.0,
    max_duration_per_char: float = float("inf"),
    min_duration_per_word: float = 0.0,
    max_duration_per_word: float = float("inf"),
    lowercase_transcripts: bool = True,
) -> Iterable[AudioRecord]:
    summary = LoaderSummary()
    found_usable_record = False

    try:
        for row in iter_manifest_rows(dataset_root):
            summary.scanned += 1
            record = _build_cv_record(
                row,
                resolve_audio=lambda manifest_row: _resolve_audio(
                    manifest_row,
                    dataset_root=dataset_root,
                ),
                scanned_rows=summary.scanned,
                summary=summary,
                min_transcript_chars=min_transcript_chars,
                max_transcript_chars=max_transcript_chars,
                max_symbol_ratio=max_symbol_ratio,
                min_audio_duration_sec=min_audio_duration_sec,
                max_audio_duration_sec=max_audio_duration_sec,
                min_chars_per_second=min_chars_per_second,
                max_chars_per_second=max_chars_per_second,
                min_words_per_second=min_words_per_second,
                max_words_per_second=max_words_per_second,
                min_duration_per_char=min_duration_per_char,
                max_duration_per_char=max_duration_per_char,
                min_duration_per_word=min_duration_per_word,
                max_duration_per_word=max_duration_per_word,
                lowercase_transcripts=lowercase_transcripts,
            )
            if record is None:
                continue
            found_usable_record = True
            if not _record_split_matches(
                record,
                split=split,
                seed=seed,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
            ):
                summary.skipped_split += 1
                continue
            yield record
            summary.selected += 1
            if max_samples is not None and summary.selected >= max_samples:
                return
    finally:
        _log_loader_summary(
            source=dataset_root,
            split=split,
            summary=summary,
            max_samples=max_samples,
        )

    if not found_usable_record:
        raise RuntimeError("No usable records were found in the dataset manifests.")


def iter_records_from_source(
    source: str | Path,
    *,
    split: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    max_samples: int | None = None,
    min_transcript_chars: int = 1,
    max_transcript_chars: int = 400,
    max_symbol_ratio: float = 0.5,
    min_audio_duration_sec: float = 0.01,
    max_audio_duration_sec: float = 30.0,
    min_chars_per_second: float = 0.0,
    max_chars_per_second: float = float("inf"),
    min_words_per_second: float = 0.0,
    max_words_per_second: float = float("inf"),
    min_duration_per_char: float = 0.0,
    max_duration_per_char: float = float("inf"),
    min_duration_per_word: float = 0.0,
    max_duration_per_word: float = float("inf"),
    lowercase_transcripts: bool = True,
    hf_token: str | None = None,
    cache_dir: str | Path | None = None,
) -> Iterable[AudioRecord]:
    summary = LoaderSummary()
    found_usable_record = False

    try:
        for row in iter_manifest_rows_from_source(
            source,
            hf_token=hf_token,
            cache_dir=cache_dir,
        ):
            summary.scanned += 1
            record = _build_cv_record(
                row,
                resolve_audio=lambda manifest_row: resolve_audio_from_source(
                    manifest_row,
                    source=source,
                ),
                scanned_rows=summary.scanned,
                summary=summary,
                min_transcript_chars=min_transcript_chars,
                max_transcript_chars=max_transcript_chars,
                max_symbol_ratio=max_symbol_ratio,
                min_audio_duration_sec=min_audio_duration_sec,
                max_audio_duration_sec=max_audio_duration_sec,
                min_chars_per_second=min_chars_per_second,
                max_chars_per_second=max_chars_per_second,
                min_words_per_second=min_words_per_second,
                max_words_per_second=max_words_per_second,
                min_duration_per_char=min_duration_per_char,
                max_duration_per_char=max_duration_per_char,
                min_duration_per_word=min_duration_per_word,
                max_duration_per_word=max_duration_per_word,
                lowercase_transcripts=lowercase_transcripts,
            )
            if record is None:
                continue
            found_usable_record = True
            if not _record_split_matches(
                record,
                split=split,
                seed=seed,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
            ):
                summary.skipped_split += 1
                continue
            yield record
            summary.selected += 1
            if max_samples is not None and summary.selected >= max_samples:
                return
    finally:
        _log_loader_summary(
            source=source,
            split=split,
            summary=summary,
            max_samples=max_samples,
        )

    if not found_usable_record:
        raise RuntimeError("No usable records were found in the dataset manifests.")


def load_corpus_texts(
    dataset_root: Path,
    deduplicate: bool = False,
    max_samples: int | None = None,
    lowercase_transcripts: bool = True,
) -> list[str]:
    texts = list(
        iter_corpus_texts(
            dataset_root=dataset_root,
            deduplicate=deduplicate,
            max_samples=max_samples,
            lowercase_transcripts=lowercase_transcripts,
        )
    )
    if not texts:
        raise RuntimeError("No usable transcripts were found in the dataset manifests.")
    return texts


def iter_corpus_texts_from_repo(
    repo_id: str,
    token: str | None,
    deduplicate: bool = False,
    max_samples: int | None = None,
    lowercase_transcripts: bool = True,
) -> Iterable[str]:
    seen: set[str] = set()
    yielded = 0
    for manifest_url in _iter_repo_manifest_urls(repo_id, token=token):
        for row in _iter_remote_rows(manifest_url, token=token, batch_size=8192):
            try:
                transcript = _extract_transcript(row, lowercase=lowercase_transcripts)
            except KeyError:
                continue
            if deduplicate:
                if transcript in seen:
                    continue
                seen.add(transcript)
            yield transcript
            yielded += 1
            if max_samples is not None and yielded >= max_samples:
                return


def iter_corpus_texts(
    dataset_root: Path,
    deduplicate: bool = False,
    max_samples: int | None = None,
    lowercase_transcripts: bool = True,
) -> Iterable[str]:
    seen: set[str] = set()
    yielded = 0
    for row in iter_manifest_rows(dataset_root):
        try:
            transcript = _extract_transcript(row, lowercase=lowercase_transcripts)
        except KeyError:
            continue
        if deduplicate:
            if transcript in seen:
                continue
            seen.add(transcript)
        yield transcript
        yielded += 1
        if max_samples is not None and yielded >= max_samples:
            break


def feature_cache_path(
    feature_cache_dir: str | Path | None,
    utterance_id: str,
    featurizer: AudioFeaturizer,
    cache_key_extra: dict[str, object] | None = None,
) -> Path | None:
    if feature_cache_dir is None:
        return None
    feature_cache_path = Path(feature_cache_dir)
    feature_cache_path.mkdir(parents=True, exist_ok=True)
    safe_utterance_id = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_" for char in utterance_id
    ).strip("._")
    if not safe_utterance_id:
        safe_utterance_id = hashlib.sha256(utterance_id.encode("utf-8")).hexdigest()[:16]
    elif safe_utterance_id != utterance_id:
        safe_utterance_id = f"{safe_utterance_id[:96]}_{hashlib.sha256(utterance_id.encode('utf-8')).hexdigest()[:8]}"
    cache_config: dict[str, object] = {"featurizer": featurizer.config_dict()}
    if cache_key_extra:
        cache_config["extra"] = cache_key_extra
    frontend_hash = hashlib.sha256(repr(cache_config).encode("utf-8")).hexdigest()[:12]
    return feature_cache_path / f"{safe_utterance_id}_{frontend_hash}.pt"


def max_reasonable_feature_frames(record: AudioRecord) -> int:
    estimated_frames = max(1, int(record.estimated_frames))
    return max(20_000, estimated_frames * 4, estimated_frames + 1_024)


def normalize_feature_tensor(features: Tensor, expected_feature_bins: int) -> Tensor | None:
    if features.dim() != 2:
        return None
    if features.size(1) == expected_feature_bins:
        return features
    if features.size(0) == expected_feature_bins and features.size(1) > 0:
        return features.transpose(0, 1).contiguous()
    return None


def feature_tensor_is_plausible(
    record: AudioRecord,
    features: Tensor,
    *,
    expected_feature_bins: int,
) -> bool:
    normalized = normalize_feature_tensor(features, expected_feature_bins)
    if normalized is None:
        return False
    return 0 < normalized.size(0) <= max_reasonable_feature_frames(record)


_RUST_FEATURE_PAYLOAD_MAGIC = b"SFCF32L1"
_RUST_FEATURE_PAYLOAD_HEADER_SIZE = len(_RUST_FEATURE_PAYLOAD_MAGIC) + 8


def _load_feature_cache_payload(payload: bytes) -> Tensor:
    if not payload.startswith(_RUST_FEATURE_PAYLOAD_MAGIC):
        return torch.load(io.BytesIO(payload), map_location="cpu")
    if len(payload) < _RUST_FEATURE_PAYLOAD_HEADER_SIZE:
        raise ValueError("Rust feature cache payload is shorter than its header.")
    rows, cols = struct.unpack_from("<II", payload, len(_RUST_FEATURE_PAYLOAD_MAGIC))
    expected_size = _RUST_FEATURE_PAYLOAD_HEADER_SIZE + int(rows) * int(cols) * 4
    if len(payload) != expected_size:
        raise ValueError(
            "Rust feature cache payload has invalid size: "
            f"got {len(payload)} bytes, expected {expected_size}."
        )
    values = torch.frombuffer(
        bytearray(payload[_RUST_FEATURE_PAYLOAD_HEADER_SIZE:]),
        dtype=torch.float32,
    ).clone()
    return values.reshape(int(rows), int(cols))


class ShardedParquetFeatureCache:
    def __init__(
        self,
        root: str | Path,
        *,
        num_shards: int = 64,
        commit_every: int = 64,
    ) -> None:
        self.root = Path(root)
        self.num_shards = int(num_shards)
        self.commit_every = max(1, int(commit_every))
        self.shard_dir = self.root / "feature_shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self._pending_rows: dict[int, list[dict[str, object]]] = {}
        self._part_counters: dict[int, int] = {}
        self._pid: int | None = None
        self._lock = threading.RLock()
        atexit.register(self.close)

    def _key(self, utterance_id: str, featurizer: AudioFeaturizer) -> str:
        cache_config: dict[str, object] = {"featurizer": featurizer.config_dict()}
        frontend_hash = hashlib.sha256(repr(cache_config).encode("utf-8")).hexdigest()[:12]
        return hashlib.sha256(f"{utterance_id}:{frontend_hash}".encode("utf-8")).hexdigest()

    def _shard_index(self, key: str) -> int:
        return int(key[:8], 16) % self.num_shards

    def _ensure_process_state(self) -> None:
        current_pid = os.getpid()
        if self._pid != current_pid:
            self.close()
            self._pid = current_pid

    def _shard_path(self, shard_index: int) -> Path:
        path = self.shard_dir / f"features_{shard_index:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _part_path(self, shard_index: int) -> Path:
        counter = self._part_counters.get(shard_index, 0) + 1
        self._part_counters[shard_index] = counter
        return (
            self._shard_path(shard_index)
            / f"part_{os.getpid()}_{time.time_ns()}_{counter:06d}.parquet"
        )

    def close(self) -> None:
        with self._lock:
            self.flush()
            self._pid = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_pending_rows"] = {}
        state["_part_counters"] = {}
        state["_pid"] = None
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._pending_rows = {}
        self._part_counters = {}
        self._pid = None
        self._lock = threading.RLock()

    def _append_row(
        self,
        shard_index: int,
        *,
        key: str,
        payload: bytes | None,
        deleted: bool,
    ) -> None:
        self._ensure_process_state()
        rows = self._pending_rows.setdefault(shard_index, [])
        rows.append({"key": key, "payload": payload, "deleted": deleted})
        if len(rows) >= self.commit_every:
            self._flush_shard(shard_index)

    def _flush_shard(self, shard_index: int) -> None:
        rows = self._pending_rows.get(shard_index, [])
        if not rows:
            return
        frame = pl.DataFrame(
            {
                "key": [str(row["key"]) for row in rows],
                "payload": [row["payload"] for row in rows],
                "deleted": [bool(row["deleted"]) for row in rows],
            },
            schema={
                "key": pl.String,
                "payload": pl.Binary,
                "deleted": pl.Boolean,
            },
        )
        frame.write_parquet(self._part_path(shard_index))
        self._pending_rows[shard_index] = []

    def load(self, utterance_id: str, featurizer: AudioFeaturizer) -> Tensor | None:
        with self._lock:
            key = self._key(utterance_id, featurizer)
            shard_index = self._shard_index(key)
            self._ensure_process_state()
            for row in reversed(self._pending_rows.get(shard_index, [])):
                if row["key"] != key:
                    continue
                if row["deleted"]:
                    return None
                payload = row["payload"]
                if not isinstance(payload, bytes):
                    return None
                return _load_feature_cache_payload(payload)
            part_paths = sorted(
                self._shard_path(shard_index).glob("part_*.parquet"),
                key=lambda path: (path.stat().st_mtime_ns, path.name),
                reverse=True,
            )
            for path in part_paths:
                matches = (
                    pl.read_parquet(path, columns=["key", "payload", "deleted"])
                    .filter(pl.col("key") == key)
                    .tail(1)
                )
                if matches.is_empty():
                    continue
                row = matches.row(0, named=True)
                if row["deleted"]:
                    return None
                payload = row["payload"]
                if payload is None:
                    return None
                return _load_feature_cache_payload(payload)
            return None

    def store(self, utterance_id: str, featurizer: AudioFeaturizer, features: Tensor) -> None:
        with self._lock:
            key = self._key(utterance_id, featurizer)
            buffer = io.BytesIO()
            torch.save(features, buffer)
            shard_index = self._shard_index(key)
            self._append_row(
                shard_index,
                key=key,
                payload=buffer.getvalue(),
                deleted=False,
            )

    def flush(self) -> None:
        with self._lock:
            for shard_index in list(self._pending_rows):
                self._flush_shard(shard_index)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def delete(self, utterance_id: str, featurizer: AudioFeaturizer) -> None:
        with self._lock:
            key = self._key(utterance_id, featurizer)
            shard_index = self._shard_index(key)
            self._append_row(shard_index, key=key, payload=None, deleted=True)
            self._flush_shard(shard_index)


class ASRDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[AudioRecord],
        tokenizer: Tokenizer,
        featurizer: AudioFeaturizer,
        specaugment: SpecAugment | None = None,
        waveform_augment: WaveformAugment | None = None,
        feature_cache_dir: str | Path | None = None,
        feature_cache_format: str = "file",
        return_waveforms: bool = False,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.featurizer = featurizer
        self.specaugment = specaugment
        self.waveform_augment = waveform_augment
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir is not None else None
        self.feature_cache_format = str(feature_cache_format)
        if self.feature_cache_format not in {"file", "parquet"}:
            raise ValueError(
                "feature_cache_format must be either 'file' or 'parquet', "
                f"got {feature_cache_format!r}"
            )
        self.feature_cache = (
            ShardedParquetFeatureCache(self.feature_cache_dir)
            if self.feature_cache_dir is not None and self.feature_cache_format == "parquet"
            else None
        )
        self.return_waveforms = return_waveforms
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        if self.feature_cache is not None:
            self.feature_cache.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _feature_cache_path(self, record: AudioRecord) -> Path | None:
        return feature_cache_path(self.feature_cache_dir, record.utterance_id, self.featurizer)

    def _load_waveform(
        self,
        record: AudioRecord,
        *,
        waveform_augment_enabled: bool,
    ) -> tuple[Tensor, int]:
        waveform, sample_rate = load_audio(record.audio_path, record.audio_bytes)
        if waveform_augment_enabled and self.waveform_augment is not None:
            waveform, sample_rate = self.waveform_augment(waveform, sample_rate)
        return waveform, sample_rate

    def _compute_features_from_waveform(self, waveform: Tensor, sample_rate: int) -> Tensor:
        return self.featurizer(waveform, sample_rate)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any] | None:
        record = self.records[index]
        cache_path = self._feature_cache_path(record)
        waveform_augment_enabled = (
            self.waveform_augment is not None and self.waveform_augment.is_enabled()
        )
        waveform: Tensor | None = None
        sample_rate: int | None = None
        use_cache = cache_path is not None and not waveform_augment_enabled
        features: Tensor | None = None
        use_sharded_cache = use_cache and self.feature_cache is not None
        use_file_cache = use_cache and self.feature_cache_format == "file"
        if use_sharded_cache:
            features = self.feature_cache.load(record.utterance_id, self.featurizer)
        if features is None and use_cache and cache_path is not None and cache_path.exists():
            features = torch.load(cache_path, map_location="cpu")
            if use_sharded_cache:
                self.feature_cache.store(record.utterance_id, self.featurizer, features)
        if features is not None:
            if not feature_tensor_is_plausible(
                record,
                features,
                expected_feature_bins=self.featurizer.n_mels,
            ):
                logger.warning(
                    "discarding suspicious cached features for utterance_id=%s shape=%s "
                    "estimated_frames=%s",
                    record.utterance_id,
                    tuple(features.shape),
                    int(record.estimated_frames),
                )
                if self.feature_cache is not None:
                    self.feature_cache.delete(record.utterance_id, self.featurizer)
                if cache_path is not None:
                    cache_path.unlink(missing_ok=True)
                waveform, sample_rate = self._load_waveform(
                    record,
                    waveform_augment_enabled=waveform_augment_enabled,
                )
                features = self._compute_features_from_waveform(waveform, sample_rate)
                if use_sharded_cache:
                    self.feature_cache.store(record.utterance_id, self.featurizer, features)
                elif use_file_cache and cache_path is not None:
                    torch.save(features, cache_path)
        else:
            waveform, sample_rate = self._load_waveform(
                record,
                waveform_augment_enabled=waveform_augment_enabled,
            )
            features = self._compute_features_from_waveform(waveform, sample_rate)
            if use_sharded_cache:
                self.feature_cache.store(record.utterance_id, self.featurizer, features)
            elif use_file_cache and cache_path is not None:
                torch.save(features, cache_path)
        features = normalize_feature_tensor(features, self.featurizer.n_mels)
        if features is None or not feature_tensor_is_plausible(
            record,
            features,
            expected_feature_bins=self.featurizer.n_mels,
        ):
            logger.warning(
                "skipping utterance_id=%s due to invalid feature shape/length shape=%s "
                "estimated_frames=%s max_reasonable_frames=%s",
                record.utterance_id,
                tuple(features.shape) if features is not None else None,
                int(record.estimated_frames),
                max_reasonable_feature_frames(record),
            )
            return None
        if self.specaugment is not None:
            features = self.specaugment(features)
        target_ids = torch.tensor(self.tokenizer.encode(record.transcript), dtype=torch.long)
        batch_item = {
            "features": features,
            "feature_length": features.size(0),
            "feature_padding_value": float(getattr(self.featurizer, "padding_value", 0.0)),
            "targets": target_ids,
            "target_length": target_ids.numel(),
            "transcript": record.transcript,
            "utterance_id": record.utterance_id,
            "speaker_id": record.speaker_id,
            "has_speaker_id": record.has_speaker_id,
        }
        if self.return_waveforms:
            if waveform is None or sample_rate is None:
                waveform, sample_rate = self._load_waveform(
                    record,
                    waveform_augment_enabled=waveform_augment_enabled,
                )
            mono_waveform = waveform.mean(dim=0) if waveform.dim() > 1 else waveform.reshape(-1)
            batch_item.update(
                {
                    "waveform": mono_waveform.contiguous(),
                    "waveform_length": int(mono_waveform.numel()),
                    "sample_rate": int(sample_rate),
                }
            )
        return batch_item


class LengthBucketBatchSampler(BatchSampler):
    def __init__(
        self,
        records: list[AudioRecord],
        batch_size: int,
        shuffle: bool,
        longest_first: bool = False,
        seed: int = 0,
        progress_logger: logging.Logger | None = None,
        progress_label: str = "dataloader",
    ) -> None:
        self.records = records
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.longest_first = longest_first
        self.seed = int(seed)
        self.epoch = 0
        self.progress_logger = progress_logger
        self.progress_label = progress_label

    def _batch_order_key(self, batch: list[int]) -> tuple[int, int]:
        max_frames = max(max(1, _record_estimated_frames(self.records, index)) for index in batch)
        return max_frames, len(batch)

    def _build_batches(self) -> list[list[int]]:
        indices = _sorted_indices_with_progress(
            self.records,
            lambda index: _record_estimated_frames(self.records, index),
            progress_logger=self.progress_logger,
            progress_label=self.progress_label,
            phase="length-bucket sampler",
        )
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        if self.longest_first:
            batches.sort(key=self._batch_order_key, reverse=True)
        return batches

    def __iter__(self):
        batches = getattr(self, "_batches", None)
        if batches is None:
            batches = self._batches = self._build_batches()
        if not self.longest_first and self.shuffle:
            order = _epoch_shuffled_order(len(batches), seed=self.seed, epoch=self.epoch)
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self) -> int:
        batches = getattr(self, "_batches", None)
        if batches is None:
            batches = self._batches = self._build_batches()
        return len(batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class MaxFramesBatchSampler(BatchSampler):
    def __init__(
        self,
        records: list[AudioRecord],
        max_batch_frames: int,
        shuffle: bool,
        longest_first: bool = False,
        seed: int = 0,
        progress_logger: logging.Logger | None = None,
        progress_label: str = "dataloader",
    ) -> None:
        self.records = records
        self.max_batch_frames = max_batch_frames
        self.shuffle = shuffle
        self.longest_first = longest_first
        self.seed = int(seed)
        self.epoch = 0
        self.progress_logger = progress_logger
        self.progress_label = progress_label
        self._sorted_indices = _sorted_indices_with_progress(
            self.records,
            lambda index: _record_estimated_frames(self.records, index),
            progress_logger=self.progress_logger,
            progress_label=self.progress_label,
            phase="max-frames sampler",
        )
        self._batches = self._build_batches()

    def _batch_order_key(self, batch: list[int]) -> tuple[int, int]:
        max_frames = max(max(1, _record_estimated_frames(self.records, index)) for index in batch)
        return len(batch) * max_frames, max_frames

    def _iter_batches(self) -> Iterable[list[int]]:
        current_batch: list[int] = []
        current_max = 0
        for index in self._sorted_indices:
            frames = max(1, _record_estimated_frames(self.records, index))
            proposed_size = len(current_batch) + 1
            proposed_max = max(current_max, frames)
            if current_batch and proposed_size * proposed_max > self.max_batch_frames:
                yield current_batch
                current_batch = []
                current_max = 0
            current_batch.append(index)
            current_max = max(current_max, frames)
        if current_batch:
            yield current_batch

    def _build_batches(self) -> list[list[int]]:
        start_time = time.perf_counter()
        total_records = len(self._sorted_indices)
        batches: list[list[int]] = []
        seen = 0
        if self.progress_logger is not None:
            self.progress_logger.info(
                "%s max-frames sampler building batches records=%s budget=%s",
                self.progress_label,
                total_records,
                self.max_batch_frames,
            )
        for batch in self._iter_batches():
            batches.append(batch)
            seen += len(batch)
            if self.progress_logger is not None and _should_log_progress(seen, total_records):
                _log_progress(
                    self.progress_logger,
                    f"{self.progress_label} max-frames sampler built batches={len(batches)}",
                    seen,
                    total_records,
                    start_time,
                )
        if self.longest_first:
            batches.sort(key=self._batch_order_key, reverse=True)
        return batches

    def __iter__(self):
        batches = self._batches
        if not self.longest_first and self.shuffle:
            order = _epoch_shuffled_order(len(batches), seed=self.seed, epoch=self.epoch)
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class DurationBatchSampler(BatchSampler):
    def __init__(
        self,
        records: list[AudioRecord],
        max_batch_duration_sec: float,
        shuffle: bool,
        longest_first: bool = False,
        seed: int = 0,
        progress_logger: logging.Logger | None = None,
        progress_label: str = "dataloader",
    ) -> None:
        self.records = records
        self.max_batch_duration_sec = max_batch_duration_sec
        self.shuffle = shuffle
        self.longest_first = longest_first
        self.seed = int(seed)
        self.epoch = 0
        self.progress_logger = progress_logger
        self.progress_label = progress_label
        self._sorted_indices = _sorted_indices_with_progress(
            self.records,
            lambda index: (
                _record_duration_seconds(self.records, index),
                _record_estimated_frames(self.records, index),
            ),
            progress_logger=self.progress_logger,
            progress_label=self.progress_label,
            phase="duration sampler",
        )
        self._batches = self._build_batches()

    def _record_duration_seconds(self, record: AudioRecord) -> float:
        if record.num_samples > 0 and record.sample_rate > 0:
            return max(0.0, record.num_samples / record.sample_rate)
        return 0.0

    def _batch_order_key(self, batch: list[int]) -> tuple[float, int]:
        total_duration = sum(_record_duration_seconds(self.records, index) for index in batch)
        max_frames = max(max(1, _record_estimated_frames(self.records, index)) for index in batch)
        return total_duration, max_frames

    def _iter_batches(self) -> Iterable[list[int]]:
        current_batch: list[int] = []
        current_duration = 0.0
        for index in self._sorted_indices:
            duration = _record_duration_seconds(self.records, index)
            if current_batch and current_duration + duration > self.max_batch_duration_sec:
                yield current_batch
                current_batch = []
                current_duration = 0.0
            current_batch.append(index)
            current_duration += duration
        if current_batch:
            yield current_batch

    def _build_batches(self) -> list[list[int]]:
        start_time = time.perf_counter()
        total_records = len(self._sorted_indices)
        batches: list[list[int]] = []
        seen = 0
        if self.progress_logger is not None:
            self.progress_logger.info(
                "%s duration sampler building batches records=%s budget_sec=%.2f",
                self.progress_label,
                total_records,
                self.max_batch_duration_sec,
            )
        for batch in self._iter_batches():
            batches.append(batch)
            seen += len(batch)
            if self.progress_logger is not None and _should_log_progress(seen, total_records):
                _log_progress(
                    self.progress_logger,
                    f"{self.progress_label} duration sampler built batches={len(batches)}",
                    seen,
                    total_records,
                    start_time,
                )
        if self.longest_first:
            batches.sort(key=self._batch_order_key, reverse=True)
        return batches

    def __iter__(self):
        batches = self._batches
        if not self.longest_first and self.shuffle:
            order = _epoch_shuffled_order(len(batches), seed=self.seed, epoch=self.epoch)
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class AdaptiveBatchSampler(BatchSampler):
    def __init__(
        self,
        records: list[AudioRecord],
        target_batch_units: int,
        unit: str,
        shuffle: bool,
        longest_first: bool = False,
        seed: int = 0,
        progress_logger: logging.Logger | None = None,
        progress_label: str = "dataloader",
    ) -> None:
        if unit not in {"frames", "tokens"}:
            raise ValueError("unit must be one of {'frames', 'tokens'}")
        self.records = records
        self.target_batch_units = target_batch_units
        self.unit = unit
        self.shuffle = shuffle
        self.longest_first = longest_first
        self.seed = int(seed)
        self.epoch = 0
        self.progress_logger = progress_logger
        self.progress_label = progress_label
        self._sorted_indices = _sorted_indices_with_progress(
            self.records,
            lambda index: (
                _record_estimated_frames(self.records, index),
                _record_token_length(self.records, index),
            ),
            progress_logger=self.progress_logger,
            progress_label=self.progress_label,
            phase="adaptive sampler",
        )
        self._batches = self._build_batches()

    def _record_units(self, index: int) -> int:
        if self.unit == "frames":
            return max(1, _record_estimated_frames(self.records, index))
        return max(1, _record_token_length(self.records, index))

    def _iter_batches(self) -> Iterable[list[int]]:
        current_batch: list[int] = []
        current_units = 0
        for index in self._sorted_indices:
            units = self._record_units(index)
            if current_batch and current_units + units > self.target_batch_units:
                yield current_batch
                current_batch = []
                current_units = 0
            current_batch.append(index)
            current_units += units
        if current_batch:
            yield current_batch

    def _batch_order_key(self, batch: list[int]) -> tuple[int, int]:
        total_units = sum(self._record_units(index) for index in batch)
        max_frames = max(max(1, _record_estimated_frames(self.records, index)) for index in batch)
        return total_units, max_frames

    def _build_batches(self) -> list[list[int]]:
        start_time = time.perf_counter()
        total_records = len(self._sorted_indices)
        batches: list[list[int]] = []
        seen = 0
        if self.progress_logger is not None:
            self.progress_logger.info(
                "%s adaptive sampler building batches records=%s unit=%s budget=%s",
                self.progress_label,
                total_records,
                self.unit,
                self.target_batch_units,
            )
        for batch in self._iter_batches():
            batches.append(batch)
            seen += len(batch)
            if self.progress_logger is not None and _should_log_progress(seen, total_records):
                _log_progress(
                    self.progress_logger,
                    f"{self.progress_label} adaptive sampler built batches={len(batches)}",
                    seen,
                    total_records,
                    start_time,
                )
        if self.longest_first:
            batches.sort(key=self._batch_order_key, reverse=True)
        return batches

    def __iter__(self):
        batches = self._batches
        if not self.longest_first and self.shuffle:
            order = _epoch_shuffled_order(len(batches), seed=self.seed, epoch=self.epoch)
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class DistributedIndexSampler(Sampler[int]):
    def __init__(
        self,
        dataset_size: int,
        *,
        rank: int,
        world_size: int,
        shuffle: bool,
        seed: int = 0,
        drop_last: bool = False,
        pad_to_world_size: bool = False,
    ) -> None:
        if world_size < 1:
            raise ValueError("world_size must be at least 1")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size - 1}], got {rank}")
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.pad_to_world_size = bool(pad_to_world_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _all_indices(self) -> list[int]:
        indices = list(range(self.dataset_size))
        if self.shuffle:
            order = _epoch_shuffled_order(self.dataset_size, seed=self.seed, epoch=self.epoch)
            indices = [indices[index] for index in order]
        if self.drop_last:
            usable = (len(indices) // self.world_size) * self.world_size
            return indices[:usable]
        if self.pad_to_world_size and indices:
            total_size = math.ceil(len(indices) / self.world_size) * self.world_size
            if total_size > len(indices):
                indices = indices + indices[: total_size - len(indices)]
        return indices

    def __iter__(self):
        indices = self._all_indices()
        yield from indices[self.rank : len(indices) : self.world_size]

    def __len__(self) -> int:
        if self.dataset_size <= 0:
            return 0
        if self.drop_last:
            return self.dataset_size // self.world_size
        if self.pad_to_world_size:
            return math.ceil(self.dataset_size / self.world_size)
        remaining = max(0, self.dataset_size - self.rank)
        return (remaining + self.world_size - 1) // self.world_size


class DistributedBatchSampler(BatchSampler):
    def __init__(
        self,
        batch_sampler: BatchSampler,
        *,
        rank: int,
        world_size: int,
        drop_last: bool = False,
        pad_to_world_size: bool = False,
    ) -> None:
        if world_size < 1:
            raise ValueError("world_size must be at least 1")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size - 1}], got {rank}")
        self.batch_sampler = batch_sampler
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.drop_last = bool(drop_last)
        self.pad_to_world_size = bool(pad_to_world_size)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)

    def _all_batches(self) -> list[list[int]]:
        return [list(batch) for batch in self.batch_sampler]

    def __iter__(self):
        batches = self._all_batches()
        if self.drop_last:
            usable = (len(batches) // self.world_size) * self.world_size
            batches = batches[:usable]
        elif self.pad_to_world_size and batches:
            total_size = math.ceil(len(batches) / self.world_size) * self.world_size
            if total_size > len(batches):
                batches = batches + batches[: total_size - len(batches)]
        yield from batches[self.rank : len(batches) : self.world_size]

    def __len__(self) -> int:
        total_batches = len(self.batch_sampler)
        if total_batches <= 0:
            return 0
        if self.drop_last:
            return total_batches // self.world_size
        if self.pad_to_world_size:
            return math.ceil(total_batches / self.world_size)
        remaining = max(0, total_batches - self.rank)
        return (remaining + self.world_size - 1) // self.world_size


class IndexBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler: Sampler[int],
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        self.sampler = sampler
        self.batch_size = int(batch_size)

    def __iter__(self):
        batch: list[int] = []
        for index in self.sampler:
            batch.append(int(index))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        sampler_length = len(self.sampler)  # type: ignore[arg-type]
        return math.ceil(sampler_length / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class EpochIndexSampler(Sampler[int]):
    def __init__(
        self,
        dataset_size: int,
        *,
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        indices = list(range(self.dataset_size))
        if self.shuffle:
            order = _epoch_shuffled_order(self.dataset_size, seed=self.seed, epoch=self.epoch)
            indices = [indices[index] for index in order]
        yield from indices

    def __len__(self) -> int:
        return self.dataset_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class YomikomiDataLoader:
    def __init__(
        self,
        dataset: ASRDataset,
        *,
        batch_sampler: BatchSampler,
        collate_fn: Callable[[list[dict[str, Any] | None]], dict[str, Any] | None],
        num_workers: int = 0,
        prefetch_buffer_size: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.sampler = getattr(batch_sampler, "sampler", None)
        self.collate_fn = collate_fn
        self.num_workers = max(0, int(num_workers))
        self.prefetch_buffer_size = prefetch_buffer_size
        try:
            import yomikomi
        except ImportError as exc:
            raise ImportError(
                "Yomikomi DataLoader backend requested, but the 'yomikomi' package is not "
                "installed. Install yomikomi or use --dataloader-backend torch."
            ) from exc
        self._yomikomi = yomikomi

    def __iter__(self):
        stream = self._yomikomi.stream(
            _ThreadSafeIterator(self.batch_sampler),
            field="indices",
        )
        if self.num_workers > 0:
            kwargs: dict[str, Any] = {"num_threads": self.num_workers}
            if self.prefetch_buffer_size is not None:
                kwargs["buffer_size"] = self.prefetch_buffer_size
            stream = stream.prefetch(**kwargs)
        return self._iter_collated_batches(stream)

    def _iter_collated_batches(self, stream):
        for item in stream:
            indices = item["indices"]
            yield self.collate_fn([self.dataset[int(index)] for index in indices])

    def __len__(self) -> int:
        return len(self.batch_sampler)


class RustParquetFeatureDataLoader:
    def __init__(
        self,
        dataset: ASRDataset,
        *,
        batch_sampler: BatchSampler,
        num_workers: int = 0,
        prefetch_batches: int = 2,
        progress_logger: logging.Logger | None = None,
        progress_label: str = "dataloader",
    ) -> None:
        if dataset.feature_cache_dir is None:
            raise ValueError("rust-parquet backend requires dataset.feature_cache_dir")
        if dataset.feature_cache_format != "parquet":
            raise ValueError("rust-parquet backend requires feature_cache_format='parquet'")
        if dataset.return_waveforms:
            raise ValueError("rust-parquet backend does not support return_waveforms=True")
        if dataset.waveform_augment is not None and dataset.waveform_augment.is_enabled():
            raise ValueError("rust-parquet backend cannot be used with waveform augmentation")
        try:
            from asr_features import RustParquetFeatureCacheReader
        except ImportError as exc:
            raise ImportError(
                "rust-parquet DataLoader backend requested, but asr_features was not built "
                "with RustParquetFeatureCacheReader."
            ) from exc
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.sampler = getattr(batch_sampler, "sampler", None)
        self.num_workers = max(0, int(num_workers))
        self.prefetch_batches = max(1, int(prefetch_batches))
        self.progress_logger = progress_logger
        self.progress_label = progress_label
        self._logged_first_batch = False
        index_start_time = time.perf_counter()
        if self.progress_logger is not None:
            self.progress_logger.info(
                "%s rust-parquet feature cache index build started cache_dir=%s batches=%s "
                "num_workers=%s prefetch_batches=%s",
                self.progress_label,
                dataset.feature_cache_dir,
                len(batch_sampler),
                self.num_workers,
                self.prefetch_batches,
            )
        self.reader = RustParquetFeatureCacheReader(str(dataset.feature_cache_dir))
        if self.progress_logger is not None:
            self.progress_logger.info(
                "%s rust-parquet feature cache index ready entries=%s elapsed=%.1fs",
                self.progress_label,
                self.reader.len(),
                time.perf_counter() - index_start_time,
            )

    def __iter__(self):
        iterator = iter(self.batch_sampler)
        if self.num_workers <= 0:
            for indices in iterator:
                yield self._load_batch(list(indices))
            return
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = set()
            for _ in range(self.prefetch_batches):
                try:
                    indices = list(next(iterator))
                except StopIteration:
                    break
                futures.add(executor.submit(self._load_batch, indices))
            while futures:
                ready, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in ready:
                    yield future.result()
                    try:
                        indices = list(next(iterator))
                    except StopIteration:
                        continue
                    futures.add(executor.submit(self._load_batch, indices))

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def _load_batch(self, indices: list[int]) -> dict[str, Any] | None:
        start_time = time.perf_counter()
        records = [self.dataset.records[int(index)] for index in indices]
        keys = [self.dataset.feature_cache._key(record.utterance_id, self.dataset.featurizer) for record in records]  # type: ignore[union-attr]
        rust_features = self.reader.fetch_many(keys)
        items: list[dict[str, Any] | None] = []
        rust_hits = 0
        fallbacks = 0
        for index, record, features_array in zip(indices, records, rust_features, strict=True):
            if features_array is None:
                fallbacks += 1
                items.append(self.dataset[int(index)])
                continue
            features = torch.from_numpy(features_array).to(dtype=torch.float32)
            features = normalize_feature_tensor(features, self.dataset.featurizer.n_mels)
            if features is None or not feature_tensor_is_plausible(
                record,
                features,
                expected_feature_bins=self.dataset.featurizer.n_mels,
            ):
                fallbacks += 1
                items.append(self.dataset[int(index)])
                continue
            rust_hits += 1
            if self.dataset.specaugment is not None:
                features = self.dataset.specaugment(features)
            target_ids = torch.tensor(
                self.dataset.tokenizer.encode(record.transcript),
                dtype=torch.long,
            )
            items.append(
                {
                    "features": features,
                    "feature_length": features.size(0),
                    "feature_padding_value": float(
                        getattr(self.dataset.featurizer, "padding_value", 0.0)
                    ),
                    "targets": target_ids,
                    "target_length": target_ids.numel(),
                    "transcript": record.transcript,
                    "utterance_id": record.utterance_id,
                    "speaker_id": record.speaker_id,
                    "has_speaker_id": record.has_speaker_id,
                }
            )
        batch = collate_asr_batch(items)
        if self.progress_logger is not None and not self._logged_first_batch:
            self._logged_first_batch = True
            self.progress_logger.info(
                "%s rust-parquet first loaded batch size=%s rust_hits=%s fallbacks=%s "
                "elapsed=%.1fs",
                self.progress_label,
                len(indices),
                rust_hits,
                fallbacks,
                time.perf_counter() - start_time,
            )
        return batch


class _ThreadSafeIterator:
    def __init__(self, iterable) -> None:
        self._iterator = iter(iterable)
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            return next(self._iterator)


def _record_is_valid(record: AudioRecord) -> bool:
    try:
        if record.audio_path is not None and Path(record.audio_path).exists():
            try:
                torchaudio.info(record.audio_path)
                return True
            except ImportError:
                _load_wave_from_stdlib(record.audio_path)
                return True
        if record.audio_bytes is not None:
            try:
                torchaudio.info(io.BytesIO(record.audio_bytes))
                return True
            except ImportError:
                _load_wave_from_stdlib(io.BytesIO(record.audio_bytes))
                return True
    except Exception:
        return False
    return False


def prevalidate_records(
    records: list[AudioRecord],
    num_workers: int = 4,
) -> list[AudioRecord]:
    if num_workers <= 1:
        return [record for record in records if _record_is_valid(record)]

    validated_records: list[AudioRecord] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for record, is_valid in zip(records, executor.map(_record_is_valid, records), strict=True):
            if is_valid:
                validated_records.append(record)
    return validated_records


def estimate_record_frames(
    record: AudioRecord,
    hop_length: int,
    featurizer: AudioFeaturizer | None = None,
    force_audio_metadata_probe: bool = False,
) -> int:
    num_samples = int(record.num_samples)
    sample_rate = int(record.sample_rate)
    if force_audio_metadata_probe or num_samples <= 0 or sample_rate <= 0:
        num_samples, sample_rate = probe_audio_metadata(record.audio_path, record.audio_bytes)
    if num_samples <= 0 or sample_rate <= 0:
        return max(0, int(record.estimated_frames))
    object.__setattr__(record, "num_samples", num_samples)
    object.__setattr__(record, "sample_rate", sample_rate)
    estimated_frames = estimate_feature_frames_from_metadata(
        num_samples,
        sample_rate,
        hop_length=hop_length,
        featurizer=featurizer,
    )
    object.__setattr__(record, "estimated_frames", estimated_frames)
    return estimated_frames


def materialize_record_metadata(
    records: list[AudioRecord],
    hop_length: int,
    num_workers: int = 4,
    featurizer: AudioFeaturizer | None = None,
    force_audio_metadata_probe: bool = False,
    progress_logger: logging.Logger | None = None,
    progress_label: str = "records",
) -> list[AudioRecord]:
    def populate(record: AudioRecord) -> AudioRecord:
        estimate_record_frames(
            record,
            hop_length=hop_length,
            featurizer=featurizer,
            force_audio_metadata_probe=force_audio_metadata_probe,
        )
        return record

    total = len(records)
    start_time = time.perf_counter()
    if progress_logger is not None:
        progress_logger.info(
            "%s materializing audio metadata records=%s metadata_workers=%s force_probe=%s",
            progress_label,
            total,
            num_workers,
            force_audio_metadata_probe,
        )
    if num_workers <= 1:
        for completed, record in enumerate(records, start=1):
            populate(record)
            if progress_logger is not None and _should_log_progress(completed, total):
                _log_progress(
                    progress_logger,
                    f"{progress_label} materialized audio metadata",
                    completed,
                    total,
                    start_time,
                )
        return records
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for completed, _ in enumerate(executor.map(populate, records), start=1):
            if progress_logger is not None and _should_log_progress(completed, total):
                _log_progress(
                    progress_logger,
                    f"{progress_label} materialized audio metadata",
                    completed,
                    total,
                    start_time,
                )
    return records


def collate_asr_batch(batch: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    feature_lengths = torch.tensor([item["feature_length"] for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)
    max_feature_length = int(feature_lengths.max().item())

    padded_features = []
    targets = []
    transcripts = []
    utterance_ids = []
    speaker_ids = []
    has_speaker_ids = []
    include_waveforms = all("waveform" in item for item in batch)
    padded_waveforms = []
    waveform_lengths = []
    sample_rates = []
    max_waveform_length = (
        max(int(item["waveform_length"]) for item in batch) if include_waveforms else 0
    )
    for item in batch:
        feature = item["features"]
        if feature.size(0) < max_feature_length:
            feature = F.pad(
                feature,
                (0, 0, 0, max_feature_length - feature.size(0)),
                value=float(item.get("feature_padding_value", 0.0)),
            )
        padded_features.append(feature)
        targets.append(item["targets"])
        transcripts.append(item["transcript"])
        utterance_ids.append(item["utterance_id"])
        speaker_ids.append(item["speaker_id"])
        has_speaker_ids.append(item["has_speaker_id"])
        if include_waveforms:
            waveform = item["waveform"]
            if waveform.numel() < max_waveform_length:
                waveform = F.pad(waveform, (0, max_waveform_length - waveform.numel()))
            padded_waveforms.append(waveform)
            waveform_lengths.append(int(item["waveform_length"]))
            sample_rates.append(int(item["sample_rate"]))

    collated = {
        "features": torch.stack(padded_features, dim=0),
        "feature_lengths": feature_lengths,
        "targets": torch.cat(targets, dim=0),
        "target_lengths": target_lengths,
        "transcripts": transcripts,
        "utterance_ids": utterance_ids,
        "speaker_ids": speaker_ids,
        "has_speaker_ids": has_speaker_ids,
    }
    if include_waveforms:
        collated["waveforms"] = torch.stack(padded_waveforms, dim=0)
        collated["waveform_lengths"] = torch.tensor(waveform_lengths, dtype=torch.long)
        collated["sample_rates"] = torch.tensor(sample_rates, dtype=torch.long)
    return collated


def create_dataloader(
    dataset: ASRDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    bucket_by_length: bool = False,
    max_batch_duration_sec: float | None = None,
    max_batch_frames: int | None = None,
    adaptive_batch_unit: str | None = None,
    adaptive_batch_budget: int | None = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    metadata_workers: int = 4,
    force_audio_metadata_probe: bool = False,
    longest_batches_first: bool = False,
    multiprocessing_context: str | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    pad_distributed_batches: bool = False,
    in_order: bool = True,
    worker_threads: int | None = 1,
    backend: str = "torch",
    yomikomi_prefetch_buffer_size: int | None = None,
    rust_prefetch_batches: int = 2,
    progress_logger: logging.Logger | None = None,
    progress_label: str = "dataloader",
) -> DataLoader[dict[str, Any]] | YomikomiDataLoader | RustParquetFeatureDataLoader:
    stage_start_time = time.perf_counter()
    if progress_logger is not None:
        progress_logger.info(
            "%s create_dataloader started samples=%s batch_size=%s shuffle=%s "
            "num_workers=%s worker_threads=%s backend=%s",
            progress_label,
            len(dataset),
            batch_size,
            shuffle,
            num_workers,
            worker_threads if num_workers > 0 else "none",
            backend,
        )
    if hasattr(dataset.records, "populate_metadata"):
        dataset.records.populate_metadata(
            hop_length=dataset.featurizer.hop_length,
            num_workers=metadata_workers,
            featurizer=dataset.featurizer,
            force_audio_metadata_probe=force_audio_metadata_probe,
            progress_logger=progress_logger,
            progress_label=progress_label,
        )
    else:
        materialize_record_metadata(
            dataset.records,
            hop_length=dataset.featurizer.hop_length,
            num_workers=metadata_workers,
            featurizer=dataset.featurizer,
            force_audio_metadata_probe=force_audio_metadata_probe,
            progress_logger=progress_logger,
            progress_label=progress_label,
        )
    if progress_logger is not None:
        progress_logger.info(
            "%s metadata ready elapsed=%.1fs",
            progress_label,
            time.perf_counter() - stage_start_time,
        )
    if adaptive_batch_unit == "tokens" and hasattr(dataset.records, "populate_token_lengths"):
        dataset.records.populate_token_lengths(
            dataset.tokenizer,
            progress_logger=progress_logger,
            progress_label=progress_label,
        )
    if backend not in {"torch", "yomikomi", "rust-parquet"}:
        raise ValueError(
            "backend must be one of {'torch', 'yomikomi', 'rust-parquet'}, "
            f"got {backend!r}."
        )

    def make_loader(
        *,
        batch_sampler: BatchSampler | None = None,
        sampler: Sampler[int] | None = None,
        loader_batch_size: int | None = None,
        loader_shuffle: bool = False,
    ):
        if backend == "yomikomi":
            if batch_sampler is None:
                resolved_sampler = sampler
                if resolved_sampler is None:
                    resolved_sampler = EpochIndexSampler(
                        len(dataset),
                        shuffle=loader_shuffle,
                        seed=seed,
                    )
                batch_sampler = IndexBatchSampler(
                    resolved_sampler,
                    batch_size=batch_size if loader_batch_size is None else loader_batch_size,
                )
            return YomikomiDataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_asr_batch,
                num_workers=num_workers,
                prefetch_buffer_size=yomikomi_prefetch_buffer_size,
            )
        if backend == "rust-parquet":
            if batch_sampler is None:
                resolved_sampler = sampler
                if resolved_sampler is None:
                    resolved_sampler = EpochIndexSampler(
                        len(dataset),
                        shuffle=loader_shuffle,
                        seed=seed,
                    )
                batch_sampler = IndexBatchSampler(
                    resolved_sampler,
                    batch_size=batch_size if loader_batch_size is None else loader_batch_size,
                )
            return RustParquetFeatureDataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                prefetch_batches=rust_prefetch_batches,
                progress_logger=progress_logger,
                progress_label=progress_label,
            )
        if batch_sampler is not None:
            return DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_kwargs)
        return DataLoader(
            dataset,
            batch_size=batch_size if loader_batch_size is None else loader_batch_size,
            shuffle=loader_shuffle,
            sampler=sampler,
            **dataloader_kwargs,
        )

    dataloader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": collate_asr_batch,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
        dataloader_kwargs["in_order"] = in_order
        if worker_threads is not None and worker_threads > 0:
            dataloader_kwargs["worker_init_fn"] = functools.partial(
                _limit_dataloader_worker_threads,
                num_threads=int(worker_threads),
            )
        resolved_multiprocessing_context = _dataloader_multiprocessing_context(
            num_workers,
            multiprocessing_context=multiprocessing_context,
        )
        if resolved_multiprocessing_context is not None:
            dataloader_kwargs["multiprocessing_context"] = resolved_multiprocessing_context
    if adaptive_batch_unit is not None and adaptive_batch_budget is not None:
        batch_sampler = AdaptiveBatchSampler(
            dataset.records,
            target_batch_units=adaptive_batch_budget,
            unit=adaptive_batch_unit,
            shuffle=shuffle,
            longest_first=longest_batches_first,
            seed=seed,
            progress_logger=progress_logger,
            progress_label=progress_label,
        )
        if distributed and world_size > 1:
            batch_sampler = DistributedBatchSampler(
                batch_sampler,
                rank=rank,
                world_size=world_size,
                pad_to_world_size=pad_distributed_batches,
            )
        return make_loader(batch_sampler=batch_sampler)
    if max_batch_duration_sec is not None:
        batch_sampler = DurationBatchSampler(
            dataset.records,
            max_batch_duration_sec=max_batch_duration_sec,
            shuffle=shuffle,
            longest_first=longest_batches_first,
            seed=seed,
            progress_logger=progress_logger,
            progress_label=progress_label,
        )
        if distributed and world_size > 1:
            batch_sampler = DistributedBatchSampler(
                batch_sampler,
                rank=rank,
                world_size=world_size,
                pad_to_world_size=pad_distributed_batches,
            )
        return make_loader(batch_sampler=batch_sampler)
    if max_batch_frames is not None:
        batch_sampler = MaxFramesBatchSampler(
            dataset.records,
            max_batch_frames=max_batch_frames,
            shuffle=shuffle,
            longest_first=longest_batches_first,
            seed=seed,
            progress_logger=progress_logger,
            progress_label=progress_label,
        )
        if distributed and world_size > 1:
            batch_sampler = DistributedBatchSampler(
                batch_sampler,
                rank=rank,
                world_size=world_size,
                pad_to_world_size=pad_distributed_batches,
            )
        return make_loader(batch_sampler=batch_sampler)
    if bucket_by_length:
        batch_sampler = LengthBucketBatchSampler(
            dataset.records,
            batch_size=batch_size,
            shuffle=shuffle,
            longest_first=longest_batches_first,
            seed=seed,
            progress_logger=progress_logger,
            progress_label=progress_label,
        )
        if distributed and world_size > 1:
            batch_sampler = DistributedBatchSampler(
                batch_sampler,
                rank=rank,
                world_size=world_size,
                pad_to_world_size=pad_distributed_batches,
            )
        return make_loader(batch_sampler=batch_sampler)
    if distributed and world_size > 1:
        sampler = DistributedIndexSampler(
            len(dataset),
            rank=rank,
            world_size=world_size,
            shuffle=shuffle,
            seed=seed,
            pad_to_world_size=pad_distributed_batches,
        )
        return make_loader(sampler=sampler, loader_batch_size=batch_size, loader_shuffle=False)
    return make_loader(loader_batch_size=batch_size, loader_shuffle=shuffle)
