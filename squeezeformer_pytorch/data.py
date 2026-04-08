from __future__ import annotations

import hashlib
import io
import logging
import multiprocessing as mp
import os
import re
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
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
from torch.utils.data import BatchSampler, DataLoader, Dataset

from .asr import Tokenizer
from .frontend import AudioFeaturizer, SpecAugment, WaveformAugment

TRANSCRIPT_COLUMNS = ("sentence", "transcript", "transcription", "text", "normalized_text")
AUDIO_COLUMNS = ("path", "audio")


logger = logging.getLogger("train")


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
        "skipped_audio_too_short=%s skipped_audio_too_long=%s skipped_split=%s max_samples=%s",
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
        reader = pl.read_csv_batched(
            payload,
            separator="\t",
            infer_schema_length=1000,
            batch_size=batch_size,
        )
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            yield from batches[0].iter_rows(named=True)
        return
    if suffix == ".parquet":
        logger.info("loading remote parquet manifest %s", source_url)
        yield from pl.read_parquet(payload).iter_rows(named=True)
        return
    raise ValueError(f"Unsupported remote manifest format for {source_url}.")


def _iter_manifest_file_rows(path: Path, *, batch_size: int) -> Iterable[dict[str, Any]]:
    if path.suffix == ".tsv":
        reader = pl.read_csv_batched(
            path,
            separator="\t",
            infer_schema_length=1000,
            batch_size=batch_size,
        )
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            yield from batches[0].iter_rows(named=True)
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

    utterance_id = str(row.get("id") or audio_path or scanned_rows)
    raw_speaker_id = row.get("client_id") or row.get("speaker_id") or row.get("speaker")
    speaker_id = str(raw_speaker_id) if raw_speaker_id not in {None, ""} else None
    estimated_frames = max(1, int((duration_seconds * 16000) / 160))
    return AudioRecord(
        audio_path=audio_path,
        audio_bytes=audio_bytes,
        transcript=transcript,
        utterance_id=utterance_id,
        speaker_id=speaker_id,
        has_speaker_id=speaker_id is not None,
        estimated_frames=estimated_frames,
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
        return 0, 0

    try:
        waveform, sample_rate = load_audio(audio_path, audio_bytes)
    except Exception:
        return 0, 0
    return int(waveform.size(-1)), int(sample_rate)


def load_audio(audio_path: str | None, audio_bytes: bytes | None) -> tuple[Tensor, int]:
    try:
        if audio_path is not None and Path(audio_path).exists():
            return torchaudio.load(audio_path)
        if audio_bytes is not None:
            return torchaudio.load(io.BytesIO(audio_bytes))
    except ImportError:
        pass

    if audio_path is not None and Path(audio_path).exists():
        return _load_wave_from_stdlib(audio_path)
    if audio_bytes is not None:
        return _load_wave_from_stdlib(io.BytesIO(audio_bytes))
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


class ASRDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[AudioRecord],
        tokenizer: Tokenizer,
        featurizer: AudioFeaturizer,
        specaugment: SpecAugment | None = None,
        waveform_augment: WaveformAugment | None = None,
        feature_cache_dir: str | Path | None = None,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.featurizer = featurizer
        self.specaugment = specaugment
        self.waveform_augment = waveform_augment
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir is not None else None
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

    def _feature_cache_path(self, record: AudioRecord) -> Path | None:
        return feature_cache_path(self.feature_cache_dir, record.utterance_id, self.featurizer)

    def _compute_features(
        self,
        record: AudioRecord,
        *,
        waveform_augment_enabled: bool,
    ) -> Tensor:
        waveform, sample_rate = load_audio(record.audio_path, record.audio_bytes)
        if waveform_augment_enabled and self.waveform_augment is not None:
            waveform, sample_rate = self.waveform_augment(waveform, sample_rate)
        return self.featurizer(waveform, sample_rate)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any] | None:
        record = self.records[index]
        cache_path = self._feature_cache_path(record)
        waveform_augment_enabled = (
            self.waveform_augment is not None and self.waveform_augment.is_enabled()
        )
        use_cache = cache_path is not None and not waveform_augment_enabled
        if use_cache and cache_path.exists():
            features = torch.load(cache_path, map_location="cpu")
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
                cache_path.unlink(missing_ok=True)
                features = self._compute_features(
                    record,
                    waveform_augment_enabled=waveform_augment_enabled,
                )
                if use_cache:
                    torch.save(features, cache_path)
        else:
            features = self._compute_features(
                record,
                waveform_augment_enabled=waveform_augment_enabled,
            )
            if use_cache:
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
        return {
            "features": features,
            "feature_length": features.size(0),
            "targets": target_ids,
            "target_length": target_ids.numel(),
            "transcript": record.transcript,
            "utterance_id": record.utterance_id,
            "speaker_id": record.speaker_id,
            "has_speaker_id": record.has_speaker_id,
        }


class LengthBucketBatchSampler(BatchSampler):
    def __init__(
        self,
        records: list[AudioRecord],
        batch_size: int,
        shuffle: bool,
        longest_first: bool = False,
    ) -> None:
        self.records = records
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.longest_first = longest_first

    def _batch_order_key(self, batch: list[int]) -> tuple[int, int]:
        max_frames = max(max(1, self.records[index].estimated_frames) for index in batch)
        return max_frames, len(batch)

    def __iter__(self):
        indices = sorted(
            range(len(self.records)), key=lambda index: self.records[index].estimated_frames
        )
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        if self.longest_first:
            batches.sort(key=self._batch_order_key, reverse=True)
        elif self.shuffle:
            order = torch.randperm(len(batches)).tolist()
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self) -> int:
        return (len(self.records) + self.batch_size - 1) // self.batch_size


class MaxFramesBatchSampler(BatchSampler):
    def __init__(
        self,
        records: list[AudioRecord],
        max_batch_frames: int,
        shuffle: bool,
        longest_first: bool = False,
    ) -> None:
        self.records = records
        self.max_batch_frames = max_batch_frames
        self.shuffle = shuffle
        self.longest_first = longest_first
        self._sorted_indices = sorted(
            range(len(self.records)), key=lambda index: self.records[index].estimated_frames
        )
        self._num_batches = self._count_batches()

    def _batch_order_key(self, batch: list[int]) -> tuple[int, int]:
        max_frames = max(max(1, self.records[index].estimated_frames) for index in batch)
        return len(batch) * max_frames, max_frames

    def _iter_batches(self) -> Iterable[list[int]]:
        current_batch: list[int] = []
        current_max = 0
        for index in self._sorted_indices:
            frames = max(1, self.records[index].estimated_frames)
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

    def _count_batches(self) -> int:
        return sum(1 for _ in self._iter_batches())

    def __iter__(self):
        batches = list(self._iter_batches())
        if self.longest_first:
            batches.sort(key=self._batch_order_key, reverse=True)
        elif self.shuffle:
            order = torch.randperm(len(batches)).tolist()
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self) -> int:
        return self._num_batches


class AdaptiveBatchSampler(BatchSampler):
    def __init__(
        self,
        records: list[AudioRecord],
        target_batch_units: int,
        unit: str,
        shuffle: bool,
        longest_first: bool = False,
    ) -> None:
        if unit not in {"frames", "tokens"}:
            raise ValueError("unit must be one of {'frames', 'tokens'}")
        self.records = records
        self.target_batch_units = target_batch_units
        self.unit = unit
        self.shuffle = shuffle
        self.longest_first = longest_first
        self._sorted_indices = sorted(
            range(len(self.records)),
            key=lambda index: (
                self.records[index].estimated_frames,
                len(self.records[index].transcript),
            ),
        )
        self._num_batches = self._count_batches()

    def _record_units(self, record: AudioRecord) -> int:
        if self.unit == "frames":
            return max(1, record.estimated_frames)
        return max(1, len(record.transcript))

    def _iter_batches(self) -> Iterable[list[int]]:
        current_batch: list[int] = []
        current_units = 0
        for index in self._sorted_indices:
            units = self._record_units(self.records[index])
            if current_batch and current_units + units > self.target_batch_units:
                yield current_batch
                current_batch = []
                current_units = 0
            current_batch.append(index)
            current_units += units
        if current_batch:
            yield current_batch

    def _batch_order_key(self, batch: list[int]) -> tuple[int, int]:
        total_units = sum(self._record_units(self.records[index]) for index in batch)
        max_frames = max(max(1, self.records[index].estimated_frames) for index in batch)
        return total_units, max_frames

    def _count_batches(self) -> int:
        return sum(1 for _ in self._iter_batches())

    def __iter__(self):
        batches = list(self._iter_batches())
        if self.longest_first:
            batches.sort(key=self._batch_order_key, reverse=True)
        elif self.shuffle:
            order = torch.randperm(len(batches)).tolist()
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self) -> int:
        return self._num_batches


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


def estimate_record_frames(record: AudioRecord, hop_length: int) -> int:
    if record.estimated_frames > 0 and record.num_samples > 0 and record.sample_rate > 0:
        return record.estimated_frames
    num_samples, sample_rate = probe_audio_metadata(record.audio_path, record.audio_bytes)
    if num_samples <= 0 or sample_rate <= 0:
        return 0
    object.__setattr__(record, "num_samples", num_samples)
    object.__setattr__(record, "sample_rate", sample_rate)
    estimated_frames = max(1, int(num_samples / hop_length))
    object.__setattr__(record, "estimated_frames", estimated_frames)
    return estimated_frames


def materialize_record_metadata(
    records: list[AudioRecord],
    hop_length: int,
    num_workers: int = 4,
) -> list[AudioRecord]:
    def populate(record: AudioRecord) -> AudioRecord:
        estimate_record_frames(record, hop_length=hop_length)
        return record

    if num_workers <= 1:
        return [populate(record) for record in records]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for _ in executor.map(populate, records):
            pass
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
    for item in batch:
        feature = item["features"]
        if feature.size(0) < max_feature_length:
            feature = F.pad(feature, (0, 0, 0, max_feature_length - feature.size(0)))
        padded_features.append(feature)
        targets.append(item["targets"])
        transcripts.append(item["transcript"])
        utterance_ids.append(item["utterance_id"])
        speaker_ids.append(item["speaker_id"])
        has_speaker_ids.append(item["has_speaker_id"])

    return {
        "features": torch.stack(padded_features, dim=0),
        "feature_lengths": feature_lengths,
        "targets": torch.cat(targets, dim=0),
        "target_lengths": target_lengths,
        "transcripts": transcripts,
        "utterance_ids": utterance_ids,
        "speaker_ids": speaker_ids,
        "has_speaker_ids": has_speaker_ids,
    }


def create_dataloader(
    dataset: ASRDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    bucket_by_length: bool = False,
    max_batch_frames: int | None = None,
    adaptive_batch_unit: str | None = None,
    adaptive_batch_budget: int | None = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    metadata_workers: int = 4,
    longest_batches_first: bool = False,
    multiprocessing_context: str | None = None,
) -> DataLoader[dict[str, Any]]:
    if hasattr(dataset.records, "populate_metadata"):
        dataset.records.populate_metadata(
            hop_length=dataset.featurizer.hop_length,
            num_workers=metadata_workers,
        )
    else:
        materialize_record_metadata(
            dataset.records,
            hop_length=dataset.featurizer.hop_length,
            num_workers=metadata_workers,
        )
    dataloader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": collate_asr_batch,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
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
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_kwargs)
    if max_batch_frames is not None:
        batch_sampler = MaxFramesBatchSampler(
            dataset.records,
            max_batch_frames=max_batch_frames,
            shuffle=shuffle,
            longest_first=longest_batches_first,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_kwargs)
    if bucket_by_length:
        batch_sampler = LengthBucketBatchSampler(
            dataset.records,
            batch_size=batch_size,
            shuffle=shuffle,
            longest_first=longest_batches_first,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **dataloader_kwargs,
    )
