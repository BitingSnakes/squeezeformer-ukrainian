from __future__ import annotations

import argparse
import array
import base64
import hashlib
import json
import mmap
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.distributed as dist

from squeezeformer_pytorch.data import (
    AudioFeaturizer,
    AudioRecord,
    download_dataset,
    estimate_feature_frames_from_metadata,
    iter_records,
    iter_records_from_source,
    load_audio,
    prevalidate_records,
    probe_audio_metadata,
    read_binary_source,
)


def _progress_log_interval(total: int) -> int:
    if total <= 0:
        return 1
    return max(1_000, min(50_000, total // 20 or 1))


def _log_metadata_progress(
    progress_logger: Logger,
    message: str,
    completed: int,
    total: int,
    start_time: float,
) -> None:
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


def _resolve_dataset_roots(args: argparse.Namespace) -> list[Path]:
    sources = list(args.dataset_source or [])
    if not sources:
        sources = [args.dataset_repo]

    dataset_roots: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        dataset_root = download_dataset(
            repo_id=source,
            token=args.hf_token,
            cache_dir=args.cache_dir,
        ).resolve()
        if dataset_root in seen:
            continue
        seen.add(dataset_root)
        dataset_roots.append(dataset_root)
    return dataset_roots


def _resolve_sources(
    raw_sources: list[str] | None, *, fallback: str | None = None
) -> list[str | Path]:
    sources = list(raw_sources or [])
    if not sources:
        if fallback is None:
            return []
        sources = [fallback]

    resolved_sources: list[str | Path] = []
    seen: set[str] = set()
    for source in sources:
        source_path = Path(source).expanduser()
        resolved_source: str | Path = source_path.resolve() if source_path.exists() else source
        dedupe_key = str(resolved_source)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        resolved_sources.append(resolved_source)
    return resolved_sources


def _resolve_dataset_sources(args: argparse.Namespace) -> list[str | Path]:
    return _resolve_sources(args.dataset_source, fallback=args.dataset_repo)


def _resolve_validation_dataset_sources(args: argparse.Namespace) -> list[str | Path]:
    return _resolve_sources(args.validation_dataset_source)


class DiskBackedRecordStore:
    def __init__(
        self,
        records_path: Path,
        offsets: array.array,
        estimated_frames: array.array,
        num_samples: array.array | None = None,
        sample_rates: array.array | None = None,
        *,
        start: int = 0,
        step: int = 1,
        count: int | None = None,
    ) -> None:
        self.records_path = records_path
        self.offsets = offsets
        self.estimated_frames = estimated_frames
        self.num_samples = num_samples
        self.sample_rates = sample_rates
        self.start = start
        self.step = step
        self.count = count
        self._handle = None
        self._handle_pid: int | None = None

    def _global_index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.start + (index * self.step)

    def _open_handle(self):
        current_pid = os.getpid()
        if self._handle_pid != current_pid and self._handle is not None and not self._handle.closed:
            self._handle.close()
            self._handle = None
        if self._handle is None or self._handle.closed:
            self._handle = self.records_path.open("rb")
            self._handle_pid = current_pid
        return self._handle

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        self._handle = None
        self._handle_pid = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._handle = None
        self._handle_pid = None

    def __len__(self) -> int:
        total = len(self.offsets)
        if self.start >= total:
            return 0
        available = ((total - self.start - 1) // self.step) + 1
        if self.count is None:
            return available
        return max(0, min(available, self.count))

    def __getitem__(self, index: int) -> AudioRecord:
        global_index = self._global_index(index)
        handle = self._open_handle()
        handle.seek(self.offsets[global_index])
        payload = json.loads(handle.readline().decode("utf-8"))
        return AudioRecord(
            audio_path=payload["audio_path"],
            audio_bytes=_load_cached_audio_bytes(payload, records_path=self.records_path),
            transcript=payload["transcript"],
            utterance_id=payload["utterance_id"],
            estimated_frames=int(self.estimated_frames[global_index]),
            speaker_id=payload["speaker_id"],
            has_speaker_id=bool(payload["has_speaker_id"]),
            num_samples=(
                int(self.num_samples[global_index]) if self.num_samples is not None else 0
            ),
            sample_rate=(
                int(self.sample_rates[global_index]) if self.sample_rates is not None else 0
            ),
        )

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def shard(
        self, rank: int, world_size: int, *, allow_uneven: bool = False
    ) -> "DiskBackedRecordStore":
        local_length = len(self)
        if allow_uneven:
            local_count = max(0, (local_length - rank + world_size - 1) // world_size)
        else:
            local_count = local_length // world_size
        return DiskBackedRecordStore(
            self.records_path,
            self.offsets,
            self.estimated_frames,
            self.num_samples,
            self.sample_rates,
            start=self.start + rank,
            step=self.step * world_size,
            count=local_count,
        )

    def populate_metadata(
        self,
        hop_length: int,
        num_workers: int = 4,
        featurizer: AudioFeaturizer | None = None,
        force_audio_metadata_probe: bool = False,
        progress_logger: Logger | None = None,
        progress_label: str = "records",
    ) -> None:
        total = len(self)
        start_time = time.perf_counter()
        if (
            not force_audio_metadata_probe
            and self.num_samples is not None
            and self.sample_rates is not None
        ):
            if progress_logger is not None:
                progress_logger.info(
                    "%s estimating frames from cached audio metadata records=%s",
                    progress_label,
                    total,
                )
            for completed, index in enumerate(range(total), start=1):
                global_index = self._global_index(index)
                num_samples = int(self.num_samples[global_index])
                sample_rate = int(self.sample_rates[global_index])
                if num_samples <= 0 or sample_rate <= 0:
                    if progress_logger is not None and (
                        completed == total or completed % _progress_log_interval(total) == 0
                    ):
                        _log_metadata_progress(
                            progress_logger,
                            f"{progress_label} estimated frames from cached metadata",
                            completed,
                            total,
                            start_time,
                        )
                    continue
                frames = estimate_feature_frames_from_metadata(
                    num_samples,
                    sample_rate,
                    hop_length=hop_length,
                    featurizer=featurizer,
                )
                self.estimated_frames[global_index] = max(0, int(frames))
                if progress_logger is not None and (
                    completed == total or completed % _progress_log_interval(total) == 0
                ):
                    _log_metadata_progress(
                        progress_logger,
                        f"{progress_label} estimated frames from cached metadata",
                        completed,
                        total,
                        start_time,
                    )
        if force_audio_metadata_probe:
            global_indices = [self._global_index(index) for index in range(len(self))]
        else:
            global_indices = [
                self._global_index(index)
                for index in range(len(self))
                if (
                    int(self.estimated_frames[self._global_index(index)]) <= 0
                    or self.num_samples is None
                    or int(self.num_samples[self._global_index(index)]) <= 0
                    or self.sample_rates is None
                    or int(self.sample_rates[self._global_index(index)]) <= 0
                )
            ]
        if not global_indices:
            if progress_logger is not None:
                progress_logger.info(
                    "%s audio metadata complete records=%s probe_needed=0 elapsed=%.1fs",
                    progress_label,
                    total,
                    time.perf_counter() - start_time,
                )
            return

        def populate(global_index: int) -> tuple[int, int, int, int]:
            handle = self.records_path.open("rb")
            try:
                handle.seek(self.offsets[global_index])
                payload = json.loads(handle.readline().decode("utf-8"))
            finally:
                handle.close()
            audio_bytes = _load_cached_audio_bytes(payload, records_path=self.records_path)
            num_samples, sample_rate = probe_audio_metadata(payload["audio_path"], audio_bytes)
            frames = estimate_feature_frames_from_metadata(
                num_samples,
                sample_rate,
                hop_length=hop_length,
                featurizer=featurizer,
            )
            return global_index, frames, num_samples, sample_rate

        probe_total = len(global_indices)
        if progress_logger is not None:
            progress_logger.info(
                "%s probing missing audio metadata records=%s metadata_workers=%s force_probe=%s",
                progress_label,
                probe_total,
                num_workers,
                force_audio_metadata_probe,
            )

        def store_populated(global_index: int, frames: int, num_samples: int, sample_rate: int):
            self.estimated_frames[global_index] = max(0, int(frames))
            if self.num_samples is not None:
                self.num_samples[global_index] = max(0, int(num_samples))
            if self.sample_rates is not None:
                self.sample_rates[global_index] = max(0, int(sample_rate))

        probe_start_time = time.perf_counter()
        if num_workers <= 1:
            iterator = (populate(global_index) for global_index in global_indices)
            for completed, (global_index, frames, num_samples, sample_rate) in enumerate(
                iterator, start=1
            ):
                store_populated(global_index, frames, num_samples, sample_rate)
                if progress_logger is not None and (
                    completed == probe_total or completed % _progress_log_interval(probe_total) == 0
                ):
                    _log_metadata_progress(
                        progress_logger,
                        f"{progress_label} probed audio metadata",
                        completed,
                        probe_total,
                        probe_start_time,
                    )
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for completed, (global_index, frames, num_samples, sample_rate) in enumerate(
                    executor.map(populate, global_indices), start=1
                ):
                    store_populated(global_index, frames, num_samples, sample_rate)
                    if progress_logger is not None and (
                        completed == probe_total
                        or completed % _progress_log_interval(probe_total) == 0
                    ):
                        _log_metadata_progress(
                            progress_logger,
                            f"{progress_label} probed audio metadata",
                            completed,
                            probe_total,
                            probe_start_time,
                        )


class _BinaryIndexView:
    def __init__(
        self,
        path: Path,
        *,
        item_size: int,
        fmt: str,
        writable: bool = False,
    ) -> None:
        self.path = path
        self.item_size = item_size
        self.fmt = fmt
        self.writable = writable
        self._length = path.stat().st_size // item_size if path.exists() else 0
        self._handle = None
        self._mmap = None
        self._pid: int | None = None

    def _ensure_open(self):
        current_pid = os.getpid()
        if self._pid != current_pid:
            self.close()
        elif self._mmap is not None:
            try:
                self._mmap.size()
            except (BufferError, OSError, ValueError):
                self.close()
        if self._mmap is None:
            mode = "r+b" if self.writable else "rb"
            self._handle = self.path.open(mode)
            access = mmap.ACCESS_WRITE if self.writable else mmap.ACCESS_READ
            self._mmap = mmap.mmap(self._handle.fileno(), length=0, access=access)
            self._pid = current_pid
        return self._mmap

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        self._mmap = None
        self._handle = None
        self._pid = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_mmap"] = None
        state["_handle"] = None
        state["_pid"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._mmap = None
        self._handle = None
        self._pid = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> int:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        return int(struct.unpack_from(self.fmt, self._ensure_open(), index * self.item_size)[0])

    def __setitem__(self, index: int, value: int) -> None:
        if not self.writable:
            raise TypeError("Index view is read-only.")
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        struct.pack_into(self.fmt, self._ensure_open(), index * self.item_size, int(value))

    def __iter__(self):
        for index in range(self._length):
            yield self[index]


def _record_index_path(records_path: Path, suffix: str) -> Path:
    return records_path.with_suffix(records_path.suffix + suffix)


def _build_disk_backed_record_store(
    dataset_sources: list[str | Path],
    *,
    split: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    max_samples: int | None,
    min_transcript_chars: int,
    max_transcript_chars: int,
    max_symbol_ratio: float,
    lowercase_transcripts: bool,
    records_path: Path,
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
    hf_token: str | None = None,
    cache_dir: str | None = None,
) -> DiskBackedRecordStore:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    audio_blob_dir = records_path.parent / f"{records_path.stem}_audio_blobs"
    offsets_path = _record_index_path(records_path, ".offsets.u64")
    estimated_frames_path = _record_index_path(records_path, ".estimated_frames.u32")
    num_samples_path = _record_index_path(records_path, ".num_samples.u64")
    sample_rates_path = _record_index_path(records_path, ".sample_rates.u32")
    written = 0
    with (
        records_path.open("wb") as handle,
        offsets_path.open("wb") as offsets_handle,
        estimated_frames_path.open("wb") as estimated_frames_handle,
        num_samples_path.open("wb") as num_samples_handle,
        sample_rates_path.open("wb") as sample_rates_handle,
    ):
        for dataset_source in dataset_sources:
            remaining_samples = None
            if max_samples is not None:
                remaining_samples = max_samples - written
                if remaining_samples <= 0:
                    break
            record_iterator = (
                iter_records(
                    dataset_root=dataset_source,
                    split=split,
                    seed=seed,
                    val_fraction=val_fraction,
                    test_fraction=test_fraction,
                    max_samples=remaining_samples,
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
                if isinstance(dataset_source, Path)
                and dataset_source.exists()
                and dataset_source.is_dir()
                else iter_records_from_source(
                    dataset_source,
                    split=split,
                    seed=seed,
                    val_fraction=val_fraction,
                    test_fraction=test_fraction,
                    max_samples=remaining_samples,
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
                    hf_token=hf_token,
                    cache_dir=cache_dir,
                )
            )
            for record in record_iterator:
                audio_bytes = record.audio_bytes
                if audio_bytes is None and record.audio_path is not None:
                    try:
                        audio_bytes = read_binary_source(
                            record.audio_path,
                            token=hf_token,
                            cache_dir=cache_dir,
                        )
                    except Exception:
                        audio_bytes = None
                preserve_audio_bytes = audio_bytes is not None and not (
                    record.audio_path is not None and Path(record.audio_path).exists()
                )
                audio_blob_path: str | None = None
                if preserve_audio_bytes:
                    audio_blob_dir.mkdir(parents=True, exist_ok=True)
                    blob_name = hashlib.sha256(audio_bytes).hexdigest() + ".bin"
                    blob_path = audio_blob_dir / blob_name
                    if not blob_path.exists():
                        blob_path.write_bytes(audio_bytes)
                    audio_blob_path = str(blob_path.relative_to(records_path.parent))
                offsets_handle.write(struct.pack("<Q", handle.tell()))
                payload = json.dumps(
                    {
                        "audio_path": record.audio_path,
                        "audio_blob_path": audio_blob_path,
                        "transcript": record.transcript,
                        "utterance_id": record.utterance_id,
                        "speaker_id": record.speaker_id,
                        "has_speaker_id": record.has_speaker_id,
                    },
                    ensure_ascii=False,
                )
                handle.write(payload.encode("utf-8"))
                handle.write(b"\n")
                estimated_frames_handle.write(
                    struct.pack("<I", max(0, int(record.estimated_frames)))
                )
                num_samples_handle.write(struct.pack("<Q", max(0, int(record.num_samples))))
                sample_rates_handle.write(struct.pack("<I", max(0, int(record.sample_rate))))
                written += 1
    if written == 0:
        raise RuntimeError(
            f"Split '{split}' is empty after applying the current split fractions across "
            "all dataset sources."
        )
    return DiskBackedRecordStore(
        records_path,
        _BinaryIndexView(offsets_path, item_size=8, fmt="<Q"),
        _BinaryIndexView(
            estimated_frames_path,
            item_size=4,
            fmt="<I",
            writable=True,
        ),
        _BinaryIndexView(
            num_samples_path,
            item_size=8,
            fmt="<Q",
            writable=True,
        ),
        _BinaryIndexView(
            sample_rates_path,
            item_size=4,
            fmt="<I",
            writable=True,
        ),
    )


def _open_disk_backed_record_store(records_path: Path) -> DiskBackedRecordStore:
    offsets_path = _record_index_path(records_path, ".offsets.u64")
    estimated_frames_path = _record_index_path(records_path, ".estimated_frames.u32")
    num_samples_path = _record_index_path(records_path, ".num_samples.u64")
    sample_rates_path = _record_index_path(records_path, ".sample_rates.u32")
    return DiskBackedRecordStore(
        records_path,
        _BinaryIndexView(offsets_path, item_size=8, fmt="<Q"),
        _BinaryIndexView(
            estimated_frames_path,
            item_size=4,
            fmt="<I",
            writable=True,
        ),
        (
            _BinaryIndexView(
                num_samples_path,
                item_size=8,
                fmt="<Q",
                writable=True,
            )
            if num_samples_path.exists()
            else None
        ),
        (
            _BinaryIndexView(
                sample_rates_path,
                item_size=4,
                fmt="<I",
                writable=True,
            )
            if sample_rates_path.exists()
            else None
        ),
    )


def _load_cached_audio_bytes(payload: dict[str, object], *, records_path: Path) -> bytes | None:
    audio_blob_path = payload.get("audio_blob_path")
    if isinstance(audio_blob_path, str) and audio_blob_path:
        blob_path = Path(audio_blob_path)
        if not blob_path.is_absolute():
            blob_path = records_path.parent / blob_path
        return blob_path.read_bytes()
    audio_bytes_payload = payload.get("audio_bytes")
    if isinstance(audio_bytes_payload, str) and audio_bytes_payload:
        return base64.b64decode(audio_bytes_payload)
    return None


_OPUS_HEADER_SCAN_BYTES = 64 * 1024


def _audio_header_looks_like_opus(header: bytes) -> bool:
    return b"OpusHead" in header[:_OPUS_HEADER_SCAN_BYTES]


def _read_header_from_path(audio_path: str | None) -> bytes:
    if not audio_path:
        return b""
    try:
        path = Path(audio_path)
    except OSError:
        return b""
    if not path.exists():
        return b""
    try:
        with path.open("rb") as handle:
            return handle.read(_OPUS_HEADER_SCAN_BYTES)
    except OSError:
        return b""


def _path_declares_opus(audio_path: str | None) -> bool:
    if not audio_path:
        return False
    return Path(urlparse(audio_path).path).suffix.lower() == ".opus"


def _record_looks_like_opus(record: AudioRecord) -> bool:
    if _path_declares_opus(record.audio_path):
        return True
    if record.audio_bytes is not None and _audio_header_looks_like_opus(record.audio_bytes):
        return True
    return _audio_header_looks_like_opus(_read_header_from_path(record.audio_path))


def _find_opus_probe_record(
    records: list[AudioRecord] | DiskBackedRecordStore,
) -> tuple[str | None, bytes | None, str] | None:
    if isinstance(records, DiskBackedRecordStore):
        with records.records_path.open("rb") as handle:
            for global_index in range(records.start, len(records.offsets), records.step):
                handle.seek(records.offsets[global_index])
                payload = json.loads(handle.readline().decode("utf-8"))
                audio_path = payload.get("audio_path")
                if not isinstance(audio_path, str):
                    audio_path = None
                audio_blob_path = payload.get("audio_blob_path")
                header = b""
                if isinstance(audio_blob_path, str) and audio_blob_path:
                    blob_path = Path(audio_blob_path)
                    if not blob_path.is_absolute():
                        blob_path = records.records_path.parent / blob_path
                    header = _read_header_from_path(str(blob_path))
                elif audio_path is not None:
                    header = _read_header_from_path(audio_path)
                if _path_declares_opus(audio_path) or _audio_header_looks_like_opus(header):
                    return (
                        audio_path,
                        _load_cached_audio_bytes(payload, records_path=records.records_path),
                        audio_path or f"blob:{audio_blob_path}",
                    )
        return None

    for record in records:
        if _record_looks_like_opus(record):
            return record.audio_path, record.audio_bytes, record.audio_path or record.utterance_id
    return None


def _ensure_opus_decode_support(
    records: list[AudioRecord] | DiskBackedRecordStore,
    *,
    split: str,
) -> None:
    probe = _find_opus_probe_record(records)
    if probe is None:
        return
    audio_path, audio_bytes, description = probe
    try:
        load_audio(audio_path, audio_bytes)
    except Exception as error:
        raise RuntimeError(
            f"Split '{split}' contains Opus audio ({description}), but this runtime cannot decode "
            "it with torchaudio. Install torchaudio with Opus/FFmpeg support or convert those "
            "files to WAV before training."
        ) from error


def _load_records_from_dataset_roots(
    dataset_sources: list[str | Path],
    *,
    split: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    max_samples: int | None,
    min_transcript_chars: int,
    max_transcript_chars: int,
    max_symbol_ratio: float,
    lowercase_transcripts: bool,
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
    hf_token: str | None = None,
    cache_dir: str | None = None,
) -> list:
    records = []
    for dataset_source in dataset_sources:
        remaining_samples = None
        if max_samples is not None:
            remaining_samples = max_samples - len(records)
            if remaining_samples <= 0:
                break
        iterator = (
            iter_records(
                dataset_root=dataset_source,
                split=split,
                seed=seed,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                max_samples=remaining_samples,
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
            if isinstance(dataset_source, Path)
            and dataset_source.exists()
            and dataset_source.is_dir()
            else iter_records_from_source(
                dataset_source,
                split=split,
                seed=seed,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                max_samples=remaining_samples,
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
                hf_token=hf_token,
                cache_dir=cache_dir,
            )
        )
        records.extend(iterator)
    if not records:
        raise RuntimeError(
            f"Split '{split}' is empty after applying the current split fractions across "
            "all dataset sources."
        )
    return records


def _prevalidate_loaded_records(
    records: list[AudioRecord],
    *,
    split: str,
    num_workers: int,
) -> list[AudioRecord]:
    validated_records = prevalidate_records(records, num_workers=num_workers)
    if not validated_records:
        raise RuntimeError(f"Split '{split}' is empty after audio prevalidation.")
    return validated_records


def _distributed_barrier() -> None:
    if not dist.is_initialized():
        return
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        dist.barrier(device_ids=[torch.cuda.current_device()])
        return
    dist.barrier()


def _load_train_val_records(
    args: argparse.Namespace,
    train_dataset_sources: list[str | Path],
    validation_dataset_sources: list[str | Path] | None = None,
    *,
    lowercase_transcripts: bool,
    output_dir: Path,
    distributed: bool = False,
    is_main_process: bool = True,
) -> tuple[list[AudioRecord] | DiskBackedRecordStore, list[AudioRecord] | DiskBackedRecordStore]:
    validation_dataset_sources = validation_dataset_sources or []
    use_external_validation = bool(validation_dataset_sources)
    train_val_fraction = 0.0 if use_external_validation else args.val_fraction
    train_test_fraction = 0.0 if use_external_validation else args.test_fraction
    validation_split = "train" if use_external_validation else "validation"
    validation_val_fraction = 0.0 if use_external_validation else args.val_fraction
    validation_test_fraction = 0.0 if use_external_validation else args.test_fraction

    if args.record_cache:
        common_record_store_kwargs = {
            "seed": args.seed,
            "min_transcript_chars": args.min_transcript_chars,
            "max_transcript_chars": args.max_transcript_chars,
            "max_symbol_ratio": args.max_symbol_ratio,
            "lowercase_transcripts": lowercase_transcripts,
            "hf_token": args.hf_token,
            "cache_dir": args.cache_dir,
            "min_chars_per_second": args.min_chars_per_second,
            "max_chars_per_second": args.max_chars_per_second,
            "min_words_per_second": args.min_words_per_second,
            "max_words_per_second": args.max_words_per_second,
            "min_duration_per_char": args.min_duration_per_char,
            "max_duration_per_char": args.max_duration_per_char,
            "min_duration_per_word": args.min_duration_per_word,
            "max_duration_per_word": args.max_duration_per_word,
        }
        if hasattr(args, "min_audio_duration_sec"):
            common_record_store_kwargs["min_audio_duration_sec"] = args.min_audio_duration_sec
        if hasattr(args, "max_audio_duration_sec"):
            common_record_store_kwargs["max_audio_duration_sec"] = args.max_audio_duration_sec
        record_store_dir = (
            Path(args.record_cache_dir)
            if args.record_cache_dir is not None
            else output_dir / "record_cache"
        )
        train_records_path = record_store_dir / "train.jsonl"
        val_records_path = record_store_dir / "validation.jsonl"
        if distributed and not is_main_process:
            if not dist.is_initialized():
                raise RuntimeError(
                    "Distributed record cache loading requires initialized process group."
                )
            _distributed_barrier()
            train_records = _open_disk_backed_record_store(train_records_path)
            val_records = _open_disk_backed_record_store(val_records_path)
        else:
            train_records = _build_disk_backed_record_store(
                train_dataset_sources,
                split="train",
                val_fraction=train_val_fraction,
                test_fraction=train_test_fraction,
                max_samples=args.max_train_samples,
                records_path=train_records_path,
                **common_record_store_kwargs,
            )
            val_records = _build_disk_backed_record_store(
                validation_dataset_sources or train_dataset_sources,
                split=validation_split,
                val_fraction=validation_val_fraction,
                test_fraction=validation_test_fraction,
                max_samples=args.max_val_samples,
                records_path=val_records_path,
                **common_record_store_kwargs,
            )
            if distributed and dist.is_initialized():
                _distributed_barrier()
        if args.prevalidate_audio:
            raise ValueError(
                "--prevalidate-audio is not supported with the disk-backed training record store. "
                "Leave it disabled for large multi-source training runs or use --no-record-cache."
            )
        return train_records, val_records

    common_load_kwargs = {
        "seed": args.seed,
        "min_transcript_chars": args.min_transcript_chars,
        "max_transcript_chars": args.max_transcript_chars,
        "max_symbol_ratio": args.max_symbol_ratio,
        "lowercase_transcripts": lowercase_transcripts,
        "hf_token": args.hf_token,
        "cache_dir": args.cache_dir,
        "min_chars_per_second": args.min_chars_per_second,
        "max_chars_per_second": args.max_chars_per_second,
        "min_words_per_second": args.min_words_per_second,
        "max_words_per_second": args.max_words_per_second,
        "min_duration_per_char": args.min_duration_per_char,
        "max_duration_per_char": args.max_duration_per_char,
        "min_duration_per_word": args.min_duration_per_word,
        "max_duration_per_word": args.max_duration_per_word,
    }
    if hasattr(args, "min_audio_duration_sec"):
        common_load_kwargs["min_audio_duration_sec"] = args.min_audio_duration_sec
    if hasattr(args, "max_audio_duration_sec"):
        common_load_kwargs["max_audio_duration_sec"] = args.max_audio_duration_sec

    train_records = _load_records_from_dataset_roots(
        train_dataset_sources,
        split="train",
        val_fraction=train_val_fraction,
        test_fraction=train_test_fraction,
        max_samples=args.max_train_samples,
        **common_load_kwargs,
    )
    val_records = _load_records_from_dataset_roots(
        validation_dataset_sources or train_dataset_sources,
        split=validation_split,
        val_fraction=validation_val_fraction,
        test_fraction=validation_test_fraction,
        max_samples=args.max_val_samples,
        **common_load_kwargs,
    )
    if args.prevalidate_audio:
        train_records = _prevalidate_loaded_records(
            train_records,
            split="train",
            num_workers=args.prevalidate_workers,
        )
        val_records = _prevalidate_loaded_records(
            val_records,
            split="validation",
            num_workers=args.prevalidate_workers,
        )
    return train_records, val_records


def _shard_records_for_rank(
    records: list[AudioRecord] | DiskBackedRecordStore,
    *,
    rank: int,
    world_size: int,
    allow_uneven: bool = False,
) -> list[AudioRecord] | DiskBackedRecordStore:
    if world_size <= 1:
        return records
    if hasattr(records, "shard"):
        return records.shard(rank, world_size, allow_uneven=allow_uneven)
    if allow_uneven:
        return records[rank::world_size]
    usable = (len(records) // world_size) * world_size
    return records[:usable][rank:usable:world_size]


def _record_store_duration_hours(
    records: list[AudioRecord] | DiskBackedRecordStore,
    *,
    hop_length: int,
    sample_rate: int = 16000,
) -> float:
    if sample_rate <= 0:
        return 0.0
    if (
        hasattr(records, "num_samples")
        and getattr(records, "num_samples") is not None
        and hasattr(records, "sample_rates")
        and getattr(records, "sample_rates") is not None
    ):
        total_seconds = 0.0
        for num_samples, record_sample_rate in zip(
            records.num_samples,
            records.sample_rates,
            strict=True,
        ):
            if int(num_samples) > 0 and int(record_sample_rate) > 0:
                total_seconds += int(num_samples) / int(record_sample_rate)
        return total_seconds / 3600.0
    if not hasattr(records, "estimated_frames") and all(
        int(record.num_samples) > 0 and int(record.sample_rate) > 0 for record in records
    ):
        total_seconds = sum(int(record.num_samples) / int(record.sample_rate) for record in records)
        return total_seconds / 3600.0
    total_frames = 0
    if hasattr(records, "estimated_frames"):
        total_frames = sum(int(value) for value in records.estimated_frames)
    else:
        total_frames = sum(max(0, int(record.estimated_frames)) for record in records)
    total_seconds = (float(total_frames) * float(hop_length)) / float(sample_rate)
    return total_seconds / 3600.0


def _frames_to_minutes(
    total_frames: int,
    *,
    hop_length: int,
    sample_rate: int = 16000,
) -> float:
    if sample_rate <= 0:
        return 0.0
    total_seconds = (float(total_frames) * float(hop_length)) / float(sample_rate)
    return total_seconds / 60.0


def _build_split_audit(
    split_records: dict[str, list[AudioRecord] | DiskBackedRecordStore],
    *,
    hop_length: int,
) -> dict[str, object]:
    speaker_sets = {
        split_name: {
            record.speaker_id for record in records if record.has_speaker_id and record.speaker_id
        }
        for split_name, records in split_records.items()
    }
    counts = {
        split_name: {
            "samples": len(records),
            "speakers": len(speaker_sets[split_name]),
            "records_with_speaker_id": sum(int(record.has_speaker_id) for record in records),
            "hours": _record_store_duration_hours(records, hop_length=hop_length),
        }
        for split_name, records in split_records.items()
    }
    overlaps = {}
    split_names = list(split_records)
    for index, split_name in enumerate(split_names):
        for other_name in split_names[index + 1 :]:
            key = f"{split_name}_vs_{other_name}"
            overlaps[key] = len(speaker_sets[split_name] & speaker_sets[other_name])
    speaker_counts = [item["speakers"] for item in counts.values() if item["speakers"] > 0]
    balance_ratio = (
        max(speaker_counts) / max(1, min(speaker_counts)) if len(speaker_counts) >= 2 else 1.0
    )
    return {
        "counts": counts,
        "hours": {split_name: item["hours"] for split_name, item in counts.items()},
        "total_hours": sum(float(item["hours"]) for item in counts.values()),
        "speaker_overlaps": overlaps,
        "speaker_balance_ratio": balance_ratio,
        "speaker_id_available": all(
            item["records_with_speaker_id"] == item["samples"] for item in counts.values()
        ),
    }
