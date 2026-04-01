from __future__ import annotations

import hashlib
import io
import multiprocessing as mp
import re
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl
import torch
import torchaudio
from huggingface_hub import list_repo_files, snapshot_download
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset

from .asr import Tokenizer

TRANSCRIPT_COLUMNS = ("sentence", "transcript", "transcription", "text", "normalized_text")
AUDIO_COLUMNS = ("path", "audio")


@dataclass(frozen=True)
class CVRecord:
    audio_path: str | None
    audio_bytes: bytes | None
    transcript: str
    utterance_id: str
    estimated_frames: int
    speaker_id: str | None = None
    has_speaker_id: bool = False
    num_samples: int = 0
    sample_rate: int = 0


class AudioFeaturizer(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        preemphasis: float = 0.97,
        normalize_signal: bool = True,
        normalize_feature: bool = True,
        normalize_per_frame: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.preemphasis = preemphasis
        self.normalize_signal = normalize_signal
        self.normalize_feature = normalize_feature
        self.normalize_per_frame = normalize_per_frame
        self.hop_length = hop_length
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def forward(self, waveform: Tensor, sample_rate: int) -> Tensor:
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                sample_rate,
                self.sample_rate,
            )[0]
        if self.normalize_signal:
            waveform = waveform - waveform.mean()
            waveform = waveform / waveform.abs().amax().clamp_min(1e-6)
        if self.preemphasis > 0:
            waveform = torch.cat(
                [waveform[:1], waveform[1:] - self.preemphasis * waveform[:-1]],
                dim=0,
            )
        features = self.mel(waveform)
        features = torch.log(features.clamp_min(1e-5)).transpose(0, 1)
        if self.normalize_feature:
            if self.normalize_per_frame:
                mean = features.mean(dim=-1, keepdim=True)
                std = features.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
            else:
                mean = features.mean(dim=0, keepdim=True)
                std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
            features = (features - mean) / std
        return features

    def config_dict(self) -> dict[str, object]:
        return {
            "sample_rate": self.sample_rate,
            "preemphasis": self.preemphasis,
            "normalize_signal": self.normalize_signal,
            "normalize_feature": self.normalize_feature,
            "normalize_per_frame": self.normalize_per_frame,
            "hop_length": self.hop_length,
        }


class SpecAugment(torch.nn.Module):
    def __init__(
        self,
        num_freq_masks: int = 2,
        freq_mask_param: int = 27,
        num_time_masks: int = 5,
        time_mask_max_ratio: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_freq_masks = num_freq_masks
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.time_mask_max_ratio = time_mask_max_ratio

    def forward(self, features: Tensor) -> Tensor:
        augmented = features.clone()
        time_steps, feature_bins = augmented.shape

        for _ in range(self.num_freq_masks):
            width = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
            if width == 0 or width >= feature_bins:
                continue
            start = int(torch.randint(0, feature_bins - width + 1, (1,)).item())
            augmented[:, start : start + width] = 0

        max_time_width = max(1, int(time_steps * self.time_mask_max_ratio))
        for _ in range(self.num_time_masks):
            width = int(torch.randint(0, max_time_width + 1, (1,)).item())
            if width == 0 or width >= time_steps:
                continue
            start = int(torch.randint(0, time_steps - width + 1, (1,)).item())
            augmented[start : start + width, :] = 0

        return augmented


class WaveformAugment(torch.nn.Module):
    def __init__(
        self,
        speed_perturb_prob: float = 0.0,
        speed_factors: tuple[float, ...] = (0.9, 1.0, 1.1),
        noise_prob: float = 0.0,
        noise_snr_db_range: tuple[float, float] = (10.0, 30.0),
        reverb_prob: float = 0.0,
        reverb_decay_range: tuple[float, float] = (0.15, 0.5),
        reverb_delay_ms_range: tuple[float, float] = (8.0, 35.0),
    ) -> None:
        super().__init__()
        self.speed_perturb_prob = speed_perturb_prob
        self.speed_factors = speed_factors
        self.noise_prob = noise_prob
        self.noise_snr_db_range = noise_snr_db_range
        self.reverb_prob = reverb_prob
        self.reverb_decay_range = reverb_decay_range
        self.reverb_delay_ms_range = reverb_delay_ms_range

    def is_enabled(self) -> bool:
        return any(
            probability > 0.0
            for probability in (
                self.speed_perturb_prob,
                self.noise_prob,
                self.reverb_prob,
            )
        )

    def forward(self, waveform: Tensor, sample_rate: int) -> tuple[Tensor, int]:
        augmented = waveform
        current_sample_rate = sample_rate
        if self.speed_perturb_prob > 0 and torch.rand(1).item() < self.speed_perturb_prob:
            factor = self.speed_factors[int(torch.randint(0, len(self.speed_factors), (1,)).item())]
            if factor != 1.0:
                target_rate = max(1, int(round(current_sample_rate * factor)))
                augmented = torchaudio.functional.resample(
                    augmented,
                    current_sample_rate,
                    target_rate,
                )
                augmented = torchaudio.functional.resample(
                    augmented,
                    target_rate,
                    current_sample_rate,
                )
        if self.noise_prob > 0 and torch.rand(1).item() < self.noise_prob:
            augmented = self._add_noise(augmented)
        if self.reverb_prob > 0 and torch.rand(1).item() < self.reverb_prob:
            augmented = self._add_reverb(augmented, current_sample_rate)
        return augmented, current_sample_rate

    def _add_noise(self, waveform: Tensor) -> Tensor:
        low, high = self.noise_snr_db_range
        snr_db = float(torch.empty(1).uniform_(low, high).item())
        signal_power = waveform.pow(2).mean().clamp_min(1e-8)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        return (waveform + noise).clamp(-1.0, 1.0)

    def _add_reverb(self, waveform: Tensor, sample_rate: int) -> Tensor:
        low_decay, high_decay = self.reverb_decay_range
        low_delay_ms, high_delay_ms = self.reverb_delay_ms_range
        decay = float(torch.empty(1).uniform_(low_decay, high_decay).item())
        delay_ms = float(torch.empty(1).uniform_(low_delay_ms, high_delay_ms).item())
        delay_samples = max(1, int(sample_rate * delay_ms / 1000.0))
        impulse_length = min(waveform.size(-1), max(delay_samples * 4, delay_samples + 1))
        impulse = waveform.new_zeros(1, 1, impulse_length)
        impulse[0, 0, 0] = 1.0
        for tap in range(1, 4):
            index = min(impulse_length - 1, tap * delay_samples)
            impulse[0, 0, index] += decay**tap
        impulse = impulse / impulse.abs().sum().clamp_min(1e-6)
        reverberated = F.conv1d(
            waveform.unsqueeze(0),
            impulse.expand(waveform.size(0), -1, -1),
            padding=impulse_length - 1,
            groups=waveform.size(0),
        )[0]
        return reverberated[..., : waveform.size(-1)].clamp(-1.0, 1.0)


def normalize_transcript(text: str) -> str:
    normalized = text.strip().lower()
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


def iter_manifest_rows(dataset_root: Path, batch_size: int = 8192) -> Iterable[dict[str, Any]]:
    manifest_paths = list(_iter_manifest_paths(dataset_root))
    if not manifest_paths:
        raise FileNotFoundError(f"No TSV or Parquet manifest files found under {dataset_root}.")

    for path in manifest_paths:
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
            continue

        # Process Parquet files one at a time to avoid concatenating all manifests into memory.
        yield from pl.read_parquet(path).iter_rows(named=True)


def _extract_transcript(row: dict[str, Any]) -> str:
    for column in TRANSCRIPT_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return normalize_transcript(value)
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


def download_cv22_dataset(
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


def load_cv22_records(
    dataset_root: Path,
    split: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    max_samples: int | None = None,
    min_transcript_chars: int = 1,
    max_transcript_chars: int = 400,
    max_symbol_ratio: float = 0.5,
) -> list[CVRecord]:
    records = list(
        iter_cv22_records(
            dataset_root=dataset_root,
            split=split,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            max_samples=max_samples,
            min_transcript_chars=min_transcript_chars,
            max_transcript_chars=max_transcript_chars,
            max_symbol_ratio=max_symbol_ratio,
        )
    )
    if not records:
        raise RuntimeError(f"Split '{split}' is empty after applying the current split fractions.")
    return records


def iter_cv22_records(
    dataset_root: Path,
    split: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    max_samples: int | None = None,
    min_transcript_chars: int = 1,
    max_transcript_chars: int = 400,
    max_symbol_ratio: float = 0.5,
) -> Iterable[CVRecord]:
    selected = 0
    found_usable_record = False
    scanned = 0
    train_cutoff = max(0.0, 1.0 - val_fraction - test_fraction)

    for row in iter_manifest_rows(dataset_root):
        scanned += 1
        try:
            transcript = _extract_transcript(row)
            audio_path, audio_bytes = _resolve_audio(row, dataset_root=dataset_root)
        except KeyError:
            continue
        if not transcript_is_usable(
            transcript,
            min_chars=min_transcript_chars,
            max_chars=max_transcript_chars,
            max_symbol_ratio=max_symbol_ratio,
        ):
            continue
        found_usable_record = True
        utterance_id = str(row.get("id") or audio_path or scanned)
        raw_speaker_id = row.get("client_id") or row.get("speaker_id") or row.get("speaker")
        speaker_id = str(raw_speaker_id) if raw_speaker_id not in {None, ""} else None
        duration_seconds = (
            row.get("duration") or row.get("duration_seconds") or row.get("audio_duration")
        )
        if isinstance(duration_seconds, (float, int)):
            estimated_frames = max(1, int((float(duration_seconds) * 16000) / 160))
        else:
            estimated_frames = 0
        record = CVRecord(
            audio_path=audio_path,
            audio_bytes=audio_bytes,
            transcript=transcript,
            utterance_id=utterance_id,
            speaker_id=speaker_id,
            has_speaker_id=speaker_id is not None,
            estimated_frames=estimated_frames,
        )
        split_key = record.speaker_id or record.utterance_id
        score = _hash_to_unit_interval(split_key, seed=seed)
        if split == "train" and score >= train_cutoff:
            continue
        if split == "validation" and not (
            train_cutoff <= score < train_cutoff + val_fraction
        ):
            continue
        if split == "test" and score < train_cutoff + val_fraction:
            continue
        yield record
        selected += 1
        if max_samples is not None and selected >= max_samples:
            return

    if not found_usable_record:
        raise RuntimeError("No usable records were found in the dataset manifests.")


def load_cv22_corpus_texts(
    dataset_root: Path,
    deduplicate: bool = False,
    max_samples: int | None = None,
) -> list[str]:
    texts = list(
        iter_cv22_corpus_texts(
            dataset_root=dataset_root,
            deduplicate=deduplicate,
            max_samples=max_samples,
        )
    )
    if not texts:
        raise RuntimeError("No usable transcripts were found in the dataset manifests.")
    return texts


def iter_cv22_corpus_texts_from_repo(
    repo_id: str,
    token: str | None,
    deduplicate: bool = False,
    max_samples: int | None = None,
) -> Iterable[str]:
    seen: set[str] = set()
    yielded = 0
    storage_options = _hf_storage_options(token)
    for repo_path in sorted(list_repo_files(repo_id, repo_type="dataset", token=token)):
        if repo_path.endswith(".parquet"):
            lazy_frame = pl.scan_parquet(
                f"hf://datasets/{repo_id}/{repo_path}",
                storage_options=storage_options,
            )
            transcript_column = _select_transcript_column(lazy_frame.collect_schema().names())
            if transcript_column is None:
                continue
            frame = lazy_frame.select(pl.col(transcript_column)).collect()
            rows = ({transcript_column: value} for value in frame[transcript_column].to_list())
        elif repo_path.endswith(".tsv"):
            lazy_frame = pl.scan_csv(
                f"hf://datasets/{repo_id}/{repo_path}",
                separator="\t",
                infer_schema_length=1000,
                storage_options=storage_options,
            )
            transcript_column = _select_transcript_column(lazy_frame.collect_schema().names())
            if transcript_column is None:
                continue
            frame = lazy_frame.select(pl.col(transcript_column)).collect()
            rows = ({transcript_column: value} for value in frame[transcript_column].to_list())
        else:
            continue

        for row in rows:
            try:
                transcript = _extract_transcript(row)
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


def iter_cv22_corpus_texts(
    dataset_root: Path,
    deduplicate: bool = False,
    max_samples: int | None = None,
) -> Iterable[str]:
    seen: set[str] = set()
    yielded = 0
    for row in iter_manifest_rows(dataset_root):
        try:
            transcript = _extract_transcript(row)
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
) -> Path | None:
    if feature_cache_dir is None:
        return None
    feature_cache_path = Path(feature_cache_dir)
    feature_cache_path.mkdir(parents=True, exist_ok=True)
    frontend_hash = hashlib.sha256(repr(featurizer.config_dict()).encode("utf-8")).hexdigest()[:12]
    return feature_cache_path / f"{utterance_id}_{frontend_hash}.pt"


class CV22ASRDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[CVRecord],
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

    def _feature_cache_path(self, record: CVRecord) -> Path | None:
        return feature_cache_path(self.feature_cache_dir, record.utterance_id, self.featurizer)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        cache_path = self._feature_cache_path(record)
        waveform_augment_enabled = (
            self.waveform_augment is not None and self.waveform_augment.is_enabled()
        )
        use_cache = cache_path is not None and not waveform_augment_enabled
        if use_cache and cache_path.exists():
            features = torch.load(cache_path, map_location="cpu")
        else:
            waveform, sample_rate = load_audio(record.audio_path, record.audio_bytes)
            if waveform_augment_enabled:
                waveform, sample_rate = self.waveform_augment(waveform, sample_rate)
            features = self.featurizer(waveform, sample_rate)
            if use_cache:
                torch.save(features, cache_path)
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
    def __init__(self, records: list[CVRecord], batch_size: int, shuffle: bool) -> None:
        self.records = records
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = sorted(
            range(len(self.records)), key=lambda index: self.records[index].estimated_frames
        )
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            order = torch.randperm(len(batches)).tolist()
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self) -> int:
        return (len(self.records) + self.batch_size - 1) // self.batch_size


class MaxFramesBatchSampler(BatchSampler):
    def __init__(
        self,
        records: list[CVRecord],
        max_batch_frames: int,
        shuffle: bool,
    ) -> None:
        self.records = records
        self.max_batch_frames = max_batch_frames
        self.shuffle = shuffle
        self._batches = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        sorted_indices = sorted(
            range(len(self.records)), key=lambda index: self.records[index].estimated_frames
        )
        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_max = 0
        for index in sorted_indices:
            frames = max(1, self.records[index].estimated_frames)
            proposed_size = len(current_batch) + 1
            proposed_max = max(current_max, frames)
            if current_batch and proposed_size * proposed_max > self.max_batch_frames:
                batches.append(current_batch)
                current_batch = []
                current_max = 0
            current_batch.append(index)
            current_max = max(current_max, frames)
        if current_batch:
            batches.append(current_batch)
        return batches

    def __iter__(self):
        batches = list(self._batches)
        if self.shuffle:
            order = torch.randperm(len(batches)).tolist()
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)


class AdaptiveBatchSampler(BatchSampler):
    def __init__(
        self,
        records: list[CVRecord],
        target_batch_units: int,
        unit: str,
        shuffle: bool,
    ) -> None:
        if unit not in {"frames", "tokens"}:
            raise ValueError("unit must be one of {'frames', 'tokens'}")
        self.records = records
        self.target_batch_units = target_batch_units
        self.unit = unit
        self.shuffle = shuffle
        self._batches = self._build_batches()

    def _record_units(self, record: CVRecord) -> int:
        if self.unit == "frames":
            return max(1, record.estimated_frames)
        return max(1, len(record.transcript))

    def _build_batches(self) -> list[list[int]]:
        sorted_indices = sorted(
            range(len(self.records)),
            key=lambda index: (
                self.records[index].estimated_frames,
                len(self.records[index].transcript),
            ),
        )
        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_units = 0
        for index in sorted_indices:
            units = self._record_units(self.records[index])
            if current_batch and current_units + units > self.target_batch_units:
                batches.append(current_batch)
                current_batch = []
                current_units = 0
            current_batch.append(index)
            current_units += units
        if current_batch:
            batches.append(current_batch)
        return batches

    def __iter__(self):
        batches = list(self._batches)
        if self.shuffle:
            order = torch.randperm(len(batches)).tolist()
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)


def _record_is_valid(record: CVRecord) -> bool:
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
    records: list[CVRecord],
    num_workers: int = 4,
) -> list[CVRecord]:
    if num_workers <= 1:
        return [record for record in records if _record_is_valid(record)]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        validity = list(executor.map(_record_is_valid, records))
    return [record for record, is_valid in zip(records, validity, strict=True) if is_valid]


def estimate_record_frames(record: CVRecord, hop_length: int) -> int:
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
    records: list[CVRecord],
    hop_length: int,
    num_workers: int = 4,
) -> list[CVRecord]:
    def populate(record: CVRecord) -> CVRecord:
        estimate_record_frames(record, hop_length=hop_length)
        return record

    if num_workers <= 1:
        return [populate(record) for record in records]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(populate, records))


def collate_asr_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
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
    dataset: CV22ASRDataset,
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
) -> DataLoader[dict[str, Any]]:
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
        if sys.platform.startswith("linux"):
            dataloader_kwargs["multiprocessing_context"] = mp.get_context("fork")
    if adaptive_batch_unit is not None and adaptive_batch_budget is not None:
        batch_sampler = AdaptiveBatchSampler(
            dataset.records,
            target_batch_units=adaptive_batch_budget,
            unit=adaptive_batch_unit,
            shuffle=shuffle,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_kwargs)
    if max_batch_frames is not None:
        batch_sampler = MaxFramesBatchSampler(
            dataset.records,
            max_batch_frames=max_batch_frames,
            shuffle=shuffle,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_kwargs)
    if bucket_by_length:
        batch_sampler = LengthBucketBatchSampler(
            dataset.records, batch_size=batch_size, shuffle=shuffle
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **dataloader_kwargs,
    )
