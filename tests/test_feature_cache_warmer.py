from __future__ import annotations

from pathlib import Path

import torch

from squeezeformer_pytorch.data import AudioFeaturizer, AudioRecord
from squeezeformer_pytorch.runtime_types import FeatureCacheFormat
from squeezeformer_pytorch.training.feature_cache_warmer import FeatureCacheWarmDataset, parse_args


def test_parse_feature_cache_warmer_accepts_training_and_warmer_args(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--cache-warm-split",
            "both",
            "--cache-warm-workers",
            "4",
            "--device",
            "cpu",
            "--feature-cache-dir",
            str(tmp_path / "cache"),
            "--feature-cache-format",
            "parquet",
        ]
    )

    assert args.cache_warm_split == "both"
    assert args.cache_warm_workers == 4
    assert args.feature_cache_dir == str(tmp_path / "cache")
    assert args.feature_cache_format == FeatureCacheFormat.PARQUET


def test_feature_cache_warmer_writes_file_cache(tmp_path: Path, monkeypatch) -> None:
    load_calls = 0

    def fake_load_audio(
        audio_path: str | None, audio_bytes: bytes | None
    ) -> tuple[torch.Tensor, int]:
        nonlocal load_calls
        load_calls += 1
        return torch.ones(1, 320), 16_000

    monkeypatch.setattr(
        "squeezeformer_pytorch.training.feature_cache_warmer.load_audio", fake_load_audio
    )
    dataset = FeatureCacheWarmDataset(
        [AudioRecord("dummy.wav", None, "це тест", "utt0", estimated_frames=2)],
        featurizer=AudioFeaturizer(),
        feature_cache_dir=tmp_path,
        feature_cache_format="file",
    )

    first = dataset[0]
    second = dataset[0]

    assert first["status"] == "written"
    assert second["status"] == "hit"
    assert load_calls == 1
    assert list(tmp_path.glob("*.pt"))
    assert not (tmp_path / "feature_shards").exists()


def test_feature_cache_warmer_writes_parquet_cache(tmp_path: Path, monkeypatch) -> None:
    load_calls = 0

    def fake_load_audio(
        audio_path: str | None, audio_bytes: bytes | None
    ) -> tuple[torch.Tensor, int]:
        nonlocal load_calls
        load_calls += 1
        return torch.ones(1, 320), 16_000

    monkeypatch.setattr(
        "squeezeformer_pytorch.training.feature_cache_warmer.load_audio", fake_load_audio
    )
    dataset = FeatureCacheWarmDataset(
        [AudioRecord("dummy.wav", None, "це тест", "utt0", estimated_frames=2)],
        featurizer=AudioFeaturizer(),
        feature_cache_dir=tmp_path,
        feature_cache_format="parquet",
    )

    first = dataset[0]
    second = dataset[0]
    dataset.close()

    assert first["status"] == "written"
    assert second["status"] == "hit"
    assert load_calls == 1
    assert list((tmp_path / "feature_shards").glob("features_*/part_*.parquet"))
    assert not list(tmp_path.glob("*.pt"))
