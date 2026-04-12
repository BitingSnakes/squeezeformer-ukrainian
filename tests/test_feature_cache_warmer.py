from __future__ import annotations

from pathlib import Path

import torch

from squeezeformer_pytorch.data import AudioFeaturizer, AudioRecord, ShardedParquetFeatureCache
from squeezeformer_pytorch.runtime_types import FeatureCacheFormat
from squeezeformer_pytorch.training.feature_cache_warmer import (
    FeatureCacheWarmDataset,
    _resolve_cache_warm_splits,
    parse_args,
)


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


def test_parse_feature_cache_warmer_accepts_comma_separated_splits() -> None:
    args = parse_args(["--cache-warm-split", "train,validation", "--device", "cpu"])

    assert args.cache_warm_split == "train,validation"
    assert _resolve_cache_warm_splits(args.cache_warm_split) == {"train", "validation"}


def test_resolve_cache_warm_splits_expands_both() -> None:
    assert _resolve_cache_warm_splits("both") == {"train", "validation"}
    assert _resolve_cache_warm_splits("train,both") == {"train", "validation"}


def test_resolve_cache_warm_splits_rejects_unknown_split() -> None:
    try:
        _resolve_cache_warm_splits("train,test")
    except ValueError as error:
        assert "test" in str(error)
    else:
        raise AssertionError("Expected invalid cache warm split to be rejected.")


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


def test_feature_cache_warmer_can_return_features_for_main_process_write(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_load_audio(
        audio_path: str | None, audio_bytes: bytes | None
    ) -> tuple[torch.Tensor, int]:
        return torch.ones(1, 320), 16_000

    monkeypatch.setattr(
        "squeezeformer_pytorch.training.feature_cache_warmer.load_audio", fake_load_audio
    )
    featurizer = AudioFeaturizer()
    dataset = FeatureCacheWarmDataset(
        [AudioRecord("dummy.wav", None, "це тест", "utt0", estimated_frames=2)],
        featurizer=featurizer,
        feature_cache_dir=tmp_path,
        feature_cache_format="parquet",
        write_cache=False,
    )

    item = dataset[0]

    assert item["status"] == "written"
    assert item["features"].shape == (3, featurizer.n_mels)
    assert not list((tmp_path / "feature_shards").glob("features_*/part_*.parquet"))


def test_sharded_parquet_feature_cache_close_flushes_buffered_rows(tmp_path: Path) -> None:
    featurizer = AudioFeaturizer()
    cache = ShardedParquetFeatureCache(tmp_path, commit_every=64)

    cache.store("utt0", featurizer, torch.ones(3, featurizer.n_mels))

    assert not list((tmp_path / "feature_shards").glob("features_*/part_*.parquet"))
    cache.close()
    assert list((tmp_path / "feature_shards").glob("features_*/part_*.parquet"))
