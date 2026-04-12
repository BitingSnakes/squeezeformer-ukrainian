from __future__ import annotations

import time
from pathlib import Path

import torch

from feature_cache_warmer.cli import (
    FeatureCacheWarmDataset,
    _ffmpeg_decode_audio_source,
    _ffprobe_audio_source,
    _load_skip_list,
    _resolve_cache_warm_splits,
    parse_args,
)
from squeezeformer_pytorch.data import AudioFeaturizer, AudioRecord, ShardedParquetFeatureCache
from squeezeformer_pytorch.runtime_types import FeatureCacheFormat


def test_parse_feature_cache_warmer_accepts_training_and_warmer_args(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--cache-warm-split",
            "both",
            "--cache-warm-workers",
            "4",
            "--cache-warm-timeout",
            "60",
            "--cache-warm-record-timeout",
            "7",
            "--cache-warm-ffprobe-timeout",
            "0.2",
            "--cache-warm-ffmpeg-timeout",
            "0.3",
            "--cache-warm-skip-list",
            str(tmp_path / "skip.txt"),
            "--cache-warm-failed-list",
            str(tmp_path / "failed.tsv"),
            "--cache-warm-audio-source",
            "bytes",
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
    assert args.cache_warm_timeout == 60
    assert args.cache_warm_record_timeout == 7
    assert args.cache_warm_ffprobe_timeout == 0.2
    assert args.cache_warm_ffmpeg_timeout == 0.3
    assert args.cache_warm_skip_list == str(tmp_path / "skip.txt")
    assert args.cache_warm_failed_list == str(tmp_path / "failed.tsv")
    assert args.cache_warm_audio_source == "bytes"
    assert args.feature_cache_dir == str(tmp_path / "cache")
    assert args.feature_cache_format == FeatureCacheFormat.PARQUET


def test_parse_feature_cache_warmer_accepts_comma_separated_splits() -> None:
    args = parse_args(["--cache-warm-split", "train,validation", "--device", "cpu"])

    assert args.cache_warm_split == "train,validation"
    assert _resolve_cache_warm_splits(args.cache_warm_split) == {"train", "validation"}


def test_parse_feature_cache_warmer_rejects_subsecond_timeouts() -> None:
    for flag in ("--cache-warm-timeout", "--cache-warm-record-timeout"):
        try:
            parse_args([flag, "0.01", "--device", "cpu"])
        except ValueError as error:
            assert "at least 1 second" in str(error)
        else:
            raise AssertionError(f"Expected {flag} to reject subsecond timeout.")


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

    monkeypatch.setattr("feature_cache_warmer.cli.load_audio", fake_load_audio)
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

    monkeypatch.setattr("feature_cache_warmer.cli.load_audio", fake_load_audio)
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

    monkeypatch.setattr("feature_cache_warmer.cli.load_audio", fake_load_audio)
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


def test_feature_cache_warmer_record_timeout_marks_record_failed(
    tmp_path: Path, monkeypatch
) -> None:
    def slow_load_audio(
        audio_path: str | None, audio_bytes: bytes | None
    ) -> tuple[torch.Tensor, int]:
        time.sleep(1.0)
        return torch.ones(1, 320), 16_000

    monkeypatch.setattr("feature_cache_warmer.cli.load_audio", slow_load_audio)
    dataset = FeatureCacheWarmDataset(
        [AudioRecord("dummy.wav", None, "це тест", "utt0", estimated_frames=2)],
        featurizer=AudioFeaturizer(),
        feature_cache_dir=tmp_path,
        feature_cache_format="file",
        record_timeout_seconds=0.01,
    )

    item = dataset[0]

    assert item["status"] == "failed"
    assert "timed out" in item["error"]
    assert not list(tmp_path.glob("*.pt"))


def test_feature_cache_warmer_skip_list_skips_before_loading_audio(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_load_audio(
        audio_path: str | None, audio_bytes: bytes | None
    ) -> tuple[torch.Tensor, int]:
        raise AssertionError("skip list should avoid audio load")

    monkeypatch.setattr("feature_cache_warmer.cli.load_audio", fake_load_audio)
    dataset = FeatureCacheWarmDataset(
        [
            AudioRecord(
                "/data/common_voice_uk_24203951.opus",
                None,
                "це тест",
                "/data/common_voice_uk_24203951.opus",
                estimated_frames=2,
            )
        ],
        featurizer=AudioFeaturizer(),
        feature_cache_dir=tmp_path,
        feature_cache_format="file",
        skipped_audio_sources={"common_voice_uk_24203951.opus"},
    )

    item = dataset[0]

    assert item["status"] == "skipped"
    assert item["utterance_id"] == "/data/common_voice_uk_24203951.opus"
    assert not list(tmp_path.glob("*.pt"))


def test_feature_cache_warmer_ffprobe_rejects_bad_audio_before_decode(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_load_audio(
        audio_path: str | None, audio_bytes: bytes | None
    ) -> tuple[torch.Tensor, int]:
        raise AssertionError("ffprobe failure should avoid audio load")

    def fake_run(*args, **kwargs):
        return type(
            "Completed",
            (),
            {"returncode": 1, "stdout": b"", "stderr": b"invalid data"},
        )()

    monkeypatch.setattr("feature_cache_warmer.cli.load_audio", fake_load_audio)
    monkeypatch.setattr("feature_cache_warmer.cli.subprocess.run", fake_run)
    dataset = FeatureCacheWarmDataset(
        [AudioRecord("dummy.wav", b"bad", "це тест", "utt0", estimated_frames=2)],
        featurizer=AudioFeaturizer(),
        feature_cache_dir=tmp_path,
        feature_cache_format="file",
        ffprobe_timeout_seconds=0.2,
    )

    item = dataset[0]

    assert item["status"] == "failed"
    assert "ffprobe failed" in item["error"]
    assert not list(tmp_path.glob("*.pt"))


def test_ffprobe_audio_source_accepts_audio_stream(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return type("Completed", (), {"returncode": 0, "stdout": b"audio\n", "stderr": b""})()

    monkeypatch.setattr("feature_cache_warmer.cli.subprocess.run", fake_run)

    error = _ffprobe_audio_source(
        AudioRecord("dummy.wav", b"ok", "це тест", "utt0", estimated_frames=2),
        0.2,
    )

    assert error is None


def test_feature_cache_warmer_ffmpeg_rejects_bad_audio_before_decode(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_load_audio(
        audio_path: str | None, audio_bytes: bytes | None
    ) -> tuple[torch.Tensor, int]:
        raise AssertionError("ffmpeg failure should avoid audio load")

    def fake_run(*args, **kwargs):
        return type(
            "Completed",
            (),
            {"returncode": 1, "stdout": b"", "stderr": b"invalid data"},
        )()

    monkeypatch.setattr("feature_cache_warmer.cli.load_audio", fake_load_audio)
    monkeypatch.setattr("feature_cache_warmer.cli.subprocess.run", fake_run)
    dataset = FeatureCacheWarmDataset(
        [AudioRecord("dummy.wav", b"bad", "це тест", "utt0", estimated_frames=2)],
        featurizer=AudioFeaturizer(),
        feature_cache_dir=tmp_path,
        feature_cache_format="file",
        ffmpeg_timeout_seconds=0.2,
    )

    item = dataset[0]

    assert item["status"] == "failed"
    assert "ffmpeg decode validation failed" in item["error"]
    assert not list(tmp_path.glob("*.pt"))


def test_ffmpeg_decode_audio_source_accepts_decodable_audio(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return type("Completed", (), {"returncode": 0, "stdout": b"", "stderr": b""})()

    monkeypatch.setattr("feature_cache_warmer.cli.subprocess.run", fake_run)

    error = _ffmpeg_decode_audio_source(
        AudioRecord("dummy.wav", b"ok", "це тест", "utt0", estimated_frames=2),
        0.2,
    )

    assert error is None


def test_load_skip_list_accepts_paths_and_basenames(tmp_path: Path) -> None:
    skip_path = tmp_path / "skip.txt"
    skip_path.write_text(
        "# comment\n/data/common_voice_uk_24203951.opus\nutt0\n",
        encoding="utf-8",
    )

    values = _load_skip_list(skip_path)

    assert "/data/common_voice_uk_24203951.opus" in values
    assert "common_voice_uk_24203951.opus" in values
    assert "utt0" in values


def test_sharded_parquet_feature_cache_close_flushes_buffered_rows(tmp_path: Path) -> None:
    featurizer = AudioFeaturizer()
    cache = ShardedParquetFeatureCache(tmp_path, commit_every=64)

    cache.store("utt0", featurizer, torch.ones(3, featurizer.n_mels))

    assert not list((tmp_path / "feature_shards").glob("features_*/part_*.parquet"))
    cache.close()
    assert list((tmp_path / "feature_shards").glob("features_*/part_*.parquet"))
