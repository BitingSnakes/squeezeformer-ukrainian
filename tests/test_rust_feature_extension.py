from __future__ import annotations

import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest


def test_build_featurizer_from_config_uses_rust_wrappers() -> None:
    from squeezeformer_pytorch.frontend import (
        RustAudioFeaturizer,
        RustW2VBertFeatureExtractor,
        build_featurizer_from_config,
        zipformer_paper_featurizer_config,
    )

    squeezeformer = build_featurizer_from_config({})
    zipformer = build_featurizer_from_config(
        zipformer_paper_featurizer_config(),
        use_zipformer=True,
    )
    w2v_bert = build_featurizer_from_config(
        {"type": "w2v_bert", "model_source": "facebook/w2v-bert-2.0"},
        use_w2v_bert=True,
    )

    assert isinstance(squeezeformer, RustAudioFeaturizer)
    assert squeezeformer.frontend_type == "squeezeformer"
    assert isinstance(zipformer, RustAudioFeaturizer)
    assert zipformer.frontend_type == "zipformer"
    assert isinstance(w2v_bert, RustW2VBertFeatureExtractor)


def test_rust_feature_extension_extracts_numpy_features() -> None:
    pytest.importorskip("asr_features")
    import numpy as np

    from asr_features import (
        extract_squeezeformer,
        extract_w2v_bert,
        extract_zipformer,
    )

    waveform = np.sin(np.arange(16_000, dtype=np.float32) * 0.01).astype(np.float32)

    squeezeformer = extract_squeezeformer(waveform, 16_000)
    zipformer = extract_zipformer(waveform, 16_000)
    w2v_bert = extract_w2v_bert(waveform, 16_000)

    assert squeezeformer.dtype == np.float32
    assert squeezeformer.shape[1] == 80
    assert zipformer.dtype == np.float32
    assert zipformer.shape[1] == 80
    assert w2v_bert.dtype == np.float32
    assert w2v_bert.shape[1] == 160


def test_rust_featurizer_modules_extract_torch_features() -> None:
    pytest.importorskip("asr_features")
    import torch

    from squeezeformer_pytorch.frontend import build_featurizer_from_config

    waveform = torch.sin(torch.arange(16_000, dtype=torch.float32) * 0.01)
    squeezeformer = build_featurizer_from_config({})
    w2v_bert = build_featurizer_from_config(
        {"type": "w2v_bert", "model_source": "facebook/w2v-bert-2.0"},
        use_w2v_bert=True,
    )

    squeezeformer_features = squeezeformer(waveform, 16_000)
    w2v_bert_features = w2v_bert(waveform, 16_000)

    assert squeezeformer_features.dtype == torch.float32
    assert squeezeformer_features.shape[1] == 80
    assert w2v_bert_features.dtype == torch.float32
    assert w2v_bert_features.shape[1] == 160


def test_rust_record_cache_subcommand_matches_python_store(tmp_path) -> None:
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not available")

    train_dir = tmp_path / "train-source"
    val_dir = tmp_path / "validation-source"
    (train_dir / "audio").mkdir(parents=True)
    (val_dir / "audio").mkdir(parents=True)
    for path in (
        train_dir / "audio" / "a.wav",
        train_dir / "audio" / "b.wav",
        val_dir / "audio" / "c.wav",
    ):
        _write_test_wav(path)
    (train_dir / "train.tsv").write_text(
        "id\tpath\tsentence\tduration\tclient_id\n"
        "utt1\taudio/a.wav\tПривіт , світе!\t1.0\tspk1\n"
        "utt2\taudio/b.wav\tДругий запис\t1.5\tspk2\n",
        encoding="utf-8",
    )
    (val_dir / "validation.tsv").write_text(
        "id\tpath\tsentence\tduration\tclient_id\n"
        "val1\taudio/c.wav\tВалідація\t2.0\tspk3\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "record-cache"

    subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str((Path.cwd() / "rust_feature_cache_warmer" / "Cargo.toml").resolve()),
            "--",
            "record-cache",
            "--record-cache-dir",
            str(cache_dir),
            "--dataset-source",
            str(train_dir),
            "--validation-dataset-source",
            str(val_dir),
            "--require-readable-audio",
            "--progress-interval",
            "0",
        ],
        check=True,
    )

    from squeezeformer_pytorch.training.data_loading import _open_disk_backed_record_store

    train = _open_disk_backed_record_store(cache_dir / "train.jsonl")
    validation = _open_disk_backed_record_store(cache_dir / "validation.jsonl")

    assert len(train) == 2
    assert len(validation) == 1
    assert train[0].transcript == "Привіт, світе!"
    assert train[0].utterance_id == "utt1"
    assert train[0].estimated_frames == 101
    assert train[0].num_samples == 16_000
    assert validation[0].utterance_id == "val1"
    for suffix, item_size in {
        ".offsets.u64": 8,
        ".estimated_frames.u32": 4,
        ".num_samples.u64": 8,
        ".sample_rates.u32": 4,
        ".transcript_lengths.u32": 4,
        ".token_lengths.u32": 4,
    }.items():
        assert (cache_dir / f"train.jsonl{suffix}").stat().st_size == len(train) * item_size

    feature_cache_dir = tmp_path / "feature-cache" / "train"
    subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str((Path.cwd() / "rust_feature_cache_warmer" / "Cargo.toml").resolve()),
            "--",
            "--input-record-cache",
            str(cache_dir / "train.jsonl"),
            "--cache-dir",
            str(feature_cache_dir),
            "--frontend",
            "zipformer",
            "--threads",
            "2",
            "--rows-per-part",
            "1",
        ],
        check=True,
    )

    from squeezeformer_pytorch.data import ShardedParquetFeatureCache
    from squeezeformer_pytorch.frontend import (
        build_featurizer_from_config,
        zipformer_paper_featurizer_config,
    )

    feature_cache = ShardedParquetFeatureCache(feature_cache_dir)
    features = feature_cache.load(
        "utt1",
        build_featurizer_from_config(zipformer_paper_featurizer_config(), use_zipformer=True),
    )
    assert features is not None
    assert features.shape[1] == 80


def _write_test_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        frames = bytearray()
        for index in range(16_000):
            sample = int(12_000 * math.sin(index * 0.03))
            frames.extend(struct.pack("<h", sample))
        handle.writeframes(bytes(frames))
