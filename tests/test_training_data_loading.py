from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import squeezeformer_pytorch.training.data_loading as data_loading
from squeezeformer_pytorch.data import AudioRecord
from squeezeformer_pytorch.training.data_loading import (
    _build_disk_backed_record_store,
    _disk_backed_record_store_exists,
    _build_split_audit,
    _load_train_val_records,
    _record_index_path,
    _shard_records_for_rank,
)


def test_shard_records_for_rank_allow_uneven_preserves_tail_samples() -> None:
    records = list(range(10))

    rank_0 = _shard_records_for_rank(records, rank=0, world_size=3, allow_uneven=True)
    rank_1 = _shard_records_for_rank(records, rank=1, world_size=3, allow_uneven=True)
    rank_2 = _shard_records_for_rank(records, rank=2, world_size=3, allow_uneven=True)

    assert rank_0 == [0, 3, 6, 9]
    assert rank_1 == [1, 4, 7]
    assert rank_2 == [2, 5, 8]


def test_shard_records_for_rank_even_mode_drops_tail_samples() -> None:
    records = list(range(10))

    rank_0 = _shard_records_for_rank(records, rank=0, world_size=3)
    rank_1 = _shard_records_for_rank(records, rank=1, world_size=3)
    rank_2 = _shard_records_for_rank(records, rank=2, world_size=3)

    assert rank_0 == [0, 3, 6]
    assert rank_1 == [1, 4, 7]
    assert rank_2 == [2, 5, 8]


def test_build_disk_backed_record_store_can_require_readable_audio(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "present.wav").write_bytes(b"not decoded during record build")
    (dataset_root / "train.tsv").write_text(
        "path\tsentence\tid\tduration\n"
        "present.wav\tнаявний запис\tutt0\t0.3\n"
        "missing.wav\tвідсутній запис\tutt1\t0.3\n",
        encoding="utf-8",
    )

    store = _build_disk_backed_record_store(
        [dataset_root],
        split="train",
        seed=13,
        val_fraction=0.0,
        test_fraction=0.0,
        max_samples=None,
        min_transcript_chars=1,
        max_transcript_chars=400,
        max_symbol_ratio=0.5,
        lowercase_transcripts=True,
        records_path=tmp_path / "records" / "train.jsonl",
        require_readable_audio=True,
    )

    assert len(store) == 1
    assert store[0].utterance_id == "utt0"


def test_build_disk_backed_record_store_can_require_audio_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "present.wav").write_bytes(b"path audio should not be used")

    def fake_iter_records(*args, **kwargs):
        return iter(
            [
                AudioRecord(
                    "present.wav",
                    None,
                    "path only",
                    "path-only",
                    estimated_frames=2,
                ),
                AudioRecord(
                    "present.wav",
                    b"embedded bytes",
                    "bytes",
                    "bytes-record",
                    estimated_frames=2,
                ),
            ]
        )

    monkeypatch.setattr(data_loading, "iter_records", fake_iter_records)

    store = _build_disk_backed_record_store(
        [dataset_root],
        split="train",
        seed=13,
        val_fraction=0.0,
        test_fraction=0.0,
        max_samples=None,
        min_transcript_chars=1,
        max_transcript_chars=400,
        max_symbol_ratio=0.5,
        lowercase_transcripts=True,
        records_path=tmp_path / "records" / "train.jsonl",
        require_audio_bytes=True,
    )

    assert len(store) == 1
    record = store[0]
    assert record.utterance_id == "bytes-record"
    assert record.audio_bytes == b"embedded bytes"


def test_load_train_val_records_reuses_existing_record_cache(
    tmp_path: Path, monkeypatch
) -> None:
    train_root = tmp_path / "train-data"
    train_root.mkdir()
    (train_root / "train.tsv").write_text(
        "path\tsentence\tid\tduration\n"
        "train.wav\ttraining sample\ttrain-utt\t0.3\n",
        encoding="utf-8",
    )
    validation_root = tmp_path / "validation-data"
    validation_root.mkdir()
    (validation_root / "validation.tsv").write_text(
        "path\tsentence\tid\tduration\n"
        "validation.wav\tvalidation sample\tvalidation-utt\t0.3\n",
        encoding="utf-8",
    )
    record_cache_dir = tmp_path / "cached-metadata"
    _build_disk_backed_record_store(
        [train_root],
        split="train",
        seed=13,
        val_fraction=0.0,
        test_fraction=0.0,
        max_samples=None,
        min_transcript_chars=1,
        max_transcript_chars=400,
        max_symbol_ratio=0.5,
        lowercase_transcripts=True,
        records_path=record_cache_dir / "train.jsonl",
    )
    _build_disk_backed_record_store(
        [validation_root],
        split="train",
        seed=13,
        val_fraction=0.0,
        test_fraction=0.0,
        max_samples=None,
        min_transcript_chars=1,
        max_transcript_chars=400,
        max_symbol_ratio=0.5,
        lowercase_transcripts=True,
        records_path=record_cache_dir / "validation.jsonl",
    )

    def fail_build(*args, **kwargs):
        raise AssertionError("existing record cache should be opened, not rebuilt")

    monkeypatch.setattr(data_loading, "_build_disk_backed_record_store", fail_build)

    args = SimpleNamespace(
        record_cache=True,
        record_cache_dir=str(record_cache_dir),
        seed=13,
        val_fraction=0.0,
        test_fraction=0.0,
        max_train_samples=None,
        max_val_samples=None,
        min_transcript_chars=1,
        max_transcript_chars=400,
        max_symbol_ratio=0.5,
        min_chars_per_second=0.0,
        max_chars_per_second=float("inf"),
        min_words_per_second=0.0,
        max_words_per_second=float("inf"),
        min_duration_per_char=0.0,
        max_duration_per_char=float("inf"),
        min_duration_per_word=0.0,
        max_duration_per_word=float("inf"),
        hf_token=None,
        cache_dir=None,
        prevalidate_audio=False,
    )

    train_records, validation_records = _load_train_val_records(
        args,
        [train_root],
        [validation_root],
        lowercase_transcripts=True,
        output_dir=tmp_path / "artifacts",
    )

    assert train_records[0].utterance_id == "train-utt"
    assert validation_records[0].utterance_id == "validation-utt"


def test_record_cache_reuse_rejects_mismatched_index_lengths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "train.tsv").write_text(
        "path\tsentence\tid\tduration\n"
        "a.wav\tfirst sample\tutt0\t0.3\n"
        "b.wav\tsecond sample\tutt1\t0.3\n",
        encoding="utf-8",
    )
    records_path = tmp_path / "cached-metadata" / "train.jsonl"
    _build_disk_backed_record_store(
        [dataset_root],
        split="train",
        seed=13,
        val_fraction=0.0,
        test_fraction=0.0,
        max_samples=None,
        min_transcript_chars=1,
        max_transcript_chars=400,
        max_symbol_ratio=0.5,
        lowercase_transcripts=True,
        records_path=records_path,
    )
    estimated_frames_path = _record_index_path(records_path, ".estimated_frames.u32")
    estimated_frames_path.write_bytes(estimated_frames_path.read_bytes()[:4])

    assert not _disk_backed_record_store_exists(records_path)


def test_split_audit_does_not_load_disk_backed_audio_blobs(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(
        '{"audio_path":null,"audio_blob_path":"missing.bin","transcript":"sample",'
        '"utterance_id":"utt0","speaker_id":"speaker-a","has_speaker_id":true}\n',
        encoding="utf-8",
    )
    _record_index_path(records_path, ".offsets.u64").write_bytes((0).to_bytes(8, "little"))
    _record_index_path(records_path, ".estimated_frames.u32").write_bytes(
        (100).to_bytes(4, "little")
    )
    _record_index_path(records_path, ".num_samples.u64").write_bytes(
        (16_000).to_bytes(8, "little")
    )
    _record_index_path(records_path, ".sample_rates.u32").write_bytes(
        (16_000).to_bytes(4, "little")
    )
    store = data_loading._open_disk_backed_record_store(records_path)

    audit = _build_split_audit({"train": store}, hop_length=160)

    assert audit["counts"]["train"]["speakers"] == 1
    assert audit["counts"]["train"]["records_with_speaker_id"] == 1
