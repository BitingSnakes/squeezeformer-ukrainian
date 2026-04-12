from __future__ import annotations

from pathlib import Path

from squeezeformer_pytorch.training.data_loading import (
    _build_disk_backed_record_store,
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
