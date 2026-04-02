import json
import logging
import math
import sys
from dataclasses import asdict
from pathlib import Path

import polars as pl
import pytest
import torch

from squeezeformer_pytorch import (
    NGramLanguageModel,
    SentencePieceTokenizer,
    SqueezeformerCTC,
    build_squeezeformer_encoder,
    squeezeformer_variant,
    tokenizer_from_dict,
)
from squeezeformer_pytorch.asr import load_lm_scorer
from squeezeformer_pytorch.checkpoints import load_checkpoint, save_checkpoint
from squeezeformer_pytorch.data import (
    AdaptiveBatchSampler,
    AudioFeaturizer,
    CV22ASRDataset,
    CVRecord,
    MaxFramesBatchSampler,
    SpecAugment,
    WaveformAugment,
    create_dataloader,
    iter_cv22_corpus_texts,
    iter_cv22_records,
    load_cv22_corpus_texts,
    load_cv22_records,
    normalize_transcript,
    transcript_is_usable,
)
from squeezeformer_pytorch.runtime_types import DTypeChoice, OptimizerChoice
from train import (
    ExponentialMovingAverage,
    _average_topk_checkpoints,
    _build_fp8_recipe,
    _configure_console_logger,
    _update_top_checkpoints,
    _validate_device_argument,
    _validate_fp8_runtime,
    _variant_defaults,
    build_optimizer,
    build_paper_scheduler,
    resolve_device,
    speaker_level_metrics,
)


def expected_subsampled_length(length: int) -> int:
    value = length
    for _ in range(2):
        value = math.floor(value / 2)
    return value


@torch.no_grad()
def test_variant_forward_shapes() -> None:
    batch = 2
    lengths = torch.tensor([160, 123], dtype=torch.int64)
    features = torch.randn(batch, int(lengths.max().item()), 80)

    for variant in ["xs", "sm", "m"]:
        model = build_squeezeformer_encoder(variant)
        model.eval()
        outputs, output_lengths = model(features, lengths)

        cfg = squeezeformer_variant(variant)
        assert outputs.shape[0] == batch
        assert outputs.shape[2] == cfg.d_model
        expected_lengths = torch.tensor(
            [expected_subsampled_length(160), expected_subsampled_length(123)]
        )
        assert torch.equal(output_lengths, expected_lengths)
        assert outputs.shape[1] == int(output_lengths.max().item())
        assert torch.isfinite(outputs).all()


@torch.no_grad()
def test_flash_attention_backend_forward_shapes() -> None:
    lengths = torch.tensor([160, 123], dtype=torch.int64)
    features = torch.randn(2, int(lengths.max().item()), 80)
    model = build_squeezeformer_encoder("xs", attention_backend="flash")
    model.eval()

    outputs, output_lengths = model(features, lengths)

    assert outputs.shape == (2, int(output_lengths.max().item()), model.config.d_model)
    assert torch.equal(
        output_lengths,
        torch.tensor([expected_subsampled_length(160), expected_subsampled_length(123)]),
    )
    assert torch.isfinite(outputs).all()


@torch.no_grad()
def test_transformer_engine_padding_path_preserves_shapes_without_te_runtime() -> None:
    lengths = torch.tensor([160, 123], dtype=torch.int64)
    features = torch.randn(2, int(lengths.max().item()), 80)
    model = build_squeezeformer_encoder(
        "xs",
        attention_backend="flash",
        use_transformer_engine=True,
    )
    model.eval()

    outputs, output_lengths = model(features, lengths)

    assert outputs.shape == (2, int(output_lengths.max().item()), model.config.d_model)
    assert torch.equal(
        output_lengths,
        torch.tensor([expected_subsampled_length(160), expected_subsampled_length(123)]),
    )
    assert torch.isfinite(outputs).all()


@torch.no_grad()
def test_temporal_unet_recovers_subsampled_resolution() -> None:
    model = build_squeezeformer_encoder("sm")
    model.eval()
    lengths = torch.tensor([200], dtype=torch.int64)
    features = torch.randn(1, 200, 80)

    outputs, output_lengths = model(features, lengths)

    assert output_lengths.item() == expected_subsampled_length(200)
    assert outputs.shape[1] == output_lengths.item()


@torch.no_grad()
def test_temporal_unet_handles_odd_length_sequences() -> None:
    model = build_squeezeformer_encoder("sm")
    model.eval()
    lengths = torch.tensor([555], dtype=torch.int64)
    features = torch.randn(1, 555, 80)

    outputs, output_lengths = model(features, lengths)

    assert output_lengths.item() == expected_subsampled_length(555)
    assert outputs.shape[1] == output_lengths.item()
    assert torch.isfinite(outputs).all()


def test_variant_table_matches_paper() -> None:
    expected = {
        "xs": (16, 144, 4),
        "s": (18, 196, 4),
        "sm": (16, 256, 4),
        "m": (20, 324, 4),
        "ml": (18, 512, 8),
        "l": (22, 640, 8),
    }
    for name, values in expected.items():
        cfg = squeezeformer_variant(name)
        assert (cfg.num_layers, cfg.d_model, cfg.num_heads) == values


def test_stochastic_depth_enabled_for_larger_variants() -> None:
    assert squeezeformer_variant("xs").stochastic_depth_rate == 0.0
    assert squeezeformer_variant("m").stochastic_depth_rate > 0.0
    assert (
        squeezeformer_variant("l").stochastic_depth_rate
        > squeezeformer_variant("m").stochastic_depth_rate
    )


def test_sentencepiece_tokenizer_roundtrip(tmp_path: Path) -> None:
    tokenizer = SentencePieceTokenizer.train(
        ["привіт світе", "це короткий тест", "мовна модель"],
        model_prefix=tmp_path / "spm",
        vocab_size=24,
    )
    token_ids = tokenizer.encode("привіт світе")
    assert token_ids
    restored = tokenizer_from_dict(tokenizer.to_dict())
    assert restored.decode(token_ids)


def test_safetensors_checkpoint_roundtrip(tmp_path: Path) -> None:
    tokenizer = SentencePieceTokenizer.train(
        ["привіт світе", "це короткий тест", "мовна модель"],
        model_prefix=tmp_path / "spm_ckpt",
        vocab_size=24,
    )
    model = SqueezeformerCTC(
        encoder_config=squeezeformer_variant("xs"),
        vocab_size=tokenizer.vocab_size,
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "encoder_config": asdict(model.encoder_config),
        "tokenizer": tokenizer.to_dict(),
        "featurizer_config": {"sample_rate": 16_000},
        "training_args": {"dtype": "bfloat16"},
    }

    checkpoint_path = tmp_path / "checkpoint_best.safetensors"
    save_checkpoint(checkpoint, checkpoint_path)
    restored = load_checkpoint(checkpoint_path)

    assert checkpoint_path.exists()
    assert checkpoint_path.with_suffix(".json").exists()
    assert restored["encoder_config"] == checkpoint["encoder_config"]
    assert restored["tokenizer"] == checkpoint["tokenizer"]
    assert restored["training_args"] == checkpoint["training_args"]
    for key, value in checkpoint["model_state_dict"].items():
        assert torch.equal(restored["model_state_dict"][key], value)


def test_ngram_lm_prefers_seen_extension(tmp_path: Path) -> None:
    lm = NGramLanguageModel.train(
        ["abba", "abba", "abbb"],
        order=3,
        alpha=0.1,
    )
    assert lm.score_extension("abb") > lm.score_extension("abz")
    save_path = tmp_path / "shallow_fusion_lm.json"
    lm.save(save_path)
    loaded = NGramLanguageModel.load(save_path)
    assert loaded.score_text("abba") > loaded.score_text("abza")


def test_validate_device_argument_accepts_xla_aliases() -> None:
    assert _validate_device_argument("xla") == "xla"
    assert _validate_device_argument("xla:0") == "xla:0"
    assert _validate_device_argument("tpu:1") == "tpu:1"


def test_resolve_device_requires_torch_xla_for_xla_alias() -> None:
    try:
        import torch_xla.core.xla_model  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="torch_xla"):
            resolve_device("xla")


def test_validate_fp8_runtime_rejects_non_cuda_device() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        _validate_fp8_runtime(torch.device("cpu"), squeezeformer_variant("xs"))


def test_validate_fp8_runtime_rejects_incompatible_hidden_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTE:
        @staticmethod
        def is_fp8_available() -> tuple[bool, str | None]:
            return True, None

    monkeypatch.setattr("train.te", _FakeTE())
    monkeypatch.setattr("train.transformer_engine_available", lambda: True)

    with pytest.raises(ValueError, match="divisible by 16"):
        _validate_fp8_runtime(torch.device("cuda"), squeezeformer_variant("s"))


def test_build_fp8_recipe_uses_requested_recipe_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeFormat:
        HYBRID = "hybrid-format"
        E4M3 = "e4m3-format"

    class _FakeDelayedScaling:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr("train.Format", _FakeFormat)
    monkeypatch.setattr("train.DelayedScaling", _FakeDelayedScaling)

    args = type(
        "Args",
        (),
        {
            "dtype": DTypeChoice.FP8,
            "fp8_format": "e4m3",
            "fp8_amax_history_len": 8,
            "fp8_amax_compute_algo": "most_recent",
        },
    )()
    recipe = _build_fp8_recipe(args)

    assert isinstance(recipe, _FakeDelayedScaling)
    assert recipe.kwargs == {
        "fp8_format": _FakeFormat.E4M3,
        "amax_history_len": 8,
        "amax_compute_algo": "most_recent",
    }


def test_lm_scorer_factory_spec_loads_saved_model(tmp_path: Path) -> None:
    lm = NGramLanguageModel.train(["це тест", "це ще тест"], order=2, alpha=0.1)
    lm_path = tmp_path / "lm.json"
    lm.save(lm_path)
    scorer = load_lm_scorer(f"squeezeformer_pytorch.lm:load_saved_ngram_scorer:{lm_path}")
    assert scorer is not None
    assert scorer("це") > scorer("цz")


def test_load_cv22_corpus_texts_normalizes_and_deduplicates(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\na.wav\t Це   Тест \nb.wav\tце тест\nc.wav\tМовна   Модель\n",
        encoding="utf-8",
    )
    texts = load_cv22_corpus_texts(tmp_path, deduplicate=False)
    assert texts == ["це тест", "це тест", "мовна модель"]
    deduped = load_cv22_corpus_texts(tmp_path, deduplicate=True)
    assert deduped == ["це тест", "мовна модель"]


def test_normalize_transcript_preserves_case_when_lowercase_disabled() -> None:
    assert normalize_transcript(" Це   Тест ", lowercase=False) == "Це Тест"


def test_load_cv22_corpus_texts_preserves_case_when_lowercase_disabled(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\na.wav\t Це   Тест \nb.wav\tце тест\nc.wav\tМовна   Модель\n",
        encoding="utf-8",
    )
    texts = load_cv22_corpus_texts(tmp_path, deduplicate=False, lowercase_transcripts=False)
    assert texts == ["Це Тест", "це тест", "Мовна Модель"]


def test_iter_cv22_corpus_texts_reads_root_level_parquet(tmp_path: Path) -> None:
    pl.DataFrame(
        {
            "path": ["a.wav", "b.wav"],
            "sentence": [" Це   Тест ", "Мовна   Модель"],
        }
    ).write_parquet(tmp_path / "train.parquet")
    texts = list(iter_cv22_corpus_texts(tmp_path))
    assert texts == ["це тест", "мовна модель"]


def test_load_cv22_records_works_without_speaker_id_field(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\tid\tduration\na.wav\tце тест\tutt0\t0.3\nb.wav\tмовна модель\tutt1\t0.3\n",
        encoding="utf-8",
    )
    records = load_cv22_records(
        dataset_root=tmp_path,
        split="train",
        seed=13,
        val_fraction=0.0,
        test_fraction=0.0,
    )
    assert len(records) == 2
    assert all(record.speaker_id is None for record in records)
    assert all(not record.has_speaker_id for record in records)


def test_load_cv22_records_preserves_case_when_lowercase_disabled(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\tid\tduration\na.wav\tЦе Тест\tutt0\t0.3\n",
        encoding="utf-8",
    )
    records = load_cv22_records(
        dataset_root=tmp_path,
        split="train",
        seed=13,
        val_fraction=0.0,
        test_fraction=0.0,
        lowercase_transcripts=False,
    )
    assert [record.transcript for record in records] == ["Це Тест"]


def test_average_topk_checkpoints_logs_shape_mismatch_without_rank_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = logging.getLogger("train")
    logger.handlers.clear()
    _configure_console_logger(rank=0, is_main_process=True)

    topk_dir = tmp_path / "checkpoints_topk"
    topk_dir.mkdir()
    metadata = [
        {"path": "ckpt_a.pt"},
        {"path": "ckpt_b.pt"},
    ]
    (topk_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": {"weight": torch.zeros(2, 2)},
            "tokenizer": {"type": "character", "symbols": ["а"]},
            "encoder_config": asdict(squeezeformer_variant("xs")),
        },
        topk_dir / "ckpt_a.pt",
    )
    torch.save(
        {
            "model_state_dict": {"weight": torch.zeros(3, 3)},
            "tokenizer": {"type": "character", "symbols": ["а"]},
            "encoder_config": asdict(squeezeformer_variant("xs")),
        },
        topk_dir / "ckpt_b.pt",
    )
    monkeypatch.setattr("train._export_inference_checkpoint", lambda checkpoint, path: path)

    averaged_path = _average_topk_checkpoints(tmp_path)

    assert averaged_path == tmp_path / "checkpoint_topk_avg.pt"
    output = capsys.readouterr().out
    assert "Skipping checkpoint" in output
    assert "rank=0" in output


def test_update_top_checkpoints_removes_incompatible_existing_entries(tmp_path: Path) -> None:
    topk_dir = tmp_path / "checkpoints_topk"
    topk_dir.mkdir()
    metadata = [
        {"epoch": 7, "val_wer": 1.0, "path": "old.pt"},
    ]
    (topk_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    save_checkpoint(
        {
            "model_state_dict": {"weight": torch.zeros(3, 3)},
            "tokenizer": {"type": "character", "symbols": ["а"]},
            "encoder_config": asdict(squeezeformer_variant("xs")),
        },
        topk_dir / "old.pt",
    )

    _update_top_checkpoints(
        output_dir=tmp_path,
        checkpoint={
            "model_state_dict": {"weight": torch.zeros(2, 2)},
            "tokenizer": {"type": "character", "symbols": ["а"]},
            "encoder_config": asdict(squeezeformer_variant("xs")),
        },
        epoch=8,
        val_wer=0.9,
        keep_top_k=2,
    )

    saved_metadata = json.loads((topk_dir / "metadata.json").read_text(encoding="utf-8"))
    assert saved_metadata == [
        {"epoch": 8, "val_wer": 0.9, "path": "checkpoint_epoch=0008_valwer=0.900000.pt"}
    ]
    assert not (topk_dir / "old.pt").exists()


def test_iter_cv22_records_streams_split_selection(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\tid\tspeaker_id\tduration\n"
        "a.wav\tце тест\tutt0\tspk0\t0.3\n"
        "b.wav\tмовна модель\tutt1\tspk1\t0.3\n"
        "c.wav\tще приклад\tutt2\tspk2\t0.3\n",
        encoding="utf-8",
    )
    streamed = list(
        iter_cv22_records(
            dataset_root=tmp_path,
            split="train",
            seed=13,
            val_fraction=0.2,
            test_fraction=0.2,
            max_samples=2,
        )
    )
    loaded = load_cv22_records(
        dataset_root=tmp_path,
        split="train",
        seed=13,
        val_fraction=0.2,
        test_fraction=0.2,
        max_samples=2,
    )
    assert [record.utterance_id for record in streamed] == [
        record.utterance_id for record in loaded
    ]
    assert len(streamed) <= 2


def test_transcript_filter_rejects_pathological_rows() -> None:
    assert transcript_is_usable(
        "це нормальний рядок",
        min_chars=3,
        max_chars=40,
        max_symbol_ratio=0.5,
    )
    assert not transcript_is_usable("", min_chars=1, max_chars=40, max_symbol_ratio=0.5)
    assert not transcript_is_usable("а", min_chars=3, max_chars=40, max_symbol_ratio=0.5)
    assert not transcript_is_usable("!" * 8, min_chars=1, max_chars=40, max_symbol_ratio=0.5)
    assert not transcript_is_usable("дуже" * 20, min_chars=1, max_chars=10, max_symbol_ratio=0.5)


def test_max_frames_batch_sampler_respects_frame_budget() -> None:
    records = [
        CVRecord(None, None, "a", "0", estimated_frames=20, speaker_id="s0", has_speaker_id=True),
        CVRecord(None, None, "b", "1", estimated_frames=25, speaker_id="s1", has_speaker_id=True),
        CVRecord(None, None, "c", "2", estimated_frames=40, speaker_id="s2", has_speaker_id=True),
        CVRecord(None, None, "d", "3", estimated_frames=45, speaker_id="s3", has_speaker_id=True),
    ]
    sampler = MaxFramesBatchSampler(records, max_batch_frames=90, shuffle=False)
    batches = list(iter(sampler))
    assert batches
    for batch in batches:
        max_frames = max(records[index].estimated_frames for index in batch)
        assert len(batch) * max_frames <= 90


def test_adaptive_batch_sampler_respects_token_budget() -> None:
    records = [
        CVRecord(
            None,
            None,
            "aaa",
            "0",
            estimated_frames=20,
            speaker_id="s0",
            has_speaker_id=True,
        ),
        CVRecord(
            None,
            None,
            "bbbb",
            "1",
            estimated_frames=25,
            speaker_id="s1",
            has_speaker_id=True,
        ),
        CVRecord(
            None,
            None,
            "cc",
            "2",
            estimated_frames=40,
            speaker_id="s2",
            has_speaker_id=True,
        ),
        CVRecord(
            None,
            None,
            "dddd",
            "3",
            estimated_frames=45,
            speaker_id="s3",
            has_speaker_id=True,
        ),
    ]
    sampler = AdaptiveBatchSampler(records, target_batch_units=6, unit="tokens", shuffle=False)
    batches = list(iter(sampler))
    assert batches
    for batch in batches:
        total_tokens = sum(len(records[index].transcript) for index in batch)
        assert total_tokens <= 6


def test_ema_decay_warmup_increases_toward_target() -> None:
    model = torch.nn.Linear(4, 4)
    ema = ExponentialMovingAverage(model, decay=0.9, warmup_steps=4)
    observed = []
    for _ in range(4):
        ema.update(model)
        observed.append(ema.current_decay())
    assert observed[0] < observed[-1]
    assert observed[-1] == 0.9


def test_ema_update_normalizes_shadow_dtype_before_math() -> None:
    model = torch.nn.Linear(4, 4)
    ema = ExponentialMovingAverage(model, decay=0.9)
    ema.shadow = {name: tensor.to(torch.float64) for name, tensor in ema.shadow.items()}

    ema.update(model)
    backup = ema.apply_to(model)

    assert backup
    for name, parameter in model.named_parameters():
        assert ema.shadow[name].dtype == parameter.dtype


def test_speaker_metrics_mark_missing_speaker_ids_unavailable() -> None:
    metrics = speaker_level_metrics(
        speaker_ids=[None, None],
        has_speaker_ids=[False, False],
        references=["це тест", "мовна модель"],
        hypotheses=["це тест", "мовна модель"],
    )
    assert metrics["speaker_count"] == 0
    assert metrics["speaker_id_available"] is False
    assert metrics["missing_speaker_id_samples"] == 2


def test_time_reduction_kernel_matches_paper() -> None:
    model = build_squeezeformer_encoder("sm")
    reduction = model.time_reduce["7"]
    assert reduction.kernel_size == 3


def test_variant_time_unet_indices_match_reference_layout() -> None:
    expected = {
        "xs": ((7,), (15,)),
        "s": ((8,), (17,)),
        "sm": ((7,), (15,)),
        "m": ((9,), (19,)),
        "ml": ((8,), (17,)),
        "l": ((10,), (21,)),
    }
    for variant, (reduce_idx, recover_idx) in expected.items():
        config = squeezeformer_variant(variant)
        assert config.time_reduce_idx == reduce_idx
        assert config.time_recover_idx == recover_idx


def test_default_block_pattern_matches_paper_layout() -> None:
    config = squeezeformer_variant("sm")
    assert config.block_pattern == ("M", "s", "C", "s")


def test_paper_defaults_use_sentencepiece_compatible_variant_defaults() -> None:
    assert _variant_defaults("sm").peak_lr == 2e-3
    assert _variant_defaults("sm").num_time_masks == 5
    assert _variant_defaults("m").peak_lr == 1.5e-3
    assert _variant_defaults("m").num_time_masks == 7
    assert _variant_defaults("l").peak_lr == 1e-3
    assert _variant_defaults("l").num_time_masks == 10


def test_paper_scheduler_warmup_hold_decay() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_paper_scheduler(
        optimizer,
        steps_per_epoch=2,
        warmup_epochs=1,
        hold_epochs=1,
        decay_exponent=1.0,
    )
    observed = []
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    assert observed[0] == 1.0
    assert observed[1] == 1.0
    assert observed[2] == 1.0
    assert observed[3] < observed[2]
    assert observed[4] < observed[3]


def test_specaugment_preserves_shape() -> None:
    augment = SpecAugment(
        num_freq_masks=2,
        freq_mask_param=4,
        num_time_masks=2,
        time_mask_max_ratio=0.2,
    )
    features = torch.ones(20, 8)
    augmented = augment(features)
    assert augmented.shape == features.shape


def test_feature_cache_is_used_when_waveform_augment_is_effectively_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [len(text)]

    load_calls = 0

    def fake_load_audio(
        audio_path: str | None, audio_bytes: bytes | None
    ) -> tuple[torch.Tensor, int]:
        nonlocal load_calls
        load_calls += 1
        return torch.ones(1, 320), 16_000

    monkeypatch.setattr("squeezeformer_pytorch.data.load_audio", fake_load_audio)

    dataset = CV22ASRDataset(
        records=[CVRecord("dummy.wav", None, "це тест", "utt0", estimated_frames=2)],
        tokenizer=DummyTokenizer(),
        featurizer=AudioFeaturizer(),
        waveform_augment=WaveformAugment(
            speed_perturb_prob=0.0,
            noise_prob=0.0,
            reverb_prob=0.0,
        ),
        feature_cache_dir=tmp_path,
    )

    first_item = dataset[0]
    second_item = dataset[0]

    assert load_calls == 1
    assert torch.equal(first_item["features"], second_item["features"])


def test_create_dataloader_uses_fork_context_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [len(text)]

    captured: dict[str, object] = {}

    class FakeDataLoader:
        def __init__(self, dataset, *args, **kwargs) -> None:
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "squeezeformer_pytorch.data.materialize_record_metadata", lambda *args, **kwargs: args[0]
    )
    monkeypatch.setattr("squeezeformer_pytorch.data.DataLoader", FakeDataLoader)

    dataset = CV22ASRDataset(
        records=[CVRecord("dummy.wav", None, "це тест", "utt0", estimated_frames=2)],
        tokenizer=DummyTokenizer(),
        featurizer=AudioFeaturizer(),
    )
    create_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        prefetch_factor=1,
    )

    kwargs = captured["kwargs"]
    if sys.platform.startswith("linux"):
        assert kwargs["multiprocessing_context"].get_start_method() == "fork"
    else:
        assert "multiprocessing_context" not in kwargs


def test_muon_optimizer_partition_uses_encoder_hidden_weights() -> None:
    config = squeezeformer_variant("xs")
    model = SqueezeformerCTC(encoder_config=config, vocab_size=32)
    optimizers, optimizer_names = build_optimizer(
        model=model,
        optimizer_name=OptimizerChoice.MUON,
        muon_lr=1e-3,
        adamw_lr=5e-4,
        muon_weight_decay=1e-4,
        adamw_weight_decay=5e-5,
    )
    assert optimizer_names == ["muon", "adamw_aux"]
    muon_params = optimizers[0].param_groups[0]["params"]
    adamw_params = optimizers[1].param_groups[0]["params"] + optimizers[1].param_groups[1]["params"]
    assert muon_params
    assert adamw_params
    assert all(parameter.ndim == 2 for parameter in muon_params)
