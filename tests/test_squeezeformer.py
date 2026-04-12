import array
import json
import logging
import math
import os
import pickle
import sys
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import polars as pl
import pytest
import torch

import squeezeformer_pytorch.asr as squeezeformer_asr
import squeezeformer_pytorch.model as squeezeformer_model
import train
from squeezeformer_pytorch import (
    NGramLanguageModel,
    SentencePieceTokenizer,
    SqueezeformerCTC,
    build_squeezeformer_encoder,
    squeezeformer_variant,
    tokenizer_from_dict,
)
from squeezeformer_pytorch.asr import (
    CharacterTokenizer,
    _logaddexp,
    load_lm_scorer,
)
from squeezeformer_pytorch.checkpoints import load_checkpoint, save_checkpoint
from squeezeformer_pytorch.data import (
    AdaptiveBatchSampler,
    ASRDataset,
    AudioFeaturizer,
    AudioRecord,
    DistributedBatchSampler,
    DistributedIndexSampler,
    DurationBatchSampler,
    MaxFramesBatchSampler,
    SpecAugment,
    WaveformAugment,
    collate_asr_batch,
    create_dataloader,
    iter_corpus_texts,
    iter_manifest_rows_from_source,
    iter_records,
    iter_records_from_source,
    load_corpus_texts,
    load_records,
    normalize_transcript,
    probe_audio_metadata,
    transcript_is_usable,
)
from squeezeformer_pytorch.frontend import (
    estimate_num_feature_frames,
    zipformer_paper_featurizer_config,
)
from squeezeformer_pytorch.masking import (
    make_attention_mask,
    make_padding_mask,
    make_sequence_mask,
)
from squeezeformer_pytorch.runtime_types import DecodeStrategy, DTypeChoice, OptimizerChoice
from squeezeformer_pytorch.secrets import REDACTED, sanitize_for_serialization
from squeezeformer_pytorch.training.data_loading import _shard_records_for_rank
from train import (
    DiskBackedRecordStore,
    ExponentialMovingAverage,
    _aed_cross_entropy_loss,
    _average_topk_checkpoints,
    _build_disk_backed_record_store,
    _build_fp8_recipe,
    _build_trackio_cli_arguments_table,
    _build_trackio_grouped_metrics,
    _build_trackio_run_name,
    _configure_console_logger,
    _decode_train_preview_hypotheses,
    _ensure_opus_decode_support,
    _launch_trackio_ui,
    _load_records_from_dataset_roots,
    _load_train_val_records,
    _resolve_dataset_roots,
    _resolve_dataset_sources,
    _resolve_optimizer_learning_rates,
    _should_warn_on_blank_starvation,
    _update_top_checkpoints,
    _validate_device_argument,
    _validate_fp8_runtime,
    _validate_resume_tokenizer_configuration,
    _variant_defaults,
    build_optimizer,
    build_paper_scheduler,
    parse_args,
    resolve_device,
    speaker_level_metrics,
)
from zipformer_pytorch.asr import zipformer_variant as zipformer_asr_variant


def expected_subsampled_length(length: int) -> int:
    value = length
    for _ in range(2):
        value = math.floor(value / 2)
    return value


def test_estimate_num_feature_frames_matches_torchaudio_frontend() -> None:
    featurizer = AudioFeaturizer(**zipformer_paper_featurizer_config())
    for num_samples in (1, 159, 160, 399, 400, 16_000, 32_000):
        actual_frames = featurizer(torch.zeros(num_samples), 16_000).size(0)
        estimated_frames = estimate_num_feature_frames(
            num_samples,
            sample_rate=16_000,
            target_sample_rate=featurizer.sample_rate,
            n_fft=featurizer.n_fft,
            win_length=featurizer.win_length,
            hop_length=featurizer.hop_length,
            backend=featurizer.backend,
        )

        assert estimated_frames == actual_frames


def test_probe_audio_metadata_falls_back_to_load_when_info_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_info(_source):
        raise AttributeError("torchaudio.info is unavailable")

    def fake_load(_source):
        return torch.zeros(1, 48_000), 48_000

    monkeypatch.setattr("squeezeformer_pytorch.data.torchaudio.info", fake_info, raising=False)
    monkeypatch.setattr("squeezeformer_pytorch.data.torchaudio.load", fake_load)

    assert probe_audio_metadata(None, b"opus") == (48_000, 48_000)


def test_audio_preview_filename_strips_source_audio_extension() -> None:
    assert (
        train._audio_preview_filename(0, "/content/cv22/common_voice_uk_38194628.wav")
        == "sample_000000_content_cv22_common_voice_uk_38194628"
    )


def test_save_audio_preview_samples_writes_wav_transcripts_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        AudioRecord("a.opus", None, "перший текст", "utt/0", estimated_frames=10),
        AudioRecord("b.opus", None, "другий текст", "utt/1", estimated_frames=20),
    ]
    saved_audio: list[tuple[str, tuple[int, ...], int]] = []

    def fake_load_audio(audio_path: str | None, audio_bytes: bytes | None):
        del audio_bytes
        length = 4 if audio_path == "a.opus" else 6
        return torch.arange(length, dtype=torch.float32).unsqueeze(0), 48_000

    def fake_save(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
        saved_audio.append((path, tuple(waveform.shape), sample_rate))
        Path(path).write_bytes(b"wav")

    monkeypatch.setattr("train.load_audio", fake_load_audio)
    monkeypatch.setattr("train.torchaudio.save", fake_save)

    count = train._save_audio_preview_samples(
        records,
        output_dir=tmp_path,
        sample_count=2,
        logger=logging.getLogger("test"),
    )

    preview_dir = tmp_path / "audio_previews"
    manifest = [
        json.loads(line)
        for line in (preview_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert count == 2
    assert len(saved_audio) == 2
    assert saved_audio[0][1:] == ((1, 4), 48_000)
    assert saved_audio[1][1:] == ((1, 6), 48_000)
    assert manifest[0]["utterance_id"] == "utt/0"
    assert manifest[0]["transcript"] == "перший текст"
    assert manifest[0]["sample_rate"] == 48_000
    assert Path(manifest[0]["audio_path"]).name.startswith("sample_000000_utt_0")
    assert Path(manifest[0]["audio_path"]).suffix == ".wav"
    assert Path(manifest[0]["transcript_path"]).read_text(encoding="utf-8") == "перший текст\n"


def test_validate_resume_tokenizer_configuration_rejects_tokenizer_type_mismatch() -> None:
    checkpoint = {
        "tokenizer": {
            "type": "sentencepiece",
            "model_proto_b64": "CgkKBG1vY2sQAA==",
        }
    }

    with pytest.raises(RuntimeError) as error:
        _validate_resume_tokenizer_configuration(
            checkpoint=checkpoint,
            checkpoint_path=Path("checkpoint_last.pt"),
            requested_tokenizer_type="character",
            tokenizer_path=None,
        )

    assert "uses tokenizer type 'sentencepiece'" in str(error.value)


def test_build_trackio_run_name_uses_wall_clock_timestamp() -> None:
    run_name = _build_trackio_run_name(
        trackio_project="demo-run",
        output_dir=Path("artifacts/demo"),
        start_epoch=1,
        global_step=0,
        process_start_timestamp=1_712_750_400.0,
    )

    assert run_name == "demo-run_20240410_120000"


def test_should_warn_on_blank_starvation_detects_zero_blank_argmax() -> None:
    assert _should_warn_on_blank_starvation(
        global_step=25,
        avg_blank_probability=0.004,
        argmax_blank_fraction=0.0,
        avg_top_nonblank_probability=0.04,
    )


def test_should_warn_on_blank_starvation_ignores_healthy_blank_signal() -> None:
    assert not _should_warn_on_blank_starvation(
        global_step=25,
        avg_blank_probability=0.6,
        argmax_blank_fraction=0.8,
        avg_top_nonblank_probability=0.2,
    )


def test_decode_train_preview_hypotheses_returns_greedy_and_beam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_decode_batch(*_args, **kwargs):
        calls.append(kwargs)
        strategy = kwargs["strategy"]
        if strategy == DecodeStrategy.GREEDY:
            return ["greedy"]
        return ["beam"]

    monkeypatch.setattr(train, "decode_batch", fake_decode_batch)

    greedy, beam = _decode_train_preview_hypotheses(
        log_probs=torch.zeros(1, 2, 3),
        output_lengths=torch.tensor([2]),
        tokenizer=object(),
        beam_size=5,
        lm_scorer=None,
        lm_weight=0.0,
        beam_length_bonus=0.25,
    )

    assert (greedy, beam) == ("greedy", "beam")
    assert [call["strategy"] for call in calls] == [DecodeStrategy.GREEDY, DecodeStrategy.BEAM]
    assert calls[1]["beam_length_bonus"] == 0.25


def test_validate_resume_tokenizer_configuration_rejects_tokenizer_path_mismatch(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    CharacterTokenizer(symbols=["а", "б"]).save(tokenizer_path)
    checkpoint = {"tokenizer": CharacterTokenizer(symbols=["а", "в"]).to_dict()}

    with pytest.raises(RuntimeError) as error:
        _validate_resume_tokenizer_configuration(
            checkpoint=checkpoint,
            checkpoint_path=Path("checkpoint_last.pt"),
            requested_tokenizer_type="character",
            tokenizer_path=str(tokenizer_path),
        )

    assert "does not match the tokenizer loaded" in str(error.value)


def test_build_trackio_cli_arguments_table_records_explicit_arguments() -> None:
    table = _build_trackio_cli_arguments_table(
        [
            "--device",
            "cpu",
            "--batch-size=16",
            "--dataset-source",
            "train-a.parquet",
            "--dataset-source",
            "train-b.parquet",
            "--no-record-cache",
            "--muon-decay-exponent",
            "-0.5",
        ]
    )

    assert table is not None
    assert table.data.to_dict("records") == [
        {"position": 1, "argument": "--device", "value": "cpu"},
        {"position": 2, "argument": "--batch-size", "value": "16"},
        {"position": 3, "argument": "--dataset-source", "value": "train-a.parquet"},
        {"position": 4, "argument": "--dataset-source", "value": "train-b.parquet"},
        {"position": 5, "argument": "--no-record-cache", "value": True},
        {"position": 6, "argument": "--muon-decay-exponent", "value": "-0.5"},
    ]


def test_build_trackio_grouped_metrics_returns_slash_prefixed_metrics() -> None:
    metrics = _build_trackio_grouped_metrics(
        groups={
            "losses": {"train_loss_step": 1.25, "train_aed_loss_step": 0.2},
            "optimization": {"grad_norm": 4.5},
        },
    )

    assert metrics == {
        "losses/train_loss_step": 1.25,
        "losses/train_aed_loss_step": 0.2,
        "optimization/grad_norm": 4.5,
    }


def test_launch_trackio_ui_uses_trackio_show_without_blocking(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_show(**kwargs):
        captured["kwargs"] = kwargs
        captured["trackio_dir_during_call"] = os.environ.get("TRACKIO_DIR")
        captured["gradio_share_during_call"] = os.environ.get("GRADIO_SHARE")
        return ("app", "http://127.0.0.1:7860/", "https://demo.gradio.live", "https://full")

    monkeypatch.setattr(train.trackio, "show", fake_show)
    monkeypatch.delenv("TRACKIO_DIR", raising=False)
    monkeypatch.delenv("GRADIO_SHARE", raising=False)

    result = _launch_trackio_ui(
        trackio_dir=tmp_path / "trackio",
        logger=logging.getLogger("test"),
        project="demo-project",
    )

    assert result == ("app", "http://127.0.0.1:7860/", "https://demo.gradio.live", "https://full")
    assert captured["kwargs"] == {
        "project": "demo-project",
        "open_browser": False,
        "block_thread": False,
    }
    assert captured["trackio_dir_during_call"] == str(tmp_path / "trackio")
    assert captured["gradio_share_during_call"] == "True"
    assert "TRACKIO_DIR" not in os.environ
    assert "GRADIO_SHARE" not in os.environ


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


def test_squeezeformer_training_outputs_include_audio_teacher_states() -> None:
    config = squeezeformer_variant("xs")
    model = SqueezeformerCTC(
        encoder_config=config,
        vocab_size=8,
        audio_teacher_enabled=True,
        audio_teacher_hidden_size=24,
    )
    features = torch.randn(2, 48, 80)
    feature_lengths = torch.tensor([48, 40], dtype=torch.long)
    targets = torch.tensor([1, 2, 3, 1, 2], dtype=torch.long)
    target_lengths = torch.tensor([3, 2], dtype=torch.long)

    outputs = model(
        features,
        feature_lengths,
        return_training_outputs=True,
        targets=targets,
        target_lengths=target_lengths,
        blank_id=0,
    )

    assert "audio_teacher_student_states" in outputs
    assert outputs["audio_teacher_student_states"].shape == (2, 24)
    assert torch.isfinite(outputs).all()


@torch.no_grad()
def test_transformer_engine_padding_path_preserves_shapes_without_te_runtime() -> None:
    lengths = torch.tensor([160, 123], dtype=torch.int64)
    features = torch.randn(2, int(lengths.max().item()), 80)
    model = build_squeezeformer_encoder(
        "xs",
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
def test_encoder_zeroes_padded_suffixes_after_each_block() -> None:
    model = build_squeezeformer_encoder("xs")
    model.eval()

    features = torch.randn(2, 16, 80)
    lengths = torch.tensor([16, 12], dtype=torch.int64)
    outputs, output_lengths = model(features, lengths)

    assert outputs.shape == (2, int(output_lengths.max().item()), model.config.d_model)
    assert torch.count_nonzero(outputs[1, int(output_lengths[1].item()) :, :]) == 0


@torch.no_grad()
def test_convolution_module_masks_padded_suffix_before_batch_norm_statistics() -> None:
    module = squeezeformer_model.ConvolutionModule(
        dim=4,
        kernel_size=3,
        expansion_factor=2,
        dropout=0.0,
    )
    module.train()

    lengths = torch.tensor([5, 3], dtype=torch.int64)
    pad_mask = make_sequence_mask(lengths, max_length=5)

    clean = torch.randn(2, 5, 4)
    masked = clean.clone()
    masked[1, 3:, :] = 0.0

    polluted = clean.clone()
    polluted[1, 3:, :] = 1_000.0

    clean_out = module(masked, pad_mask=pad_mask)

    module_patched = squeezeformer_model.ConvolutionModule(
        dim=4,
        kernel_size=3,
        expansion_factor=2,
        dropout=0.0,
    )
    module_patched.load_state_dict(module.state_dict())
    module_patched.train()
    polluted_out = module_patched(polluted, pad_mask=pad_mask)

    assert torch.allclose(clean_out[:, :3, :], polluted_out[:, :3, :], atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_ctc_head_defaults_to_zero_blank_bias() -> None:
    model = SqueezeformerCTC(
        encoder_config=squeezeformer_variant("xs"),
        vocab_size=16,
    )

    assert torch.isclose(model.classifier.bias[0], torch.tensor(0.0))
    assert torch.count_nonzero(model.classifier.bias[1:]) == 0


def test_ctc_length_diagnostics_count_repeated_adjacent_targets() -> None:
    targets = torch.tensor([1, 1, 2, 3, 4, 4], dtype=torch.long)
    target_lengths = torch.tensor([3, 3], dtype=torch.long)
    minimum_lengths = SqueezeformerCTC._ctc_minimum_alignment_lengths(
        targets,
        target_lengths,
    )

    diagnostics = SqueezeformerCTC._ctc_length_diagnostics(
        torch.tensor([3, 4], dtype=torch.long),
        targets,
        torch.tensor([3, 3], dtype=torch.long),
    )

    assert torch.equal(minimum_lengths, torch.tensor([4, 4], dtype=torch.long))
    assert diagnostics["impossible_sample_count"] == 1.0
    assert diagnostics["tight_sample_count"] == 2.0


@torch.no_grad()
def test_ctc_log_probs_promote_bfloat16_inputs_to_float32() -> None:
    model = SqueezeformerCTC(
        encoder_config=squeezeformer_variant("xs"),
        vocab_size=8,
    )
    model.classifier = model.classifier.to(dtype=torch.bfloat16)
    x = torch.randn(2, 5, model.encoder_config.d_model, dtype=torch.bfloat16)

    _logits, log_probs = model._chunked_logits_and_log_probs_from_classifier(model.classifier, x)

    assert log_probs.dtype == torch.float32


def test_chunked_ctc_loss_requests_float32_log_softmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SqueezeformerCTC(
        encoder_config=squeezeformer_variant("xs"),
        vocab_size=8,
    )
    model.classifier = model.classifier.to(dtype=torch.bfloat16)
    x = torch.randn(2, 6, model.encoder_config.d_model, dtype=torch.bfloat16)
    output_lengths = torch.tensor([6, 5], dtype=torch.long)
    targets = torch.tensor([1, 2, 3, 1, 2], dtype=torch.long)
    target_lengths = torch.tensor([3, 2], dtype=torch.long)

    observed_dtypes: list[torch.dtype | None] = []
    original_log_softmax = squeezeformer_asr.F.log_softmax

    def wrapped_log_softmax(*args, **kwargs):
        observed_dtypes.append(kwargs.get("dtype"))
        return original_log_softmax(*args, **kwargs)

    monkeypatch.setattr(squeezeformer_asr.F, "log_softmax", wrapped_log_softmax)

    loss = model._chunked_ctc_loss_from_classifier(
        model.classifier,
        x,
        output_lengths,
        targets,
        target_lengths,
        blank_id=0,
    )

    assert loss.item() >= 0.0
    assert observed_dtypes == [torch.float32]


def test_sentencepiece_tokenizer_rejects_empty_piece_nonblank_token() -> None:
    class FakeProcessor:
        def vocab_size(self) -> int:
            return 3

        def decode(self, token_ids) -> str:
            mapping = {
                1: "a",
                2: " ",
            }
            return "".join(mapping[token_id] for token_id in token_ids)

        def id_to_piece(self, token_id: int) -> str:
            return {
                0: "<pad>",
                1: "a",
                2: "",
            }[token_id]

    with pytest.raises(ValueError) as error:
        SentencePieceTokenizer(processor=FakeProcessor(), model_proto=b"fake")

    assert "with empty pieces" in str(error.value)


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


@torch.no_grad()
def test_encoder_handles_physically_tiny_inputs_without_crashing() -> None:
    model = build_squeezeformer_encoder("sm")
    model.eval()

    for length in (1, 2, 3, 4, 5, 7):
        lengths = torch.tensor([length], dtype=torch.int64)
        features = torch.randn(1, length, 80)

        outputs, output_lengths = model(features, lengths)

        assert output_lengths.item() >= 1
        assert outputs.shape == (1, int(output_lengths.item()), model.config.d_model)
        assert torch.isfinite(outputs).all()


@torch.no_grad()
def test_encoder_can_return_mid_layer_hidden_state() -> None:
    model = build_squeezeformer_encoder("xs")
    model.eval()
    lengths = torch.tensor([160, 123], dtype=torch.int64)
    features = torch.randn(2, int(lengths.max().item()), 80)

    outputs, output_lengths, intermediates, intermediate_lengths = model.forward_with_intermediates(
        features,
        lengths,
        intermediate_layer_indices=(7,),
    )
    intermediate = intermediates[7]
    captured_lengths = intermediate_lengths[7]

    assert outputs.shape[0] == intermediate.shape[0] == 2
    assert outputs.shape[2] == intermediate.shape[2] == model.config.d_model
    assert torch.equal(output_lengths, torch.tensor([40, 30]))
    assert torch.equal(captured_lengths, torch.tensor([20, 15]))
    assert intermediate.shape[1] == int(captured_lengths.max().item())
    assert torch.isfinite(intermediate).all()


@torch.no_grad()
def test_ctc_model_can_emit_training_only_aed_logits() -> None:
    model = SqueezeformerCTC(
        encoder_config=squeezeformer_variant("xs"),
        vocab_size=10,
        aed_decoder_enabled=True,
        aed_decoder_layers=1,
        aed_decoder_heads=4,
        liberta_distill_enabled=True,
    )
    model.eval()
    lengths = torch.tensor([160, 123], dtype=torch.int64)
    features = torch.randn(2, int(lengths.max().item()), 80)
    decoder_inputs = torch.tensor(
        [
            [1, 4, 5, 0],
            [1, 6, 0, 0],
        ],
        dtype=torch.int64,
    )

    logits, output_lengths, hidden = model.aed_forward(features, lengths, decoder_inputs)
    projected = model.project_aed_hidden_for_liberta(
        hidden, torch.tensor([3, 2], dtype=torch.int64)
    )

    assert logits.shape == (2, 4, 13)
    assert hidden.shape == (2, 4, model.encoder_config.d_model)
    assert output_lengths.shape == (2,)
    assert projected.shape == (2, 1024)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(projected).all()


def test_sequence_mask_helpers_share_consistent_semantics() -> None:
    lengths = torch.tensor([3, 1], dtype=torch.int64)

    sequence_mask = make_sequence_mask(lengths, max_length=4)
    padding_mask = make_padding_mask(lengths, max_length=4)
    attention_mask = make_attention_mask(lengths, max_length=4)

    assert torch.equal(
        sequence_mask,
        torch.tensor([[True, True, True, False], [True, False, False, False]]),
    )
    assert torch.equal(padding_mask, ~sequence_mask)
    assert torch.equal(
        attention_mask[0],
        sequence_mask[0].unsqueeze(0) & sequence_mask[0].unsqueeze(1),
    )
    assert not attention_mask[1, 1, 1]


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


def test_default_adamw_learning_rate_is_capped_when_not_explicitly_set() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
        ]
    )

    peak_lr, muon_lr, adamw_lr = _resolve_optimizer_learning_rates(
        args,
        variant_defaults=_variant_defaults("sm"),
    )

    assert peak_lr == 2e-3
    assert muon_lr == 2e-3
    assert adamw_lr == 3e-4


def test_explicit_learning_rate_still_applies_to_default_adamw_path() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--learning-rate",
            "0.0015",
        ]
    )

    peak_lr, muon_lr, adamw_lr = _resolve_optimizer_learning_rates(
        args,
        variant_defaults=_variant_defaults("sm"),
    )

    assert peak_lr == 0.0015
    assert muon_lr == 0.0015
    assert adamw_lr == 0.0015


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


def test_character_tokenizer_encode_fails_on_oov_character() -> None:
    tokenizer = CharacterTokenizer(symbols=["а", "б"])

    with pytest.raises(ValueError) as error:
        tokenizer.encode("абв")

    assert "'в'" in str(error.value)


def test_logaddexp_matches_torch_for_finite_and_infinite_inputs() -> None:
    pairs = [
        (float("-inf"), -3.0),
        (-2.0, float("-inf")),
        (-5.0, -1.0),
        (-0.5, -0.75),
    ]

    for left, right in pairs:
        expected = torch.logaddexp(torch.tensor(left), torch.tensor(right)).item()
        assert _logaddexp(left, right) == pytest.approx(expected)


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


def test_safetensors_checkpoint_metadata_redacts_hf_tokens(tmp_path: Path) -> None:
    tokenizer = SentencePieceTokenizer.train(
        ["привіт світе", "це короткий тест", "мовна модель"],
        model_prefix=tmp_path / "spm_scrub_ckpt",
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
        "training_args": {
            "hf_token": "hf_abcdefghijklmnopqrstuvwxyz123456",
            "nested": {
                "authorization": "Bearer hf_abcdefghijklmnopqrstuvwxyz123456",
            },
        },
    }

    checkpoint_path = tmp_path / "checkpoint_last.safetensors"
    save_checkpoint(checkpoint, checkpoint_path)

    metadata = json.loads(checkpoint_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["training_args"]["hf_token"] == REDACTED
    assert REDACTED in metadata["training_args"]["nested"]["authorization"]
    assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in json.dumps(metadata)


def test_sanitize_for_serialization_redacts_recursive_secret_fields() -> None:
    payload = {
        "hf_token": "hf_abcdefghijklmnopqrstuvwxyz123456",
        "training_args": {
            "api_key": "hf_abcdefghijklmnopqrstuvwxyz123456",
            "dataset_repo": "speech-uk/squeezeformer-sm",
            "nested": [
                {"authorization": "Bearer hf_abcdefghijklmnopqrstuvwxyz123456"},
                "hf_abcdefghijklmnopqrstuvwxyz123456",
            ],
        },
        "tokenizer": {"type": "sentencepiece"},
    }

    sanitized = sanitize_for_serialization(payload)

    assert sanitized["hf_token"] == REDACTED
    assert sanitized["training_args"]["api_key"] == REDACTED
    assert REDACTED in sanitized["training_args"]["nested"][0]["authorization"]
    assert sanitized["training_args"]["nested"][1] == REDACTED
    assert sanitized["tokenizer"] == payload["tokenizer"]


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


def test_validate_device_argument_accepts_standard_torch_devices() -> None:
    assert _validate_device_argument("cpu") == "cpu"
    assert _validate_device_argument("cuda:0") == "cuda:0"


def test_resolve_device_uses_torch_device() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


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


def test_validate_fp8_runtime_accepts_zipformer_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTE:
        @staticmethod
        def is_fp8_available() -> tuple[bool, str | None]:
            return True, None

    monkeypatch.setattr("train.te", _FakeTE())
    monkeypatch.setattr("train.transformer_engine_available", lambda: True)

    _validate_fp8_runtime(torch.device("cuda"), zipformer_asr_variant("xs"))


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


def test_load_corpus_texts_normalizes_and_deduplicates(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\na.wav\t Це   Тест \nb.wav\tце тест\nc.wav\tМовна   Модель\n",
        encoding="utf-8",
    )
    texts = load_corpus_texts(tmp_path, deduplicate=False)
    assert texts == ["це тест", "це тест", "мовна модель"]
    deduped = load_corpus_texts(tmp_path, deduplicate=True)
    assert deduped == ["це тест", "мовна модель"]


def test_normalize_transcript_preserves_case_when_lowercase_disabled() -> None:
    assert normalize_transcript(" Це   Тест ", lowercase=False) == "Це Тест"


def test_load_corpus_texts_preserves_case_when_lowercase_disabled(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\na.wav\t Це   Тест \nb.wav\tце тест\nc.wav\tМовна   Модель\n",
        encoding="utf-8",
    )
    texts = load_corpus_texts(tmp_path, deduplicate=False, lowercase_transcripts=False)
    assert texts == ["Це Тест", "це тест", "Мовна Модель"]


def test_iter_corpus_texts_reads_root_level_parquet(tmp_path: Path) -> None:
    pl.DataFrame(
        {
            "path": ["a.wav", "b.wav"],
            "sentence": [" Це   Тест ", "Мовна   Модель"],
        }
    ).write_parquet(tmp_path / "train.parquet")
    texts = list(iter_corpus_texts(tmp_path))
    assert texts == ["це тест", "мовна модель"]


def test_load_records_works_without_speaker_id_field(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\tid\tduration\na.wav\tце тест\tutt0\t0.3\nb.wav\tмовна модель\tutt1\t0.3\n",
        encoding="utf-8",
    )
    records = load_records(
        dataset_root=tmp_path,
        split="train",
        seed=13,
        val_fraction=0.0,
        test_fraction=0.0,
    )
    assert len(records) == 2
    assert all(record.speaker_id is None for record in records)
    assert all(not record.has_speaker_id for record in records)


def test_load_records_estimates_centered_torchaudio_frames(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\tid\tduration\na.wav\tце тест\tutt0\t1.0\n",
        encoding="utf-8",
    )

    records = load_records(
        dataset_root=tmp_path,
        split="train",
        seed=13,
        val_fraction=0.0,
        test_fraction=0.0,
    )

    assert records[0].num_samples == 16_000
    assert records[0].estimated_frames == 101


def test_load_records_preserves_case_when_lowercase_disabled(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\tid\tduration\na.wav\tЦе Тест\tutt0\t0.3\n",
        encoding="utf-8",
    )
    records = load_records(
        dataset_root=tmp_path,
        split="train",
        seed=13,
        val_fraction=0.0,
        test_fraction=0.0,
        lowercase_transcripts=False,
    )
    assert [record.transcript for record in records] == ["Це Тест"]


def test_resolve_dataset_roots_uses_dataset_source_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    def fake_download_dataset(
        repo_id: str,
        token: str | None,
        cache_dir: str | None = None,
        force_download: bool = False,
        allow_patterns: list[str] | None = None,
    ) -> Path:
        del token, cache_dir, force_download, allow_patterns
        return {"source-a": first, "source-b": second}[repo_id]

    monkeypatch.setattr("train.download_dataset", fake_download_dataset)

    args = type(
        "Args",
        (),
        {
            "dataset_repo": "speech-uk/cv22",
            "dataset_source": ["source-a", "source-b", "source-a"],
            "hf_token": None,
            "cache_dir": None,
        },
    )()

    assert _resolve_dataset_roots(args) == [first.resolve(), second.resolve()]


def test_resolve_dataset_sources_preserves_remote_parquet_urls() -> None:
    args = type(
        "Args",
        (),
        {
            "dataset_repo": "speech-uk/cv22",
            "dataset_source": [
                "https://huggingface.co/datasets/speech-uk/cv22/resolve/main/0.parquet",
                "https://huggingface.co/datasets/speech-uk/cv22/resolve/main/0.parquet",
            ],
        },
    )()

    assert _resolve_dataset_sources(args) == [
        "https://huggingface.co/datasets/speech-uk/cv22/resolve/main/0.parquet"
    ]


def test_parse_args_supports_no_record_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--device", "cpu", "--no-record-cache"],
    )

    args = parse_args()

    assert args.record_cache is False


def test_parse_args_supports_max_batch_duration_sec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--device", "cpu", "--max-batch-duration-sec", "12.5"],
    )

    args = parse_args()

    assert args.max_batch_duration_sec == 12.5


def test_parse_args_supports_audio_teacher_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--device",
            "cpu",
            "--audio-teacher",
            "--audio-teacher-model-name",
            "facebook/wav2vec2-bert-2.0",
            "--audio-teacher-device",
            "cpu",
            "--audio-teacher-weight",
            "0.2",
            "--audio-teacher-layer",
            "5",
        ],
    )

    args = parse_args()

    assert args.audio_teacher is True
    assert args.audio_teacher_model_name == "facebook/wav2vec2-bert-2.0"
    assert args.audio_teacher_device == "cpu"
    assert args.audio_teacher_weight == pytest.approx(0.2)
    assert args.audio_teacher_layer == 5


def test_upload_checkpoint_folder_to_hf_uses_cli_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "checkpoint_last.pt").write_bytes(b"checkpoint")
    calls: list[dict[str, object]] = []

    class FakeHfApi:
        def upload_folder(self, **kwargs):
            calls.append(kwargs)
            return type("CommitInfo", (), {"commit_url": "https://huggingface.co/commit"})()

    monkeypatch.setattr(train, "HfApi", lambda: FakeHfApi())

    args = type(
        "Args",
        (),
        {
            "hf_upload_checkpoints": True,
            "hf_upload_repo_id": "speech-uk/checkpoints",
            "hf_upload_repo_type": "dataset",
            "hf_upload_path_in_repo": "/runs/demo/",
            "hf_upload_revision": "main",
            "hf_upload_checkpoint_format": "safetensors",
            "hf_upload_allow_pattern": ["checkpoint*"],
            "hf_upload_ignore_pattern": ["*.tmp"],
            "hf_upload_fail_on_error": True,
            "hf_upload_token": "upload-token",
            "hf_token": "hf_abcdefghijklmnopqrstuvwxyz123456",
        },
    )()

    upload_info = train._upload_checkpoint_folder_to_hf(
        args=args,
        output_dir=tmp_path,
        logger=logging.getLogger("test"),
        commit_message="Upload checkpoints",
    )

    assert upload_info is not None
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "speech-uk/checkpoints"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["folder_path"] == tmp_path
    assert calls[0]["path_in_repo"] == "runs/demo"
    assert calls[0]["revision"] == "main"
    assert calls[0]["allow_patterns"] == ["checkpoint*"]
    assert calls[0]["ignore_patterns"] == [
        "record_cache/**",
        "feature_cache/**",
        "trackio/**",
        "audio_previews/**",
        "*.pt",
        "**/*.pt",
        "*.tmp",
    ]
    assert calls[0]["token"] == "upload-token"


def test_initialize_hf_checkpoint_repository_creates_repo_with_upload_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeHfApi:
        def create_repo(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(train, "HfApi", lambda: FakeHfApi())

    args = type(
        "Args",
        (),
        {
            "hf_upload_checkpoints": True,
            "hf_upload_repo_id": "speech-uk/checkpoints",
            "hf_upload_repo_type": "model",
            "hf_upload_path_in_repo": "runs/demo",
            "hf_upload_checkpoint_format": "all",
            "hf_upload_create_repo": True,
            "hf_upload_private": True,
            "hf_upload_fail_on_error": True,
            "hf_upload_token": "upload-token",
            "hf_token": "dataset-token",
        },
    )()

    train._initialize_hf_checkpoint_repository(
        args,
        output_dir=tmp_path,
        logger=logging.getLogger("test"),
    )

    assert calls == [
        {
            "repo_id": "speech-uk/checkpoints",
            "token": "upload-token",
            "private": True,
            "repo_type": None,
            "exist_ok": True,
        }
    ]


def test_hf_upload_checkpoint_format_pt_ignores_safetensors_artifacts() -> None:
    args = type(
        "Args",
        (),
        {
            "hf_upload_checkpoint_format": "pt",
            "hf_upload_ignore_pattern": ["*.tmp"],
        },
    )()

    assert train._hf_upload_ignore_patterns(args) == [
        "record_cache/**",
        "feature_cache/**",
        "trackio/**",
        "audio_previews/**",
        "*.safetensors",
        "**/*.safetensors",
        "checkpoint_*.json",
        "**/checkpoint_*.json",
        "*.tmp",
    ]


def test_parse_args_supports_alignment_filter_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--device",
            "cpu",
            "--min-chars-per-second",
            "1.5",
            "--max-chars-per-second",
            "18",
            "--min-words-per-second",
            "0.5",
            "--max-words-per-second",
            "5",
            "--min-duration-per-char",
            "0.02",
            "--max-duration-per-char",
            "0.6",
            "--min-duration-per-word",
            "0.15",
            "--max-duration-per-word",
            "3.0",
        ],
    )

    args = parse_args()

    assert args.min_chars_per_second == 1.5
    assert args.max_chars_per_second == 18.0
    assert args.min_words_per_second == 0.5
    assert args.max_words_per_second == 5.0
    assert args.min_duration_per_char == 0.02
    assert args.max_duration_per_char == 0.6
    assert args.min_duration_per_word == 0.15
    assert args.max_duration_per_word == 3.0


def test_parse_args_supports_legacy_token_alignment_filter_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--device",
            "cpu",
            "--min-tokens-per-second",
            "0.5",
            "--max-tokens-per-second",
            "5",
            "--min-duration-per-token",
            "0.15",
            "--max-duration-per-token",
            "3.0",
        ],
    )

    args = parse_args()

    assert args.min_words_per_second == 0.5
    assert args.max_words_per_second == 5.0
    assert args.min_duration_per_word == 0.15
    assert args.max_duration_per_word == 3.0


def test_load_records_from_dataset_roots_combines_sources_with_global_limit(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "train.tsv").write_text(
        "path\tsentence\tid\tduration\n"
        "a.wav\tперший запис\tutt0\t0.3\n"
        "b.wav\tдругий запис\tutt1\t0.3\n",
        encoding="utf-8",
    )
    (second / "train.tsv").write_text(
        "path\tsentence\tid\tduration\n"
        "c.wav\tтретій запис\tutt2\t0.3\n"
        "d.wav\tчетвертий запис\tutt3\t0.3\n",
        encoding="utf-8",
    )

    records = _load_records_from_dataset_roots(
        [first, second],
        split="train",
        seed=13,
        val_fraction=0.0,
        test_fraction=0.0,
        max_samples=3,
        min_transcript_chars=1,
        max_transcript_chars=400,
        max_symbol_ratio=0.5,
        lowercase_transcripts=True,
    )

    assert [record.utterance_id for record in records] == ["utt0", "utt1", "utt2"]


def test_build_disk_backed_record_store_combines_sources_with_global_limit(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "train.tsv").write_text(
        "path\tsentence\tid\tduration\n"
        "a.wav\tперший запис\tutt0\t0.3\n"
        "b.wav\tдругий запис\tutt1\t0.3\n",
        encoding="utf-8",
    )
    (second / "train.tsv").write_text(
        "path\tsentence\tid\tduration\n"
        "c.wav\tтретій запис\tutt2\t0.3\n"
        "d.wav\tчетвертий запис\tutt3\t0.3\n",
        encoding="utf-8",
    )

    store = _build_disk_backed_record_store(
        [first, second],
        split="train",
        seed=13,
        val_fraction=0.0,
        test_fraction=0.0,
        max_samples=3,
        min_transcript_chars=1,
        max_transcript_chars=400,
        max_symbol_ratio=0.5,
        lowercase_transcripts=True,
        records_path=tmp_path / "records" / "train.jsonl",
    )

    assert isinstance(store, DiskBackedRecordStore)
    assert len(store) == 3
    assert [record.utterance_id for record in store] == ["utt0", "utt1", "utt2"]
    assert (tmp_path / "records" / "train.jsonl.offsets.u64").exists()
    assert (tmp_path / "records" / "train.jsonl.estimated_frames.u32").exists()
    assert (tmp_path / "records" / "train.jsonl.num_samples.u64").exists()
    assert (tmp_path / "records" / "train.jsonl.sample_rates.u32").exists()
    assert all(record.num_samples > 0 for record in store)
    assert all(record.sample_rate > 0 for record in store)


def test_build_disk_backed_record_store_reads_parquet_manifest_file(tmp_path: Path) -> None:
    manifest_path = tmp_path / "train.parquet"
    pl.DataFrame(
        {
            "path": ["a.wav", "b.wav"],
            "sentence": ["перший запис", "другий запис"],
            "id": ["utt0", "utt1"],
            "duration": [0.3, 0.3],
        }
    ).write_parquet(manifest_path)

    store = _build_disk_backed_record_store(
        [manifest_path],
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
    )

    assert isinstance(store, DiskBackedRecordStore)
    assert len(store) == 2
    assert [record.utterance_id for record in store] == ["utt0", "utt1"]


def test_load_train_val_records_without_record_cache_uses_in_memory_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_rows = [
        AudioRecord(
            audio_path="a.wav",
            audio_bytes=None,
            transcript="перший запис",
            utterance_id="utt0",
            estimated_frames=10,
        )
    ]
    val_rows = [
        AudioRecord(
            audio_path="b.wav",
            audio_bytes=None,
            transcript="другий запис",
            utterance_id="utt1",
            estimated_frames=12,
        )
    ]
    calls: list[str] = []

    def fake_load_records_from_dataset_roots(
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
        min_chars_per_second: float,
        max_chars_per_second: float,
        min_words_per_second: float,
        max_words_per_second: float,
        min_duration_per_char: float,
        max_duration_per_char: float,
        min_duration_per_word: float,
        max_duration_per_word: float,
        lowercase_transcripts: bool,
        hf_token: str | None = None,
        cache_dir: str | None = None,
    ) -> list[AudioRecord]:
        del (
            dataset_sources,
            seed,
            val_fraction,
            test_fraction,
            max_samples,
            min_transcript_chars,
            max_transcript_chars,
            max_symbol_ratio,
            min_chars_per_second,
            max_chars_per_second,
            min_words_per_second,
            max_words_per_second,
            min_duration_per_char,
            max_duration_per_char,
            min_duration_per_word,
            max_duration_per_word,
            lowercase_transcripts,
            hf_token,
            cache_dir,
        )
        calls.append(split)
        return train_rows if split == "train" else val_rows

    def fake_build_disk_backed_record_store(*args, **kwargs):
        raise AssertionError("disk-backed store should not be used when --no-record-cache is set")

    monkeypatch.setattr(
        "train._load_records_from_dataset_roots", fake_load_records_from_dataset_roots
    )
    monkeypatch.setattr(
        "train._build_disk_backed_record_store", fake_build_disk_backed_record_store
    )

    args = type(
        "Args",
        (),
        {
            "record_cache": False,
            "record_cache_dir": None,
            "seed": 13,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "max_train_samples": None,
            "max_val_samples": None,
            "min_transcript_chars": 1,
            "max_transcript_chars": 400,
            "max_symbol_ratio": 0.5,
            "min_chars_per_second": 0.0,
            "max_chars_per_second": float("inf"),
            "min_words_per_second": 0.0,
            "max_words_per_second": float("inf"),
            "min_duration_per_char": 0.0,
            "max_duration_per_char": float("inf"),
            "min_duration_per_word": 0.0,
            "max_duration_per_word": float("inf"),
            "hf_token": None,
            "cache_dir": None,
            "prevalidate_audio": False,
            "prevalidate_workers": 1,
        },
    )()

    train_records, val_records = _load_train_val_records(
        args,
        [tmp_path / "data"],
        lowercase_transcripts=True,
        output_dir=tmp_path / "artifacts",
    )

    assert calls == ["train", "validation"]
    assert train_records == train_rows
    assert val_records == val_rows


def test_ensure_opus_decode_support_fails_fast_for_in_memory_opus_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        AudioRecord(
            audio_path="sample.opus",
            audio_bytes=None,
            transcript="приклад",
            utterance_id="utt0",
            estimated_frames=10,
        )
    ]

    def fake_load_audio(audio_path: str | None, audio_bytes: bytes | None):
        del audio_path, audio_bytes
        raise RuntimeError("missing opus decoder")

    monkeypatch.setattr("train.load_audio", fake_load_audio)

    with pytest.raises(RuntimeError, match="contains Opus audio"):
        _ensure_opus_decode_support(records, split="train")


def test_ensure_opus_decode_support_checks_disk_backed_record_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records_path = tmp_path / "train.jsonl"
    payload = {
        "audio_path": "sample.opus",
        "audio_blob_path": None,
        "transcript": "приклад",
        "utterance_id": "utt0",
        "speaker_id": None,
        "has_speaker_id": False,
    }
    records_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    store = DiskBackedRecordStore(
        records_path,
        array.array("Q", [0]),
        array.array("I", [1]),
        array.array("Q", [16]),
        array.array("I", [16_000]),
    )
    observed: list[tuple[str | None, bytes | None]] = []

    def fake_load_audio(audio_path: str | None, audio_bytes: bytes | None):
        observed.append((audio_path, audio_bytes))
        return torch.zeros(1, 16), 16_000

    monkeypatch.setattr("train.load_audio", fake_load_audio)

    _ensure_opus_decode_support(store, split="train")

    assert observed == [("sample.opus", None)]


def test_shard_records_for_rank_slices_in_memory_records() -> None:
    records = [
        AudioRecord("a.wav", None, "a", "utt0", 1),
        AudioRecord("b.wav", None, "b", "utt1", 1),
        AudioRecord("c.wav", None, "c", "utt2", 1),
        AudioRecord("d.wav", None, "d", "utt3", 1),
    ]

    shard = _shard_records_for_rank(records, rank=1, world_size=2)

    assert [record.utterance_id for record in shard] == ["utt1", "utt3"]


def test_disk_backed_record_store_shard_views_rows(tmp_path: Path) -> None:
    records_path = tmp_path / "train.jsonl"
    records_path.write_text(
        '{"audio_path":"a.wav","transcript":"a","utterance_id":"utt0","speaker_id":null,"has_speaker_id":false}\n'
        '{"audio_path":"b.wav","transcript":"b","utterance_id":"utt1","speaker_id":null,"has_speaker_id":false}\n'
        '{"audio_path":"c.wav","transcript":"c","utterance_id":"utt2","speaker_id":null,"has_speaker_id":false}\n'
        '{"audio_path":"d.wav","transcript":"d","utterance_id":"utt3","speaker_id":null,"has_speaker_id":false}\n',
        encoding="utf-8",
    )
    offsets = [0]
    content = records_path.read_text(encoding="utf-8")
    running = 0
    for line in content.splitlines(keepends=True)[:-1]:
        running += len(line)
        offsets.append(running)

    store = DiskBackedRecordStore(
        records_path,
        array.array("Q", offsets),
        array.array("I", [10, 20, 30, 40]),
        array.array("Q", [160, 320, 480, 640]),
        array.array("I", [16_000, 16_000, 16_000, 16_000]),
    )

    shard = store.shard(rank=1, world_size=2)

    assert len(shard) == 2
    assert [record.utterance_id for record in shard] == ["utt1", "utt3"]


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
    metadata = {
        "compatibility_signature": "legacy-test-signature",
        "items": [
            {"path": "ckpt_a.pt"},
            {"path": "ckpt_b.pt"},
        ],
    }
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
    metadata = {
        "items": [
            {"epoch": 7, "val_wer": 1.0, "path": "old.pt"},
        ]
    }
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
    assert saved_metadata["items"] == [
        {"epoch": 8, "val_wer": 0.9, "path": "checkpoint_epoch=0008_valwer=0.900000.pt"}
    ]
    assert isinstance(saved_metadata["compatibility_signature"], str)
    assert not (topk_dir / "old.pt").exists()


def test_iter_records_streams_split_selection(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\tid\tspeaker_id\tduration\n"
        "a.wav\tце тест\tutt0\tspk0\t0.3\n"
        "b.wav\tмовна модель\tutt1\tspk1\t0.3\n"
        "c.wav\tще приклад\tutt2\tspk2\t0.3\n",
        encoding="utf-8",
    )
    streamed = list(
        iter_records(
            dataset_root=tmp_path,
            split="train",
            seed=13,
            val_fraction=0.2,
            test_fraction=0.2,
            max_samples=2,
        )
    )
    loaded = load_records(
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


def test_iter_manifest_rows_from_source_reads_single_manifest_file(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\tduration\na.wav\tце тест\t0.3\nb.wav\tмовна модель\t0.4\n",
        encoding="utf-8",
    )

    rows = list(iter_manifest_rows_from_source(manifest))

    assert rows == [
        {"path": "a.wav", "sentence": "це тест", "duration": 0.3},
        {"path": "b.wav", "sentence": "мовна модель", "duration": 0.4},
    ]


def test_iter_manifest_rows_from_source_caches_remote_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"path\tsentence\tduration\na.wav\tce test\t0.3\n"
    calls: list[str] = []

    class _FakeResponse(BytesIO):
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.close()

    def fake_urlopen(request):
        calls.append(request.full_url)
        return _FakeResponse(payload)

    monkeypatch.setattr("squeezeformer_pytorch.data.urlopen", fake_urlopen)
    source_url = "https://example.com/train.tsv"

    first_rows = list(iter_manifest_rows_from_source(source_url, cache_dir=tmp_path))
    second_rows = list(iter_manifest_rows_from_source(source_url, cache_dir=tmp_path))

    assert first_rows == [{"path": "a.wav", "sentence": "ce test", "duration": 0.3}]
    assert second_rows == first_rows
    assert calls == [source_url]


def test_iter_records_from_source_matches_local_loader(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\tid\tspeaker_id\tduration\n"
        "a.wav\tце тест\tutt0\tspk0\t0.3\n"
        "b.wav\tмовна модель\tutt1\tspk1\t0.3\n"
        "c.wav\tще приклад\tutt2\tspk2\t0.3\n",
        encoding="utf-8",
    )

    from_source = list(
        iter_records_from_source(
            manifest,
            split="train",
            seed=13,
            val_fraction=0.2,
            test_fraction=0.2,
            max_samples=2,
        )
    )
    from_root = list(
        iter_records(
            dataset_root=tmp_path,
            split="train",
            seed=13,
            val_fraction=0.2,
            test_fraction=0.2,
            max_samples=2,
        )
    )

    assert [record.utterance_id for record in from_source] == [
        record.utterance_id for record in from_root
    ]


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


def test_iter_records_filters_alignment_outliers(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\tid\tduration\n"
        "a.wav\tце нормальний приклад\tutt0\t1.5\n"
        "b.wav\tдуже довгий текст для короткого аудіо\tutt1\t0.2\n"
        "c.wav\tтак\tutt2\t8.0\n",
        encoding="utf-8",
    )

    records = list(
        iter_records(
            dataset_root=tmp_path,
            split="train",
            seed=13,
            val_fraction=0.0,
            test_fraction=0.0,
            min_chars_per_second=2.0,
            max_chars_per_second=20.0,
            min_words_per_second=0.5,
            max_words_per_second=6.0,
            min_duration_per_char=0.03,
            max_duration_per_char=0.5,
            min_duration_per_word=0.2,
            max_duration_per_word=3.0,
        )
    )

    assert [record.utterance_id for record in records] == ["utt0"]


def test_max_frames_batch_sampler_respects_frame_budget() -> None:
    records = [
        AudioRecord(
            None, None, "a", "0", estimated_frames=20, speaker_id="s0", has_speaker_id=True
        ),
        AudioRecord(
            None, None, "b", "1", estimated_frames=25, speaker_id="s1", has_speaker_id=True
        ),
        AudioRecord(
            None, None, "c", "2", estimated_frames=40, speaker_id="s2", has_speaker_id=True
        ),
        AudioRecord(
            None, None, "d", "3", estimated_frames=45, speaker_id="s3", has_speaker_id=True
        ),
    ]
    sampler = MaxFramesBatchSampler(records, max_batch_frames=90, shuffle=False)
    batches = list(iter(sampler))
    assert batches
    for batch in batches:
        max_frames = max(records[index].estimated_frames for index in batch)
        assert len(batch) * max_frames <= 90


def test_distributed_batch_sampler_repartitions_batches_each_epoch() -> None:
    records = [
        AudioRecord(None, None, str(index), str(index), estimated_frames=30) for index in range(8)
    ]
    batch_sampler_rank0 = MaxFramesBatchSampler(
        records,
        max_batch_frames=30,
        shuffle=True,
        seed=7,
    )
    batch_sampler_rank1 = MaxFramesBatchSampler(
        records,
        max_batch_frames=30,
        shuffle=True,
        seed=7,
    )
    rank0 = DistributedBatchSampler(
        batch_sampler_rank0,
        rank=0,
        world_size=2,
    )
    rank1 = DistributedBatchSampler(
        batch_sampler_rank1,
        rank=1,
        world_size=2,
    )

    rank0.set_epoch(0)
    rank1.set_epoch(0)
    epoch0_rank0 = list(rank0)
    epoch0_rank1 = list(rank1)
    epoch0_combined = [
        batch for pair in zip(epoch0_rank0, epoch0_rank1, strict=False) for batch in pair if batch
    ]

    rank0.set_epoch(1)
    rank1.set_epoch(1)
    epoch1_rank0 = list(rank0)
    epoch1_rank1 = list(rank1)
    epoch1_combined = [
        batch for pair in zip(epoch1_rank0, epoch1_rank1, strict=False) for batch in pair if batch
    ]

    assert epoch0_combined != epoch1_combined
    flattened = sorted(index for batch in epoch0_rank0 + epoch0_rank1 for index in batch)
    assert flattened == list(range(len(records)))


def test_length_bucket_sampler_reuses_precomputed_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import squeezeformer_pytorch.data as data_module

    records = [
        AudioRecord(None, None, str(index), str(index), estimated_frames=10 + index)
        for index in range(20)
    ]
    sort_calls = 0
    original_sort = data_module._sorted_indices_with_progress

    def wrapped_sort(*args, **kwargs):
        nonlocal sort_calls
        sort_calls += 1
        return original_sort(*args, **kwargs)

    monkeypatch.setattr(data_module, "_sorted_indices_with_progress", wrapped_sort)
    sampler = data_module.LengthBucketBatchSampler(records, batch_size=2, shuffle=True, seed=7)

    sampler.set_epoch(0)
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)

    assert sort_calls == 1
    assert epoch0 != epoch1


def test_distributed_index_sampler_can_leave_validation_uneven() -> None:
    rank0 = DistributedIndexSampler(
        5,
        rank=0,
        world_size=2,
        shuffle=False,
        pad_to_world_size=False,
    )
    rank1 = DistributedIndexSampler(
        5,
        rank=1,
        world_size=2,
        shuffle=False,
        pad_to_world_size=False,
    )

    assert list(rank0) == [0, 2, 4]
    assert list(rank1) == [1, 3]


def test_duration_batch_sampler_respects_duration_budget() -> None:
    records = [
        AudioRecord(
            None,
            None,
            "a",
            "0",
            estimated_frames=20,
            speaker_id="s0",
            has_speaker_id=True,
            num_samples=16_000,
            sample_rate=16_000,
        ),
        AudioRecord(
            None,
            None,
            "b",
            "1",
            estimated_frames=25,
            speaker_id="s1",
            has_speaker_id=True,
            num_samples=24_000,
            sample_rate=16_000,
        ),
        AudioRecord(
            None,
            None,
            "c",
            "2",
            estimated_frames=40,
            speaker_id="s2",
            has_speaker_id=True,
            num_samples=32_000,
            sample_rate=16_000,
        ),
        AudioRecord(
            None,
            None,
            "d",
            "3",
            estimated_frames=45,
            speaker_id="s3",
            has_speaker_id=True,
            num_samples=48_000,
            sample_rate=16_000,
        ),
    ]
    sampler = DurationBatchSampler(records, max_batch_duration_sec=3.0, shuffle=False)
    batches = list(iter(sampler))
    assert batches
    for batch in batches:
        total_duration = sum(
            records[index].num_samples / records[index].sample_rate for index in batch
        )
        assert total_duration <= 3.0


def test_duration_batch_sampler_longest_first_orders_heaviest_batches_first() -> None:
    records = [
        AudioRecord(
            None,
            None,
            "a",
            "0",
            estimated_frames=10,
            speaker_id="s0",
            has_speaker_id=True,
            num_samples=16_000,
            sample_rate=16_000,
        ),
        AudioRecord(
            None,
            None,
            "b",
            "1",
            estimated_frames=20,
            speaker_id="s1",
            has_speaker_id=True,
            num_samples=16_000,
            sample_rate=16_000,
        ),
        AudioRecord(
            None,
            None,
            "c",
            "2",
            estimated_frames=30,
            speaker_id="s2",
            has_speaker_id=True,
            num_samples=48_000,
            sample_rate=16_000,
        ),
        AudioRecord(
            None,
            None,
            "d",
            "3",
            estimated_frames=40,
            speaker_id="s3",
            has_speaker_id=True,
            num_samples=48_000,
            sample_rate=16_000,
        ),
    ]
    sampler = DurationBatchSampler(
        records,
        max_batch_duration_sec=3.0,
        shuffle=False,
        longest_first=True,
    )

    batches = list(iter(sampler))

    observed = [
        sum(records[index].num_samples / records[index].sample_rate for index in batch)
        for batch in batches
    ]
    assert observed == sorted(observed, reverse=True)


def test_adaptive_batch_sampler_respects_token_budget() -> None:
    records = [
        AudioRecord(
            None,
            None,
            "aaa",
            "0",
            estimated_frames=20,
            speaker_id="s0",
            has_speaker_id=True,
        ),
        AudioRecord(
            None,
            None,
            "bbbb",
            "1",
            estimated_frames=25,
            speaker_id="s1",
            has_speaker_id=True,
        ),
        AudioRecord(
            None,
            None,
            "cc",
            "2",
            estimated_frames=40,
            speaker_id="s2",
            has_speaker_id=True,
        ),
        AudioRecord(
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


def test_adaptive_batch_sampler_uses_disk_backed_token_lengths_without_json(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "missing.jsonl"
    offsets = array.array("Q", [0, 0, 0])
    estimated_frames = array.array("I", [20, 25, 40])
    num_samples = array.array("Q", [16_000, 16_000, 16_000])
    sample_rates = array.array("I", [16_000, 16_000, 16_000])
    transcript_lengths = array.array("I", [100, 100, 100])
    token_lengths = array.array("I", [3, 4, 2])
    store = DiskBackedRecordStore(
        records_path,
        offsets,
        estimated_frames,
        num_samples,
        sample_rates,
        transcript_lengths,
        token_lengths,
    )

    sampler = AdaptiveBatchSampler(store, target_batch_units=6, unit="tokens", shuffle=False)
    batches = list(sampler)

    assert batches
    for batch in batches:
        assert sum(token_lengths[index] for index in batch) <= 6


def test_adaptive_batch_sampler_respects_frame_budget_and_keeps_all_indices() -> None:
    records = [
        AudioRecord(None, None, "a", "0", estimated_frames=20),
        AudioRecord(None, None, "bb", "1", estimated_frames=25),
        AudioRecord(None, None, "ccc", "2", estimated_frames=40),
        AudioRecord(None, None, "dddd", "3", estimated_frames=45),
        AudioRecord(None, None, "eeeee", "4", estimated_frames=70),
    ]
    sampler = AdaptiveBatchSampler(records, target_batch_units=90, unit="frames", shuffle=False)

    batches = list(iter(sampler))

    assert sorted(index for batch in batches for index in batch) == list(range(len(records)))
    for batch in batches:
        total_frames = sum(records[index].estimated_frames for index in batch)
        assert total_frames <= 90


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
    model = build_squeezeformer_encoder("sm")
    assert model.blocks[0].block_pattern == ("M", "s", "C", "s")


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

    dataset = ASRDataset(
        records=[AudioRecord("dummy.wav", None, "це тест", "utt0", estimated_frames=2)],
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
    assert list((tmp_path / "feature_shards").glob("features_*.sqlite3"))
    assert not list(tmp_path.glob("*.pt"))


def test_invalid_cached_features_are_recomputed(
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

    dataset = ASRDataset(
        records=[AudioRecord("dummy.wav", None, "це тест", "utt0", estimated_frames=2)],
        tokenizer=DummyTokenizer(),
        featurizer=AudioFeaturizer(),
        feature_cache_dir=tmp_path,
    )
    cache_path = dataset._feature_cache_path(dataset.records[0])
    assert cache_path is not None
    torch.save(torch.ones(25_000, dataset.featurizer.n_mels), cache_path)

    item = dataset[0]

    assert item is not None
    assert load_calls == 1
    assert item["features"].shape == (3, dataset.featurizer.n_mels)


def test_collate_asr_batch_filters_invalid_items() -> None:
    batch = collate_asr_batch(
        [
            None,
            {
                "features": torch.ones(4, 80),
                "feature_length": 4,
                "targets": torch.tensor([1, 2], dtype=torch.long),
                "target_length": 2,
                "transcript": "це тест",
                "utterance_id": "utt0",
                "speaker_id": None,
                "has_speaker_id": False,
            },
        ]
    )

    assert batch is not None
    assert batch["features"].shape == (1, 4, 80)
    assert batch["feature_lengths"].tolist() == [4]


def test_collate_asr_batch_includes_waveforms_when_present() -> None:
    batch = collate_asr_batch(
        [
            {
                "features": torch.ones(4, 80),
                "feature_length": 4,
                "targets": torch.tensor([1, 2], dtype=torch.long),
                "target_length": 2,
                "transcript": "це тест",
                "utterance_id": "utt0",
                "speaker_id": None,
                "has_speaker_id": False,
                "waveform": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
                "waveform_length": 3,
                "sample_rate": 16_000,
            },
            {
                "features": torch.ones(2, 80),
                "feature_length": 2,
                "targets": torch.tensor([3], dtype=torch.long),
                "target_length": 1,
                "transcript": "тест",
                "utterance_id": "utt1",
                "speaker_id": "spk1",
                "has_speaker_id": True,
                "waveform": torch.tensor([4.0, 5.0], dtype=torch.float32),
                "waveform_length": 2,
                "sample_rate": 16_000,
            },
        ]
    )

    assert batch is not None
    assert batch["waveforms"].shape == (2, 3)
    assert batch["waveform_lengths"].tolist() == [3, 2]
    assert batch["sample_rates"].tolist() == [16_000, 16_000]
    assert batch["waveforms"][1].tolist() == [4.0, 5.0, 0.0]
    assert batch["targets"].tolist() == [1, 2, 3]
    assert batch["target_lengths"].tolist() == [2, 1]
    assert batch["transcripts"] == ["це тест", "тест"]
    assert batch["utterance_ids"] == ["utt0", "utt1"]
    assert batch["speaker_ids"] == [None, "spk1"]
    assert batch["has_speaker_ids"] == [False, True]


def test_asr_dataset_returns_waveform_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [len(text)]

    def fake_load_audio(
        audio_path: str | None, audio_bytes: bytes | None
    ) -> tuple[torch.Tensor, int]:
        del audio_path, audio_bytes
        return torch.tensor([[1.0, 3.0, 5.0], [3.0, 5.0, 7.0]], dtype=torch.float32), 16_000

    monkeypatch.setattr("squeezeformer_pytorch.data.load_audio", fake_load_audio)

    dataset = ASRDataset(
        records=[AudioRecord("dummy.wav", None, "це тест", "utt0", estimated_frames=2)],
        tokenizer=DummyTokenizer(),
        featurizer=AudioFeaturizer(),
        return_waveforms=True,
    )

    item = dataset[0]

    assert item is not None
    assert torch.equal(item["waveform"], torch.tensor([2.0, 4.0, 6.0], dtype=torch.float32))
    assert item["waveform_length"] == 3
    assert item["sample_rate"] == 16_000


def test_aed_cross_entropy_loss_matches_transposed_reference() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([[1, 2, 0], [3, 0, 4]], dtype=torch.long)

    expected = torch.nn.functional.cross_entropy(
        logits.float().transpose(1, 2),
        targets,
        ignore_index=0,
        reduction="sum",
    ) / targets.ne(0).sum().clamp_min(1)

    observed = _aed_cross_entropy_loss(logits, targets, pad_id=0)

    torch.testing.assert_close(observed, expected)


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

    dataset = ASRDataset(
        records=[AudioRecord("dummy.wav", None, "це тест", "utt0", estimated_frames=2)],
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
        assert kwargs["in_order"] is True
    else:
        assert "multiprocessing_context" not in kwargs


def test_create_dataloader_can_disable_in_order_worker_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    dataset = ASRDataset(
        records=[AudioRecord("dummy.wav", None, "це тест", "utt0", estimated_frames=2)],
        tokenizer=DummyTokenizer(),
        featurizer=AudioFeaturizer(),
    )
    create_dataloader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        prefetch_factor=1,
        in_order=False,
    )

    assert captured["kwargs"]["in_order"] is False


def test_create_dataloader_uses_spawn_context_when_distributed_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)

    dataset = ASRDataset(
        records=[AudioRecord("dummy.wav", None, "це тест", "utt0", estimated_frames=2)],
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
        assert kwargs["multiprocessing_context"].get_start_method() == "spawn"
    else:
        assert "multiprocessing_context" not in kwargs


def test_create_dataloader_wraps_max_frames_sampler_for_distributed_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    dataset = ASRDataset(
        records=[
            AudioRecord("dummy.wav", None, f"utt-{index}", f"utt-{index}", estimated_frames=32)
            for index in range(4)
        ],
        tokenizer=DummyTokenizer(),
        featurizer=AudioFeaturizer(),
    )
    create_dataloader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        max_batch_frames=64,
        distributed=True,
        rank=0,
        world_size=2,
        seed=13,
        pad_distributed_batches=True,
    )

    kwargs = captured["kwargs"]
    assert isinstance(kwargs["batch_sampler"], DistributedBatchSampler)


def test_disk_backed_record_store_is_pickle_safe_after_open(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    payload = {
        "audio_path": "dummy.wav",
        "audio_blob_path": None,
        "transcript": "це тест",
        "utterance_id": "utt0",
        "speaker_id": None,
        "has_speaker_id": False,
    }
    records_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    offsets = array.array("Q", [0])
    estimated_frames = array.array("I", [2])
    num_samples = array.array("Q", [3200])
    sample_rates = array.array("I", [16_000])
    store = DiskBackedRecordStore(
        records_path, offsets, estimated_frames, num_samples, sample_rates
    )

    # Force the store to open its underlying file handle before pickling.
    record = store[0]
    assert record.transcript == "це тест"

    restored = pickle.loads(pickle.dumps(store))

    assert len(restored) == 1
    assert restored[0].utterance_id == "utt0"


def test_disk_backed_record_store_recomputes_centered_frame_estimate(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    payload = {
        "audio_path": "dummy.wav",
        "audio_blob_path": None,
        "transcript": "це тест",
        "utterance_id": "utt0",
        "speaker_id": None,
        "has_speaker_id": False,
    }
    records_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    offsets = array.array("Q", [0])
    estimated_frames = array.array("I", [20])
    num_samples = array.array("Q", [3200])
    sample_rates = array.array("I", [16_000])
    store = DiskBackedRecordStore(
        records_path, offsets, estimated_frames, num_samples, sample_rates
    )
    featurizer = AudioFeaturizer(**zipformer_paper_featurizer_config())

    store.populate_metadata(
        hop_length=featurizer.hop_length,
        num_workers=1,
        featurizer=featurizer,
    )

    assert store[0].estimated_frames == 21


def test_disk_backed_record_store_force_probe_overwrites_manifest_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records_path = tmp_path / "records.jsonl"
    payload = {
        "audio_path": "dummy.wav",
        "audio_blob_path": None,
        "transcript": "це тест",
        "utterance_id": "utt0",
        "speaker_id": None,
        "has_speaker_id": False,
    }
    records_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    offsets = array.array("Q", [0])
    estimated_frames = array.array("I", [303])
    num_samples = array.array("Q", [48_384])
    sample_rates = array.array("I", [16_000])
    store = DiskBackedRecordStore(
        records_path, offsets, estimated_frames, num_samples, sample_rates
    )
    featurizer = AudioFeaturizer(**zipformer_paper_featurizer_config())

    monkeypatch.setattr(
        "squeezeformer_pytorch.training.data_loading.probe_audio_metadata",
        lambda _audio_path, _audio_bytes: (145_920, 48_000),
    )

    store.populate_metadata(
        hop_length=featurizer.hop_length,
        num_workers=1,
        featurizer=featurizer,
        force_audio_metadata_probe=True,
    )

    record = store[0]
    assert record.num_samples == 145_920
    assert record.sample_rate == 48_000
    assert record.estimated_frames == 305


def test_muon_optimizer_partition_uses_encoder_hidden_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMuon(torch.optim.Optimizer):
        def __init__(
            self,
            params: object,
            lr: float,
            weight_decay: float,
            adjust_lr_fn: str,
        ) -> None:
            defaults = {
                "lr": lr,
                "weight_decay": weight_decay,
                "adjust_lr_fn": adjust_lr_fn,
            }
            super().__init__(params, defaults)

        def step(self, closure=None):  # type: ignore[no-untyped-def]
            return None

    import train

    monkeypatch.setattr(train, "ExternalMuon", FakeMuon)
    monkeypatch.setattr(train.torch.optim, "Muon", None, raising=False)

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
