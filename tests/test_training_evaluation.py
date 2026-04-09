from __future__ import annotations

import importlib
import logging
import sys
import types
from argparse import Namespace
from pathlib import Path

import torch

sys.modules.setdefault(
    "trackio",
    types.SimpleNamespace(
        init=lambda **_kwargs: None,
        log=lambda *_args, **_kwargs: None,
        Table=type("Table", (), {}),
    ),
)

CharacterTokenizer = importlib.import_module("squeezeformer_pytorch.asr").CharacterTokenizer
DTypeChoice = importlib.import_module("squeezeformer_pytorch.runtime_types").DTypeChoice
ValidationModelSource = importlib.import_module(
    "squeezeformer_pytorch.runtime_types"
).ValidationModelSource
squeezeformer_variant = importlib.import_module("squeezeformer_pytorch.model").squeezeformer_variant
training_evaluation = importlib.import_module("squeezeformer_pytorch.training.evaluation")


def test_greedy_decode_ignores_padded_frames() -> None:
    tokenizer = CharacterTokenizer(symbols=["a", "b"])
    log_probs = torch.log(
        torch.tensor(
            [
                [
                    [0.01, 0.98, 0.01],
                    [0.01, 0.98, 0.01],
                    [0.01, 0.01, 0.98],
                ]
            ],
            dtype=torch.float32,
        )
    )

    decoded = training_evaluation.greedy_decode(
        log_probs,
        output_lengths=torch.tensor([2]),
        tokenizer=tokenizer,
    )

    assert decoded == ["a"]


def test_beam_decode_ignores_padded_frames() -> None:
    tokenizer = CharacterTokenizer(symbols=["a", "b"])
    log_probs = torch.log(
        torch.tensor(
            [
                [
                    [0.01, 0.98, 0.01],
                    [0.01, 0.98, 0.01],
                    [0.01, 0.01, 0.98],
                ]
            ],
            dtype=torch.float32,
        )
    )

    decoded = training_evaluation.decode_batch(
        log_probs,
        output_lengths=torch.tensor([2]),
        tokenizer=tokenizer,
        strategy=training_evaluation.DecodeStrategy.BEAM,
        beam_size=2,
    )

    assert decoded == ["a"]


def test_beam_decode_length_bonus_can_prefer_nonempty_prefix() -> None:
    tokenizer = CharacterTokenizer(symbols=["a"])
    log_probs = torch.log(
        torch.tensor(
            [
                [
                    [0.90, 0.10],
                    [0.90, 0.10],
                ]
            ],
            dtype=torch.float32,
        )
    )

    decoded_without_bonus = training_evaluation.decode_batch(
        log_probs,
        output_lengths=torch.tensor([2]),
        tokenizer=tokenizer,
        strategy=training_evaluation.DecodeStrategy.BEAM,
        beam_size=2,
        beam_length_bonus=0.0,
    )
    decoded_with_bonus = training_evaluation.decode_batch(
        log_probs,
        output_lengths=torch.tensor([2]),
        tokenizer=tokenizer,
        strategy=training_evaluation.DecodeStrategy.BEAM,
        beam_size=2,
        beam_length_bonus=2.0,
    )

    assert decoded_without_bonus == [""]
    assert decoded_with_bonus == ["a"]


def test_ctc_batch_diagnostics_capture_blank_and_nonblank_signal() -> None:
    tokenizer = CharacterTokenizer(symbols=["a", "b"])
    log_probs = torch.log(
        torch.tensor(
            [
                [
                    [0.80, 0.15, 0.05],
                    [0.10, 0.70, 0.20],
                    [0.85, 0.10, 0.05],
                ]
            ],
            dtype=torch.float32,
        )
    )

    diagnostics = training_evaluation.summarize_ctc_batch_diagnostics(
        training_evaluation.ctc_batch_diagnostics(
            log_probs,
            output_lengths=torch.tensor([2]),
            tokenizer=tokenizer,
            target_lengths=torch.tensor([1]),
        )
    )

    assert diagnostics["argmax_blank_fraction"] == 0.5
    assert diagnostics["avg_output_frames"] == 2.0
    assert diagnostics["avg_target_tokens"] == 1.0
    assert diagnostics["target_tokens_per_frame"] == 0.5
    assert diagnostics["impossible_sample_fraction"] == 0.0
    assert diagnostics["tight_sample_fraction"] == 0.0
    assert diagnostics["avg_blank_probability"] > diagnostics["avg_top_nonblank_probability"]


def test_ctc_batch_diagnostics_capture_impossible_and_tight_samples() -> None:
    tokenizer = CharacterTokenizer(symbols=["a"])
    log_probs = torch.log(
        torch.tensor(
            [
                [[0.8, 0.2], [0.8, 0.2]],
                [[0.8, 0.2], [0.8, 0.2]],
            ],
            dtype=torch.float32,
        )
    )

    diagnostics = training_evaluation.summarize_ctc_batch_diagnostics(
        training_evaluation.ctc_batch_diagnostics(
            log_probs,
            output_lengths=torch.tensor([1, 2]),
            tokenizer=tokenizer,
            target_lengths=torch.tensor([2, 2]),
        )
    )

    assert diagnostics["impossible_sample_fraction"] == 0.5
    assert diagnostics["tight_sample_fraction"] == 1.0


def test_ctc_logit_diagnostics_capture_margin_and_entropy_signal() -> None:
    tokenizer = CharacterTokenizer(symbols=["a", "b"])
    logits = torch.tensor(
        [
            [
                [3.0, 1.0, 0.0],
                [0.5, 2.0, 1.0],
                [9.0, 9.0, 9.0],
            ]
        ],
        dtype=torch.float32,
    )

    diagnostics = training_evaluation.summarize_ctc_logit_diagnostics(
        training_evaluation.ctc_logit_diagnostics(
            logits,
            output_lengths=torch.tensor([2]),
            tokenizer=tokenizer,
        )
    )

    assert diagnostics["avg_top_logit"] == 2.5
    assert diagnostics["avg_top2_margin"] == 1.5
    assert diagnostics["avg_blank_logit"] == 1.75
    assert diagnostics["avg_blank_nonblank_margin"] == 0.25
    assert diagnostics["avg_entropy"] > 0.0


def test_encoder_output_diagnostics_capture_mean_std_and_norm() -> None:
    encoded = torch.tensor(
        [
            [
                [1.0, -1.0],
                [3.0, 1.0],
                [100.0, 100.0],
            ]
        ],
        dtype=torch.float32,
    )

    diagnostics = training_evaluation.summarize_encoder_output_diagnostics(
        training_evaluation.encoder_output_diagnostics(
            encoded,
            output_lengths=torch.tensor([2]),
        )
    )

    assert diagnostics["avg_mean"] == 1.0
    assert abs(diagnostics["avg_std"] - torch.tensor(2.0**0.5).item()) < 1e-6
    assert diagnostics["avg_token_l2_norm"] > 2.2


def test_top_emitted_token_histogram_reports_dominant_tokens() -> None:
    tokenizer = CharacterTokenizer(symbols=["a", "b"])
    log_probs = torch.log(
        torch.tensor(
            [
                [
                    [0.05, 0.90, 0.05],
                    [0.05, 0.90, 0.05],
                    [0.05, 0.05, 0.90],
                    [0.90, 0.05, 0.05],
                ]
            ],
            dtype=torch.float32,
        )
    )

    histogram = training_evaluation.top_emitted_token_histogram(
        log_probs,
        output_lengths=torch.tensor([3]),
        tokenizer=tokenizer,
        top_k=3,
    )

    assert histogram[0][2] == "a"
    assert abs(histogram[0][1] - (2.0 / 3.0)) < 1e-6
    assert histogram[1][2] == "b"


def test_top_emitted_token_histogram_uses_sentencepiece_piece_when_decode_is_empty() -> None:
    class DummyProcessor:
        def decode(self, ids):
            if list(ids) == [2]:
                return ""
            return "x"

        def id_to_piece(self, token_id):
            if token_id == 2:
                return "▁"
            return "x"

    class DummyTokenizer:
        blank_id = 0
        processor = DummyProcessor()

        def decode(self, token_ids):
            return self.processor.decode(token_ids)

    log_probs = torch.log(
        torch.tensor(
            [
                [
                    [0.01, 0.01, 0.98],
                    [0.01, 0.01, 0.98],
                ]
            ],
            dtype=torch.float32,
        )
    )

    histogram = training_evaluation.top_emitted_token_histogram(
        log_probs,
        output_lengths=torch.tensor([2]),
        tokenizer=DummyTokenizer(),
        top_k=1,
    )

    assert histogram[0][2] == "▁"


def test_evaluate_restores_model_mode(monkeypatch) -> None:
    class DummyTokenizer:
        blank_id = 0

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.aed_decoder = None
            self.intermediate_ctc_layers = ()

        def forward(self, *_args, **_kwargs):
            return {
                "encoded": torch.zeros(1, 2, 4),
                "output_lengths": torch.tensor([2]),
                "main_log_probs": torch.zeros(1, 2, 2),
                "main_ctc_loss": torch.tensor(1.25),
                "intermediate_ctc_losses": {},
            }

    monkeypatch.setattr(training_evaluation, "decode_batch", lambda *_args, **_kwargs: ["test"])
    monkeypatch.setattr(training_evaluation, "char_error_rate", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(training_evaluation, "word_error_rate", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(training_evaluation, "length_bucket_metrics", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        training_evaluation,
        "speaker_level_metrics",
        lambda *_args, **_kwargs: {
            "speaker_count": 0.0,
            "speaker_macro_wer": 0.0,
            "speaker_id_available": 0.0,
            "missing_speaker_id_samples": 0.0,
        },
    )
    monkeypatch.setattr(training_evaluation, "collect_examples", lambda *_args, **_kwargs: ([], []))

    batch = {
        "features": torch.zeros(1, 4, 3),
        "feature_lengths": torch.tensor([4]),
        "targets": torch.tensor([[1, 1]]),
        "target_lengths": torch.tensor([2]),
        "transcripts": ["test"],
        "utterance_ids": ["utt-1"],
        "speaker_ids": [None],
        "has_speaker_ids": [False],
    }
    model = DummyModel()
    model.train()

    training_evaluation.evaluate(
        model=model,
        dataloader=[batch],
        criterion=torch.nn.CTCLoss(),
        tokenizer=DummyTokenizer(),
        device=torch.device("cpu"),
        dtype=DTypeChoice.FLOAT32,
    )

    assert model.training is True


def test_merge_evaluation_shards_combines_all_examples(monkeypatch) -> None:
    monkeypatch.setattr(training_evaluation, "char_error_rate", lambda refs, hyps: float(len(refs)))
    monkeypatch.setattr(
        training_evaluation, "word_error_rate", lambda refs, hyps: float(len(hyps)) / 10.0
    )
    monkeypatch.setattr(
        training_evaluation,
        "length_bucket_metrics",
        lambda refs, hyps: {"samples_total": float(len(refs))},
    )
    monkeypatch.setattr(
        training_evaluation,
        "speaker_level_metrics",
        lambda speaker_ids, has_speaker_ids, refs, hyps: {
            "speaker_count": len([speaker_id for speaker_id in speaker_ids if speaker_id]),
            "speaker_macro_wer": 0.25,
            "speaker_id_available": False,
            "missing_speaker_id_samples": len([flag for flag in has_speaker_ids if not flag]),
            "per_speaker": {},
        },
    )
    monkeypatch.setattr(
        training_evaluation,
        "collect_examples",
        lambda utterance_ids, speaker_ids, refs, hyps, limit: (
            [
                {
                    "utterance_id": utterance_ids[0],
                    "speaker_id": "",
                    "reference": refs[0],
                    "hypothesis": hyps[0],
                }
            ],
            [
                {
                    "utterance_id": utterance_ids[-1],
                    "speaker_id": "",
                    "reference": refs[-1],
                    "hypothesis": hyps[-1],
                }
            ],
        ),
    )

    merged = training_evaluation._merge_evaluation_shards(
        [
            {
                "total_loss": 2.0,
                "total_main_ctc_loss": 4.0,
                "total_intermediate_ctc_loss": 6.0,
                "total_combined_ctc_loss": 8.0,
                "total_aed_loss": 10.0,
                "total_liberta_distill_loss": 12.0,
                "total_audio_teacher_loss": 14.0,
                "total_batches": 2,
                "total_samples": 1.0,
                "references": ["a"],
                "hypotheses": ["b"],
                "utterance_ids": ["utt-1"],
                "speaker_ids": ["spk-1"],
                "has_speaker_ids": [True],
            },
            {
                "total_loss": 18.0,
                "total_main_ctc_loss": 24.0,
                "total_intermediate_ctc_loss": 30.0,
                "total_combined_ctc_loss": 36.0,
                "total_aed_loss": 42.0,
                "total_liberta_distill_loss": 48.0,
                "total_audio_teacher_loss": 54.0,
                "total_batches": 2,
                "total_samples": 3.0,
                "references": ["c"],
                "hypotheses": ["d"],
                "utterance_ids": ["utt-2"],
                "speaker_ids": [None],
                "has_speaker_ids": [False],
            },
        ],
        example_limit=1,
    )

    assert merged["metrics"]["loss"] == 5.0
    assert merged["metrics"]["main_ctc_loss"] == 7.0
    assert merged["metrics"]["intermediate_ctc_loss"] == 9.0
    assert merged["metrics"]["combined_ctc_loss"] == 11.0
    assert merged["metrics"]["aed_loss"] == 13.0
    assert merged["metrics"]["liberta_distill_loss"] == 15.0
    assert merged["metrics"]["audio_teacher_loss"] == 17.0
    assert merged["metrics"]["cer"] == 2.0
    assert merged["metrics"]["wer"] == 0.2
    assert merged["metrics"]["avg_blank_probability"] == 0.0
    assert merged["metrics"]["decoded_empty_fraction"] == 0.0
    assert merged["metrics"]["decoded_avg_char_length"] == 1.0
    assert merged["metrics"]["decoded_avg_word_length"] == 1.0
    assert merged["metrics"]["samples_total"] == 2.0
    assert merged["metrics"]["speaker_count"] == 1.0
    assert merged["metrics"]["speaker_macro_wer"] == 0.25
    assert merged["metrics"]["speaker_id_available"] == 0.0
    assert merged["metrics"]["missing_speaker_id_samples"] == 1.0
    assert merged["hardest_examples"][0]["utterance_id"] == "utt-1"
    assert merged["random_examples"][0]["utterance_id"] == "utt-2"


def test_evaluate_and_checkpoint_saves_validated_ema_weights_and_resume_raw(
    monkeypatch, tmp_path: Path
) -> None:
    class DummyTokenizer:
        blank_id = 0

        def to_dict(self) -> dict[str, object]:
            return {"type": "character", "symbols": ["a"]}

    class DummyFeaturizer:
        def config_dict(self) -> dict[str, object]:
            return {"sample_rate": 16_000}

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ema = training_evaluation.ExponentialMovingAverage(model, decay=0.9)
    ema.shadow["weight"] = torch.full_like(ema.shadow["weight"], 2.0)

    captured_checkpoints: list[dict[str, object]] = []
    monkeypatch.setattr(
        training_evaluation,
        "evaluate",
        lambda *args, **kwargs: {
            "metrics": {
                "loss": 1.0,
                "main_ctc_loss": 1.0,
                "intermediate_ctc_loss": 0.0,
                "combined_ctc_loss": 1.0,
                "aed_loss": 0.0,
                "liberta_distill_loss": 0.0,
                "audio_teacher_loss": 0.0,
                "cer": 0.5,
                "wer": 0.5,
                "avg_blank_probability": 0.1,
                "decoded_empty_fraction": 0.0,
                "decoded_avg_char_length": 4.0,
                "decoded_avg_word_length": 1.0,
            },
            "timings": {},
            "hardest_examples": [],
            "random_examples": [],
            "speaker_metrics": {},
        },
    )
    monkeypatch.setattr(training_evaluation.trackio, "log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        training_evaluation,
        "save_checkpoint",
        lambda checkpoint, _path: captured_checkpoints.append(
            {
                key: (
                    {
                        nested_key: nested_value.detach().clone()
                        if isinstance(nested_value, torch.Tensor)
                        else nested_value
                        for nested_key, nested_value in value.items()
                    }
                    if isinstance(value, dict)
                    else value
                )
                for key, value in checkpoint.items()
            }
        ),
    )
    monkeypatch.setattr(
        training_evaluation,
        "_export_inference_checkpoint",
        lambda _checkpoint, path: path.with_suffix(".safetensors"),
    )
    monkeypatch.setattr(training_evaluation, "_update_top_checkpoints", lambda **_kwargs: None)
    monkeypatch.setattr(training_evaluation, "_average_topk_checkpoints", lambda _output_dir: None)

    best_val_wer = training_evaluation._evaluate_and_checkpoint(
        model=model,
        val_loader=[],
        criterion=torch.nn.CTCLoss(blank=0, zero_infinity=True),
        tokenizer=DummyTokenizer(),
        device=torch.device("cpu"),
        dtype=DTypeChoice.FLOAT32,
        fp8_recipe=None,
        decode_strategy=training_evaluation.DecodeStrategy.GREEDY,
        beam_size=1,
        lm_scorer=None,
        lm_weight=0.0,
        beam_length_bonus=0.1,
        example_limit=1,
        intermediate_ctc_weight=0.0,
        aed_loss_weight=0.0,
        liberta_teacher=None,
        liberta_distill_weight=0.0,
        audio_teacher=None,
        audio_teacher_weight=0.0,
        audio_teacher_objective="hidden_mse",
        ema=ema,
        validation_model_source=ValidationModelSource.EMA,
        train_metrics={"train_loss": 2.0},
        epoch=1,
        global_step=10,
        output_dir=tmp_path,
        encoder_config=squeezeformer_variant("xs"),
        featurizer=DummyFeaturizer(),
        optimizers=[optimizer],
        optimizer_names=["adamw"],
        schedulers=[scheduler],
        scaler=scaler,
        args=Namespace(keep_top_k=1, validation_model_source=ValidationModelSource.EMA),
        best_val_wer=float("inf"),
        split_audit={},
        logger=logging.getLogger("test"),
        save_last_checkpoint=False,
        report_stem="epoch_0001_step_00000010",
        distributed=False,
        is_main_process=True,
    )

    assert best_val_wer == 0.5
    assert captured_checkpoints
    saved_checkpoint = captured_checkpoints[0]
    assert float(saved_checkpoint["model_state_dict"]["weight"].item()) == 2.0
    assert float(saved_checkpoint["resume_model_state_dict"]["weight"].item()) == 1.0
    assert saved_checkpoint["validation_model_source"] == ValidationModelSource.EMA
    assert float(model.weight.item()) == 1.0
