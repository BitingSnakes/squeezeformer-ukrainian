from __future__ import annotations

import importlib
import sys
import types

import torch

sys.modules.setdefault(
    "trackio",
    types.SimpleNamespace(init=lambda **_kwargs: None, log=lambda *_args, **_kwargs: None),
)

CharacterTokenizer = importlib.import_module("squeezeformer_pytorch.asr").CharacterTokenizer
DTypeChoice = importlib.import_module("squeezeformer_pytorch.runtime_types").DTypeChoice
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
    monkeypatch.setattr(training_evaluation, "word_error_rate", lambda refs, hyps: float(len(hyps)) / 10.0)
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
            [{"utterance_id": utterance_ids[0], "speaker_id": "", "reference": refs[0], "hypothesis": hyps[0]}],
            [{"utterance_id": utterance_ids[-1], "speaker_id": "", "reference": refs[-1], "hypothesis": hyps[-1]}],
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
                "references": ["a"],
                "hypotheses": ["b"],
                "utterance_ids": ["utt-1"],
                "speaker_ids": ["spk-1"],
                "has_speaker_ids": [True],
            },
            {
                "total_loss": 6.0,
                "total_main_ctc_loss": 8.0,
                "total_intermediate_ctc_loss": 10.0,
                "total_combined_ctc_loss": 12.0,
                "total_aed_loss": 14.0,
                "total_liberta_distill_loss": 16.0,
                "total_audio_teacher_loss": 18.0,
                "total_batches": 2,
                "references": ["c"],
                "hypotheses": ["d"],
                "utterance_ids": ["utt-2"],
                "speaker_ids": [None],
                "has_speaker_ids": [False],
            },
        ],
        example_limit=1,
    )

    assert merged["metrics"]["loss"] == 2.0
    assert merged["metrics"]["main_ctc_loss"] == 3.0
    assert merged["metrics"]["intermediate_ctc_loss"] == 4.0
    assert merged["metrics"]["combined_ctc_loss"] == 5.0
    assert merged["metrics"]["aed_loss"] == 6.0
    assert merged["metrics"]["liberta_distill_loss"] == 7.0
    assert merged["metrics"]["audio_teacher_loss"] == 8.0
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
