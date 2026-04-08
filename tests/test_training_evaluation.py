from __future__ import annotations

import sys
import types

import torch

sys.modules.setdefault(
    "trackio",
    types.SimpleNamespace(init=lambda **_kwargs: None, log=lambda *_args, **_kwargs: None),
)

from squeezeformer_pytorch.runtime_types import DTypeChoice
from squeezeformer_pytorch.training import evaluation as training_evaluation


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
