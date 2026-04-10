from __future__ import annotations

import torch

from zipformer_pytorch.asr import ZipformerCTC, ZipformerConfig


def test_zipformer_ctc_returns_training_outputs_with_main_log_probs() -> None:
    model = ZipformerCTC(
        encoder_config=ZipformerConfig(input_dim=8, model_dim=16, num_layers=2, num_heads=4),
        vocab_size=6,
    )
    features = torch.randn(2, 12, 8)
    feature_lengths = torch.tensor([12, 9], dtype=torch.long)
    targets = torch.tensor([1, 2, 1, 2, 3], dtype=torch.long)
    target_lengths = torch.tensor([3, 2], dtype=torch.long)

    outputs = model(
        features,
        feature_lengths,
        return_training_outputs=True,
        targets=targets,
        target_lengths=target_lengths,
        blank_id=0,
        return_main_log_probs=True,
    )

    assert outputs["encoded"].shape == (2, 12, 16)
    assert outputs["output_lengths"].tolist() == [12, 9]
    assert outputs["main_log_probs"].shape == (2, 12, 6)
    assert outputs["main_ctc_loss"] is not None
    assert outputs["intermediate_ctc_losses"] == {}
    assert outputs["blank_logit_regularization_loss"].dtype == torch.float32
