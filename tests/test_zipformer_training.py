from __future__ import annotations

import torch

from zipformer_pytorch.asr import (
    ZipformerConfig,
    ZipformerCTC,
    ZipformerEncoder,
    _make_padding_mask,
)


def _run_zipformer_encoder_without_block_masks(
    encoder: ZipformerEncoder,
    features: torch.Tensor,
    feature_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_lengths = feature_lengths.to(dtype=torch.int64).clamp_(1, features.size(1))
    padding_mask = _make_padding_mask(output_lengths, max_length=features.size(1)).unsqueeze(-1)

    x = encoder.input_projection(encoder.input_norm(features))
    x = encoder.input_dropout(x)
    x = x.masked_fill(~padding_mask, 0.0)
    for block in encoder.blocks:
        x = block(x, mask=None)
        x = x.masked_fill(~padding_mask, 0.0)
    x = encoder.output_norm(x)
    x = x.masked_fill(~padding_mask, 0.0)
    return x, output_lengths


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


def test_zipformer_ctc_forward_runs_on_meta_device() -> None:
    model = ZipformerCTC(
        encoder_config=ZipformerConfig(input_dim=8, model_dim=16, num_layers=2, num_heads=4),
        vocab_size=6,
    ).to("meta")
    features = torch.randn(2, 12, 8, device="meta")
    feature_lengths = torch.tensor([12, 9], dtype=torch.long, device="meta")

    logits, output_lengths = model(features, feature_lengths)

    assert logits.device.type == "meta"
    assert output_lengths.device.type == "meta"


def test_zipformer_encoder_masks_padding_inside_blocks() -> None:
    torch.manual_seed(0)
    encoder = ZipformerEncoder(
        ZipformerConfig(input_dim=8, model_dim=16, num_layers=2, num_heads=4)
    ).eval()
    valid_length = torch.tensor([7], dtype=torch.long)
    base = torch.randn(1, 7, 8)
    short = torch.cat([base, torch.zeros(1, 2, 8)], dim=1)
    long = torch.cat([base, torch.zeros(1, 8, 8)], dim=1)

    with torch.no_grad():
        old_short, _ = _run_zipformer_encoder_without_block_masks(encoder, short, valid_length)
        old_long, _ = _run_zipformer_encoder_without_block_masks(encoder, long, valid_length)
        new_short, _ = encoder(short, valid_length)
        new_long, _ = encoder(long, valid_length)

    old_diff = (old_short[:, :7] - old_long[:, :7]).abs().max().item()
    new_diff = (new_short[:, :7] - new_long[:, :7]).abs().max().item()

    assert old_diff > 0.05
    assert new_diff < 0.01
    assert new_diff < old_diff * 0.1
