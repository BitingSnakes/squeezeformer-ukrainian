from __future__ import annotations

import torch

from zipformer_pytorch.asr import (
    ZipformerConfig,
    ZipformerCTC,
    ZipformerEncoder,
    zipformer_variant,
)
from zipformer_pytorch.zipformer.zipformer import (
    ActivationBalancer,
    DownsampledZipformerStack,
    PairwiseDownsample,
    PairwiseUpsample,
    Whiten,
)


def _tiny_zipformer_config() -> ZipformerConfig:
    return ZipformerConfig(
        input_dim=8,
        output_downsampling_factor=2,
        downsampling_factor=(1, 2),
        encoder_dim=(16, 24),
        num_encoder_layers=(1, 1),
        num_heads=(4, 4),
        query_head_dim=(8,),
        value_head_dim=(4,),
        pos_head_dim=(4,),
        feedforward_dim=(32, 48),
        cnn_module_kernel=(5, 5),
        pos_dim=16,
        dropout=0.0,
    )


def test_zipformer_ctc_returns_training_outputs_with_main_log_probs() -> None:
    model = ZipformerCTC(
        encoder_config=_tiny_zipformer_config(),
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

    assert outputs["encoded"].shape == (2, 3, 24)
    assert outputs["output_lengths"].tolist() == [3, 3]
    assert outputs["main_log_probs"].shape == (2, 3, 6)
    assert outputs["main_ctc_loss"] is not None
    assert outputs["intermediate_ctc_losses"] == {}
    assert outputs["blank_logit_regularization_loss"].dtype == torch.float32


def test_zipformer_ctc_forward_runs_on_meta_device() -> None:
    model = ZipformerCTC(
        encoder_config=_tiny_zipformer_config(),
        vocab_size=6,
    ).to("meta")
    features = torch.randn(2, 12, 8, device="meta")
    feature_lengths = torch.tensor([12, 9], dtype=torch.long, device="meta")

    logits, output_lengths = model(features, feature_lengths)

    assert logits.device.type == "meta"
    assert output_lengths.device.type == "meta"


def test_zipformer_encoder_masks_padding_inside_blocks() -> None:
    torch.manual_seed(0)
    encoder = ZipformerEncoder(_tiny_zipformer_config()).eval()
    valid_length = torch.tensor([7], dtype=torch.long)
    base = torch.randn(1, 7, 8)
    short = torch.cat([base, torch.zeros(1, 2, 8)], dim=1)
    long = torch.cat([base, torch.zeros(1, 8, 8)], dim=1)

    with torch.no_grad():
        new_short, _ = encoder(short, valid_length)
        new_long, _ = encoder(long, valid_length)

    new_diff = (new_short[:, :2] - new_long[:, :2]).abs().max().item()

    assert new_diff < 0.01


def test_zipformer_variants_match_paper_scale_ladder() -> None:
    s = zipformer_variant("s")
    m = zipformer_variant("m")
    l = zipformer_variant("l")

    assert s.encoder_dim == (192, 256, 256, 256, 256, 256)
    assert s.num_encoder_layers == (2, 2, 2, 2, 2, 2)
    assert s.feedforward_dim == (512, 768, 768, 768, 768, 768)

    assert m.encoder_dim == (192, 256, 384, 512, 384, 256)
    assert m.num_encoder_layers == (2, 2, 3, 4, 3, 2)
    assert m.feedforward_dim == (512, 768, 1024, 1536, 1024, 768)

    assert l.encoder_dim == (192, 256, 512, 768, 512, 256)
    assert l.num_encoder_layers == (2, 2, 4, 5, 4, 2)
    assert l.feedforward_dim == (512, 768, 1536, 2048, 1536, 768)

    assert zipformer_variant("sm") == m
    assert zipformer_variant("ml") == l


def test_zipformer_uses_pairwise_resampling_for_reduced_rate_stacks() -> None:
    config = ZipformerConfig(
        input_dim=8,
        output_downsampling_factor=2,
        downsampling_factor=(1, 4, 8),
        encoder_dim=(16, 24, 32),
        num_encoder_layers=(1, 1, 1),
        num_heads=(4, 4, 4),
        query_head_dim=(8,),
        value_head_dim=(4,),
        pos_head_dim=(4,),
        feedforward_dim=(32, 48, 64),
        cnn_module_kernel=(5, 5, 5),
        pos_dim=16,
        dropout=0.0,
    )
    encoder = ZipformerEncoder(config)
    stack4 = encoder.encoder.stacks[1]
    stack8 = encoder.encoder.stacks[2]

    assert isinstance(stack4, DownsampledZipformerStack)
    assert isinstance(stack4.downsample, PairwiseDownsample)
    assert isinstance(stack4.upsample, PairwiseUpsample)
    assert len(stack4.downsample.stages) == 2
    assert len(stack4.upsample.stages) == 2

    assert isinstance(stack8, DownsampledZipformerStack)
    assert len(stack8.downsample.stages) == 3
    assert len(stack8.upsample.stages) == 3


def test_zipformer_blocks_include_balancers_and_whiteners() -> None:
    encoder = ZipformerEncoder(_tiny_zipformer_config())
    conv_embed = encoder.encoder.conv_embed
    stack = encoder.encoder.stacks[0]
    block = stack.blocks[0]

    assert isinstance(conv_embed.conv1_balancer, ActivationBalancer)
    assert isinstance(conv_embed.conv2_balancer, ActivationBalancer)
    assert isinstance(conv_embed.conv3_balancer, ActivationBalancer)
    assert isinstance(block.feed_forward1.balancer, ActivationBalancer)
    assert isinstance(block.attention_weights.key_whiten, Whiten)
    assert isinstance(block.self_attention1.value_whiten, Whiten)
    assert isinstance(block.conv1.input_balancer, ActivationBalancer)
    assert isinstance(block.conv1.depthwise_balancer, ActivationBalancer)
    assert isinstance(block.block_balancer, ActivationBalancer)
    assert isinstance(block.whiten, Whiten)
