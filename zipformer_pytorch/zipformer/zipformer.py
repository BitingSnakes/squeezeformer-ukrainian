from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _make_padding_mask(lengths: Tensor, *, max_length: int) -> Tensor:
    return torch.arange(max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def _ceil_divide(lengths: Tensor, factor: int) -> Tensor:
    return (lengths + factor - 1) // factor


def _conv_out_length(lengths: Tensor, *, kernel_size: int, stride: int, padding: int) -> Tensor:
    return ((lengths + 2 * padding - kernel_size) // stride) + 1


def _convert_num_channels(x: Tensor, num_channels: int) -> Tensor:
    current_channels = x.size(-1)
    if current_channels == num_channels:
        return x
    if current_channels > num_channels:
        return x[..., :num_channels]
    pad = x.new_zeros(*x.shape[:-1], num_channels - current_channels)
    return torch.cat((x, pad), dim=-1)


def _mask_tensor(x: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return x
    return x.masked_fill(~mask.unsqueeze(-1), 0.0)


class SwooshR(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x - 1.0) - 0.08 * x - 0.313261687


class SwooshL(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x - 4.0) - 0.08 * x - 0.035


class BiasNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1.0e-8) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        if x.size(-1) != self.num_features:
            raise ValueError(
                f"BiasNorm expected {self.num_features} channels, got {x.size(-1)}."
            )
        bias = self.bias.view(*([1] * (x.ndim - 1)), self.num_features)
        rms = (x - bias).pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.log_scale.exp()


class BypassModule(nn.Module):
    def __init__(
        self,
        num_features: int,
        *,
        batch_count: int = 0,
        warmup_batches: int = 20_000,
        warmup_min: float = 0.9,
        steady_min: float = 0.2,
        max_value: float = 1.0,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_features))
        self.batch_count = batch_count
        self.warmup_batches = warmup_batches
        self.warmup_min = warmup_min
        self.steady_min = steady_min
        self.max_value = max_value

    def _current_scale(self) -> Tensor:
        minimum = self.warmup_min if self.training and self.batch_count < self.warmup_batches else self.steady_min
        return self.scale.clamp(min=minimum, max=self.max_value)

    def forward(self, residual: Tensor, update: Tensor) -> Tensor:
        scale = self._current_scale().view(1, 1, -1)
        return residual + (update - residual) * scale


class Downsample(nn.Module):
    def __init__(self, factor: int = 2) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError(f"Downsample factor must be >= 1, got {factor}.")
        self.factor = factor
        self.weights = nn.Parameter(torch.zeros(factor))

    def _downsample(self, x: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
        if self.factor == 1:
            return x, mask

        batch_size, time_steps, channels = x.shape
        output_time = (time_steps + self.factor - 1) // self.factor
        pad = output_time * self.factor - time_steps
        if pad > 0:
            pad_values = x[:, -1:, :].expand(batch_size, pad, channels)
            x = torch.cat((x, pad_values), dim=1)
            if mask is not None:
                pad_mask = mask[:, -1:].expand(batch_size, pad)
                mask = torch.cat((mask, pad_mask), dim=1)

        x = x.view(batch_size, output_time, self.factor, channels)
        weights = self.weights.softmax(dim=0).view(1, 1, self.factor, 1)

        if mask is None:
            return (x * weights).sum(dim=2), None

        window_mask = mask.view(batch_size, output_time, self.factor).unsqueeze(-1).to(x.dtype)
        masked_weights = weights * window_mask
        denom = masked_weights.sum(dim=2).clamp_min(1.0e-8)
        y = (x * masked_weights).sum(dim=2) / denom
        output_mask = mask.view(batch_size, output_time, self.factor).any(dim=2)
        y = y.masked_fill(~output_mask.unsqueeze(-1), 0.0)
        return y, output_mask

    def forward(self, x: Tensor) -> Tensor:
        y, _ = self._downsample(x, None)
        return y

    def apply_masked(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        y, output_mask = self._downsample(x, mask)
        assert output_mask is not None
        return y, output_mask


class Upsample(nn.Module):
    def __init__(self, factor: int = 2) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError(f"Upsample factor must be >= 1, got {factor}.")
        self.factor = factor

    def _upsample(
        self,
        x: Tensor,
        mask: Tensor | None,
        *,
        target_length: int | None,
    ) -> tuple[Tensor, Tensor | None]:
        if self.factor == 1:
            if target_length is None or target_length == x.size(1):
                return x, mask
            return x[:, :target_length], None if mask is None else mask[:, :target_length]

        y = x.repeat_interleave(self.factor, dim=1)
        output_mask = None if mask is None else mask.repeat_interleave(self.factor, dim=1)
        if target_length is not None:
            y = y[:, :target_length]
            if output_mask is not None:
                output_mask = output_mask[:, :target_length]
        return y, output_mask

    def forward(self, x: Tensor) -> Tensor:
        y, _ = self._upsample(x, None, target_length=None)
        return y

    def apply_masked(
        self,
        x: Tensor,
        mask: Tensor,
        *,
        target_length: int,
    ) -> tuple[Tensor, Tensor]:
        y, output_mask = self._upsample(x, mask, target_length=target_length)
        assert output_mask is not None
        return y, output_mask


class ConvNextBlock2d(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.pointwise_in = nn.Linear(channels, 384)
        self.activation = SwooshL()
        self.pointwise_out = nn.Linear(384, channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pointwise_in(x)
        x = self.activation(x)
        x = self.pointwise_out(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x


class ConvEmbed(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=(1, 2), padding=1),
            SwooshR(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=(2, 2), padding=1),
            SwooshR(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=(1, 2), padding=1),
            SwooshR(),
        )
        self.convnext = ConvNextBlock2d(128)

        freq_dim = input_dim
        for _ in range(3):
            freq_dim = (freq_dim + 1) // 2
        self.output_projection = nn.Linear(128 * freq_dim, output_dim)
        self.output_norm = BiasNorm(output_dim)

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        mask = _make_padding_mask(lengths, max_length=x.size(1))
        x = _mask_tensor(x, mask)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.convnext(x)

        batch_size, channels, time_steps, freq_dim = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size, time_steps, freq_dim * channels)
        x = self.output_projection(x)
        x = self.output_norm(x)

        output_lengths = _conv_out_length(lengths, kernel_size=3, stride=1, padding=1)
        output_lengths = _conv_out_length(output_lengths, kernel_size=3, stride=2, padding=1)
        output_lengths = _conv_out_length(output_lengths, kernel_size=3, stride=1, padding=1)
        output_mask = _make_padding_mask(output_lengths, max_length=x.size(1))
        x = _mask_tensor(x, output_mask)
        return x, output_lengths


class CompactRelPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0, length_factor: float = 1.0) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"CompactRelPositionalEncoding requires an even dimension, got {embed_dim}.")
        if length_factor < 1.0:
            raise ValueError(f"length_factor must be >= 1.0, got {length_factor}.")
        self.embed_dim = embed_dim
        self.length_factor = length_factor
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence_length: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        positions = torch.arange(
            -(sequence_length - 1),
            sequence_length,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)
        frequencies = 1 + torch.arange(self.embed_dim // 2, device=device, dtype=torch.float32)

        compression_length = math.sqrt(self.embed_dim)
        compressed = compression_length * positions.sign() * (
            (positions.abs() + compression_length).log() - math.log(compression_length)
        )
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)
        angles = torch.atan(compressed / length_scale)

        encoding = torch.zeros(positions.size(0), self.embed_dim, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.cos(angles * frequencies)
        encoding[:, 1::2] = torch.sin(angles * frequencies)
        encoding[:, -1] = 1.0
        return self.dropout(encoding).to(dtype=dtype)


class MultiHeadAttentionWeights(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.scale = query_head_dim ** -0.5
        self.query_proj = nn.Linear(embed_dim, num_heads * query_head_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, num_heads * query_head_dim, bias=False)
        self.pos_query_proj = nn.Linear(embed_dim, num_heads * pos_head_dim, bias=False)
        self.pos_proj = nn.Linear(pos_dim, num_heads * pos_head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, pos_emb: Tensor, mask: Tensor | None) -> Tensor:
        batch_size, time_steps, _ = x.shape
        q = self.query_proj(x).view(batch_size, time_steps, self.num_heads, self.query_head_dim)
        k = self.key_proj(x).view(batch_size, time_steps, self.num_heads, self.query_head_dim)
        p = self.pos_query_proj(x).view(batch_size, time_steps, self.num_heads, self.pos_head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        p = p.permute(0, 2, 1, 3)

        content_scores = torch.einsum("bhtd,bhsd->bhts", q, k) * self.scale

        rel = self.pos_proj(pos_emb).view(2 * time_steps - 1, self.num_heads, self.pos_head_dim)
        rel = rel.permute(1, 0, 2)
        relative_index = (
            torch.arange(time_steps, device=x.device).unsqueeze(1)
            - torch.arange(time_steps, device=x.device).unsqueeze(0)
            + time_steps
            - 1
        )
        rel = rel[:, relative_index]
        pos_scores = torch.einsum("bhtd,htsd->bhts", p, rel)

        scores = content_scores + pos_scores
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], -1.0e4)

        attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, :, None], 0.0)
        return self.dropout(attn)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, value_head_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.value_proj = nn.Linear(embed_dim, num_heads * value_head_dim, bias=True)
        self.output_proj = nn.Linear(num_heads * value_head_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_weights: Tensor) -> Tensor:
        batch_size, time_steps, _ = x.shape
        v = self.value_proj(x).view(batch_size, time_steps, self.num_heads, -1)
        v = v.permute(0, 2, 1, 3)
        y = torch.einsum("bhts,bhsd->bhtd", attn_weights, v)
        y = y.permute(0, 2, 1, 3).reshape(batch_size, time_steps, -1)
        return self.dropout(self.output_proj(y))


class NonLinearAttention(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden_dim * 3, bias=True)
        self.output_proj = nn.Linear(hidden_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_weights: Tensor) -> Tensor:
        a, b, c = self.proj(x).chunk(3, dim=-1)
        attended = torch.einsum("bij,bjd->bid", attn_weights[:, 0], torch.tanh(b) * c)
        return self.dropout(self.output_proj(a * attended))


class FeedForwardModule(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.activation = SwooshL()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=True)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return self.out_dropout(x)


class ZipformerConvModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.0) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"Zipformer convolution kernel must be odd, got {kernel_size}.")
        self.input_proj = nn.Linear(dim, dim * 2, bias=True)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.activation = SwooshR()
        self.output_proj = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        x, gate = self.input_proj(x).chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)
        x = x.transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(1), 0.0)
        x = self.depthwise(x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        x = self.output_proj(x)
        x = self.dropout(x)
        return _mask_tensor(x, mask)


class ZipformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        mult: int = 4,
        *,
        feedforward_dim: int | None = None,
        dim_head: int = 32,
        pos_dim: int = 48,
        pos_head_dim: int = 4,
        value_head_dim: int | None = None,
        dropout: float = 0.0,
        conv_kernel_size: int = 31,
    ) -> None:
        super().__init__()
        ff_dim = dim * mult if feedforward_dim is None else feedforward_dim
        if value_head_dim is None:
            value_head_dim = max(1, dim // heads)

        self.attention_weights = MultiHeadAttentionWeights(
            dim,
            pos_dim=pos_dim,
            num_heads=heads,
            query_head_dim=dim_head,
            pos_head_dim=pos_head_dim,
            dropout=dropout,
        )
        self.feed_forward1 = FeedForwardModule(dim, (ff_dim * 3) // 4, dropout)
        self.non_linear_attention = NonLinearAttention(dim, (dim * 3) // 4, dropout)
        self.self_attention1 = SelfAttention(dim, heads, value_head_dim, dropout)
        self.conv1 = ZipformerConvModule(dim, kernel_size=conv_kernel_size, dropout=dropout)
        self.feed_forward2 = FeedForwardModule(dim, ff_dim, dropout)
        self.mid_bypass = BypassModule(dim)
        self.self_attention2 = SelfAttention(dim, heads, value_head_dim, dropout)
        self.conv2 = ZipformerConvModule(dim, kernel_size=conv_kernel_size, dropout=dropout)
        self.feed_forward3 = FeedForwardModule(dim, (ff_dim * 5) // 4, dropout)
        self.output_norm = BiasNorm(dim)
        self.output_bypass = BypassModule(dim)

    def forward(self, x: Tensor, pos_emb: Tensor, mask: Tensor | None = None) -> Tensor:
        residual = x
        attn_weights = self.attention_weights(x, pos_emb, mask)

        x = _mask_tensor(x + self.feed_forward1(x), mask)
        x = _mask_tensor(x + self.non_linear_attention(x, attn_weights), mask)
        x = _mask_tensor(x + self.self_attention1(x, attn_weights), mask)
        x = _mask_tensor(x + self.conv1(x, mask), mask)
        x = _mask_tensor(x + self.feed_forward2(x), mask)

        x = _mask_tensor(self.mid_bypass(residual, x), mask)
        x = _mask_tensor(x + self.self_attention2(x, attn_weights), mask)
        x = _mask_tensor(x + self.conv2(x, mask), mask)
        x = _mask_tensor(x + self.feed_forward3(x), mask)
        x = self.output_norm(x)
        x = _mask_tensor(self.output_bypass(residual, x), mask)
        return x


class ZipformerStack(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_layers: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_dim: int,
        conv_kernel_size: int,
        pos_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.positional_encoding = CompactRelPositionalEncoding(pos_dim, dropout=0.0)
        self.blocks = nn.ModuleList(
            [
                ZipformerBlock(
                    dim,
                    heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dim_head=query_head_dim,
                    pos_dim=pos_dim,
                    pos_head_dim=pos_head_dim,
                    value_head_dim=value_head_dim,
                    dropout=dropout,
                    conv_kernel_size=conv_kernel_size,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        pos_emb = self.positional_encoding(x.size(1), device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, pos_emb, mask=mask)
        return x


class DownsampledZipformerStack(nn.Module):
    def __init__(self, stack: ZipformerStack, *, dim: int, downsample: int) -> None:
        super().__init__()
        self.downsample = Downsample(downsample)
        self.stack = stack
        self.upsample = Upsample(downsample)
        self.output_bypass = BypassModule(dim)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        residual = x
        x, stack_mask = self.downsample.apply_masked(x, mask)
        x = self.stack(x, stack_mask)
        x, _ = self.upsample.apply_masked(x, stack_mask, target_length=residual.size(1))
        x = _mask_tensor(x, mask)
        return _mask_tensor(self.output_bypass(residual, x), mask)


class Zipformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_downsampling_factor: int,
        downsampling_factor: Sequence[int],
        encoder_dim: Sequence[int],
        num_encoder_layers: Sequence[int],
        num_heads: Sequence[int],
        query_head_dim: Sequence[int],
        pos_head_dim: Sequence[int],
        value_head_dim: Sequence[int],
        feedforward_dim: Sequence[int],
        cnn_module_kernel: Sequence[int],
        pos_dim: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        tuple_lengths = {
            len(downsampling_factor),
            len(encoder_dim),
            len(num_encoder_layers),
            len(num_heads),
            len(query_head_dim),
            len(pos_head_dim),
            len(value_head_dim),
            len(feedforward_dim),
            len(cnn_module_kernel),
        }
        if len(tuple_lengths) != 1:
            raise ValueError("All Zipformer stack parameter tuples must have the same length.")

        self.encoder_dim = tuple(encoder_dim)
        self.downsampling_factor = tuple(downsampling_factor)
        self.output_downsampling_factor = output_downsampling_factor
        self.output_dim = max(self.encoder_dim)

        self.conv_embed = ConvEmbed(input_dim, self.encoder_dim[0])
        stacks: list[nn.Module] = []
        for stack_index, dim in enumerate(self.encoder_dim):
            stack = ZipformerStack(
                dim=dim,
                num_layers=num_encoder_layers[stack_index],
                num_heads=num_heads[stack_index],
                query_head_dim=query_head_dim[stack_index],
                pos_head_dim=pos_head_dim[stack_index],
                value_head_dim=value_head_dim[stack_index],
                feedforward_dim=feedforward_dim[stack_index],
                conv_kernel_size=cnn_module_kernel[stack_index],
                pos_dim=pos_dim,
                dropout=dropout,
            )
            if self.downsampling_factor[stack_index] == 1:
                stacks.append(stack)
            else:
                stacks.append(
                    DownsampledZipformerStack(
                        stack,
                        dim=dim,
                        downsample=self.downsampling_factor[stack_index],
                    )
                )
        self.stacks = nn.ModuleList(stacks)
        self.output_downsample = Downsample(output_downsampling_factor)
        self.batch_count = 0

    def set_batch_count(self, batch_count: int) -> None:
        self.batch_count = batch_count
        for module in self.modules():
            if hasattr(module, "batch_count"):
                module.batch_count = batch_count

    def _get_full_dim_output(self, outputs: list[Tensor]) -> Tensor:
        pieces = [outputs[-1]]
        current_dim = self.encoder_dim[-1]
        for stack_index in range(len(outputs) - 2, -1, -1):
            stack_dim = self.encoder_dim[stack_index]
            if stack_dim > current_dim:
                pieces.append(outputs[stack_index][..., current_dim:stack_dim])
                current_dim = stack_dim
        return torch.cat(pieces, dim=-1)

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        x, lengths = self.conv_embed(x, lengths)
        mask = _make_padding_mask(lengths, max_length=x.size(1))

        outputs: list[Tensor] = []
        for stack_index, stack in enumerate(self.stacks):
            x = _convert_num_channels(x, self.encoder_dim[stack_index])
            x = _mask_tensor(x, mask)
            x = stack(x, mask)
            outputs.append(x)

        x = self._get_full_dim_output(outputs)
        x, mask = self.output_downsample.apply_masked(x, mask)
        lengths = _ceil_divide(lengths, self.output_downsampling_factor)
        x = _mask_tensor(x, mask)
        return x, lengths
