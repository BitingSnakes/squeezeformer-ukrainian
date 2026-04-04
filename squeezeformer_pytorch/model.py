from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .masking import make_attention_mask, make_sequence_mask

try:
    import transformer_engine.pytorch as te
except ImportError, OSError:
    te = None


FP8_SHAPE_ALIGNMENT = 16


def _subsample_length(
    lengths: Tensor,
    kernel_size: int = 3,
    stride: int = 2,
    repeats: int = 2,
) -> Tensor:
    output = lengths.to(dtype=torch.int64)
    for _ in range(repeats):
        if kernel_size != 3 or stride != 2:
            raise ValueError(
                "The current implementation expects the paper's fixed 3x3 stride-2 subsampler."
            )
        output = torch.div(output, stride, rounding_mode="floor")
    return output


def transformer_engine_available() -> bool:
    return te is not None


def _is_fp8_compatible_linear_shape(in_features: int, out_features: int) -> bool:
    return in_features % FP8_SHAPE_ALIGNMENT == 0 and out_features % FP8_SHAPE_ALIGNMENT == 0


def _pad_tensor_along_dim(x: Tensor, dim: int, target_size: int) -> Tensor:
    if x.size(dim) >= target_size:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = target_size - x.size(dim)
    return torch.cat([x, x.new_zeros(pad_shape)], dim=dim)


def _pad_mask_to_length(mask: Tensor | None, target_length: int) -> Tensor | None:
    if mask is None or mask.size(-1) >= target_length:
        return mask
    pad_shape = list(mask.shape)
    pad_shape[-1] = target_length - mask.size(-1)
    return torch.cat([mask, mask.new_zeros(pad_shape, dtype=mask.dtype)], dim=-1)


def _pad_attn_mask_to_length(mask: Tensor | None, target_length: int) -> Tensor | None:
    if mask is None or mask.size(-1) >= target_length:
        return mask
    pad_amount = target_length - mask.size(-1)
    pad_dims = [0, pad_amount, 0, pad_amount]
    return F.pad(mask, pad_dims, value=False)


def _padded_sequence_length(length: int) -> int:
    return math.ceil(length / FP8_SHAPE_ALIGNMENT) * FP8_SHAPE_ALIGNMENT


def make_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    use_transformer_engine: bool = False,
) -> nn.Module:
    if (
        use_transformer_engine
        and te is not None
        and _is_fp8_compatible_linear_shape(in_features, out_features)
    ):
        return te.Linear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)


def make_layer_norm(dim: int, *, use_transformer_engine: bool = False) -> nn.Module:
    # Keep LayerNorm on PyTorch kernels. Transformer Engine FP8 support is still
    # applied through TE Linear modules, while TE LayerNorm has shown unstable
    # kernel launch behavior on this model's [batch, time, hidden] activations.
    return nn.LayerNorm(dim)


def apply_linear_with_fp8_padding(module: nn.Module, x: Tensor) -> Tensor:
    if te is None or not isinstance(module, te.Linear) or x.dim() < 2:
        return module(x)

    original_shape = x.shape
    flat = x.reshape(-1, original_shape[-1])
    padded_rows = _padded_sequence_length(flat.size(0))
    if padded_rows != flat.size(0):
        flat = _pad_tensor_along_dim(flat, dim=0, target_size=padded_rows)
    flat = module(flat)
    flat = flat[: math.prod(original_shape[:-1])]
    return flat.reshape(*original_shape[:-1], flat.size(-1))


class ScaleBiasLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale + self.bias


class RelativePositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_length: int = 5000) -> None:
        super().__init__()
        positions = torch.arange(max_length - 1, -max_length, -1.0).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(2 * max_length - 1, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        length = x.size(1)
        center = self.pe.size(1) // 2
        start = center - length + 1
        end = center + length
        return self.pe[:, start:end].to(dtype=x.dtype, device=x.device)


class RelPositionMultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_transformer_engine: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = make_linear(dim, dim, use_transformer_engine=use_transformer_engine)
        self.key = make_linear(dim, dim, use_transformer_engine=use_transformer_engine)
        self.value = make_linear(dim, dim, use_transformer_engine=use_transformer_engine)
        self.pos_proj = make_linear(
            dim,
            dim,
            bias=False,
            use_transformer_engine=use_transformer_engine,
        )
        self.out_proj = make_linear(dim, dim, use_transformer_engine=use_transformer_engine)
        self.pos_bias_u = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: Tensor) -> Tensor:
        batch, time, _ = x.shape
        return x.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _relative_shift(x: Tensor) -> Tensor:
        batch, heads, time_q, time_k = x.shape
        zero_pad = x.new_zeros(batch, heads, time_q, 1)
        x = torch.cat([zero_pad, x], dim=-1)
        x = x.view(batch, heads, time_k + 1, time_q)
        x = x[:, :, 1:, :].view(batch, heads, time_q, time_k)
        return x

    def forward(self, x: Tensor, pos: Tensor, mask: Tensor | None = None) -> Tensor:
        query = self._shape(apply_linear_with_fp8_padding(self.query, x))
        key = self._shape(apply_linear_with_fp8_padding(self.key, x))
        value = self._shape(apply_linear_with_fp8_padding(self.value, x))
        pos_proj = self._shape(apply_linear_with_fp8_padding(self.pos_proj, pos))

        query_u = query + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        query_v = query + self.pos_bias_v.unsqueeze(0).unsqueeze(2)

        content_scores = torch.matmul(query_u, key.transpose(-1, -2))
        position_scores = torch.matmul(query_v, pos_proj.transpose(-1, -2))
        position_scores = self._relative_shift(position_scores)
        position_scores = position_scores[..., : content_scores.size(-1)]

        scores = (content_scores + position_scores) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, value)
        context = context.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.dim)
        return apply_linear_with_fp8_padding(self.out_proj, context)


class FlashMultiHeadAttention(nn.Module):
    """PyTorch SDPA-backed MHA.

    On supported CUDA shapes/dtypes, scaled_dot_product_attention dispatches to
    FlashAttention kernels automatically.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_transformer_engine: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = make_linear(dim, dim, use_transformer_engine=use_transformer_engine)
        self.key = make_linear(dim, dim, use_transformer_engine=use_transformer_engine)
        self.value = make_linear(dim, dim, use_transformer_engine=use_transformer_engine)
        self.out_proj = make_linear(dim, dim, use_transformer_engine=use_transformer_engine)
        self.dropout_p = dropout

    def _shape(self, x: Tensor) -> Tensor:
        batch, time, _ = x.shape
        return x.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _to_sdpa_mask(mask: Tensor) -> Tensor:
        return mask.unsqueeze(1)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        query = self._shape(apply_linear_with_fp8_padding(self.query, x))
        key = self._shape(apply_linear_with_fp8_padding(self.key, x))
        value = self._shape(apply_linear_with_fp8_padding(self.value, x))
        attn_mask = self._to_sdpa_mask(mask) if mask is not None else None
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        context = context.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.dim)
        return apply_linear_with_fp8_padding(self.out_proj, context)


class FeedForwardModule(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        residual_factor: float = 1.0,
        adaptive_scale: bool = True,
        use_transformer_engine: bool = False,
    ) -> None:
        super().__init__()
        self.input_transform: nn.Module = (
            ScaleBiasLayer(dim)
            if adaptive_scale
            else make_layer_norm(dim, use_transformer_engine=use_transformer_engine)
        )
        self.linear1 = make_linear(
            dim,
            expansion_factor * dim,
            use_transformer_engine=use_transformer_engine,
        )
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = make_linear(
            expansion_factor * dim,
            dim,
            use_transformer_engine=use_transformer_engine,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.residual_factor = residual_factor

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.input_transform(x)
        x = apply_linear_with_fp8_padding(self.linear1, x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = apply_linear_with_fp8_padding(self.linear2, x)
        x = self.dropout2(x)
        return residual + self.residual_factor * x


class AttentionModule(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        adaptive_scale: bool = True,
        attention_backend: str = "relative",
        use_transformer_engine: bool = False,
    ) -> None:
        super().__init__()
        self.input_transform: nn.Module = (
            ScaleBiasLayer(dim)
            if adaptive_scale
            else make_layer_norm(dim, use_transformer_engine=use_transformer_engine)
        )
        if attention_backend == "relative":
            self.attn: nn.Module = RelPositionMultiHeadAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                use_transformer_engine=use_transformer_engine,
            )
        elif attention_backend == "flash":
            self.attn = FlashMultiHeadAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                use_transformer_engine=use_transformer_engine,
            )
        else:
            raise ValueError(f"Unsupported attention backend: {attention_backend}")
        self.attention_backend = attention_backend
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, pos: Tensor, mask: Tensor | None = None) -> Tensor:
        residual = x
        x = self.input_transform(x)
        if self.attention_backend == "relative":
            x = self.attn(x, pos=pos, mask=mask)
        else:
            x = self.attn(x, mask=mask)
        x = self.dropout(x)
        return residual + x


class ConvolutionModule(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        use_glu: bool = False,
        adaptive_scale: bool = True,
        use_transformer_engine: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length depthwise convolution")
        hidden_dim = dim * expansion_factor
        self.input_transform: nn.Module = (
            ScaleBiasLayer(dim)
            if adaptive_scale
            else make_layer_norm(dim, use_transformer_engine=use_transformer_engine)
        )
        self.pointwise_in = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.use_glu = use_glu
        if use_glu:
            raise NotImplementedError(
                "The paper's final Squeezeformer replaces GLU with Swish everywhere."
            )
        self.activation1 = nn.SiLU()
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.activation2 = nn.SiLU()
        self.pointwise_out = nn.Conv1d(hidden_dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        residual = x
        x = self.input_transform(x)
        x = x.transpose(1, 2)
        x = self.pointwise_in(x)
        if self.use_glu:
            x = F.glu(torch.cat([x, x], dim=1), dim=1)
        else:
            x = self.activation1(x)
        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(1).to(dtype=x.dtype)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation2(x)
        x = self.pointwise_out(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return residual + x


class MHSAFFModule(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_expansion_factor: int,
        dropout: float,
        attention_backend: str,
        use_transformer_engine: bool = False,
    ) -> None:
        super().__init__()
        self.attn = AttentionModule(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            adaptive_scale=True,
            attention_backend=attention_backend,
            use_transformer_engine=use_transformer_engine,
        )
        self.mid_norm = make_layer_norm(dim, use_transformer_engine=use_transformer_engine)
        self.ff = FeedForwardModule(
            dim=dim,
            expansion_factor=ff_expansion_factor,
            dropout=dropout,
            residual_factor=1.0,
            adaptive_scale=True,
            use_transformer_engine=use_transformer_engine,
        )
        self.out_norm = make_layer_norm(dim, use_transformer_engine=use_transformer_engine)

    def forward(self, x: Tensor, pos: Tensor, attn_mask: Tensor | None) -> Tensor:
        x = self.attn(x, pos=pos, mask=attn_mask)
        x = self.mid_norm(x)
        x = self.ff(x)
        return self.out_norm(x)


class ConvFFModule(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        conv_expansion_factor: int,
        ff_expansion_factor: int,
        dropout: float,
        use_transformer_engine: bool = False,
    ) -> None:
        super().__init__()
        self.conv = ConvolutionModule(
            dim=dim,
            kernel_size=kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout=dropout,
            use_glu=False,
            adaptive_scale=True,
            use_transformer_engine=use_transformer_engine,
        )
        self.mid_norm = make_layer_norm(dim, use_transformer_engine=use_transformer_engine)
        self.ff = FeedForwardModule(
            dim=dim,
            expansion_factor=ff_expansion_factor,
            dropout=dropout,
            residual_factor=1.0,
            adaptive_scale=True,
            use_transformer_engine=use_transformer_engine,
        )
        self.out_norm = make_layer_norm(dim, use_transformer_engine=use_transformer_engine)

    def forward(self, x: Tensor, pad_mask: Tensor | None) -> Tensor:
        x = self.conv(x, pad_mask=pad_mask)
        x = self.mid_norm(x)
        x = self.ff(x)
        return self.out_norm(x)


class SqueezeformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        ff_expansion_factor: int,
        conv_expansion_factor: int,
        dropout: float,
        block_pattern: tuple[str, ...],
        attention_backend: str,
        drop_path_rate: float = 0.0,
        use_transformer_engine: bool = False,
    ) -> None:
        super().__init__()
        self.block_pattern = block_pattern
        self.drop_path_rate = drop_path_rate
        layers: list[nn.Module] = []
        for token in block_pattern:
            if token == "M":
                layers.append(
                    MHSAFFModule(
                        dim=dim,
                        num_heads=num_heads,
                        ff_expansion_factor=ff_expansion_factor,
                        dropout=dropout,
                        attention_backend=attention_backend,
                        use_transformer_engine=use_transformer_engine,
                    )
                )
            elif token == "C":
                layers.append(
                    ConvFFModule(
                        dim=dim,
                        kernel_size=kernel_size,
                        conv_expansion_factor=conv_expansion_factor,
                        ff_expansion_factor=ff_expansion_factor,
                        dropout=dropout,
                        use_transformer_engine=use_transformer_engine,
                    )
                )
            elif token == "s":
                layers.append(nn.Identity())
            else:
                raise ValueError(f"Unsupported block token: {token}")
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        attn_mask: Tensor | None,
        pad_mask: Tensor | None,
    ) -> Tensor:
        residual = x
        for token, layer in zip(self.block_pattern, self.layers, strict=True):
            if token == "M":
                x = layer(x, pos=pos, attn_mask=attn_mask)
            elif token == "C":
                x = layer(x, pad_mask=pad_mask)
        if self.training and self.drop_path_rate > 0:
            keep_prob = 1.0 - self.drop_path_rate
            shape = (x.size(0),) + (1,) * (x.dim() - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            binary_mask = torch.floor(random_tensor)
            x = residual + (x - residual) * binary_mask / keep_prob
        return x


class Conv2dSubsampling(nn.Module):
    def __init__(self, in_features: int, channels: int, depthwise_separable: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.channels = channels
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, stride=2)
        self.depthwise_separable = depthwise_separable
        if depthwise_separable:
            self.conv2_dw = nn.Conv2d(channels, channels, kernel_size=3, stride=2, groups=channels)
            self.conv2_pw = nn.Conv2d(channels, channels, kernel_size=1)
        else:
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2)
        self.activation = nn.ReLU()
        out_features = self._output_freq_bins(in_features) * channels
        self.output_dim = out_features

    def _output_freq_bins(self, features: int) -> int:
        value = features
        for _ in range(2):
            value = math.floor((value - 1) / 2) + 1
        return value

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = F.pad(x, (0, 1, 0, 1))
        x = self.activation(self.conv1(x))
        x = F.pad(x, (0, 1, 0, 1))
        if self.depthwise_separable:
            x = self.conv2_pw(self.conv2_dw(x))
        else:
            x = self.conv2(x)
        x = self.activation(x)
        batch, channels, time, freq = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(batch, time, channels * freq)


class TimeReductionLayer(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, stride: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        x = x.transpose(1, 2)
        pad_mask = make_sequence_mask(lengths, max_length=x.size(-1))
        x = x * pad_mask.unsqueeze(1).to(dtype=x.dtype)
        x = F.pad(x, (0, max(0, self.kernel_size - self.stride)))
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)
        lengths = torch.div(lengths, self.stride, rounding_mode="floor")
        return x, lengths


class TimeRecoveryLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        stride: int = 2,
        use_transformer_engine: bool = False,
    ) -> None:
        super().__init__()
        self.proj = make_linear(dim, dim, use_transformer_engine=use_transformer_engine)
        self.stride = stride

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = torch.repeat_interleave(x, repeats=self.stride, dim=1)
        target_length = skip.size(1)
        if x.size(1) < target_length:
            pad_length = target_length - x.size(1)
            x = F.pad(x, (0, 0, 0, pad_length), mode="replicate")
        else:
            x = x[:, :target_length, :]
        return apply_linear_with_fp8_padding(self.proj, x) + skip


@dataclass(frozen=True)
class SqueezeformerConfig:
    input_features: int = 80
    d_model: int = 256
    num_layers: int = 16
    num_heads: int = 4
    kernel_size: int = 31
    ff_expansion_factor: int = 4
    conv_expansion_factor: int = 2
    dropout: float = 0.1
    depthwise_subsampling: bool = True
    block_pattern: tuple[str, ...] = ("M", "s", "C", "s")
    time_reduction_kernel_size: int = 3
    activation_checkpointing: bool = False
    stochastic_depth_rate: float = 0.0
    time_reduce_idx: tuple[int, ...] = (7,)
    time_recover_idx: tuple[int, ...] = (15,)
    attention_backend: str = "relative"

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_pattern", tuple(self.block_pattern))
        object.__setattr__(self, "time_reduce_idx", tuple(self.time_reduce_idx))
        object.__setattr__(self, "time_recover_idx", tuple(self.time_recover_idx))


VARIANT_CONFIGS: dict[str, SqueezeformerConfig] = {
    "xs": SqueezeformerConfig(
        d_model=144,
        num_layers=16,
        num_heads=4,
        time_reduce_idx=(7,),
        time_recover_idx=(15,),
    ),
    "s": SqueezeformerConfig(
        d_model=196,
        num_layers=18,
        num_heads=4,
        time_reduce_idx=(8,),
        time_recover_idx=(17,),
    ),
    "sm": SqueezeformerConfig(
        d_model=256,
        num_layers=16,
        num_heads=4,
        time_reduce_idx=(7,),
        time_recover_idx=(15,),
    ),
    "m": SqueezeformerConfig(
        d_model=324,
        num_layers=20,
        num_heads=4,
        stochastic_depth_rate=0.03,
        time_reduce_idx=(9,),
        time_recover_idx=(19,),
    ),
    "ml": SqueezeformerConfig(
        d_model=512,
        num_layers=18,
        num_heads=8,
        stochastic_depth_rate=0.05,
        time_reduce_idx=(8,),
        time_recover_idx=(17,),
    ),
    "l": SqueezeformerConfig(
        d_model=640,
        num_layers=22,
        num_heads=8,
        stochastic_depth_rate=0.1,
        time_reduce_idx=(10,),
        time_recover_idx=(21,),
    ),
}


def squeezeformer_variant(name: str) -> SqueezeformerConfig:
    key = name.lower()
    if key not in VARIANT_CONFIGS:
        raise KeyError(f"Unknown Squeezeformer variant: {name}")
    return replace(VARIANT_CONFIGS[key])


class SqueezeformerEncoder(nn.Module):
    """Encoder-only Squeezeformer.

    This follows the paper's architecture deltas and the released reference configs:
    depthwise-separable 4x subsampling, MF/CF block layout, scaled post-LN residual
    modules, and a single temporal downsample/upsample pair.
    """

    def __init__(self, config: SqueezeformerConfig, use_transformer_engine: bool = False) -> None:
        super().__init__()
        self.config = config
        self.use_transformer_engine = use_transformer_engine
        self.subsampling = Conv2dSubsampling(
            in_features=config.input_features,
            channels=config.d_model,
            depthwise_separable=config.depthwise_subsampling,
        )
        self.input_projection = make_linear(
            self.subsampling.output_dim,
            config.d_model,
            use_transformer_engine=use_transformer_engine,
        )
        self.input_norm = make_layer_norm(
            config.d_model,
            use_transformer_engine=use_transformer_engine,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.positional_encoding = RelativePositionalEncoding(config.d_model)
        self.time_reduce = nn.ModuleDict(
            {
                str(i): TimeReductionLayer(
                    config.d_model,
                    kernel_size=config.time_reduction_kernel_size,
                )
                for i in config.time_reduce_idx
            }
        )
        self.time_recover = nn.ModuleDict(
            {
                str(i): TimeRecoveryLayer(
                    config.d_model,
                    use_transformer_engine=use_transformer_engine,
                )
                for i in config.time_recover_idx
            }
        )
        self.blocks = nn.ModuleList(
            [
                SqueezeformerBlock(
                    dim=config.d_model,
                    num_heads=config.num_heads,
                    kernel_size=config.kernel_size,
                    ff_expansion_factor=config.ff_expansion_factor,
                    conv_expansion_factor=config.conv_expansion_factor,
                    dropout=config.dropout,
                    block_pattern=config.block_pattern,
                    attention_backend=config.attention_backend,
                    drop_path_rate=(
                        config.stochastic_depth_rate * layer_index / max(1, config.num_layers - 1)
                    ),
                    use_transformer_engine=use_transformer_engine,
                )
                for layer_index in range(config.num_layers)
            ]
        )

    def _forward_impl(
        self,
        features: Tensor,
        lengths: Tensor,
        intermediate_layer_indices: tuple[int, ...] = (),
        post_block_transforms: dict[int, Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]]
        | None = None,
    ) -> tuple[Tensor, Tensor, dict[int, Tensor], dict[int, Tensor]]:
        if features.dim() != 3:
            raise ValueError(
                f"Expected features with shape [batch, time, mel], got {tuple(features.shape)}"
            )
        invalid_indices = [
            layer_index
            for layer_index in intermediate_layer_indices
            if not 0 <= layer_index < len(self.blocks)
        ]
        if invalid_indices:
            raise ValueError(
                "intermediate_layer_indices must reference encoder blocks within "
                f"[0, {len(self.blocks) - 1}], got {invalid_indices}."
            )

        x = self.subsampling(features)
        lengths = _subsample_length(lengths)
        x = apply_linear_with_fp8_padding(self.input_projection, x)
        x = x * math.sqrt(self.config.d_model)
        x = self.dropout(x)
        x = self.input_norm(x)

        recover_stack: list[tuple[Tensor, Tensor]] = []
        intermediate_xs: dict[int, Tensor] = {}
        intermediate_lengths: dict[int, Tensor] = {}
        post_block_transforms = post_block_transforms or {}
        for layer_index, block in enumerate(self.blocks):
            if layer_index in self.config.time_reduce_idx:
                recover_stack.append((x, lengths))
                x, lengths = self.time_reduce[str(layer_index)](x, lengths)

            if layer_index in self.config.time_recover_idx:
                if not recover_stack:
                    raise RuntimeError(
                        "Encountered a recovery layer without a matching reduced activation"
                    )
                skip_x, skip_lengths = recover_stack.pop()
                x = self.time_recover[str(layer_index)](x, skip_x)
                lengths = skip_lengths

            if self.use_transformer_engine:
                padded_time = _padded_sequence_length(x.size(1))
                x = _pad_tensor_along_dim(x, dim=1, target_size=padded_time)
                pos = self.positional_encoding(x)
                attn_mask = _pad_attn_mask_to_length(
                    make_attention_mask(lengths, max_length=x.size(1)),
                    target_length=padded_time,
                )
                pad_mask = _pad_mask_to_length(
                    make_sequence_mask(lengths, max_length=x.size(1)),
                    target_length=padded_time,
                )
            else:
                pos = self.positional_encoding(x)
                attn_mask = make_attention_mask(lengths, max_length=x.size(1))
                pad_mask = make_sequence_mask(lengths, max_length=x.size(1))
            if self.config.activation_checkpointing and self.training:
                x = activation_checkpoint(
                    lambda a, b, c, d: block(a, pos=b, attn_mask=c, pad_mask=d),
                    x,
                    pos,
                    attn_mask,
                    pad_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, pos=pos, attn_mask=attn_mask, pad_mask=pad_mask)
            x = x[:, : int(lengths.max().item()), :]
            if layer_index in intermediate_layer_indices:
                intermediate_xs[layer_index] = x
                intermediate_lengths[layer_index] = lengths
            if layer_index in post_block_transforms:
                x, lengths = post_block_transforms[layer_index](x, lengths)
                x = x[:, : int(lengths.max().item()), :]

        return x, lengths, intermediate_xs, intermediate_lengths

    def forward(self, features: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        x, lengths, _, _ = self._forward_impl(features, lengths)
        return x, lengths

    def forward_with_intermediates(
        self,
        features: Tensor,
        lengths: Tensor,
        intermediate_layer_indices: tuple[int, ...],
        post_block_transforms: dict[int, Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]]
        | None = None,
    ) -> tuple[Tensor, Tensor, dict[int, Tensor], dict[int, Tensor]]:
        x, output_lengths, intermediate_xs, intermediate_lengths = self._forward_impl(
            features,
            lengths,
            intermediate_layer_indices=intermediate_layer_indices,
            post_block_transforms=post_block_transforms,
        )
        missing_indices = [
            layer_index
            for layer_index in intermediate_layer_indices
            if layer_index not in intermediate_xs or layer_index not in intermediate_lengths
        ]
        if missing_indices:
            raise RuntimeError(
                f"Failed to capture intermediate encoder outputs for layers {missing_indices}."
            )
        return x, output_lengths, intermediate_xs, intermediate_lengths


def build_squeezeformer_encoder(
    variant: str = "sm",
    **overrides: int | float | tuple[int, ...] | bool | str,
) -> SqueezeformerEncoder:
    use_transformer_engine = bool(overrides.pop("use_transformer_engine", False))
    config = replace(squeezeformer_variant(variant), **overrides)
    return SqueezeformerEncoder(config, use_transformer_engine=use_transformer_engine)
