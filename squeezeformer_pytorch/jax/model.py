from __future__ import annotations

import math
from dataclasses import dataclass, replace

import flax.linen as nn
import jax
import jax.numpy as jnp


def _subsample_length(
    lengths: jnp.ndarray,
    kernel_size: int = 3,
    stride: int = 2,
    repeats: int = 2,
) -> jnp.ndarray:
    output = lengths.astype(jnp.int32)
    for _ in range(repeats):
        if kernel_size != 3 or stride != 2:
            raise ValueError("JAX subsampling follows the fixed 3x3 stride-2 frontend.")
        output = output // stride
    return output


def _make_pad_mask(lengths: jnp.ndarray, max_length: int | None = None) -> jnp.ndarray:
    if max_length is None:
        max_length = int(lengths.max())
    positions = jnp.arange(max_length, dtype=jnp.int32)
    return positions[None, :] < lengths[:, None]


def _make_attn_mask(lengths: jnp.ndarray, max_length: int) -> jnp.ndarray:
    pad_mask = _make_pad_mask(lengths, max_length=max_length)
    return pad_mask[:, None, :] & pad_mask[:, :, None]


class ScaleBiasLayer(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        bias = self.param("bias", nn.initializers.zeros, (self.dim,))
        return x * scale + bias


class IdentityLayer(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


class RelativePositionalEncoding(nn.Module):
    dim: int
    max_length: int = 5000

    def setup(self) -> None:
        positions = jnp.arange(self.max_length - 1, -self.max_length, -1.0)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.dim, 2, dtype=jnp.float32) * (-math.log(10000.0) / self.dim)
        )
        pe = jnp.zeros((2 * self.max_length - 1, self.dim), dtype=jnp.float32)
        pe = pe.at[:, 0::2].set(jnp.sin(positions * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(positions * div_term))
        self.pe = pe

    def __call__(self, length: int, dtype: jnp.dtype) -> jnp.ndarray:
        center = self.pe.shape[0] // 2
        start = center - length + 1
        end = center + length
        return self.pe[start:end][None, ...].astype(dtype)


class RelPositionMultiHeadAttention(nn.Module):
    dim: int
    num_heads: int
    dropout: float = 0.0

    def setup(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} must be divisible by num_heads={self.num_heads}")
        self.head_dim = self.dim // self.num_heads

        def dense(*, use_bias: bool = True) -> nn.Dense:
            return nn.Dense(
                self.dim,
                use_bias=use_bias,
                kernel_init=nn.initializers.xavier_uniform(),
            )

        self.query = dense()
        self.key = dense()
        self.value = dense()
        self.pos_proj = dense(use_bias=False)
        self.out_proj = dense()
        self.pos_bias_u = self.param(
            "pos_bias_u",
            nn.initializers.zeros,
            (self.num_heads, self.head_dim),
        )
        self.pos_bias_v = self.param(
            "pos_bias_v",
            nn.initializers.zeros,
            (self.num_heads, self.head_dim),
        )
        self.attn_dropout = nn.Dropout(rate=self.dropout)

    def _shape(self, x: jnp.ndarray) -> jnp.ndarray:
        batch, time, _ = x.shape
        return x.reshape(batch, time, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    @staticmethod
    def _relative_shift(x: jnp.ndarray) -> jnp.ndarray:
        batch, heads, time_q, time_k = x.shape
        zero_pad = jnp.zeros((batch, heads, time_q, 1), dtype=x.dtype)
        x = jnp.concatenate([zero_pad, x], axis=-1)
        x = x.reshape(batch, heads, time_k + 1, time_q)
        x = x[:, :, 1:, :].reshape(batch, heads, time_q, time_k)
        return x

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        pos: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        query = self._shape(self.query(x))
        key = self._shape(self.key(x))
        value = self._shape(self.value(x))
        pos_proj = self._shape(self.pos_proj(pos))

        query_u = query + self.pos_bias_u[None, :, None, :]
        query_v = query + self.pos_bias_v[None, :, None, :]

        content_scores = jnp.einsum("bhtd,bhsd->bhts", query_u, key)
        position_scores = jnp.einsum("bhtd,bhsd->bhts", query_v, pos_proj)
        position_scores = self._relative_shift(position_scores)
        position_scores = position_scores[..., : content_scores.shape[-1]]

        scores = (content_scores + position_scores) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = jnp.where(mask[:, None, :, :], scores, jnp.finfo(scores.dtype).min)

        attn = nn.softmax(scores, axis=-1)
        attn = self.attn_dropout(attn, deterministic=deterministic)
        context = jnp.einsum("bhts,bhsd->bhtd", attn, value)
        context = context.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.dim)
        return self.out_proj(context)


class MultiHeadAttention(nn.Module):
    dim: int
    num_heads: int
    dropout: float = 0.0

    def setup(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} must be divisible by num_heads={self.num_heads}")
        self.head_dim = self.dim // self.num_heads

        def dense() -> nn.Dense:
            return nn.Dense(
                self.dim,
                use_bias=True,
                kernel_init=nn.initializers.xavier_uniform(),
            )

        self.query = dense()
        self.key = dense()
        self.value = dense()
        self.out_proj = dense()
        self.attn_dropout = nn.Dropout(rate=self.dropout)

    def _shape(self, x: jnp.ndarray) -> jnp.ndarray:
        batch, time, _ = x.shape
        return x.reshape(batch, time, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        query = self._shape(self.query(x))
        key = self._shape(self.key(x))
        value = self._shape(self.value(x))
        scores = jnp.einsum("bhtd,bhsd->bhts", query, key) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = jnp.where(mask[:, None, :, :], scores, jnp.finfo(scores.dtype).min)
        attn = nn.softmax(scores, axis=-1)
        attn = self.attn_dropout(attn, deterministic=deterministic)
        context = jnp.einsum("bhts,bhsd->bhtd", attn, value)
        context = context.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.dim)
        return self.out_proj(context)


class FeedForwardModule(nn.Module):
    dim: int
    expansion_factor: int = 4
    dropout: float = 0.1
    residual_factor: float = 1.0
    adaptive_scale: bool = True

    def setup(self) -> None:
        self.input_transform = (
            ScaleBiasLayer(self.dim) if self.adaptive_scale else nn.LayerNorm(epsilon=1e-5)
        )
        hidden_dim = self.expansion_factor * self.dim
        self.linear1 = nn.Dense(hidden_dim, kernel_init=nn.initializers.xavier_uniform())
        self.linear2 = nn.Dense(self.dim, kernel_init=nn.initializers.xavier_uniform())
        self.dropout1 = nn.Dropout(rate=self.dropout)
        self.dropout2 = nn.Dropout(rate=self.dropout)

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        residual = x
        x = self.input_transform(x)
        x = self.linear1(x)
        x = nn.silu(x)
        x = self.dropout1(x, deterministic=deterministic)
        x = self.linear2(x)
        x = self.dropout2(x, deterministic=deterministic)
        return residual + self.residual_factor * x


class AttentionModule(nn.Module):
    dim: int
    num_heads: int
    dropout: float = 0.1
    adaptive_scale: bool = True
    attention_backend: str = "relative"

    def setup(self) -> None:
        self.input_transform = (
            ScaleBiasLayer(self.dim) if self.adaptive_scale else nn.LayerNorm(epsilon=1e-5)
        )
        if self.attention_backend == "relative":
            self.attn = RelPositionMultiHeadAttention(
                dim=self.dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )
        elif self.attention_backend == "flash":
            self.attn = MultiHeadAttention(
                dim=self.dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )
        else:
            raise ValueError(f"Unsupported attention backend: {self.attention_backend}")
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        pos: jnp.ndarray,
        mask: jnp.ndarray | None,
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        residual = x
        x = self.input_transform(x)
        if self.attention_backend == "relative":
            x = self.attn(x, pos=pos, mask=mask, deterministic=deterministic)
        else:
            x = self.attn(x, mask=mask, deterministic=deterministic)
        x = self.dropout_layer(x, deterministic=deterministic)
        return residual + x


class ConvolutionModule(nn.Module):
    dim: int
    kernel_size: int = 31
    expansion_factor: int = 2
    dropout: float = 0.1
    adaptive_scale: bool = True

    def setup(self) -> None:
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length depthwise convolution")
        self.hidden_dim = self.dim * self.expansion_factor
        self.input_transform = (
            ScaleBiasLayer(self.dim) if self.adaptive_scale else nn.LayerNorm(epsilon=1e-5)
        )
        self.pointwise_in = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(1,),
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.depthwise = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.kernel_size,),
            padding="SAME",
            feature_group_count=self.hidden_dim,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.batch_norm = nn.BatchNorm(momentum=0.9, epsilon=1e-5)
        self.pointwise_out = nn.Conv(
            features=self.dim,
            kernel_size=(1,),
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        pad_mask: jnp.ndarray | None = None,
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        residual = x
        x = self.input_transform(x)
        x = self.pointwise_in(x)
        x = nn.silu(x)
        if pad_mask is not None:
            x = x * pad_mask[..., None].astype(x.dtype)
        x = self.depthwise(x)
        x = self.batch_norm(x, use_running_average=deterministic)
        x = nn.silu(x)
        x = self.pointwise_out(x)
        x = self.dropout_layer(x, deterministic=deterministic)
        return residual + x


class MHSAFFModule(nn.Module):
    dim: int
    num_heads: int
    ff_expansion_factor: int
    dropout: float
    attention_backend: str

    def setup(self) -> None:
        self.attn = AttentionModule(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            adaptive_scale=True,
            attention_backend=self.attention_backend,
        )
        self.mid_norm = nn.LayerNorm(epsilon=1e-5)
        self.ff = FeedForwardModule(
            dim=self.dim,
            expansion_factor=self.ff_expansion_factor,
            dropout=self.dropout,
            residual_factor=1.0,
            adaptive_scale=True,
        )
        self.out_norm = nn.LayerNorm(epsilon=1e-5)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        pos: jnp.ndarray,
        attn_mask: jnp.ndarray | None,
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        x = self.attn(x, pos=pos, mask=attn_mask, deterministic=deterministic)
        x = self.mid_norm(x)
        x = self.ff(x, deterministic=deterministic)
        return self.out_norm(x)


class ConvFFModule(nn.Module):
    dim: int
    kernel_size: int
    conv_expansion_factor: int
    ff_expansion_factor: int
    dropout: float

    def setup(self) -> None:
        self.conv = ConvolutionModule(
            dim=self.dim,
            kernel_size=self.kernel_size,
            expansion_factor=self.conv_expansion_factor,
            dropout=self.dropout,
            adaptive_scale=True,
        )
        self.mid_norm = nn.LayerNorm(epsilon=1e-5)
        self.ff = FeedForwardModule(
            dim=self.dim,
            expansion_factor=self.ff_expansion_factor,
            dropout=self.dropout,
            residual_factor=1.0,
            adaptive_scale=True,
        )
        self.out_norm = nn.LayerNorm(epsilon=1e-5)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        pad_mask: jnp.ndarray | None,
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        x = self.conv(x, pad_mask=pad_mask, deterministic=deterministic)
        x = self.mid_norm(x)
        x = self.ff(x, deterministic=deterministic)
        return self.out_norm(x)


class SqueezeformerBlock(nn.Module):
    dim: int
    num_heads: int
    kernel_size: int
    ff_expansion_factor: int
    conv_expansion_factor: int
    dropout: float
    block_pattern: tuple[str, ...]
    attention_backend: str
    drop_path_rate: float = 0.0

    def setup(self) -> None:
        layers: list[nn.Module] = []
        for token in self.block_pattern:
            if token == "M":
                layers.append(
                    MHSAFFModule(
                        dim=self.dim,
                        num_heads=self.num_heads,
                        ff_expansion_factor=self.ff_expansion_factor,
                        dropout=self.dropout,
                        attention_backend=self.attention_backend,
                    )
                )
            elif token == "C":
                layers.append(
                    ConvFFModule(
                        dim=self.dim,
                        kernel_size=self.kernel_size,
                        conv_expansion_factor=self.conv_expansion_factor,
                        ff_expansion_factor=self.ff_expansion_factor,
                        dropout=self.dropout,
                    )
                )
            elif token == "s":
                layers.append(IdentityLayer())
            else:
                raise ValueError(f"Unsupported block token: {token}")
        self.layers = tuple(layers)

    def _drop_path(
        self,
        x: jnp.ndarray,
        residual: jnp.ndarray,
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        if deterministic or self.drop_path_rate <= 0:
            return x
        keep_prob = 1.0 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + jax.random.uniform(
            self.make_rng("dropout"),
            shape=shape,
            dtype=x.dtype,
        )
        binary_mask = jnp.floor(random_tensor)
        return residual + (x - residual) * binary_mask / keep_prob

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        pos: jnp.ndarray,
        attn_mask: jnp.ndarray | None,
        pad_mask: jnp.ndarray | None,
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        residual = x
        for token, layer in zip(self.block_pattern, self.layers, strict=True):
            if token == "M":
                x = layer(x, pos=pos, attn_mask=attn_mask, deterministic=deterministic)
            elif token == "C":
                x = layer(x, pad_mask=pad_mask, deterministic=deterministic)
        if self.drop_path_rate > 0:
            x = self._drop_path(x, residual, deterministic=deterministic)
        return x


class Conv2dSubsampling(nn.Module):
    in_features: int
    channels: int
    depthwise_separable: bool = True

    def setup(self) -> None:
        self.conv1 = nn.Conv(
            features=self.channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            kernel_init=nn.initializers.kaiming_normal(),
        )
        if self.depthwise_separable:
            self.conv2_dw = nn.Conv(
                features=self.channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                feature_group_count=self.channels,
                kernel_init=nn.initializers.kaiming_normal(),
            )
            self.conv2_pw = nn.Conv(
                features=self.channels,
                kernel_size=(1, 1),
                padding="VALID",
                kernel_init=nn.initializers.kaiming_normal(),
            )
        else:
            self.conv2 = nn.Conv(
                features=self.channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                kernel_init=nn.initializers.kaiming_normal(),
            )
        self.output_dim = self._output_freq_bins(self.in_features) * self.channels

    def _output_freq_bins(self, features: int) -> int:
        value = features
        for _ in range(2):
            value = math.floor((value - 1) / 2) + 1
        return value

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x[..., None]
        x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
        x = nn.relu(self.conv1(x))
        x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
        if self.depthwise_separable:
            x = self.conv2_pw(self.conv2_dw(x))
        else:
            x = self.conv2(x)
        x = nn.relu(x)
        batch, time, freq, channels = x.shape
        return x.reshape(batch, time, freq * channels)


class TimeReductionLayer(nn.Module):
    dim: int
    kernel_size: int = 3
    stride: int = 2

    def setup(self) -> None:
        self.depthwise = nn.Conv(
            features=self.dim,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding="VALID",
            feature_group_count=self.dim,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.pointwise = nn.Conv(
            features=self.dim,
            kernel_size=(1,),
            padding="VALID",
            kernel_init=nn.initializers.kaiming_normal(),
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, lengths: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        pad_mask = _make_pad_mask(lengths, max_length=x.shape[1])
        x = x * pad_mask[..., None].astype(x.dtype)
        x = jnp.pad(x, ((0, 0), (0, max(0, self.kernel_size - self.stride)), (0, 0)))
        x = self.depthwise(x)
        x = self.pointwise(x)
        lengths = lengths // self.stride
        return x, lengths


class TimeRecoveryLayer(nn.Module):
    dim: int
    stride: int = 2

    def setup(self) -> None:
        self.proj = nn.Dense(self.dim, kernel_init=nn.initializers.xavier_uniform())

    @nn.compact
    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray) -> jnp.ndarray:
        x = jnp.repeat(x, repeats=self.stride, axis=1)
        target_length = skip.shape[1]
        if x.shape[1] < target_length:
            pad_length = target_length - x.shape[1]
            x = jnp.pad(x, ((0, 0), (0, pad_length), (0, 0)), mode="edge")
        else:
            x = x[:, :target_length, :]
        return self.proj(x) + skip


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
    stochastic_depth_rate: float = 0.0
    time_reduce_idx: tuple[int, ...] = (7,)
    time_recover_idx: tuple[int, ...] = (15,)
    attention_backend: str = "relative"


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
    config: SqueezeformerConfig

    def setup(self) -> None:
        config = self.config
        self.subsampling = Conv2dSubsampling(
            in_features=config.input_features,
            channels=config.d_model,
            depthwise_separable=config.depthwise_subsampling,
        )
        self.input_projection = nn.Dense(
            config.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
        )
        self.input_norm = nn.LayerNorm(epsilon=1e-5)
        self.dropout = nn.Dropout(rate=config.dropout)
        self.positional_encoding = RelativePositionalEncoding(config.d_model)
        self.time_reduce = {
            index: TimeReductionLayer(
                config.d_model,
                kernel_size=config.time_reduction_kernel_size,
            )
            for index in config.time_reduce_idx
        }
        self.time_recover = {
            index: TimeRecoveryLayer(config.d_model) for index in config.time_recover_idx
        }
        self.blocks = tuple(
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
                    config.stochastic_depth_rate * index / max(1, config.num_layers - 1)
                ),
            )
            for index in range(config.num_layers)
        )

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        lengths: jnp.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if features.ndim != 3:
            raise ValueError(
                f"Expected features with shape [batch, time, mel], got {features.shape}"
            )

        x = self.subsampling(features)
        lengths = _subsample_length(lengths)
        x = self.input_projection(x)
        x = x * math.sqrt(self.config.d_model)
        x = self.dropout(x, deterministic=deterministic)
        x = self.input_norm(x)

        recover_stack: list[tuple[jnp.ndarray, jnp.ndarray]] = []
        for layer_index, block in enumerate(self.blocks):
            if layer_index in self.config.time_reduce_idx:
                recover_stack.append((x, lengths))
                x, lengths = self.time_reduce[layer_index](x, lengths)

            if layer_index in self.config.time_recover_idx:
                if not recover_stack:
                    raise RuntimeError("Encountered a recovery layer without a matching reduction")
                skip_x, skip_lengths = recover_stack.pop()
                x = self.time_recover[layer_index](x, skip_x)
                lengths = skip_lengths

            pos = self.positional_encoding(x.shape[1], x.dtype)
            attn_mask = _make_attn_mask(lengths, max_length=x.shape[1])
            pad_mask = _make_pad_mask(lengths, max_length=x.shape[1])
            x = block(
                x,
                pos=pos,
                attn_mask=attn_mask,
                pad_mask=pad_mask,
                deterministic=deterministic,
            )

        return x, lengths


class SqueezeformerCTC(nn.Module):
    encoder_config: SqueezeformerConfig
    vocab_size: int

    def setup(self) -> None:
        self.encoder = SqueezeformerEncoder(self.encoder_config)
        self.classifier = nn.Dense(
            self.vocab_size,
            kernel_init=nn.initializers.xavier_uniform(),
        )

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        feature_lengths: jnp.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        encoded, output_lengths = self.encoder(
            features,
            feature_lengths,
            deterministic=deterministic,
        )
        return self.classifier(encoded), output_lengths

    def log_probs(
        self,
        features: jnp.ndarray,
        feature_lengths: jnp.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        logits, output_lengths = self(
            features,
            feature_lengths,
            deterministic=deterministic,
        )
        return nn.log_softmax(logits), output_lengths


def build_squeezeformer_encoder(
    variant: str = "sm",
    **overrides: int | float | tuple[int, ...] | bool | str,
) -> SqueezeformerEncoder:
    config = replace(squeezeformer_variant(variant), **overrides)
    return SqueezeformerEncoder(config)
