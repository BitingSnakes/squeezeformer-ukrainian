from __future__ import annotations

from typing import Any


def _load_extension():
    try:
        import feature_cache_warmer_rust
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "feature_cache_warmer_rust is not installed. Build it with "
            "`cd rust_feature_cache_warmer && maturin develop --features python --release` "
            "or install the maturin-built wheel."
        ) from error
    return feature_cache_warmer_rust


def extract_squeezeformer(waveform: Any, sample_rate: int, **kwargs: Any):
    return _load_extension().extract_squeezeformer(waveform, sample_rate, **kwargs)


def extract_zipformer(waveform: Any, sample_rate: int, **kwargs: Any):
    return _load_extension().extract_zipformer(waveform, sample_rate, **kwargs)


def extract_w2v_bert(waveform: Any, sample_rate: int, **kwargs: Any):
    return _load_extension().extract_w2v_bert(waveform, sample_rate, **kwargs)


__all__ = [
    "extract_squeezeformer",
    "extract_w2v_bert",
    "extract_zipformer",
]
