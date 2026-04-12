from __future__ import annotations

import pytest


def test_rust_feature_extension_extracts_numpy_features() -> None:
    pytest.importorskip("feature_cache_warmer_rust")
    import numpy as np

    from feature_cache_warmer.rust_features import (
        extract_squeezeformer,
        extract_w2v_bert,
        extract_zipformer,
    )

    waveform = np.sin(np.arange(16_000, dtype=np.float32) * 0.01).astype(np.float32)

    squeezeformer = extract_squeezeformer(waveform, 16_000)
    zipformer = extract_zipformer(waveform, 16_000)
    w2v_bert = extract_w2v_bert(waveform, 16_000)

    assert squeezeformer.dtype == np.float32
    assert squeezeformer.shape[1] == 80
    assert zipformer.dtype == np.float32
    assert zipformer.shape[1] == 80
    assert w2v_bert.dtype == np.float32
    assert w2v_bert.shape[1] == 160
