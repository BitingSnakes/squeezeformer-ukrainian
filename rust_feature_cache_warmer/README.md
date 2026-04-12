# ASR features

This crate reads dataset parquet manifest files, decodes audio with Symphonia, computes
log-mel features with RustFFT, resamples non-16 kHz audio with Rubato, and writes
the sharded parquet cache layout consumed by `ShardedParquetFeatureCache`.
Ogg/Vorbis is enabled through Symphonia's Ogg demuxer support. If Symphonia
cannot decode a codec such as Opus, the CLI falls back to `ffmpeg`; pass
`--no-ffmpeg-fallback` to disable that.

Example:

```bash
cargo run --release --manifest-path rust_feature_cache_warmer/Cargo.toml -- \
  --input /data/cv22/train.parquet \
  --cache-dir artifacts/feature-cache/train \
  --frontend squeezeformer \
  --threads 8
```

Use `--frontend zipformer` for the paper frontend defaults, or `--frontend w2v-bert`
for the W2V-BERT/SeamlessM4T-style 160-dimensional stacked fbank frontend.
Use `--input-folder /path/to/manifests` instead of `--input` to process every
`.parquet` file under a directory recursively. Repeat `--input-folder` to combine
several manifest roots; relative audio paths are resolved against the folder that
contributed each parquet file unless `--source-base` is set explicitly.

To warm exactly the records selected by the training loader, build the
Python-compatible disk-backed record cache first, then feed a split JSONL file
to the feature warmer:

```bash
cargo run --release --manifest-path rust_feature_cache_warmer/Cargo.toml -- record-cache \
  --record-cache-dir /content/cache-cv-zipformer \
  --dataset-source /content/cv22-opus/ \
  --validation-dataset-source /content/cv10-uk-testset-clean-punctuated/data/ \
  --require-readable-audio

cargo run --release --manifest-path rust_feature_cache_warmer/Cargo.toml -- \
  --input-record-cache /content/cache-cv-zipformer/train.jsonl \
  --cache-dir /content/feature-cache-parquet-cv-zipformer/train \
  --frontend zipformer \
  --threads 26

cargo run --release --manifest-path rust_feature_cache_warmer/Cargo.toml -- \
  --input-record-cache /content/cache-cv-zipformer/validation.jsonl \
  --cache-dir /content/feature-cache-parquet-cv-zipformer/validation \
  --frontend zipformer \
  --threads 26
```

The `record-cache` subcommand mirrors the Python `--record-cache-dir` layout:
`train.jsonl`, `validation.jsonl`, `.offsets.u64`, `.estimated_frames.u32`,
`.num_samples.u64`, `.sample_rates.u32`, `.transcript_lengths.u32`,
`.token_lengths.u32`, plus `<split>_audio_blobs/*.bin` when embedded audio bytes
need to be preserved. It currently targets local TSV/Parquet files or
directories; use the Python loader for Hugging Face repo ids or remote manifest
URLs.

Logging uses `env_logger` and defaults to `info`. Set
`RUST_LOG=asr_features=debug` to include decode, resample, batch, and
shard flush details, or `RUST_LOG=asr_features=trace` for per-row
feature extraction logs:

```bash
RUST_LOG=asr_features=debug cargo run --release --manifest-path rust_feature_cache_warmer/Cargo.toml -- \
  --input /data/cv22/train.parquet \
  --cache-dir artifacts/feature-cache/train
```

The output directory is a split cache root, matching the Python warmer:

```text
artifacts/feature-cache/train/
  feature_shards/
    features_00/
      part_rust_...parquet
```

The parquet row schema is the same as the Python cache (`key`, `payload`,
`deleted`). The payload is a compact Rust-native f32 matrix format; the Python
loader in `squeezeformer_pytorch.data` understands both this payload and legacy
`torch.save` payloads.

## Frontend compatibility

- `squeezeformer` mirrors `AudioFeaturizer()` defaults: 16 kHz audio, `n_fft=400`,
  `win_length=400`, `hop_length=160`, 80 HTK mel bins, pre-emphasis `0.97`,
  signal normalization, and per-bin feature normalization.
- `zipformer` mirrors `zipformer_paper_featurizer_config()`: the same STFT/mel
  layout, but no pre-emphasis and no signal or feature normalization.
- `w2v-bert` mirrors the repository's `W2VBertFeatureExtractor` cache contract
  and the Hugging Face `SeamlessM4TFeatureExtractor` algorithm: 16 kHz audio,
  80-bin Kaldi fbank, Povey window, pre-emphasis `0.97`, per-mel unbiased
  variance normalization, and stride-2 stacking to 160 features.

The cache key is generated with the same Python `repr({"featurizer": ...})`
hash inputs used by `ShardedParquetFeatureCache`, so a matching Python
featurizer will find the Rust-written shard without a manifest sidecar.

## Python extension

The feature extraction code is also exposed as an optional PyO3 extension. Build
and install it into the active Python environment with:

```bash
cd rust_feature_cache_warmer
maturin develop --features python --release
```

The extension module is `asr_features`:

```python
import numpy as np
from asr_features import extract_w2v_bert

waveform = np.zeros(16_000, dtype=np.float32)
features = extract_w2v_bert(waveform, 16_000)
assert features.shape[1] == 160
```

The repository's `build_featurizer_from_config()` factory now returns PyTorch
modules backed by this extension for Squeezeformer, Zipformer, and W2V-BERT.
That factory is used by `train.py`, `evaluate.py`, `inference.py`, and the
Python cache warmers, so actual feature extraction no longer goes through the
torchaudio/Hugging Face frontend path unless a test deliberately monkeypatches
the script-local compatibility aliases.

## Parallelism

Feature extraction runs in a Rayon thread pool. Set `--threads N` to control the
number of decode/extract workers. The default `--threads 0` uses Rayon's default
thread count. Cache shard writes stay on the main thread so parquet parts remain
well-formed and deterministic within each input batch.
