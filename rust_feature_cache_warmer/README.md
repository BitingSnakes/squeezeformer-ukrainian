# Rust feature cache warmer

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
`.parquet` file under a directory recursively.

Logging uses `env_logger` and defaults to `info`. Set
`RUST_LOG=feature_cache_warmer=debug` to include decode, resample, batch, and
shard flush details, or `RUST_LOG=feature_cache_warmer=trace` for per-row
feature extraction logs:

```bash
RUST_LOG=feature_cache_warmer=debug cargo run --release --manifest-path rust_feature_cache_warmer/Cargo.toml -- \
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

## Parallelism

Feature extraction runs in a Rayon thread pool. Set `--threads N` to control the
number of decode/extract workers. The default `--threads 0` uses Rayon's default
thread count. Cache shard writes stay on the main thread so parquet parts remain
well-formed and deterministic within each input batch.
