# Ukrainian ASR Training Toolkit

This repository contains PyTorch training, evaluation, inference, and export tooling for
Ukrainian ASR experiments on `speech-uk/cv22` and compatible Common Voice-style datasets.

The default model is a standalone Squeezeformer CTC implementation based on
[arXiv:2206.00888v2](https://arxiv.org/abs/2206.00888v2). The same data and checkpointing
pipeline also supports Zipformer, Zipformer transducer, and W2V-BERT fine-tuning paths.

## Contents

- [squeezeformer_pytorch/model.py](squeezeformer_pytorch/model.py): Squeezeformer encoder
  and published size variants
- [squeezeformer_pytorch/asr.py](squeezeformer_pytorch/asr.py): CTC wrapper, tokenizers,
  beam search, AED and distillation hooks
- [zipformer_pytorch](zipformer_pytorch): Zipformer CTC and transducer models
- [w2v_bert](w2v_bert): W2V-BERT CTC fine-tuning wrapper
- [squeezeformer_pytorch/data.py](squeezeformer_pytorch/data.py): dataset download,
  manifests, transcript normalization, featurization, caching, and bucketing
- [squeezeformer_pytorch/lm.py](squeezeformer_pytorch/lm.py): character n-gram
  shallow-fusion LM
- [train.py](train.py): training entrypoint
- [evaluate.py](evaluate.py): checkpoint evaluation entrypoint
- [inference.py](inference.py): single-file transcription and Gradio ASR UI
- [extract_features.py](extract_features.py): offline log-mel feature cache warming
- [train_lm.py](train_lm.py): shallow-fusion LM training
- [quantize.py](quantize.py): TorchAO int8 weight-only checkpoint conversion
- [benchmark.py](benchmark.py): synthetic throughput, memory, and decoding benchmark
- [hparam_tuner.py](hparam_tuner.py): hardware-sensitive command estimator
- [tests](tests): architecture, data loading, CLI, evaluation, inference, and training checks

## Model Options

Squeezeformer is the default architecture:

```bash
uv run python train.py --variant sm --output-dir artifacts/squeezeformer-sm
```

Available Squeezeformer variants:

| Variant | Layers | Dim | Heads |
| --- | ---: | ---: | ---: |
| `xs` | 16 | 144 | 4 |
| `s` | 18 | 196 | 4 |
| `sm` | 16 | 256 | 4 |
| `m` | 20 | 324 | 4 |
| `ml` | 18 | 512 | 8 |
| `l` | 22 | 640 | 8 |

The Squeezeformer encoder includes the main paper-aligned pieces used by this repo:
depthwise-separable 4x acoustic subsampling, MF/CF block ordering, scaled post-LN
residual modules, relative-position attention, Temporal U-Net reduction/recovery, and
published size presets.

Other training paths:

```bash
# Zipformer CTC
uv run python train.py --zipformer --variant sm --output-dir artifacts/zipformer-sm

# Zipformer transducer
uv run python train.py \
  --zipformer \
  --zipformer-transducer \
  --variant sm \
  --output-dir artifacts/zipformer-transducer-sm

# W2V-BERT fine-tuning
uv run python train.py --w2v-bert --output-dir artifacts/w2v-bert
```

W2V-BERT uses `facebook/w2v-bert-2.0` by default. Use
`--w2v-bert-model-path /path/to/snapshot` for a local Hugging Face snapshot.

## Setup

The project targets Python 3.12 and is designed for `uv`.

CPU or general development setup:

```bash
uv sync --extra train --extra quantize --group dev
```

Training, evaluation, inference, and `extract_features.py` use the Rust feature
frontend through the local PyO3 package `asr-features`. `uv sync`
builds it from [rust_feature_cache_warmer](rust_feature_cache_warmer), so the
environment needs a Rust toolchain. To rebuild only the extension in an active
environment:

```bash
cd rust_feature_cache_warmer
maturin develop --features python --release
```

CUDA setup:

```bash
# Choose exactly one CUDA group per environment.
uv sync --extra train --group cu126
uv sync --extra train --group cu12
uv sync --extra train --group cu13
```

CUDA group intent:

- `cu126`: PyTorch 2.10 CUDA 12.6 wheels for hosts with older R560-series drivers
- `cu12`: PyTorch 2.11 CUDA 12.8 wheels and CUDA 12 NPP
- `cu13`: PyTorch 2.11 CUDA 13.0 wheels and CUDA 13 NPP

Optional dependency groups:

- `.[train]`: training, metrics, Polars, Trackio, and Transformers
- `.[fp8]`: Transformer Engine FP8 support
- `.[quantize]`: TorchAO post-training quantization

## Dataset Access

The default dataset is the gated Hugging Face dataset `speech-uk/cv22`.

Before training or evaluation:

1. Accept the dataset terms at `https://huggingface.co/datasets/speech-uk/cv22`.
2. Export a token with access to the dataset:

```bash
export HF_TOKEN=your_huggingface_token
```

The loader downloads snapshots with `huggingface_hub.snapshot_download()` and reads TSV or
Parquet manifests through Polars. Local Common Voice-style directories are also supported:

```bash
uv run python train.py --dataset-repo /path/to/local/commonvoice
```

To train from multiple sources, repeat `--dataset-source`. When this flag is present,
`train.py` ignores `--dataset-repo` and uses the source list.

```bash
HF_TOKEN=... uv run python train.py \
  --dataset-source speech-uk/cv22 \
  --dataset-source /mnt/extra_commonvoice \
  --output-dir artifacts/multi-source
```

For large multi-source runs, records are streamed into JSONL indexes under
`OUTPUT_DIR/record_cache` or `--record-cache-dir`. Do not combine this disk-backed path
with `--prevalidate-audio`; that mode is intentionally blocked because it requires an
in-memory split.

The upstream dataset exposes a source train split. This repo creates deterministic internal
`train`, `validation`, and `test` splits from that data. Defaults are:

- `--val-fraction 0.1`
- `--test-fraction 0.1`

That yields an effective `80 / 10 / 10` split. If speaker metadata is available, splitting
is speaker-aware; otherwise it falls back to utterance-level hashing and records the missing
speaker identity in the split audit.

## Quick Start

Run a small CPU smoke test:

```bash
HF_TOKEN=... uv run python train.py \
  --variant xs \
  --device cpu \
  --dtype bfloat16 \
  --output-dir artifacts/smoke \
  --epochs 1 \
  --max-train-samples 64 \
  --max-val-samples 16
```

Estimate runtime-sensitive values for a larger command:

```bash
uv run python hparam_tuner.py \
  --variant sm \
  --optimizer muon \
  --tokenizer sentencepiece \
  --spm-vocab-size 128 \
  --device cpu \
  --dtype bfloat16 \
  --feature-cache-dir artifacts/feature_cache \
  --decode-strategy beam \
  --beam-size 8 \
  --output-dir artifacts/cv22-sm \
  --epochs 10 \
  --emit-format both
```

Train with caching, bucketing, augmentation, and beam-search validation:

```bash
HF_TOKEN=... uv run python train.py \
  --variant sm \
  --optimizer muon \
  --tokenizer sentencepiece \
  --spm-vocab-size 128 \
  --device cpu \
  --dtype bfloat16 \
  --gradient-accumulation-steps 4 \
  --feature-cache-dir artifacts/feature_cache \
  --max-batch-frames 12000 \
  --speed-perturb-prob 0.5 \
  --noise-prob 0.2 \
  --reverb-prob 0.1 \
  --decode-strategy beam \
  --beam-size 8 \
  --output-dir artifacts/cv22-sm \
  --epochs 10
```

Resume training:

```bash
HF_TOKEN=... uv run python train.py \
  --resume artifacts/cv22-sm/checkpoint_last.pt \
  --output-dir artifacts/cv22-sm
```

Distributed training:

```bash
HF_TOKEN=... uv run squeezeformer-torchrun --nproc_per_node=2 train.py \
  --distributed \
  --variant sm \
  --output-dir artifacts/cv22-sm-ddp
```

The launcher sets `OMP_NUM_THREADS` automatically when it is unset. Export
`OMP_NUM_THREADS` before launch if you want a specific value.

## Training Features

Important defaults:

- `--optimizer muon`
- `--dtype bfloat16`
- `--tokenizer sentencepiece`
- `--spm-vocab-size 128`
- `--warmup-epochs 20`
- `--hold-epochs 160`
- `--decay-exponent 1.0`
- `--bucket-by-length`
- `--metadata-workers 4`
- `--pin-memory`
- `--persistent-workers`
- `--ema-decay 0.0`

Training supports:

- Muon for encoder 2D hidden weights, with AdamW for the remaining parameters
- separate Muon and AdamW LR, weight decay, and scheduler overrides
- SentencePiece or character tokenization
- gradient accumulation and clipping
- optional EMA checkpoints and EMA validation
- optional `torch.compile`
- optional activation checkpointing inside encoder blocks
- on-disk feature caching
- length bucketing and dynamic batching by frames, duration, or transcript tokens
- audio metadata materialization and optional prevalidation
- transcript and duration filtering
- greedy or beam-search validation decoding
- optional AED loss for Squeezeformer
- optional LiBERTa transcript distillation for Squeezeformer
- optional wav2vec2-BERT audio-teacher distillation for Squeezeformer
- optional training-time shallow-fusion LM fitting
- top-k checkpoint retention and checkpoint averaging
- per-checkpoint JSON evaluation reports
- hardest and random decoded example logging in Trackio
- WER/CER by utterance-length bucket and speaker-level aggregates
- single-node distributed training through `torchrun`

Variant-aware Squeezeformer defaults:

- `xs`, `s`, `sm`: peak LR `2e-3`, time masks `5`
- `m`: peak LR `1.5e-3`, time masks `7`
- `ml`, `l`: peak LR `1e-3`, time masks `10`

## Tokenization

Supported tokenizer modes:

- `sentencepiece`
- `character`

Default training behavior:

- `--tokenizer sentencepiece`
- `--spm-vocab-size 128`
- `--spm-model-type unigram`

SentencePiece runs write `tokenizer.json` and `tokenizer.model`. Evaluation and inference
reload tokenizer settings from checkpoint metadata.

## Audio Frontend

The default frontend produces 80-bin log-mel features with pre-emphasis, waveform
normalization, and feature normalization.

Common frontend and augmentation flags:

- `--n-mels`
- `--preemphasis`
- `--normalize-signal/--no-normalize-signal`
- `--normalize-feature/--no-normalize-feature`
- `--normalize-per-frame/--no-normalize-per-frame`
- `--num-freq-masks`
- `--freq-mask-param`
- `--num-time-masks`
- `--time-mask-max-ratio`
- `--speed-perturb-prob`
- `--speed-factors`
- `--noise-prob`
- `--noise-snr-db-min`
- `--noise-snr-db-max`
- `--reverb-prob`

## Feature Caching

Training, evaluation, and offline extraction can write reusable frontend tensors to disk:

- training writes under `FEATURE_CACHE_DIR/train`
- validation writes under `FEATURE_CACHE_DIR/validation`
- evaluation writes under `FEATURE_CACHE_DIR/<split>`
- offline extraction writes under `FEATURE_CACHE_DIR/<split>`

Each cached file is a `.pt` tensor keyed by utterance id and frontend config hash.

Warm a small cache during training:

```bash
HF_TOKEN=... uv run python train.py \
  --variant xs \
  --device cpu \
  --dtype bfloat16 \
  --output-dir artifacts/cache_warm \
  --epochs 1 \
  --feature-cache-dir artifacts/feature_cache \
  --max-train-samples 1000 \
  --max-val-samples 200
```

Extract features without training:

```bash
HF_TOKEN=... uv run python extract_features.py \
  --dataset-repo speech-uk/cv22 \
  --split train \
  --feature-cache-dir artifacts/feature_cache
```

Cache notes:

- cache reuse is disabled for training samples when waveform augmentation is enabled
- changing frontend settings produces a new cache key
- the cache stores post-featurization model inputs, not encoder embeddings
- offline extraction streams split selection from manifests instead of building the full
  record list in memory

## Evaluation

Evaluate a checkpoint:

```bash
HF_TOKEN=... uv run python evaluate.py \
  --checkpoint artifacts/cv22-sm/checkpoint_best.pt \
  --split test \
  --device cpu \
  --dtype bfloat16 \
  --decode-strategy beam \
  --beam-size 8
```

`evaluate.py` prints and logs:

- loss
- CER and WER
- per-bucket CER/WER for `short`, `medium`, and `long` utterances
- speaker-level aggregate metrics
- sample count
- hardest and random decoded example pairs

## Inference

Transcribe one file with the default Hugging Face checkpoint:

```bash
uv run python inference.py --audio path/to/audio.wav
```

Launch the local Gradio app:

```bash
uv run python inference.py --gradio
```

Launch the app with an explicit checkpoint:

```bash
uv run python inference.py \
  --gradio \
  --checkpoint speech-uk/squeezeformer-bf16-lm-sm-moredata \
  --host 0.0.0.0 \
  --port 7860
```

Useful inference flags:

- `--checkpoint`: local `.pt` or `.safetensors` checkpoint path, Hugging Face checkpoint URL,
  or Hugging Face repo id
- `--checkpoint-metadata`: optional JSON sidecar path for `.safetensors`
- `--device`: inference device such as `cpu` or `cuda:0`
- `--dtype`: `float32`, `bfloat16`, `float16`, or `fp8`
- `--host`: Gradio bind host, default `127.0.0.1`
- `--port`: Gradio bind port, default `7860`
- `--share`: create a public Gradio share link

When a Hugging Face repo id is provided, inference looks for `checkpoint_best.pt`, then
falls back to `checkpoint_best.safetensors` plus its metadata sidecar.

## Decoding And Shallow Fusion

Training-time validation and `evaluate.py` support:

- `--decode-strategy greedy|beam`
- `--beam-size`
- `--beam-length-bonus`
- `--lm-scorer`
- `--lm-weight`

`--lm-scorer` expects `module:function` and returns a scalar score for a decoded prefix.

Train and save the bundled character n-gram LM during ASR training:

```bash
HF_TOKEN=... uv run python train.py \
  --output-dir artifacts/cv22-sm \
  --fit-shallow-fusion-lm \
  --decode-strategy beam \
  --beam-size 8 \
  --lm-weight 0.2
```

Train it from a text corpus:

```bash
uv run python train_lm.py \
  --corpus data/lm_corpus.txt \
  --output artifacts/shallow_fusion_lm.json \
  --order 3 \
  --alpha 0.1
```

Train it directly from dataset transcripts:

```bash
HF_TOKEN=... uv run python train_lm.py \
  --dataset-repo speech-uk/cv22 \
  --output artifacts/shallow_fusion_lm.json \
  --deduplicate \
  --order 3 \
  --alpha 0.1
```

Use a saved LM explicitly:

```bash
uv run python evaluate.py \
  --checkpoint artifacts/cv22-sm/checkpoint_best.pt \
  --decode-strategy beam \
  --lm-scorer squeezeformer_pytorch.lm:load_saved_ngram_scorer:artifacts/cv22-sm/shallow_fusion_lm.json \
  --lm-weight 0.2
```

## Quantization

Use [quantize.py](quantize.py) to convert a checkpoint into a TorchAO int8 weight-only
checkpoint:

```bash
uv run python quantize.py \
  --checkpoint artifacts/cv22-sm/checkpoint_best.pt \
  --output artifacts/cv22-sm/checkpoint_best.torchao-int8.safetensors
```

Run inference with the quantized checkpoint:

```bash
uv run python inference.py \
  --checkpoint artifacts/cv22-sm/checkpoint_best.torchao-int8.safetensors \
  --audio path/to/audio.wav \
  --device cpu \
  --dtype float32
```

Quantization notes:

- quantized checkpoints use TorchAO `Int8WeightOnlyConfig`
- `.safetensors` checkpoints store weights in the safetensors file and quantization metadata
  in the adjacent `.json` sidecar
- [inference.py](inference.py) detects TorchAO checkpoints automatically
- TorchAO quantized checkpoints do not support `--dtype fp8`
- quantization changes the inference format; it does not preserve Transformer Engine FP8 modules

## Checkpoints

Training writes:

- `checkpoint_last.pt`
- `checkpoint_last.safetensors` plus `checkpoint_last.json`
- `checkpoint_best.pt`
- `checkpoint_best.safetensors` plus `checkpoint_best.json`
- `checkpoint_topk_avg.pt`
- `checkpoint_topk_avg.safetensors` plus `checkpoint_topk_avg.json`
- `checkpoints_topk/`
- `checkpoints_topk/metadata.json`
- `eval_reports/epoch_XXXX.json`
- `split_audit.json`
- `tokenizer.json`
- `tokenizer.model` for SentencePiece runs
- `train_summary.json`

Resume restores model weights, optimizer state, scheduler state, grad scaler state, EMA
state, epoch, global step, and best validation WER.

## Trackio

Training and evaluation log metrics to Trackio.

Local dashboard:

```bash
trackio show --project "squeezeformer-cv22"
```

Shared dashboard:

```bash
GRADIO_SHARE=True uv run python -c "import trackio; trackio.show()"
```

To sync logs to a Hugging Face Space, pass:

```bash
--trackio-space-id username/space-name
```

## Development Checks

Run the local checks:

```bash
uv run ruff check .
uv run pytest -q
uv run python -c "import train, evaluate; print('imports_ok')"
```

Benchmark synthetic runtime behavior:

```bash
uv run python benchmark.py \
  --variant sm \
  --batch-size 8 \
  --time-steps 512 \
  --decode-strategy beam \
  --beam-size 8
```

Run the local resumable training smoke test:

```bash
uv run python scripts/smoke_resume_train.py
```

## Current Limits

- This is not the original large-scale Squeezeformer paper training environment.
- Distributed training is basic single-node `torchrun` support, not a full multi-node recipe.
- The bundled LM is a simple character n-gram shallow-fusion scorer, not a full external LM
  training stack.
- Live gated-dataset behavior still depends on the current Hugging Face dataset contents and
  accepted access terms.

## Sources

- Squeezeformer paper: [arXiv:2206.00888v2](https://arxiv.org/abs/2206.00888v2)
- Dataset card: `https://huggingface.co/datasets/speech-uk/cv22`
- Trackio: `https://pypi.org/project/trackio/`
