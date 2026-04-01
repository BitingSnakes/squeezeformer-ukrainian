# Squeezeformer PyTorch

This repository contains a standalone PyTorch implementation of the Squeezeformer encoder from [arXiv-2206.00888v2](/workspace/arXiv-2206.00888v2), plus CTC training and evaluation scripts for the gated Hugging Face dataset `speech-uk/cv22`.

## What Is Here

- [model.py](/workspace/squeezeformer_pytorch/model.py): encoder architecture and published size variants
- [asr.py](/workspace/squeezeformer_pytorch/asr.py): CTC wrapper, tokenizer implementations, beam search, LM scorer hook
- [lm.py](/workspace/squeezeformer_pytorch/lm.py): concrete character n-gram shallow-fusion LM
- [data.py](/workspace/squeezeformer_pytorch/data.py): dataset download, Polars manifest loading, transcript normalization, featurization, caching, bucketing
- [metrics.py](/workspace/squeezeformer_pytorch/metrics.py): CER and WER through `jiwer`
- [train.py](/workspace/train.py): training entrypoint
- [train_lm.py](/workspace/train_lm.py): train a shallow-fusion n-gram LM from a text corpus or dataset transcripts
- [export_cv22_corpus.py](/workspace/export_cv22_corpus.py): export normalized cv22 transcripts as an LM corpus
- [evaluate.py](/workspace/evaluate.py): evaluation entrypoint
- [benchmark.py](/workspace/benchmark.py): synthetic throughput, memory, and decode-speed benchmark
- [tests](/workspace/tests): architecture and training utility checks

## Architecture Fidelity

The encoder implements the paper’s main Squeezeformer changes:

- depthwise-separable 4x acoustic subsampling
- explicit MF/CF block pattern, default `M,s,C,s`
- scaled post-LN residual modules
- relative-position attention
- Temporal U-Net time reduction and recovery
- paper-aligned reduction kernel size `3`
- published size variants: `xs`, `s`, `sm`, `m`, `ml`, `l`

Published variant table used in code:

- `xs`: 16 layers, 144 dim, 4 heads
- `s`: 18 layers, 196 dim, 4 heads
- `sm`: 16 layers, 256 dim, 4 heads
- `m`: 20 layers, 324 dim, 4 heads
- `ml`: 18 layers, 512 dim, 8 heads
- `l`: 22 layers, 640 dim, 8 heads

## Environment Setup

The repository is designed to run in a local `uv` environment.

```bash
uv venv .venv
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install polars jiwer sentencepiece huggingface_hub trackio pytest ruff
```

## Dataset Access

The scripts use `speech-uk/cv22`:

- dataset page: `https://huggingface.co/datasets/speech-uk/cv22`
- the dataset is gated
- you must accept the dataset terms on Hugging Face before download works

Set a token before training or evaluation:

```bash
export HF_TOKEN=your_huggingface_token
```

The loader downloads the dataset snapshot with `huggingface_hub.snapshot_download()` and reads TSV or Parquet manifests with `polars`.

For local development and smoke tests, `--dataset-repo` can also point at a local directory that contains Common Voice-style manifests and audio files.

Important: the dataset itself exposes only a source train split. This repo creates deterministic internal `train`, `validation`, and `test` splits from that source data. When real speaker metadata such as `client_id` or `speaker_id` is available, the split is speaker-aware. When the dataset does not expose speaker identity, the pipeline falls back to utterance-level hashing so training still works, and the split audit marks speaker identity as unavailable.

Default split behavior:

- `--val-fraction 0.1`
- `--test-fraction 0.1`

That yields an effective `80 / 10 / 10` train/validation/test split.

## Tokenization

Supported tokenizer modes:

- `sentencepiece`
- `character`

Default training behavior:

- `--tokenizer sentencepiece`
- `--spm-vocab-size 128`
- `--spm-model-type unigram`

SentencePiece runs write both:

- `tokenizer.json`
- `tokenizer.model`

Evaluation reloads the tokenizer from checkpoint metadata, so you do not need to specify tokenizer settings again.

## Audio Frontend

The frontend is now configurable and closer to the paper’s recipe:

- 80-bin log-mel features
- pre-emphasis, default `0.97`
- waveform normalization, enabled by default
- feature normalization, enabled by default
- configurable per-frame normalization if you want it
- configurable SpecAugment frequency and time masking
- optional speed perturbation
- optional additive noise
- optional synthetic reverb

Relevant training flags:

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
- `--reverb-decay-min`
- `--reverb-decay-max`
- `--reverb-delay-ms-min`
- `--reverb-delay-ms-max`

## Training Features

[train.py](/workspace/train.py) now includes:

- SentencePiece-128 by default
- Muon by default, with AdamW on auxiliary parameter groups
- separate Muon vs AdamW LR and weight-decay controls
- separate Muon vs AdamW scheduler overrides
- no-decay filtering for biases, norms, and scale parameters
- paper-style warmup, hold, and inverse-power decay scheduling
- gradient clipping
- gradient accumulation
- EMA checkpoints and EMA-based validation
- EMA warmup
- optional `torch.compile`
- optional activation checkpointing inside encoder blocks
- feature caching on disk
- up-front audio metadata and frame-count materialization
- length bucketing
- optional max-frames batching instead of fixed utterance-count batches
- adaptive batch scaling by effective frames or transcript tokens
- dataloader tuning knobs for `pin_memory`, `persistent_workers`, and `prefetch_factor`
- optional audio prevalidation before training
- transcript filtering for too-short, too-long, empty, and symbol-heavy rows
- top-k checkpoint retention
- checkpoint resume with optimizer, scheduler, scaler, EMA, and global step state
- greedy or beam-search validation decoding
- LM scorer hook for beam search
- optional training-time fit of a concrete shallow-fusion n-gram LM
- hardest and random decoded example logging in `trackio`
- WER/CER metrics broken out by utterance-length bucket
- speaker-level metrics and split audits
- automatic top-k checkpoint averaging
- per-checkpoint JSON evaluation reports
- basic single-node distributed training through `torchrun`

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
- `--ema-decay 0.999`

Variant-aware defaults derived from the paper:

- `xs`, `s`, `sm`: peak LR `2e-3`, time masks `5`
- `m`: peak LR `1.5e-3`, time masks `7`
- `ml`, `l`: peak LR `1e-3`, time masks `10`

## Quick Start

Minimal CPU smoke test:

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

More complete run with caching, bucketing, and beam-search validation:

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
  --batch-size 8 \
  --epochs 10
```

Additional data-quality and batching controls:

- `--min-transcript-chars`
- `--max-transcript-chars`
- `--max-symbol-ratio`
- `--max-batch-frames`
- `--adaptive-batch-unit frames|tokens`
- `--adaptive-batch-budget`
- `--metadata-workers`

Additional optimization controls:

- `--grad-clip-norm`
- `--ema-warmup-steps`
- `--muon-warmup-epochs`
- `--muon-hold-epochs`
- `--muon-decay-exponent`
- `--adamw-warmup-epochs`
- `--adamw-hold-epochs`
- `--adamw-decay-exponent`

Resume training:

```bash
HF_TOKEN=... uv run python train.py \
  --resume artifacts/cv22-sm/checkpoint_last.pt \
  --output-dir artifacts/cv22-sm
```

Distributed training:

```bash
HF_TOKEN=... torchrun --nproc_per_node=2 train.py \
  --distributed \
  --variant sm \
  --output-dir artifacts/cv22-sm-ddp
```

Local resumable smoke test:

```bash
uv run python scripts/smoke_resume_train.py
```

Evaluate:

```bash
HF_TOKEN=... uv run python evaluate.py \
  --checkpoint artifacts/cv22-sm/checkpoint_best.pt \
  --split test \
  --device cpu \
  --dtype bfloat16 \
  --decode-strategy beam \
  --beam-size 8
```

Benchmark:

```bash
uv run python benchmark.py \
  --variant sm \
  --batch-size 8 \
  --time-steps 512 \
  --decode-strategy beam \
  --beam-size 8
```

## Optimizer Behavior

The default optimizer path is mixed:

- Muon is applied to encoder 2D hidden weights
- AdamW is applied to the remaining parameters
- AdamW splits decay and no-decay parameter groups

Useful knobs:

- `--optimizer muon|adamw`
- `--learning-rate`
- `--muon-learning-rate`
- `--adamw-learning-rate`
- `--weight-decay`
- `--muon-weight-decay`
- `--adamw-weight-decay`

If the Muon path is not what you want, use `--optimizer adamw`.

## Decoding And LM Hook

Both training-time validation and `evaluate.py` support:

- `--decode-strategy greedy|beam`
- `--beam-size`
- `--lm-scorer`
- `--lm-weight`

`--lm-scorer` expects a Python callable in `module:function` form that takes a decoded prefix string and returns a scalar score. Example:

```bash
uv run python evaluate.py \
  --checkpoint artifacts/cv22-sm/checkpoint_best.pt \
  --decode-strategy beam \
  --lm-scorer my_lm:score_text \
  --lm-weight 0.2
```

This repo also includes a concrete character n-gram shallow-fusion LM in [lm.py](/workspace/squeezeformer_pytorch/lm.py).

Train and save it automatically during ASR training:

```bash
HF_TOKEN=... uv run python train.py \
  --output-dir artifacts/cv22-sm \
  --fit-shallow-fusion-lm \
  --decode-strategy beam \
  --beam-size 8 \
  --lm-weight 0.2
```

Train it from a standalone text corpus:

```bash
uv run python train_lm.py \
  --corpus data/lm_corpus.txt \
  --output artifacts/shallow_fusion_lm.json \
  --order 3 \
  --alpha 0.1
```

Or train it directly from a Hugging Face/local dataset by using the transcript field as text input:

```bash
HF_TOKEN=... uv run python train_lm.py \
  --dataset-repo speech-uk/cv22 \
  --output artifacts/shallow_fusion_lm.json \
  --deduplicate \
  --order 3 \
  --alpha 0.1
```

Build that corpus directly from cv22:

```bash
HF_TOKEN=... uv run python export_cv22_corpus.py \
  --dataset-repo speech-uk/cv22 \
  --output artifacts/cv22_corpus.txt \
  --deduplicate
```

Then train the LM:

```bash
uv run python train_lm.py \
  --corpus artifacts/cv22_corpus.txt \
  --output artifacts/shallow_fusion_lm.json
```

Use a saved LM explicitly from the generic hook:

```bash
uv run python evaluate.py \
  --checkpoint artifacts/cv22-sm/checkpoint_best.pt \
  --decode-strategy beam \
  --lm-scorer squeezeformer_pytorch.lm:load_saved_ngram_scorer:artifacts/cv22-sm/shallow_fusion_lm.json \
  --lm-weight 0.2
```

## Checkpoints

Training writes:

- `checkpoint_last.pt`
- `checkpoint_best.pt`
- `checkpoint_topk_avg.pt`
- `checkpoints_topk/` with the best `--keep-top-k` checkpoints by validation WER
- `checkpoints_topk/metadata.json`
- `eval_reports/epoch_XXXX.json`
- `split_audit.json`
- `tokenizer.json`
- `tokenizer.model` for SentencePiece runs
- `train_summary.json`

Resume loads:

- model weights
- optimizer state
- scheduler state
- grad scaler state
- EMA state
- epoch
- global step
- best validation WER

## Evaluation Outputs

[evaluate.py](/workspace/evaluate.py) prints and logs:

- loss
- CER
- WER
- per-bucket CER/WER for `short`, `medium`, and `long` utterances
- speaker-level aggregate metrics
- sample count
- hardest decoded example pairs
- random decoded example pairs

## Trackio

Training and evaluation log metrics to `trackio`.

Local dashboard:

```bash
trackio show --project "squeezeformer-cv22"
```

To sync logs to a Hugging Face Space, pass:

- `--trackio-space-id username/space-name`

## Validation

Current local checks:

```bash
uv run ruff check /workspace
uv run pytest -q
uv run python -c "import train, evaluate; print('imports_ok')"
```

## Current Limits

- This is still not the original large-scale paper training environment.
- The LM interface is only a scorer hook; there is no bundled external LM package or fusion recipe.
- There is no distributed training support.
- The dataset integration is practical and tested structurally, but the first full gated run is still the final contract check against the live dataset contents.

## Sources

- Paper source: [arXiv-2206.00888v2](/workspace/arXiv-2206.00888v2)
- Dataset card: `https://huggingface.co/datasets/speech-uk/cv22`
- `trackio`: `https://pypi.org/project/trackio/`
