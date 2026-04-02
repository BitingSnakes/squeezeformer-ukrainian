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
- [extract_features.py](/workspace/extract_features.py): extract and cache frontend log-mel features without training
- [evaluate.py](/workspace/evaluate.py): evaluation entrypoint
- [benchmark.py](/workspace/benchmark.py): synthetic throughput, memory, and decode-speed benchmark
- [hparam_tuner.py](/workspace/hparam_tuner.py): estimate hardware-sensitive `train.py` values and emit a ready command
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

CPU setup:

```bash
uv venv .venv
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install polars jiwer sentencepiece huggingface_hub trackio pytest ruff
```

TPU setup:

```bash
uv venv .venv
source .venv/bin/activate

# Install PyTorch, TorchAudio, and Torch/XLA builds that match your TPU runtime.
uv pip install torch torchaudio torch_xla

# Install the project dependencies.
uv pip install -e .
uv pip install pytest ruff
```

For TPU runs, install a `torch_xla` build that matches your `torch` version and runtime, then
pass `--device xla` or `--device xla:N` to `train.py` or `evaluate.py`. The current TPU path is
single-process only; the existing `torchrun` distributed path remains CUDA/CPU-oriented.

On managed TPU environments such as Colab or Kaggle, prefer creating the `uv` environment after
the TPU runtime is already attached. If the runtime preinstalls `torch`, `torch_xla`, or related
TPU packages, keep those versions aligned when installing into `.venv`.

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

For transcript-only workflows such as [train_lm.py](/workspace/train_lm.py), the repo now
downloads only manifest files and streams transcript rows instead of materializing the full
corpus in RAM. That keeps LM preparation usable even when the underlying dataset is tens of GB.

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

## Feature Extraction

This repo supports both on-the-fly cache warming through training/evaluation and a standalone
offline extractor via [extract_features.py](/workspace/extract_features.py).

When caching is enabled:

- training writes cached tensors under `FEATURE_CACHE_DIR/train`
- validation writes cached tensors under `FEATURE_CACHE_DIR/validation`
- evaluation writes cached tensors under `FEATURE_CACHE_DIR/<split>`
- offline extraction writes cached tensors under `FEATURE_CACHE_DIR/<split>`
- each cached file is a `.pt` tensor keyed by utterance id and frontend config hash

Example: warm the train and validation feature cache during a short run

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

Example: extract and cache features for the evaluation split without training

```bash
HF_TOKEN=... uv run python evaluate.py \
  --checkpoint artifacts/squeezeformer-cv22/checkpoint_last.pt \
  --split test \
  --device cpu \
  --dtype bfloat16 \
  --feature-cache-dir artifacts/feature_cache
```

Example: extract and cache the train split directly, including for large local datasets,
without first launching a train/eval job

```bash
HF_TOKEN=... uv run python extract_features.py \
  --dataset-repo speech-uk/cv22 \
  --split train \
  --feature-cache-dir artifacts/feature_cache
```

Notes:

- cache reuse is disabled for training samples when waveform augmentation is enabled
- changing frontend settings such as `--preemphasis` or normalization flags produces a new cache key
- the cache stores model inputs after featurization, not raw audio embeddings from the encoder
- the offline extractor now streams split selection from dataset manifests instead of building the full record list in RAM first

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
- hardware-sensitive hyperparameter estimation through `hparam_tuner.py`
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

Estimate hardware-sensitive values for the fuller training command:

```bash
uv run python hparam_tuner.py \
  --variant sm \
  --optimizer muon \
  --tokenizer sentencepiece \
  --spm-vocab-size 128 \
  --device cpu \
  --dtype bfloat16 \
  --feature-cache-dir artifacts/feature_cache \
  --speed-perturb-prob 0.5 \
  --noise-prob 0.2 \
  --reverb-prob 0.1 \
  --decode-strategy beam \
  --beam-size 8 \
  --output-dir artifacts/cv22-sm \
  --epochs 10 \
  --emit-format both
```

`hparam_tuner.py` keeps the user-selected settings and estimates the knobs most likely to
depend on hardware or runtime pressure:

- `--batch-size`
- `--max-batch-frames`
- `--gradient-accumulation-steps`
- `--num-workers`
- `--metadata-workers`
- `--prefetch-factor`
- `--beam-size` for CPU beam-search runs

`hparam_tuner.py` also accepts `--device xla` for TPU-oriented estimates. That path uses
conservative TPU heuristics instead of live HBM probing, so treat the output as a starting
point and adjust `--max-batch-frames` or accumulation if your runtime still OOMs.

Example CPU estimate for the command below:

```bash
uv run python train.py \
  --variant sm \
  --optimizer muon \
  --tokenizer sentencepiece \
  --spm-vocab-size 128 \
  --device cpu \
  --dtype bfloat16 \
  --gradient-accumulation-steps 4 \
  --feature-cache-dir artifacts/feature_cache \
  --max-batch-frames 13500 \
  --speed-perturb-prob 0.5 \
  --noise-prob 0.2 \
  --reverb-prob 0.1 \
  --decode-strategy beam \
  --beam-size 4 \
  --output-dir artifacts/cv22-sm \
  --batch-size 9 \
  --epochs 10 \
  --num-workers 8 \
  --metadata-workers 8 \
  --prefetch-factor 2 \
  --no-pin-memory \
  --persistent-workers
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

Evaluate on TPU:

```bash
HF_TOKEN=... uv run python evaluate.py \
  --checkpoint artifacts/cv22-sm/checkpoint_best.pt \
  --split test \
  --device xla \
  --dtype bfloat16
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
- `checkpoint_last.safetensors` plus `checkpoint_last.json` for inference/evaluation
- `checkpoint_best.pt`
- `checkpoint_best.safetensors` plus `checkpoint_best.json` for inference/evaluation
- `checkpoint_topk_avg.pt`
- `checkpoint_topk_avg.safetensors` plus `checkpoint_topk_avg.json` for inference/evaluation
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

Shared dashboard:

```bash
GRADIO_SHARE=True uv run python -c "import trackio; trackio.show()"
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
