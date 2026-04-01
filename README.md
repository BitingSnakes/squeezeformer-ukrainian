# Squeezeformer PyTorch

This repository contains a standalone PyTorch implementation of the Squeezeformer encoder from the paper source in [arXiv-2206.00888v2](/workspace/arXiv-2206.00888v2), plus minimal training and evaluation scripts for CTC-based ASR on the gated Hugging Face dataset `speech-uk/cv22`.

The code is intentionally small and explicit:

- `squeezeformer_pytorch/model.py`: encoder architecture
- `squeezeformer_pytorch/asr.py`: CTC wrapper and character tokenizer
- `squeezeformer_pytorch/data.py`: dataset download, Polars manifest loading, audio featurization, batching
- `squeezeformer_pytorch/metrics.py`: CER and WER
- `train.py`: training entrypoint
- `evaluate.py`: evaluation entrypoint
- `tests/`: forward-shape and variant checks

## What Is Implemented

The encoder follows the paper’s main architectural changes:

- depthwise-separable 4x acoustic subsampling
- Squeezeformer MF/CF block structure
- scaled post-LN residual modules
- relative-position attention
- Temporal U-Net style time reduction and time recovery
- published model variants: `xs`, `s`, `sm`, `m`, `ml`, `l`

This is an encoder-first implementation, not a full reproduction of the authors’ original training stack. In particular:

- decoding is greedy CTC only
- tokenization is character-level, not SentencePiece
- training defaults are practical local defaults, not the paper’s TPU-scale recipe

SentencePiece tokenization is also supported as a runtime option for the training and evaluation scripts.

## Requirements

The project uses a local `uv` environment.

Core runtime dependencies:

- `torch`
- `torchaudio`
- `polars`
- `jiwer`
- `sentencepiece`
- `huggingface_hub`
- `trackio`
- `pytest`
- `ruff`

If `.venv` does not exist yet:

```bash
uv venv .venv
```

Install dependencies:

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install polars jiwer sentencepiece huggingface_hub trackio pytest ruff
```

## Dataset Access

Training uses `speech-uk/cv22` from Hugging Face:

- dataset page: `https://huggingface.co/datasets/speech-uk/cv22`
- the dataset is gated
- you must log in on Hugging Face, accept the access conditions, and provide a token

Export a token before training or evaluation:

```bash
export HF_TOKEN=your_huggingface_token
```

The loader downloads the dataset snapshot with `huggingface_hub.snapshot_download()` and uses `polars` to read the dataset manifests from TSV or Parquet files.

## Quick Start

Train a small run:

```bash
HF_TOKEN=... uv run python train.py \
  --variant sm \
  --tokenizer sentencepiece \
  --output-dir artifacts/cv22-sm \
  --batch-size 8 \
  --epochs 10
```

Evaluate the best checkpoint:

```bash
HF_TOKEN=... uv run python evaluate.py \
  --checkpoint artifacts/cv22-sm/checkpoint_best.pt
```

## Training Script

`train.py` does the following:

1. downloads the dataset snapshot from Hugging Face
2. loads manifests with Polars
3. creates deterministic train/validation/test splits using a hash of utterance id
4. builds a character vocabulary from the training transcripts
5. extracts 80-bin log-mel features with `torchaudio`
6. trains a Squeezeformer encoder with a CTC head
7. logs metrics to `trackio`
8. saves:
   - `checkpoint_last.pt`
   - `checkpoint_best.pt`
   - `tokenizer.json`
   - `train_summary.json`

Common options:

- `--variant`: one of `xs`, `s`, `sm`, `m`, `ml`, `l`
- `--output-dir`: checkpoint directory
- `--batch-size`
- `--epochs`
- `--learning-rate`
- `--max-train-samples`
- `--max-val-samples`
- `--trackio-project`
- `--trackio-space-id`
- `--tokenizer`: `character` or `sentencepiece`
- `--spm-vocab-size`
- `--spm-model-type`

Example with smaller subsets for a smoke test:

```bash
HF_TOKEN=... uv run python train.py \
  --variant xs \
  --output-dir artifacts/smoke \
  --epochs 1 \
  --max-train-samples 64 \
  --max-val-samples 16
```

## Evaluation Script

`evaluate.py` loads a saved checkpoint and computes:

- loss
- CER
- WER

It also logs the evaluation summary to `trackio`.

Example:

```bash
HF_TOKEN=... uv run python evaluate.py \
  --checkpoint artifacts/cv22-sm/checkpoint_best.pt \
  --split test
```

## Trackio Logging

Training and evaluation use `trackio` for experiment logging.

Local logging works out of the box:

```python
trackio.init(project="squeezeformer-cv22")
trackio.log({"train_loss": 0.42})
trackio.finish()
```

To view a local dashboard:

```bash
trackio show --project "squeezeformer-cv22"
```

If you want logs to sync to a Hugging Face Space, pass:

- `--trackio-space-id username/space-name`

## Architecture Notes

The paper specifies the macro-architecture clearly, but some implementation details are only implicit in the text. This code resolves those gaps using the paper’s cited public reference implementation where needed, especially for:

- exact block layout
- variant-specific reduction and recovery indices
- convolution module expansion behavior

The implemented defaults align with the published variant table:

- `xs`: 16 layers, 144 dim, 4 heads
- `s`: 18 layers, 196 dim, 4 heads
- `sm`: 16 layers, 256 dim, 4 heads
- `m`: 20 layers, 324 dim, 4 heads
- `ml`: 18 layers, 512 dim, 8 heads
- `l`: 22 layers, 640 dim, 8 heads

## Project Structure

```text
/workspace
├── README.md
├── pyproject.toml
├── train.py
├── evaluate.py
├── squeezeformer_pytorch
│   ├── __init__.py
│   ├── model.py
│   ├── asr.py
│   ├── data.py
│   └── metrics.py
├── tests
│   ├── conftest.py
│   └── test_squeezeformer.py
└── arXiv-2206.00888v2
```

## Validation

Current local checks:

```bash
uv run ruff check /workspace
uv run pytest -q
uv run python -c "import train, evaluate; print('imports_ok')"
```

## Known Limitations

- This is not a full paper reproduction.
- The dataset loader is designed for Common Voice style manifests and should be considered correct-by-inspection until the first real gated dataset run confirms every column in `speech-uk/cv22`.
- The tokenizer is character-based rather than SentencePiece.
- No beam search or language model decoding is included.
- No distributed training support is included.

## Next Steps

Good follow-up improvements:

- add SentencePiece tokenization
- add checkpoint resume support
- add mixed precision training
- add beam search decoding
- add explicit manifest schema tests once the gated dataset is available locally
