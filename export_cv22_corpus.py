from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from squeezeformer_pytorch.data import (
    download_cv22_dataset,
    iter_cv22_corpus_texts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export normalized cv22 transcripts as a newline-delimited corpus for train_lm.py."
        )
    )
    parser.add_argument("--dataset-repo", default="speech-uk/cv22")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", default="artifacts/cv22_corpus.txt")
    parser.add_argument(
        "--deduplicate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop repeated transcript lines before writing the corpus.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = download_cv22_dataset(
        repo_id=args.dataset_repo,
        token=args.hf_token,
        cache_dir=args.cache_dir,
        allow_patterns=["*.tsv", "*.parquet", "**/*.tsv", "**/*.parquet"],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for text in iter_cv22_corpus_texts(
            dataset_root=dataset_root,
            deduplicate=args.deduplicate,
            max_samples=args.max_samples,
        ):
            handle.write(text)
            handle.write("\n")
            line_count += 1
    if line_count == 0:
        raise RuntimeError("No usable transcripts were found in the dataset manifests.")

    summary = {
        "dataset_root": str(dataset_root),
        "output": str(output_path),
        "lines": line_count,
        "deduplicate": args.deduplicate,
        "max_samples": args.max_samples,
        "next_step": f"uv run python train_lm.py --corpus {output_path}",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
