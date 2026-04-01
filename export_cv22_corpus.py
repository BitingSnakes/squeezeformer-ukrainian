from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from squeezeformer_pytorch.data import (
    iter_cv22_corpus_texts,
    iter_cv22_corpus_texts_from_repo,
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
    dataset_path = Path(args.dataset_repo)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        if dataset_path.exists():
            texts = iter_cv22_corpus_texts(
                dataset_root=dataset_path,
                deduplicate=args.deduplicate,
                max_samples=args.max_samples,
            )
            dataset_source = str(dataset_path.resolve())
        else:
            texts = iter_cv22_corpus_texts_from_repo(
                repo_id=args.dataset_repo,
                token=args.hf_token,
                deduplicate=args.deduplicate,
                max_samples=args.max_samples,
            )
            dataset_source = args.dataset_repo
        for text in texts:
            handle.write(text)
            handle.write("\n")
            line_count += 1
    if line_count == 0:
        raise RuntimeError("No usable transcripts were found in the dataset manifests.")

    summary = {
        "dataset_root": dataset_source,
        "output": str(output_path),
        "lines": line_count,
        "deduplicate": args.deduplicate,
        "max_samples": args.max_samples,
        "next_step": f"uv run python train_lm.py --corpus {output_path}",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
