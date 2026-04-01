from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from squeezeformer_pytorch.data import (
    iter_cv22_corpus_texts,
    iter_cv22_corpus_texts_from_repo,
)
from squeezeformer_pytorch.lm import NGramLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a shallow-fusion character n-gram LM from a newline-delimited corpus."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--corpus",
        help="Path to a newline-delimited text corpus.",
    )
    source_group.add_argument(
        "--dataset-repo",
        help="Hugging Face dataset repo or local dataset directory to extract transcripts from.",
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--deduplicate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop repeated transcript lines before LM training.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--output",
        default="artifacts/shallow_fusion_lm.json",
        help="Path to the LM JSON artifact.",
    )
    parser.add_argument("--order", type=int, default=3, help="n-gram order.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Add-alpha smoothing value.")
    parser.add_argument(
        "--preview-text",
        default=None,
        help="Optional text to score after training.",
    )
    return parser.parse_args()


def iter_corpus_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            normalized = line.strip()
            if normalized:
                yield normalized


def count_texts(
    texts: Iterable[str],
    progress_description: str,
    total: int | None = None,
) -> tuple[Iterable[str], dict[str, int]]:
    counter = {"lines": 0}
    progress = tqdm(desc=progress_description, total=total, unit="line")

    def iterator() -> Iterable[str]:
        try:
            for text in texts:
                counter["lines"] += 1
                progress.update()
                yield text
        finally:
            progress.close()

    return iterator(), counter


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.corpus is not None:
        corpus_path = Path(args.corpus)
        texts, counter = count_texts(
            iter_corpus_lines(corpus_path),
            progress_description="Reading corpus",
        )
        input_source = str(corpus_path)
        input_type = "corpus"
    else:
        dataset_path = Path(args.dataset_repo)
        if dataset_path.exists():
            texts, counter = count_texts(
                iter_cv22_corpus_texts(
                    dataset_root=dataset_path,
                    deduplicate=args.deduplicate,
                    max_samples=args.max_samples,
                ),
                progress_description="Reading dataset",
                total=args.max_samples,
            )
            input_source = str(dataset_path.resolve())
            input_type = "dataset"
        else:
            texts, counter = count_texts(
                iter_cv22_corpus_texts_from_repo(
                    repo_id=args.dataset_repo,
                    token=args.hf_token,
                    deduplicate=args.deduplicate,
                    max_samples=args.max_samples,
                ),
                progress_description="Streaming dataset",
                total=args.max_samples,
            )
            input_source = args.dataset_repo
            input_type = "dataset_repo"
    try:
        lm = NGramLanguageModel.train(texts, order=args.order, alpha=args.alpha)
    except ValueError as error:
        if counter["lines"] == 0:
            raise ValueError(
                f"{input_type} '{input_source}' does not contain any non-empty lines"
            ) from error
        raise
    lm.save(output_path)

    summary: dict[str, object] = {
        "input_type": input_type,
        "input_source": input_source,
        "output": str(output_path),
        "lines": counter["lines"],
        "order": lm.order,
        "alpha": lm.alpha,
        "vocabulary_size": len(lm.vocabulary),
        "lm_scorer_spec": f"squeezeformer_pytorch.lm:load_saved_ngram_scorer:{output_path}",
    }
    if args.dataset_repo is not None:
        summary["deduplicate"] = args.deduplicate
        summary["max_samples"] = args.max_samples
    if args.preview_text:
        summary["preview_text"] = args.preview_text
        summary["preview_log_score"] = lm.score_text(args.preview_text)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
