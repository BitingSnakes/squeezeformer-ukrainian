from __future__ import annotations

from typing import Iterable

from jiwer import cer, wer


def char_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    references_list = list(references)
    hypotheses_list = list(hypotheses)
    if not references_list:
        return 0.0
    return float(cer(references_list, hypotheses_list))


def word_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    references_list = list(references)
    hypotheses_list = list(hypotheses)
    if not references_list:
        return 0.0
    return float(wer(references_list, hypotheses_list))
