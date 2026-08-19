from __future__ import annotations

import logging
import re
import string
import unicodedata
from collections import Counter
from typing import Sequence


LOGGER = logging.getLogger(__name__)
_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    rouge_scorer = None


_ROUGE = (
    rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    if rouge_scorer is not None
    else None
)


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = "".join(" " if _is_punctuation(char) else char for char in text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(candidate: str, reference: str) -> float:
    return float(normalize_answer(candidate) == normalize_answer(reference))


def token_f1(candidate: str, reference: str) -> float:
    candidate_tokens = normalize_answer(candidate).split()
    reference_tokens = normalize_answer(reference).split()
    if not candidate_tokens or not reference_tokens:
        return float(candidate_tokens == reference_tokens)
    overlap = Counter(candidate_tokens) & Counter(reference_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(candidate_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_references(
    candidate: str,
    references: Sequence[str],
    *,
    allow_rouge_fallback: bool = False,
) -> dict[str, float | str]:
    if not references:
        raise ValueError("At least one reference is required")

    result: dict[str, float | str] = {
        "exact_match": max(exact_match(candidate, reference) for reference in references),
        "token_f1": max(token_f1(candidate, reference) for reference in references),
    }

    if _ROUGE is not None:
        rouge_scores = [_ROUGE.score(reference, candidate) for reference in references]
        result.update(
            {
                "rouge1_recall": max(score["rouge1"].recall for score in rouge_scores),
                "rouge1_f1": max(score["rouge1"].fmeasure for score in rouge_scores),
                "rouge_l_f1": max(score["rougeL"].fmeasure for score in rouge_scores),
                "rouge_backend": "rouge-score/stemmed",
            }
        )
    else:
        if not allow_rouge_fallback:
            raise RuntimeError(
                "rouge-score is required for formal scoring but is not installed. "
                "Install the project with 'pip install -e .', or set "
                "metrics.allow_rouge_fallback=true only for smoke tests."
            )
        LOGGER.warning(
            "rouge-score is not installed; using the dependency-free unstemmed fallback"
        )
        fallback = [_fallback_rouge(candidate, reference) for reference in references]
        result.update(
            {
                "rouge1_recall": max(score["rouge1_recall"] for score in fallback),
                "rouge1_f1": max(score["rouge1_f1"] for score in fallback),
                "rouge_l_f1": max(score["rouge_l_f1"] for score in fallback),
                "rouge_backend": "fallback/unstemmed",
            }
        )

    candidate_words = word_count(candidate)
    reference_words = [word_count(reference) for reference in references]
    mean_reference_words = sum(reference_words) / len(reference_words)
    result.update(
        {
            "candidate_words": float(candidate_words),
            "reference_words_mean": mean_reference_words,
            "length_ratio": (
                candidate_words / mean_reference_words if mean_reference_words else 0.0
            ),
        }
    )
    return result


def compute_bertscore_f1(
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
    *,
    model_type: str,
) -> list[float]:
    if len(candidates) != len(references):
        raise ValueError("BERTScore candidates and reference groups must have equal length")
    try:
        from bert_score import score as bert_score
    except ImportError as error:  # pragma: no cover - optional dependency.
        raise RuntimeError(
            "BERTScore is enabled but bert-score is not installed. "
            "Install the project with: pip install -e '.[semantic]'"
        ) from error

    flat_candidates: list[str] = []
    flat_references: list[str] = []
    owners: list[int] = []
    for owner, (candidate, candidate_references) in enumerate(zip(candidates, references)):
        if not candidate_references:
            raise ValueError(f"Candidate {owner} has no references")
        for reference in candidate_references:
            flat_candidates.append(candidate)
            flat_references.append(reference)
            owners.append(owner)

    _, _, f1 = bert_score(
        flat_candidates,
        flat_references,
        model_type=model_type,
        verbose=True,
    )
    maxima = [0.0] * len(candidates)
    for owner, value in zip(owners, f1.tolist()):
        maxima[owner] = max(maxima[owner], float(value))
    return maxima


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text), flags=re.UNICODE))


def _is_punctuation(char: str) -> bool:
    return char in string.punctuation or unicodedata.category(char).startswith("P")


def _fallback_rouge(candidate: str, reference: str) -> dict[str, float]:
    candidate_tokens = normalize_answer(candidate).split()
    reference_tokens = normalize_answer(reference).split()
    if not candidate_tokens or not reference_tokens:
        equal = float(candidate_tokens == reference_tokens)
        return {
            "rouge1_recall": equal,
            "rouge1_f1": equal,
            "rouge_l_f1": equal,
        }

    unigram_overlap = sum((Counter(candidate_tokens) & Counter(reference_tokens)).values())
    rouge1_precision = unigram_overlap / len(candidate_tokens)
    rouge1_recall = unigram_overlap / len(reference_tokens)
    rouge1_f1 = _f1(rouge1_precision, rouge1_recall)

    lcs = _lcs_length(candidate_tokens, reference_tokens)
    rouge_l_precision = lcs / len(candidate_tokens)
    rouge_l_recall = lcs / len(reference_tokens)
    return {
        "rouge1_recall": rouge1_recall,
        "rouge1_f1": rouge1_f1,
        "rouge_l_f1": _f1(rouge_l_precision, rouge_l_recall),
    }


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
