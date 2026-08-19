from __future__ import annotations

import csv
import json
import math
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import binomtest

from .config import ExperimentConfig
from .data import Example, load_examples
from .io import read_jsonl, write_jsonl
from .metrics import compute_bertscore_f1, score_references
from .validate import (
    validate_generation_rows,
    validate_judgment_row,
)


_LABELS = ("CORRECT", "MINOR_ERROR", "MAJOR_ERROR")
_NUMERIC_METRICS = (
    "strict_correct",
    "non_major",
    "exact_match",
    "token_f1",
    "rouge1_recall",
    "rouge1_f1",
    "rouge_l_f1",
    "candidate_words",
    "reference_words_mean",
    "length_ratio",
)


def score_run(config: ExperimentConfig) -> dict[str, Any]:
    rows, coverage, examples = _load_and_validate_inputs(config)
    if not rows:
        raise ValueError("Judgment files contain no records")

    scored: list[dict[str, Any]] = []
    for row in rows:
        references = [str(value) for value in row["references"]]
        lexical = score_references(
            str(row["answer"]),
            references,
            allow_rouge_fallback=config.metrics.allow_rouge_fallback,
        )
        scored.append({**row, **lexical})

    if config.metrics.bertscore:
        values = compute_bertscore_f1(
            [str(row["answer"]) for row in scored],
            [[str(value) for value in row["references"]] for row in scored],
            model_type=config.metrics.bertscore_model,
        )
        for row, value in zip(scored, values):
            row["bertscore_f1"] = value

    metrics_dir = config.run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = metrics_dir / "per_example.jsonl"
    coverage_path = metrics_dir / "coverage.json"
    write_jsonl(per_example_path, scored)
    _write_json(coverage_path, coverage)

    coverage_by_pair = {
        (str(item["model"]), str(item["condition"])): item
        for item in coverage
    }
    summaries = _condition_summaries(scored, coverage_by_pair)
    paired = _paired_summaries(
        scored,
        expected_n=len(examples),
        bootstrap_samples=config.metrics.bootstrap_samples,
        confidence_level=config.metrics.confidence_level,
        seed=config.run.seed,
    )

    summary_json = metrics_dir / "summary.json"
    paired_json = metrics_dir / "paired.json"
    _write_json(summary_json, summaries)
    _write_json(paired_json, paired)
    _write_csv(metrics_dir / "summary.csv", summaries)
    _write_csv(metrics_dir / "paired.csv", paired)

    is_partial = any(bool(item["partial"]) for item in coverage)
    return {
        "records": len(scored),
        "partial": is_partial,
        "per_example": str(per_example_path),
        "coverage": str(coverage_path),
        "summary": str(summary_json),
        "paired": str(paired_json),
    }


def _load_and_validate_inputs(
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Example]]:
    examples = load_examples(config.data.normalized_path)
    examples_by_id = {example.example_id: example for example in examples}
    if len(examples_by_id) != len(examples):
        counts = Counter(example.example_id for example in examples)
        duplicates = sorted(
            example_id for example_id, count in counts.items() if count > 1
        )
        raise RuntimeError(
            "Normalized data contains duplicate example_id values: "
            + ", ".join(repr(value) for value in duplicates)
        )

    expected_ids = set(examples_by_id)
    judgment_dir = config.run_dir / "judgments"
    generation_dir = config.run_dir / "generations"

    expected_paths = {
        judgment_dir / f"{model}.{condition}.jsonl"
        for model in config.models
        for condition in config.generation.conditions
    }
    actual_paths = set(judgment_dir.glob("*.jsonl")) if judgment_dir.exists() else set()
    extra_paths = sorted(actual_paths - expected_paths)
    if extra_paths:
        rendered = ", ".join(str(path) for path in extra_paths)
        raise RuntimeError(f"Unexpected judgment files: {rendered}")

    all_rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for model in sorted(config.models):
        for condition in config.generation.conditions:
            judgment_path = judgment_dir / f"{model}.{condition}.jsonl"
            generation_path = generation_dir / f"{model}.{condition}.jsonl"
            if not judgment_path.exists():
                missing_ids = sorted(expected_ids)
                coverage.append(
                    _coverage_row(
                        model=model,
                        condition=condition,
                        expected_n=len(examples),
                        generation_n=0,
                        judgment_n=0,
                        missing_ids=missing_ids,
                        file_exists=False,
                    )
                )
                incomplete.append(
                    f"{model}/{condition}: missing judgment file"
                )
                continue
            if not generation_path.exists():
                raise RuntimeError(
                    f"Judgments exist without their generation file: {generation_path}"
                )

            generation_rows = list(
                read_jsonl(generation_path, tolerate_trailing_partial=True)
            )
            validate_generation_rows(
                rows=generation_rows,
                examples_by_id=examples_by_id,
                config=config,
                model_key=model,
                condition=condition,
                path=generation_path,
            )
            generation_by_id = {
                str(row["example_id"]): row for row in generation_rows
            }

            judgment_rows = list(
                read_jsonl(judgment_path, tolerate_trailing_partial=True)
            )
            seen: set[str] = set()
            for row in judgment_rows:
                example_id = str(row.get("example_id", ""))
                if not example_id or example_id in seen:
                    raise RuntimeError(
                        f"Duplicate or missing example_id {example_id!r} "
                        f"in {judgment_path}"
                    )
                seen.add(example_id)
                validate_judgment_row(
                    row=row,
                    generation=generation_by_id.get(example_id),
                    example=examples_by_id.get(example_id),
                    config=config,
                    model=model,
                    condition=condition,
                    path=judgment_path,
                    generation_path=generation_path,
                    context="input",
                )

            missing_ids = sorted(expected_ids - seen)
            coverage.append(
                _coverage_row(
                    model=model,
                    condition=condition,
                    expected_n=len(examples),
                    generation_n=len(generation_rows),
                    judgment_n=len(judgment_rows),
                    missing_ids=missing_ids,
                    file_exists=True,
                )
            )
            if missing_ids:
                incomplete.append(
                    f"{model}/{condition}: {len(missing_ids)} missing judgments"
                )
            all_rows.extend(judgment_rows)

    if incomplete:
        details = "\n".join(f"- {item}" for item in incomplete)
        raise RuntimeError(
            "Experiment is incomplete; refusing to compute formal metrics.\n"
            f"{details}\n"
            "Finish all configured generation and Judge runs before scoring."
        )
    return all_rows, coverage, examples


def _coverage_row(
    *,
    model: str,
    condition: str,
    expected_n: int,
    generation_n: int,
    judgment_n: int,
    missing_ids: list[str],
    file_exists: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "condition": condition,
        "expected_n": expected_n,
        "generation_n": generation_n,
        "judgment_n": judgment_n,
        "coverage_rate": judgment_n / expected_n if expected_n else 0.0,
        "missing_n": len(missing_ids),
        "missing_ids": missing_ids,
        "file_exists": file_exists,
        "partial": judgment_n != expected_n,
    }


def _condition_summaries(
    rows: Sequence[dict[str, Any]],
    coverage_by_pair: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["condition"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (model, condition), group in sorted(grouped.items()):
        labels = Counter(str(row["label"]) for row in group)
        coverage = coverage_by_pair[(model, condition)]
        summary: dict[str, Any] = {
            "model": model,
            "condition": condition,
            "n": len(group),
            "expected_n": coverage["expected_n"],
            "coverage_rate": coverage["coverage_rate"],
            "partial": coverage["partial"],
            "correct_count": labels["CORRECT"],
            "minor_error_count": labels["MINOR_ERROR"],
            "major_error_count": labels["MAJOR_ERROR"],
            "rouge_backend": _single_value(group, "rouge_backend"),
        }
        for metric in _NUMERIC_METRICS:
            summary[f"mean_{metric}"] = _mean(group, metric)
        if "bertscore_f1" in group[0]:
            summary["mean_bertscore_f1"] = _mean(group, "bertscore_f1")
        summaries.append(summary)
    return summaries


def _paired_summaries(
    rows: Sequence[dict[str, Any]],
    *,
    expected_n: int,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["model"]), str(row["condition"]))
        example_id = str(row["example_id"])
        if example_id in indexed[key]:
            raise RuntimeError(
                f"Duplicate example_id {example_id!r} in paired inputs for "
                f"{key[0]}/{key[1]}"
            )
        indexed[key][example_id] = row

    models = sorted({model for model, _ in indexed})
    summaries: list[dict[str, Any]] = []
    for model in models:
        gold = indexed.get((model, "gold"), {})
        closed = indexed.get((model, "closed_book"), {})
        if not gold or not closed:
            continue
        gold_ids = set(gold)
        closed_ids = set(closed)
        if gold_ids != closed_ids:
            missing_from_gold = sorted(closed_ids - gold_ids)
            missing_from_closed = sorted(gold_ids - closed_ids)
            raise RuntimeError(
                f"Condition ID mismatch for {model}: "
                f"missing from gold={missing_from_gold}, "
                f"missing from closed_book={missing_from_closed}"
            )
        if len(gold_ids) != expected_n:
            raise RuntimeError(
                f"Incomplete paired inputs for {model}: expected {expected_n}, "
                f"found {len(gold_ids)}"
            )
        common = sorted(gold_ids)

        gold_strict = np.asarray([float(gold[key]["strict_correct"]) for key in common])
        closed_strict = np.asarray([float(closed[key]["strict_correct"]) for key in common])
        gold_non_major = np.asarray([float(gold[key]["non_major"]) for key in common])
        closed_non_major = np.asarray([float(closed[key]["non_major"]) for key in common])

        rescue = int(np.sum((closed_strict == 0) & (gold_strict == 1)))
        harm = int(np.sum((closed_strict == 1) & (gold_strict == 0)))
        both_correct = int(np.sum((closed_strict == 1) & (gold_strict == 1)))
        both_wrong = int(np.sum((closed_strict == 0) & (gold_strict == 0)))
        closed_wrong = rescue + both_wrong
        closed_correct = harm + both_correct

        model_seed = seed + zlib.crc32(model.encode("utf-8"))
        strict_low, strict_high = paired_bootstrap_interval(
            gold_strict,
            closed_strict,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=model_seed,
        )
        non_major_low, non_major_high = paired_bootstrap_interval(
            gold_non_major,
            closed_non_major,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=model_seed + 1,
        )

        transitions = Counter(
            (str(closed[key]["label"]), str(gold[key]["label"]))
            for key in common
        )
        summary: dict[str, Any] = {
            "model": model,
            "n_expected": expected_n,
            "n_paired": len(common),
            "partial": len(common) != expected_n,
            "gold_strict_accuracy": float(np.mean(gold_strict)),
            "closed_book_strict_accuracy": float(np.mean(closed_strict)),
            "strict_context_gain": float(np.mean(gold_strict - closed_strict)),
            "strict_context_gain_ci_low": strict_low,
            "strict_context_gain_ci_high": strict_high,
            "gold_non_major_rate": float(np.mean(gold_non_major)),
            "closed_book_non_major_rate": float(np.mean(closed_non_major)),
            "non_major_context_gain": float(np.mean(gold_non_major - closed_non_major)),
            "non_major_context_gain_ci_low": non_major_low,
            "non_major_context_gain_ci_high": non_major_high,
            "closed_wrong_gold_correct": rescue,
            "closed_correct_gold_correct": both_correct,
            "closed_wrong_gold_wrong": both_wrong,
            "closed_correct_gold_wrong": harm,
            "strict_rescue_rate": rescue / closed_wrong if closed_wrong else None,
            "strict_harm_rate": harm / closed_correct if closed_correct else None,
            "mcnemar_exact_p": exact_mcnemar_p(rescue, harm),
        }
        for closed_label in _LABELS:
            for gold_label in _LABELS:
                key = (
                    f"transition_{closed_label.lower()}_to_"
                    f"{gold_label.lower()}"
                )
                summary[key] = transitions[(closed_label, gold_label)]

        for metric in ("token_f1", "rouge_l_f1", "rouge1_recall"):
            gold_values = np.asarray([float(gold[key][metric]) for key in common])
            closed_values = np.asarray([float(closed[key][metric]) for key in common])
            summary[f"{metric}_context_gain"] = float(
                np.mean(gold_values - closed_values)
            )
        if "bertscore_f1" in gold[common[0]] and "bertscore_f1" in closed[common[0]]:
            gold_values = np.asarray([float(gold[key]["bertscore_f1"]) for key in common])
            closed_values = np.asarray([float(closed[key]["bertscore_f1"]) for key in common])
            summary["bertscore_f1_context_gain"] = float(
                np.mean(gold_values - closed_values)
            )
        summaries.append(summary)
    return summaries


def paired_bootstrap_interval(
    left: np.ndarray,
    right: np.ndarray,
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("paired bootstrap inputs must be non-empty one-dimensional arrays")
    rng = np.random.default_rng(seed)
    differences = left - right
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selection = rng.integers(0, differences.size, size=differences.size)
        estimates[index] = float(np.mean(differences[selection]))
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def exact_mcnemar_p(gold_only_correct: int, closed_only_correct: int) -> float:
    discordant = gold_only_correct + closed_only_correct
    if discordant == 0:
        return 1.0
    return float(
        binomtest(
            min(gold_only_correct, closed_only_correct),
            discordant,
            0.5,
        ).pvalue
    )


def _mean(rows: Iterable[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return float(math.fsum(values) / len(values))


def _single_value(rows: Sequence[dict[str, Any]], field: str) -> Any:
    values = {row.get(field) for row in rows}
    return next(iter(values)) if len(values) == 1 else "mixed"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
