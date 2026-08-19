from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .config import Condition, ExperimentConfig
from .data import Example
from .io import record_sha256
from .prompts import (
    GENERATION_PROMPT_VERSION,
    GENERATION_SYSTEM_PROMPT,
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    build_generation_user_prompt,
    build_judge_user_prompt,
    prompt_sha256,
)
from .runtime import judge_server_matches_config
from .schema import JudgeResult


def generation_parameters(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "temperature": config.generation.temperature,
        "top_p": config.generation.top_p,
        "top_k": config.generation.top_k,
        "min_p": config.generation.min_p,
        "presence_penalty": config.generation.presence_penalty,
        "repetition_penalty": config.generation.repetition_penalty,
        "max_tokens": config.generation.max_tokens,
        "seed": config.run.seed,
    }


def judge_parameters(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "enable_thinking": config.judge.enable_thinking,
        "deterministic_inference": config.judge.deterministic_inference,
        "temperature": config.judge.temperature,
        "top_p": config.judge.top_p,
        "top_k": config.judge.top_k,
        "min_p": config.judge.min_p,
        "presence_penalty": config.judge.presence_penalty,
        "repetition_penalty": config.judge.repetition_penalty,
        "max_tokens": config.judge.max_tokens,
        "seed": config.run.seed,
    }


def validate_generation_rows(
    *,
    rows: list[dict[str, Any]],
    examples_by_id: dict[str, Example],
    config: ExperimentConfig,
    model_key: str,
    served_model: str | None = None,
    condition: Condition,
    path: Path,
    context: Literal["resume", "input"] = "input",
) -> None:
    seen: set[str] = set()
    resolved_model = served_model or config.models[model_key].served_model
    parameters = generation_parameters(config)
    for row in rows:
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in seen:
            raise RuntimeError(
                f"Duplicate or missing example_id {example_id!r} in {path}"
            )
        seen.add(example_id)
        example = examples_by_id.get(example_id)
        if example is None:
            raise RuntimeError(f"Unknown example_id {example_id!r} in {path}")

        user_prompt = build_generation_user_prompt(
            condition=condition,
            question=example.question,
            title=example.title,
            passage=example.passage,
        )
        expected = {
            "source_sha256": record_sha256(example.model_dump(mode="json")),
            "model": model_key,
            "served_model": resolved_model,
            "condition": condition,
            "question": example.question,
            "references": example.references,
            "prompt_version": GENERATION_PROMPT_VERSION,
            "prompt_sha256": prompt_sha256(GENERATION_SYSTEM_PROMPT, user_prompt),
            "generation_parameters": parameters,
        }
        for field, value in expected.items():
            if row.get(field) != value:
                if context == "resume":
                    raise RuntimeError(
                        f"Cannot resume {path}: {field} mismatch for {example_id}. "
                        "Use a new run.name for changed data, prompts, models, or parameters."
                    )
                raise RuntimeError(
                    f"Invalid generation input {path}: {field} mismatch "
                    f"for {example_id}. Regenerate under a new run.name."
                )
        if not isinstance(row.get("answer"), str):
            raise RuntimeError(f"Invalid answer for {example_id!r} in {path}")


def validate_judgment_row(
    *,
    row: dict[str, Any],
    generation: dict[str, Any] | None,
    example: Example | None,
    config: ExperimentConfig,
    model: str,
    condition: Condition,
    path: Path,
    generation_path: Path | None = None,
    server: dict[str, Any] | None = None,
    context: Literal["resume", "input"] = "input",
) -> None:
    example_id = str(row.get("example_id", ""))
    if example is None:
        if context == "resume":
            raise RuntimeError(f"Unknown example_id {example_id!r} in {path}")
        raise RuntimeError(f"Unexpected example_id {example_id!r} in {path}")
    if generation is None:
        if context == "resume":
            raise RuntimeError(f"Unknown example_id {example_id!r} in {path}")
        source = generation_path or path
        raise RuntimeError(
            f"Judgment {example_id!r} has no matching generation "
            f"in {source}"
        )

    candidate = str(generation["answer"])
    user_prompt = build_judge_user_prompt(
        question=example.question,
        references=example.references,
        candidate=candidate,
    )
    expected = {
        "source_sha256": record_sha256(example.model_dump(mode="json")),
        "generation_sha256": record_sha256(generation),
        "model": model,
        "condition": condition,
        "question": example.question,
        "references": example.references,
        "answer": candidate,
        "judge_model": config.judge.served_model,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_prompt_sha256": prompt_sha256(JUDGE_SYSTEM_PROMPT, user_prompt),
        "judge_parameters": judge_parameters(config),
    }
    for field, value in expected.items():
        if row.get(field) != value:
            if context == "resume":
                raise RuntimeError(
                    f"Cannot resume {path}: {field} mismatch for {example_id}. "
                    "Use a new run.name for changed inputs or Judge settings."
                )
            raise RuntimeError(
                f"Invalid judgment {path}: {field} mismatch for {example_id}"
            )

    _validate_judge_server(
        row=row,
        config=config,
        example_id=example_id,
        path=path,
        server=server,
        context=context,
    )

    parsed = JudgeResult.model_validate(
        {"label": row.get("label"), "reason": row.get("reason")}
    )
    if row.get("strict_correct") != int(parsed.label == "CORRECT"):
        raise RuntimeError(f"Invalid strict_correct for {example_id!r} in {path}")
    if row.get("non_major") != int(parsed.label in {"CORRECT", "MINOR_ERROR"}):
        raise RuntimeError(f"Invalid non_major for {example_id!r} in {path}")


def validate_judgment_rows(
    *,
    rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    examples_by_id: dict[str, Example],
    config: ExperimentConfig,
    model_key: str,
    condition: Condition,
    path: Path,
    server: dict[str, Any] | None = None,
    context: Literal["resume", "input"] = "resume",
) -> None:
    generation_by_id = {str(row["example_id"]): row for row in generation_rows}
    seen: set[str] = set()
    for row in rows:
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in seen:
            raise RuntimeError(
                f"Duplicate or missing example_id {example_id!r} in {path}"
            )
        seen.add(example_id)
        validate_judgment_row(
            row=row,
            generation=generation_by_id.get(example_id),
            example=examples_by_id.get(example_id),
            config=config,
            model=model_key,
            condition=condition,
            path=path,
            server=server,
            context=context,
        )


def _validate_judge_server(
    *,
    row: dict[str, Any],
    config: ExperimentConfig,
    example_id: str,
    path: Path,
    server: dict[str, Any] | None,
    context: Literal["resume", "input"],
) -> None:
    recorded = row.get("judge_server")
    if server is not None:
        if recorded != server:
            if context == "resume":
                raise RuntimeError(
                    f"Cannot resume {path}: judge_server mismatch for {example_id}. "
                    "The running Judge server does not match the recorded runtime."
                )
            raise RuntimeError(
                f"Invalid judgment {path}: judge_server mismatch for {example_id}"
            )
        return
    if recorded is None:
        return
    if not isinstance(recorded, dict) or not judge_server_matches_config(
        config, recorded
    ):
        raise RuntimeError(
            f"Invalid judgment {path}: judge_server does not match the YAML "
            f"for {example_id}"
        )
