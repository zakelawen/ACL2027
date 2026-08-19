from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from tqdm import tqdm

from .api import ChatClient, ChatCompletionError, retry_async
from .runtime import ensure_judge_server_snapshot, verify_judge_server
from .config import Condition, ExperimentConfig
from .data import Example, load_examples
from .io import (
    append_jsonl,
    drop_resolved_rows,
    exclusive_output_lock,
    read_jsonl,
    record_sha256,
    sort_jsonl,
)
from .prompts import (
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_prompt,
    prompt_sha256,
)
from .schema import JUDGE_JSON_SCHEMA, JudgeLabel, JudgeResult
from .validate import (
    judge_parameters,
    validate_generation_rows,
    validate_judgment_rows,
)


LOGGER = logging.getLogger(__name__)

__all__ = [
    "JUDGE_JSON_SCHEMA",
    "JudgeLabel",
    "JudgeResult",
    "judge_answers",
    "judge_parameters",
    "validate_generation_rows",
]


class JudgeExampleError(Exception):
    """Per-example structured-output failure that should be quarantined."""

    def __init__(
        self,
        message: str,
        *,
        raw_content: str | None = None,
        finish_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_content = raw_content
        self.finish_reason = finish_reason


def is_quarantine_error(error: BaseException) -> bool:
    return isinstance(
        error,
        (
            JudgeExampleError,
            ChatCompletionError,
            ValidationError,
            json.JSONDecodeError,
        ),
    )


async def judge_answers(
    config: ExperimentConfig,
    *,
    model_keys: list[str],
    conditions: list[Condition],
    limit: int | None = None,
) -> list[Path]:
    unknown = set(model_keys) - set(config.models)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    examples = load_examples(config.data.normalized_path)
    examples_by_id = {example.example_id: example for example in examples}
    client = ChatClient(
        base_url=config.judge.base_url,
        api_key=config.judge.api_key,
        timeout_seconds=config.judge.request_timeout_seconds,
    )
    try:
        await retry_async(
            lambda: client.require_model(config.judge.served_model),
            attempts=config.judge.max_retries,
            base_seconds=config.judge.retry_base_seconds,
            description=f"judge server check for {config.judge.served_model}",
        )
        info = await retry_async(
            client.get_server_info,
            attempts=config.judge.max_retries,
            base_seconds=config.judge.retry_base_seconds,
            description="judge /server_info",
        )
        server = verify_judge_server(config, info)
        server = ensure_judge_server_snapshot(config, server)
        outputs: list[Path] = []
        for model_key in model_keys:
            for condition in conditions:
                success_path, failed_path = await _judge_file(
                    config=config,
                    client=client,
                    examples_by_id=examples_by_id,
                    model_key=model_key,
                    condition=condition,
                    limit=limit,
                    server=server,
                )
                outputs.append(success_path)
                if failed_path.exists():
                    outputs.append(failed_path)
        return outputs
    finally:
        await client.close()


async def _judge_file(
    *,
    config: ExperimentConfig,
    client: ChatClient,
    examples_by_id: dict[str, Example],
    model_key: str,
    condition: Condition,
    limit: int | None,
    server: dict[str, Any],
) -> tuple[Path, Path]:
    generation_path = config.run_dir / "generations" / f"{model_key}.{condition}.jsonl"
    if not generation_path.exists():
        raise FileNotFoundError(f"Missing generation file: {generation_path}")

    all_rows = list(read_jsonl(generation_path, tolerate_trailing_partial=True))
    validate_generation_rows(
        rows=all_rows,
        examples_by_id=examples_by_id,
        config=config,
        model_key=model_key,
        condition=condition,
        path=generation_path,
        context="input",
    )
    rows = all_rows[:limit] if limit is not None else all_rows
    ordered_ids = [str(row["example_id"]) for row in all_rows]

    output_path = config.run_dir / "judgments" / f"{model_key}.{condition}.jsonl"
    failed_path = (
        config.run_dir / "judgments" / "failed" / f"{model_key}.{condition}.jsonl"
    )
    with exclusive_output_lock(output_path):
        existing = (
            list(read_jsonl(output_path, tolerate_trailing_partial=True))
            if output_path.exists()
            else []
        )
        validate_judgment_rows(
            rows=existing,
            generation_rows=all_rows,
            examples_by_id=examples_by_id,
            config=config,
            model_key=model_key,
            condition=condition,
            path=output_path,
            server=server,
            context="resume",
        )
        done = {str(row["example_id"]) for row in existing}
        drop_resolved_rows(failed_path, done)
        pending = [row for row in rows if str(row["example_id"]) not in done]
        semaphore = asyncio.Semaphore(config.judge.concurrency)
        parameters = judge_parameters(config)

        async def worker(row: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                example_id = str(row["example_id"])
                example = examples_by_id[example_id]
                candidate = str(row["answer"])
                user_prompt = build_judge_user_prompt(
                    question=example.question,
                    references=example.references,
                    candidate=candidate,
                )
                messages = [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]

                async def request_and_validate() -> tuple[JudgeResult, Any]:
                    response = await client.chat(
                        model=config.judge.served_model,
                        messages=messages,
                        max_tokens=config.judge.max_tokens,
                        temperature=config.judge.temperature,
                        top_p=config.judge.top_p,
                        seed=config.run.seed,
                        presence_penalty=config.judge.presence_penalty,
                        frequency_penalty=0.0,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "correctness_judge",
                                "strict": True,
                                "schema": JUDGE_JSON_SCHEMA,
                            },
                        },
                        extra_body={
                            "top_k": config.judge.top_k,
                            "min_p": config.judge.min_p,
                            "repetition_penalty": config.judge.repetition_penalty,
                            "chat_template_kwargs": {
                                "enable_thinking": config.judge.enable_thinking,
                            },
                        },
                    )
                    try:
                        parsed = JudgeResult.model_validate(
                            json.loads(response.content)
                        )
                    except (json.JSONDecodeError, ValidationError) as error:
                        raise JudgeExampleError(
                            f"invalid structured judge output: {error}",
                            raw_content=response.content,
                            finish_reason=response.finish_reason,
                        ) from error
                    return parsed, response

                try:
                    parsed, response = await retry_async(
                        request_and_validate,
                        attempts=config.judge.max_retries,
                        base_seconds=config.judge.retry_base_seconds,
                        description=f"judge {model_key}/{condition}/{example_id}",
                    )
                except Exception as error:
                    if not is_quarantine_error(error):
                        raise
                    LOGGER.warning(
                        "Quarantining judge failure %s/%s/%s: %s",
                        model_key,
                        condition,
                        example_id,
                        error,
                    )
                    return {
                        "status": "failed",
                        "row": {
                            "example_id": example_id,
                            "model": model_key,
                            "condition": condition,
                            "answer": candidate,
                            "error": str(error),
                            "error_type": type(error).__name__,
                            "raw_judge_response": getattr(error, "raw_content", None)
                            or getattr(error, "content", None),
                            "finish_reason": getattr(error, "finish_reason", None),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    }

                return {
                    "status": "ok",
                    "row": {
                        "example_id": example_id,
                        "source_sha256": record_sha256(
                            example.model_dump(mode="json")
                        ),
                        "generation_sha256": record_sha256(row),
                        "model": model_key,
                        "condition": condition,
                        "question": example.question,
                        "references": example.references,
                        "answer": candidate,
                        "label": parsed.label,
                        "reason": parsed.reason.strip(),
                        "raw_judge_response": response.content,
                        "strict_correct": int(parsed.label == "CORRECT"),
                        "non_major": int(
                            parsed.label in {"CORRECT", "MINOR_ERROR"}
                        ),
                        "judge_model": config.judge.served_model,
                        "judge_prompt_version": JUDGE_PROMPT_VERSION,
                        "judge_prompt_sha256": prompt_sha256(
                            JUDGE_SYSTEM_PROMPT, user_prompt
                        ),
                        "judge_parameters": parameters,
                        "judge_server": server,
                        "finish_reason": response.finish_reason,
                        "usage": response.usage_dict(),
                        "latency_seconds": response.latency_seconds,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                }

        tasks = [asyncio.create_task(worker(row)) for row in pending]
        completed_in_scope = sum(
            str(row["example_id"]) in done for row in rows
        )
        with tqdm(
            total=len(rows),
            initial=completed_in_scope,
            desc=f"judge {model_key}/{condition}",
        ) as bar:
            try:
                for future in asyncio.as_completed(tasks):
                    result = await future
                    if result["status"] == "ok":
                        append_jsonl(output_path, result["row"])
                        done.add(str(result["row"]["example_id"]))
                    else:
                        append_jsonl(failed_path, result["row"])
                    bar.update(1)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        drop_resolved_rows(failed_path, done)
        sort_jsonl(output_path, ordered_ids)
        if failed_path.exists():
            sort_jsonl(failed_path, ordered_ids)
            remaining = sum(1 for _ in read_jsonl(failed_path))
            LOGGER.warning(
                "Judge finished %s/%s with %d quarantined failure(s) in %s",
                model_key,
                condition,
                remaining,
                failed_path,
            )
        return output_path, failed_path
