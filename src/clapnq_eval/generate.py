from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .api import ChatClient, ChatCompletionError, retry_async
from .runtime import (
    verify_generator_server,
    verify_model_identity,
    versions_match,
)
from .config import Condition, ExperimentConfig
from .data import Example, load_examples
from .io import (
    append_jsonl,
    exclusive_output_lock,
    read_jsonl,
    record_sha256,
    sort_jsonl,
)
from .prompts import (
    GENERATION_PROMPT_VERSION,
    GENERATION_SYSTEM_PROMPT,
    build_generation_user_prompt,
    prompt_sha256,
)
from .validate import generation_parameters, validate_generation_rows


LOGGER = logging.getLogger(__name__)


async def generate_answers(
    config: ExperimentConfig,
    *,
    model_key: str,
    conditions: list[Condition],
    limit: int | None = None,
) -> list[Path]:
    if model_key not in config.models:
        choices = ", ".join(sorted(config.models))
        raise ValueError(f"Unknown model {model_key!r}. Choose from: {choices}")
    invalid = set(conditions) - set(config.generation.conditions)
    if invalid:
        raise ValueError(f"Conditions are disabled in config: {sorted(invalid)}")

    all_examples = load_examples(config.data.normalized_path)
    examples = all_examples[:limit] if limit is not None else all_examples
    examples_by_id = {example.example_id: example for example in all_examples}

    served_model = config.models[model_key].served_model
    client = ChatClient(
        base_url=config.generation.base_url,
        api_key=config.generation.api_key,
        timeout_seconds=config.generation.request_timeout_seconds,
    )
    try:
        await retry_async(
            lambda: client.require_model(served_model),
            attempts=config.generation.max_retries,
            base_seconds=config.generation.retry_base_seconds,
            description=f"generator server check for {served_model}",
        )
        await _verify_live_generator(config, client, model_key, served_model)
        outputs: list[Path] = []
        for condition in conditions:
            output = await _generate_condition(
                config=config,
                client=client,
                examples=examples,
                all_examples=all_examples,
                examples_by_id=examples_by_id,
                model_key=model_key,
                served_model=served_model,
                condition=condition,
            )
            outputs.append(output)
        return outputs
    finally:
        await client.close()


async def _generate_condition(
    *,
    config: ExperimentConfig,
    client: ChatClient,
    examples: list[Example],
    all_examples: list[Example],
    examples_by_id: dict[str, Example],
    model_key: str,
    served_model: str,
    condition: Condition,
) -> Path:
    output_path = config.run_dir / "generations" / f"{model_key}.{condition}.jsonl"
    with exclusive_output_lock(output_path):
        existing = (
            list(read_jsonl(output_path, tolerate_trailing_partial=True))
            if output_path.exists()
            else []
        )
        validate_generation_rows(
            rows=existing,
            examples_by_id=examples_by_id,
            config=config,
            model_key=model_key,
            served_model=served_model,
            condition=condition,
            path=output_path,
            context="resume",
        )
        done = {str(row["example_id"]) for row in existing}
        pending = [example for example in examples if example.example_id not in done]
        semaphore = asyncio.Semaphore(config.generation.concurrency)
        parameters = generation_parameters(config)

        async def worker(example: Example) -> dict[str, Any]:
            async with semaphore:
                user_prompt = build_generation_user_prompt(
                    condition=condition,
                    question=example.question,
                    title=example.title,
                    passage=example.passage,
                )
                messages = [
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
                try:
                    result = await retry_async(
                        lambda: client.chat(
                            model=served_model,
                            messages=messages,
                            max_tokens=config.generation.max_tokens,
                            temperature=config.generation.temperature,
                            top_p=config.generation.top_p,
                            seed=config.run.seed,
                            presence_penalty=config.generation.presence_penalty,
                            frequency_penalty=0.0,
                            extra_body={
                                "top_k": config.generation.top_k,
                                "min_p": config.generation.min_p,
                                "repetition_penalty": config.generation.repetition_penalty,
                            },
                        ),
                        attempts=config.generation.max_retries,
                        base_seconds=config.generation.retry_base_seconds,
                        description=(
                            f"generation {model_key}/{condition}/{example.example_id}"
                        ),
                    )
                except ChatCompletionError as error:
                    raise ChatCompletionError(
                        f"{error} ({model_key}/{condition}/{example.example_id})",
                        content=error.content,
                        finish_reason=error.finish_reason,
                    ) from error
                return {
                    "example_id": example.example_id,
                    "source_sha256": record_sha256(example.model_dump(mode="json")),
                    "model": model_key,
                    "served_model": served_model,
                    "model_path": str(config.models[model_key].model_path),
                    "condition": condition,
                    "question": example.question,
                    "references": example.references,
                    "answer": result.content,
                    "finish_reason": result.finish_reason,
                    "usage": result.usage_dict(),
                    "latency_seconds": result.latency_seconds,
                    "prompt_version": GENERATION_PROMPT_VERSION,
                    "prompt_sha256": prompt_sha256(
                        GENERATION_SYSTEM_PROMPT, user_prompt
                    ),
                    "generation_parameters": parameters,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }

        tasks = [asyncio.create_task(worker(example)) for example in pending]
        completed_in_scope = sum(
            example.example_id in done for example in examples
        )
        failures: list[BaseException] = []
        with tqdm(
            total=len(examples),
            initial=completed_in_scope,
            desc=f"generate {model_key}/{condition}",
        ) as bar:
            try:
                for future in asyncio.as_completed(tasks):
                    try:
                        row = await future
                    except Exception as error:
                        failures.append(error)
                        LOGGER.error(
                            "generation failed for %s/%s: %s",
                            model_key,
                            condition,
                            error,
                        )
                        bar.update(1)
                        continue
                    append_jsonl(output_path, row)
                    bar.update(1)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        if failures:
            preview = "; ".join(str(error) for error in failures[:8])
            extra = "" if len(failures) <= 8 else f" (+{len(failures) - 8} more)"
            raise RuntimeError(
                f"{len(failures)} generation(s) failed for {model_key}/{condition}: "
                f"{preview}{extra}"
            )

        sort_jsonl(output_path, [example.example_id for example in all_examples])
        return output_path


async def _verify_live_generator(
    config: ExperimentConfig,
    client: ChatClient,
    model_key: str,
    served_model: str,
) -> None:
    model_config = config.models[model_key]
    info: dict[str, Any] | None = None
    try:
        info = await retry_async(
            client.get_server_info,
            attempts=config.generation.max_retries,
            base_seconds=config.generation.retry_base_seconds,
            description=f"generator /server_info for {served_model}",
        )
    except RuntimeError:
        info = None
    card = await retry_async(
        lambda: client.get_model_card(served_model),
        attempts=config.generation.max_retries,
        base_seconds=config.generation.retry_base_seconds,
        description=f"generator /v1/models for {served_model}",
    )
    verify_generator_server(
        expected_path=model_config.model_path,
        served_model=served_model,
        info=info,
        model_card=card,
    )
    verify_model_identity(model_config.model_path, model_config.model_identity)
    version = await retry_async(
        client.get_engine_version,
        attempts=config.generation.max_retries,
        base_seconds=config.generation.retry_base_seconds,
        description=f"generator /version for {served_model}",
    )
    if not versions_match(version, model_config.vllm_version):
        raise RuntimeError(
            f"Generator vLLM version {version!r} does not match "
            f"models.{model_key}.vllm_version={model_config.vllm_version!r}."
        )
