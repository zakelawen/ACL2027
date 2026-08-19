from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class ChatCompletionError(ValueError):
    """Non-retryable completion failure that still carries the raw payload."""

    def __init__(
        self,
        message: str,
        *,
        content: str | None = None,
        finish_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.content = content
        self.finish_reason = finish_reason

_EXTRA_BODY_KEYS = frozenset({
    "top_k",
    "min_p",
    "repetition_penalty",
    "chat_template_kwargs",
})


@dataclass(frozen=True)
class ChatResult:
    content: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_seconds: float

    def usage_dict(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def openai_compat_root(base_url: str) -> str:
    return str(base_url).rstrip("/").removesuffix("/v1")


class ChatClient:
    def __init__(self, *, base_url: str, api_key: str, timeout_seconds: float) -> None:
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    async def close(self) -> None:
        await self._client.close()

    async def require_model(self, model: str) -> None:
        response = await self._client.models.list()
        available = {item.id for item in response.data}
        if model not in available:
            rendered = ", ".join(sorted(available)) or "<none>"
            raise RuntimeError(f"Model {model!r} is not served. Available models: {rendered}")

    async def get_server_info(self) -> dict[str, Any]:
        import httpx

        root = openai_compat_root(self._base_url)
        last_status: int | None = None
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as http:
            for path in ("/server_info", "/get_server_info"):
                response = await http.get(root + path)
                last_status = response.status_code
                if response.status_code != 200:
                    continue
                payload = response.json()
                if not isinstance(payload, dict):
                    raise RuntimeError(
                        f"Judge {path} returned a non-object payload"
                    )
                return payload
        raise RuntimeError(
            f"Judge /server_info is unavailable (HTTP {last_status})"
        )

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        presence_penalty: float | None = None,
        frequency_penalty: float = 0.0,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "seed": seed,
        }
        if presence_penalty is not None:
            request["presence_penalty"] = presence_penalty
        if response_format is not None:
            request["response_format"] = response_format
        if extra_body:
            unexpected = set(extra_body) - _EXTRA_BODY_KEYS
            if unexpected:
                raise ValueError(f"Unsupported extra_body keys: {sorted(unexpected)}")
            overlap = set(extra_body) & set(request)
            if overlap:
                raise ValueError(f"extra_body duplicates standard parameters: {sorted(overlap)}")
            request["extra_body"] = extra_body

        started = time.perf_counter()
        response = await self._client.chat.completions.create(**request)
        latency = time.perf_counter() - started
        if not response.choices:
            raise ChatCompletionError("Chat completion returned no choices")
        choice = response.choices[0]
        raw_content = choice.message.content
        text = raw_content if isinstance(raw_content, str) else None
        if choice.finish_reason == "length":
            raise ChatCompletionError(
                "Chat completion was truncated at the token limit",
                content=text,
                finish_reason=choice.finish_reason,
            )
        if text is None:
            raise ChatCompletionError(
                "Chat completion returned no text content",
                content=None,
                finish_reason=choice.finish_reason,
            )

        usage = response.usage
        return ChatResult(
            content=text.strip(),
            finish_reason=choice.finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            latency_seconds=latency,
        )


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_seconds: float,
    description: str,
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as error:
            if not _is_retryable_error(error):
                raise
            last_error = error
            if attempt == attempts:
                break
            exponential = base_seconds * (2 ** (attempt - 1))
            delay = exponential * random.uniform(0.8, 1.2)
            LOGGER.warning(
                "%s failed (%d/%d): %s; retrying in %.1fs",
                description,
                attempt,
                attempts,
                error,
                delay,
            )
            await asyncio.sleep(delay)
    assert last_error is not None
    raise RuntimeError(f"{description} failed after {attempts} attempts") from last_error


def _is_retryable_error(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
    return False

