from __future__ import annotations

import unittest

from clapnq_eval.api import (
    ChatCompletionError,
    _is_retryable_error,
    completed_chat_text,
    retry_async,
)


class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_retryable_validation_error_fails_immediately(self) -> None:
        calls = 0

        async def operation() -> None:
            nonlocal calls
            calls += 1
            raise ValueError("invalid structured output")

        with self.assertRaisesRegex(ValueError, "invalid structured output"):
            await retry_async(
                operation,
                attempts=5,
                base_seconds=0.001,
                description="validation",
            )
        self.assertEqual(calls, 1)

    def test_httpx_connection_errors_are_retryable(self) -> None:
        import httpx

        self.assertTrue(
            _is_retryable_error(httpx.ConnectError("connection refused"))
        )
        request = httpx.Request("GET", "http://127.0.0.1/server_info")
        response = httpx.Response(503, request=request)
        self.assertTrue(_is_retryable_error(httpx.HTTPStatusError("busy", request=request, response=response)))
        self.assertFalse(_is_retryable_error(httpx.HTTPStatusError("bad", request=request, response=httpx.Response(400, request=request))))


class CompletedTextTests(unittest.TestCase):
    def test_stop_text_is_stripped(self) -> None:
        self.assertEqual(
            completed_chat_text("  answer  ", finish_reason="stop"),
            "answer",
        )

    def test_truncated_generation_is_accepted_when_allowed(self) -> None:
        text = completed_chat_text(
            "partial answer",
            finish_reason="length",
            allow_truncated=True,
        )
        self.assertEqual(text, "partial answer")

    def test_truncated_generation_is_rejected_by_default(self) -> None:
        with self.assertRaises(ChatCompletionError) as ctx:
            completed_chat_text("partial answer", finish_reason="length")
        self.assertEqual(ctx.exception.finish_reason, "length")

    def test_empty_truncated_text_is_rejected(self) -> None:
        with self.assertRaises(ChatCompletionError):
            completed_chat_text("   ", finish_reason="length", allow_truncated=True)


if __name__ == "__main__":
    unittest.main()
