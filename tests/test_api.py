from __future__ import annotations

import unittest

from clapnq_eval.api import _is_retryable_error, retry_async


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


if __name__ == "__main__":
    unittest.main()
