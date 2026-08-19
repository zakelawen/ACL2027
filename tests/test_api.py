from __future__ import annotations

import unittest

from clapnq_eval.api import retry_async


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


if __name__ == "__main__":
    unittest.main()
