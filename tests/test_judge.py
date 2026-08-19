from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from clapnq_eval.api import ChatCompletionError
from clapnq_eval.judge import JudgeExampleError, is_quarantine_error
from clapnq_eval.schema import JudgeResult


class JudgeQuarantineTests(unittest.TestCase):
    def test_structured_output_failures_are_quarantined(self) -> None:
        self.assertTrue(is_quarantine_error(JudgeExampleError("bad json")))
        self.assertTrue(
            is_quarantine_error(
                ChatCompletionError("truncated", content="{", finish_reason="length")
            )
        )
        self.assertTrue(is_quarantine_error(json.JSONDecodeError("bad", "{", 0)))
        try:
            JudgeResult.model_validate({"label": "CORRECT"})
        except ValidationError as error:
            self.assertTrue(is_quarantine_error(error))

    def test_infrastructure_failures_are_not_quarantined(self) -> None:
        self.assertFalse(is_quarantine_error(RuntimeError("failed after 5 attempts")))
        self.assertFalse(is_quarantine_error(FileNotFoundError("missing")))


if __name__ == "__main__":
    unittest.main()
