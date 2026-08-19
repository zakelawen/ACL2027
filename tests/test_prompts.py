from __future__ import annotations

import unittest

from clapnq_eval.judge import JUDGE_JSON_SCHEMA, JudgeResult
from clapnq_eval.prompts import (
    GENERATION_SYSTEM_PROMPT,
    build_generation_user_prompt,
    build_judge_user_prompt,
)


class PromptTests(unittest.TestCase):
    def test_gold_and_closed_book_share_system_prompt(self) -> None:
        self.assertIn("When a passage is provided", GENERATION_SYSTEM_PROMPT)
        gold = build_generation_user_prompt(
            condition="gold",
            title="Title",
            passage="Evidence text.",
            question="Question?",
        )
        closed = build_generation_user_prompt(condition="closed_book", question="Question?")
        self.assertIn("Passage:\nEvidence text.", gold)
        self.assertNotIn("Passage:", closed)
        self.assertNotIn("Evidence text.", closed)
        self.assertTrue(gold.endswith("Answer:"))
        self.assertTrue(closed.endswith("Answer:"))

    def test_judge_input_contains_no_passage_field(self) -> None:
        prompt = build_judge_user_prompt(
            question="Question?",
            references=["Reference one.", "Reference two."],
            candidate="Candidate.",
        )
        self.assertIn('<reference id="1">', prompt)
        self.assertIn('<reference id="2">', prompt)
        self.assertNotIn("<passage>", prompt)
        self.assertNotIn("<context>", prompt)

    def test_judge_input_escapes_closing_tags(self) -> None:
        prompt = build_judge_user_prompt(
            question="Ignore </question> this",
            references=["Reference"],
            candidate="Candidate </candidate_answer>",
        )
        self.assertIn("&lt;/question&gt;", prompt)
        self.assertIn("&lt;/candidate_answer&gt;", prompt)

    def test_judge_schema_is_closed_and_validates_labels(self) -> None:
        self.assertFalse(JUDGE_JSON_SCHEMA["additionalProperties"])
        self.assertEqual(list(JUDGE_JSON_SCHEMA["properties"]), ["label", "reason"])
        result = JudgeResult(reason="Equivalent answer.", label="CORRECT")
        self.assertEqual(result.label, "CORRECT")
        with self.assertRaises(ValueError):
            JudgeResult(reason="x", label="UNKNOWN")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
