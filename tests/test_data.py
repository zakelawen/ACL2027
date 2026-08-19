from __future__ import annotations

import unittest
from pathlib import Path

from clapnq_eval.data import parse_clapnq_record
from clapnq_eval.io import read_jsonl


FIXTURE = Path(__file__).parent / "fixtures" / "sample.jsonl"


class DataTests(unittest.TestCase):
    def test_official_fields_are_mapped(self) -> None:
        first = next(read_jsonl(FIXTURE))
        example = parse_clapnq_record(first)
        self.assertEqual(example.example_id, "-547956488374826249")
        self.assertEqual(example.question, "What is the answer?")
        self.assertEqual(example.title, "Example")
        self.assertEqual(example.passage, "The answer is alpha.")
        self.assertEqual(example.references, ["The answer is alpha."])

    def test_skip_na_and_duplicate_references_are_removed(self) -> None:
        rows = list(read_jsonl(FIXTURE))
        example = parse_clapnq_record(rows[2])
        self.assertEqual(
            example.references,
            ["The object is blue.", "Blue is the described color."],
        )
        self.assertEqual(example.selected_sentences, ["The object is blue."])

    def test_no_valid_reference_raises(self) -> None:
        record = {
            "id": "x",
            "input": "Question?",
            "passages": [{"title": "T", "text": "P"}],
            "output": [
                {"answer": "NA", "meta": {"skip": False}},
                {"answer": "", "meta": {"skip": True}},
            ],
        }
        with self.assertRaisesRegex(ValueError, "reference"):
            parse_clapnq_record(record)

    def test_multiple_gold_passages_raise(self) -> None:
        record = {
            "id": "x",
            "input": "Question?",
            "passages": [{"title": "A", "text": "A"}, {"title": "B", "text": "B"}],
            "output": [{"answer": "Answer", "meta": {"skip": False}}],
        }
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_clapnq_record(record)


if __name__ == "__main__":
    unittest.main()
