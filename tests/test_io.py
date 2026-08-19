from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from clapnq_eval.io import (
    append_jsonl,
    completed_ids,
    drop_resolved_rows,
    read_jsonl,
    sort_jsonl,
    write_jsonl,
)


class JsonlTests(unittest.TestCase):
    def test_write_append_and_resume_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            write_jsonl(path, [{"example_id": "1", "value": 1}])
            append_jsonl(path, {"example_id": "2", "value": 2})
            self.assertEqual(completed_ids(path), {"1", "2"})
            self.assertEqual(len(list(read_jsonl(path))), 2)

    def test_incomplete_final_line_is_tolerated_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"example_id":"1"}\n{"example_id":', encoding="utf-8")
            rows = list(read_jsonl(path, tolerate_trailing_partial=True))
            self.assertEqual(rows, [{"example_id": "1"}])

            append_jsonl(path, {"example_id": "2"})
            self.assertEqual(
                list(read_jsonl(path)),
                [{"example_id": "1"}, {"example_id": "2"}],
            )

    def test_invalid_terminated_final_line_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"ok":1}\nnot-json\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                list(read_jsonl(path, tolerate_trailing_partial=True))

    def test_sort_jsonl_follows_requested_id_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            write_jsonl(
                path,
                [{"example_id": "b"}, {"example_id": "a"}, {"example_id": "c"}],
            )
            sort_jsonl(path, ["a", "b", "c"])
            self.assertEqual(
                [row["example_id"] for row in read_jsonl(path)],
                ["a", "b", "c"],
            )

    def test_drop_resolved_rows_removes_file_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.jsonl"
            write_jsonl(
                path,
                [{"example_id": "1"}, {"example_id": "2"}],
            )
            drop_resolved_rows(path, {"1"})
            self.assertEqual(
                [row["example_id"] for row in read_jsonl(path)],
                ["2"],
            )
            drop_resolved_rows(path, {"2"})
            self.assertFalse(path.exists())

    def test_utf8_truncated_tail_is_tolerated_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            complete = json.dumps({"example_id": "1", "text": "ok"}, ensure_ascii=False)
            # Truncate a multibyte character mid-sequence (U+4E2D is e4 b8 ad).
            truncated = b'{"example_id":"2","text":"' + b"\xe4\xb8"
            path.write_bytes((complete + "\n").encode("utf-8") + truncated)
            rows = list(read_jsonl(path, tolerate_trailing_partial=True))
            self.assertEqual(rows, [{"example_id": "1", "text": "ok"}])

            append_jsonl(path, {"example_id": "2", "text": "recovered"})
            self.assertEqual(
                [row["example_id"] for row in read_jsonl(path)],
                ["1", "2"],
            )

    def test_invalid_middle_line_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"ok":1}\nnot-json\n{"ok":2}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                list(read_jsonl(path, tolerate_trailing_partial=True))


if __name__ == "__main__":
    unittest.main()
