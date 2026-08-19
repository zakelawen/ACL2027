from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from clapnq_eval.config import ExperimentConfig, ModelConfig, load_config
from clapnq_eval.data import Example
from clapnq_eval.generate import generation_parameters
from clapnq_eval.io import read_jsonl, record_sha256, write_jsonl
from clapnq_eval.judge import judge_parameters
from clapnq_eval.prompts import (
    GENERATION_PROMPT_VERSION,
    GENERATION_SYSTEM_PROMPT,
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    build_generation_user_prompt,
    build_judge_user_prompt,
    prompt_sha256,
)
from clapnq_eval.report import score_run


CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


class ReportTests(unittest.TestCase):
    def test_formal_score_rejects_missing_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, example = self._config(Path(directory))
            self._write_condition(
                config,
                example,
                condition="gold",
                answer="Reference answer.",
                label="CORRECT",
            )

            with self.assertRaisesRegex(RuntimeError, "Experiment is incomplete"):
                score_run(config)
            self.assertFalse((config.run_dir / "metrics").exists())

    def test_complete_score_reports_label_transition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, example = self._config(Path(directory))
            self._write_condition(
                config,
                example,
                condition="gold",
                answer="Reference answer.",
                label="CORRECT",
            )
            self._write_condition(
                config,
                example,
                condition="closed_book",
                answer="Wrong answer.",
                label="MAJOR_ERROR",
            )

            result = score_run(config)
            self.assertFalse(result["partial"])
            paired = json.loads(
                Path(str(result["paired"])).read_text(encoding="utf-8")
            )
            self.assertEqual(len(paired), 1)
            self.assertEqual(
                paired[0]["transition_major_error_to_correct"],
                1,
            )
            self.assertEqual(paired[0]["strict_context_gain"], 1.0)
            self.assertIn("exact_match_context_gain", paired[0])
            self.assertIn("exact_match_context_gain_ci_low", paired[0])
            self.assertIn("rouge1_f1_context_gain_ci_high", paired[0])

    def test_score_rejects_missing_judgment_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, example = self._config(Path(directory))
            self._write_complete_conditions(config, example)
            path = config.run_dir / "judgments" / "test-model.closed_book.jsonl"
            write_jsonl(path, [])
            with self.assertRaisesRegex(RuntimeError, "missing judgments"):
                score_run(config)

    def test_score_rejects_duplicate_judgment_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, example = self._config(Path(directory))
            self._write_complete_conditions(config, example)
            path = config.run_dir / "judgments" / "test-model.gold.jsonl"
            rows = list(read_jsonl(path))
            write_jsonl(path, [rows[0], rows[0]])
            with self.assertRaisesRegex(RuntimeError, "Duplicate or missing example_id"):
                score_run(config)

    def test_score_rejects_condition_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, example = self._config(Path(directory))
            self._write_complete_conditions(config, example)
            path = config.run_dir / "judgments" / "test-model.closed_book.jsonl"
            row = next(read_jsonl(path))
            row["example_id"] = "unexpected-example"
            write_jsonl(path, [row])
            with self.assertRaisesRegex(RuntimeError, "Unexpected example_id"):
                score_run(config)

    def test_score_rejects_missing_judge_server_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, example = self._config(Path(directory))
            self._write_complete_conditions(config, example)
            (config.run_dir / "judge_server.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "run-level Judge snapshot"):
                score_run(config)

    def test_score_rejects_mixed_judge_server_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, example = self._config(Path(directory))
            self._write_complete_conditions(config, example)
            path = config.run_dir / "judgments" / "test-model.closed_book.jsonl"
            row = next(read_jsonl(path))
            row["judge_server"] = {
                **self._judge_server(config),
                "random_seed": 7,
            }
            write_jsonl(path, [row])
            with self.assertRaisesRegex(RuntimeError, "judge_server mismatch"):
                score_run(config)

    def test_score_rejects_a_single_configured_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, example = self._config(Path(directory))
            config.generation.conditions = ["gold"]
            self._write_condition(
                config,
                example,
                condition="gold",
                answer="Reference answer.",
                label="CORRECT",
            )
            with self.assertRaisesRegex(RuntimeError, "gold and closed_book"):
                score_run(config)

    def test_score_ignores_quarantined_failures_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, example = self._config(Path(directory))
            self._write_complete_conditions(config, example)
            failed = (
                config.run_dir / "judgments" / "failed" / "test-model.gold.jsonl"
            )
            failed.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(failed, [{"example_id": "example-1", "error": "bad json"}])
            result = score_run(config)
            self.assertFalse(result["partial"])

    def test_score_rejects_normalized_reference_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, example = self._config(Path(directory))
            self._write_complete_conditions(config, example)
            path = config.run_dir / "judgments" / "test-model.gold.jsonl"
            row = next(read_jsonl(path))
            row["references"] = ["Stale reference answer."]
            write_jsonl(path, [row])
            with self.assertRaisesRegex(RuntimeError, "references mismatch"):
                score_run(config)

    def _write_complete_conditions(
        self, config: ExperimentConfig, example: Example
    ) -> None:
        self._write_condition(
            config,
            example,
            condition="gold",
            answer="Reference answer.",
            label="CORRECT",
        )
        self._write_condition(
            config,
            example,
            condition="closed_book",
            answer="Wrong answer.",
            label="MAJOR_ERROR",
        )

    def _config(self, root: Path) -> tuple[ExperimentConfig, Example]:
        config = load_config(CONFIG)
        config.run.name = "test"
        config.run.output_dir = root / "runs"
        config.data.normalized_path = root / "data.jsonl"
        config.models = {
            "test-model": ModelConfig(
                served_model="served-model",
                model_path=root / "fake-model",
            )
        }
        config.generation.conditions = ["gold", "closed_book"]
        config.metrics.bootstrap_samples = 100
        config.metrics.allow_rouge_fallback = True
        self._write_snapshot(config)

        example = Example(
            example_id="example-1",
            question="What is the answer?",
            title="Test",
            passage="The answer is supported here.",
            references=["Reference answer."],
            selected_sentences=["The answer is supported here."],
        )
        write_jsonl(config.data.normalized_path, [example.model_dump(mode="json")])
        return config, example

    def _write_condition(
        self,
        config: ExperimentConfig,
        example: Example,
        *,
        condition: str,
        answer: str,
        label: str,
    ) -> None:
        generation_prompt = build_generation_user_prompt(
            condition=condition,
            question=example.question,
            title=example.title,
            passage=example.passage,
        )
        generation = {
            "example_id": example.example_id,
            "source_sha256": record_sha256(example.model_dump(mode="json")),
            "model": "test-model",
            "served_model": "served-model",
            "model_path": str(config.models["test-model"].model_path),
            "condition": condition,
            "question": example.question,
            "references": example.references,
            "answer": answer,
            "prompt_version": GENERATION_PROMPT_VERSION,
            "prompt_sha256": prompt_sha256(
                GENERATION_SYSTEM_PROMPT,
                generation_prompt,
            ),
            "generation_parameters": generation_parameters(config),
        }
        generation_path = (
            config.run_dir
            / "generations"
            / f"test-model.{condition}.jsonl"
        )
        write_jsonl(generation_path, [generation])

        judge_prompt = build_judge_user_prompt(
            question=example.question,
            references=example.references,
            candidate=answer,
        )
        judgment = {
            "example_id": example.example_id,
            "source_sha256": record_sha256(example.model_dump(mode="json")),
            "generation_sha256": record_sha256(generation),
            "model": "test-model",
            "condition": condition,
            "question": example.question,
            "references": example.references,
            "answer": answer,
            "label": label,
            "reason": "Decisive semantic comparison.",
            "strict_correct": int(label == "CORRECT"),
            "non_major": int(label in {"CORRECT", "MINOR_ERROR"}),
            "judge_model": config.judge.served_model,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_prompt_sha256": prompt_sha256(
                JUDGE_SYSTEM_PROMPT,
                judge_prompt,
            ),
            "judge_parameters": judge_parameters(config),
            "judge_server": self._judge_server(config),
        }
        judgment_path = (
            config.run_dir
            / "judgments"
            / f"test-model.{condition}.jsonl"
        )
        write_jsonl(judgment_path, [judgment])

    def _judge_server(self, config: ExperimentConfig) -> dict[str, object]:
        return {
            "enable_deterministic_inference": True,
            "sglang_version": config.judge.sglang_version,
            "model_path": str(config.judge.model_path),
            "served_model_name": config.judge.served_model,
            "random_seed": config.run.seed,
        }

    def _write_snapshot(self, config: ExperimentConfig) -> None:
        path = config.run_dir / "judge_server.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._judge_server(config), sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
