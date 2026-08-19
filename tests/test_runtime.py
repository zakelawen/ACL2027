from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from clapnq_eval.config import load_config
from clapnq_eval.runtime import (
    ensure_judge_server_snapshot,
    summarize_judge_server,
    verify_generator_server,
    verify_judge_server,
)


CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def _info(config=None, **overrides: object) -> dict[str, object]:
    config = config or load_config(CONFIG)
    payload: dict[str, object] = {
        "version": config.judge.sglang_version,
        "enable_deterministic_inference": True,
        "model_path": str(config.judge.model_path),
        "served_model_name": config.judge.served_model,
        "random_seed": config.run.seed,
        "tp_size": 2,
    }
    payload.update(overrides)
    return payload


class RuntimeTests(unittest.TestCase):
    def test_verify_accepts_matching_server(self) -> None:
        config = load_config(CONFIG)
        summary = verify_judge_server(config, _info(config))
        self.assertTrue(summary["enable_deterministic_inference"])
        self.assertEqual(summary["sglang_version"], config.judge.sglang_version)
        self.assertEqual(summary["model_path"], str(config.judge.model_path))

    def test_verify_rejects_deterministic_mismatch(self) -> None:
        config = load_config(CONFIG)
        with self.assertRaisesRegex(RuntimeError, "enable_deterministic_inference=False"):
            verify_judge_server(config, _info(config, enable_deterministic_inference=False))

    def test_verify_rejects_seed_mismatch(self) -> None:
        config = load_config(CONFIG)
        with self.assertRaisesRegex(RuntimeError, "random_seed=7"):
            verify_judge_server(config, _info(config, random_seed=7))

    def test_verify_rejects_model_path_mismatch(self) -> None:
        config = load_config(CONFIG)
        with self.assertRaisesRegex(RuntimeError, "model_path"):
            verify_judge_server(config, _info(config, model_path="/tmp/other-weights"))

    def test_verify_rejects_version_mismatch(self) -> None:
        config = load_config(CONFIG)
        with self.assertRaisesRegex(RuntimeError, "sglang_version"):
            verify_judge_server(config, _info(config, version="0.0.0"))

    def test_verify_rejects_missing_deterministic_field(self) -> None:
        config = load_config(CONFIG)
        info = _info(config)
        del info["enable_deterministic_inference"]
        with self.assertRaisesRegex(RuntimeError, "did not report"):
            verify_judge_server(config, info)

    def test_verify_rejects_served_model_mismatch(self) -> None:
        config = load_config(CONFIG)
        with self.assertRaisesRegex(RuntimeError, "served_model_name"):
            verify_judge_server(config, _info(config, served_model_name="other-model"))

    def test_summarize_keeps_known_fields_only(self) -> None:
        summary = summarize_judge_server(
            _info(unrelated_internal_state=[{"skip": True}])
        )
        self.assertNotIn("unrelated_internal_state", summary)
        self.assertIn("random_seed", summary)

    def test_ensure_snapshot_is_create_once(self) -> None:
        config = load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            config.run.output_dir = Path(directory)
            config.run.name = "signed"
            first = verify_judge_server(config, _info(config))
            ensure_judge_server_snapshot(config, first)
            ensure_judge_server_snapshot(config, first)
            other = dict(first)
            other["random_seed"] = 7
            with self.assertRaisesRegex(RuntimeError, "does not match the live server"):
                ensure_judge_server_snapshot(config, other)
            stored = json.loads(
                (config.run_dir / "judge_server.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored, first)

    def test_generator_rejects_reported_path_mismatch(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected"):
            verify_generator_server(
                expected_path=Path("/mnt/model/Qwen2.5-7B-Instruct"),
                served_model="qwen2.5-7b",
                info=None,
                model_card={"root": "/tmp/other-weights"},
            )


if __name__ == "__main__":
    unittest.main()
