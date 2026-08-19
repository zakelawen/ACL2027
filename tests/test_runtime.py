from __future__ import annotations

import unittest
from pathlib import Path

from clapnq_eval.config import load_config
from clapnq_eval.runtime import summarize_judge_server, verify_judge_server


CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def _info(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": "0.5.17",
        "enable_deterministic_inference": True,
        "model_path": "/models/Qwen3.8-27B-FP8",
        "served_model_name": "qwen3.8-27b-judge",
        "random_seed": 20260819,
        "tp_size": 2,
    }
    payload.update(overrides)
    return payload


class RuntimeTests(unittest.TestCase):
    def test_verify_accepts_matching_server(self) -> None:
        config = load_config(CONFIG)
        summary = verify_judge_server(config, _info())
        self.assertTrue(summary["enable_deterministic_inference"])
        self.assertEqual(summary["sglang_version"], "0.5.17")
        self.assertEqual(summary["model_path"], "/models/Qwen3.8-27B-FP8")

    def test_verify_rejects_deterministic_mismatch(self) -> None:
        config = load_config(CONFIG)
        with self.assertRaisesRegex(RuntimeError, "enable_deterministic_inference=False"):
            verify_judge_server(config, _info(enable_deterministic_inference=False))

    def test_verify_rejects_missing_deterministic_field(self) -> None:
        config = load_config(CONFIG)
        info = _info()
        del info["enable_deterministic_inference"]
        with self.assertRaisesRegex(RuntimeError, "did not report"):
            verify_judge_server(config, info)

    def test_verify_rejects_served_model_mismatch(self) -> None:
        config = load_config(CONFIG)
        with self.assertRaisesRegex(RuntimeError, "served_model_name"):
            verify_judge_server(config, _info(served_model_name="other-model"))

    def test_summarize_keeps_known_fields_only(self) -> None:
        summary = summarize_judge_server(
            _info(unrelated_internal_state=[{"skip": True}])
        )
        self.assertNotIn("unrelated_internal_state", summary)
        self.assertIn("random_seed", summary)


if __name__ == "__main__":
    unittest.main()
