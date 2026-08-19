from __future__ import annotations

import unittest
from pathlib import Path

from clapnq_eval.config import load_config


CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


class ConfigTests(unittest.TestCase):
    def test_official_non_thinking_judge_profile(self) -> None:
        config = load_config(CONFIG)
        self.assertFalse(config.judge.enable_thinking)
        self.assertEqual(config.judge.temperature, 0.7)
        self.assertEqual(config.judge.top_p, 0.8)
        self.assertEqual(config.judge.top_k, 20)
        self.assertEqual(config.judge.min_p, 0.0)
        self.assertEqual(config.judge.presence_penalty, 0.0)
        self.assertEqual(config.judge.repetition_penalty, 1.0)

    def test_paths_resolve_from_project_root(self) -> None:
        config = load_config(CONFIG)
        self.assertTrue(config.data.raw_path.is_absolute())
        self.assertEqual(config.run.output_dir.name, "runs")
        self.assertEqual(config.run.output_dir.parent, CONFIG.parents[1])

    def test_project_root_follows_pyproject_not_config_depth(self) -> None:
        config = load_config(CONFIG)
        self.assertEqual(config.data.raw_path.parents[2], CONFIG.parents[1])
        self.assertTrue((config.run.output_dir.parent / "pyproject.toml").is_file())


if __name__ == "__main__":
    unittest.main()
