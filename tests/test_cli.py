from __future__ import annotations

import unittest
from pathlib import Path

from clapnq_eval.cli import build_parser, _select_conditions
from clapnq_eval.config import load_config


CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


class CliTests(unittest.TestCase):
    def test_all_conditions_follow_the_yaml(self) -> None:
        config = load_config(CONFIG)
        parser = build_parser()
        self.assertEqual(
            _select_conditions("all", config, parser),
            ["gold", "closed_book"],
        )
        self.assertEqual(
            _select_conditions("gold", config, parser),
            ["gold"],
        )

    def test_unknown_condition_is_rejected(self) -> None:
        config = load_config(CONFIG)
        parser = build_parser()
        with self.assertRaises(SystemExit):
            _select_conditions("topk_5", config, parser)

    def test_all_follows_a_custom_condition_list(self) -> None:
        config = load_config(CONFIG)
        config.generation.conditions = ["gold"]
        parser = build_parser()
        self.assertEqual(_select_conditions("all", config, parser), ["gold"])


if __name__ == "__main__":
    unittest.main()
