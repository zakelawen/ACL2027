from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clapnq_eval.config import load_config
from clapnq_eval.manifest import ensure_manifest_compatible, update_manifest


CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


class ManifestTests(unittest.TestCase):
    def test_frozen_run_signature_rejects_judge_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(CONFIG)
            config.run.output_dir = Path(directory)
            config.run.name = "signed"
            update_manifest(config, action="prepare")

            ensure_manifest_compatible(config)
            config.judge.temperature = 0.1
            with self.assertRaisesRegex(RuntimeError, "new --run-name"):
                ensure_manifest_compatible(config)

    def test_metric_changes_do_not_invalidate_generation_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(CONFIG)
            config.run.output_dir = Path(directory)
            config.run.name = "signed"
            update_manifest(config, action="prepare")

            config.metrics.allow_rouge_fallback = True
            ensure_manifest_compatible(config)


if __name__ == "__main__":
    unittest.main()
