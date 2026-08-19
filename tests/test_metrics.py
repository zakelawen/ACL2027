from __future__ import annotations

import unittest

import numpy as np

from clapnq_eval.metrics import exact_match, score_references, token_f1
from clapnq_eval.report import exact_mcnemar_p, paired_bootstrap_interval


class MetricTests(unittest.TestCase):
    def test_normalized_exact_match(self) -> None:
        self.assertEqual(exact_match("The Eiffel Tower!", "eiffel tower"), 1.0)

    def test_token_f1_partial_overlap(self) -> None:
        self.assertGreater(token_f1("alpha beta", "alpha gamma"), 0.0)
        self.assertLess(token_f1("alpha beta", "alpha gamma"), 1.0)

    def test_multiple_references_take_maximum(self) -> None:
        scores = score_references(
            "Paris is the answer.",
            ["London.", "Paris is the answer."],
            allow_rouge_fallback=True,
        )
        self.assertEqual(scores["exact_match"], 1.0)
        self.assertEqual(scores["token_f1"], 1.0)
        self.assertEqual(scores["rouge_l_f1"], 1.0)

    def test_paired_bootstrap_constant_difference(self) -> None:
        left = np.ones(8)
        right = np.zeros(8)
        low, high = paired_bootstrap_interval(
            left,
            right,
            samples=200,
            confidence_level=0.95,
            seed=7,
        )
        self.assertEqual(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_exact_mcnemar(self) -> None:
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(5, 0), 0.0625)


if __name__ == "__main__":
    unittest.main()
