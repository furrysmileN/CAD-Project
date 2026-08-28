from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.v6_feature_scorer import score_critical_fact
from rq2_harness.v6b_pair_generator import (
    PAIR_KINDS,
    generate_pair,
    generate_pairs,
    operations_differ_only_in_critical,
)


class V6bPairGeneratorTests(unittest.TestCase):
    def test_four_kinds_and_single_fact_diff(self) -> None:
        pairs = generate_pairs(12)
        kinds = {pair["kind"] for pair in pairs}
        self.assertEqual(kinds, set(PAIR_KINDS))
        for pair in pairs:
            self.assertTrue(operations_differ_only_in_critical(pair), pair["pair_id"])
            self.assertNotEqual(pair["spec_a"]["critical_fact"]["value"], pair["spec_b"]["critical_fact"]["value"])
            self.assertEqual(pair["spec_a"]["critical_fact"]["category"], pair["spec_b"]["critical_fact"]["category"])
            self.assertEqual(pair["spec_a"]["critical_fact"]["fact_id"], pair["spec_b"]["critical_fact"]["fact_id"])

    def test_oracle_separates_a_and_b_plans(self) -> None:
        for index in range(4):
            pair = generate_pair(index)
            latent_a = pair["spec_a"]
            self.assertTrue(score_critical_fact(pair["spec_a"]["gt_plan"], latent_a)["exact"], pair["kind"])
            self.assertFalse(score_critical_fact(pair["spec_b"]["gt_plan"], latent_a)["exact"], pair["kind"])

    def test_hidden_b_has_back_hole(self) -> None:
        pair = generate_pair(2)
        self.assertEqual(pair["kind"], "hidden_presence")
        ids_a = {op["id"] for op in pair["spec_a"]["operations"]}
        ids_b = {op["id"] for op in pair["spec_b"]["operations"]}
        self.assertNotIn("back_hole", ids_a)
        self.assertIn("back_hole", ids_b)
        self.assertFalse(pair["spec_a"]["critical_fact"]["value"])
        self.assertTrue(pair["spec_b"]["critical_fact"]["value"])

    def test_through_pair_b_is_blind(self) -> None:
        pair = generate_pair(1)
        self.assertEqual(pair["kind"], "through_vs_blind")
        self.assertEqual(pair["spec_a"]["critical_fact"]["value"], "through")
        self.assertEqual(pair["spec_b"]["critical_fact"]["value"], "blind")


if __name__ == "__main__":
    unittest.main()
