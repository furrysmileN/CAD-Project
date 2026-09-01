from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.v6_feature_scorer import score_critical_fact
from rq2_harness.v6b_diag_conditions import fact_sentence
from rq2_harness.v7_pair_generator import (
    FORBIDDEN_FAMILIES,
    HOSTS,
    KINDS,
    MIN_OPS,
    assert_not_v6b_shell,
    generate_pair,
    generate_pairs,
    operations_differ_only_in_critical,
)


class V7PairGeneratorTests(unittest.TestCase):
    def test_hosts_and_kinds_cross_not_old_shells(self) -> None:
        pairs = generate_pairs(16)
        hosts = {pair["host"] for pair in pairs}
        kinds = {pair["kind"] for pair in pairs}
        self.assertEqual(hosts, set(HOSTS))
        self.assertEqual(kinds, set(KINDS))
        self.assertTrue(hosts.isdisjoint(FORBIDDEN_FAMILIES))
        for pair in pairs:
            assert_not_v6b_shell(pair)
            self.assertGreaterEqual(len(pair["spec_a"]["operations"]), MIN_OPS)
            self.assertTrue(operations_differ_only_in_critical(pair), pair["pair_id"])
            self.assertNotEqual(pair["spec_a"]["critical_fact"]["value"], pair["spec_b"]["critical_fact"]["value"])
            ids = {op["id"] for op in pair["spec_a"]["operations"]}
            self.assertGreaterEqual(len(ids), MIN_OPS)

    def test_l2_l3_live_on_new_hosts(self) -> None:
        l2 = [generate_pair(i) for i in range(16) if generate_pair(i)["kind"] == "through_vs_blind"]
        l3 = [generate_pair(i) for i in range(16) if generate_pair(i)["kind"] == "hidden_presence"]
        self.assertGreaterEqual(len({p["host"] for p in l2}), 3)
        self.assertGreaterEqual(len({p["host"] for p in l3}), 3)
        for pair in l2:
            self.assertEqual(pair["spec_b"]["critical_fact"]["value"], "blind")
        for pair in l3:
            self.assertFalse(pair["spec_a"]["critical_fact"]["value"])
            self.assertTrue(pair["spec_b"]["critical_fact"]["value"])
            self.assertIn("back_hole", {op["id"] for op in pair["spec_b"]["operations"]})

    def test_oracle_separates_a_and_b(self) -> None:
        for index in range(16):
            pair = generate_pair(index)
            latent_a = pair["spec_a"]
            self.assertTrue(score_critical_fact(pair["spec_a"]["gt_plan"], latent_a)["exact"], pair["pair_id"])
            self.assertFalse(score_critical_fact(pair["spec_b"]["gt_plan"], latent_a)["exact"], pair["pair_id"])

    def test_flange_and_shaft_present(self) -> None:
        families = {generate_pair(i)["family"] for i in range(16)}
        self.assertIn("flange_neck", families)
        self.assertIn("stepped_shaft_collar", families)

    def test_tb_sentence_still_works(self) -> None:
        pair = generate_pair(0)
        text = fact_sentence(pair["kind"], pair["spec_b"])
        self.assertNotIn("v7_probe_", text)
        self.assertNotIn(pair["family"], text)


class V7RunnerGuardTests(unittest.TestCase):
    def test_refuses_non_v7_output_root(self) -> None:
        from rq2_harness.v7_runner import run_v7

        with self.assertRaises(ValueError):
            run_v7(
                Path(__file__).resolve().parents[1] / "configs" / "v6b_diag_c2b.yaml",
                dry_run=True,
                limit=1,
            )


if __name__ == "__main__":
    unittest.main()
