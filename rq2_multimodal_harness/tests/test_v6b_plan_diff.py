from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.v6b_plan_diff import classify_c3_c5, critical_plan_edit, unrelated_plan_edit_count


class V6bPlanDiffTests(unittest.TestCase):
    def test_pocket_depth_change_is_critical_edit(self) -> None:
        latent = {
            "critical_fact": {
                "fact_id": "pocket.depth",
                "category": "depth",
                "value": 0.12,
                "operation_id": "pocket",
            }
        }
        c3 = {
            "operations": [
                {"id": "block", "op": "box", "combine": "new", "size": [0.92, 0.70, 0.40]},
                {"id": "pocket", "op": "box", "combine": "cut", "size": [0.34, 0.26, 0.12]},
                {"id": "end_slot", "op": "slot", "combine": "cut", "depth": 0.10},
            ]
        }
        c5 = {
            "operations": [
                {"id": "block", "op": "box", "combine": "new", "size": [0.92, 0.70, 0.40]},
                {"id": "pocket", "op": "box", "combine": "cut", "size": [0.34, 0.26, 0.24]},
                {"id": "end_slot", "op": "slot", "combine": "cut", "depth": 0.10},
            ]
        }
        self.assertTrue(critical_plan_edit(c3, c5, latent))
        self.assertEqual(unrelated_plan_edit_count(c3, c5, latent), 0)
        row = classify_c3_c5(c3, c5, latent, kind="pocket_depth", gt_b=0.24)
        self.assertEqual(row["category_label"], "critical_changed_to_B")

    def test_unrelated_only_size_change(self) -> None:
        latent = {
            "critical_fact": {
                "fact_id": "pocket.depth",
                "category": "depth",
                "value": 0.12,
                "operation_id": "pocket",
            }
        }
        c3 = {
            "operations": [
                {"id": "block", "op": "box", "combine": "new", "size": [0.92, 0.70, 0.40]},
                {"id": "pocket", "op": "box", "combine": "cut", "size": [0.34, 0.26, 0.12]},
            ]
        }
        c5 = {
            "operations": [
                {"id": "block", "op": "box", "combine": "new", "size": [0.88, 0.70, 0.40]},
                {"id": "pocket", "op": "box", "combine": "cut", "size": [0.34, 0.26, 0.12]},
            ]
        }
        self.assertFalse(critical_plan_edit(c3, c5, latent))
        self.assertEqual(unrelated_plan_edit_count(c3, c5, latent), 1)
        row = classify_c3_c5(c3, c5, latent, kind="pocket_depth", gt_b=0.24)
        self.assertEqual(row["category_label"], "unrelated_only")

    def test_missing_back_hole(self) -> None:
        latent = {
            "critical_fact": {
                "fact_id": "back_hole.hidden_presence",
                "category": "hidden_presence",
                "value": False,
                "operation_id": "back_hole",
            }
        }
        c3 = {
            "operations": [
                {"id": "body", "op": "box", "combine": "new", "size": [0.80, 0.50, 0.42]},
            ]
        }
        c5 = {
            "operations": [
                {"id": "body", "op": "box", "combine": "new", "size": [0.82, 0.50, 0.42]},
            ]
        }
        row = classify_c3_c5(c3, c5, latent, kind="hidden_presence", gt_b=True)
        self.assertEqual(row["category_label"], "missing_required_op")
        self.assertFalse(row["critical_plan_edit"])
        self.assertGreaterEqual(row["unrelated_plan_edit_count"], 1)

    def test_through_mode_not_updated(self) -> None:
        latent = {
            "critical_fact": {
                "fact_id": "hole_blind.through_vs_blind",
                "category": "through_vs_blind",
                "value": "through",
                "operation_id": "hole_blind",
            }
        }
        c3 = {
            "operations": [
                {"id": "base", "op": "box", "combine": "new", "size": [1.0, 0.72, 0.28]},
                {"id": "through_hole", "op": "hole", "combine": "cut", "depth": 0.28, "workplane": "XY"},
            ]
        }
        c5 = {
            "operations": [
                {"id": "base", "op": "box", "combine": "new", "size": [1.0, 0.70, 0.28]},
                {"id": "through_hole", "op": "hole", "combine": "cut", "depth": 0.28, "workplane": "XY"},
            ]
        }
        row = classify_c3_c5(c3, c5, latent, kind="through_vs_blind", gt_b="blind")
        self.assertEqual(row["category_label"], "mode_not_updated")


if __name__ == "__main__":
    unittest.main()
