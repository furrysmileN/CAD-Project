from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.repair_v21 import REPAIR_VERSION, repair_plan_v21


def base_plan(operation: dict) -> dict:
    return {
        "schema_version": "harnesscad.plan.v2",
        "sample_id": "s_test",
        "coordinate_system": {"units": "normalized", "origin": [0, 0, 0], "longest_bbox_edge": 1.0},
        "operations": [operation],
    }


class RepairV21Tests(unittest.TestCase):
    def test_valid_plan_is_unchanged_and_not_mutated(self) -> None:
        plan = base_plan(
            {"id": "base", "op": "cylinder", "combine": "new", "center": [0, 0, 0], "radius": 0.2, "height": 1, "axis": [0, 0, 1]}
        )
        original = copy.deepcopy(plan)
        repaired, log = repair_plan_v21(plan)
        self.assertEqual(plan, original)
        self.assertEqual(repaired, original)
        self.assertFalse(log["changed"])
        self.assertEqual(log["before_sha256"], log["after_sha256"])
        self.assertEqual(log["repair_count"], 0)
        self.assertEqual(log["repair_version"], REPAIR_VERSION)

    def test_non_unit_axes_are_normalized_only_on_schema_axis_paths(self) -> None:
        plan = base_plan(
            {"id": "base", "op": "cylinder", "combine": "new", "center": [2, 0, 0], "radius": 0.2, "height": 1, "axis": [0, 0, 2]}
        )
        repaired, log = repair_plan_v21(plan, ["unit_axis"])
        self.assertEqual(repaired["operations"][0]["center"], [2, 0, 0])
        self.assertEqual(repaired["operations"][0]["axis"], [0.0, 0.0, 1.0])
        self.assertEqual(log["repair_codes"], ["normalize_unit_axis"])
        self.assertEqual(log["changed_paths"], ["$.operations[0].axis"])

    def test_zero_and_nonfinite_axes_are_not_repaired(self) -> None:
        for vector in ([0, 0, 0], [math.inf, 0, 0]):
            plan = base_plan(
                {"id": "base", "op": "cylinder", "combine": "new", "center": [0, 0, 0], "radius": 0.2, "height": 1, "axis": vector}
            )
            repaired, log = repair_plan_v21(plan, ["unit_axis"])
            self.assertEqual(repaired, plan)
            self.assertFalse(log["changed"])

    def test_numeric_strings_are_converted_only_at_numeric_paths(self) -> None:
        plan = base_plan(
            {
                "id": "base",
                "op": "cylinder",
                "combine": "new",
                "center": ["0", "0.0", "1e-1"],
                "radius": "0.25",
                "height": "1",
                "axis": ["0", "0", "2"],
            }
        )
        plan["metadata"] = {"label": "0.5 mm"}
        repaired, log = repair_plan_v21(plan, ["number", "unit_axis"])
        operation = repaired["operations"][0]
        self.assertEqual(operation["center"], [0, 0.0, 0.1])
        self.assertEqual(operation["radius"], 0.25)
        self.assertEqual(operation["height"], 1)
        self.assertEqual(operation["axis"], [0.0, 0.0, 1.0])
        self.assertEqual(repaired["metadata"]["label"], "0.5 mm")
        self.assertIn("coerce_numeric_string", log["repair_codes"])
        self.assertIn("normalize_unit_axis", log["repair_codes"])

    def test_non_numeric_or_unit_strings_are_not_converted(self) -> None:
        plan = base_plan(
            {"id": "base", "op": "sphere", "combine": "new", "center": [0, 0, 0], "radius": "0.5 mm"}
        )
        repaired, log = repair_plan_v21(plan, ["number"])
        self.assertEqual(repaired, plan)
        self.assertFalse(log["changed"])

    def test_pattern_count_requires_integer_string(self) -> None:
        plan = base_plan(
            {
                "id": "base",
                "op": "linear_pattern",
                "combine": "new",
                "source": "prior",
                "direction": [1, 0, 0],
                "count": "3",
                "spacing": "0.2",
            }
        )
        repaired, log = repair_plan_v21(plan, ["number"])
        self.assertEqual(repaired["operations"][0]["count"], 3)
        self.assertEqual(repaired["operations"][0]["spacing"], 0.2)
        self.assertIn("coerce_integer_string", log["repair_codes"])
        plan["operations"][0]["count"] = "3.0"
        repaired, _ = repair_plan_v21(plan, ["number"])
        self.assertEqual(repaired["operations"][0]["count"], "3.0")

    def test_polygon_closes_and_removes_adjacent_duplicates(self) -> None:
        plan = base_plan(
            {
                "id": "base",
                "op": "polygon_extrude",
                "combine": "new",
                "workplane": "XY",
                "points": [[0, 0], [1, 0], [1, 0], [0, 1]],
                "depth": 0.2,
                "centered": True,
                "offset": [0, 0, 0],
            }
        )
        repaired, log = repair_plan_v21(plan, ["polygon"])
        self.assertEqual(repaired["operations"][0]["points"], [[0, 0], [1, 0], [0, 1], [0, 0]])
        self.assertEqual(
            log["repair_codes"],
            ["remove_consecutive_duplicate_vertex", "close_polygon"],
        )

    def test_self_intersecting_polygon_is_not_repaired(self) -> None:
        plan = base_plan(
            {
                "id": "base",
                "op": "polygon_extrude",
                "combine": "new",
                "workplane": "XY",
                "points": [[0, 0], [1, 1], [0, 1], [1, 0]],
                "depth": 0.2,
                "centered": True,
                "offset": [0, 0, 0],
            }
        )
        repaired, log = repair_plan_v21(plan, ["polygon"])
        self.assertEqual(repaired, plan)
        self.assertFalse(log["changed"])

    def test_missing_fields_are_not_invented(self) -> None:
        plan = base_plan({"id": "base", "op": "sphere", "combine": "new", "center": [0, 0, 0]})
        repaired, log = repair_plan_v21(plan)
        self.assertEqual(repaired, plan)
        self.assertNotIn("radius", repaired["operations"][0])
        self.assertFalse(log["changed"])

    def test_ambiguous_rotate_and_revolve_are_not_repaired(self) -> None:
        plans = [
            base_plan(
                {
                    "id": "base",
                    "op": "transform",
                    "combine": "new",
                    "source": "prior",
                    "rotate": [0, 0, 3.14159],
                }
            ),
            base_plan(
                {
                    "id": "base",
                    "op": "revolve_profile",
                    "combine": "new",
                    "workplane": "XY",
                    "profile": [[0, 0], [1, 0], [1, 1], [0, 0]],
                    "axis": [0, 0, 1],
                    "angle": 360,
                    "offset": [0, 0, 0],
                }
            ),
        ]
        for plan in plans:
            repaired, log = repair_plan_v21(plan, ["rotate_revolve"])
            self.assertEqual(repaired, plan)
            self.assertFalse(log["changed"])

    def test_unambiguous_rotate_and_revolve_aliases_are_canonicalized(self) -> None:
        plan = base_plan(
            {
                "id": "base",
                "op": "revolve_profile",
                "combine": "new",
                "workplane": "XY",
                "profile": [[0.2, 0], [0.4, 0], [0.4, 1], [0.2, 0]],
                "axis": {"axisStart": [0, 0], "axisEnd": [0, 1]},
                "angle": 360,
                "offset": [0, 0, 0],
            }
        )
        repaired, log = repair_plan_v21(plan, ["rotate_revolve"])
        self.assertEqual(repaired["operations"][0]["axis"], [[0, 0], [0, 1]])
        self.assertEqual(log["repair_codes"], ["canonicalize_revolve_axis_alias"])

    def test_repair_is_idempotent(self) -> None:
        plan = base_plan(
            {"id": "base", "op": "cylinder", "combine": "new", "center": ["0", 0, 0], "radius": "0.2", "height": 1, "axis": [0, 0, 2]}
        )
        first, first_log = repair_plan_v21(plan)
        second, second_log = repair_plan_v21(first)
        self.assertTrue(first_log["changed"])
        self.assertEqual(second, first)
        self.assertFalse(second_log["changed"])
        self.assertEqual(second_log["repair_count"], 0)


if __name__ == "__main__":
    unittest.main()
