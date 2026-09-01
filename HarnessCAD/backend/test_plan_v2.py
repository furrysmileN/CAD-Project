from __future__ import annotations

import copy
import unittest

from .harness_api_v2 import RunRequest, compile_plan_v2, run_endpoint, validate_plan_v2
from .test_harness_v2 import base_plan as base_plan_v1


def plan_v2(operations: list[dict]) -> dict:
    return {
        "schema_version": "harnesscad.plan.v2",
        "sample_id": "plan_v2_test",
        "coordinate_system": {
            "units": "normalized",
            "origin": [0.0, 0.0, 0.0],
            "longest_bbox_edge": 1.0,
        },
        "metadata": {"suite": "unittest"},
        "operations": operations,
    }


def box_operation() -> dict:
    return {
        "id": "base",
        "op": "box",
        "combine": "new",
        "center": [0.0, 0.0, 0.0],
        "size": [1.0, 0.6, 0.2],
    }


class PlanV2SchemaTests(unittest.TestCase):
    def test_compilation_is_deterministic(self):
        plan = plan_v2([box_operation()])
        self.assertEqual(compile_plan_v2(plan), compile_plan_v2(copy.deepcopy(plan)))

    def test_extra_field_is_rejected(self):
        plan = plan_v2([box_operation()])
        plan["operations"][0]["python"] = "import os"
        validation = validate_plan_v2(plan)
        self.assertFalse(validation["valid"])
        self.assertIn("extra_field", {issue["code"] for issue in validation["issues"]})

    def test_forward_and_unknown_references_are_rejected(self):
        transform = {
            "id": "moved",
            "op": "transform",
            "combine": "new",
            "source": "later",
            "translate": [0.1, 0.0, 0.0],
        }
        validation = validate_plan_v2(plan_v2([transform, box_operation()]))
        self.assertFalse(validation["valid"])
        self.assertIn("invalid_reference", {issue["code"] for issue in validation["issues"]})

    def test_bad_coordinate_and_non_unit_axis_are_rejected(self):
        operation = {
            "id": "bad",
            "op": "cylinder",
            "combine": "new",
            "center": [9.0, 0.0, 0.0],
            "radius": 0.2,
            "height": 1.0,
            "axis": [0.0, 0.0, 0.5],
        }
        validation = validate_plan_v2(plan_v2([operation]))
        self.assertFalse(validation["valid"])
        codes = {issue["code"] for issue in validation["issues"]}
        self.assertIn("invalid_number", codes)
        self.assertIn("axis_not_unit_length", codes)

    def test_open_and_self_intersecting_polygons_are_rejected(self):
        operation = {
            "id": "poly",
            "op": "polygon_extrude",
            "combine": "new",
            "workplane": "XY",
            "points": [[-0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5]],
            "depth": 0.2,
            "centered": True,
            "offset": [0.0, 0.0, 0.0],
        }
        validation = validate_plan_v2(plan_v2([operation]))
        self.assertFalse(validation["valid"])
        self.assertIn("polygon_not_closed", {issue["code"] for issue in validation["issues"]})


class PlanV2ExecutionTests(unittest.TestCase):
    def run_success(self, operations: list[dict]) -> dict:
        result = run_endpoint(RunRequest(plan=plan_v2(operations), timeout_sec=30.0))
        self.assertIn(result["status"], {"success", "success_with_warnings"}, result.get("error"))
        self.assertIsNotNone(result["stepUrl"])
        self.assertIsNotNone(result["modelUrl"])
        self.assertEqual(len(result["operationTrace"]), len(operations))
        return result

    def test_polygon_extrude_success(self):
        self.run_success(
            [
                {
                    "id": "polygon",
                    "op": "polygon_extrude",
                    "combine": "new",
                    "workplane": "XY",
                    "points": [[-0.5, -0.3], [0.5, -0.3], [0.5, 0.3], [-0.5, 0.3], [-0.5, -0.3]],
                    "depth": 0.2,
                    "centered": True,
                    "offset": [0.0, 0.0, 0.0],
                }
            ]
        )

    def test_revolve_profile_success(self):
        self.run_success(
            [
                {
                    "id": "revolved",
                    "op": "revolve_profile",
                    "combine": "new",
                    "workplane": "XY",
                    "profile": [[0.0, -0.5], [0.2, -0.5], [0.2, 0.5], [0.0, 0.5], [0.0, -0.5]],
                    "axis": [[0.0, -1.0], [0.0, 1.0]],
                    "angle": 360.0,
                    "offset": [0.0, 0.0, 0.0],
                }
            ]
        )

    def test_global_hole_and_slot_success(self):
        self.run_success(
            [
                box_operation(),
                {
                    "id": "round_hole",
                    "op": "hole",
                    "combine": "cut",
                    "workplane": "XZ",
                    "center": [-0.25, 0.0, 0.0],
                    "diameter": 0.16,
                    "depth": 0.8,
                },
                {
                    "id": "straight_slot",
                    "op": "slot",
                    "combine": "cut",
                    "workplane": "YZ",
                    "center": [0.25, 0.0, 0.0],
                    "length": 0.25,
                    "width": 0.1,
                    "depth": 1.2,
                    "angle": 90.0,
                },
            ]
        )

    def test_translate_and_rotate_transform_success(self):
        base = box_operation()
        base["size"] = [0.5, 0.3, 0.2]
        self.run_success(
            [
                base,
                {
                    "id": "moved",
                    "op": "transform",
                    "combine": "add",
                    "source": "base",
                    "translate": [0.25, 0.0, 0.0],
                    "rotate": {"origin": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0], "angle": 180.0},
                },
            ]
        )

    def test_fillet_success(self):
        self.run_success(
            [box_operation(), {"id": "rounded", "op": "fillet", "radius": 0.03, "edge_axis": "Z"}]
        )

    def test_chamfer_success(self):
        self.run_success([box_operation(), {"id": "beveled", "op": "chamfer", "distance": 0.03}])

    def test_linear_pattern_success(self):
        seed = box_operation()
        seed["center"] = [-0.4, 0.0, 0.0]
        seed["size"] = [0.2, 0.3, 0.2]
        self.run_success(
            [
                seed,
                {
                    "id": "row",
                    "op": "linear_pattern",
                    "combine": "add",
                    "source": "base",
                    "direction": [1.0, 0.0, 0.0],
                    "count": 5,
                    "spacing": 0.2,
                },
            ]
        )

    def test_empty_geometry_is_traced(self):
        plan = plan_v2(
            [
                box_operation(),
                {
                    "id": "remove_all",
                    "op": "box",
                    "combine": "cut",
                    "center": [0.0, 0.0, 0.0],
                    "size": [2.0, 2.0, 2.0],
                },
            ]
        )
        result = run_endpoint(RunRequest(plan=plan, timeout_sec=30.0))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure"]["code"], "empty_after_operation")
        self.assertEqual(result["failure"]["operationId"], "remove_all")

    def test_episode_v2_still_runs_plan_v1(self):
        result = run_endpoint(RunRequest(plan=base_plan_v1(), timeout_sec=30.0))
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["operationTrace"]), 2)


if __name__ == "__main__":
    unittest.main()
