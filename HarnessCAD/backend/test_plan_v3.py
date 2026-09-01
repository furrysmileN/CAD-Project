from __future__ import annotations

import unittest

from .harness_api_v2 import RunRequest, compile_plan_v2, run_endpoint, validate_plan_v2
from .plan_v3_schema import validate_plan_v3


def plan_v3(operations: list[dict]) -> dict:
    return {
        "schema_version": "harnesscad.plan.v3",
        "sample_id": "plan_v3_test",
        "coordinate_system": {
            "units": "normalized",
            "origin": [0.0, 0.0, 0.0],
            "longest_bbox_edge": 1.0,
        },
        "operations": operations,
    }


CLOSED_SQUARE = [[-0.15, -0.15], [0.15, -0.15], [0.15, 0.15], [-0.15, 0.15], [-0.15, -0.15]]
ARC_WIRE = [
    {"kind": "move", "to": [-0.2, 0.0]},
    {"kind": "three_point_arc", "through": [0.0, 0.2], "to": [0.2, 0.0]},
    {"kind": "three_point_arc", "through": [0.0, -0.2], "to": [-0.2, 0.0]},
]


class PlanV3SchemaTests(unittest.TestCase):
    def test_v2_still_rejects_sweep(self):
        plan = {
            "schema_version": "harnesscad.plan.v2",
            "sample_id": "old",
            "coordinate_system": {"units": "normalized", "origin": [0.0, 0.0, 0.0], "longest_bbox_edge": 1.0},
            "operations": [
                {
                    "id": "tube",
                    "op": "sweep_profile",
                    "combine": "new",
                    "workplane": "XY",
                    "profile": CLOSED_SQUARE,
                    "path": [[0.0, 0.0, -0.3], [0.0, 0.0, 0.3]],
                    "offset": [0.0, 0.0, 0.0],
                }
            ],
        }
        self.assertFalse(validate_plan_v2(plan)["valid"])

    def test_path_xor_helix(self):
        op = {
            "id": "tube",
            "op": "sweep_profile",
            "combine": "new",
            "workplane": "XY",
            "profile": CLOSED_SQUARE,
            "path": [[0.0, 0.0, -0.3], [0.0, 0.0, 0.3]],
            "helix": {"radius": 0.2, "pitch": 0.08, "turns": 2.0, "axis": [0.0, 0.0, 1.0], "center": [0.0, 0.0, 0.0]},
            "offset": [0.0, 0.0, 0.0],
        }
        self.assertFalse(validate_plan_v3(plan_v3([op]))["valid"])


class PlanV3ExecutionTests(unittest.TestCase):
    def run_success(self, operations: list[dict]) -> dict:
        result = run_endpoint(RunRequest(plan=plan_v3(operations), timeout_sec=30.0))
        self.assertIn(result["status"], {"success", "success_with_warnings"}, result.get("error"))
        self.assertIsNotNone(result["stepUrl"])
        return result

    def test_arc_extrude(self):
        self.run_success(
            [
                {
                    "id": "disc",
                    "op": "polygon_extrude",
                    "combine": "new",
                    "workplane": "XY",
                    "wire": ARC_WIRE,
                    "depth": 0.12,
                    "centered": True,
                    "offset": [0.0, 0.0, 0.0],
                }
            ]
        )

    def test_sweep_path(self):
        self.run_success(
            [
                {
                    "id": "tube",
                    "op": "sweep_profile",
                    "combine": "new",
                    "workplane": "XY",
                    "profile": CLOSED_SQUARE,
                    "path": [[0.0, 0.0, -0.35], [0.0, 0.15, 0.0], [0.0, 0.0, 0.35]],
                    "offset": [0.0, 0.0, 0.0],
                }
            ]
        )

    def test_sweep_helix(self):
        self.run_success(
            [
                {
                    "id": "spring",
                    "op": "sweep_profile",
                    "combine": "new",
                    "workplane": "XY",
                    "profile": [[0.18, -0.03], [0.24, -0.03], [0.24, 0.03], [0.18, 0.03], [0.18, -0.03]],
                    "helix": {
                        "radius": 0.21,
                        "pitch": 0.08,
                        "turns": 3.0,
                        "axis": [0.0, 0.0, 1.0],
                        "center": [0.0, 0.0, 0.0],
                    },
                    "offset": [0.0, 0.0, 0.0],
                }
            ]
        )

    def test_loft(self):
        self.run_success(
            [
                {
                    "id": "taper",
                    "op": "loft_profiles",
                    "combine": "new",
                    "workplane": "XY",
                    "profiles": [
                        {
                            "points": [[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3], [-0.3, -0.3]],
                            "offset": [0.0, 0.0, -0.2],
                        },
                        {
                            "points": [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1]],
                            "offset": [0.0, 0.0, 0.2],
                        },
                    ],
                }
            ]
        )

    def test_compile_deterministic(self):
        plan = plan_v3(
            [
                {
                    "id": "tube",
                    "op": "sweep_profile",
                    "combine": "new",
                    "workplane": "XY",
                    "profile": CLOSED_SQUARE,
                    "path": [[0.0, 0.0, -0.2], [0.0, 0.0, 0.2]],
                    "offset": [0.0, 0.0, 0.0],
                }
            ]
        )
        self.assertEqual(compile_plan_v2(plan), compile_plan_v2(plan))
