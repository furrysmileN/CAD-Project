from __future__ import annotations

import unittest

from .plan_v2_compiler import compile_plan_v2
from .plan_v31_schema import validate_plan_v31


def _plan(path_wire: list[dict]) -> dict:
    return {
        "schema_version": "harnesscad.plan.v3.1",
        "sample_id": "arc_oracle",
        "coordinate_system": {
            "units": "normalized",
            "origin": [0, 0, 0],
            "longest_bbox_edge": 1.0,
        },
        "operations": [
            {
                "id": "arc_sweep",
                "op": "sweep_profile",
                "combine": "new",
                "workplane": "YZ",
                "wire": [
                    {"kind": "move", "to": [0.1, 0]},
                    {
                        "kind": "three_point_arc",
                        "through": [0, 0.1],
                        "to": [-0.1, 0],
                    },
                    {
                        "kind": "three_point_arc",
                        "through": [0, -0.1],
                        "to": [0.1, 0],
                    },
                ],
                "path_wire": path_wire,
                "sweep_mode": "frenet",
                "offset": [0.5, 0, 0.3],
            }
        ],
    }


class PlanV31Tests(unittest.TestCase):
    def test_valid_3d_arc_compiles_to_typed_cadquery_edge(self) -> None:
        plan = _plan(
            [
                {"kind": "move", "to": [0.5, 0, 0.3]},
                {"kind": "line", "to": [0.2, 0, 0.3]},
                {
                    "kind": "three_point_arc",
                    "through": [-0.0121, 0, 0.2121],
                    "to": [-0.1, 0, 0],
                },
                {"kind": "line", "to": [-0.1, 0, -0.5]},
            ]
        )
        self.assertEqual(validate_plan_v31(plan), {"valid": True, "issues": []})
        source = compile_plan_v2(plan)
        self.assertIn("cq.Edge.makeThreePointArc", source)
        self.assertIn("isFrenet=op.get", source)

    def test_collinear_arc_and_invalid_mode_fail_closed(self) -> None:
        plan = _plan(
            [
                {"kind": "move", "to": [0, 0, 0]},
                {
                    "kind": "three_point_arc",
                    "through": [0.5, 0, 0],
                    "to": [1, 0, 0],
                },
            ]
        )
        plan["operations"][0]["sweep_mode"] = "guess"
        validation = validate_plan_v31(plan)
        self.assertFalse(validation["valid"])
        self.assertEqual(
            {issue["code"] for issue in validation["issues"]},
            {"degenerate_path_arc", "invalid_sweep_mode"},
        )


if __name__ == "__main__":
    unittest.main()
