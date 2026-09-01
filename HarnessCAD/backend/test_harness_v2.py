from __future__ import annotations

import copy
import unittest

from .harness_api_v2 import RunRequest, run_endpoint


def base_plan() -> dict:
    return {
        "schema_version": "harnesscad.plan.v1",
        "sample_id": "v2_test",
        "coordinate_system": {
            "units": "normalized",
            "origin": [0.0, 0.0, 0.0],
            "longest_bbox_edge": 1.0,
        },
        "operations": [
            {
                "id": "base",
                "primitive": "box",
                "combine": "new",
                "center": [0.0, 0.0, 0.0],
                "size": [1.0, 0.6, 0.2],
            },
            {
                "id": "hole",
                "primitive": "cylinder",
                "combine": "cut",
                "center": [0.0, 0.0, 0.0],
                "radius": 0.1,
                "height": 0.4,
                "axis": [0.0, 0.0, 1.0],
            },
        ],
    }


class HarnessV2Tests(unittest.TestCase):
    def run_plan(self, plan: dict) -> dict:
        return run_endpoint(RunRequest(plan=plan, timeout_sec=30.0))

    def test_success_trace(self):
        result = self.run_plan(base_plan())
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["operationTrace"]), 2)
        self.assertTrue(result["metrics"]["canonicalFrame"])
        self.assertEqual(result["metrics"]["solidCount"], 1)

    def test_empty_cut_is_classified(self):
        plan = base_plan()
        plan["operations"][0]["size"] = [1.0, 1.6, 0.2]
        plan["operations"][1]["radius"] = 1.8
        result = self.run_plan(plan)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure"]["code"], "empty_after_operation")
        self.assertEqual(result["failure"]["operationId"], "hole")
        self.assertEqual(result["operationTrace"][-1]["after"]["solidCount"], 0)

    def test_noncanonical_geometry_is_warning(self):
        plan = base_plan()
        plan["operations"][0]["size"] = [1.0, 1.6, 0.2]
        result = self.run_plan(plan)
        self.assertEqual(result["status"], "success_with_warnings")
        self.assertFalse(result["metrics"]["canonicalFrame"])
        warning_codes = {warning["code"] for warning in result["warnings"]}
        self.assertIn("declared_scale_mismatch_likely", warning_codes)
        self.assertIn("noncanonical_final_geometry", warning_codes)

    def test_disconnected_add_is_recorded(self):
        plan = base_plan()
        plan["operations"] = plan["operations"][:1]
        plan["operations"].append(
            {
                "id": "remote_sphere",
                "primitive": "sphere",
                "combine": "add",
                "center": [1.5, 0.0, 0.0],
                "radius": 0.2,
            }
        )
        result = self.run_plan(plan)
        self.assertEqual(result["status"], "success_with_warnings")
        self.assertEqual(result["metrics"]["solidCount"], 2)
        self.assertIn("multiple_solids_after_operation", result["operationTrace"][-1]["warnings"])

    def test_invalid_plan_stops_before_compile(self):
        plan = base_plan()
        plan["operations"][0]["combine"] = "cut"
        result = self.run_plan(plan)
        self.assertEqual(result["status"], "validation_failed")
        self.assertEqual(result["failure"]["code"], "plan_validation_failed")
        self.assertEqual(result["operationTrace"], [])


if __name__ == "__main__":
    unittest.main()
