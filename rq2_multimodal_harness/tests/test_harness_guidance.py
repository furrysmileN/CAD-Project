from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.feedback import resolve_feedback_config
from rq2_harness.harness_guidance import (
    GENERATOR_EXTRUDE,
    GENERATOR_SWEEP_PLANE,
    build_guidance,
    infer_decisions,
    infer_pose,
)
from rq2_harness.hvc_oracles import ORACLES
from rq2_harness.pc_prompting import build_pc_messages
from rq2_harness.prompting import parse_plan_response, validate_plan

LEAK_MARKERS = (
    "family",
    "stem",
    "slotted_plate",
    "cable_routing_panel",
    "pipe_elbow",
    "benchcad",
)


def _images(root: Path, prefix: str) -> list[dict[str, str]]:
    result = []
    for index, view in enumerate(("front", "side", "top", "isometric")):
        path = root / f"{prefix}_{view}.png"
        Image.new("RGB", (8, 8), (index * 40, 20, 10)).save(path)
        result.append({"view": view, "path": str(path), "sha256": f"{prefix}-{view}"})
    return result


def _row(root: Path) -> dict:
    return {
        "sample_id": "secret_family_0001",
        "family": "secret_family",
        "difficulty": "hard",
        "gt_code": {"path": str(root / "secret.py"), "sha256": "gt-secret-hash"},
        "step": {"path": str(root / "secret.step"), "sha256": "gt-step-hash"},
        "text": {"L1": "short plate", "L3": "1. Extrude a 20 mm plate"},
        "text_encodings": {"T1": {"text": "short plate"}, "T2": {"text": "1. Extrude a 20 mm plate"}},
        "render_encodings": {"I1": {"images": _images(root, "I1"), "params": {}}},
        "point_cloud": {
            "path": str(root / "cloud.npy"),
            "sha256": "point-hash",
            "encoding": {"images": _images(root, "P"), "params": {"encoding_version": "rq2.pc_depth_contour.v1"}},
        },
    }


def _plan(operations: list[dict]) -> dict:
    return {
        "schema_version": "harnesscad.plan.v3",
        "sample_id": "s_guidance",
        "coordinate_system": {"units": "normalized", "origin": [0, 0, 0], "longest_bbox_edge": 1.0},
        "operations": operations,
    }


class PoseGuidanceTests(unittest.TestCase):
    def test_slotted_plate_bbox_is_standing(self) -> None:
        pose = infer_pose([0.026, 1.0, 0.53])
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertEqual(pose["thin_axis"], "X")
        self.assertEqual(pose["workplane"], "YZ")
        self.assertEqual(pose["shape_class"], "plate")

    def test_cable_panel_bbox_is_flat(self) -> None:
        pose = infer_pose([1.0, 0.45, 0.01])
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertEqual(pose["thin_axis"], "Z")
        self.assertEqual(pose["workplane"], "XY")
        self.assertEqual(pose["shape_class"], "plate")

    def test_guidance_payload_has_no_family_leak(self) -> None:
        evidence = {
            "frame": {"bbox_size": [0.026, 1.0, 0.53]},
            "sections": {
                "YZ": {"outer": {"bbox_size": [1.0, 0.53]}, "holes": []},
            },
        }
        payload = build_guidance(evidence)
        blob = json.dumps(payload, ensure_ascii=False).lower()
        for marker in LEAK_MARKERS:
            self.assertNotIn(marker, blob, marker)
        self.assertNotIn("round-tube", blob)
        self.assertNotIn("[structure_recipes]", blob)

    def test_decisions_are_extent_not_part_names(self) -> None:
        plate = infer_decisions(bbox_size=[0.026, 1.0, 0.53])
        self.assertEqual(plate["generator"]["id"], GENERATOR_EXTRUDE)
        self.assertEqual(plate["topology"]["kind"], "unmeasured")
        self.assertFalse(plate["topology"]["inner_radius_known"])

        two_long = infer_decisions(
            {
                "frame": {"bbox_size": [1.0, 0.389, 1.0]},
                "sections": {
                    "XY": {"outer": {"bbox_size": [0.40, 0.39]}, "holes": []},
                    "XZ": {"outer": {"bbox_size": [1.0, 1.0]}, "holes": []},
                    "YZ": {"outer": {"bbox_size": [0.39, 0.41]}, "holes": []},
                },
            }
        )
        self.assertEqual(two_long["generator"]["id"], GENERATOR_SWEEP_PLANE)
        self.assertEqual(two_long["pose"]["workplane"], "XZ")
        self.assertEqual(two_long["topology"]["kind"], "solid")
        self.assertFalse(two_long["topology"]["inner_radius_known"])

        hollow = infer_decisions(
            {
                "frame": {"bbox_size": [1.0, 1.0, 0.2]},
                "sections": {
                    "XY": {
                        "outer": {"bbox_size": [1.0, 1.0]},
                        "holes": [{"radius": 0.22, "confidence": 0.8}],
                    }
                },
            }
        )
        self.assertEqual(hollow["topology"]["kind"], "hollow")
        self.assertTrue(hollow["topology"]["inner_radius_known"])
        self.assertEqual(hollow["sizes"]["hole_radii"], [0.22])

    def test_offset_ring_finds_hollow_without_family_name(self) -> None:
        rng = np.random.default_rng(0)
        theta = rng.uniform(0.0, 2.0 * np.pi, 900)
        radius = rng.choice([0.22, 0.30], 900)
        height = rng.uniform(-0.45, 0.45, 900)
        points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), height))
        hollow = infer_decisions(
            {"frame": {"bbox_size": [0.6, 0.6, 1.0]}, "sections": {}},
            points=points,
        )
        self.assertEqual(hollow["topology"]["kind"], "hollow")
        self.assertTrue(hollow["topology"]["inner_radius_known"])
        self.assertGreater(hollow["topology"]["inner_radius"], 0.1)
        self.assertGreater(hollow["topology"]["outer_radius"], hollow["topology"]["inner_radius"])

        box = rng.uniform(-0.4, 0.4, size=(900, 3))
        solid = infer_decisions(
            {"frame": {"bbox_size": [0.8, 0.8, 0.8]}, "sections": {}},
            points=box,
        )
        self.assertEqual(solid["topology"]["kind"], "solid")
        self.assertFalse(solid["topology"]["inner_radius_known"])

        theta = rng.uniform(0.0, 2.0 * np.pi, 800)
        height = rng.uniform(-0.45, 0.45, 800)
        skin = np.column_stack((0.28 * np.cos(theta), 0.28 * np.sin(theta), height))
        skin_solid = infer_decisions(
            {"frame": {"bbox_size": [0.56, 0.56, 0.9]}, "sections": {}},
            points=skin,
        )
        self.assertEqual(skin_solid["topology"]["kind"], "solid")

        blob = json.dumps(hollow, ensure_ascii=False).lower()
        for marker in LEAK_MARKERS:
            self.assertNotIn(marker, blob, marker)


class PromptV5Tests(unittest.TestCase):
    def test_v5_messages_include_pose_and_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = {
                "schema": "point_evidence.v1",
                "cloud_id": "c_test",
                "content_hash": "abc",
                "frame": {"center": [0, 0, 0], "bbox_size": [0.026, 1.0, 0.53]},
                "quality": {"point_count": 2048, "valid_ratio": 1.0, "degenerate": False},
                "sections": {},
                "hypotheses": [],
                "uncertainties": [],
            }
            messages, audit = build_pc_messages(
                _row(root),
                "I1P_geom",
                evidence=evidence,
                image_max_edge=8,
                plan_prompt_version="v5",
            )
            serialized = json.dumps(messages, ensure_ascii=False)
            self.assertIn("[POSE]", serialized)
            self.assertIn("[CONSTRUCTION_LAWS]", serialized)
            self.assertIn("[DECISIONS]", serialized)
            self.assertIn("sweep_profile", serialized)
            self.assertNotIn("[STRUCTURE_RECIPES]", serialized)
            self.assertNotIn("Round-tube bend", serialized)
            self.assertEqual(audit["plan_prompt_version"], "v5")
            self.assertEqual(audit["guidance"]["pose"]["thin_axis"], "X")
            self.assertEqual(audit["guidance"]["pose"]["workplane"], "YZ")
            self.assertEqual(audit["guidance"]["decisions"]["generator"], GENERATOR_EXTRUDE)
            for marker in ("slotted_plate", "secret_family", "gt-secret"):
                self.assertNotIn(marker, serialized)

    def test_i1_does_not_read_point_cloud_topology(self) -> None:
        rng = np.random.default_rng(0)
        theta = rng.uniform(0.0, 2.0 * np.pi, 900)
        radius = rng.choice([0.22, 0.30], 900)
        height = rng.uniform(-0.45, 0.45, 900)
        cloud = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), height))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "cloud.npy", cloud)
            row = _row(root)
            i1_messages, i1_audit = build_pc_messages(
                row,
                "I1",
                image_max_edge=8,
                plan_prompt_version="v5",
            )
            i1_text = json.dumps(i1_messages, ensure_ascii=False)
            self.assertIn("[CONSTRUCTION_LAWS]", i1_text)
            self.assertNotIn("[DECISIONS]\nThis-task answers", i1_text)
            self.assertNotIn("[POSE]", i1_text)
            self.assertNotIn("offset slice shows an outer ring", i1_text)
            self.assertIsNone(i1_audit["guidance"]["pose"])
            self.assertEqual(i1_audit["guidance"]["decisions"]["topology"], "unmeasured")

            evidence = {
                "schema": "point_evidence.v1",
                "cloud_id": "c_test",
                "content_hash": "abc",
                "frame": {"center": [0, 0, 0], "bbox_size": [0.6, 0.6, 1.0]},
                "quality": {"point_count": 900, "valid_ratio": 1.0, "degenerate": False},
                "sections": {},
                "hypotheses": [],
                "uncertainties": [],
            }
            geom_messages, geom_audit = build_pc_messages(
                row,
                "P_geom",
                evidence=evidence,
                image_max_edge=8,
                plan_prompt_version="v5",
            )
            geom_text = json.dumps(geom_messages, ensure_ascii=False)
            self.assertIn("[DECISIONS]", geom_text)
            self.assertIn("hollow", geom_text)
            self.assertEqual(geom_audit["guidance"]["decisions"]["topology"], "hollow")

    def test_legal_sweep_and_ring_revolve_pass_v5(self) -> None:
        sweep = ORACLES[5]
        self.assertEqual(validate_plan(sweep, "v5"), [])
        parsed = parse_plan_response(json.dumps(sweep), plan_version="v5")
        self.assertTrue(parsed["ok"], parsed.get("issues"))

        ring = _plan(
            [
                {
                    "id": "elbow",
                    "op": "revolve_profile",
                    "combine": "new",
                    "workplane": "XY",
                    "profile": [[0.22, -0.03], [0.30, -0.03], [0.30, 0.03], [0.22, 0.03], [0.22, -0.03]],
                    "axis": [[0, 0], [0, 1]],
                    "angle": 90,
                    "offset": [0, 0, 0],
                }
            ]
        )
        self.assertEqual(validate_plan(ring, "v5"), [])

    def test_solid_rectangle_revolve_still_legal(self) -> None:
        solid = _plan(
            [
                {
                    "id": "chunk",
                    "op": "revolve_profile",
                    "combine": "new",
                    "workplane": "XY",
                    "profile": [[0.05, -0.2], [0.4, -0.2], [0.4, 0.2], [0.05, 0.2], [0.05, -0.2]],
                    "axis": [[0, 0], [0, 1]],
                    "angle": 90,
                    "offset": [0, 0, 0],
                }
            ]
        )
        self.assertEqual(validate_plan(solid, "v5"), [])

    def test_feedback_yaml_can_select_v5(self) -> None:
        block = resolve_feedback_config({"feedback": {"arm": "C", "plan_prompt_version": "v5"}})
        self.assertEqual(block["plan_prompt_version"], "v5")
        self.assertTrue(block["enabled"])
        frozen = resolve_feedback_config({"feedback": {"arm": "C"}})
        self.assertEqual(frozen["plan_prompt_version"], "v3")


if __name__ == "__main__":
    unittest.main()
