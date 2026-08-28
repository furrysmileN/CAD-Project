from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.analysis import _wilcoxon, complement_gain
from rq2_harness.common import state_path
from rq2_harness.conditions import modalities_for
from rq2_harness.geometry import bbox_metrics, canonicalize_points, chamfer_distance, f_score, score_step_pair
from rq2_harness.pointcloud import CAMERAS, encode_point_cloud, normalize_points, render_depth_contour
from rq2_harness.prepare import deterministic_stratified_sample
from rq2_harness.prompting import build_messages, parse_plan_response
from rq2_harness.runner import _should_skip_state


class SamplingTests(unittest.TestCase):
    def test_deterministic_stratified_sample(self) -> None:
        rows = [
            {"stem": f"s{i}", "family": f"f{i % 3}", "difficulty": f"d{i % 2}", "complexity_bin": i % 3}
            for i in range(30)
        ]
        first = deterministic_stratified_sample(rows, 12, 42)
        second = deterministic_stratified_sample(reversed(rows), 12, 42)
        self.assertEqual([row["stem"] for row in first], [row["stem"] for row in second])
        self.assertEqual(len({row["stem"] for row in first}), 12)


class ConditionAndLeakageTests(unittest.TestCase):
    def _row(self, root: Path) -> dict:
        image_path = root / "render.png"
        pc_path = root / "pc.png"
        Image.new("RGB", (8, 8), "white").save(image_path)
        Image.new("RGB", (8, 8), "black").save(pc_path)
        return {
            "sample_id": "secret_family_001",
            "family": "secret_family",
            "difficulty": "hard",
            "text": {"L1": "short", "L3": "make a plate"},
            "gt_code": {"path": "secret.py", "sha256": "gt-secret-hash"},
            "images": [{"path": str(image_path), "sha256": "image-hash", "view": "front"}],
            "point_cloud": {
                "sha256": "point-hash",
                "encoding": {
                    "params": {"resolution": 8},
                    "images": [{"path": str(pc_path), "sha256": "pc-view-hash", "view": "front"}],
                },
            },
        }

    def test_condition_isolation_and_no_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = self._row(Path(directory))
            for condition in ("T", "I", "P", "TI", "TP", "IP", "TIP"):
                messages, audit = build_messages(row, condition, image_max_edge=8)
                serialized = json.dumps(messages)
                self.assertEqual(set(audit["allowed_modalities"]), set(modalities_for(condition)))
                self.assertNotIn("condition", serialized.lower())
                self.assertNotIn("secret_family", serialized)
                self.assertNotIn("hard", serialized)
                self.assertNotIn("gt-secret-hash", serialized)
                image_count = serialized.count("data:image/png;base64,")
                self.assertEqual(image_count, int("I" in condition) + int("P" in condition))
                self.assertEqual("L3 DESCRIPTION:" in serialized, "T" in condition)

    def test_v2_prompt_uses_extended_schema_without_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = self._row(Path(directory))
            messages, audit = build_messages(row, "TIP", image_max_edge=8, plan_version="v2")
            serialized = json.dumps(messages)
            self.assertEqual(audit["plan_version"], "v2")
            self.assertIn("harnesscad.plan.v2", serialized)
            self.assertIn("polygon_extrude", serialized)
            self.assertNotIn("secret_family", serialized)


class PointCloudTests(unittest.TestCase):
    def test_deterministic_encoding(self) -> None:
        points = np.array(
            [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, 1], [0, 0, 0]],
            dtype=np.float32,
        )
        normalized, _ = normalize_points(points)
        first = render_depth_contour(normalized, CAMERAS["isometric"], resolution=64, padding=0.06)
        second = render_depth_contour(normalized, CAMERAS["isometric"], resolution=64, padding=0.06)
        self.assertEqual(first.tobytes(), second.tobytes())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "points.npy"
            np.save(source, points)
            a = encode_point_cloud(source, root / "cache", views=["front"], resolution=64, padding=0.06, encoding_version="test")
            b = encode_point_cloud(source, root / "cache", views=["front"], resolution=64, padding=0.06, encoding_version="test")
            self.assertEqual(a["cache_key"], b["cache_key"])
            self.assertEqual(a["images"][0]["sha256"], b["images"][0]["sha256"])


class ParsingTests(unittest.TestCase):
    def test_markdown_and_single_syntax_repair(self) -> None:
        raw = """```json
        {"schema_version":"harnesscad.plan.v1","sample_id":"s_1",
        "coordinate_system":{"units":"normalized","origin":[0,0,0],"longest_bbox_edge":1.0},
        "operations":[{"id":"base","primitive":"box","combine":"new","center":[0,0,0],"size":[1,0.5,0.2],}],}
        ```"""
        parsed = parse_plan_response(raw)
        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["repair"]["kind"], "json_syntax")
        self.assertEqual(len(parsed["plan"]["operations"]), 1)

    def test_format_repair_removes_but_does_not_invent_geometry(self) -> None:
        raw = json.dumps(
            {
                "schema_version": "harnesscad.plan.v1",
                "sample_id": "s_1",
                "coordinate_system": {"units": "normalized", "origin": [0, 0, 0], "longest_bbox_edge": 1.0},
                "operations": [{"id": 7, "primitive": "sphere", "combine": "new", "center": [0, 0, 0], "radius": 0.5, "comment": "x"}],
                "explanation": "remove me",
            }
        )
        parsed = parse_plan_response(raw)
        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["repair"]["kind"], "field_format")
        self.assertNotIn("explanation", parsed["plan"])
        self.assertNotIn("comment", parsed["plan"]["operations"][0])
        self.assertEqual(parsed["plan"]["operations"][0]["radius"], 0.5)

    def test_v2_plan_parsing_and_field_repair(self) -> None:
        raw = json.dumps(
            {
                "schema_version": "harnesscad.plan.v2",
                "sample_id": "s_2",
                "coordinate_system": {"units": "normalized", "origin": [0, 0, 0], "longest_bbox_edge": 1.0},
                "operations": [
                    {
                        "id": "base",
                        "op": "polygon_extrude",
                        "combine": "new",
                        "workplane": "XY",
                        "points": [[-0.5, -0.2], [0.5, -0.2], [0.5, 0.2], [-0.5, 0.2], [-0.5, -0.2]],
                        "depth": 0.2,
                        "centered": True,
                        "offset": [0, 0, 0],
                        "comment": "remove only",
                    }
                ],
            }
        )
        parsed = parse_plan_response(raw, plan_version="v2")
        self.assertTrue(parsed["ok"], parsed["issues"])
        self.assertEqual(parsed["repair"]["kind"], "field_format")
        self.assertNotIn("comment", parsed["plan"]["operations"][0])


class UtilityAndMetricTests(unittest.TestCase):
    def test_state_path(self) -> None:
        self.assertEqual(state_path(Path("state"), "a/b", "TI"), Path("state") / "a_b" / "TI.json")

    def test_only_model_outcomes_are_resumed_as_complete(self) -> None:
        for status in ("completed", "parse_failed", "episode_failed"):
            self.assertTrue(_should_skip_state(status, dry_run=False))
        for status in ("dry_run", "running", "fatal_api_error", "task_failed", None):
            self.assertFalse(_should_skip_state(status, dry_run=False))
        self.assertTrue(_should_skip_state("completed", dry_run=True))

    def test_complement_gain(self) -> None:
        scores = {"T": 0.2, "I": 0.5, "P": 0.4, "TI": 0.7, "TIP": 0.8}
        self.assertAlmostEqual(complement_gain(scores, "TI"), 0.2)
        self.assertAlmostEqual(complement_gain(scores, "TIP"), 0.3)

    def test_wilcoxon_requires_paired_rows(self) -> None:
        self.assertEqual(_wilcoxon([], []), (None, None, "no_pairs"))
        self.assertEqual(_wilcoxon([1.0], []), (None, None, "length_mismatch"))

    def test_known_point_geometry(self) -> None:
        cube = np.array(list(np.ndindex(2, 2, 2)), dtype=float)
        transformed = cube * 4 + np.array([10, -3, 2])
        cube_n, _ = canonicalize_points(cube)
        transformed_n, _ = canonicalize_points(transformed)
        self.assertAlmostEqual(chamfer_distance(cube_n, transformed_n), 0.0)
        metrics = bbox_metrics(cube_n, transformed_n)
        self.assertAlmostEqual(metrics["aspect_l1"], 0.0)
        self.assertAlmostEqual(metrics["scale_ratio"], 1.0)
        self.assertAlmostEqual(metrics["center_distance"], 0.0)

    def test_f_score_identity_and_separation(self) -> None:
        cube = np.array(list(np.ndindex(2, 2, 2)), dtype=float)
        cube_n, _ = canonicalize_points(cube)
        identity = f_score(cube_n, cube_n, 0.01)
        self.assertAlmostEqual(identity["precision"], 1.0)
        self.assertAlmostEqual(identity["recall"], 1.0)
        self.assertAlmostEqual(identity["f1"], 1.0)
        separated = f_score(cube_n, cube_n + np.array([100.0, 0.0, 0.0]), 0.01)
        self.assertAlmostEqual(separated["precision"], 0.0)
        self.assertAlmostEqual(separated["recall"], 0.0)
        self.assertAlmostEqual(separated["f1"], 0.0)

    def test_identical_step_geometry_when_cadquery_available(self) -> None:
        try:
            import cadquery as cq
        except ImportError:
            self.skipTest("cadquery unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "box.step"
            cq.exporters.export(cq.Workplane("XY").box(1.0, 0.5, 0.25), str(path))
            metrics = score_step_pair(path, path, n_points=256, seed=42, voxel_resolution=16, tau=0.25)
            self.assertTrue(metrics["valid"])
            self.assertAlmostEqual(metrics["shape_only_cd"], 0.0, places=12)
            self.assertAlmostEqual(metrics["common_frame_cd"], 0.0, places=12)
            self.assertAlmostEqual(metrics["joint_quality"], 1.0, places=12)
            self.assertEqual(metrics["metrics_version"], "rq2.geometry.v2")
            self.assertAlmostEqual(metrics["fscore_shape"]["f1"], 1.0, places=12)
            self.assertAlmostEqual(metrics["fscore_common"]["f1"], 1.0, places=12)
            self.assertEqual(metrics["fscore_shape"]["tau"], 0.01)
            self.assertEqual(metrics["shape_voxel_iou"]["status"], "ok")
            self.assertAlmostEqual(metrics["shape_voxel_iou"]["value"], 1.0, places=12)

    def test_invalid_metrics_schema_includes_new_fields(self) -> None:
        metrics = score_step_pair(Path("missing_pred.step"), Path("missing_gt.step"))
        self.assertFalse(metrics["valid"])
        self.assertIsNone(metrics["fscore_shape"])
        self.assertIsNone(metrics["fscore_common"])
        self.assertEqual(metrics["shape_voxel_iou"]["status"], "not_computed")
        self.assertEqual(metrics["metrics_version"], "rq2.geometry.v2")


if __name__ == "__main__":
    unittest.main()
