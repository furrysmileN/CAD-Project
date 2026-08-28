from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.pc_conditions import (
    CONFIRM_CONDITION_IDS,
    SCREEN_CONDITION_IDS,
    parse_condition,
    validate_conditions,
)
from rq2_harness.pc_fsm import ToolLoopConfig, ToolLoopState, apply_query, classify_model_output
from rq2_harness.pc_prompting import (
    MAX_EVIDENCE_TOKENS,
    audit_payload,
    build_pc_messages,
    compact_evidence_for_prompt,
    estimate_tokens,
)
from rq2_harness.pointcloud.canonical import CanonicalTransform
from rq2_harness.pointcloud.evidence import EVIDENCE_SCHEMA, build_evidence
from rq2_harness.pointcloud.io import PointCloudError, clean_points, hash_points
from rq2_harness.pointcloud.service import PointCloudService
from rq2_harness.pointcloud.tools import PointCloudSession, execute_tool, parse_query_request


def _images(root: Path, prefix: str) -> list[dict[str, str]]:
    result = []
    for index, view in enumerate(("front", "side", "top", "isometric")):
        path = root / f"{prefix}_{view}.png"
        Image.new("RGB", (8, 8), (index * 40, 20, 10)).save(path)
        result.append({"view": view, "path": str(path), "sha256": f"{prefix}-{view}"})
    return result


def _row(root: Path) -> dict:
    npy = root / "cloud.npy"
    np.save(npy, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64))
    return {
        "sample_id": "secret_family_0001",
        "family": "secret_family",
        "difficulty": "hard",
        "gt_code": {"path": str(root / "secret.py"), "sha256": "gt-secret-hash"},
        "step": {"path": str(root / "secret.step"), "sha256": "gt-step-hash"},
        "text": {"L1": "short plate", "L3": "detailed secret"},
        "text_encodings": {"T1": {"text": "short plate", "sha256": "t1"}},
        "render_encodings": {"I1": {"images": _images(root, "I1"), "params": {}}},
        "point_cloud": {
            "path": str(npy),
            "sha256": "point-hash",
            "encoding": {"images": _images(root, "P"), "params": {"encoding_version": "rq2.pc_depth_contour.v1"}},
        },
    }


def _tiny_evidence(npy: Path) -> dict:
    return build_evidence(npy)


class PCConditionTests(unittest.TestCase):
    def test_nine_screen_ids_and_no_proj_geom_mix(self) -> None:
        self.assertEqual(len(SCREEN_CONDITION_IDS), 9)
        self.assertEqual(len(CONFIRM_CONDITION_IDS), 8)
        self.assertNotIn("P_geom_static", CONFIRM_CONDITION_IDS)
        specs = validate_conditions(SCREEN_CONDITION_IDS)
        self.assertEqual(len(specs), 9)
        self.assertTrue(parse_condition("I1P_geom").tools)
        self.assertFalse(parse_condition("P_geom_static").tools)
        self.assertTrue(parse_condition("P_geom_static").point_geom)
        self.assertFalse(parse_condition("P_proj").point_geom)
        with self.assertRaises(ValueError):
            parse_condition("TIP")
        with self.assertRaises(ValueError):
            validate_conditions(["I1", "I1"])


class PointCloudIOTests(unittest.TestCase):
    def test_nan_empty_and_canonical_roundtrip(self) -> None:
        with self.assertRaises(PointCloudError):
            clean_points(np.array([[np.nan, 0, 0]], dtype=np.float64))
        points = np.array([[10.0, 2.0, 0.0], [12.0, 2.0, 0.5], [10.0, 3.0, 0.5]], dtype=np.float64)
        cleaned, quality = clean_points(points)
        self.assertFalse(quality["degenerate"])
        transform = CanonicalTransform.from_points(cleaned)
        canonical = transform.forward(cleaned)
        restored = transform.inverse(canonical)
        np.testing.assert_allclose(restored, cleaned, atol=1e-9)
        matrix = transform.to_matrix()
        ones = np.concatenate([cleaned, np.ones((len(cleaned), 1))], axis=1)
        via_matrix = ones @ matrix
        np.testing.assert_allclose(via_matrix[:, :3], canonical, atol=1e-9)
        inv = transform.inverse_matrix()
        np.testing.assert_allclose(matrix @ inv, np.eye(4), atol=1e-9)

    def test_hash_stable(self) -> None:
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        self.assertEqual(hash_points(points), hash_points(points[::-1]))


class EvidenceAndToolTests(unittest.TestCase):
    def test_schema_and_seed_repeat(self) -> None:
        rng = np.random.default_rng(0)
        points = rng.normal(size=(256, 3))
        points[:, 2] *= 0.1
        with tempfile.TemporaryDirectory() as directory:
            npy = Path(directory) / "a.npy"
            np.save(npy, points)
            first = build_evidence(npy, ransac_seed=42)
            second = build_evidence(npy, ransac_seed=42)
            self.assertEqual(first["schema"], EVIDENCE_SCHEMA)
            self.assertEqual(first["content_hash"], second["content_hash"])
            compact = compact_evidence_for_prompt(first)
            self.assertLessEqual(estimate_tokens(json.dumps(compact)), MAX_EVIDENCE_TOKENS)
            self.assertNotIn("source_sha256", compact)

    def test_cloud_id_isolation_and_param_validation(self) -> None:
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        transform = CanonicalTransform.from_points(points)
        session = PointCloudSession(
            cloud_id="c_bound",
            points_canonical=transform.forward(points),
            transform=transform,
        )
        denied = execute_tool(
            session,
            {"tool": "get_pointcloud_summary", "params": {"cloud_id": "c_other"}},
        )
        self.assertFalse(denied["ok"])
        self.assertEqual(denied["error"]["code"], "PermissionError")
        bad = execute_tool(session, {"tool": "measure_pointcloud", "params": {"cloud_id": "c_bound"}})
        self.assertFalse(bad["ok"])
        ok = execute_tool(
            session,
            {
                "tool": "measure_pointcloud",
                "params": {"cloud_id": "c_bound", "measurement_type": "bbox_size"},
            },
        )
        self.assertTrue(ok["ok"])
        missing_step = execute_tool(
            session,
            {
                "tool": "compare_cad_to_cloud",
                "params": {"cloud_id": "c_bound", "candidate_step_id": "cand_0"},
            },
        )
        self.assertFalse(missing_step["ok"])

    def test_fsm_budget(self) -> None:
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        transform = CanonicalTransform.from_points(points)
        session = PointCloudSession(
            cloud_id="c_bound",
            points_canonical=transform.forward(points),
            transform=transform,
        )
        state = ToolLoopState()
        config = ToolLoopConfig(max_pre_queries=2, max_post_queries=1)
        base = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
        request = {"tool": "get_pointcloud_summary", "params": {"cloud_id": "c_bound"}}
        for _ in range(2):
            apply_query(
                base_messages=base,
                raw_response=json.dumps(request),
                request=request,
                session=session,
                state=state,
                config=config,
            )
        self.assertEqual(state.pre_queries, 2)
        _, trace = apply_query(
            base_messages=base,
            raw_response=json.dumps(request),
            request=request,
            session=session,
            state=state,
            config=config,
        )
        self.assertEqual(trace["error"]["code"], "BudgetError")
        compare = {
            "tool": "compare_cad_to_cloud",
            "params": {"cloud_id": "c_bound", "candidate_step_id": "cand_0"},
        }
        _, trace = apply_query(
            base_messages=base,
            raw_response=json.dumps(compare),
            request=compare,
            session=session,
            state=state,
            config=config,
        )
        self.assertEqual(trace["error"]["code"], "BudgetError")
        self.assertEqual(classify_model_output('{"tool":"detect_symmetry","params":{"cloud_id":"c_bound"}}')["kind"], "query_request")
        self.assertEqual(classify_model_output('{"schema_version":"harnesscad.plan.v2"}')["kind"], "plan")

    def test_parse_query_wrapped(self) -> None:
        text = '{"query_request": {"tool": "detect_symmetry", "params": {"cloud_id": "c"}}}'
        parsed = parse_query_request(text)
        # 外层带 query_request 时仍能被 FSM 识别
        classified = classify_model_output(text)
        self.assertIn(classified["kind"], {"query_request", "plan"})


class PromptLeakageTests(unittest.TestCase):
    def test_condition_slots_and_no_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _row(root)
            evidence = _tiny_evidence(Path(row["point_cloud"]["path"]))
            for condition_id in SCREEN_CONDITION_IDS:
                spec = parse_condition(condition_id)
                messages, audit = build_pc_messages(row, spec, evidence=evidence, image_max_edge=8)
                serialized = json.dumps(messages)
                self.assertNotIn("secret_family", serialized)
                self.assertNotIn("gt-secret-hash", serialized)
                self.assertNotIn(".npy", serialized)
                self.assertNotIn(".step", serialized)
                self.assertNotIn(str(row["sample_id"]), serialized)
                report = audit_payload(
                    messages,
                    spec,
                    sample_id=row["sample_id"],
                    family=row["family"],
                    gt_hash="gt-secret-hash",
                )
                self.assertTrue(report["ok"], report["issues"])
                self.assertEqual(set(audit["allowed_modalities"]), set(spec.modalities))
                if spec.text:
                    self.assertIn("[TEXT_INTENT]", serialized)
                    self.assertIn("short plate", serialized)
                    self.assertNotIn("detailed secret", serialized)
                else:
                    self.assertNotIn("[TEXT_INTENT]", serialized)
                    self.assertNotIn("short plate", serialized)
                if spec.point_geom:
                    self.assertIn("[POINT_OBSERVATION]", serialized)
                    self.assertIn("[POINT_HYPOTHESIS]", serialized)
                if spec.point_proj:
                    self.assertIn("projected contour", serialized)
                if spec.tools:
                    self.assertIn("[POINT_TOOLS]", serialized)
                else:
                    self.assertNotIn("[POINT_TOOLS]", serialized)

    def test_evidence_path_does_not_read_gt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npy = root / "cloud.npy"
            np.save(npy, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64))
            step = root / "gt.step"
            step.write_text("STEP", encoding="utf-8")

            real_open = open

            def guarded_open(path, *args, **kwargs):
                name = Path(str(path)).name.lower()
                if name.endswith(".step") or name.endswith(".stp") or name.endswith(".py"):
                    raise AssertionError(f"证据路径读取了 GT: {path}")
                return real_open(path, *args, **kwargs)

            service = PointCloudService()
            with mock.patch("builtins.open", guarded_open):
                evidence = service.prepare_evidence(npy, root / "evidence", "sample")
            self.assertEqual(evidence["schema"], EVIDENCE_SCHEMA)


if __name__ == "__main__":
    unittest.main()
