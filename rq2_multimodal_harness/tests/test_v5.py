from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.pc_conditions import (
    SCREEN_CONDITION_IDS,
    V5_ABLATION_IDS,
    V5_CONFIRM_IDS,
    V5_TOOL_IDS,
    parse_condition,
    validate_conditions,
)
from rq2_harness.pc_prompting import apply_evidence_profile, audit_payload, build_pc_messages, compact_evidence_for_prompt
from rq2_harness.pc_runner import _state_path, _task_order
from rq2_harness.pc_tool_fsm import append_query_turn, parse_query_or_submit
from rq2_harness.pointcloud.evidence import build_evidence
from rq2_harness.pc_analysis import _geometry_values
from rq2_harness.v5_reaudit import load_screen_ids
from rq2_harness.v5_shuffle import build_shuffle_mapping


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
        "text": {"L1": "short plate", "L3": "1. Extrude a 20 mm plate\n2. Drill a hole"},
        "text_encodings": {"T1": {"text": "short plate"}, "T2": {"text": "1. Extrude a 20 mm plate"}},
        "render_encodings": {"I1": {"images": _images(root, "I1"), "params": {}}},
        "point_cloud": {
            "path": str(npy),
            "sha256": "point-hash",
            "encoding": {"images": _images(root, "P"), "params": {"encoding_version": "rq2.pc_depth_contour.v1"}},
        },
    }


class V5ConditionTests(unittest.TestCase):
    def test_v4_screen_ids_unchanged(self) -> None:
        self.assertEqual(len(SCREEN_CONDITION_IDS), 9)
        self.assertTrue(parse_condition("I1P_geom").tools)

    def test_v5_ids_parse(self) -> None:
        validate_conditions(V5_ABLATION_IDS)
        validate_conditions(V5_CONFIRM_IDS)
        validate_conditions(V5_TOOL_IDS)
        self.assertEqual(parse_condition("P_bbox").resolved_profile, "bbox")
        self.assertTrue(parse_condition("P_shuffle").shuffle)
        self.assertEqual(parse_condition("T2I1").text_level, "T2")
        self.assertEqual(parse_condition("OPTIONAL_TOOL").tool_protocol, "query_or_submit")


class V5EvidenceProfileTests(unittest.TestCase):
    def test_bbox_omits_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npy = Path(tmp) / "c.npy"
            np.save(npy, np.random.default_rng(0).normal(size=(64, 3)))
            evidence = build_evidence(npy)
            bbox = apply_evidence_profile(evidence, "bbox")
            self.assertNotIn("sections", bbox)
            self.assertNotIn("symmetry_candidates", bbox)
            full = compact_evidence_for_prompt(evidence, profile="full")
            self.assertIn("sections", full)
            partial = apply_evidence_profile(evidence, "partial")
            for block in (partial.get("sections") or {}).values():
                self.assertEqual(block.get("holes"), [])


class V5PromptTests(unittest.TestCase):
    def test_t2_and_bbox_no_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _row(root)
            evidence = build_evidence(Path(row["point_cloud"]["path"]))
            messages, audit = build_pc_messages(row, "T2I1P_geom", evidence=evidence)
            self.assertEqual(audit.get("text_level"), "T2")
            payload = audit_payload(messages, parse_condition("T2I1P_geom"), sample_id=row["sample_id"])
            self.assertTrue(payload["ok"], payload["issues"])
            messages_b, _ = build_pc_messages(row, "P_bbox", evidence=evidence)
            serialized = json.dumps(messages_b)
            self.assertIn("[POINT_OBSERVATION]", serialized)
            self.assertNotIn("short plate", serialized)
            cam_audit = audit_payload(messages_b, parse_condition("P_bbox"), sample_id=row["sample_id"], family="cam")
            self.assertTrue(cam_audit["ok"], cam_audit["issues"])


class V5ShuffleTests(unittest.TestCase):
    def test_derangement(self) -> None:
        rows = [
            {"sample_id": "a", "difficulty": "easy", "complexity_bin": 0, "family": "fa", "frame": {"bbox_size": [1, 1, 1]}},
            {"sample_id": "b", "difficulty": "easy", "complexity_bin": 0, "family": "fb", "frame": {"bbox_size": [1, 1, 1.1]}},
            {"sample_id": "c", "difficulty": "hard", "complexity_bin": 2, "family": "fc", "frame": {"bbox_size": [2, 1, 1]}},
        ]
        payload = build_shuffle_mapping(rows, seed=1)
        mapping = payload["mapping"]
        self.assertEqual(set(mapping), {"a", "b", "c"})
        self.assertEqual(set(mapping.values()), {"a", "b", "c"})
        self.assertTrue(all(src != dst for src, dst in mapping.items()))


class V5ToolProtocolTests(unittest.TestCase):
    def test_parse_query_or_submit(self) -> None:
        query = parse_query_or_submit(
            '{"action":"query","query":{"tool":"query_cross_section","arguments":{"origin":[0,0,0],"normal":[0,0,1]}},"plan":null}'
        )
        self.assertEqual(query["kind"], "query")
        plan = parse_query_or_submit('{"action":"submit_plan","query":null,"plan":{"operations":[]}}')
        self.assertEqual(plan["kind"], "submit_plan")
        bare = parse_query_or_submit('{"operations":[{"op":"box"}]}')
        self.assertEqual(bare["kind"], "submit_plan")

    def test_query_history_accumulates(self) -> None:
        messages = [{"role": "user", "content": "start"}]
        once = append_query_turn(messages, request={"tool": "a"}, result={"ok": True, "n": 1})
        twice = append_query_turn(once, request={"tool": "b"}, result={"ok": True, "n": 2})
        self.assertEqual(len(twice), 5)
        self.assertIn("Query history is accumulated", twice[-1]["content"])
        self.assertIn('"n": 1', twice[2]["content"] if isinstance(twice[2].get("content"), str) else "")


class V5RunnerLayoutTests(unittest.TestCase):
    def test_repeat_state_path_and_control_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _state_path(Path(tmp), "s1", "P_geom", 2)
            self.assertEqual(path.name, "r02.json")
            self.assertEqual(path.parent.name, "P_geom")
            legacy = _state_path(Path(tmp), "s1", "P_geom", 0)
            self.assertEqual(legacy.name, "P_geom.json")
        rows = [{"sample_id": "a"}, {"sample_id": "b"}]
        specs = validate_conditions(["P_geom", "I1P_shuffle"])
        tasks = _task_order(
            rows,
            specs,
            seed=1,
            repeat_ids=(1, 2, 3),
            control_ids=frozenset({"I1P_shuffle"}),
            control_repeat=1,
        )
        self.assertEqual(len(tasks), 2 * 3 + 2 * 1)
        shuffle_repeats = {rid for row, spec, rid in tasks if spec.condition_id == "I1P_shuffle"}
        self.assertEqual(shuffle_repeats, {1})


class V5AxisDegeneracyTests(unittest.TestCase):
    def test_cylinder_keeps_only_revolution_axis(self) -> None:
        from rq2_harness.evidence_audit import _axis_alignment, _well_defined_pca_axes

        axes = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        defined = _well_defined_pca_axes(axes, [1.0, 0.32, 0.31])
        self.assertEqual(len(defined), 1)
        self.assertGreater(abs(float(defined[0][2])), 0.99)
        gt = {"axes": axes, "eigenvalue_ratios": [1.0, 0.30, 0.29]}
        # 径向平面转 45° 不应再拖垮分数
        rotated = [[0, 0, 1], [0.707, 0.707, 0], [-0.707, 0.707, 0]]
        score = _axis_alignment(rotated, [1.0, 0.32, 0.31], gt)
        self.assertIsNotNone(score)
        self.assertGreater(float(score), 0.99)

    def test_similar_pair_is_undefined(self) -> None:
        from rq2_harness.evidence_audit import _well_defined_pca_axes

        axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.assertEqual(_well_defined_pca_axes(axes, [1.0, 0.95, 0.05]), [])
        self.assertEqual(_well_defined_pca_axes(axes, [1.0, 0.99, 0.98]), [])

    def test_box_keeps_all_isolated_axes(self) -> None:
        from rq2_harness.evidence_audit import _well_defined_pca_axes

        axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        defined = _well_defined_pca_axes(axes, [1.0, 0.50, 0.20])
        self.assertEqual(len(defined), 3)


class V5HeldoutTests(unittest.TestCase):
    def test_screen_ids_count(self) -> None:
        path = Path(__file__).resolve().parents[1] / "outputs" / "encoding_screen_n20" / "selection_summary.json"
        if not path.is_file():
            self.skipTest("selection_summary missing")
        ids = load_screen_ids(path)
        self.assertEqual(len(ids), 20)

    def test_geometry_exports_common_frame_cd(self) -> None:
        values = _geometry_values(
            {
                "status": "completed",
                "geometry": {
                    "joint_quality": 0.4,
                    "common_frame_cd": 0.12,
                    "fscore_common": {"f1": 0.3},
                    "bbox": {"scale_log_abs": 0.01},
                },
            }
        )
        self.assertEqual(values["common_frame_cd"], 0.12)
        self.assertEqual(values["f1_common"], 0.3)


if __name__ == "__main__":
    unittest.main()
