from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.pc_conditions import V8_I1_ABLATION_IDS, parse_condition, validate_conditions
from rq2_harness.pc_prompting import apply_evidence_profile, audit_payload, build_pc_messages
from rq2_harness.pointcloud.evidence import build_evidence
from rq2_harness.v8_autopsy import classify_regimes, classify_residual, hamilton_quotas, sample_n20


def _images(root: Path, prefix: str) -> list[dict[str, str]]:
    result = []
    for index, view in enumerate(("view_0", "view_2", "view_4", "view_6")):
        path = root / f"{prefix}_{view}.png"
        Image.new("RGB", (8, 8), (index * 40, 20, 10)).save(path)
        result.append({"view": view, "path": str(path), "sha256": f"{prefix}-{view}"})
    return result


class V8ConditionTests(unittest.TestCase):
    def test_i1_ablation_ids(self) -> None:
        specs = validate_conditions(V8_I1_ABLATION_IDS)
        self.assertEqual([spec.condition_id for spec in specs], list(V8_I1_ABLATION_IDS))
        bbox = parse_condition("I1P_bbox")
        self.assertTrue(bbox.images)
        self.assertTrue(bbox.point_geom)
        self.assertFalse(bbox.tools)
        self.assertEqual(bbox.resolved_profile, "bbox")
        self.assertEqual(parse_condition("I1P_axes").resolved_profile, "axes")
        self.assertEqual(parse_condition("I1P_sym").resolved_profile, "sym")


class V8AutopsyLabelTests(unittest.TestCase):
    def test_regimes(self) -> None:
        both = classify_regimes(0.55, 0.20, 0.40)
        self.assertTrue(both["pc_helps_image"])
        self.assertTrue(both["image_helps_pc"])
        self.assertFalse(both["both_still_bad"])
        bad = classify_regimes(0.20, 0.18, 0.19)
        self.assertTrue(bad["both_still_bad"])
        self.assertEqual(bad["regime"], "weak_or_mixed")
        good = classify_regimes(0.80, 0.50, 0.60)
        self.assertTrue(good["already_good"])

    def test_residual_priority(self) -> None:
        scale = classify_residual(
            bbox_scale=0.08,
            shape_cd=0.02,
            common_cd=0.08,
            voxel=0.6,
            i1p_jq=0.4,
            plan_holes=0,
            gt_holes=0,
            evidence_holes=0,
            gt_thru=0,
            plan_thru_like=0,
            jaccard_i1=1.0,
            jaccard_p=1.0,
        )
        self.assertEqual(scale["residual_primary"], "scale")

    def test_sample_deterministic(self) -> None:
        rows = []
        for difficulty in ("easy", "medium", "hard"):
            for bin_id in (1, 2):
                for index in range(8):
                    rows.append(
                        {
                            "sample_id": f"{difficulty}_{bin_id}_{index:02d}",
                            "difficulty": difficulty,
                            "complexity_bin": bin_id,
                            "family": "fam",
                        }
                    )
        once = sample_n20(rows, seed=20260828, n=20)
        twice = sample_n20(rows, seed=20260828, n=20)
        self.assertEqual(once["sample_ids"], twice["sample_ids"])
        self.assertEqual(len(once["sample_ids"]), 20)
        quotas = hamilton_quotas({("a", 1): 50, ("b", 1): 50}, 20)
        self.assertEqual(sum(quotas.values()), 20)


class V8PromptTests(unittest.TestCase):
    def test_i1p_bbox_no_sections_no_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npy = root / "cloud.npy"
            np.save(npy, np.random.default_rng(0).normal(size=(64, 3)))
            row = {
                "sample_id": "secret_family_0001",
                "family": "secret_family",
                "images": _images(root, "I1"),
                "point_cloud": {"path": str(npy), "sha256": "point-hash"},
            }
            evidence = build_evidence(npy)
            compact = apply_evidence_profile(evidence, "bbox")
            self.assertNotIn("sections", compact)
            self.assertNotIn("symmetry_candidates", compact)
            messages, audit = build_pc_messages(row, "I1P_bbox", evidence=evidence)
            self.assertEqual(audit.get("evidence_profile"), "bbox")
            payload = audit_payload(
                messages, parse_condition("I1P_bbox"), sample_id=row["sample_id"], family="secret_family"
            )
            self.assertTrue(payload["ok"], payload["issues"])
            serialized = json.dumps(messages)
            self.assertNotIn("[POINT_TOOLS]", serialized)
            self.assertNotIn("secret_family_0001", serialized)
            self.assertEqual(payload["image_count"], 4)


if __name__ == "__main__":
    unittest.main()
