from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.v6_analysis import analyze_v6, mock_rows
from rq2_harness.v6_carriers import list_carriers, strip_carriers
from rq2_harness.v6_conditions import FORBIDDEN_MARKERS, audit_v6_payload, build_v6_messages, parse_condition
from rq2_harness.v6_corruptions import build_p_wrong, unchanged_except_critical
from rq2_harness.v6_evidence_builder import attach_primary_critical, build_p_comp
from rq2_harness.v6_fact_masks import build_p_repeat, repeat_contains_critical
from rq2_harness.v6_feature_scorer import score_critical_fact
from rq2_harness.v6_latent_generator import FAMILIES, generate_one, generate_split, parameter_signature


class V6LatentTests(unittest.TestCase):
    def test_five_families_and_unique_signatures(self) -> None:
        specs = generate_split("pilot", 20)
        self.assertEqual({spec["family"] for spec in specs}, set(FAMILIES))
        sigs = [parameter_signature(spec) for spec in specs]
        self.assertEqual(len(sigs), len(set(sigs)))
        confirm = generate_split("confirm", 20)
        overlap = set(sigs) & {parameter_signature(spec) for spec in confirm}
        self.assertFalse(overlap)

    def test_gt_plan_not_required_in_prompt_fields(self) -> None:
        spec = generate_one("pilot", 0)
        self.assertIn("gt_plan", spec)
        self.assertTrue(spec["critical_fact"]["recoverable_from_pointcloud"])
        self.assertFalse(spec["critical_fact"]["visibility_in_images"])


def _synthetic_plate_with_blind_hole(n: int = 3500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-0.36, 0.36, n)
    z = rng.uniform(-0.085, 0.085, n)
    face = rng.integers(0, 6, n)
    x = np.where(face == 0, -0.5, np.where(face == 1, 0.5, x))
    y = np.where(face == 2, -0.36, np.where(face == 3, 0.36, y))
    z = np.where(face == 4, -0.085, np.where(face == 5, 0.085, z))
    center = np.array([-0.20, 0.12])
    radius = 0.06
    depth = 0.09
    xy = np.column_stack((x, y))
    in_hole = (np.linalg.norm(xy - center, axis=1) < radius) & (face == 5)
    z = np.where(in_hole, 0.085 - depth, z)
    wall_n = 400
    theta = rng.uniform(0, 2 * np.pi, wall_n)
    wall = np.column_stack(
        (
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            rng.uniform(0.085 - depth, 0.085, wall_n),
        )
    )
    return np.vstack([np.column_stack((x, y, z)), wall])


class V6EvidenceTests(unittest.TestCase):
    def test_repeat_and_wrong(self) -> None:
        spec = generate_one("pilot", 0)
        spec["critical_fact"] = {
            "fact_id": "hole_blind.depth",
            "category": "depth",
            "value": 0.09,
            "field": "depth",
        }
        with tempfile.TemporaryDirectory() as tmp:
            npy = Path(tmp) / "c.npy"
            np.save(npy, _synthetic_plate_with_blind_hole())
            p_comp = attach_primary_critical(build_p_comp(npy), spec["critical_fact"])
            p_repeat = build_p_repeat(p_comp, spec["critical_fact"])
            p_wrong = build_p_wrong(p_comp, spec["critical_fact"], sample_id=spec["sample_id"])
            self.assertFalse(repeat_contains_critical(p_repeat, spec["critical_fact"]))
            self.assertTrue(unchanged_except_critical(p_comp, p_wrong, spec["critical_fact"]))
            self.assertIn("corruption", p_wrong)
            dumped = json.dumps(p_comp)
            self.assertNotIn('"latent_spec"', dumped)
            self.assertNotIn('"gt_plan"', dumped)

    def test_primary_depth_not_bbox_and_within_tolerance(self) -> None:
        critical = {"fact_id": "hole_blind.depth", "category": "depth", "value": 0.09}
        with tempfile.TemporaryDirectory() as tmp:
            npy = Path(tmp) / "c.npy"
            np.save(npy, _synthetic_plate_with_blind_hole())
            p_comp = attach_primary_critical(build_p_comp(npy), critical)
            primary = next(item for item in p_comp["cad_facts"] if item.get("role") == "primary_critical")
            self.assertNotEqual(primary.get("source"), "point_cloud_bbox")
            self.assertAlmostEqual(float(primary["value"]), 0.09, delta=0.04)

    def test_hidden_absence_on_plain_plate(self) -> None:
        critical = {"fact_id": "back_hole.hidden_presence", "category": "hidden_presence", "value": False}
        with tempfile.TemporaryDirectory() as tmp:
            npy = Path(tmp) / "c.npy"
            np.save(npy, _synthetic_plate_with_blind_hole())
            p_comp = attach_primary_critical(build_p_comp(npy), critical)
            primary = next(item for item in p_comp["cad_facts"] if item.get("role") == "primary_critical")
            self.assertFalse(bool(primary["value"]))

    def test_strip_carriers_removes_section_and_duplicate_facts(self) -> None:
        evidence = {
            "cad_facts": [
                {"fact_id": "hole_blind.depth", "category": "depth", "value": 0.09, "role": "primary_critical"},
                {"fact_id": "section_xy/hole_00", "category": "depth", "value": 0.09, "role": "measured"},
            ],
            "sections": {"XY": {"holes": [{"radius": 0.06, "depth": 0.09}, {"radius": 0.05, "depth": 0.40}]}},
            "hypotheses": [{"note": "hole_blind.depth looks like 0.09"}],
        }
        critical = {"fact_id": "hole_blind.depth", "category": "depth"}
        stripped = strip_carriers(evidence, critical)
        self.assertFalse(list_carriers(stripped, critical))
        self.assertEqual(len(stripped["sections"]["XY"]["holes"]), 1)
        self.assertEqual(stripped["sections"]["XY"]["holes"][0]["depth"], 0.40)


class V6PayloadTests(unittest.TestCase):
    def test_neutral_ids_and_no_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            views = []
            for i, name in enumerate(("front", "side", "top", "isometric")):
                path = root / f"{name}.png"
                Image.new("RGB", (16, 16), (i * 40, 10, 20)).save(path)
                views.append({"view": name, "path": str(path), "sha256": name})
            evidence = {
                "schema": "point_evidence.v6",
                "cloud_id": "c_deadbeef",
                "frame": {"bbox_size": [1, 1, 1]},
                "quality": {},
                "cad_facts": [{"fact_id": "hole.depth", "value": 0.1, "role": "primary_critical"}],
                "hypotheses": [],
                "uncertainties": [],
            }
            row = {
                "sample_id": "v6_pilot_0001",
                "family": "plate_holes",
                "images": {"views": views},
                "evidence": {"p_comp": evidence, "p_repeat": evidence, "p_wrong": evidence},
            }
            for cid in ("C0", "C1", "C2", "C3", "C4", "C5"):
                spec = parse_condition(cid)
                messages = build_v6_messages(row, spec)
                audit = audit_v6_payload(messages, spec, sample_id=row["sample_id"], family=row["family"])
                self.assertTrue(audit["ok"], audit["issues"])
                blob = json.dumps(messages)
                self.assertNotIn("C3_COMPLEMENT", blob)
                self.assertNotIn(row["sample_id"], blob)
                self.assertNotIn("plate_holes", blob)
                for marker in ("latent_spec", "gt_code"):
                    self.assertNotIn(marker, blob.lower())
            self.assertIn("C0_BASE", FORBIDDEN_MARKERS)


class V6FeatureAndStatsTests(unittest.TestCase):
    def test_feature_depth(self) -> None:
        latent = {"critical_fact": {"fact_id": "hole_blind.depth", "category": "depth", "value": 0.10, "operation_id": "hole_blind"}}
        plan = {"operations": [{"id": "hole_blind", "op": "hole", "depth": 0.10, "combine": "cut"}]}
        scores = score_critical_fact(plan, latent)
        self.assertTrue(scores["exact"])
        self.assertTrue(scores["within_tolerance"])

    def test_oracle_pocket_box_depth(self) -> None:
        latent = {"critical_fact": {"fact_id": "pocket.depth", "category": "depth", "value": 0.16, "operation_id": "pocket"}}
        gt = {"operations": [{"id": "pocket", "op": "box", "combine": "cut", "size": [0.3, 0.2, 0.16]}]}
        cf = {"operations": [{"id": "pocket", "op": "box", "combine": "cut", "size": [0.3, 0.2, 0.31]}]}
        self.assertTrue(score_critical_fact(gt, latent)["exact"])
        self.assertFalse(score_critical_fact(cf, latent)["exact"])

    def test_oracle_flange_spacing_json_rounding(self) -> None:
        latent = {"critical_fact": {"category": "offset_or_spacing", "value": 0.2847}}
        gt = {
            "operations": [
                {"id": "bolt_0", "op": "hole", "center": [-0.1424, -0.1424, 0.05], "combine": "cut"},
                {"id": "bolt_1", "op": "hole", "center": [0.1424, -0.1424, 0.05], "combine": "cut"},
            ]
        }
        cf = copy.deepcopy(gt)
        for op in cf["operations"]:
            op["center"][0] = round(float(op["center"][0]) * 1.5, 4)
            op["center"][1] = round(float(op["center"][1]) * 1.5, 4)
        self.assertTrue(score_critical_fact(gt, latent)["exact"])
        self.assertFalse(score_critical_fact(cf, latent)["exact"])

    def test_blind_depth_ignores_through_hole(self) -> None:
        latent = {"critical_fact": {"fact_id": "hole_blind.depth", "category": "depth", "value": 0.16, "operation_id": "hole_blind"}}
        mixed = {
            "operations": [
                {"id": "base", "op": "box", "combine": "new", "size": [1.0, 0.72, 0.42]},
                {"id": "hole_1", "op": "hole", "combine": "cut", "depth": 0.42},
                {"id": "hole_2", "op": "hole", "combine": "cut", "depth": 0.16},
            ]
        }
        only_through = {
            "operations": [
                {"id": "base", "op": "box", "combine": "new", "size": [1.0, 0.72, 0.42]},
                {"id": "hole_1", "op": "hole", "combine": "cut", "depth": 0.42},
                {"id": "hole_2", "op": "hole", "combine": "cut", "depth": 0.42},
            ]
        }
        self.assertTrue(score_critical_fact(mixed, latent)["exact"])
        self.assertIsNone(score_critical_fact(only_through, latent)["pred_value"])

    def test_pocket_named_hole_not_through(self) -> None:
        latent = {"critical_fact": {"fact_id": "pocket.depth", "category": "depth", "value": 0.1304, "operation_id": "pocket"}}
        plan = {
            "operations": [
                {"id": "base_block", "op": "box", "combine": "new", "size": [0.92, 0.70, 0.40]},
                {"id": "through_hole", "op": "hole", "combine": "cut", "depth": 0.4348},
                {"id": "top_pocket", "op": "hole", "combine": "cut", "depth": 0.1304},
            ]
        }
        scores = score_critical_fact(plan, latent)
        self.assertEqual(scores["pred_value"], 0.1304)
        self.assertTrue(scores["within_tolerance"])

    def test_oracle_revolve_axis_and_radius(self) -> None:
        axis_latent = {"critical_fact": {"category": "axis_or_symmetry", "value": [0.0, 0.0, 1.0]}}
        radius_latent = {"critical_fact": {"category": "radius_or_width", "value": 0.24}}
        gt = {
            "operations": [
                {
                    "id": "shaft",
                    "op": "revolve_profile",
                    "workplane": "XZ",
                    "axis": [[0.0, -1.0], [0.0, 1.0]],
                    "profile": [[0.12, -0.5], [0.24, 0.5], [0.0, 0.5]],
                }
            ]
        }
        cf = copy.deepcopy(gt)
        cf["operations"][0]["workplane"] = "YZ"
        cf["operations"][0]["profile"][1][0] = 0.32
        self.assertTrue(score_critical_fact(gt, axis_latent)["exact"])
        self.assertFalse(score_critical_fact(cf, axis_latent)["exact"])
        self.assertTrue(score_critical_fact(gt, radius_latent)["exact"])
        self.assertFalse(score_critical_fact(cf, radius_latent)["exact"])

    def test_mock_signs(self) -> None:
        jq = analyze_v6(mock_rows(20), endpoint="first_attempt", metric="joint_quality")
        cd = analyze_v6(mock_rows(20), endpoint="first_attempt", metric="common_frame_cd", invert=True)
        for row in jq["contrasts_holm"]:
            self.assertGreater(row["mean_delta"], 0)
        self.assertGreater(cd["contrasts_holm"][0]["mean_delta"], 0)
        self.assertGreater(jq["k6"]["mean_delta"], 0)


if __name__ == "__main__":
    unittest.main()
