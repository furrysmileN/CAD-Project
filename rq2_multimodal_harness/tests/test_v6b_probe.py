from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.v6_runner import _resolve_latent_path
from rq2_harness.v6b_probe_analysis import analyze_v6b_probe

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "outputs" / "v6_information_complementarity" / "pilot_v2_minimal_pairs"


class V6bProbeHarnessTests(unittest.TestCase):
    def test_resolve_latent_from_dir(self) -> None:
        row = {"sample_id": "v6b_probe_0000a"}
        config = {"paths": {"latent_dir": str(PAIRS / "latent_specs")}}
        path = _resolve_latent_path(row, config)
        self.assertTrue(path.is_file())
        spec = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(spec["critical_fact"]["value"], 0.16)

    def test_analysis_detects_follow_and_identical(self) -> None:
        def _row(sample: str, cond: str, pred: float, exact: bool, sha: str, gt_a=0.16, gt_b=0.30) -> dict:
            return {
                "sample_id": sample,
                "kind": "blind_depth",
                "condition": cond,
                "repeat_id": 1,
                "status": "completed",
                "category": "depth",
                "gt_a": gt_a,
                "gt_b": gt_b,
                "features": {"exact": exact, "within_tolerance": exact, "pred_value": pred},
                "geometry": {"joint_quality": 0.5},
                "plan_sha256": sha,
            }

        follow_rows = []
        for cond, pred, exact, sha in (
            ("C2", 0.16, True, "a"),
            ("C3", 0.16, True, "a"),
            ("C4", 0.10, False, "b"),
            ("C5", 0.30, False, "c"),
        ):
            follow_rows.append(_row("s0", cond, pred, exact, sha))
        summary = analyze_v6b_probe(follow_rows)
        self.assertTrue(summary["gates"]["pass_probe_go"])
        self.assertEqual(summary["value_follow"]["C5_follow_strict"], 1)

        stuck = [
            _row("s1", cond, 0.16, True, "same")
            for cond in ("C2", "C3", "C4", "C5")
        ]
        stuck_summary = analyze_v6b_probe(stuck)
        self.assertFalse(stuck_summary["gates"]["pass_probe_go"])
        self.assertEqual(stuck_summary["plan_identity"]["c3_c4_c5"], 1)

        default_b = [
            _row("s2", cond, False, False, "z", gt_a=True, gt_b=False)
            for cond in ("C2", "C3", "C4", "C5")
        ]
        for row in default_b:
            row["kind"] = "hidden_presence"
            row["category"] = "hidden_presence"
            row["features"]["pred_value"] = False
        default_summary = analyze_v6b_probe(default_b)
        self.assertEqual(default_summary["value_follow"]["C5_follow_b"], 1)
        self.assertEqual(default_summary["value_follow"]["C5_follow_strict"], 0)
        self.assertFalse(default_summary["gates"]["pass_probe_go"])


if __name__ == "__main__":
    unittest.main()
