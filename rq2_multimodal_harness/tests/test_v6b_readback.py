from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.v6_manifest import attach_evidence_payloads, read_manifest
from rq2_harness.v6b_readback import (
    KIND_FIELD,
    audit_readback_payload,
    build_readback_messages,
    expected_from_evidence,
    parse_readback_response,
    predicted_value,
    score_readback,
)

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "outputs" / "v6_information_complementarity" / "pilot_v2_minimal_pairs"


def _eligible_row(pair_id: str) -> dict:
    for row in read_manifest(PAIRS / "manifest_probe.jsonl"):
        if row.get("pair_id") == pair_id:
            return attach_evidence_payloads(row)
    raise AssertionError(pair_id)


class V6bReadbackTests(unittest.TestCase):
    def test_expected_full_repeat_counterfactual(self) -> None:
        row = _eligible_row("v6b_probe_0000")
        kind = row["kind"]
        critical = row["critical_fact"]
        full = expected_from_evidence(row["evidence"]["p_comp"], kind=kind, critical=critical)
        repeat = expected_from_evidence(row["evidence"]["p_repeat"], kind=kind, critical=critical)
        cf = expected_from_evidence(row["evidence"]["p_counterfactual"], kind=kind, critical=critical)
        self.assertTrue(full["present"])
        self.assertFalse(repeat["present"])
        self.assertTrue(cf["present"])
        self.assertEqual(full["field"], "hole_depth")
        self.assertAlmostEqual(float(full["value"]), 0.16, places=3)
        self.assertAlmostEqual(float(cf["value"]), 0.30, places=3)

    def test_payload_has_evidence_not_plan_or_images(self) -> None:
        row = _eligible_row("v6b_probe_0001")
        messages = build_readback_messages(row["evidence"]["p_comp"])
        audit = audit_readback_payload(
            messages,
            sample_id=row["sample_id"],
            family=row["family"],
            pair_id=row["pair_id"],
        )
        self.assertTrue(audit["ok"], audit["issues"])
        blob = json.dumps(messages)
        self.assertIn("[POINT_OBSERVATION]", blob)
        self.assertIn("[READBACK_CONSTRAINTS]", blob)
        self.assertNotIn("[PLAN_CONSTRAINTS]", blob)
        self.assertNotIn("data:image/png;base64,", blob)
        self.assertNotIn(row["pair_id"], blob)

    def test_score_full_and_follow(self) -> None:
        parsed = {
            "hole_depth": 0.16,
            "through_or_blind": None,
            "pocket_depth": None,
            "hidden_feature_present": None,
            "evidence_source": "point_cloud_top_depression",
            "confidence": 1.0,
        }
        expected = {"present": True, "value": 0.16, "category": "depth", "field": "hole_depth"}
        full = score_readback(parsed, expected, kind="blind_depth", foil_value=0.30)
        self.assertTrue(full["match"])
        cf_parsed = dict(parsed)
        cf_parsed["hole_depth"] = 0.30
        follow = score_readback(cf_parsed, {"present": True, "value": 0.30, "category": "depth"}, kind="blind_depth", foil_value=0.16)
        self.assertTrue(follow["follow_counterfactual"])
        stuck = score_readback(parsed, {"present": True, "value": 0.30, "category": "depth"}, kind="blind_depth", foil_value=0.16)
        self.assertFalse(stuck["follow_counterfactual"])
        missing = score_readback(
            {**parsed, "hole_depth": None},
            {"present": False, "value": None, "category": "depth"},
            kind="blind_depth",
            foil_value=0.16,
        )
        self.assertTrue(missing["match"])
        leaked = score_readback(
            parsed,
            {"present": False, "value": None, "category": "depth"},
            kind="blind_depth",
            foil_value=0.16,
        )
        self.assertFalse(leaked["match"])

    def test_parse_and_kind_fields(self) -> None:
        parsed = parse_readback_response('```json\n{"through_or_blind": "blind", "hole_depth": null}\n```')
        self.assertTrue(parsed["ok"])
        self.assertEqual(predicted_value(parsed["parsed"], "through_vs_blind"), "blind")
        self.assertEqual(set(KIND_FIELD), {"blind_depth", "through_vs_blind", "pocket_depth", "hidden_presence"})


if __name__ == "__main__":
    unittest.main()
