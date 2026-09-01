from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.v6b_diag_conditions import (
    audit_diag_payload,
    build_diag_messages,
    fact_sentence,
    parse_diag_condition,
)
from rq2_harness.v6b_pair_generator import generate_pair


class V6bDiagConditionTests(unittest.TestCase):
    def test_fact_sentence_no_leakage(self) -> None:
        for index, kind in enumerate(("blind_depth", "through_vs_blind", "hidden_presence", "pocket_depth")):
            pair = generate_pair(index)
            text = fact_sentence(kind, pair["spec_b"])
            self.assertNotIn("v6b_probe_", text)
            self.assertNotIn("counterfactual", text.lower())
            self.assertNotIn(pair["spec_b"]["family"], text)
            self.assertNotIn(pair["pair_id"], text)
            if kind == "pocket_depth":
                self.assertIn("0.24", text)
            if kind == "through_vs_blind":
                self.assertIn("blind", text)
            if kind == "hidden_presence":
                self.assertIn("back face", text)

    def test_c2b_payload_has_point_no_image(self) -> None:
        spec = parse_diag_condition("C2B")
        row = {
            "evidence": {"p_counterfactual": {"schema": "point_evidence.v6", "cad_facts": [], "hypotheses": [], "uncertainties": []}},
        }
        messages = build_diag_messages(row, spec)
        audit = audit_diag_payload(messages, spec, sample_id="v6b_probe_0000a", family="plate_holes")
        self.assertTrue(audit["ok"], audit["issues"])
        serialized = str(messages)
        self.assertIn("[POINT_OBSERVATION]", serialized)
        self.assertNotIn("[IMAGE_OBSERVATION]", serialized)
        self.assertEqual(audit["image_count"], 0)

    def test_tb_payload_has_text_fact_no_point(self) -> None:
        spec = parse_diag_condition("TB")
        row = {"text_fact": "The hole is blind, with a normalized depth of 0.12."}
        messages = build_diag_messages(row, spec)
        audit = audit_diag_payload(messages, spec, sample_id="v6b_probe_0001a", family="plate_holes")
        self.assertTrue(audit["ok"], audit["issues"])
        blob = str(messages)
        self.assertIn("[TEXT_FACT]", blob)
        self.assertNotIn("[POINT_OBSERVATION]", blob)


if __name__ == "__main__":
    unittest.main()
