from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rq2_harness.cq_sandbox import audit_cadquery_source, extract_cadquery_source, run_cadquery_sandbox
from rq2_harness.hvc_ops import classify_code, coverage_report
from rq2_harness.hvc_oracles import ORACLES
from rq2_harness.prompting import parse_plan_response, validate_plan


class HvcOpsTests(unittest.TestCase):
    def test_advanced_and_coverage_gate(self):
        code = "import cadquery as cq\ncq.Workplane('XY').circle(1).sweep(cq.Wire.makeHelix(1, 4, 2))\n"
        row = classify_code(code, "coil_spring")
        self.assertEqual(row["stratum"], "advanced")
        self.assertTrue(row["v3_expressible"])
        report = coverage_report([{**row, "stratum": "advanced"}] * 10)
        self.assertTrue(report["pass_gate"])

    def test_denied_import(self):
        issues = audit_cadquery_source("import os\nresult = os.getcwd()\n")
        self.assertTrue(any(item["code"] == "denied_import" for item in issues))

    def test_extract_fence(self):
        text = "here\n```python\nimport cadquery as cq\nresult = cq.Workplane('XY').box(1,1,1)\n```\n"
        self.assertIn("cadquery", extract_cadquery_source(text))

    def test_sandbox_box(self):
        source = "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)\n"
        with tempfile.TemporaryDirectory() as tmp:
            out = run_cadquery_sandbox(source, Path(tmp) / "box.step", timeout_sec=30)
        self.assertTrue(out["ok"], out.get("issues"))

    def test_v5_messages_include_guidance(self):
        from rq2_harness.common import read_jsonl
        from rq2_harness.hvc_runner import _harness_messages

        manifest = Path(__file__).resolve().parents[1] / "outputs" / "harness_vs_cadrille" / "manifest_n40.jsonl"
        if not manifest.is_file():
            self.skipTest("HVC manifest missing")
        row = next(iter(read_jsonl(manifest)))
        messages, audit = _harness_messages(row, image_max_edge=256, plan_version="v5")
        texts = [item.get("text", "") for item in messages[1]["content"] if item.get("type") == "text"]
        self.assertTrue(any("[POSE]" in text or "[CONSTRUCTION_LAWS]" in text for text in texts))
        self.assertIn("guidance", audit)

    def test_v4_prompt_parse(self):
        plan = ORACLES[5]
        text = json_dumps(plan)
        parsed = parse_plan_response(text, plan_version="v4")
        self.assertTrue(parsed["ok"], parsed.get("issues"))
        self.assertEqual(validate_plan(plan, "v4"), [])


def json_dumps(plan: dict) -> str:
    import json

    return json.dumps(plan)


class AnalysisGateTests(unittest.TestCase):
    def test_pending_gpu_excluded_from_means(self):
        from rq2_harness.hvc_analysis import analyze_hvc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                '{"sample_id":"a","stratum":"standard","step":{"path":"x"}}\n',
                encoding="utf-8",
            )
            state_dir = root / "cut2" / "state" / "a"
            state_dir.mkdir(parents=True)
            (state_dir / "cadrille_rl.json").write_text(
                '{"arm":"cadrille_rl","sample_id":"a","stratum":"standard","status":"pending_gpu","geometry":{"joint_quality":0.0,"success":false}}',
                encoding="utf-8",
            )
            (state_dir / "qwen_raw.json").write_text(
                '{"arm":"qwen_raw","sample_id":"a","stratum":"standard","status":"completed","geometry":{"joint_quality":0.2,"success":true}}',
                encoding="utf-8",
            )
            (state_dir / "qwen_harness.json").write_text(
                '{"arm":"qwen_harness","sample_id":"a","stratum":"standard","status":"completed","geometry":{"joint_quality":0.4,"success":true}}',
                encoding="utf-8",
            )
            report = analyze_hvc(
                {
                    "paths": {"output_root": str(root / "cut2"), "manifest": str(manifest)},
                    "cut3": {"jq_delta_gate": 0.03},
                }
            )
            self.assertEqual(report["cadrille_note"], "cadrille_pending_gpu")
            self.assertIsNone(report["means"].get("cadrille_rl"))
            self.assertAlmostEqual(report["delta_harness_raw"], 0.2)
            self.assertTrue(report["gates"]["proceed_cut3"])


class OracleCompileTests(unittest.TestCase):
    def test_ten_oracles_execute(self):
        from rq2_harness.backend import run_episode

        self.assertEqual(len(ORACLES), 10)
        config = {"episode_version": "v2", "root": "HarnessCAD/HarnessCAD", "timeout_sec": 30}
        for plan in ORACLES:
            result = run_episode(plan, config)
            status = (result.get("response") or {}).get("status")
            self.assertIn(status, {"success", "success_with_warnings"}, f"{plan['sample_id']}: {result}")
