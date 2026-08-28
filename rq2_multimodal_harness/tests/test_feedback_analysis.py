# -*- coding: utf-8 -*-
"""RQ2b 反馈实验分析函数的单元测试：构造合成 state 校验修复率与 token 增量口径。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.encoding_analysis import (
    FEEDBACK_CONDITIONS,
    analyze_feedback_experiment,
    feedback_arm_summary,
    feedback_task_row,
    load_feedback_states,
)

MANIFEST = {
    "s1": {"sample_id": "s1", "family": "f1", "difficulty": "easy", "complexity_bin": 0},
    "s2": {"sample_id": "s2", "family": "f2", "difficulty": "medium", "complexity_bin": 1},
}


def _round(round_index: int, usage: dict | None, failure_kind: str | None = None) -> dict:
    record = {
        "round": round_index,
        "temperature": 0.3 if round_index else 0,
        "api": {"usage": usage},
    }
    if failure_kind:
        record["failure"] = {"kind": failure_kind}
    return record


class FeedbackTaskRowTests(unittest.TestCase):
    def test_fixed_after_schema_failure(self):
        state = {
            "sample_id": "s1",
            "condition_id": "T2",
            "status": "completed",
            "geometry": {"joint_quality": 0.42},
            "feedback": {
                "arm": "C",
                "kept_round": 1,
                "rounds": [
                    _round(0, {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}, "schema"),
                    _round(1, {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220}),
                ],
            },
        }
        row = feedback_task_row(state, arm="C")
        self.assertTrue(row["completed"])
        self.assertTrue(row["fixed"])
        self.assertEqual(row["fixed_at_round"], 1)
        self.assertEqual(row["round0_failure_kind"], "format")
        self.assertIsNone(row["round1_failure_kind"])
        self.assertEqual(row["n_rounds"], 2)
        self.assertEqual(row["input_tokens"], 300.0)
        self.assertEqual(row["feedback_input_tokens"], 200.0)

    def test_new_error_kind_detected(self):
        state = {
            "sample_id": "s2",
            "condition_id": "I1",
            "status": "episode_failed",
            "geometry": {},
            "feedback": {
                "arm": "C",
                "kept_round": 1,
                "rounds": [
                    _round(0, {"prompt_tokens": 100, "completion_tokens": 10}, "schema"),
                    _round(1, {"prompt_tokens": 100, "completion_tokens": 10}, "execution"),
                ],
            },
        }
        row = feedback_task_row(state, arm="C")
        self.assertFalse(row["fixed"])
        self.assertTrue(row["new_error_kind"])
        self.assertEqual(row["round1_failure_kind"], "execution")

    def test_legacy_state_without_feedback_block(self):
        state = {
            "sample_id": "s1",
            "condition_id": "P1",
            "status": "parse_failed",
            "geometry": {},
            "api": {"usage": {"prompt_tokens": 90, "completion_tokens": 9}},
        }
        row = feedback_task_row(state, arm="A0")
        self.assertEqual(row["n_rounds"], 1)
        self.assertEqual(row["round0_failure_kind"], "format")
        self.assertEqual(row["input_tokens"], 90.0)

    def test_backend_plan_validation_classified_as_format(self):
        state = {
            "sample_id": "s1",
            "condition_id": "T2",
            "status": "episode_failed",
            "geometry": {},
            "feedback": {
                "arm": "B2",
                "kept_round": 1,
                "rounds": [
                    _round(0, {"prompt_tokens": 100, "completion_tokens": 10}, "execution"),
                    _round(1, {"prompt_tokens": 100, "completion_tokens": 10}, "execution"),
                ],
            },
        }
        for record in state["feedback"]["rounds"]:
            record["failure"] = {
                "kind": "execution",
                "failure": {"code": "plan_validation_failed", "message": "rejected"},
            }
        row = feedback_task_row(state, arm="B2")
        self.assertEqual(row["round0_failure_kind"], "format")
        self.assertEqual(row["round1_failure_kind"], "format")
        self.assertFalse(row["new_error_kind"])

    def test_legacy_a0_plan_validation_failed_is_format(self):
        state = {
            "sample_id": "s1",
            "condition_id": "T2",
            "status": "episode_failed",
            "geometry": {},
            "api": {"usage": {"prompt_tokens": 90, "completion_tokens": 9}},
            "episode": {
                "response": {
                    "failure": {"code": "plan_validation_failed", "message": "rejected"},
                }
            },
        }
        row = feedback_task_row(state, arm="A0")
        self.assertEqual(row["round0_failure_kind"], "format")

    def test_legacy_a0_operation_exception_is_execution(self):
        state = {
            "sample_id": "s1",
            "condition_id": "T2",
            "status": "episode_failed",
            "geometry": {},
            "api": {"usage": {"prompt_tokens": 90, "completion_tokens": 9}},
            "episode": {
                "response": {
                    "failure": {"code": "operation_exception", "message": "boom"},
                }
            },
        }
        row = feedback_task_row(state, arm="A0")
        self.assertEqual(row["round0_failure_kind"], "execution")


class FeedbackArmSummaryTests(unittest.TestCase):
    def test_fix_rates_by_failure_type(self):
        rows = [
            feedback_task_row(
                {
                    "sample_id": "s1",
                    "condition_id": "T2",
                    "status": "completed",
                    "geometry": {"joint_quality": 0.5},
                    "feedback": {
                        "arm": "B1",
                        "kept_round": 1,
                        "rounds": [
                            _round(0, {}, "schema"),
                            _round(1, {}),
                        ],
                    },
                },
                arm="B1",
            ),
            feedback_task_row(
                {
                    "sample_id": "s2",
                    "condition_id": "T2",
                    "status": "episode_failed",
                    "geometry": {},
                    "feedback": {
                        "arm": "B1",
                        "kept_round": 1,
                        "rounds": [
                            _round(0, {}, "execution"),
                            _round(1, {}, "execution"),
                        ],
                    },
                },
                arm="B1",
            ),
        ]
        summary = feedback_arm_summary(rows)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["format_fix_rate"], 1.0)
        self.assertEqual(summary["execution_fix_rate"], 0.0)
        self.assertEqual(summary["fixed_at_round1"], 1)


class AnalyzeFeedbackExperimentTests(unittest.TestCase):
    def test_end_to_end_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "arm" / "state" / "s1"
            state_dir.mkdir(parents=True)
            state = {
                "sample_id": "s1",
                "condition_id": "T2",
                "status": "completed",
                "geometry": {"joint_quality": 0.4},
                "feedback": {
                    "arm": "A1",
                    "kept_round": 0,
                    "rounds": [_round(0, {"prompt_tokens": 10, "completion_tokens": 2})],
                },
            }
            (state_dir / "T2.json").write_text(json.dumps(state), encoding="utf-8")
            manifest_path = root / "sample_manifest.jsonl"
            with manifest_path.open("w", encoding="utf-8") as handle:
                for sample in MANIFEST.values():
                    handle.write(json.dumps(sample) + "\n")
            out = root / "analysis"
            result = analyze_feedback_experiment(
                {"A1": root / "arm"},
                out,
                manifest_path=manifest_path,
            )
            self.assertEqual(result["arm_summary"]["A1"]["n"], 1)
            self.assertTrue((out / "FEEDBACK_REPORT_ZH.md").is_file())
            self.assertTrue((out / "feedback_arm_summary.csv").is_file())
            self.assertTrue((out / "feedback_task_rows.csv").is_file())

    def test_load_skips_non_subset_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state" / "s1"
            state_dir.mkdir(parents=True)
            base = {
                "sample_id": "s1",
                "status": "completed",
                "geometry": {"joint_quality": 0.4},
                "feedback": {"arm": "A0", "kept_round": 0, "rounds": [_round(0, {})]},
            }
            for condition in ("T2", "T1"):
                state = dict(base, condition_id=condition)
                (state_dir / f"{condition}.json").write_text(
                    json.dumps(state), encoding="utf-8"
                )
            rows = load_feedback_states(root, "A0", MANIFEST)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["condition"], "T2")
            self.assertIn("T2", FEEDBACK_CONDITIONS)


if __name__ == "__main__":
    unittest.main()
