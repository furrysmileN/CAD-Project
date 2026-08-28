# -*- coding: utf-8 -*-
"""RQ2b 确认实验分析函数单元测试：合成 state 校验行级口径、修复率与配对检验。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.confirm_analysis import (
    CONFIRM_CONDITIONS,
    analyze_confirmation,
    arm_summary,
    bootstrap_paired_delta,
    mcnemar_exact,
    task_row,
    wilcoxon_signed_rank,
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


class ConfirmTaskRowTests(unittest.TestCase):
    def test_fixed_after_execution_failure(self):
        state = {
            "sample_id": "s1",
            "condition_id": "T",
            "status": "completed",
            "geometry": {"joint_quality": 0.42},
            "feedback": {
                "arm": "C",
                "kept_round": 1,
                "rounds": [
                    _round(0, {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}, "execution"),
                    _round(1, {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220}),
                ],
            },
        }
        row = task_row(state, arm="C")
        self.assertTrue(row["completed"])
        self.assertTrue(row["fixed"])
        self.assertEqual(row["fixed_at_round"], 1)
        self.assertEqual(row["round0_failure_kind"], "execution")
        self.assertIsNone(row["round1_failure_kind"])
        self.assertEqual(row["n_rounds"], 2)
        self.assertEqual(row["input_tokens"], 300.0)
        self.assertEqual(row["feedback_input_tokens"], 200.0)

    def test_new_error_kind_detected(self):
        state = {
            "sample_id": "s1",
            "condition_id": "I",
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
        row = task_row(state, arm="C")
        self.assertFalse(row["fixed"])
        self.assertTrue(row["new_error_kind"])
        self.assertEqual(row["round1_failure_kind"], "execution")

    def test_legacy_state_without_feedback_block(self):
        state = {
            "sample_id": "s1",
            "condition_id": "P",
            "status": "parse_failed",
            "geometry": {},
            "api": {"usage": {"prompt_tokens": 90, "completion_tokens": 9}},
        }
        row = task_row(state, arm="A0")
        self.assertEqual(row["n_rounds"], 1)
        self.assertEqual(row["round0_failure_kind"], "format")
        self.assertEqual(row["input_tokens"], 90.0)

    def test_legacy_episode_failed_execution(self):
        state = {
            "sample_id": "s1",
            "condition_id": "T",
            "status": "episode_failed",
            "geometry": {},
            "api": {"usage": {"prompt_tokens": 90, "completion_tokens": 9}},
            "episode": {
                "response": {
                    "failure": {"code": "operation_exception", "message": "boom"},
                }
            },
        }
        row = task_row(state, arm="A0")
        self.assertEqual(row["round0_failure_kind"], "execution")

    def test_legacy_episode_failed_plan_validation_is_format(self):
        state = {
            "sample_id": "s1",
            "condition_id": "T",
            "status": "episode_failed",
            "geometry": {},
            "api": {"usage": {"prompt_tokens": 90, "completion_tokens": 9}},
            "episode": {
                "response": {
                    "failure": {"code": "plan_validation_failed", "message": "rejected"},
                }
            },
        }
        row = task_row(state, arm="A0")
        self.assertEqual(row["round0_failure_kind"], "format")


class ConfirmArmSummaryTests(unittest.TestCase):
    def test_fix_rates_by_failure_type(self):
        rows = [
            task_row(
                {
                    "sample_id": "s1",
                    "condition_id": "T",
                    "status": "completed",
                    "geometry": {"joint_quality": 0.5},
                    "feedback": {
                        "arm": "C",
                        "kept_round": 1,
                        "rounds": [
                            _round(0, {}, "schema"),
                            _round(1, {}),
                        ],
                    },
                },
                arm="C",
            ),
            task_row(
                {
                    "sample_id": "s2",
                    "condition_id": "T",
                    "status": "episode_failed",
                    "geometry": {},
                    "feedback": {
                        "arm": "C",
                        "kept_round": 1,
                        "rounds": [
                            _round(0, {}, "execution"),
                            _round(1, {}, "execution"),
                        ],
                    },
                },
                arm="C",
            ),
        ]
        summary = arm_summary(rows)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["format_fix_rate"], 1.0)
        self.assertEqual(summary["execution_fix_rate"], 0.0)
        self.assertEqual(summary["fixed_at_round1"], 1)


class ConfirmStatsTests(unittest.TestCase):
    def test_mcnemar_exact(self):
        result = mcnemar_exact(
            __import__("numpy").array([True, True, True, False]),
            __import__("numpy").array([True, False, True, True]),
        )
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["only_a_completed"], 1)
        self.assertEqual(result["only_b_completed"], 1)
        self.assertAlmostEqual(result["p_value"], 1.0)

    def test_wilcoxon_signed_rank_all_positive(self):
        import numpy as np

        result = wilcoxon_signed_rank(np.array([0.1, 0.2, 0.3]))
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["stat"], 0.0)
        self.assertGreater(result["mean_diff"], 0.0)

    def test_bootstrap_paired_delta(self):
        deltas = {
            ("s1", "T"): 0.1,
            ("s1", "I"): 0.2,
            ("s2", "T"): -0.1,
            ("s2", "I"): 0.0,
        }
        result = bootstrap_paired_delta(
            ["s1", "s2"], deltas, repeats=200, seed=42
        )
        self.assertEqual(result["repeats"], 200)
        self.assertAlmostEqual(result["observed_mean"], 0.05)


class AnalyzeConfirmationTests(unittest.TestCase):
    def _write_manifest(self, root: Path) -> Path:
        manifest_path = root / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for sample in MANIFEST.values():
                handle.write(json.dumps(sample) + "\n")
        return manifest_path

    def _write_state(self, state_dir: Path, sample_id: str, condition: str, state: dict) -> None:
        target = state_dir / sample_id
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{condition}.json").write_text(json.dumps(state), encoding="utf-8")

    def _make_state(self, sample_id: str, condition: str, status: str, jq: float = 0.0) -> dict:
        return {
            "sample_id": sample_id,
            "condition_id": condition,
            "status": status,
            "geometry": {"joint_quality": jq} if status == "completed" else {},
            "feedback": {
                "arm": "C",
                "kept_round": 0,
                "rounds": [_round(0, {"prompt_tokens": 10, "completion_tokens": 2})],
            },
        }

    def test_end_to_end_writes_outputs_and_paired_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in ("A0", "C"):
                state_dir = root / arm
                state_dir.mkdir(parents=True)
                for sample_id, condition in (("s1", "T"), ("s2", "I")):
                    self._write_state(
                        state_dir,
                        sample_id,
                        condition,
                        self._make_state(
                            sample_id,
                            condition,
                            "completed" if (arm == "C" or condition == "T") else "episode_failed",
                            jq=0.4 if (arm == "C" or condition == "T") else 0.0,
                        ),
                    )
            manifest_path = self._write_manifest(root)
            out = root / "analysis"
            result = analyze_confirmation(
                {"A0": root / "A0", "C": root / "C"},
                out,
                manifest_path,
                bootstrap_repeats=50,
                seed=42,
            )
            self.assertEqual(result["arm_summary"]["A0"]["n"], 2)
            self.assertEqual(result["arm_summary"]["C"]["n"], 2)
            self.assertIn("a0_vs_c", result["paired"])
            self.assertTrue((out / "CONFIRM_REPORT_ZH.md").is_file())
            self.assertTrue((out / "confirm_arm_summary.csv").is_file())
            self.assertTrue((out / "confirm_task_rows.csv").is_file())
            self.assertTrue((out / "confirm_analysis.json").is_file())

    def test_load_skips_non_confirm_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "A0"
            state_dir.mkdir(parents=True)
            base = {
                "sample_id": "s1",
                "status": "completed",
                "geometry": {"joint_quality": 0.4},
                "feedback": {"arm": "A0", "kept_round": 0, "rounds": [_round(0, {})]},
            }
            for condition in ("T", "T1"):
                state = dict(base, condition_id=condition)
                self._write_state(state_dir, "s1", condition, state)
            manifest_path = self._write_manifest(root)
            out = root / "analysis"
            result = analyze_confirmation(
                {"A0": state_dir},
                out,
                manifest_path,
                bootstrap_repeats=10,
                seed=42,
            )
            self.assertEqual(result["arm_summary"]["A0"]["n"], 1)
            rows = []
            for path in sorted((state_dir / "s1").glob("*.json")):
                state = json.loads(path.read_text(encoding="utf-8"))
                if state["condition_id"] in CONFIRM_CONDITIONS:
                    rows.append(task_row(state, arm="A0"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["condition"], "T")


if __name__ == "__main__":
    unittest.main()
