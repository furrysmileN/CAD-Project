from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.encoding_analysis import (
    ENCODING_CONDITIONS,
    analyze_encoding_screen,
    bootstrap_summary,
    interaction_rows,
    parse_condition,
    state_to_row,
)


def synthetic_state(
    sample_id: str,
    condition: str,
    quality: float,
    *,
    valid: bool = True,
    parse_ok: bool = True,
    schema_valid: bool = True,
    execution_success: bool = True,
) -> dict:
    episode_status = "completed" if execution_success else "failed"
    return {
        "sample_id": sample_id,
        "condition": condition,
        "status": "completed" if execution_success else "episode_failed",
        "parse": {
            "ok": parse_ok,
            "plan": {
                "operations": [
                    {"id": "base", "op": "sphere"},
                    {"id": "cut", "op": "hole"},
                ]
            },
        },
        "episode": {
            "response": {
                "status": episode_status,
                "validation": {
                    "valid": schema_valid,
                    "issues": [] if schema_valid else [{"code": "bad_schema"}],
                    "planSummary": {"operationCount": 2},
                },
                "failure": None if execution_success else {"code": "operation_exception"},
            }
        },
        "geometry": {
            "valid": valid,
            "joint_quality": quality,
            "shape_only_cd": 0.1,
            "common_frame_cd": 0.2,
            "voxel_iou": {"value": 0.8},
        },
        "api": {
            "latency_sec": 2.0,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        },
    }


class EncodingAnalysisTests(unittest.TestCase):
    def test_condition_space_has_exactly_63_valid_conditions(self) -> None:
        self.assertEqual(len(ENCODING_CONDITIONS), 63)
        self.assertEqual(len(set(ENCODING_CONDITIONS)), 63)
        self.assertEqual(parse_condition("T2I1P3"), {"T": 2, "I": 1, "P": 3})
        with self.assertRaises(ValueError):
            parse_condition("I1T2")

    def test_failure_aware_quality_and_cost(self) -> None:
        state = synthetic_state("s1", "T1", 0.9, valid=False)
        row = state_to_row(
            state,
            {"sample_id": "s1", "family": "part"},
            {
                "input_per_million_tokens": 10.0,
                "output_per_million_tokens": 20.0,
                "latency_per_second": 0.5,
            },
        )
        self.assertEqual(row["joint_quality"], 0.0)
        self.assertEqual(row["failure_stage"], "geometry")
        self.assertEqual(row["operation_count"], 2)
        self.assertAlmostEqual(row["estimated_cost"], 1.002)

    def test_bootstrap_is_deterministic_and_reports_wins(self) -> None:
        first = bootstrap_summary([1.0, 0.0, -0.5], seed=4, repeats=100)
        second = bootstrap_summary([1.0, 0.0, -0.5], seed=4, repeats=100)
        self.assertEqual(first, second)
        self.assertEqual((first["wins"], first["ties"], first["losses"]), (1, 1, 1))
        self.assertAlmostEqual(first["median"], 0.0)

    def test_only_preregistered_comparison_families_are_emitted(self) -> None:
        rows = [
            state_to_row(synthetic_state("s1", condition, index / 10))
            for index, condition in enumerate(("T1", "T2", "T1I1", "T2I1", "I1", "T1I1P1"))
        ]
        output = interaction_rows(rows, seed=3, bootstrap_repeats=10)
        families = {row["comparison_family"] for row in output}
        self.assertEqual(
            families,
            {
                "single_modality_encoding",
                "encoding_replacement_fixed_others",
                "direct_bimodal_gain",
                "trimodal_increment",
                "encoding_marginal_mean",
            },
        )
        self.assertNotIn("all_pairwise", families)
        comparison = next(
            row for row in output
            if row["comparison_family"] == "encoding_replacement_fixed_others"
            and row["left"] == "T1I1" and row["right"] == "T2I1"
        )
        self.assertEqual(comparison["n"], 1)
        self.assertAlmostEqual(comparison["mean"], 0.1)

    def test_end_to_end_with_missing_conditions_writes_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "outputs" / "encoding_screen_n20"
            state_dir = experiment / "state" / "s1"
            state_dir.mkdir(parents=True)
            (experiment / "sample_manifest.jsonl").write_text(
                json.dumps({
                    "sample_id": "s1",
                    "family": "synthetic",
                    "difficulty": "easy",
                    "complexity": 2,
                    "complexity_bin": "low",
                }) + "\n",
                encoding="utf-8",
            )
            states = {
                "T1": synthetic_state("s1", "T1", 0.2),
                "T2": synthetic_state("s1", "T2", 0.4),
                "I1": synthetic_state("s1", "I1", 0.3),
                "T1I1": synthetic_state("s1", "T1I1", 0.6),
                "T2I1": synthetic_state(
                    "s1", "T2I1", 0.7, execution_success=False, valid=False
                ),
                "T1I1P1": synthetic_state("s1", "T1I1P1", 0.8),
            }
            for condition, state in states.items():
                (state_dir / f"{condition}.json").write_text(
                    json.dumps(state), encoding="utf-8"
                )
            output = root / "analysis"
            result = analyze_encoding_screen(
                experiment,
                output,
                bootstrap_repeats=20,
                seed=11,
                cost_config={
                    "input_per_million_tokens": 1.0,
                    "output_per_million_tokens": 2.0,
                },
            )
            expected = {
                "encoding_condition_summary.csv",
                "encoding_task_rows.csv",
                "encoding_failure_summary.csv",
                "encoding_interactions.csv",
                "encoding_cost_summary.csv",
                "encoding_pareto.csv",
                "encoding_screen.json",
                "ENCODING_SCREEN_REPORT_ZH.md",
            }
            self.assertTrue(all((output / name).is_file() for name in expected))
            self.assertEqual(result["n_tasks"], len(states))
            self.assertEqual(result["missing_task_count"], 63 - len(states))
            self.assertGreaterEqual(len(list((output / "figures").glob("*.png"))), 3)
            with (output / "encoding_failure_summary.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                failures = list(csv.DictReader(handle))
            self.assertEqual(failures[0]["stage"], "execution")
            report = (output / "ENCODING_SCREEN_REPORT_ZH.md").read_text(encoding="utf-8")
            self.assertIn("不作显著性宣称", report)
            self.assertIn("缺失 sample-condition：57", report)

    def test_no_states_raises_instead_of_creating_empty_figures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state").mkdir()
            (root / "sample_manifest.jsonl").write_text(
                '{"sample_id":"s1"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "没有可分析"):
                analyze_encoding_screen(root, bootstrap_repeats=5)


if __name__ == "__main__":
    unittest.main()
