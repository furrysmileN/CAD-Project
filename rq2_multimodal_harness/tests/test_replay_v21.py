from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness import replay_v21
from rq2_harness.backend import run_episode
from rq2_harness.replay_analysis import _bootstrap_values, _exact_mcnemar


def sphere_plan() -> dict:
    return {
        "schema_version": "harnesscad.plan.v2",
        "sample_id": "s_test",
        "coordinate_system": {"units": "normalized", "origin": [0, 0, 0], "longest_bbox_edge": 1.0},
        "operations": [
            {"id": "base", "op": "sphere", "combine": "new", "center": [0, 0, 0], "radius": 0.5}
        ],
    }


class ReplayV21Tests(unittest.TestCase):
    def test_replay_module_has_no_api_client_dependency(self) -> None:
        source = inspect.getsource(replay_v21)
        self.assertNotIn("api_client", source)
        self.assertNotIn("chat_completion", source)

    def test_setting_parse_repairs_numeric_axis_without_changing_raw(self) -> None:
        plan = sphere_plan()
        plan["operations"] = [
            {
                "id": "base",
                "op": "cylinder",
                "combine": "new",
                "center": [0, 0, 0],
                "radius": "0.2",
                "height": 1,
                "axis": [0, 0, 2],
            }
        ]
        raw = json.dumps(plan)
        parsed, log = replay_v21._parse_for_setting(raw, ["number", "unit_axis"])
        self.assertTrue(parsed["ok"])
        self.assertTrue(log["changed"])
        self.assertEqual(parsed["plan"]["operations"][0]["axis"], [0.0, 0.0, 1.0])
        self.assertIn('"axis": [0, 0, 2]', raw)

    def test_task_fingerprint_changes_with_repair_setting(self) -> None:
        common = {
            "baseline_state_sha256": "a",
            "raw_response_sha256": "b",
            "plan_sha256": "c",
            "code_fingerprint": "d",
            "scoring_fingerprint": "e",
        }
        first = replay_v21._task_fingerprint(setting="R1", **common)
        second = replay_v21._task_fingerprint(setting="R4", **common)
        self.assertNotEqual(first, second)

    def test_tree_snapshot_detects_no_read_only_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample" / "T.json"
            path.parent.mkdir()
            path.write_text('{"status":"completed"}\n', encoding="utf-8")
            first = replay_v21._tree_snapshot(root)
            _ = path.read_text(encoding="utf-8")
            second = replay_v21._tree_snapshot(root)
            self.assertEqual(first, second)

    def test_exact_mcnemar_and_bootstrap(self) -> None:
        result = _exact_mcnemar([False, False, True, True], [True, False, True, False])
        self.assertEqual(result["rescued"], 1)
        self.assertEqual(result["regressed"], 1)
        self.assertEqual(result["p_value"], 1.0)
        first = _bootstrap_values([0.1, 0.2, -0.1], seed=42, repeats=100)
        second = _bootstrap_values([0.1, 0.2, -0.1], seed=42, repeats=100)
        self.assertEqual(first, second)
        self.assertEqual((first["wins"], first["ties"], first["losses"]), (2, 0, 1))

    def test_episode_artifacts_can_be_routed_to_replay_root(self) -> None:
        try:
            import cadquery  # noqa: F401
        except ImportError:
            self.skipTest("cadquery unavailable")
        root = Path(__file__).resolve().parents[3] / "HarnessCAD" / "HarnessCAD"
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "runs"
            episode = run_episode(
                sphere_plan(),
                {"episode_version": "v2", "root": str(root), "timeout_sec": 30},
                run_root=run_root,
            )
            episode_path = Path(episode["episode_path"])
            self.assertTrue(episode_path.is_file())
            self.assertEqual(episode_path.parent.parent, run_root.resolve())
            self.assertTrue(Path(episode["result_step_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
