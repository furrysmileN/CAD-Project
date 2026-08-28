# -*- coding: utf-8 -*-
"""RQ2b 确认实验运行器单元测试：A0/C 双臂行为、反馈轮、断点续跑与 dry-run。"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness import confirm_runner

REPO = Path(__file__).resolve().parents[1]
PILOT_MANIFEST = REPO / "outputs" / "pilot_v2" / "manifest.jsonl"

BOX_PLAN_TEMPLATE = (
    '{{"schema_version": "harnesscad.plan.v2", "sample_id": "{sample_id}", '
    '"coordinate_system": {{"units": "normalized", "origin": [0, 0, 0], '
    '"longest_bbox_edge": 1.0}}, "operations": [{{"id": "base", "op": "box", '
    '"combine": "new", "center": [0, 0, 0], "size": [0.5, 0.5, 0.3]}}]}}'
)


def _api_result(text: str) -> dict:
    return {
        "text": text,
        "model": "test-model",
        "base_url": "test",
        "finish_reason": "stop",
        "attempt": 1,
        "attempt_latency_sec": 0.01,
        "latency_sec": 0.01,
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        "retry_errors": [],
    }


def _sample_id_from_messages(messages: list[dict]) -> str:
    content = messages[1]["content"]
    for part in content:
        if part.get("type") == "text":
            import re

            match = re.search(r'sample_id "?(s_[0-9a-f]{16})"?', part["text"])
            if match:
                return match.group(1)
    raise AssertionError("prompt 中未找到 sample_id")


def _execution_failed_episode() -> dict:
    return {
        "result_step_path": None,
        "response": {
            "status": "failed",
            "failure": {
                "code": "operation_exception",
                "message": "BRep_API: command not done",
                "operationIndex": 0,
            },
        },
    }


def _validation_failed_episode() -> dict:
    return {
        "result_step_path": None,
        "response": {
            "status": "validation_failed",
            "failure": {
                "code": "plan_validation_failed",
                "message": "plan rejected by schema validator",
            },
            "validation": {
                "valid": False,
                "issues": [
                    {
                        "code": "invalid_revolve_axis",
                        "path": "$.operations[0]",
                        "message": "axis points coincide",
                    }
                ],
            },
        },
    }


def _success_episode() -> dict:
    return {
        "result_step_path": "fake_result.step",
        "response": {"status": "success"},
    }


class ConfirmRunnerTests(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("VLM_API_KEY", "test-key")
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        manifest_src = PILOT_MANIFEST
        self.assertTrue(manifest_src.is_file(), "缺少 pilot_v2 manifest，无法运行确认实验测试")
        self.manifest_path = self.out / "manifest.jsonl"
        shutil.copyfile(manifest_src, self.manifest_path)
        self.config = confirm_runner.load_config(REPO / "configs" / "confirm_n100.yaml")
        self.config["paths"]["manifest"] = str(self.manifest_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, arm: str, responses, episode_responses=None, *, limit=1, force=False):
        arm_block = dict(self.config["arms"][arm])
        arm_block["output_dir"] = str(self.out / arm)
        self.config["arms"][arm] = arm_block
        counter = {"n": 0}

        def fake_chat(messages, settings):
            index = counter["n"]
            counter["n"] += 1
            if index >= len(responses):
                raise AssertionError("模拟响应耗尽：反馈循环超出了预期轮数")
            return _api_result(responses[index](messages))

        patches = [
            mock.patch.object(confirm_runner, "chat_completion", side_effect=fake_chat),
            mock.patch.object(
                confirm_runner,
                "score_step_pair",
                return_value={"valid": True, "joint_quality": 0.5},
            ),
        ]
        if episode_responses is not None:
            patches.append(
                mock.patch.object(confirm_runner, "run_episode", side_effect=episode_responses)
            )
        else:
            patches.append(
                mock.patch.object(
                    confirm_runner,
                    "run_episode",
                    side_effect=lambda *args, **kwargs: _success_episode(),
                )
            )
        with patches[0], patches[1], patches[2]:
            confirm_runner.run_confirmation(
                self.config,
                conditions=("T",),
                limit=limit,
                arm=arm,
                force=force,
            )
        state_files = sorted((self.out / arm / "state").glob("*/*.json"))
        return state_files

    def _load_state(self, arm: str) -> dict:
        files = sorted((self.out / arm / "state").glob("*/*.json"))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text(encoding="utf-8"))

    def test_a0_single_call_no_feedback(self):
        def round0(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        self._run("A0", [round0])
        state = self._load_state("A0")
        self.assertEqual(state["status"], "completed")
        self.assertFalse(state["feedback"]["enabled"])
        self.assertEqual(state["feedback"]["max_rounds"], 0)
        self.assertEqual(len(state["feedback"]["rounds"]), 1)
        self.assertEqual(state["plan_version"], "v2")
        self.assertEqual(state["repair"]["version"], "none")

    def test_c_arm_repair_and_schema_feedback_round(self):
        def round0(messages):
            return "this is not valid json {"

        def round1(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        self._run("C", [round0, round1])
        state = self._load_state("C")
        self.assertEqual(state["status"], "completed")
        rounds = state["feedback"]["rounds"]
        self.assertEqual(len(rounds), 2)
        self.assertEqual(rounds[0]["failure"]["kind"], "schema")
        self.assertEqual(state["feedback"]["kept_round"], 1)
        self.assertEqual(state["plan_version"], "v3")
        self.assertEqual(state["repair"]["version"], "harnesscad.repair.v2.1")

    def test_c_arm_execution_feedback_round(self):
        def round0(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        def round1(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        self._run(
            "C",
            [round0, round1],
            episode_responses=[_execution_failed_episode(), _success_episode()],
        )
        state = self._load_state("C")
        self.assertEqual(state["status"], "completed")
        rounds = state["feedback"]["rounds"]
        self.assertEqual(len(rounds), 2)
        self.assertEqual(rounds[0]["failure"]["kind"], "execution")
        self.assertEqual(state["feedback"]["kept_round"], 1)

    def test_c_arm_backend_plan_validation_failed_is_schema(self):
        def round0(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        def round1(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        self._run(
            "C",
            [round0, round1],
            episode_responses=[_validation_failed_episode(), _success_episode()],
        )
        state = self._load_state("C")
        rounds = state["feedback"]["rounds"]
        self.assertEqual(len(rounds), 2)
        self.assertEqual(rounds[0]["failure"]["kind"], "schema")
        self.assertEqual(
            rounds[0]["failure"]["issues"][0]["code"], "invalid_revolve_axis"
        )
        self.assertEqual(state["status"], "completed")

    def test_terminal_state_is_skipped_on_rerun(self):
        def round0(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        self._run("A0", [round0])
        summary = confirm_runner.run_confirmation(
            self.config,
            conditions=("T",),
            limit=1,
            arm="A0",
        )
        self.assertEqual(summary["counts"].get("skipped"), 1)
        self.assertEqual(summary["counts"].get("completed", 0), 0)

    def test_force_reruns_terminal_state(self):
        def round0(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        self._run("A0", [round0])
        self._run("A0", [round0], force=True)
        state = self._load_state("A0")
        self.assertEqual(state["status"], "completed")
        history = list((self.out / "A0" / "history").rglob("*.json"))
        self.assertEqual(len(history), 1, "强制重跑应归档旧 state")

    def test_dry_run_does_not_call_api(self):
        arm_block = dict(self.config["arms"]["A0"])
        arm_block["output_dir"] = str(self.out / "A0")
        self.config["arms"]["A0"] = arm_block
        with mock.patch.object(confirm_runner, "chat_completion") as fake_chat:
            summary = confirm_runner.run_confirmation(
                self.config,
                conditions=("T",),
                limit=1,
                arm="A0",
                dry_run=True,
            )
        fake_chat.assert_not_called()
        self.assertEqual(summary["counts"].get("dry_run"), 1)
        state = self._load_state("A0")
        self.assertEqual(state["status"], "dry_run")


class ConfirmTaskOrderTests(unittest.TestCase):
    def test_task_order_matches_pilot_v2_seed(self):
        rows = [{"sample_id": "s1"}, {"sample_id": "s2"}]
        tasks = confirm_runner._task_order(rows, ("T", "I"), seed=42)
        self.assertEqual(len(tasks), 4)
        keys = [
            hashlib_sort_key(row["sample_id"], condition)
            for row, condition in tasks
        ]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(tasks[0][1], "T")


def hashlib_sort_key(sample_id: str, condition: str) -> str:
    import hashlib

    return hashlib.sha256(f"42:{sample_id}:{condition}".encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
