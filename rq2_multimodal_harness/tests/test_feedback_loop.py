from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness import encoding_runner
from rq2_harness.encoding_conditions import parse_condition
from rq2_harness.feedback import (
    ARM_PRESETS,
    build_execution_feedback,
    build_schema_feedback,
    failure_kind_from_code,
    feedback_turn,
    resolve_feedback_config,
)

REPO = Path(__file__).resolve().parents[1]
REAL_STATE_DIR = REPO / "outputs" / "encoding_screen_n20" / "state"

BOX_PLAN_TEMPLATE = (
    '{{"schema_version": "harnesscad.plan.v2", "sample_id": "{sample_id}", '
    '"coordinate_system": {{"units": "normalized", "origin": [0, 0, 0], '
    '"longest_bbox_edge": 1.0}}, "operations": [{{"id": "base", "op": "box", '
    '"combine": "new", "center": [0, 0, 0], "size": [0.5, 0.5, 0.3]}}]}}'
)

THIN_BOX_WITH_CHAMFER = (
    '{{"schema_version": "harnesscad.plan.v2", "sample_id": "{sample_id}", '
    '"coordinate_system": {{"units": "normalized", "origin": [0, 0, 0], '
    '"longest_bbox_edge": 1.0}}, "operations": ['
    '{{"id": "seat", "op": "box", "combine": "new", "center": [0, 0, 0], "size": [0.6, 0.6, 0.05]}},'
    '{{"id": "leg1", "op": "box", "combine": "add", "center": [-0.25, -0.25, -0.25], "size": [0.04, 0.04, 0.45]}},'
    '{{"id": "leg2", "op": "box", "combine": "add", "center": [0.25, -0.25, -0.25], "size": [0.04, 0.04, 0.45]}},'
    '{{"id": "leg3", "op": "box", "combine": "add", "center": [-0.25, 0.25, -0.25], "size": [0.04, 0.04, 0.45]}},'
    '{{"id": "leg4", "op": "box", "combine": "add", "center": [0.25, 0.25, -0.25], "size": [0.04, 0.04, 0.45]}},'
    '{{"id": "backrest", "op": "box", "combine": "add", "center": [0, 0.28, 0.3], "size": [0.6, 0.04, 0.55]}},'
    '{{"id": "seat_chamfer", "op": "chamfer", "distance": 0.02}}]}}'
)


def _sample_id_from_messages(messages: list[dict]) -> str:
    content = messages[1]["content"]
    for part in content:
        if part.get("type") == "text":
            match = re.search(r'sample_id "?(s_[0-9a-f]{16})"?', part["text"])
            if match:
                return match.group(1)
    raise AssertionError("prompt 中未找到 sample_id")


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


class FeedbackConfigTests(unittest.TestCase):
    def test_arm_presets(self):
        self.assertEqual(
            resolve_feedback_config({"feedback": {"arm": "A0"}})["plan_prompt_version"],
            "v2",
        )
        a1 = resolve_feedback_config({"feedback": {"arm": "A1"}})
        self.assertEqual(a1["plan_prompt_version"], "v3")
        self.assertFalse(a1["enabled"])
        b1 = resolve_feedback_config({"feedback": {"arm": "B1"}})
        self.assertTrue(b1["enabled"])
        self.assertEqual(b1["max_rounds"], 1)
        self.assertEqual(b1["sources"], ["schema"])
        b2 = resolve_feedback_config({"feedback": {"arm": "B2"}})
        self.assertEqual(b2["max_rounds"], 2)
        self.assertEqual(b2["sources"], ["schema", "execution"])
        c = resolve_feedback_config({"feedback": {"arm": "C"}})
        self.assertEqual(c["plan_prompt_version"], "v3")
        self.assertEqual(c["sources"], ["schema", "execution"])

    def test_custom_arm_keeps_explicit_values(self):
        block = resolve_feedback_config(
            {
                "feedback": {
                    "arm": "custom",
                    "enabled": True,
                    "max_rounds": 3,
                    "sources": ["schema", "execution"],
                }
            }
        )
        self.assertTrue(block["enabled"])
        self.assertEqual(block["max_rounds"], 3)
        self.assertEqual(block["plan_prompt_version"], "v2")

    def test_all_presets_declared(self):
        for arm in ("A0", "A1", "B1", "B2", "C"):
            self.assertIn(arm, ARM_PRESETS)

    def test_failure_kind_from_code_mapping(self):
        # 后端 plan 校验拒绝必须归为格式类（runner 与分析共用的唯一映射）
        self.assertEqual(failure_kind_from_code("plan_validation_failed"), "format")
        # 执行类异常
        self.assertEqual(failure_kind_from_code("operation_exception"), "execution")
        self.assertEqual(failure_kind_from_code("invalid_shape_after_operation"), "execution")
        # 未知/缺失失败码：不武断归类，交给调用方回退
        self.assertEqual(failure_kind_from_code("unknown_code"), "execution")
        self.assertIsNone(failure_kind_from_code(None))
        self.assertIsNone(failure_kind_from_code(""))
        self.assertIsNone(failure_kind_from_code(0))


class FeedbackMessageTests(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "schema_version": "harnesscad.plan.v2",
            "sample_id": "s_test",
            "operations": [
                {"id": "base", "op": "box", "combine": "new", "center": [0, 0, 0], "size": [0.5, 0.5, 0.3]},
                {
                    "id": "rot",
                    "op": "transform",
                    "combine": "add",
                    "source": "base",
                    "rotate": {"origin": [0, 0], "axis": [0, 0, 0], "angle": 90},
                },
            ],
        }

    def test_schema_feedback_includes_issue_and_operation(self):
        text = build_schema_feedback(
            [{"code": "invalid_rotate", "path": "$.operations[1].rotate", "message": "Expected origin, unit axis and angle."}],
            self.plan,
            0,
        )
        self.assertIn("invalid_rotate", text)
        self.assertIn("$.operations[1].rotate", text)
        self.assertIn('"op": "transform"', text)

    def test_schema_feedback_round2_opener(self):
        text = build_schema_feedback([{"code": "invalid_number", "path": "$.x"}], self.plan, 1)
        self.assertIn("again", text)

    def test_execution_feedback_includes_operation(self):
        failure = {
            "code": "operation_exception",
            "message": "BRep_API: command not done",
            "operationId": "rot",
            "operationIndex": 1,
        }
        text = build_execution_feedback(failure, self.plan, 0)
        self.assertIn("operation_exception", text)
        self.assertIn("operation index: 1", text)
        self.assertIn("operation id: rot", text)
        self.assertIn('"op": "transform"', text)

    def test_feedback_turn_structure(self):
        base = [{"role": "system", "content": "sys"}, {"role": "user", "content": [{"type": "text", "text": "task"}]}]
        turn = feedback_turn(base, "previous raw", "fix it")
        self.assertEqual(len(turn), 4)
        self.assertEqual(turn[2], {"role": "assistant", "content": "previous raw"})
        self.assertEqual(turn[3]["role"], "user")
        self.assertEqual(turn[3]["content"][0]["text"], "fix it")


class FeedbackLoopEndToEndTests(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("VLM_API_KEY", "test-key")
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        run_out = self.out / "run_out"
        run_out.mkdir()
        manifest = REPO / "outputs" / "encoding_screen_n20" / "sample_manifest.jsonl"
        self.assertTrue(manifest.is_file(), "缺少冻结样本清单，无法运行端到端测试")
        shutil.copyfile(manifest, run_out / "sample_manifest.jsonl")
        self.config = encoding_runner.load_config(REPO / "configs" / "feedback_n20.yaml")
        self.config["paths"]["output_dir"] = str(run_out)

    def tearDown(self):
        self.tmp.cleanup()

    def _run_with_fake(self, arm: str, responses: list[callable]):
        self.config["feedback"]["arm"] = arm
        counter = {"n": 0}

        def fake_chat(messages, settings):
            index = counter["n"]
            counter["n"] += 1
            if index >= len(responses):
                raise AssertionError("模拟响应耗尽：反馈循环超出了预期轮数")
            return _api_result(responses[index](messages))

        with mock.patch.object(encoding_runner, "chat_completion", side_effect=fake_chat):
            encoding_runner.run_encoding_screen(
                self.config,
                conditions=(parse_condition("T1"),),
                limit=1,
            )
        state_files = list((self.out / "run_out" / "state").glob("*/*.json"))
        self.assertEqual(len(state_files), 1)
        return json.loads(state_files[0].read_text(encoding="utf-8"))

    def test_execution_feedback_round_fixes_task(self):
        def round0(messages):
            sample_id = _sample_id_from_messages(messages)
            return THIN_BOX_WITH_CHAMFER.format(sample_id=sample_id)

        def round1(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        state = self._run_with_fake("C", [round0, round1])
        self.assertEqual(state["status"], "completed")
        rounds = state["feedback"]["rounds"]
        self.assertEqual(len(rounds), 2)
        self.assertEqual(rounds[0]["failure"]["kind"], "execution")
        self.assertEqual(state["feedback"]["kept_round"], 1)
        self.assertEqual(state["feedback"]["arm"], "C")
        self.assertGreater(state["geometry"]["joint_quality"], 0.0)

    def test_schema_feedback_round_fixes_task(self):
        def round0(messages):
            return "this is not valid json {"

        def round1(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        state = self._run_with_fake("C", [round0, round1])
        self.assertEqual(state["status"], "completed")
        rounds = state["feedback"]["rounds"]
        self.assertEqual(len(rounds), 2)
        self.assertEqual(rounds[0]["failure"]["kind"], "schema")
        self.assertEqual(state["feedback"]["kept_round"], 1)

    def test_keep_best_keeps_completed_round_after_two_failures(self):
        def round0(messages):
            return "this is not valid json {"

        def round1(messages):
            sample_id = _sample_id_from_messages(messages)
            return THIN_BOX_WITH_CHAMFER.format(sample_id=sample_id)

        def round2(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        state = self._run_with_fake("C", [round0, round1, round2])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["feedback"]["kept_round"], 2)
        self.assertEqual(len(state["feedback"]["rounds"]), 3)
        self.assertEqual(state["feedback"]["rounds"][0]["failure"]["kind"], "schema")
        self.assertEqual(state["feedback"]["rounds"][1]["failure"]["kind"], "execution")

    def test_no_feedback_when_arm_a1(self):
        def round0(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        state = self._run_with_fake("A1", [round0])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(state["feedback"]["rounds"]), 1)
        self.assertFalse(state["feedback"]["enabled"])

    @staticmethod
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

    def _run_with_fake_episode(self, arm, responses, episode_responses):
        self.config["feedback"]["arm"] = arm
        counter = {"n": 0}

        def fake_chat(messages, settings):
            index = counter["n"]
            counter["n"] += 1
            if index >= len(responses):
                raise AssertionError("模拟响应耗尽：反馈循环超出了预期轮数")
            return _api_result(responses[index](messages))

        with mock.patch.object(
            encoding_runner, "chat_completion", side_effect=fake_chat
        ), mock.patch.object(
            encoding_runner, "run_episode", side_effect=episode_responses
        ):
            encoding_runner.run_encoding_screen(
                self.config,
                conditions=(parse_condition("T1"),),
                limit=1,
            )
        state_files = list((self.out / "run_out" / "state").glob("*/*.json"))
        self.assertEqual(len(state_files), 1)
        return json.loads(state_files[0].read_text(encoding="utf-8"))

    def test_backend_plan_validation_failed_records_schema_kind_and_triggers_schema_feedback(self):
        captured: dict[str, str] = {}

        def round0(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        def round1(messages):
            sample_id = _sample_id_from_messages(messages)
            captured["feedback_text"] = messages[-1]["content"][0]["text"]
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        def round2(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        state = self._run_with_fake_episode(
            "C",
            [round0, round1, round2],
            [
                self._validation_failed_episode(),
                self._validation_failed_episode(),
                self._validation_failed_episode(),
            ],
        )
        rounds = state["feedback"]["rounds"]
        self.assertEqual(len(rounds), 3)
        self.assertEqual(rounds[0]["failure"]["kind"], "schema")
        self.assertEqual(
            rounds[0]["failure"]["issues"][0]["code"], "invalid_revolve_axis"
        )
        self.assertIn("schema validator", captured["feedback_text"])
        self.assertEqual(rounds[1]["failure"]["kind"], "schema")

    def test_backend_plan_validation_failed_triggers_feedback_for_schema_only_arm(self):
        def round0(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        def round1(messages):
            sample_id = _sample_id_from_messages(messages)
            return BOX_PLAN_TEMPLATE.format(sample_id=sample_id)

        state = self._run_with_fake_episode(
            "B1",
            [round0, round1],
            [self._validation_failed_episode(), self._validation_failed_episode()],
        )
        rounds = state["feedback"]["rounds"]
        self.assertEqual(len(rounds), 2)
        self.assertEqual(rounds[0]["failure"]["kind"], "schema")


if __name__ == "__main__":
    unittest.main()
