# -*- coding: utf-8 -*-
"""错误反馈多轮修正：把校验/执行错误作为新消息发回模型，构造修正轮对话。

只构造消息文本与回合配置，不直接调用 API（由 encoding_runner 调用 chat_completion）。
"""
from __future__ import annotations

import json
import re
from typing import Any

FEEDBACK_VERSION = "harnesscad.feedback.v1"

SCHEMA_SOURCE = "schema"
EXECUTION_SOURCE = "execution"


def failure_kind_from_code(code: Any) -> str | None:
    """按失败码归一失败类别（runner 与分析共用的唯一映射）。

    ``plan_validation_failed`` 语义上属于格式/校验类失败，其余已知执行异常
    （operation_exception、invalid_shape_after_operation 等）属于执行类。
    未知或缺失的失败码返回 None，由调用方决定回退类别。
    """
    if not code:
        return None
    normalized = str(code)
    if normalized == "plan_validation_failed":
        return "format"
    return "execution"

# 实验臂预设：arm 字段决定 prompt 版本与反馈行为
ARM_PRESETS: dict[str, dict[str, Any]] = {
    "A0": {"plan_prompt_version": "v2", "enabled": False},
    "A1": {"plan_prompt_version": "v3", "enabled": False},
    "B1": {
        "plan_prompt_version": "v2",
        "enabled": True,
        "max_rounds": 1,
        "sources": [SCHEMA_SOURCE],
    },
    "B2": {
        "plan_prompt_version": "v2",
        "enabled": True,
        "max_rounds": 2,
        "sources": [SCHEMA_SOURCE, EXECUTION_SOURCE],
    },
    "C": {
        "plan_prompt_version": "v3",
        "enabled": True,
        "max_rounds": 2,
        "sources": [SCHEMA_SOURCE, EXECUTION_SOURCE],
    },
}

MAX_ISSUES_IN_FEEDBACK = 12


def resolve_feedback_config(config: dict[str, Any]) -> dict[str, Any]:
    block = dict(config.get("feedback") or {})
    arm = str(block.get("arm") or "custom")
    preset = ARM_PRESETS.get(arm)
    if preset is not None:
        block = {**block, **preset}
    block["arm"] = arm
    block.setdefault("plan_prompt_version", "v2")
    block.setdefault("enabled", False)
    block.setdefault("max_rounds", 0)
    # 反馈轮温度：round_index >= 1 的所有反馈轮统一使用，并非仅"第 2 轮"。
    # 字段名 round2_temperature 为历史命名，保留以兼容既有 config 与 state。
    block.setdefault("round2_temperature", 0.3)
    block.setdefault("sources", [SCHEMA_SOURCE])
    block.setdefault("keep_best", True)
    return block


def _operation_index(path: Any) -> int | None:
    match = re.search(r"operations\[(\d+)\]", str(path or ""))
    return int(match.group(1)) if match else None


def _operation_snippet(plan: Any, path: Any) -> dict[str, Any] | None:
    index = _operation_index(path)
    operations = plan.get("operations") if isinstance(plan, dict) else None
    if index is None or not isinstance(operations, list) or index >= len(operations):
        return None
    return operations[index]


def build_schema_feedback(
    issues: list[dict[str, Any]],
    plan: dict[str, Any],
    round_index: int,
) -> str:
    """格式/语义校验失败时的修正指令。"""
    if round_index == 0:
        opener = "Your previous plan was rejected by the schema validator."
    else:
        opener = "Your corrected plan was rejected again by the schema validator."
    lines = [
        opener,
        "Fix ONLY the reported problems and return the COMPLETE corrected JSON plan "
        "(same sample_id, same schema_version). Do not change unrelated operations.",
        "",
        "Reported problems:",
    ]
    for issue in issues[:MAX_ISSUES_IN_FEEDBACK]:
        code = issue.get("code")
        path = issue.get("path")
        message = issue.get("message")
        line = f"- [{code}] at {path}"
        if message:
            line += f": {message}"
        lines.append(line)
        snippet = _operation_snippet(plan, path)
        if snippet is not None:
            lines.append("  failing operation:")
            lines.append(
                "  " + json.dumps(snippet, ensure_ascii=False).replace("\n", " ")
            )
    lines.extend(
        [
            "",
            "Output the corrected JSON plan only.",
        ]
    )
    return "\n".join(lines)


def build_execution_feedback(
    failure: dict[str, Any],
    plan: dict[str, Any],
    round_index: int,
) -> str:
    """执行期失败的修正指令（含运行时异常与几何无效）。"""
    if round_index == 0:
        opener = "Your previous plan failed during execution."
    else:
        opener = "Your corrected plan failed again during execution."
    code = failure.get("code")
    message = failure.get("message")
    operation_id = failure.get("operationId")
    operation_index = failure.get("operationIndex")
    lines = [
        opener,
        "Fix ONLY the failing operation and return the COMPLETE corrected JSON plan "
        "(same sample_id, same schema_version). Keep all other operations unchanged "
        "unless required.",
        "",
        "Reported failure:",
        f"- code: {code}",
    ]
    if operation_index is not None:
        lines.append(f"- operation index: {operation_index}")
    if operation_id is not None:
        lines.append(f"- operation id: {operation_id}")
    if message:
        lines.append(f"- engine message: {message}")
    operations = plan.get("operations") if isinstance(plan, dict) else None
    if isinstance(operations, list) and operation_index is not None and operation_index < len(operations):
        lines.append("failing operation:")
        lines.append("  " + json.dumps(operations[operation_index], ensure_ascii=False).replace("\n", " "))
    lines.extend(
        [
            "",
            "Simplify or replace the failing operation if needed (e.g. remove a "
            "chamfer/fillet when the engine reports no suitable edges, or adjust size/"
            "position when a boolean produces empty material). Output the corrected "
            "JSON plan only.",
        ]
    )
    return "\n".join(lines)


def feedback_turn(
    base_messages: list[dict[str, Any]],
    previous_raw_response: str,
    feedback_text: str,
) -> list[dict[str, Any]]:
    """在 base_messages 后追加 assistant(上轮输出) + user(错误反馈) 构成修正轮消息。

    设计说明（非累积式）：runner 每轮都以「原始输入消息 + 本轮 assistant 输出 +
    本轮错误反馈」重建上下文，不携带更早轮次的输出与反馈。优点：每轮输入规模
    可控、token 增量稳定（图像 base64 只重发一轮），且避免早期错误方案在上下文中
    反复累积；代价：第 2 轮反馈看不到第 1 轮的修正历史。RQ2b 各臂均按此行为运行，
    历史 state 的口径以此为准；若未来改为累积式，应作为新的反馈协议版本记录，
    并在确认实验中与既有口径分开对比。
    """
    return [
        *base_messages,
        {"role": "assistant", "content": previous_raw_response},
        {"role": "user", "content": [{"type": "text", "text": feedback_text}]},
    ]
