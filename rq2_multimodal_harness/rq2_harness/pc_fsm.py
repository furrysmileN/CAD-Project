"""点云工具有限状态机（非 function calling）。

模型输出 query_request JSON 或最终 Plan。每次工具结果以
assistant(请求)+user(结果) 追加，且按 feedback_turn 的非累积模式重建：
始终基于原始输入消息 + 本轮请求 + 本轮结果，不累积更早的工具历史。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .feedback import feedback_turn
from .pc_prompting import format_geometry_feedback, format_tool_user_message
from .pointcloud.tools import PointCloudSession, execute_tool, parse_query_request


PRE_GENERATION_TOOLS = {
    "get_pointcloud_summary",
    "query_cross_section",
    "detect_symmetry",
    "fit_primitives",
    "measure_pointcloud",
}
POST_GENERATION_TOOLS = {
    "compare_cad_to_cloud",
    "localize_geometric_error",
}


@dataclass
class ToolLoopConfig:
    max_pre_queries: int = 3
    max_post_queries: int = 1
    timeout_sec: float = 8.0


@dataclass
class ToolLoopState:
    pre_queries: int = 0
    post_queries: int = 0
    has_candidate: bool = False
    traces: list[dict[str, Any]] = field(default_factory=list)

    def remaining_pre(self, config: ToolLoopConfig) -> int:
        return max(0, int(config.max_pre_queries) - self.pre_queries)

    def remaining_post(self, config: ToolLoopConfig) -> int:
        return max(0, int(config.max_post_queries) - self.post_queries)


def classify_model_output(text: str) -> dict[str, Any]:
    parsed = parse_query_request(text)
    request = parsed.get("request")
    if isinstance(request, dict) and isinstance(request.get("query_request"), dict):
        inner = request["query_request"]
        if isinstance(inner.get("tool"), str):
            return {"kind": "query_request", "request": inner, "raw": parsed.get("raw") or text}
    return parsed


def _timed_execute(session: PointCloudSession, request: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    started = time.perf_counter()
    result = execute_tool(session, request)
    elapsed = time.perf_counter() - started
    if elapsed > timeout_sec and result.get("ok"):
        return {
            "ok": False,
            "error": {
                "code": "TimeoutError",
                "message": f"工具执行 {elapsed:.2f}s 超过 {timeout_sec:.1f}s",
            },
            "elapsed_sec": elapsed,
        }
    result["elapsed_sec"] = elapsed
    return result


def apply_query(
    *,
    base_messages: list[dict[str, Any]],
    raw_response: str,
    request: dict[str, Any],
    session: PointCloudSession,
    state: ToolLoopState,
    config: ToolLoopConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """执行一次查询并返回非累积的下一轮消息 + trace。"""
    tool_name = str(request.get("tool") or "")
    post_tool = tool_name in POST_GENERATION_TOOLS
    if post_tool and not state.has_candidate:
        result = {
            "ok": False,
            "error": {
                "code": "BudgetError",
                "message": "候选 STEP 尚未注册，不能调用 compare/localize",
            },
        }
        charged = "none"
    elif state.has_candidate and state.remaining_post(config) <= 0:
        result = {
            "ok": False,
            "error": {"code": "BudgetError", "message": "生成后查询预算已耗尽，请直接输出 Plan"},
        }
        charged = "none"
    elif (not state.has_candidate) and state.remaining_pre(config) <= 0:
        result = {
            "ok": False,
            "error": {"code": "BudgetError", "message": "生成前查询预算已耗尽，请直接输出 Plan"},
        }
        charged = "none"
    else:
        result = _timed_execute(session, request, config.timeout_sec)
        if state.has_candidate:
            state.post_queries += 1
            charged = "post"
        else:
            state.pre_queries += 1
            charged = "pre"

    trace = {
        "tool": tool_name,
        "params": request.get("params"),
        "reason": request.get("reason"),
        "ok": bool(result.get("ok")),
        "result": result.get("result") if result.get("ok") else None,
        "error": result.get("error"),
        "elapsed_sec": result.get("elapsed_sec"),
        "budget": charged,
        "pre_queries": state.pre_queries,
        "post_queries": state.post_queries,
    }
    state.traces.append(trace)
    user_text = format_tool_user_message(
        {k: v for k, v in result.items() if k != "elapsed_sec"},
        remaining_pre=state.remaining_pre(config),
        remaining_post=state.remaining_post(config),
        has_candidate=state.has_candidate,
    )
    messages = feedback_turn(base_messages, raw_response, user_text)
    return messages, trace


def register_candidate(
    session: PointCloudSession,
    step_path,
    *,
    candidate_step_id: str = "cand_0",
) -> str:
    from pathlib import Path

    session.candidate_steps[candidate_step_id] = Path(step_path)
    return candidate_step_id


def geometry_followup_messages(
    base_messages: list[dict[str, Any]],
    previous_raw_response: str,
    compare_result: dict[str, Any],
    *,
    candidate_step_id: str,
) -> list[dict[str, Any]]:
    text = format_geometry_feedback(compare_result, candidate_step_id=candidate_step_id)
    return feedback_turn(base_messages, previous_raw_response, text)


ChatFn = Callable[[list[dict[str, Any]]], dict[str, Any]]


def run_until_plan_or_query(
    text: str,
) -> dict[str, Any]:
    """把单次模型输出分类为 query_request / plan / unknown。"""
    return classify_model_output(text)
