"""V5 Phase D：query_or_submit 协议 + 累积查询历史。

不改 C 臂 feedback.py 的非累积语义。本模块只服务工具实验。
"""
from __future__ import annotations

import json
from typing import Any

from .pc_fsm import ToolLoopConfig, ToolLoopState, _timed_execute
from .pointcloud.tools import PointCloudSession, parse_query_request
from .prompting import parse_plan_response


def parse_query_or_submit(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    payload: dict[str, Any] | None = None
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            payload = loaded
    except json.JSONDecodeError:
        parsed = parse_query_request(raw)
        if isinstance(parsed.get("request"), dict):
            payload = parsed["request"]
        else:
            plan = parse_plan_response(raw)
            if isinstance(plan, dict) and plan.get("operations") is not None:
                return {"kind": "submit_plan", "action": "submit_plan", "plan": plan, "raw": raw}
            return {"kind": "invalid", "error": "not_json", "raw": raw}
    if not isinstance(payload, dict):
        return {"kind": "invalid", "error": "not_object", "raw": raw}
    action = str(payload.get("action") or "").strip()
    if action in {"", "submit_plan"} and isinstance(payload.get("plan"), dict):
        return {"kind": "submit_plan", "action": "submit_plan", "plan": payload.get("plan"), "raw": raw}
    if action in {"", "submit_plan"} and isinstance(payload.get("operations"), list):
        return {"kind": "submit_plan", "action": "submit_plan", "plan": payload, "raw": raw}
    if action == "query":
        query = payload.get("query") or {}
        if not isinstance(query, dict) or not isinstance(query.get("tool"), str):
            return {"kind": "invalid", "error": "query_missing_tool", "raw": raw}
        return {
            "kind": "query",
            "action": "query",
            "request": {"tool": query.get("tool"), "params": query.get("arguments") or query.get("params") or {}},
            "raw": raw,
        }
    if action == "query_or_submit":
        return parse_query_or_submit(json.dumps({k: v for k, v in payload.items() if k != "action"}))
    return {"kind": "invalid", "error": f"unknown_action:{action or 'missing'}", "raw": raw}


def forced_cross_section_request(cloud_id: str) -> dict[str, Any]:
    return {
        "tool": "query_cross_section",
        "params": {
            "cloud_id": cloud_id,
            "origin": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
        },
        "reason": "forced diagnostic query",
    }


def append_query_turn(
    messages: list[dict[str, Any]],
    *,
    request: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """累积：在完整历史后追加 assistant 请求与 user 结果。"""
    updated = list(messages)
    updated.append({"role": "assistant", "content": json.dumps({"action": "query", "query": request}, ensure_ascii=False)})
    updated.append(
        {
            "role": "user",
            "content": (
                "[POINT_TOOL_RESULT]\n"
                + json.dumps(result, ensure_ascii=False, indent=2)[:6000]
                + "\nQuery history is accumulated. Submit a plan or issue another query."
            ),
        }
    )
    return updated


def run_forced_query(
    session: PointCloudSession,
    *,
    cloud_id: str,
    config: ToolLoopConfig,
    state: ToolLoopState,
) -> dict[str, Any]:
    request = forced_cross_section_request(cloud_id)
    result = _timed_execute(session, request, config.timeout_sec)
    state.pre_queries += 1
    state.traces.append({"tool": request["tool"], "params": request["params"], "result": result, "forced": True})
    return result
