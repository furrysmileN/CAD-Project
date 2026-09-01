"""Descriptive C2B / TB follow-B analysis. Cross-tab with frozen C5. Not a main-result."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import atomic_write_json
from .v6_manifest import read_manifest
from .v6b_plan_diff import KIND_LAYER
from .v6b_probe_analysis import _match_value


def _pred(state: dict[str, Any]) -> Any:
    return ((state.get("first_attempt") or {}).get("features") or {}).get("pred_value")


def load_condition_states(state_dir: Path, condition: str) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(state_dir.glob(f"*/{condition}/r*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        out[(str(state.get("sample_id") or ""), int(state.get("repeat_id") or 0))] = state
    return out


def analyze_diag_follow(
    diag_state_dir: Path,
    probe_state_dir: Path,
    manifest_path: Path,
    *,
    diag_condition: str,
    c2b_state_dir: Path | None = None,
) -> dict[str, Any]:
    meta = {row["sample_id"]: row for row in read_manifest(manifest_path) if row.get("eligible")}
    diag = load_condition_states(diag_state_dir, diag_condition)
    c5 = load_condition_states(probe_state_dir, "C5")
    c2b = load_condition_states(c2b_state_dir, "C2B") if c2b_state_dir else {}
    follow = {
        "n": 0,
        "diag_match_b": 0,
        "c5_match_b": 0,
        "c2b_match_b": 0,
        "both": 0,
        "diag_only": 0,
        "c5_only": 0,
        "neither": 0,
        "tb_yes_c2b_no": 0,
        "tb_no_c2b_no": 0,
        "tb_yes_c2b_yes": 0,
        "tb_no_c2b_yes": 0,
        "tb_yes_c2b_yes_c5_no": 0,
    }
    by_kind: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []
    for (sample_id, repeat_id), state in sorted(diag.items()):
        info = meta.get(sample_id) or {}
        category = str(((info.get("critical_fact") or {}).get("category")) or "")
        gt_b = (info.get("offline_audit") or {}).get("gt_b")
        kind = str(info.get("kind") or "")
        pred = _pred(state)
        diag_ok = _match_value(pred, gt_b, category)
        c5_state = c5.get((sample_id, repeat_id))
        c5_ok = _match_value(_pred(c5_state), gt_b, category) if c5_state else False
        c2b_state = c2b.get((sample_id, repeat_id))
        c2b_ok = _match_value(_pred(c2b_state), gt_b, category) if c2b_state else False
        follow["n"] += 1
        if diag_ok:
            follow["diag_match_b"] += 1
        if c5_ok:
            follow["c5_match_b"] += 1
        if c2b_ok:
            follow["c2b_match_b"] += 1
        if diag_ok and c5_ok:
            follow["both"] += 1
        elif diag_ok:
            follow["diag_only"] += 1
        elif c5_ok:
            follow["c5_only"] += 1
        else:
            follow["neither"] += 1
        if diag_condition == "TB" and c2b:
            if diag_ok and not c2b_ok:
                follow["tb_yes_c2b_no"] += 1
            elif not diag_ok and not c2b_ok:
                follow["tb_no_c2b_no"] += 1
            elif diag_ok and c2b_ok:
                follow["tb_yes_c2b_yes"] += 1
            else:
                follow["tb_no_c2b_yes"] += 1
            if diag_ok and c2b_ok and not c5_ok:
                follow["tb_yes_c2b_yes_c5_no"] += 1
        bucket = by_kind.setdefault(
            kind, {"n": 0, "diag_match_b": 0, "c5_match_b": 0, "c2b_match_b": 0}
        )
        bucket["n"] += 1
        bucket["diag_match_b"] += int(diag_ok)
        bucket["c5_match_b"] += int(c5_ok)
        bucket["c2b_match_b"] += int(c2b_ok)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "repeat_id": repeat_id,
            "kind": kind,
            "edit_layer": KIND_LAYER.get(kind),
            "gt_b": gt_b,
            "diag_pred": pred,
            "diag_match_b": diag_ok,
            "c5_match_b": c5_ok,
            "status": state.get("status"),
        }
        if c2b:
            row["c2b_match_b"] = c2b_ok
        rows.append(row)

    def _rate(num: int, den: int) -> float | None:
        return num / den if den else None

    n = follow["n"]
    interpretation = _interpret(diag_condition, follow)
    payload: dict[str, Any] = {
        "diag_condition": diag_condition,
        "n": n,
        "follow": {
            **follow,
            "diag_match_b_rate": _rate(follow["diag_match_b"], n),
            "c5_match_b_rate": _rate(follow["c5_match_b"], n),
            "c2b_match_b_rate": _rate(follow["c2b_match_b"], n) if c2b else None,
        },
        "by_kind": by_kind,
        "cross_tab": {
            "diag_yes_c5_no": follow["diag_only"],
            "diag_no_c5_no": follow["neither"],
            "diag_yes_c5_yes": follow["both"],
            "diag_no_c5_yes": follow["c5_only"],
        },
        "interpretation": interpretation,
        "note": "诊断性对照，不得并入 V6 正式 H1/H3 主结论。",
        "rows": rows,
    }
    if diag_condition == "TB" and c2b:
        payload["cross_tab_tb_c2b"] = {
            "tb_yes_c2b_no": follow["tb_yes_c2b_no"],
            "tb_no_c2b_no": follow["tb_no_c2b_no"],
            "tb_yes_c2b_yes": follow["tb_yes_c2b_yes"],
            "tb_no_c2b_yes": follow["tb_no_c2b_yes"],
            "tb_yes_c2b_yes_c5_no": follow["tb_yes_c2b_yes_c5_no"],
        }
    return payload


def _interpret(diag_condition: str, follow: dict[str, int]) -> str:
    only = follow["diag_only"]
    neither = follow["neither"]
    both = follow["both"]
    n = follow["n"] or 1
    if diag_condition == "C2B":
        if only / n >= 0.5:
            return "P_B-only 跟随而 I+P_B 不跟随：图像或视觉先验压过点云证据。"
        if neither / n >= 0.5:
            return "两者都不跟随：默认生成策略或 CAD 动作映射问题。"
        if both / n >= 0.5:
            return "两者都跟随：原 C5 阴性可能是小样本波动。"
        return "跟随随 feature 分化：信息可执行性具有 feature 依赖。"
    tb_only = follow.get("tb_yes_c2b_no") or 0
    tb_neither = follow.get("tb_no_c2b_no") or 0
    tb_both = follow.get("tb_yes_c2b_yes") or 0
    conflict = follow.get("tb_yes_c2b_yes_c5_no") or 0
    parts: list[str] = []
    if tb_only:
        parts.append(
            f"{tb_only}/{n} 条 T_B 跟随而 C2B 不跟随：能执行该事实，但 PointEvidence 权重更低。"
        )
    if tb_neither:
        parts.append(
            f"{tb_neither}/{n} 条两者都不跟随：不是点云 JSON 格式问题，而是 Plan 合成或默认策略问题。"
        )
    if conflict:
        parts.append(
            f"{conflict}/{n} 条 T_B 与 C2B 都跟随、C5 失败：跨模态冲突仲裁。"
        )
    if tb_both and not conflict:
        parts.append(f"{tb_both}/{n} 条 T_B 与 C2B 都跟随。")
    return " ".join(parts) if parts else "T_B 跟随随 feature 分化。"


def write_diag_analysis(
    diag_state_dir: Path,
    probe_state_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    diag_condition: str,
    c2b_state_dir: Path | None = None,
) -> dict[str, Any]:
    summary = analyze_diag_follow(
        diag_state_dir,
        probe_state_dir,
        manifest_path,
        diag_condition=diag_condition,
        c2b_state_dir=c2b_state_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / f"v6b_diag_{diag_condition.lower()}_descriptive.json", summary)
    return summary
