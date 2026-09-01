"""Descriptive V7 transfer analysis. Not a V6 H1/H3 result."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import atomic_write_json
from .v6_manifest import read_manifest
from .v6b_probe_analysis import _match_value
from .v7_pair_generator import KIND_LAYER

CONDITIONS = ("C3", "C5", "C2B", "TB")


def _pred(state: dict[str, Any]) -> Any:
    return ((state.get("first_attempt") or {}).get("features") or {}).get("pred_value")


def load_states(state_dir: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted(state_dir.glob("*/*/r*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        key = (str(state.get("sample_id") or ""), str(state.get("condition") or ""), int(state.get("repeat_id") or 0))
        out[key] = state
    return out


def analyze_v7(state_dir: Path, manifest_path: Path) -> dict[str, Any]:
    meta = {row["sample_id"]: row for row in read_manifest(manifest_path) if row.get("eligible")}
    states = load_states(state_dir)
    by_cond: dict[str, dict[str, int]] = {c: {"n": 0, "match_b": 0} for c in CONDITIONS}
    by_layer: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: {c: {"n": 0, "match_b": 0} for c in CONDITIONS})
    by_host: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: {c: {"n": 0, "match_b": 0} for c in CONDITIONS})
    by_kind: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: {c: {"n": 0, "match_b": 0} for c in CONDITIONS})
    rows: list[dict[str, Any]] = []
    for (sample_id, condition, repeat_id), state in sorted(states.items()):
        if condition not in CONDITIONS:
            continue
        info = meta.get(sample_id) or {}
        category = str(((info.get("critical_fact") or {}).get("category")) or "")
        gt_b = (info.get("offline_audit") or {}).get("gt_b")
        kind = str(info.get("kind") or "")
        host = str(info.get("host") or info.get("family") or "")
        layer = str(info.get("edit_layer") or KIND_LAYER.get(kind) or "")
        pred = _pred(state)
        ok = _match_value(pred, gt_b, category)
        by_cond[condition]["n"] += 1
        by_cond[condition]["match_b"] += int(ok)
        for bucket in (by_layer[layer][condition], by_host[host][condition], by_kind[kind][condition]):
            bucket["n"] += 1
            bucket["match_b"] += int(ok)
        rows.append(
            {
                "sample_id": sample_id,
                "repeat_id": repeat_id,
                "condition": condition,
                "host": host,
                "kind": kind,
                "edit_layer": layer,
                "gt_b": gt_b,
                "pred": pred,
                "match_b": ok,
                "status": state.get("status"),
            }
        )

    def _rate(item: dict[str, int]) -> float | None:
        return item["match_b"] / item["n"] if item["n"] else None

    rates = {c: _rate(by_cond[c]) for c in CONDITIONS}
    layer_rates = {
        layer: {c: _rate(by_layer[layer][c]) for c in CONDITIONS if by_layer[layer][c]["n"]}
        for layer in ("L1", "L2", "L3")
        if layer in by_layer
    }
    l1 = (layer_rates.get("L1") or {}).get("C5")
    l2 = (layer_rates.get("L2") or {}).get("C5")
    l3 = (layer_rates.get("L3") or {}).get("C5")
    t1_ok = None
    if l1 is not None and l2 is not None and l3 is not None:
        t1_ok = l1 >= l3 >= l2
    t2_ok = None
    if rates["C2B"] is not None and rates["C5"] is not None:
        t2_ok = rates["C2B"] >= rates["C5"]
    t3_ok = None
    if rates["TB"] is not None and rates["C2B"] is not None:
        t3_ok = rates["TB"] >= rates["C2B"]
    transfer = bool(t1_ok and t2_ok and t3_ok)
    if rates["TB"] is not None and rates["TB"] < 0.5:
        verdict = "T_B 跟随过低：先查新宿主上事实是否仍可执行，不解释为规律消失。"
    elif transfer:
        verdict = "描述性转移成功：新宿主上仍见 L1≥L3≥L2，且 C2B≥C5、T_B≥C2B。非正式显著。"
    else:
        verdict = "描述性转移失败：规律可能绑在 V6b 三套模板上，不得当普适机制写。不回退加 n。"
    return {
        "n_state_rows": len(rows),
        "by_condition": {c: {**by_cond[c], "rate": rates[c]} for c in CONDITIONS},
        "by_layer": {k: v for k, v in by_layer.items()},
        "by_host": {k: v for k, v in by_host.items()},
        "by_kind": {k: v for k, v in by_kind.items()},
        "hypotheses": {"T1_edit_order": t1_ok, "T2_c2b_ge_c5": t2_ok, "T3_tb_ge_c2b": t3_ok, "transfer": transfer},
        "verdict": verdict,
        "note": "不得并入 V6 正式 H1/H3。",
        "rows": rows,
    }


def write_v7_analysis(state_dir: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    summary = analyze_v7(state_dir, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "v7_transfer_descriptive.json", summary)
    return summary
