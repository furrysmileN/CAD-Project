"""Descriptive V6b probe analysis. Critical-feature is primary. No significance claims."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import atomic_write_json
from .v6_feature_scorer import DEFAULT_TOLERANCE
from .v6_manifest import read_manifest

CONDITIONS = ("C0", "C1", "C2", "C3", "C4", "C5")


def _match_value(pred: Any, expected: Any, category: str) -> bool:
    if pred is None or expected is None:
        return False
    if category == "through_vs_blind":
        return str(pred) == str(expected)
    if category == "hidden_presence":
        return bool(pred) == bool(expected)
    try:
        return abs(float(pred) - float(expected)) <= float(DEFAULT_TOLERANCE.get(category, 0.04))
    except (TypeError, ValueError):
        return False


def load_probe_rows(state_dir: Path, manifest_path: Path) -> list[dict[str, Any]]:
    meta = {row["sample_id"]: row for row in read_manifest(manifest_path) if row.get("eligible")}
    rows = []
    for path in sorted(state_dir.glob("*/*/r*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(state.get("sample_id") or "")
        info = meta.get(sample_id) or {}
        first = state.get("first_attempt") or {}
        rows.append(
            {
                "sample_id": sample_id,
                "pair_id": info.get("pair_id"),
                "kind": info.get("kind"),
                "condition": state.get("condition"),
                "repeat_id": state.get("repeat_id"),
                "status": state.get("status"),
                "features": first.get("features") or {},
                "geometry": first.get("geometry") or {},
                "plan_sha256": first.get("plan_sha256"),
                "parse_ok": first.get("parse_ok"),
                "schema_ok": first.get("schema_ok"),
                "gt_a": (info.get("offline_audit") or {}).get("gt_a"),
                "gt_b": (info.get("offline_audit") or {}).get("gt_b"),
                "category": ((info.get("critical_fact") or {}).get("category")),
            }
        )
    return rows


def _pair_means(rows: list[dict[str, Any]], field: str) -> dict[tuple[str, str], float]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["sample_id"]), str(row["condition"]))
        if field == "exact":
            buckets[key].append(1.0 if (row.get("features") or {}).get("exact") else 0.0)
        elif field == "within":
            buckets[key].append(1.0 if (row.get("features") or {}).get("within_tolerance") else 0.0)
        elif field == "jq":
            buckets[key].append(float(((row.get("geometry") or {}).get("joint_quality")) or 0.0))
    return {key: sum(vals) / len(vals) for key, vals in buckets.items() if vals}


def _condition_table(means: dict[tuple[str, str], float], samples: list[str]) -> dict[str, Any]:
    out = {}
    for cond in CONDITIONS:
        vals = [means[(sid, cond)] for sid in samples if (sid, cond) in means]
        out[cond] = {
            "n": len(vals),
            "mean": (sum(vals) / len(vals) if vals else None),
        }
    return out


def analyze_v6b_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") in {"completed", "execution_failed", "schema_failed", "parse_failed"}]
    samples = sorted({str(row["sample_id"]) for row in completed})
    exact_means = _pair_means(completed, "exact")
    within_means = _pair_means(completed, "within")
    jq_means = _pair_means(completed, "jq")
    by_kind: dict[str, dict[str, Any]] = {}
    follow = {
        "C3_match_a": 0,
        "C5_match_a": 0,
        "C5_match_b": 0,
        "C5_follow_b": 0,
        "C5_follow_strict": 0,
        "c3_c5_pred_changed": 0,
        "C2_match_a": 0,
        "n": 0,
    }
    identical = {"c3_c4": 0, "c3_c5": 0, "c4_c5": 0, "c3_c4_c5": 0, "n": 0}
    by_repeat: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in completed:
        by_repeat[(str(row["sample_id"]), int(row.get("repeat_id") or 0))][str(row["condition"])] = row
    for (_sid, _rid), conds in by_repeat.items():
        if not {"C3", "C4", "C5"} <= set(conds):
            continue
        identical["n"] += 1
        h3, h4, h5 = conds["C3"].get("plan_sha256"), conds["C4"].get("plan_sha256"), conds["C5"].get("plan_sha256")
        if h3 and h3 == h4:
            identical["c3_c4"] += 1
        if h3 and h3 == h5:
            identical["c3_c5"] += 1
        if h4 and h4 == h5:
            identical["c4_c5"] += 1
        if h3 and h3 == h4 == h5:
            identical["c3_c4_c5"] += 1
        category = str(conds["C3"].get("category") or "")
        gt_a, gt_b = conds["C3"].get("gt_a"), conds["C3"].get("gt_b")
        pred_c2 = ((conds.get("C2") or {}).get("features") or {}).get("pred_value")
        pred_c3 = (conds["C3"].get("features") or {}).get("pred_value")
        pred_c5 = (conds["C5"].get("features") or {}).get("pred_value")
        follow["n"] += 1
        if _match_value(pred_c3, gt_a, category):
            follow["C3_match_a"] += 1
        if _match_value(pred_c5, gt_a, category):
            follow["C5_match_a"] += 1
        if _match_value(pred_c5, gt_b, category):
            follow["C5_match_b"] += 1
        if _match_value(pred_c5, gt_b, category) and not _match_value(pred_c5, gt_a, category):
            follow["C5_follow_b"] += 1
        changed = pred_c3 != pred_c5
        if changed:
            follow["c3_c5_pred_changed"] += 1
        strict = bool(changed and _match_value(pred_c5, gt_b, category))
        if strict:
            follow["C5_follow_strict"] += 1
        if conds.get("C2") and _match_value(pred_c2, gt_a, category):
            follow["C2_match_a"] += 1
        kind = str(conds["C3"].get("kind") or "unknown")
        bucket = by_kind.setdefault(
            kind, {"n": 0, "C3_exact": 0, "C5_follow_b": 0, "C5_follow_strict": 0, "c3_c5_changed": 0}
        )
        bucket["n"] += 1
        if (conds["C3"].get("features") or {}).get("exact"):
            bucket["C3_exact"] += 1
        if _match_value(pred_c5, gt_b, category) and not _match_value(pred_c5, gt_a, category):
            bucket["C5_follow_b"] += 1
        if changed:
            bucket["c3_c5_changed"] += 1
        if strict:
            bucket["C5_follow_strict"] += 1

    def _rate(num: int, den: int) -> float | None:
        return (num / den) if den else None

    n_id = identical["n"]
    n_f = follow["n"]
    deltas = []
    for sid in samples:
        c3 = exact_means.get((sid, "C3"))
        c5 = exact_means.get((sid, "C5"))
        c4 = exact_means.get((sid, "C4"))
        if c3 is None:
            continue
        deltas.append(
            {
                "sample_id": sid,
                "C3_minus_C4": None if c4 is None else c3 - c4,
                "C3_minus_C5": None if c5 is None else c3 - c5,
            }
        )
    identical_rate = _rate(identical["c3_c4_c5"], n_id) or 0.0
    follow_rate = _rate(follow["C5_follow_b"], n_f) or 0.0
    strict_rate = _rate(follow["C5_follow_strict"], n_f) or 0.0
    changed_rate = _rate(follow["c3_c5_pred_changed"], n_f) or 0.0
    c3_c5_same = _rate(identical["c3_c5"], n_id) or 0.0
    gates = {
        "c3_c4_c5_not_mostly_identical": identical_rate < 0.5,
        "c3_c5_plans_differ": c3_c5_same < 0.5,
        "c5_raw_match_b_ge_50": follow_rate >= 0.5,
        "c5_strict_follow_ge_50": strict_rate >= 0.5,
        "descriptive_only": True,
    }
    gates["pass_probe_go"] = bool(gates["c3_c4_c5_not_mostly_identical"] and gates["c5_strict_follow_ge_50"])
    if identical_rate >= 0.5:
        verdict = "C3/C4/C5 仍大面积相同 Plan：读到了但未写入 CAD。不要开 confirm。"
    elif strict_rate < 0.5:
        verdict = (
            "C5 与 B 吻合含默认值膨胀；以 C3→C5 预测值是否改变为准。"
            "整体严格跟随未过 50%。不要开全量 confirm。"
        )
    else:
        verdict = "探针方向性通过：C3→C5 关键预测随反事实证据变化。仍不得做显著性宣称；confirm 另议。"
    return {
        "n_rows": len(rows),
        "n_completed_rows": len(completed),
        "n_pairs": len(samples),
        "primary_endpoint": "first_attempt",
        "primary_metric": "critical_feature_exact",
        "exact_by_condition": _condition_table(exact_means, samples),
        "within_by_condition": _condition_table(within_means, samples),
        "joint_quality_by_condition": _condition_table(jq_means, samples),
        "pair_deltas": deltas,
        "plan_identity": {
            **identical,
            "c3_c4_rate": _rate(identical["c3_c4"], n_id),
            "c3_c5_rate": _rate(identical["c3_c5"], n_id),
            "c3_c4_c5_rate": identical_rate,
        },
        "value_follow": {
            **follow,
            "C3_match_a_rate": _rate(follow["C3_match_a"], n_f),
            "C5_match_a_rate": _rate(follow["C5_match_a"], n_f),
            "C5_match_b_rate": _rate(follow["C5_match_b"], n_f),
            "C5_follow_b_rate": follow_rate,
            "C5_follow_strict_rate": strict_rate,
            "c3_c5_pred_changed_rate": changed_rate,
            "C2_match_a_rate": _rate(follow["C2_match_a"], n_f),
        },
        "by_kind": by_kind,
        "gates": gates,
        "verdict": verdict,
        "note": "n=11×2，只做描述；不得当作 confirm 显著性结论。",
    }


def write_probe_analysis(state_dir: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = load_probe_rows(state_dir, manifest_path)
    summary = analyze_v6b_probe(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "v6b_probe_descriptive.json", summary)
    return summary
