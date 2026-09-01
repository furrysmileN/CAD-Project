"""Offline C3→C5 Plan-diff for V6b. No API. Does not rewrite probe live state."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import atomic_write_json, sha256_json
from .v6_feature_scorer import DEFAULT_TOLERANCE, _ops, _select_operation, score_critical_fact
from .v6_manifest import read_manifest
from .v6b_probe_analysis import _match_value

KIND_LAYER = {
    "pocket_depth": "L1",
    "blind_depth": "L1",
    "through_vs_blind": "L2",
    "hidden_presence": "L3",
}
CATEGORIES = (
    "critical_changed_to_B",
    "critical_present_wrong_params",
    "unrelated_only",
    "missing_required_op",
    "mode_not_updated",
    "identical",
)


def _canonical_op(operation: dict[str, Any]) -> str:
    return json.dumps(operation, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _without_critical(ops: list[dict[str, Any]], critical: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not critical:
        return list(ops)
    out: list[dict[str, Any]] = []
    skipped = False
    crit_id = critical.get("id")
    for operation in ops:
        if not skipped:
            if crit_id and operation.get("id") == crit_id:
                skipped = True
                continue
            if operation == critical:
                skipped = True
                continue
        out.append(operation)
    return out


def _key_op(operation: dict[str, Any], index: int) -> tuple[Any, ...]:
    oid = operation.get("id")
    if oid:
        return ("id", str(oid))
    return ("sig", operation.get("op"), operation.get("combine"), operation.get("workplane"), index)


def unrelated_plan_edit_count(plan_a: dict[str, Any] | None, plan_b: dict[str, Any] | None, latent: dict[str, Any]) -> int:
    ops_a = [op for op in _ops(plan_a) if isinstance(op, dict)]
    ops_b = [op for op in _ops(plan_b) if isinstance(op, dict)]
    crit_a = _select_operation(plan_a, latent)
    crit_b = _select_operation(plan_b, latent)
    rest_a = _without_critical(ops_a, crit_a)
    rest_b = _without_critical(ops_b, crit_b)
    map_a = {_key_op(op, i): op for i, op in enumerate(rest_a)}
    map_b = {_key_op(op, i): op for i, op in enumerate(rest_b)}
    keys_a = Counter(map_a.keys())
    keys_b = Counter(map_b.keys())
    edits = 0
    for key in set(keys_a) | set(keys_b):
        if keys_a[key] != keys_b[key]:
            edits += abs(keys_a[key] - keys_b[key])
            continue
        if _canonical_op(map_a[key]) != _canonical_op(map_b[key]):
            edits += 1
    return int(edits)


def critical_plan_edit(plan_a: dict[str, Any] | None, plan_b: dict[str, Any] | None, latent: dict[str, Any]) -> bool:
    category = str((latent.get("critical_fact") or {}).get("category") or "")
    pred_a = score_critical_fact(plan_a, latent).get("pred_value")
    pred_b = score_critical_fact(plan_b, latent).get("pred_value")
    if pred_a is None and pred_b is None:
        return False
    if pred_a is None or pred_b is None:
        return True
    if category in {"through_vs_blind", "hidden_presence"}:
        return pred_a != pred_b
    try:
        return abs(float(pred_a) - float(pred_b)) > float(DEFAULT_TOLERANCE.get(category, 0.04))
    except (TypeError, ValueError):
        return pred_a != pred_b


def classify_c3_c5(
    plan_c3: dict[str, Any] | None,
    plan_c5: dict[str, Any] | None,
    latent: dict[str, Any],
    *,
    kind: str,
    gt_b: Any,
) -> dict[str, Any]:
    category = str((latent.get("critical_fact") or {}).get("category") or "")
    score_c3 = score_critical_fact(plan_c3, latent)
    score_c5 = score_critical_fact(plan_c5, latent)
    pred_c3 = score_c3.get("pred_value")
    pred_c5 = score_c5.get("pred_value")
    match_b = _match_value(pred_c5, gt_b, category)
    crit_edit = critical_plan_edit(plan_c3, plan_c5, latent)
    unrelated = unrelated_plan_edit_count(plan_c3, plan_c5, latent)
    hash_c3 = sha256_json(plan_c3) if isinstance(plan_c3, dict) else None
    hash_c5 = sha256_json(plan_c5) if isinstance(plan_c5, dict) else None
    same_plan = bool(hash_c3 and hash_c3 == hash_c5)
    if same_plan:
        label = "identical"
    elif kind == "hidden_presence" and bool(gt_b) is True and not pred_c5:
        label = "missing_required_op"
    elif kind == "through_vs_blind" and pred_c5 == "through":
        label = "mode_not_updated"
    elif crit_edit and match_b:
        label = "critical_changed_to_B"
    elif not crit_edit:
        label = "unrelated_only"
    else:
        label = "critical_present_wrong_params"
    return {
        "category_label": label,
        "critical_plan_edit": crit_edit,
        "unrelated_plan_edit_count": unrelated,
        "pred_c3": pred_c3,
        "pred_c5": pred_c5,
        "match_b": match_b,
        "same_plan": same_plan,
        "kind": kind,
        "edit_layer": KIND_LAYER.get(kind),
        "gt_b": gt_b,
        "c3_critical_op_id": (_select_operation(plan_c3, latent) or {}).get("id"),
        "c5_critical_op_id": (_select_operation(plan_c5, latent) or {}).get("id"),
    }


def _load_state_plans(state_dir: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted(state_dir.glob("*/*/r*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(state.get("sample_id") or "")
        condition = str(state.get("condition") or "")
        repeat_id = int(state.get("repeat_id") or 0)
        first = state.get("first_attempt") or {}
        out[(sample_id, condition, repeat_id)] = {
            "plan": first.get("plan"),
            "plan_sha256": first.get("plan_sha256"),
            "status": state.get("status"),
            "pred_value": (first.get("features") or {}).get("pred_value"),
        }
    return out


def analyze_plan_diff(state_dir: Path, manifest_path: Path, latent_dir: Path) -> dict[str, Any]:
    meta = {row["sample_id"]: row for row in read_manifest(manifest_path) if row.get("eligible")}
    states = _load_state_plans(state_dir)
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    by_kind: dict[str, dict[str, int]] = {}
    by_layer: dict[str, dict[str, int]] = {}
    for sample_id, info in sorted(meta.items()):
        latent_path = latent_dir / f"{sample_id}.json"
        latent = json.loads(latent_path.read_text(encoding="utf-8")) if latent_path.is_file() else {"critical_fact": info.get("critical_fact") or {}}
        kind = str(info.get("kind") or "")
        gt_b = (info.get("offline_audit") or {}).get("gt_b")
        for repeat_id in (1, 2):
            c3 = states.get((sample_id, "C3", repeat_id))
            c5 = states.get((sample_id, "C5", repeat_id))
            if not c3 or not c5:
                continue
            classified = classify_c3_c5(c3.get("plan"), c5.get("plan"), latent, kind=kind, gt_b=gt_b)
            record = {
                "sample_id": sample_id,
                "pair_id": info.get("pair_id"),
                "repeat_id": repeat_id,
                **classified,
            }
            rows.append(record)
            counts[classified["category_label"]] += 1
            bucket = by_kind.setdefault(kind, defaultdict(int))
            bucket[classified["category_label"]] += 1
            bucket["n"] += 1
            layer = KIND_LAYER.get(kind) or "unknown"
            layer_bucket = by_layer.setdefault(layer, defaultdict(int))
            layer_bucket[classified["category_label"]] += 1
            layer_bucket["n"] += 1
            layer_bucket["critical_plan_edit"] += int(classified["critical_plan_edit"])
            layer_bucket["unrelated_sum"] += int(classified["unrelated_plan_edit_count"])
    n = len(rows)
    summary = {
        "n": n,
        "primary_endpoint": "first_attempt",
        "note": "离线 Plan-diff。不得当作 confirm 显著性结论。",
        "counts": {key: int(counts.get(key, 0)) for key in CATEGORIES},
        "rates": {key: (counts[key] / n if n else None) for key in CATEGORIES},
        "by_kind": {kind: dict(vals) for kind, vals in by_kind.items()},
        "by_layer": {
            layer: {
                **{k: int(v) for k, v in vals.items() if k != "unrelated_sum"},
                "unrelated_plan_edit_mean": (vals["unrelated_sum"] / vals["n"]) if vals.get("n") else None,
            }
            for layer, vals in by_layer.items()
        },
        "rows": rows,
        "verdict": _verdict(counts, n),
    }
    return summary


def _verdict(counts: dict[str, int], n: int) -> str:
    if not n:
        return "无可比对。"
    unrelated = counts.get("unrelated_only", 0)
    missing = counts.get("missing_required_op", 0)
    mode = counts.get("mode_not_updated", 0)
    changed = counts.get("critical_changed_to_B", 0)
    if unrelated + missing + mode >= changed:
        return (
            "C3/C5 Plan hash 常不同，但关键字段经常不变："
            "模型改了无关操作，或未插入/未改模式。读到了上下文，却没有把反事实绑到对应 CAD 动作。"
        )
    return "关键操作变化占多数；仍只做描述。"


def write_plan_diff(state_dir: Path, manifest_path: Path, latent_dir: Path, output_dir: Path) -> dict[str, Any]:
    summary = analyze_plan_diff(state_dir, manifest_path, latent_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "v6b_plan_diff.json", summary)
    return summary
