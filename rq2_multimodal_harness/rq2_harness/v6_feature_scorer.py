"""Feature-level scoring. Latent spec is read only in this scoring stage."""
from __future__ import annotations

from typing import Any

import numpy as np

FEATURE_VERSION = "rq2.v6.feature.v3"
# Plan JSON is typically 4 decimal places; 1e-6 exact would fail GT vs rounded centers.
EXACT_NUMERIC_TOL = 5e-4
DEFAULT_TOLERANCE = {
    "depth": 0.04,
    "radius_or_width": 0.04,
    "offset_or_spacing": 0.05,
    "axis_or_symmetry": 0.15,
}
THROUGH_BODY_MARGIN = 0.02
THROUGH_ABS_MIN = 0.40

_REVOLVE_AXIS_3D = {
    "XY": [0.0, 0.0, 1.0],
    "XZ": [0.0, 0.0, 1.0],
    "YZ": [1.0, 0.0, 0.0],
}


def _ops(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(plan, dict):
        return []
    ops = plan.get("operations")
    return ops if isinstance(ops, list) else []


def _hole_ops(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [op for op in _ops(plan) if isinstance(op, dict) and op.get("op") == "hole"]


def _matched_depth(operation: dict[str, Any] | None) -> Any:
    if not operation:
        return None
    if "depth" in operation:
        return operation.get("depth")
    size = operation.get("size")
    if operation.get("op") == "box" and isinstance(size, list) and len(size) == 3:
        return size[2]
    return None


def _revolve_max_radius(operation: dict[str, Any] | None) -> Any:
    if not operation or operation.get("op") != "revolve_profile":
        return None
    radii = []
    for point in operation.get("profile") or []:
        if isinstance(point, list) and point:
            try:
                radii.append(abs(float(point[0])))
            except (TypeError, ValueError):
                continue
    return max(radii) if radii else None


def _world_axis(operation: dict[str, Any] | None) -> Any:
    if not operation:
        return None
    if operation.get("op") == "cylinder" and isinstance(operation.get("axis"), list) and len(operation["axis"]) == 3:
        return operation.get("axis")
    if operation.get("op") == "revolve_profile":
        workplane = str(operation.get("workplane") or "XZ")
        return list(_REVOLVE_AXIS_3D.get(workplane, [0.0, 0.0, 1.0]))
    return None


def _numeric_close(pred: Any, gt: Any, tol: float) -> bool:
    try:
        return abs(float(pred) - float(gt)) <= tol
    except (TypeError, ValueError):
        return False


def _axis_close(pred: Any, gt: Any, tol: float) -> bool:
    try:
        a = np.asarray(pred, dtype=np.float64)
        b = np.asarray(gt, dtype=np.float64)
        a = a / max(np.linalg.norm(a), 1e-9)
        b = b / max(np.linalg.norm(b), 1e-9)
        return float(abs(a @ b)) >= 1.0 - tol
    except Exception:
        return False


def _body_height(plan: dict[str, Any] | None) -> float | None:
    for operation in _ops(plan):
        if operation.get("op") != "box" or operation.get("combine") != "new":
            continue
        size = operation.get("size")
        if isinstance(size, list) and len(size) == 3:
            try:
                return float(size[2])
            except (TypeError, ValueError):
                return None
    return None


def _is_through_like(operation: dict[str, Any] | None, body_height: float | None) -> bool:
    depth = _matched_depth(operation)
    try:
        value = float(depth)
    except (TypeError, ValueError):
        return False
    if body_height is not None and value >= float(body_height) - THROUGH_BODY_MARGIN:
        return True
    return value >= THROUGH_ABS_MIN


def _name_score(operation: dict[str, Any], critical: dict[str, Any]) -> int:
    oid = str(operation.get("id") or "").lower()
    op_id = str(critical.get("operation_id") or "").lower()
    fact_id = str(critical.get("fact_id") or "").lower()
    if op_id and oid == op_id:
        return 10
    score = 0
    if "pocket" in fact_id and "pocket" in oid:
        score += 5
    if "blind" in fact_id and "blind" in oid:
        score += 5
    if "back" in fact_id and "back" in oid:
        score += 5
    if "hidden" in fact_id and ("back" in oid or "hidden" in oid):
        score += 5
    return score


def _select_operation(plan: dict[str, Any] | None, latent: dict[str, Any]) -> dict[str, Any] | None:
    """Pick the operation that carries the critical fact. Never use a through hole as a blind/pocket depth."""
    critical = latent.get("critical_fact") or {}
    category = str(critical.get("category") or "")
    fact_id = str(critical.get("fact_id") or "")
    ops = _ops(plan)
    if not ops:
        return None
    named = [op for op in ops if _name_score(op, critical) > 0]
    named.sort(key=lambda op: -_name_score(op, critical))
    body_h = _body_height(plan)
    holes = _hole_ops(plan)
    if category == "depth":
        if named:
            return named[0]
        if "pocket" in fact_id:
            cuts = [
                op
                for op in ops
                if op.get("combine") == "cut"
                and op.get("op") in {"box", "slot", "hole"}
                and not _is_through_like(op, body_h)
            ]
            if cuts:
                return max(cuts, key=lambda op: float(_matched_depth(op) or 0.0))
            return None
        shallow = [op for op in holes if not _is_through_like(op, body_h)]
        if shallow:
            return max(shallow, key=lambda op: float(op.get("depth") or 0.0))
        return None
    if category == "through_vs_blind":
        if named:
            return named[0]
        if len(holes) == 1:
            return holes[0]
        shallow = [op for op in holes if not _is_through_like(op, body_h)]
        if len(shallow) == 1:
            return shallow[0]
        return holes[0] if holes else None
    if category == "hidden_presence":
        if named:
            return named[0]
        xz = [op for op in holes if op.get("workplane") == "XZ"]
        return xz[0] if xz else None
    if named:
        return named[0]
    return None


def score_critical_fact(pred_plan: dict[str, Any] | None, latent: dict[str, Any]) -> dict[str, Any]:
    critical = latent.get("critical_fact") or {}
    category = str(critical.get("category") or "")
    gt = critical.get("value")
    matched = _select_operation(pred_plan, latent)
    holes = _hole_ops(pred_plan)
    exact = False
    within = False
    pred_value: Any = None
    if category == "depth":
        pred_value = _matched_depth(matched)
        exact = _numeric_close(pred_value, gt, EXACT_NUMERIC_TOL)
        within = _numeric_close(pred_value, gt, DEFAULT_TOLERANCE["depth"])
    elif category == "radius_or_width":
        if matched and "diameter" in matched:
            pred_value = matched.get("diameter")
        elif matched and "radius" in matched:
            pred_value = matched.get("radius")
        else:
            revolve = next((op for op in _ops(pred_plan) if op.get("op") == "revolve_profile"), None)
            pred_value = _revolve_max_radius(revolve)
            if pred_value is None and holes:
                pred_value = holes[0].get("diameter")
        exact = _numeric_close(pred_value, gt, EXACT_NUMERIC_TOL)
        within = _numeric_close(pred_value, gt, DEFAULT_TOLERANCE["radius_or_width"])
    elif category == "offset_or_spacing":
        holes = _hole_ops(pred_plan)
        bolts = [op for op in holes if str(op.get("id") or "").startswith("bolt_")]
        if len(bolts) >= 2:
            c0 = bolts[0].get("center") or [0, 0, 0]
            c1 = bolts[1].get("center") or [0, 0, 0]
            pred_value = abs(float(c0[0]) - float(c1[0])) or abs(float(c0[1]) - float(c1[1]))
        else:
            spacings = [op.get("spacing") for op in _ops(pred_plan) if op.get("op") == "linear_pattern"]
            pred_value = spacings[0] if spacings else None
        exact = _numeric_close(pred_value, gt, EXACT_NUMERIC_TOL)
        within = _numeric_close(pred_value, gt, DEFAULT_TOLERANCE["offset_or_spacing"])
    elif category == "through_vs_blind":
        depth = matched.get("depth") if matched else None
        if depth is None:
            pred_value = None
        else:
            pred_value = "through" if float(depth) >= 0.20 else "blind"
        exact = pred_value == gt
        within = exact
    elif category == "hidden_presence":
        pred_value = matched is not None
        exact = bool(pred_value) == bool(gt)
        within = exact
    elif category == "axis_or_symmetry":
        body = next((op for op in _ops(pred_plan) if op.get("op") in {"cylinder", "revolve_profile"}), None)
        pred_value = _world_axis(body)
        exact = _axis_close(pred_value, gt, 1e-3)
        within = _axis_close(pred_value, gt, DEFAULT_TOLERANCE["axis_or_symmetry"])
    return {
        "version": FEATURE_VERSION,
        "fact_id": critical.get("fact_id"),
        "category": category,
        "gt_value": gt,
        "pred_value": pred_value,
        "exact": exact,
        "within_tolerance": within,
        "operation_count": len(_ops(pred_plan)),
        "hole_count": len(holes),
        "add_count": sum(1 for op in _ops(pred_plan) if op.get("combine") == "add"),
        "cut_count": sum(1 for op in _ops(pred_plan) if op.get("combine") == "cut"),
    }


def empty_feature_scores() -> dict[str, Any]:
    return {
        "version": FEATURE_VERSION,
        "exact": False,
        "within_tolerance": False,
        "operation_count": 0,
        "hole_count": 0,
        "add_count": 0,
        "cut_count": 0,
    }
