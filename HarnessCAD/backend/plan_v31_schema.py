"""HarnessCAD Plan v3.1: v3 plus typed 3D path wires and sweep mode."""
from __future__ import annotations

import copy
import math
from typing import Any

from .plan_v2_schema import _issue, _vec
from .plan_v3_schema import SCHEMA_VERSION as PLAN_V3_SCHEMA_VERSION
from .plan_v3_schema import validate_plan_v3

SCHEMA_VERSION = "harnesscad.plan.v3.1"
SWEEP_MODES = {"fixed", "frenet"}


def _point(value: Any) -> list[float] | None:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
    ):
        return None
    return [float(item) for item in value]


def _same(left: list[float], right: list[float], tolerance: float = 1e-9) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(left, right))


def _validate_path_wire(
    value: Any,
    path: str,
    issues: list[dict[str, str]],
) -> list[list[float]]:
    if not isinstance(value, list) or len(value) < 2:
        issues.append(
            _issue(
                "invalid_path_wire",
                path,
                "Expected move followed by at least one line or three_point_arc.",
            )
        )
        return []
    first = value[0]
    if not isinstance(first, dict) or set(first) != {"kind", "to"} or first.get("kind") != "move":
        issues.append(
            _issue(
                "invalid_path_wire_start",
                f"{path}[0]",
                "First segment must be {kind: move, to: vec3}.",
            )
        )
        return []
    start = _point(first.get("to"))
    if start is None:
        _vec(first.get("to"), f"{path}[0].to", issues, 3)
        return []
    points = [start]
    cursor = start
    for index, segment in enumerate(value[1:], start=1):
        seg_path = f"{path}[{index}]"
        if not isinstance(segment, dict):
            issues.append(_issue("invalid_path_segment", seg_path, "Expected an object."))
            continue
        kind = segment.get("kind")
        expected = {"kind", "to"} if kind == "line" else {"kind", "through", "to"}
        if kind not in {"line", "three_point_arc"}:
            issues.append(
                _issue(
                    "unsupported_path_segment",
                    f"{seg_path}.kind",
                    "Use line or three_point_arc.",
                )
            )
            continue
        for key in sorted(set(segment) - expected):
            issues.append(
                _issue("extra_field", f"{seg_path}.{key}", "Field is not allowed.")
            )
        for key in sorted(expected - set(segment)):
            issues.append(
                _issue("missing_field", f"{seg_path}.{key}", "Required field is missing.")
            )
        destination = _point(segment.get("to"))
        if destination is None:
            _vec(segment.get("to"), f"{seg_path}.to", issues, 3)
            continue
        if _same(cursor, destination):
            issues.append(
                _issue(
                    "degenerate_path_segment",
                    seg_path,
                    "Path segment endpoints must differ.",
                )
            )
        if kind == "three_point_arc":
            through = _point(segment.get("through"))
            if through is None:
                _vec(segment.get("through"), f"{seg_path}.through", issues, 3)
            else:
                left = [through[i] - cursor[i] for i in range(3)]
                right = [destination[i] - cursor[i] for i in range(3)]
                cross = [
                    left[1] * right[2] - left[2] * right[1],
                    left[2] * right[0] - left[0] * right[2],
                    left[0] * right[1] - left[1] * right[0],
                ]
                if math.sqrt(sum(item * item for item in cross)) <= 1e-9:
                    issues.append(
                        _issue(
                            "degenerate_path_arc",
                            seg_path,
                            "Arc start, through and end must not be collinear.",
                        )
                    )
        cursor = destination
        points.append(destination)
    return points


def validate_plan_v31(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate v3.1 by lowering path-wire syntax to the already strict v3 core."""
    issues: list[dict[str, str]] = []
    if not isinstance(plan, dict):
        return {
            "valid": False,
            "issues": [_issue("not_object", "$", "Plan must be a JSON object.")],
        }
    normalized = copy.deepcopy(plan)
    normalized["schema_version"] = PLAN_V3_SCHEMA_VERSION
    operations = normalized.get("operations")
    raw_operations = plan.get("operations")
    if isinstance(operations, list) and isinstance(raw_operations, list):
        for index, (operation, raw) in enumerate(zip(operations, raw_operations)):
            if not isinstance(operation, dict) or operation.get("op") != "sweep_profile":
                continue
            raw = raw if isinstance(raw, dict) else {}
            has_path = raw.get("path") is not None
            has_path_wire = raw.get("path_wire") is not None
            has_helix = raw.get("helix") is not None
            if sum((has_path, has_path_wire, has_helix)) != 1:
                issues.append(
                    _issue(
                        "path_xor_helix",
                        f"$.operations[{index}]",
                        "Specify exactly one of path, path_wire or helix.",
                    )
                )
            if has_path_wire:
                points = _validate_path_wire(
                    raw.get("path_wire"),
                    f"$.operations[{index}].path_wire",
                    issues,
                )
                operation.pop("path_wire", None)
                operation["path"] = points if len(points) >= 2 else [[0, 0, 0], [1, 0, 0]]
            mode = raw.get("sweep_mode", "fixed")
            if mode not in SWEEP_MODES:
                issues.append(
                    _issue(
                        "invalid_sweep_mode",
                        f"$.operations[{index}].sweep_mode",
                        f"Use one of {sorted(SWEEP_MODES)}.",
                    )
                )
            operation.pop("sweep_mode", None)
    core = validate_plan_v3(normalized)
    for issue in core["issues"]:
        if (
            issue.get("code") == "extra_field"
            and (
                str(issue.get("path", "")).endswith(".path_wire")
                or str(issue.get("path", "")).endswith(".sweep_mode")
            )
        ):
            continue
        issues.append(issue)
    return {"valid": not issues, "issues": issues}
