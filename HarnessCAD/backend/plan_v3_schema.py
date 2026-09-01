"""HarnessCAD Plan v3: v2 plus sweep/loft/helix and optional arc wires."""
from __future__ import annotations

from typing import Any

from .plan_v2_schema import (
    ALL_OPERATIONS,
    COMBINES,
    COMMON_FIELDS,
    ID_PATTERN,
    MODIFIER_OPERATIONS,
    OPERATION_FIELDS,
    OPTIONAL_FIELDS,
    SAMPLE_ID_PATTERN,
    SHAPE_OPERATIONS,
    WORKPLANES,
    _closed_polygon,
    _issue,
    _json_metadata,
    _number,
    _unit_vec3,
    _validate_combine,
    _validate_coordinate_system,
    _validate_shape_operation,
    _vec,
)

SCHEMA_VERSION = "harnesscad.plan.v3"
MAX_OPERATIONS = 128

SHAPE_OPERATIONS_V3 = SHAPE_OPERATIONS | {"sweep_profile", "loft_profiles"}
ALL_OPERATIONS_V3 = SHAPE_OPERATIONS_V3 | MODIFIER_OPERATIONS

OPERATION_FIELDS_V3 = {
    **OPERATION_FIELDS,
    "polygon_extrude": {"combine", "workplane", "points", "wire", "depth", "centered", "offset"},
    "revolve_profile": {"combine", "workplane", "profile", "wire", "axis", "angle", "offset"},
    "sweep_profile": {"combine", "workplane", "profile", "wire", "path", "helix", "offset"},
    "loft_profiles": {"combine", "workplane", "profiles"},
}
OPTIONAL_FIELDS_V3 = {
    **OPTIONAL_FIELDS,
    "polygon_extrude": {"points", "wire"},
    "revolve_profile": {"profile", "wire"},
    "sweep_profile": {"profile", "wire", "path", "helix"},
}


def _validate_wire(value: object, path: str, issues: list[dict[str, str]]) -> bool:
    if not isinstance(value, list) or len(value) < 3:
        issues.append(_issue("invalid_wire", path, "Expected at least 3 wire segments starting with move."))
        return False
    first = value[0]
    if not isinstance(first, dict) or first.get("kind") != "move" or not _vec(first.get("to"), f"{path}[0].to", issues, 2):
        issues.append(_issue("invalid_wire_start", f"{path}[0]", "First segment must be kind=move with a 2D to."))
        return False
    start = [float(item) for item in first["to"]]
    cursor = start
    ok = True
    for index, segment in enumerate(value[1:], start=1):
        seg_path = f"{path}[{index}]"
        if not isinstance(segment, dict):
            issues.append(_issue("invalid_wire_segment", seg_path, "Expected an object."))
            ok = False
            continue
        kind = segment.get("kind")
        allowed = {"kind", "to", "through"}
        for key in sorted(set(segment) - allowed):
            issues.append(_issue("extra_field", f"{seg_path}.{key}", "Field is not allowed for this wire segment."))
        if kind == "line":
            if not _vec(segment.get("to"), f"{seg_path}.to", issues, 2):
                ok = False
            else:
                cursor = [float(item) for item in segment["to"]]
        elif kind == "three_point_arc":
            if not _vec(segment.get("through"), f"{seg_path}.through", issues, 2):
                ok = False
            if not _vec(segment.get("to"), f"{seg_path}.to", issues, 2):
                ok = False
            elif _vec(segment.get("through"), f"{seg_path}.through", issues, 2):
                cursor = [float(item) for item in segment["to"]]
        else:
            issues.append(_issue("unsupported_wire_kind", f"{seg_path}.kind", "Use move, line, or three_point_arc."))
            ok = False
    if ok and any(abs(a - b) > 1e-9 for a, b in zip(cursor, start)):
        issues.append(_issue("wire_not_closed", path, "Wire must return to the move start point."))
        ok = False
    return ok


def _validate_profile_or_wire(
    operation: dict[str, Any],
    path: str,
    issues: list[dict[str, str]],
    *,
    points_key: str,
) -> None:
    has_points = operation.get(points_key) is not None
    has_wire = operation.get("wire") is not None
    if has_points == has_wire:
        issues.append(_issue("profile_xor_wire", path, f"Specify exactly one of {points_key} or wire."))
        return
    if has_points:
        _closed_polygon(operation.get(points_key), f"{path}.{points_key}", issues)
    else:
        _validate_wire(operation.get("wire"), f"{path}.wire", issues)


def _validate_helix(value: object, path: str, issues: list[dict[str, str]]) -> None:
    if not isinstance(value, dict):
        issues.append(_issue("invalid_helix", path, "Expected radius, pitch, turns, axis, center."))
        return
    required = {"radius", "pitch", "turns", "axis", "center"}
    for key in sorted(required - set(value)):
        issues.append(_issue("missing_field", f"{path}.{key}", "Required field is missing."))
    for key in sorted(set(value) - required):
        issues.append(_issue("extra_field", f"{path}.{key}", "Field is not allowed."))
    _number(value.get("radius"), f"{path}.radius", issues, 0.0, 4.0, inclusive_minimum=False)
    _number(value.get("pitch"), f"{path}.pitch", issues, 0.0, 4.0, inclusive_minimum=False)
    _number(value.get("turns"), f"{path}.turns", issues, 0.25, 32.0)
    _unit_vec3(value.get("axis"), f"{path}.axis", issues)
    _vec(value.get("center"), f"{path}.center", issues, 3)


def _validate_path(value: object, path: str, issues: list[dict[str, str]]) -> None:
    if not isinstance(value, list) or len(value) < 2:
        issues.append(_issue("invalid_path", path, "Expected at least two 3D points."))
        return
    for index, point in enumerate(value):
        _vec(point, f"{path}[{index}]", issues, 3)


def _validate_loft_profiles(value: object, path: str, issues: list[dict[str, str]]) -> None:
    if not isinstance(value, list) or len(value) < 2 or len(value) > 8:
        issues.append(_issue("invalid_loft_profiles", path, "Expected 2-8 loft profiles."))
        return
    for index, profile in enumerate(value):
        prof_path = f"{path}[{index}]"
        if not isinstance(profile, dict):
            issues.append(_issue("invalid_loft_profile", prof_path, "Expected an object."))
            continue
        allowed = {"points", "wire", "offset", "workplane"}
        for key in sorted(set(profile) - allowed):
            issues.append(_issue("extra_field", f"{prof_path}.{key}", "Field is not allowed."))
        workplane = profile.get("workplane")
        if workplane is not None and workplane not in WORKPLANES:
            issues.append(_issue("invalid_workplane", f"{prof_path}.workplane", "Use XY, XZ or YZ."))
        _vec(profile.get("offset") or [0.0, 0.0, 0.0], f"{prof_path}.offset", issues, 3)
        _validate_profile_or_wire(profile, prof_path, issues, points_key="points")


def _validate_shape_v3(
    operation: dict[str, Any],
    kind: str,
    path: str,
    index: int,
    shape_ids: set[str],
    issues: list[dict[str, str]],
) -> None:
    if kind in SHAPE_OPERATIONS and kind not in {"polygon_extrude", "revolve_profile"}:
        _validate_shape_operation(operation, kind, path, index, shape_ids, issues)
        return
    _validate_combine(operation, path, index, issues)
    if kind == "polygon_extrude":
        if operation.get("workplane") not in WORKPLANES:
            issues.append(_issue("invalid_workplane", f"{path}.workplane", "Use XY, XZ or YZ."))
        _validate_profile_or_wire(operation, path, issues, points_key="points")
        _number(operation.get("depth"), f"{path}.depth", issues, 0.0, 8.0, inclusive_minimum=False)
        if not isinstance(operation.get("centered"), bool):
            issues.append(_issue("invalid_boolean", f"{path}.centered", "Expected true or false."))
        _vec(operation.get("offset"), f"{path}.offset", issues, 3)
        return
    if kind == "revolve_profile":
        if operation.get("workplane") not in WORKPLANES:
            issues.append(_issue("invalid_workplane", f"{path}.workplane", "Use XY, XZ or YZ."))
        _validate_profile_or_wire(operation, path, issues, points_key="profile")
        axis = operation.get("axis")
        if (
            not isinstance(axis, list)
            or len(axis) != 2
            or not _vec(axis[0], f"{path}.axis[0]", issues, 2)
            or not _vec(axis[1], f"{path}.axis[1]", issues, 2)
            or axis[0] == axis[1]
        ):
            issues.append(_issue("invalid_revolve_axis", f"{path}.axis", "Expected two distinct 2D points."))
        _number(operation.get("angle"), f"{path}.angle", issues, 0.0, 360.0, inclusive_minimum=False)
        _vec(operation.get("offset"), f"{path}.offset", issues, 3)
        return
    if kind == "sweep_profile":
        if operation.get("workplane") not in WORKPLANES:
            issues.append(_issue("invalid_workplane", f"{path}.workplane", "Use XY, XZ or YZ."))
        _validate_profile_or_wire(operation, path, issues, points_key="profile")
        has_path = operation.get("path") is not None
        has_helix = operation.get("helix") is not None
        if has_path == has_helix:
            issues.append(_issue("path_xor_helix", path, "Specify exactly one of path or helix."))
        if has_path:
            _validate_path(operation.get("path"), f"{path}.path", issues)
        if has_helix:
            _validate_helix(operation.get("helix"), f"{path}.helix", issues)
        _vec(operation.get("offset") or [0.0, 0.0, 0.0], f"{path}.offset", issues, 3)
        return
    if kind == "loft_profiles":
        if operation.get("workplane") not in WORKPLANES:
            issues.append(_issue("invalid_workplane", f"{path}.workplane", "Use XY, XZ or YZ."))
        _validate_loft_profiles(operation.get("profiles"), f"{path}.profiles", issues)


def validate_plan_v3(plan: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    if not isinstance(plan, dict):
        return {"valid": False, "issues": [_issue("not_object", "$", "Plan must be a JSON object.")]}
    allowed = {"schema_version", "sample_id", "coordinate_system", "metadata", "operations"}
    required = {"schema_version", "sample_id", "coordinate_system", "operations"}
    for key in sorted(required - set(plan)):
        issues.append(_issue("missing_field", f"$.{key}", "Required field is missing."))
    for key in sorted(set(plan) - allowed):
        issues.append(_issue("extra_field", f"$.{key}", "Field is not allowed by HarnessCAD Plan v3."))
    if plan.get("schema_version") != SCHEMA_VERSION:
        issues.append(_issue("invalid_schema_version", "$.schema_version", f"Expected '{SCHEMA_VERSION}'."))
    sample_id = plan.get("sample_id")
    if not isinstance(sample_id, str) or not SAMPLE_ID_PATTERN.fullmatch(sample_id) or len(sample_id) > 128:
        issues.append(_issue("invalid_sample_id", "$.sample_id", "Use 1-128 letters, digits, '.', '_' or '-'."))
    _validate_coordinate_system(plan.get("coordinate_system"), issues)
    if "metadata" in plan:
        if not isinstance(plan["metadata"], dict):
            issues.append(_issue("invalid_metadata", "$.metadata", "metadata must be a JSON object."))
        else:
            _json_metadata(plan["metadata"], "$.metadata", issues)

    operations = plan.get("operations")
    if not isinstance(operations, list) or not 1 <= len(operations) <= MAX_OPERATIONS:
        issues.append(_issue("invalid_operations", "$.operations", f"Expected 1-{MAX_OPERATIONS} operations."))
        operations = []

    seen_ids: set[str] = set()
    shape_ids: set[str] = set()
    for index, operation in enumerate(operations):
        path = f"$.operations[{index}]"
        if not isinstance(operation, dict):
            issues.append(_issue("invalid_operation", path, "Expected an object."))
            continue
        raw_kind = operation.get("op")
        kind = raw_kind if isinstance(raw_kind, str) else None
        fields = OPERATION_FIELDS_V3.get(kind)
        allowed_fields = COMMON_FIELDS | (fields or set())
        for key in sorted(set(operation) - allowed_fields):
            issues.append(_issue("extra_field", f"{path}.{key}", "Field is not allowed for this operation."))
        if kind not in ALL_OPERATIONS_V3:
            issues.append(_issue("unsupported_operation", f"{path}.op", f"Unsupported operation; use one of {sorted(ALL_OPERATIONS_V3)}."))
            continue
        required_fields = COMMON_FIELDS | fields
        required_fields -= OPTIONAL_FIELDS_V3.get(kind, set())
        for key in sorted(required_fields - set(operation)):
            issues.append(_issue("missing_field", f"{path}.{key}", "Required field is missing."))

        operation_id = operation.get("id")
        id_valid = bool(isinstance(operation_id, str) and len(operation_id) <= 64 and ID_PATTERN.fullmatch(operation_id))
        id_is_new = id_valid and operation_id not in seen_ids
        if not id_valid:
            issues.append(_issue("invalid_operation_id", f"{path}.id", "Use a letter followed by letters, digits or '_'."))
        elif not id_is_new:
            issues.append(_issue("duplicate_operation_id", f"{path}.id", "Operation id must be unique."))
        else:
            seen_ids.add(operation_id)

        if kind in SHAPE_OPERATIONS_V3:
            _validate_shape_v3(operation, kind, path, index, shape_ids, issues)
            if id_is_new:
                shape_ids.add(operation_id)
        else:
            if index == 0:
                issues.append(_issue("modifier_cannot_be_first", f"{path}.op", "A modifier requires an existing result."))
            amount_field = "radius" if kind == "fillet" else "distance"
            _number(operation.get(amount_field), f"{path}.{amount_field}", issues, 0.0, 2.0, inclusive_minimum=False)
            if not (
                operation.get("edge_axis") is None
                or isinstance(operation.get("edge_axis"), str)
                and operation.get("edge_axis") in {"X", "Y", "Z"}
            ):
                issues.append(_issue("invalid_edge_axis", f"{path}.edge_axis", "Use X, Y, Z, or omit for all edges."))

    return {"valid": not issues, "issues": issues}


# Silence unused import of v2 ALL_OPERATIONS in type checkers that flag re-exports.
_ = (ALL_OPERATIONS, COMBINES)
