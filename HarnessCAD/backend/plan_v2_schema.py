"""Strict schema and semantic validation for HarnessCAD Plan v2."""

from __future__ import annotations

import math
import re
from typing import Any


SCHEMA_VERSION = "harnesscad.plan.v2"
ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
SAMPLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
COMBINES = {"new", "add", "cut", "intersect"}
WORKPLANES = {"XY", "XZ", "YZ"}
SHAPE_OPERATIONS = {
    "box",
    "cylinder",
    "sphere",
    "polygon_extrude",
    "revolve_profile",
    "hole",
    "slot",
    "transform",
    "linear_pattern",
}
MODIFIER_OPERATIONS = {"fillet", "chamfer"}
ALL_OPERATIONS = SHAPE_OPERATIONS | MODIFIER_OPERATIONS

COMMON_FIELDS = {"id", "op"}
OPERATION_FIELDS = {
    "box": {"combine", "center", "size"},
    "cylinder": {"combine", "center", "radius", "height", "axis"},
    "sphere": {"combine", "center", "radius"},
    "polygon_extrude": {"combine", "workplane", "points", "depth", "centered", "offset"},
    "revolve_profile": {"combine", "workplane", "profile", "axis", "angle", "offset"},
    "hole": {"combine", "workplane", "center", "diameter", "depth"},
    "slot": {"combine", "workplane", "center", "length", "width", "depth", "angle"},
    "transform": {"combine", "source", "translate", "rotate"},
    "fillet": {"radius", "edge_axis"},
    "chamfer": {"distance", "edge_axis"},
    "linear_pattern": {"combine", "source", "direction", "count", "spacing"},
}
OPTIONAL_FIELDS = {
    "transform": {"translate", "rotate"},
    "fillet": {"edge_axis"},
    "chamfer": {"edge_axis"},
}


def _issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message, "severity": "error"}


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _number(
    value: object,
    path: str,
    issues: list[dict[str, str]],
    minimum: float,
    maximum: float,
    *,
    inclusive_minimum: bool = True,
) -> bool:
    valid = _is_number(value)
    if valid:
        numeric = float(value)
        valid = (numeric >= minimum if inclusive_minimum else numeric > minimum) and numeric <= maximum
    if not valid:
        bracket = "[" if inclusive_minimum else "("
        issues.append(_issue("invalid_number", path, f"Expected a finite number in {bracket}{minimum}, {maximum}]."))
    return valid


def _vec(
    value: object,
    path: str,
    issues: list[dict[str, str]],
    length: int,
    minimum: float = -4.0,
    maximum: float = 4.0,
) -> bool:
    if not isinstance(value, list) or len(value) != length:
        issues.append(_issue("invalid_vector", path, f"Expected exactly {length} numeric components."))
        return False
    return all(_number(item, f"{path}[{index}]", issues, minimum, maximum) for index, item in enumerate(value))


def _unit_vec3(value: object, path: str, issues: list[dict[str, str]]) -> bool:
    if not _vec(value, path, issues, 3, -1.0, 1.0):
        return False
    norm = math.sqrt(sum(float(item) ** 2 for item in value))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        issues.append(_issue("axis_not_unit_length", path, f"Axis norm is {norm:.8g}; expected 1."))
        return False
    return True


def _orientation(a: list[float], b: list[float], c: list[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: list[float], b: list[float], c: list[float], d: list[float]) -> bool:
    eps = 1e-12
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    proper_crossing = (
        (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps)
        and (o3 > eps and o4 < -eps or o3 < -eps and o4 > eps)
    )
    if proper_crossing:
        return True

    def on_segment(start: list[float], end: list[float], point: list[float]) -> bool:
        return (
            min(start[0], end[0]) - eps <= point[0] <= max(start[0], end[0]) + eps
            and min(start[1], end[1]) - eps <= point[1] <= max(start[1], end[1]) + eps
        )

    return (
        abs(o1) <= eps and on_segment(a, b, c)
        or abs(o2) <= eps and on_segment(a, b, d)
        or abs(o3) <= eps and on_segment(c, d, a)
        or abs(o4) <= eps and on_segment(c, d, b)
    )


def _closed_polygon(value: object, path: str, issues: list[dict[str, str]]) -> bool:
    if not isinstance(value, list) or not 4 <= len(value) <= 129:
        issues.append(_issue("invalid_polygon", path, "Expected 4-129 points, with the first point repeated last."))
        return False
    valid = True
    for index, point in enumerate(value):
        valid = _vec(point, f"{path}[{index}]", issues, 2) and valid
    if not valid:
        return False
    points = [[float(component) for component in point] for point in value]
    if points[0] != points[-1]:
        issues.append(_issue("polygon_not_closed", path, "The final point must exactly repeat the first point."))
        valid = False
    unique = points[:-1]
    if len({tuple(point) for point in unique}) < 3:
        issues.append(_issue("degenerate_polygon", path, "A polygon needs at least three distinct vertices."))
        valid = False
    elif len({tuple(point) for point in unique}) != len(unique):
        issues.append(_issue("duplicate_polygon_vertex", path, "Vertices may not repeat except for explicit closure."))
        valid = False
    for index in range(len(points) - 1):
        if points[index] == points[index + 1]:
            issues.append(_issue("duplicate_polygon_vertex", f"{path}[{index + 1}]", "Consecutive vertices must differ."))
            valid = False
    area2 = sum(
        points[index][0] * points[index + 1][1] - points[index + 1][0] * points[index][1]
        for index in range(len(points) - 1)
    )
    if abs(area2) <= 1e-10:
        issues.append(_issue("degenerate_polygon", path, "Polygon signed area must be non-zero."))
        valid = False
    segment_count = len(points) - 1
    for first in range(segment_count):
        for second in range(first + 1, segment_count):
            if second in {first, first + 1} or (first == 0 and second == segment_count - 1):
                continue
            if _segments_intersect(points[first], points[first + 1], points[second], points[second + 1]):
                issues.append(_issue("self_intersecting_polygon", path, "Polygon edges must not cross."))
                return False
    return valid


def _json_metadata(value: object, path: str, issues: list[dict[str, str]], depth: int = 0) -> bool:
    if depth > 4:
        issues.append(_issue("metadata_too_deep", path, "metadata nesting is limited to four levels."))
        return False
    if value is None or isinstance(value, bool) or _is_number(value):
        return True
    if isinstance(value, str):
        if len(value) <= 1024:
            return True
        issues.append(_issue("metadata_string_too_long", path, "metadata strings are limited to 1024 characters."))
        return False
    if isinstance(value, list):
        if len(value) > 64:
            issues.append(_issue("metadata_array_too_large", path, "metadata arrays are limited to 64 items."))
            return False
        return all(_json_metadata(item, f"{path}[{index}]", issues, depth + 1) for index, item in enumerate(value))
    if isinstance(value, dict):
        if len(value) > 64:
            issues.append(_issue("metadata_object_too_large", path, "metadata objects are limited to 64 fields."))
            return False
        valid = True
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                issues.append(_issue("invalid_metadata_key", path, "metadata keys must be 1-128 character strings."))
                valid = False
            else:
                valid = _json_metadata(item, f"{path}.{key}", issues, depth + 1) and valid
        return valid
    issues.append(_issue("invalid_metadata_value", path, "metadata must contain only bounded JSON values."))
    return False


def _validate_coordinate_system(value: object, issues: list[dict[str, str]]) -> None:
    path = "$.coordinate_system"
    if not isinstance(value, dict):
        issues.append(_issue("invalid_coordinate_system", path, "Expected an object."))
        return
    required = {"units", "origin", "longest_bbox_edge"}
    for key in sorted(required - set(value)):
        issues.append(_issue("missing_field", f"{path}.{key}", "Required field is missing."))
    for key in sorted(set(value) - required):
        issues.append(_issue("extra_field", f"{path}.{key}", "Field is not allowed."))
    if value.get("units") != "normalized":
        issues.append(_issue("invalid_units", f"{path}.units", "Expected 'normalized'."))
    origin = value.get("origin")
    if _vec(origin, f"{path}.origin", issues, 3) and any(float(item) != 0.0 for item in origin):
        issues.append(_issue("noncanonical_origin", f"{path}.origin", "Expected [0, 0, 0]."))
    edge = value.get("longest_bbox_edge")
    if not _is_number(edge) or not math.isclose(float(edge), 1.0, rel_tol=0.0, abs_tol=1e-12):
        issues.append(_issue("invalid_bbox_scale", f"{path}.longest_bbox_edge", "Expected 1.0."))


def _validate_combine(operation: dict[str, Any], path: str, index: int, issues: list[dict[str, str]]) -> None:
    combine = operation.get("combine")
    if not isinstance(combine, str) or combine not in COMBINES:
        issues.append(_issue("unsupported_combine", f"{path}.combine", "Use new, add, cut or intersect."))
    elif index == 0 and combine != "new":
        issues.append(_issue("first_operation_must_be_new", f"{path}.combine", "The first operation must use new."))
    elif index > 0 and combine == "new":
        issues.append(_issue("new_only_allowed_first", f"{path}.combine", "Only the first operation may use new."))


def _validate_shape_operation(
    operation: dict[str, Any],
    kind: str,
    path: str,
    index: int,
    shape_ids: set[str],
    issues: list[dict[str, str]],
) -> None:
    _validate_combine(operation, path, index, issues)
    if kind == "box":
        _vec(operation.get("center"), f"{path}.center", issues, 3)
        size = operation.get("size")
        if not isinstance(size, list) or len(size) != 3:
            issues.append(_issue("invalid_size", f"{path}.size", "Expected exactly three dimensions."))
        else:
            for axis, value in enumerate(size):
                _number(value, f"{path}.size[{axis}]", issues, 0.0, 8.0, inclusive_minimum=False)
    elif kind == "cylinder":
        _vec(operation.get("center"), f"{path}.center", issues, 3)
        _number(operation.get("radius"), f"{path}.radius", issues, 0.0, 4.0, inclusive_minimum=False)
        _number(operation.get("height"), f"{path}.height", issues, 0.0, 8.0, inclusive_minimum=False)
        _unit_vec3(operation.get("axis"), f"{path}.axis", issues)
    elif kind == "sphere":
        _vec(operation.get("center"), f"{path}.center", issues, 3)
        _number(operation.get("radius"), f"{path}.radius", issues, 0.0, 4.0, inclusive_minimum=False)
    elif kind == "polygon_extrude":
        if not isinstance(operation.get("workplane"), str) or operation.get("workplane") not in WORKPLANES:
            issues.append(_issue("invalid_workplane", f"{path}.workplane", "Use XY, XZ or YZ."))
        _closed_polygon(operation.get("points"), f"{path}.points", issues)
        _number(operation.get("depth"), f"{path}.depth", issues, 0.0, 8.0, inclusive_minimum=False)
        if not isinstance(operation.get("centered"), bool):
            issues.append(_issue("invalid_boolean", f"{path}.centered", "Expected true or false."))
        _vec(operation.get("offset"), f"{path}.offset", issues, 3)
    elif kind == "revolve_profile":
        if not isinstance(operation.get("workplane"), str) or operation.get("workplane") not in WORKPLANES:
            issues.append(_issue("invalid_workplane", f"{path}.workplane", "Use XY, XZ or YZ."))
        profile_valid = _closed_polygon(operation.get("profile"), f"{path}.profile", issues)
        axis = operation.get("axis")
        axis_valid = (
            isinstance(axis, list)
            and len(axis) == 2
            and _vec(axis[0], f"{path}.axis[0]", issues, 2)
            and _vec(axis[1], f"{path}.axis[1]", issues, 2)
        )
        if not axis_valid:
            if not isinstance(axis, list) or len(axis) != 2:
                issues.append(_issue("invalid_revolve_axis", f"{path}.axis", "Expected two distinct 2D points."))
        elif axis[0] == axis[1]:
            issues.append(_issue("invalid_revolve_axis", f"{path}.axis", "Axis points must differ."))
            axis_valid = False
        _number(operation.get("angle"), f"{path}.angle", issues, 0.0, 360.0, inclusive_minimum=False)
        _vec(operation.get("offset"), f"{path}.offset", issues, 3)
        if profile_valid and axis_valid:
            a = [float(item) for item in axis[0]]
            b = [float(item) for item in axis[1]]
            signs = [_orientation(a, b, [float(item) for item in point]) for point in operation["profile"][:-1]]
            if min(signs) < -1e-10 and max(signs) > 1e-10:
                issues.append(_issue("profile_crosses_axis", f"{path}.profile", "Profile vertices must stay on one side of the revolve axis."))
    elif kind in {"hole", "slot"}:
        if operation.get("combine") != "cut":
            issues.append(_issue("cut_operation_required", f"{path}.combine", f"{kind} must use cut."))
        if not isinstance(operation.get("workplane"), str) or operation.get("workplane") not in WORKPLANES:
            issues.append(_issue("invalid_workplane", f"{path}.workplane", "Use XY, XZ or YZ."))
        _vec(operation.get("center"), f"{path}.center", issues, 3)
        _number(operation.get("depth"), f"{path}.depth", issues, 0.0, 8.0, inclusive_minimum=False)
        if kind == "hole":
            _number(operation.get("diameter"), f"{path}.diameter", issues, 0.0, 8.0, inclusive_minimum=False)
        else:
            length_ok = _number(operation.get("length"), f"{path}.length", issues, 0.0, 8.0, inclusive_minimum=False)
            width_ok = _number(operation.get("width"), f"{path}.width", issues, 0.0, 8.0, inclusive_minimum=False)
            if length_ok and width_ok and float(operation["length"]) < float(operation["width"]):
                issues.append(_issue("invalid_slot_dimensions", f"{path}.length", "Slot length must be at least its width."))
            _number(operation.get("angle"), f"{path}.angle", issues, -360.0, 360.0)
    elif kind == "transform":
        source = operation.get("source")
        if not isinstance(source, str) or source not in shape_ids:
            issues.append(_issue("invalid_reference", f"{path}.source", "source must reference an earlier shape-producing operation."))
        translate = operation.get("translate")
        rotate = operation.get("rotate")
        if translate is None and rotate is None:
            issues.append(_issue("empty_transform", path, "Specify translate, rotate, or both."))
        if translate is not None:
            _vec(translate, f"{path}.translate", issues, 3, -8.0, 8.0)
        if rotate is not None:
            rotate_path = f"{path}.rotate"
            required = {"origin", "axis", "angle"}
            if not isinstance(rotate, dict):
                issues.append(_issue("invalid_rotate", rotate_path, "Expected origin, unit axis and angle."))
            else:
                for key in sorted(required - set(rotate)):
                    issues.append(_issue("missing_field", f"{rotate_path}.{key}", "Required field is missing."))
                for key in sorted(set(rotate) - required):
                    issues.append(_issue("extra_field", f"{rotate_path}.{key}", "Field is not allowed."))
                _vec(rotate.get("origin"), f"{rotate_path}.origin", issues, 3)
                _unit_vec3(rotate.get("axis"), f"{rotate_path}.axis", issues)
                _number(rotate.get("angle"), f"{rotate_path}.angle", issues, -360.0, 360.0)
    elif kind == "linear_pattern":
        source = operation.get("source")
        if not isinstance(source, str) or source not in shape_ids:
            issues.append(_issue("invalid_reference", f"{path}.source", "source must reference an earlier shape-producing operation."))
        _unit_vec3(operation.get("direction"), f"{path}.direction", issues)
        count = operation.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or not 2 <= count <= 32:
            issues.append(_issue("invalid_pattern_count", f"{path}.count", "Expected an integer in [2, 32]."))
        _number(operation.get("spacing"), f"{path}.spacing", issues, 0.0, 8.0, inclusive_minimum=False)


def validate_plan_v2(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate a Plan v2 without evaluating code or mutating the input."""
    issues: list[dict[str, str]] = []
    if not isinstance(plan, dict):
        return {"valid": False, "issues": [_issue("invalid_plan", "$", "Expected a JSON object.")]}

    allowed = {"schema_version", "sample_id", "coordinate_system", "metadata", "operations"}
    required = {"schema_version", "sample_id", "coordinate_system", "operations"}
    for key in sorted(required - set(plan)):
        issues.append(_issue("missing_field", f"$.{key}", "Required field is missing."))
    for key in sorted(set(plan) - allowed):
        issues.append(_issue("extra_field", f"$.{key}", "Field is not allowed by HarnessCAD Plan v2."))
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
    if not isinstance(operations, list) or not 1 <= len(operations) <= 64:
        issues.append(_issue("invalid_operations", "$.operations", "Expected 1-64 operations."))
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
        fields = OPERATION_FIELDS.get(kind)
        allowed_fields = COMMON_FIELDS | (fields or set())
        for key in sorted(set(operation) - allowed_fields):
            issues.append(_issue("extra_field", f"{path}.{key}", "Field is not allowed for this operation."))
        if kind not in ALL_OPERATIONS:
            issues.append(_issue("unsupported_operation", f"{path}.op", f"Unsupported operation; use one of {sorted(ALL_OPERATIONS)}."))
            continue
        required_fields = COMMON_FIELDS | fields
        required_fields -= OPTIONAL_FIELDS.get(kind, set())
        for key in sorted(required_fields - set(operation)):
            issues.append(_issue("missing_field", f"{path}.{key}", "Required field is missing."))

        operation_id = operation.get("id")
        id_valid = bool(
            isinstance(operation_id, str)
            and len(operation_id) <= 64
            and ID_PATTERN.fullmatch(operation_id)
        )
        id_is_new = id_valid and operation_id not in seen_ids
        if not id_valid:
            issues.append(_issue("invalid_operation_id", f"{path}.id", "Use a letter followed by letters, digits or '_'."))
        elif not id_is_new:
            issues.append(_issue("duplicate_operation_id", f"{path}.id", "Operation id must be unique."))
        else:
            seen_ids.add(operation_id)

        if kind in SHAPE_OPERATIONS:
            _validate_shape_operation(operation, kind, path, index, shape_ids, issues)
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
