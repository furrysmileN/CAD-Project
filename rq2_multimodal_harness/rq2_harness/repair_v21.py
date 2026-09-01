from __future__ import annotations

import copy
import math
import re
from typing import Any, Iterable

from .common import sha256_json


REPAIR_VERSION = "harnesscad.repair.v2.1"
RULES = frozenset({"number", "unit_axis", "polygon", "rotate_revolve"})
_JSON_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")
_JSON_INTEGER = re.compile(r"-?(?:0|[1-9]\d*)\Z")

_SCALAR_FIELDS = {
    "box": (),
    "cylinder": ("radius", "height"),
    "sphere": ("radius",),
    "polygon_extrude": ("depth",),
    "revolve_profile": ("angle",),
    "hole": ("diameter", "depth"),
    "slot": ("length", "width", "depth", "angle"),
    "transform": (),
    "fillet": ("radius",),
    "chamfer": ("distance",),
    "linear_pattern": ("spacing",),
    "sweep_profile": (),
    "loft_profiles": (),
}
_VECTOR_FIELDS = {
    "box": ("center", "size"),
    "cylinder": ("center", "axis"),
    "sphere": ("center",),
    "polygon_extrude": ("offset",),
    "revolve_profile": ("offset",),
    "hole": ("center",),
    "slot": ("center",),
    "transform": ("translate",),
    "fillet": (),
    "chamfer": (),
    "linear_pattern": ("direction",),
    "sweep_profile": ("offset",),
    "loft_profiles": (),
}


def _coerce_number(value: Any, *, integer: bool = False) -> Any:
    if not isinstance(value, str):
        return value
    pattern = _JSON_INTEGER if integer else _JSON_NUMBER
    if not pattern.fullmatch(value):
        return value
    try:
        result: int | float = int(value) if integer else (int(value) if _JSON_INTEGER.fullmatch(value) else float(value))
    except ValueError:
        return value
    if isinstance(result, float) and not math.isfinite(result):
        return value
    return result


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_vec(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(_finite_number(item) for item in value)


def _orientation(a: list[float], b: list[float], c: list[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: list[float], b: list[float], c: list[float], d: list[float]) -> bool:
    eps = 1e-12
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (
        o3 > eps and o4 < -eps or o3 < -eps and o4 > eps
    ):
        return True

    def on_segment(start: list[float], finish: list[float], point: list[float]) -> bool:
        return (
            min(start[0], finish[0]) - eps <= point[0] <= max(start[0], finish[0]) + eps
            and min(start[1], finish[1]) - eps <= point[1] <= max(start[1], finish[1]) + eps
        )

    return (
        abs(o1) <= eps and on_segment(a, b, c)
        or abs(o2) <= eps and on_segment(a, b, d)
        or abs(o3) <= eps and on_segment(c, d, a)
        or abs(o4) <= eps and on_segment(c, d, b)
    )


def _safe_closed_polygon(value: Any, *, revolve_axis: Any = None) -> bool:
    if not isinstance(value, list) or not 4 <= len(value) <= 129:
        return False
    if not all(_finite_vec(point, 2) and all(-4.0 <= float(item) <= 4.0 for item in point) for point in value):
        return False
    points = [[float(item) for item in point] for point in value]
    if points[0] != points[-1]:
        return False
    unique = points[:-1]
    if len({tuple(point) for point in unique}) != len(unique) or len(unique) < 3:
        return False
    if any(points[index] == points[index + 1] for index in range(len(points) - 1)):
        return False
    area2 = sum(
        points[index][0] * points[index + 1][1] - points[index + 1][0] * points[index][1]
        for index in range(len(points) - 1)
    )
    if abs(area2) <= 1e-10:
        return False
    segment_count = len(points) - 1
    for first in range(segment_count):
        for second in range(first + 1, segment_count):
            if second in {first, first + 1} or (first == 0 and second == segment_count - 1):
                continue
            if _segments_intersect(points[first], points[first + 1], points[second], points[second + 1]):
                return False
    if revolve_axis is not None:
        if (
            not isinstance(revolve_axis, list)
            or len(revolve_axis) != 2
            or not all(_finite_vec(point, 2) for point in revolve_axis)
            or revolve_axis[0] == revolve_axis[1]
        ):
            return False
        a = [float(item) for item in revolve_axis[0]]
        b = [float(item) for item in revolve_axis[1]]
        signs = [_orientation(a, b, point) for point in unique]
        if min(signs) < -1e-10 and max(signs) > 1e-10:
            return False
    return True


def _iter_numeric_paths(operation: dict[str, Any], op_path: str) -> Iterable[tuple[list[Any], str]]:
    kind = operation.get("op")
    for key in _SCALAR_FIELDS.get(kind, ()):
        if key in operation:
            yield [operation, key], f"{op_path}.{key}"
    for key in _VECTOR_FIELDS.get(kind, ()):
        value = operation.get(key)
        if isinstance(value, list):
            for index in range(len(value)):
                yield [value, index], f"{op_path}.{key}[{index}]"
    if kind in {"polygon_extrude", "revolve_profile"}:
        key = "points" if kind == "polygon_extrude" else "profile"
        value = operation.get(key)
        if isinstance(value, list):
            for point_index, point in enumerate(value):
                if isinstance(point, list):
                    for component in range(len(point)):
                        yield [point, component], f"{op_path}.{key}[{point_index}][{component}]"
    if kind == "revolve_profile":
        axis = operation.get("axis")
        if isinstance(axis, list):
            for point_index, point in enumerate(axis):
                if isinstance(point, list):
                    for component in range(len(point)):
                        yield [point, component], f"{op_path}.axis[{point_index}][{component}]"
    if kind == "transform":
        rotate = operation.get("rotate")
        if isinstance(rotate, dict):
            if "angle" in rotate:
                yield [rotate, "angle"], f"{op_path}.rotate.angle"
            for key in ("origin", "axis"):
                value = rotate.get(key)
                if isinstance(value, list):
                    for index in range(len(value)):
                        yield [value, index], f"{op_path}.rotate.{key}[{index}]"


def repair_plan_v21(plan: dict[str, Any], rules: Iterable[str] = RULES) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = tuple(dict.fromkeys(rules))
    unknown = set(selected) - RULES
    if unknown:
        raise ValueError(f"未知修复规则: {sorted(unknown)}")
    repaired = copy.deepcopy(plan)
    before_hash = sha256_json(plan)
    changes: list[tuple[str, str]] = []

    def record(code: str, path: str) -> None:
        changes.append((code, path))

    if "number" in selected:
        coordinate = repaired.get("coordinate_system")
        if isinstance(coordinate, dict):
            origin = coordinate.get("origin")
            if isinstance(origin, list):
                for index, value in enumerate(origin):
                    converted = _coerce_number(value)
                    if converted != value or type(converted) is not type(value):
                        origin[index] = converted
                        record("coerce_numeric_string", f"$.coordinate_system.origin[{index}]")
            value = coordinate.get("longest_bbox_edge")
            converted = _coerce_number(value)
            if converted != value or type(converted) is not type(value):
                coordinate["longest_bbox_edge"] = converted
                record("coerce_numeric_string", "$.coordinate_system.longest_bbox_edge")
        operations = repaired.get("operations")
        if isinstance(operations, list):
            for index, operation in enumerate(operations):
                if not isinstance(operation, dict):
                    continue
                op_path = f"$.operations[{index}]"
                for holder_and_key, path in _iter_numeric_paths(operation, op_path):
                    holder, key = holder_and_key
                    value = holder[key]
                    converted = _coerce_number(value)
                    if converted != value or type(converted) is not type(value):
                        holder[key] = converted
                        record("coerce_numeric_string", path)
                if operation.get("op") == "linear_pattern" and "count" in operation:
                    value = operation["count"]
                    converted = _coerce_number(value, integer=True)
                    if converted != value or type(converted) is not type(value):
                        operation["count"] = converted
                        record("coerce_integer_string", f"{op_path}.count")

    operations = repaired.get("operations")
    if "rotate_revolve" in selected and isinstance(operations, list):
        for index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                continue
            op_path = f"$.operations[{index}]"
            if operation.get("op") == "revolve_profile":
                axis = operation.get("axis")
                aliases = (
                    ("start", "end"),
                    ("axisStart", "axisEnd"),
                    ("axis_start", "axis_end"),
                )
                if isinstance(axis, dict):
                    for start_key, end_key in aliases:
                        if set(axis) == {start_key, end_key} and _finite_vec(axis[start_key], 2) and _finite_vec(axis[end_key], 2):
                            operation["axis"] = [copy.deepcopy(axis[start_key]), copy.deepcopy(axis[end_key])]
                            record("canonicalize_revolve_axis_alias", f"{op_path}.axis")
                            break
            if operation.get("op") == "transform":
                rotate = operation.get("rotate")
                if isinstance(rotate, dict):
                    for alias in ("angleDegrees", "angle_degrees"):
                        if set(rotate) == {"origin", "axis", alias} and _finite_vec(rotate["origin"], 3) and _finite_vec(
                            rotate["axis"], 3
                        ) and _finite_number(rotate[alias]):
                            operation["rotate"] = {
                                "origin": copy.deepcopy(rotate["origin"]),
                                "axis": copy.deepcopy(rotate["axis"]),
                                "angle": rotate[alias],
                            }
                            record("canonicalize_rotate_angle_alias", f"{op_path}.rotate")
                            break

    if "unit_axis" in selected and isinstance(operations, list):
        for index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                continue
            candidates: list[tuple[Any, str]] = []
            kind = operation.get("op")
            if kind == "cylinder":
                candidates.append((operation.get("axis"), f"$.operations[{index}].axis"))
            elif kind == "linear_pattern":
                candidates.append((operation.get("direction"), f"$.operations[{index}].direction"))
            elif kind == "transform" and isinstance(operation.get("rotate"), dict):
                candidates.append((operation["rotate"].get("axis"), f"$.operations[{index}].rotate.axis"))
            for vector, path in candidates:
                if not _finite_vec(vector, 3):
                    continue
                norm = math.sqrt(sum(float(item) ** 2 for item in vector))
                if norm <= 1e-12 or math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
                    continue
                normalized = [float(item) / norm for item in vector]
                vector[:] = normalized
                record("normalize_unit_axis", path)

    if "polygon" in selected and isinstance(operations, list):
        for index, operation in enumerate(operations):
            if not isinstance(operation, dict) or operation.get("op") not in {"polygon_extrude", "revolve_profile"}:
                continue
            key = "points" if operation["op"] == "polygon_extrude" else "profile"
            original = operation.get(key)
            if not isinstance(original, list) or not original:
                continue
            candidate: list[Any] = []
            removed = False
            for point in original:
                if candidate and point == candidate[-1]:
                    removed = True
                    continue
                candidate.append(copy.deepcopy(point))
            closed = False
            if candidate and candidate[0] != candidate[-1]:
                candidate.append(copy.deepcopy(candidate[0]))
                closed = True
            revolve_axis = operation.get("axis") if operation["op"] == "revolve_profile" else None
            if (removed or closed) and _safe_closed_polygon(candidate, revolve_axis=revolve_axis):
                operation[key] = candidate
                if removed:
                    record("remove_consecutive_duplicate_vertex", f"$.operations[{index}].{key}")
                if closed:
                    record("close_polygon", f"$.operations[{index}].{key}")

    after_hash = sha256_json(repaired)
    codes = list(dict.fromkeys(code for code, _ in changes))
    paths = list(dict.fromkeys(path for _, path in changes))
    log = {
        "repair_version": REPAIR_VERSION,
        "rules": list(selected),
        "changed": before_hash != after_hash,
        "before_sha256": before_hash,
        "after_sha256": after_hash,
        "repair_codes": codes,
        "changed_paths": paths,
        "repair_count": len(changes),
    }
    return repaired, log
