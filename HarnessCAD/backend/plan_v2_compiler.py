"""Deterministic, data-only compiler for HarnessCAD Plan v2."""

from __future__ import annotations

import json
from typing import Any

from .plan_v2_schema import validate_plan_v2
from .plan_v3_schema import SCHEMA_VERSION as PLAN_V3_SCHEMA_VERSION
from .plan_v3_schema import validate_plan_v3
from .plan_v31_schema import SCHEMA_VERSION as PLAN_V31_SCHEMA_VERSION
from .plan_v31_schema import validate_plan_v31


RUNTIME_HELPERS = r'''
import json
import math
import sys
import time
import traceback
from pathlib import Path

import cadquery as cq

STEP_PATH = Path(sys.argv[1])
STL_PATH = Path(sys.argv[2])
TRACE_PATH = Path(sys.argv[3])
TRACE = {
    "traceVersion": "harnesscad.runtime.v2",
    "status": "running",
    "operations": [],
    "stages": [],
    "warnings": [],
    "failure": None,
    "finalMetrics": None,
}
SHAPES = {}
result = None


def _write_trace():
    TRACE_PATH.write_text(json.dumps(TRACE, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _shape_metrics(shape):
    if shape is None:
        return {
            "shapeType": None, "valid": False, "solidCount": 0, "shellCount": 0,
            "faceCount": 0, "edgeCount": 0, "vertexCount": 0, "volume": 0.0,
            "area": 0.0, "bboxSize": None, "bboxCenter": None,
        }
    try:
        solids = list(shape.Solids())
    except Exception:
        solids = []
    metrics = {
        "shapeType": shape.ShapeType(),
        "valid": bool(shape.isValid()),
        "solidCount": len(solids),
        "shellCount": len(shape.Shells()),
        "faceCount": len(shape.Faces()),
        "edgeCount": len(shape.Edges()),
        "vertexCount": len(shape.Vertices()),
        "volume": float(shape.Volume()) if solids else 0.0,
        "area": float(shape.Area()) if solids else 0.0,
        "bboxSize": None,
        "bboxCenter": None,
    }
    if solids:
        bbox = shape.BoundingBox()
        metrics["bboxSize"] = [float(bbox.xlen), float(bbox.ylen), float(bbox.zlen)]
        metrics["bboxCenter"] = [
            float((bbox.xmin + bbox.xmax) / 2.0),
            float((bbox.ymin + bbox.ymax) / 2.0),
            float((bbox.zmin + bbox.zmax) / 2.0),
        ]
    return metrics


def _fail(code, stage, message, operation_index=None, operation_id=None, exception=None):
    TRACE["status"] = "failed"
    TRACE["failure"] = {
        "code": code,
        "stage": stage,
        "message": message,
        "operationIndex": operation_index,
        "operationId": operation_id,
        "exceptionType": type(exception).__name__ if exception is not None else None,
        "traceback": traceback.format_exc() if exception is not None else None,
    }
    _write_trace()
    raise SystemExit(20)


def _vector(values):
    return cq.Vector(float(values[0]), float(values[1]), float(values[2]))


def _plane(workplane, origin):
    # Explicit right-handed global bases: XY=(X,Y,+Z), XZ=(X,Z,-Y), YZ=(Y,Z,+X).
    if workplane == "XY":
        x_dir, normal = (1, 0, 0), (0, 0, 1)
    elif workplane == "XZ":
        x_dir, normal = (1, 0, 0), (0, -1, 0)
    else:
        x_dir, normal = (0, 1, 0), (1, 0, 0)
    return cq.Plane(origin=_vector(origin), xDir=_vector(x_dir), normal=_vector(normal))


def _normal(workplane):
    if workplane == "XY":
        return cq.Vector(0, 0, 1)
    if workplane == "XZ":
        return cq.Vector(0, -1, 0)
    return cq.Vector(1, 0, 0)


def _section_from_op(wp, op, points_key="points"):
    wire = op.get("wire")
    if wire:
        start = wire[0]["to"]
        wp = wp.moveTo(float(start[0]), float(start[1]))
        for segment in wire[1:]:
            if segment["kind"] == "line":
                wp = wp.lineTo(float(segment["to"][0]), float(segment["to"][1]))
            else:
                through = segment["through"]
                dest = segment["to"]
                wp = wp.threePointArc((float(through[0]), float(through[1])), (float(dest[0]), float(dest[1])))
        return wp.close()
    points = op[points_key]
    return wp.polyline(points[:-1]).close()


def _helix_wire(helix):
    pitch = float(helix["pitch"])
    height = pitch * float(helix["turns"])
    return cq.Wire.makeHelix(
        pitch,
        height,
        float(helix["radius"]),
        _vector(helix["center"]),
        _vector(helix["axis"]),
    )


def _path_wire(points):
    edges = [
        cq.Edge.makeLine(_vector(start), _vector(end))
        for start, end in zip(points, points[1:])
    ]
    return cq.Wire.assembleEdges(edges)


def _path_wire_segments(segments):
    cursor = segments[0]["to"]
    edges = []
    for segment in segments[1:]:
        destination = segment["to"]
        if segment["kind"] == "line":
            edge = cq.Edge.makeLine(_vector(cursor), _vector(destination))
        else:
            edge = cq.Edge.makeThreePointArc(
                _vector(cursor),
                _vector(segment["through"]),
                _vector(destination),
            )
        edges.append(edge)
        cursor = destination
    return cq.Wire.assembleEdges(edges)


def _build_shape(op):
    kind = op["op"]
    if kind == "box":
        return (
            cq.Workplane("XY")
            .box(float(op["size"][0]), float(op["size"][1]), float(op["size"][2]), centered=(True, True, True))
            .val()
            .translate(_vector(op["center"]))
        )
    if kind == "sphere":
        return cq.Solid.makeSphere(float(op["radius"]), _vector(op["center"]))
    if kind == "cylinder":
        axis = _vector(op["axis"])
        height = float(op["height"])
        start = _vector(op["center"]) - axis.multiply(height / 2.0)
        return cq.Solid.makeCylinder(float(op["radius"]), height, start, axis)
    if kind == "polygon_extrude":
        wire = _section_from_op(cq.Workplane(_plane(op["workplane"], op["offset"])), op, "points")
        distance = float(op["depth"]) / 2.0 if op["centered"] else float(op["depth"])
        return wire.extrude(distance, both=bool(op["centered"])).val()
    if kind == "revolve_profile":
        profile = _section_from_op(cq.Workplane(_plane(op["workplane"], op["offset"])), op, "profile")
        return profile.revolve(
            angleDegrees=float(op["angle"]),
            axisStart=tuple(op["axis"][0]),
            axisEnd=tuple(op["axis"][1]),
            combine=False,
            clean=True,
        ).val()
    if kind == "hole":
        axis = _normal(op["workplane"])
        depth = float(op["depth"])
        start = _vector(op["center"]) - axis.multiply(depth / 2.0)
        return cq.Solid.makeCylinder(float(op["diameter"]) / 2.0, depth, start, axis)
    if kind == "slot":
        plane = _plane(op["workplane"], op["center"])
        return (
            cq.Workplane(plane)
            .slot2D(float(op["length"]), float(op["width"]), float(op["angle"]))
            .extrude(float(op["depth"]) / 2.0, both=True)
            .val()
        )
    if kind == "transform":
        shape = SHAPES[op["source"]].copy()
        rotate = op.get("rotate")
        if rotate is not None:
            origin = _vector(rotate["origin"])
            shape = shape.rotate(origin, origin + _vector(rotate["axis"]), float(rotate["angle"]))
        if op.get("translate") is not None:
            shape = shape.translate(_vector(op["translate"]))
        return shape
    if kind == "linear_pattern":
        source = SHAPES[op["source"]]
        direction = _vector(op["direction"])
        spacing = float(op["spacing"])
        copies = [source.copy().translate(direction.multiply(spacing * index)) for index in range(int(op["count"]))]
        patterned = copies[0]
        for copy in copies[1:]:
            patterned = patterned.fuse(copy)
        return patterned
    if kind == "sweep_profile":
        section = _section_from_op(cq.Workplane(_plane(op["workplane"], op.get("offset") or [0.0, 0.0, 0.0])), op, "profile")
        if op.get("helix") is not None:
            path = _helix_wire(op["helix"])
        elif op.get("path_wire") is not None:
            path = _path_wire_segments(op["path_wire"])
        else:
            path = _path_wire(op["path"])
        return section.sweep(path, isFrenet=op.get("sweep_mode") == "frenet").val()
    if kind == "loft_profiles":
        wires = []
        for profile in op["profiles"]:
            workplane = profile.get("workplane") or op.get("workplane") or "XY"
            offset = profile.get("offset") or [0.0, 0.0, 0.0]
            section = _section_from_op(cq.Workplane(_plane(workplane, offset)), profile, "points")
            wires.append(section.wires().val())
        return cq.Solid.makeLoft(wires)
    raise RuntimeError("operation does not produce a standalone shape")


def _apply_operation(op, current):
    kind = op["op"]
    if kind in {"fillet", "chamfer"}:
        if op.get("edge_axis") is not None:
            selected = cq.Workplane(obj=current).edges(cq.selectors.ParallelDirSelector(_vector(
                {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[op["edge_axis"]]
            )))
        else:
            selected = cq.Workplane(obj=current).edges()
        if kind == "fillet":
            return selected.fillet(float(op["radius"])).val(), None
        return selected.chamfer(float(op["distance"])).val(), None

    shape = _build_shape(op)
    SHAPES[op["id"]] = shape
    combine = op["combine"]
    if combine == "new":
        return shape, shape
    if combine == "add":
        return current.fuse(shape), shape
    if combine == "cut":
        return current.cut(shape), shape
    return current.intersect(shape), shape
'''


def _operation_literal(operation: dict[str, Any]) -> str:
    canonical_json = json.dumps(operation, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"json.loads({json.dumps(canonical_json, ensure_ascii=True)})"


def compile_plan_v2(plan: dict[str, Any]) -> str:
    """Compile a validated v2/v3 plan to deterministic, non-user-programmable Python."""
    if plan.get("schema_version") == PLAN_V31_SCHEMA_VERSION:
        validation = validate_plan_v31(plan)
    elif plan.get("schema_version") == PLAN_V3_SCHEMA_VERSION:
        validation = validate_plan_v3(plan)
    else:
        validation = validate_plan_v2(plan)
    if not validation["valid"]:
        first = validation["issues"][0]
        raise ValueError(f"Invalid HarnessCAD Plan at {first['path']}: {first['message']}")

    lines = [
        '"""Generated deterministically by HarnessCAD Plan v2."""',
        RUNTIME_HELPERS.strip(),
        "",
    ]
    for index, operation in enumerate(plan["operations"]):
        lines.extend(
            [
                f"_op_index = {index}",
                f"_op = {_operation_literal(operation)}",
                "_op_id = _op['id']",
                "_operation = _op['op']",
                "_combine = _op.get('combine')",
                "_before = _shape_metrics(result)",
                "_op_started = time.perf_counter()",
                "try:",
                "    result, _produced_shape = _apply_operation(_op, result)",
                "    _after = _shape_metrics(result)",
                "    _duration = time.perf_counter() - _op_started",
                "    _op_warnings = []",
                "    _delta = _after['volume'] - _before['volume']",
                "    if _after['solidCount'] > 1:",
                "        _op_warnings.append('multiple_solids_after_operation')",
                "    if _op_index > 0 and abs(_delta) <= 1e-12:",
                "        _op_warnings.append('ineffective_operation')",
                "    _record = {",
                "        'index': _op_index, 'id': _op_id, 'primitive': _operation, 'operation': _operation,",
                "        'combine': _combine, 'status': 'success' if not _op_warnings else 'success_with_warnings',",
                "        'durationSec': _duration, 'before': _before, 'after': _after,",
                "        'volumeDelta': _delta, 'warnings': _op_warnings,",
                "    }",
                "    TRACE['operations'].append(_record)",
                "    _write_trace()",
                "    if _after['solidCount'] == 0 or _after['volume'] <= 1e-12:",
                "        _record['status'] = 'failed'",
                "        _fail('empty_after_operation', 'operation', 'Operation produced no solid material.', _op_index, _op_id)",
                "    if not _after['valid']:",
                "        _record['status'] = 'failed'",
                "        _fail('invalid_shape_after_operation', 'operation', 'OpenCascade reports an invalid shape.', _op_index, _op_id)",
                "except SystemExit:",
                "    raise",
                "except Exception as _exc:",
                "    TRACE['operations'].append({",
                "        'index': _op_index, 'id': _op_id, 'primitive': _operation, 'operation': _operation,",
                "        'combine': _combine, 'status': 'failed',",
                "        'durationSec': time.perf_counter() - _op_started, 'before': _before,",
                "        'after': None, 'volumeDelta': None, 'warnings': [],",
                "    })",
                "    _fail('operation_exception', 'operation', str(_exc), _op_index, _op_id, _exc)",
                "",
            ]
        )

    lines.extend(
        [
            "_export_started = time.perf_counter()",
            "try:",
            "    cq.exporters.export(result, str(STEP_PATH))",
            "except Exception as _exc:",
            "    _fail('step_export_failed', 'export_step', str(_exc), exception=_exc)",
            "TRACE['stages'].append({'stage': 'export_step', 'status': 'success', 'durationSec': time.perf_counter() - _export_started, 'bytes': STEP_PATH.stat().st_size if STEP_PATH.exists() else 0})",
            "_export_started = time.perf_counter()",
            "try:",
            "    cq.exporters.export(result, str(STL_PATH))",
            "except Exception as _exc:",
            "    _fail('stl_export_failed', 'export_stl', str(_exc), exception=_exc)",
            "_stl_bytes = STL_PATH.stat().st_size if STL_PATH.exists() else 0",
            "TRACE['stages'].append({'stage': 'export_stl', 'status': 'success' if _stl_bytes > 0 else 'failed', 'durationSec': time.perf_counter() - _export_started, 'bytes': _stl_bytes})",
            "if _stl_bytes <= 0:",
            "    _fail('empty_stl_export', 'export_stl', 'STL export produced no triangle data.')",
            "_final = _shape_metrics(result)",
            "_bbox_size = _final['bboxSize'] or [0.0, 0.0, 0.0]",
            "_bbox_center = _final['bboxCenter'] or [0.0, 0.0, 0.0]",
            "_canonical = math.isclose(max(_bbox_size), 1.0, abs_tol=1e-6) and all(abs(value) <= 1e-6 for value in _bbox_center)",
            "_final['canonicalFrame'] = _canonical",
            "TRACE['finalMetrics'] = _final",
            "if not _canonical:",
            "    TRACE['warnings'].append({'code': 'noncanonical_final_geometry', 'severity': 'warning', 'message': 'Final bbox is not centered unit-bbox geometry.'})",
            "if _final['solidCount'] > 1:",
            "    TRACE['warnings'].append({'code': 'multiple_final_solids', 'severity': 'warning', 'message': 'Final result contains multiple disconnected solids.'})",
            "TRACE['status'] = 'success_with_warnings' if TRACE['warnings'] or any(op['warnings'] for op in TRACE['operations']) else 'success'",
            "_write_trace()",
            "",
        ]
    )
    return "\n".join(lines)


def compile_plan_v3(plan: dict[str, Any]) -> str:
    return compile_plan_v2(plan)
