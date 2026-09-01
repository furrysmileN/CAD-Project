"""HarnessCAD episode v2 with preflight checks and per-operation geometry traces."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .harness_api import _shape_expression, validate_plan
from .plan_v2_compiler import compile_plan_v2 as compile_schema_plan_v2
from .plan_v2_schema import SCHEMA_VERSION as PLAN_V2_SCHEMA_VERSION
from .plan_v2_schema import validate_plan_v2 as validate_schema_plan_v2
from .plan_v3_schema import SCHEMA_VERSION as PLAN_V3_SCHEMA_VERSION
from .plan_v3_schema import validate_plan_v3 as validate_schema_plan_v3
from .plan_v31_schema import SCHEMA_VERSION as PLAN_V31_SCHEMA_VERSION
from .plan_v31_schema import validate_plan_v31 as validate_schema_plan_v31


BASE_DIR = Path(__file__).resolve().parent
HARNESS_RUNS_V2_DIR = BASE_DIR / "harness_runs_v2"
HARNESS_RUNS_V2_DIR.mkdir(parents=True, exist_ok=True)
router = APIRouter(prefix="/api/harness-v2", tags=["harness-v2"])
TRACE_VERSION = "harnesscad.episode.v2"


class PlanRequest(BaseModel):
    plan: dict[str, Any]


class RunRequest(PlanRequest):
    timeout_sec: float = Field(default=30.0, ge=1.0, le=120.0)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _operation_kind(operation: dict[str, Any]) -> str:
    return str(operation.get("op", operation.get("primitive", "unknown")))


def _primitive_bbox(operation: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    primitive = _operation_kind(operation)
    if primitive in {"transform", "linear_pattern", "fillet", "chamfer", "revolve_profile", "sweep_profile", "loft_profiles"}:
        return None
    if primitive == "polygon_extrude":
        raw_points = operation.get("points")
        if raw_points is None:
            raw_points = [segment.get("to") for segment in operation.get("wire") or [] if segment.get("to") is not None]
        if not raw_points:
            return None
        points = raw_points[:-1] if raw_points[0] == raw_points[-1] else raw_points
        depth = float(operation["depth"])
        minimum_2d = [min(float(point[axis]) for point in points) for axis in range(2)]
        maximum_2d = [max(float(point[axis]) for point in points) for axis in range(2)]
        offset = [float(value) for value in operation["offset"]]
        normal_range = (-depth / 2.0, depth / 2.0) if operation["centered"] else (0.0, depth)
        if operation["workplane"] == "XY":
            local_min, local_max = [minimum_2d[0], minimum_2d[1], normal_range[0]], [maximum_2d[0], maximum_2d[1], normal_range[1]]
        elif operation["workplane"] == "XZ":
            local_min, local_max = [minimum_2d[0], -normal_range[1], minimum_2d[1]], [maximum_2d[0], -normal_range[0], maximum_2d[1]]
        else:
            local_min, local_max = [normal_range[0], minimum_2d[0], minimum_2d[1]], [normal_range[1], maximum_2d[0], maximum_2d[1]]
        return (
            [local_min[index] + offset[index] for index in range(3)],
            [local_max[index] + offset[index] for index in range(3)],
        )
    center = [float(value) for value in operation["center"]]
    if primitive == "box":
        half = [float(value) / 2.0 for value in operation["size"]]
    elif primitive == "sphere":
        half = [float(operation["radius"])] * 3
    elif primitive == "cylinder":
        radius = float(operation["radius"])
        half_height = float(operation["height"]) / 2.0
        axis = [float(value) for value in operation["axis"]]
        half = [
            abs(axis[index]) * half_height + radius * math.sqrt(max(0.0, 1.0 - axis[index] ** 2))
            for index in range(3)
        ]
    elif primitive == "hole":
        radius = float(operation["diameter"]) / 2.0
        half_depth = float(operation["depth"]) / 2.0
        half = {
            "XY": [radius, radius, half_depth],
            "XZ": [radius, half_depth, radius],
            "YZ": [half_depth, radius, radius],
        }[operation["workplane"]]
    elif primitive == "slot":
        radial = float(operation["length"]) / 2.0
        half_depth = float(operation["depth"]) / 2.0
        half = {
            "XY": [radial, radial, half_depth],
            "XZ": [radial, half_depth, radial],
            "YZ": [half_depth, radial, radial],
        }[operation["workplane"]]
    else:
        return None
    return (
        [center[index] - half[index] for index in range(3)],
        [center[index] + half[index] for index in range(3)],
    )


def _bbox_overlap(a: tuple[list[float], list[float]], b: tuple[list[float], list[float]]) -> bool:
    return all(a[0][index] < b[1][index] and b[0][index] < a[1][index] for index in range(3))


def _bbox_contains(outer: tuple[list[float], list[float]], inner: tuple[list[float], list[float]]) -> bool:
    return all(outer[0][index] <= inner[0][index] and outer[1][index] >= inner[1][index] for index in range(3))


def _bbox_union(a: tuple[list[float], list[float]], b: tuple[list[float], list[float]]) -> tuple[list[float], list[float]]:
    return (
        [min(a[0][index], b[0][index]) for index in range(3)],
        [max(a[1][index], b[1][index]) for index in range(3)],
    )


def _bbox_intersection(
    a: tuple[list[float], list[float]], b: tuple[list[float], list[float]]
) -> tuple[list[float], list[float]] | None:
    if not _bbox_overlap(a, b):
        return None
    return (
        [max(a[0][index], b[0][index]) for index in range(3)],
        [min(a[1][index], b[1][index]) for index in range(3)],
    )


def _warning(code: str, message: str, operation_index: int | None = None, operation_id: str | None = None) -> dict[str, Any]:
    return {
        "code": code,
        "severity": "warning",
        "message": message,
        "operationIndex": operation_index,
        "operationId": operation_id,
    }


def preflight_plan(plan: dict[str, Any]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    operations = plan.get("operations", [])
    approximate_result_bbox: tuple[list[float], list[float]] | None = None

    for index, operation in enumerate(operations):
        primitive_bbox = _primitive_bbox(operation)
        combine = operation.get("combine")
        operation_id = operation["id"]

        if index == 0:
            approximate_result_bbox = primitive_bbox
            if _operation_kind(operation) == "box" and not math.isclose(max(operation["size"]), 1.0, abs_tol=1e-9):
                warnings.append(
                    _warning(
                        "declared_scale_mismatch_likely",
                        "The first box already has a longest edge different from the declared unit-bbox value 1.0.",
                        index,
                        operation_id,
                    )
                )
            continue

        if approximate_result_bbox is None or primitive_bbox is None:
            continue
        overlaps = _bbox_overlap(approximate_result_bbox, primitive_bbox)

        if combine == "cut":
            if not overlaps:
                warnings.append(
                    _warning(
                        "ineffective_cut_likely",
                        "Cutter AABB does not overlap the current approximate result AABB.",
                        index,
                        operation_id,
                    )
                )
            elif _bbox_contains(primitive_bbox, approximate_result_bbox):
                warnings.append(
                    _warning(
                        "full_removal_risk",
                        "Cutter AABB contains the whole current result AABB; this operation may remove all material.",
                        index,
                        operation_id,
                    )
                )
        elif combine == "add":
            if not overlaps:
                warnings.append(
                    _warning(
                        "disconnected_add_likely",
                        "Added primitive AABB does not overlap the current result; multiple solids are likely.",
                        index,
                        operation_id,
                    )
                )
            approximate_result_bbox = _bbox_union(approximate_result_bbox, primitive_bbox)
        elif combine == "intersect":
            intersection = _bbox_intersection(approximate_result_bbox, primitive_bbox)
            if intersection is None:
                warnings.append(
                    _warning(
                        "empty_intersection_likely",
                        "Primitive AABB does not overlap the current result; intersection is expected to be empty.",
                        index,
                        operation_id,
                    )
                )
            else:
                approximate_result_bbox = intersection

    primitive_counts = Counter(_operation_kind(operation) for operation in operations)
    combine_counts = Counter(operation["combine"] for operation in operations if "combine" in operation)
    return {
        "warnings": warnings,
        "summary": {
            "operationCount": len(operations),
            "primitiveCounts": dict(primitive_counts),
            "combineCounts": dict(combine_counts),
            "warningCount": len(warnings),
        },
    }


def _validation_with_preflight(validation: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    if validation["valid"]:
        preflight = preflight_plan(plan)
    else:
        preflight = {"warnings": [], "summary": {"operationCount": 0, "primitiveCounts": {}, "combineCounts": {}, "warningCount": 0}}
    return {
        "valid": validation["valid"],
        "issues": validation["issues"],
        "warnings": preflight["warnings"],
        "planSummary": preflight["summary"],
    }


def _validate_schema_v2_for_episode(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate Plan schema v2 and attach Episode preflight data."""
    return _validation_with_preflight(validate_schema_plan_v2(plan), plan)


def _validate_plan_for_episode(plan: dict[str, Any]) -> dict[str, Any]:
    schema_version = plan.get("schema_version") if isinstance(plan, dict) else None
    if schema_version == "harnesscad.plan.v1":
        return _validation_with_preflight(validate_plan(plan), plan)
    if schema_version == PLAN_V2_SCHEMA_VERSION:
        return _validate_schema_v2_for_episode(plan)
    if schema_version == PLAN_V3_SCHEMA_VERSION:
        return _validation_with_preflight(validate_schema_plan_v3(plan), plan)
    if schema_version == PLAN_V31_SCHEMA_VERSION:
        return _validation_with_preflight(validate_schema_plan_v31(plan), plan)
    validation = {
        "valid": False,
        "issues": [
            {
                "code": "unsupported_schema_version",
                "path": "$.schema_version",
                "message": "Use 'harnesscad.plan.v1', 'harnesscad.plan.v2', 'harnesscad.plan.v3' or 'harnesscad.plan.v3.1'.",
                "severity": "error",
            }
        ],
    }
    return _validation_with_preflight(validation, plan)


def validate_plan_v2(plan: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible Episode v2 validator for either Plan schema version."""
    return _validate_plan_for_episode(plan)


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


def _write_trace():
    TRACE_PATH.write_text(json.dumps(TRACE, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _shape_metrics(shape):
    if shape is None:
        return {
            "shapeType": None,
            "valid": False,
            "solidCount": 0,
            "shellCount": 0,
            "faceCount": 0,
            "edgeCount": 0,
            "vertexCount": 0,
            "volume": 0.0,
            "area": 0.0,
            "bboxSize": None,
            "bboxCenter": None,
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


result = None
'''


def _compile_plan_v1_episode(plan: dict[str, Any]) -> str:
    lines = [
        '"""Generated deterministically by HarnessCAD runtime v2."""',
        RUNTIME_HELPERS.strip(),
        "",
    ]
    for index, operation in enumerate(plan["operations"]):
        operation_id = json.dumps(operation["id"])
        primitive = json.dumps(operation["primitive"])
        combine = json.dumps(operation["combine"])
        shape_name = f"shape_{index:03d}"
        lines.extend(
            [
                f"_op_index = {index}",
                f"_op_id = {operation_id}",
                f"_primitive = {primitive}",
                f"_combine = {combine}",
                "_before = _shape_metrics(result)",
                "_op_started = time.perf_counter()",
                "try:",
                f"    {shape_name} = {_shape_expression(operation)}",
            ]
        )
        if operation["combine"] == "new":
            lines.append(f"    result = {shape_name}")
        elif operation["combine"] == "add":
            lines.append(f"    result = result.fuse({shape_name})")
        elif operation["combine"] == "cut":
            lines.append(f"    result = result.cut({shape_name})")
        else:
            lines.append(f"    result = result.intersect({shape_name})")
        lines.extend(
            [
                "    _after = _shape_metrics(result)",
                "    _duration = time.perf_counter() - _op_started",
                "    _op_warnings = []",
                "    _delta = _after['volume'] - _before['volume']",
                "    if _after['solidCount'] > 1:",
                "        _op_warnings.append('multiple_solids_after_operation')",
                "    if _op_index > 0 and abs(_delta) <= 1e-12:",
                "        _op_warnings.append('ineffective_operation')",
                "    _record = {",
                "        'index': _op_index, 'id': _op_id, 'primitive': _primitive, 'combine': _combine,",
                "        'status': 'success' if not _op_warnings else 'success_with_warnings',",
                "        'durationSec': _duration, 'before': _before, 'after': _after,",
                "        'volumeDelta': _delta, 'warnings': _op_warnings,",
                "    }",
                "    TRACE['operations'].append(_record)",
                "    _write_trace()",
                "    if _after['solidCount'] == 0 or _after['volume'] <= 1e-12:",
                "        _record['status'] = 'failed'",
                "        _fail('empty_after_operation', 'operation', 'Boolean/primitive operation produced no solid material.', _op_index, _op_id)",
                "    if not _after['valid']:",
                "        _record['status'] = 'failed'",
                "        _fail('invalid_shape_after_operation', 'operation', 'OpenCascade reports an invalid shape.', _op_index, _op_id)",
                "except SystemExit:",
                "    raise",
                "except Exception as _exc:",
                "    TRACE['operations'].append({",
                "        'index': _op_index, 'id': _op_id, 'primitive': _primitive, 'combine': _combine,",
                "        'status': 'failed', 'durationSec': time.perf_counter() - _op_started,",
                "        'before': _before, 'after': None, 'volumeDelta': None, 'warnings': [],",
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


def compile_plan_v2(plan: dict[str, Any]) -> str:
    """Compile HarnessCAD Plan v2 while retaining this module's public API."""
    return compile_schema_plan_v2(plan)


def _compile_plan_for_episode(plan: dict[str, Any]) -> str:
    if plan.get("schema_version") == "harnesscad.plan.v1":
        return _compile_plan_v1_episode(plan)
    return compile_plan_v2(plan)


def _environment_snapshot() -> dict[str, Any]:
    return {
        "pythonExecutable": sys.executable,
        "pythonVersion": platform.python_version(),
        "platform": platform.platform(),
        "cadqueryVersion": importlib.metadata.version("cadquery"),
        "fastapiVersion": importlib.metadata.version("fastapi"),
    }


def _artifact_manifest(run_dir: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_file():
            continue
        data = path.read_bytes()
        manifest.append({"name": path.name, "bytes": len(data), "sha256": _sha256_bytes(data)})
    return manifest


@router.post("/validate")
def validate_endpoint(payload: PlanRequest) -> dict[str, Any]:
    return _validate_plan_for_episode(payload.plan)


@router.post("/run")
def run_endpoint(payload: RunRequest) -> dict[str, Any]:
    total_started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    run_dir = HARNESS_RUNS_V2_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    plan_path = run_dir / "input_plan.json"
    _write_json(plan_path, payload.plan)

    validation_started = time.perf_counter()
    validation = _validate_plan_for_episode(payload.plan)
    validation_duration = time.perf_counter() - validation_started
    _write_json(run_dir / "validation.json", validation)
    stages: list[dict[str, Any]] = [
        {"stage": "validation", "status": "success" if validation["valid"] else "failed", "durationSec": validation_duration}
    ]
    failure: dict[str, Any] | None = None
    generated_code: str | None = None
    runtime_trace: dict[str, Any] = {"operations": [], "stages": [], "warnings": [], "finalMetrics": None}

    if not validation["valid"]:
        status = "validation_failed"
        failure = {
            "code": "plan_validation_failed",
            "stage": "validation",
            "message": "Plan schema/semantic validation failed.",
            "operationIndex": None,
            "operationId": None,
        }
        return _finalize_response(
            run_id, run_dir, payload.plan, validation, stages, status, failure, generated_code, runtime_trace, total_started
        )

    compile_started = time.perf_counter()
    generated_code = _compile_plan_for_episode(payload.plan)
    generated_path = run_dir / "generated_model.py"
    generated_path.write_text(generated_code, encoding="utf-8")
    stages.append({"stage": "compile", "status": "success", "durationSec": time.perf_counter() - compile_started})

    step_path = run_dir / "result.step"
    stl_path = run_dir / "result.stl"
    runtime_trace_path = run_dir / "runtime_trace.json"
    execution_started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(generated_path), str(step_path), str(stl_path), str(runtime_trace_path)],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=payload.timeout_sec,
            check=False,
        )
        execution_duration = time.perf_counter() - execution_started
        _write_json(
            run_dir / "process.json",
            {
                "returncode": completed.returncode,
                "durationSec": execution_duration,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
        if runtime_trace_path.exists():
            runtime_trace = json.loads(runtime_trace_path.read_text(encoding="utf-8"))
        failure = runtime_trace.get("failure")
        status = runtime_trace.get("status") or ("execution_failed" if completed.returncode else "unknown")
        stages.append(
            {
                "stage": "execution",
                "status": "success" if completed.returncode == 0 else "failed",
                "durationSec": execution_duration,
                "returncode": completed.returncode,
            }
        )
    except subprocess.TimeoutExpired as exc:
        execution_duration = time.perf_counter() - execution_started
        status = "execution_timeout"
        failure = {
            "code": "execution_timeout",
            "stage": "execution",
            "message": f"Execution exceeded {payload.timeout_sec:g} seconds.",
            "operationIndex": None,
            "operationId": None,
        }
        stages.append({"stage": "execution", "status": "failed", "durationSec": execution_duration})
        _write_json(
            run_dir / "process.json",
            {"returncode": None, "durationSec": execution_duration, "stdout": exc.stdout or "", "stderr": exc.stderr or ""},
        )

    return _finalize_response(
        run_id, run_dir, payload.plan, validation, stages, status, failure, generated_code, runtime_trace, total_started
    )


def _finalize_response(
    run_id: str,
    run_dir: Path,
    plan: dict[str, Any],
    validation: dict[str, Any],
    stages: list[dict[str, Any]],
    status: str,
    failure: dict[str, Any] | None,
    generated_code: str | None,
    runtime_trace: dict[str, Any],
    total_started: float,
) -> dict[str, Any]:
    runtime_stages = runtime_trace.get("stages", [])
    all_stages = stages + runtime_stages
    warnings = validation.get("warnings", []) + runtime_trace.get("warnings", [])
    metrics = runtime_trace.get("finalMetrics")
    episode = {
        "traceVersion": TRACE_VERSION,
        "runId": run_id,
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "failure": failure,
        "warnings": warnings,
        "validation": validation,
        "planSummary": validation.get("planSummary"),
        "operationTrace": runtime_trace.get("operations", []),
        "stageTrace": all_stages,
        "metrics": metrics,
        "environment": _environment_snapshot(),
        "provenance": {
            "planSha256": _sha256_json(plan),
            "generatedCodeSha256": _sha256_bytes(generated_code.encode("utf-8")) if generated_code else None,
        },
        "totalDurationSec": time.perf_counter() - total_started,
    }
    _write_json(run_dir / "episode_v2.json", episode)
    manifest = _artifact_manifest(run_dir)
    _write_json(run_dir / "artifact_manifest.json", manifest)

    stl_path = run_dir / "result.stl"
    step_path = run_dir / "result.step"
    has_model = stl_path.exists() and stl_path.stat().st_size > 0
    response = {
        **episode,
        "modelUrl": f"/harness-runs-v2/{run_id}/result.stl" if has_model else None,
        "stepUrl": f"/harness-runs-v2/{run_id}/result.step" if step_path.exists() and step_path.stat().st_size > 0 else None,
        "generatedCode": generated_code,
        "artifactManifest": manifest,
        "error": failure.get("message") if failure else None,
    }
    return response
