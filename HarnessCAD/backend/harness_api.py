"""Constrained Plan-Compile-Execute API for the local HarnessCAD demo."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cadquery as cq
from fastapi import APIRouter
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
HARNESS_RUNS_DIR = BASE_DIR / "harness_runs"
HARNESS_RUNS_DIR.mkdir(parents=True, exist_ok=True)
router = APIRouter(prefix="/api/harness", tags=["harness"])

SUPPORTED_PRIMITIVES = {"box", "cylinder", "sphere"}
SUPPORTED_COMBINE = {"new", "add", "cut", "intersect"}
ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
SAMPLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class HarnessPlanRequest(BaseModel):
    plan: dict[str, Any]


class HarnessRunRequest(HarnessPlanRequest):
    timeout_sec: float = Field(default=30.0, ge=1.0, le=120.0)


def _issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message, "severity": "error"}


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_vec3(
    value: object,
    path: str,
    issues: list[dict[str, str]],
    *,
    minimum: float = -2.0,
    maximum: float = 2.0,
) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        issues.append(_issue("invalid_vec3", path, "Expected an array of exactly three numbers."))
        return False
    valid = True
    for index, item in enumerate(value):
        if not _is_finite_number(item) or not minimum <= float(item) <= maximum:
            issues.append(
                _issue(
                    "invalid_vec3_component",
                    f"{path}[{index}]",
                    f"Expected a finite number in [{minimum}, {maximum}].",
                )
            )
            valid = False
    return valid


def _validate_positive(value: object, path: str, maximum: float, issues: list[dict[str, str]]) -> bool:
    if not _is_finite_number(value) or not 0.0 < float(value) <= maximum:
        issues.append(_issue("invalid_positive_number", path, f"Expected a number in (0, {maximum}]."))
        return False
    return True


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    root_allowed = {"schema_version", "sample_id", "coordinate_system", "metadata", "operations"}
    required = {"schema_version", "sample_id", "coordinate_system", "operations"}

    for key in sorted(required - set(plan)):
        issues.append(_issue("missing_field", f"$.{key}", "Required field is missing."))
    for key in sorted(set(plan) - root_allowed):
        issues.append(_issue("extra_field", f"$.{key}", "Field is not allowed by HarnessCAD Plan v1."))

    if plan.get("schema_version") != "harnesscad.plan.v1":
        issues.append(_issue("invalid_schema_version", "$.schema_version", "Expected 'harnesscad.plan.v1'."))

    sample_id = plan.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id or len(sample_id) > 128 or not SAMPLE_ID_PATTERN.fullmatch(sample_id):
        issues.append(_issue("invalid_sample_id", "$.sample_id", "Use 1-128 letters, digits, '.', '_' or '-'."))

    metadata = plan.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        issues.append(_issue("invalid_metadata", "$.metadata", "metadata must be a JSON object."))

    coordinate_system = plan.get("coordinate_system")
    if not isinstance(coordinate_system, dict):
        issues.append(_issue("invalid_coordinate_system", "$.coordinate_system", "Expected an object."))
    else:
        allowed = {"units", "origin", "longest_bbox_edge"}
        for key in sorted(set(coordinate_system) - allowed):
            issues.append(_issue("extra_field", f"$.coordinate_system.{key}", "Field is not allowed."))
        if coordinate_system.get("units") != "normalized":
            issues.append(_issue("invalid_units", "$.coordinate_system.units", "Expected 'normalized'."))
        origin = coordinate_system.get("origin")
        if _validate_vec3(origin, "$.coordinate_system.origin", issues) and any(float(x) != 0.0 for x in origin):
            issues.append(_issue("noncanonical_origin", "$.coordinate_system.origin", "Expected [0, 0, 0]."))
        edge = coordinate_system.get("longest_bbox_edge")
        if not _is_finite_number(edge) or not math.isclose(float(edge), 1.0, abs_tol=1e-12):
            issues.append(_issue("invalid_bbox_scale", "$.coordinate_system.longest_bbox_edge", "Expected 1.0."))

    operations = plan.get("operations")
    if not isinstance(operations, list) or not 1 <= len(operations) <= 64:
        issues.append(_issue("invalid_operations", "$.operations", "Expected 1-64 operations."))
        operations = []

    seen_ids: set[str] = set()
    common = {"id", "primitive", "combine", "center"}
    primitive_fields = {
        "box": {"size"},
        "cylinder": {"radius", "height", "axis"},
        "sphere": {"radius"},
    }

    for index, operation in enumerate(operations):
        path = f"$.operations[{index}]"
        if not isinstance(operation, dict):
            issues.append(_issue("invalid_operation", path, "Expected an object."))
            continue

        primitive = operation.get("primitive")
        if primitive not in SUPPORTED_PRIMITIVES:
            issues.append(_issue("unsupported_primitive", f"{path}.primitive", "Use box, cylinder or sphere."))
            allowed = common
        else:
            allowed = common | primitive_fields[primitive]
        for key in sorted(set(operation) - allowed):
            issues.append(_issue("extra_field", f"{path}.{key}", "Field is not allowed for this primitive."))
        for key in sorted(allowed - set(operation)):
            issues.append(_issue("missing_field", f"{path}.{key}", "Required field is missing."))

        operation_id = operation.get("id")
        if not isinstance(operation_id, str) or not ID_PATTERN.fullmatch(operation_id) or len(operation_id) > 64:
            issues.append(_issue("invalid_operation_id", f"{path}.id", "Use a letter followed by letters, digits or '_'."))
        elif operation_id in seen_ids:
            issues.append(_issue("duplicate_operation_id", f"{path}.id", "Operation id must be unique."))
        else:
            seen_ids.add(operation_id)

        combine = operation.get("combine")
        if combine not in SUPPORTED_COMBINE:
            issues.append(_issue("unsupported_combine", f"{path}.combine", "Use new, add, cut or intersect."))
        elif index == 0 and combine != "new":
            issues.append(_issue("first_operation_must_be_new", f"{path}.combine", "The first operation must be new."))
        elif index > 0 and combine == "new":
            issues.append(_issue("new_only_allowed_first", f"{path}.combine", "Only the first operation may be new."))

        _validate_vec3(operation.get("center"), f"{path}.center", issues)
        if primitive == "box":
            size = operation.get("size")
            if not isinstance(size, list) or len(size) != 3:
                issues.append(_issue("invalid_size", f"{path}.size", "Expected three positive dimensions."))
            else:
                for dim_index, value in enumerate(size):
                    _validate_positive(value, f"{path}.size[{dim_index}]", 4.0, issues)
        elif primitive == "sphere":
            _validate_positive(operation.get("radius"), f"{path}.radius", 2.0, issues)
        elif primitive == "cylinder":
            _validate_positive(operation.get("radius"), f"{path}.radius", 2.0, issues)
            _validate_positive(operation.get("height"), f"{path}.height", 4.0, issues)
            axis = operation.get("axis")
            if _validate_vec3(axis, f"{path}.axis", issues, minimum=-1.0, maximum=1.0):
                norm = math.sqrt(sum(float(value) ** 2 for value in axis))
                if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
                    issues.append(_issue("axis_not_unit_length", f"{path}.axis", f"Axis norm is {norm:.8g}, expected 1."))

    return {"valid": not issues, "issues": issues}


def _number(value: int | float) -> str:
    return json.dumps(float(value), allow_nan=False)


def _vector(values: list[int | float]) -> str:
    return "cq.Vector(" + ", ".join(_number(value) for value in values) + ")"


def _shape_expression(operation: dict[str, Any]) -> str:
    primitive = operation["primitive"]
    center = operation["center"]
    if primitive == "box":
        size = operation["size"]
        return (
            "cq.Workplane(\"XY\")"
            f".box({_number(size[0])}, {_number(size[1])}, {_number(size[2])}, centered=(True, True, True))"
            f".val().translate({_vector(center)})"
        )
    if primitive == "sphere":
        return f"cq.Solid.makeSphere({_number(operation['radius'])}, {_vector(center)})"
    radius = operation["radius"]
    height = operation["height"]
    axis = operation["axis"]
    start = [float(center[i]) - 0.5 * float(height) * float(axis[i]) for i in range(3)]
    return (
        f"cq.Solid.makeCylinder({_number(radius)}, {_number(height)}, "
        f"{_vector(start)}, {_vector(axis)})"
    )


def compile_plan(plan: dict[str, Any]) -> str:
    lines = [
        '"""Generated deterministically by HarnessCAD Plan v1."""',
        "import sys",
        "import cadquery as cq",
        "",
    ]
    for index, operation in enumerate(plan["operations"]):
        shape_name = f"shape_{index:03d}"
        lines.append(f"{shape_name} = {_shape_expression(operation)}")
        combine = operation["combine"]
        if combine == "new":
            lines.append(f"result = {shape_name}")
        elif combine == "add":
            lines.append(f"result = result.fuse({shape_name})")
        elif combine == "cut":
            lines.append(f"result = result.cut({shape_name})")
        else:
            lines.append(f"result = result.intersect({shape_name})")
        lines.append("")
    lines.extend(
        [
            "if __name__ == \"__main__\":",
            "    if len(sys.argv) != 3:",
            '        raise SystemExit("usage: generated_model.py OUTPUT.step OUTPUT.stl")',
            "    if result.isNull() or not result.isValid():",
            '        raise RuntimeError("CAD result is null or invalid")',
            "    cq.exporters.export(result, sys.argv[1])",
            "    cq.exporters.export(result, sys.argv[2])",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@router.post("/validate")
def validate_harness_plan(payload: HarnessPlanRequest) -> dict[str, Any]:
    return validate_plan(payload.plan)


@router.post("/run")
def run_harness_plan(payload: HarnessRunRequest) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    run_dir = HARNESS_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "input_plan.json", payload.plan)

    validation = validate_plan(payload.plan)
    _write_json(run_dir / "validation.json", validation)
    episode: dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "validation_failed" if not validation["valid"] else "validated",
        "timeout_sec": payload.timeout_sec,
        "executor_python": sys.executable,
    }
    if not validation["valid"]:
        _write_json(run_dir / "episode.json", episode)
        return {
            "runId": run_id,
            "status": "validation_failed",
            "validation": validation,
            "modelUrl": None,
            "stepUrl": None,
            "generatedCode": None,
            "metrics": None,
            "error": "Plan validation failed.",
        }

    generated_code = compile_plan(payload.plan)
    generated_path = run_dir / "generated_model.py"
    generated_path.write_text(generated_code, encoding="utf-8")
    step_path = run_dir / "result.step"
    stl_path = run_dir / "result.stl"

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(generated_path), str(step_path), str(stl_path)],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=payload.timeout_sec,
            check=False,
        )
        runtime_sec = time.perf_counter() - started
    except subprocess.TimeoutExpired as exc:
        execution = {
            "status": "timeout",
            "runtime_sec": time.perf_counter() - started,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
        _write_json(run_dir / "execution.json", execution)
        episode["status"] = "execution_timeout"
        _write_json(run_dir / "episode.json", episode)
        return {
            "runId": run_id,
            "status": "execution_timeout",
            "validation": validation,
            "modelUrl": None,
            "stepUrl": None,
            "generatedCode": generated_code,
            "metrics": None,
            "error": f"Execution exceeded {payload.timeout_sec:g} seconds.",
        }

    execution = {
        "status": "success" if completed.returncode == 0 and step_path.exists() and stl_path.exists() else "execution_failed",
        "returncode": completed.returncode,
        "runtime_sec": runtime_sec,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "step_bytes": step_path.stat().st_size if step_path.exists() else 0,
        "stl_bytes": stl_path.stat().st_size if stl_path.exists() else 0,
    }
    _write_json(run_dir / "execution.json", execution)
    if execution["status"] != "success":
        episode["status"] = "execution_failed"
        _write_json(run_dir / "episode.json", episode)
        return {
            "runId": run_id,
            "status": "execution_failed",
            "validation": validation,
            "modelUrl": None,
            "stepUrl": None,
            "generatedCode": generated_code,
            "metrics": None,
            "error": completed.stderr.strip() or "CadQuery execution failed.",
        }

    imported = cq.importers.importStep(str(step_path)).val()
    bbox = imported.BoundingBox()
    center = [
        (bbox.xmin + bbox.xmax) / 2.0,
        (bbox.ymin + bbox.ymax) / 2.0,
        (bbox.zmin + bbox.zmax) / 2.0,
    ]
    bbox_size = [bbox.xlen, bbox.ylen, bbox.zlen]
    canonical = math.isclose(max(bbox_size), 1.0, abs_tol=1e-6) and all(abs(value) <= 1e-6 for value in center)
    metrics = {
        "validShape": bool(imported.isValid()),
        "volume": float(imported.Volume()),
        "bboxSize": bbox_size,
        "bboxCenter": center,
        "canonicalFrame": canonical,
        "runtimeSec": runtime_sec,
    }
    _write_json(run_dir / "metrics.json", metrics)
    episode["status"] = "success"
    episode["metrics"] = metrics
    _write_json(run_dir / "episode.json", episode)

    return {
        "runId": run_id,
        "status": "success",
        "validation": validation,
        "modelUrl": f"/harness-runs/{run_id}/result.stl",
        "stepUrl": f"/harness-runs/{run_id}/result.step",
        "generatedCode": generated_code,
        "metrics": metrics,
        "error": None,
    }
