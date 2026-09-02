from __future__ import annotations

import argparse
import math
import runpy
import time
import traceback
from pathlib import Path
from typing import Any

from .contract import CompileRequest, CompileResult, CompileSignal, read_json, write_json


class StepValidationError(ValueError):
    def __init__(self, code: str, message: str, validation: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.validation = validation or {}


def _failure(
    request: CompileRequest,
    code: str,
    message: str,
    *,
    detail: dict[str, Any] | None = None,
    timings_s: dict[str, float] | None = None,
) -> CompileResult:
    return CompileResult(
        status="failed",
        source_path=request.source_path,
        output_dir=request.output_dir,
        parameters=request.parameters,
        signals=[CompileSignal(code=code, message=message, detail=detail or {})],
        timings_s=timings_s or {},
    )


def _part_count(model: Any, cq: Any) -> int | None:
    if isinstance(model, cq.Assembly):
        objects = getattr(model, "objects", None)
        if isinstance(objects, dict):
            # CadQuery stores a synthetic root object in addition to user parts.
            return max(0, len(objects) - 1)
    if hasattr(model, "vals"):
        try:
            return len(model.vals())
        except Exception:
            return None
    return 1


def _export_step(cq: Any, model: Any, step_path: Path) -> None:
    temporary = step_path.with_name(f".{step_path.stem}.tmp.step")
    temporary.unlink(missing_ok=True)
    candidates = [model]
    if hasattr(model, "val"):
        try:
            candidates.append(model.val())
        except Exception:
            pass

    last_error: Exception | None = None
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        try:
            cq.exporters.export(candidate, str(temporary))
            if temporary.is_file() and temporary.stat().st_size > 0:
                temporary.replace(step_path)
                return
        except Exception as exc:
            last_error = exc
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"CadQuery could not export the model as STEP: {last_error}")


def _validate_step(cq: Any, step_path: Path, expected_parts: int | None) -> dict[str, Any]:
    try:
        imported = cq.importers.importStep(str(step_path))
    except Exception as exc:
        raise StepValidationError("step_import_failed", f"CadQuery could not import exported STEP: {exc}") from exc
    shapes = list(imported.vals()) if hasattr(imported, "vals") else [imported.val()]
    shapes = [shape for shape in shapes if shape is not None]
    if not shapes:
        raise StepValidationError("empty_step", "STEP readback produced no shapes")

    invalid_indices = [index for index, shape in enumerate(shapes) if not bool(shape.isValid())]
    solids = [solid for shape in shapes for solid in shape.Solids()]
    volume = float(sum(float(solid.Volume()) for solid in solids))

    compound = shapes[0] if len(shapes) == 1 else cq.Compound.makeCompound(shapes)
    bounds = compound.BoundingBox()
    size = [float(bounds.xlen), float(bounds.ylen), float(bounds.zlen)]
    validation = {
        "brep_valid": not invalid_indices,
        "shape_count": len(shapes),
        "solid_count": len(solids),
        "expected_part_count": expected_parts,
        "volume": volume,
        "bbox_size": size,
        "step_bytes": step_path.stat().st_size,
        "units": "mm",
    }
    if not all(math.isfinite(value) for value in size + [volume]):
        raise StepValidationError(
            "nonfinite_geometry",
            "STEP readback produced non-finite geometry metrics",
            validation,
        )
    if invalid_indices:
        raise StepValidationError(
            "invalid_brep",
            f"invalid B-Rep shapes at indices {invalid_indices}",
            validation,
        )
    if not solids:
        raise StepValidationError("empty_solid", "STEP readback contains no solids", validation)
    if volume <= 0.0:
        raise StepValidationError(
            "nonpositive_volume",
            "STEP readback has non-positive solid volume",
            validation,
        )
    if max(size) <= 0.0:
        raise StepValidationError(
            "zero_size_geometry",
            "STEP readback has a zero-size bounding box",
            validation,
        )
    return validation


def compile_request(request: CompileRequest) -> CompileResult:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    source_path = Path(request.source_path).resolve()
    output_dir = Path(request.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    step_path = output_dir / "result.step"
    step_path.unlink(missing_ok=True)

    if not source_path.is_file():
        return _failure(
            request,
            "source_not_found",
            f"source file does not exist: {source_path}",
        )

    try:
        import cadquery as cq
    except Exception as exc:
        return _failure(
            request,
            "dependency_unavailable",
            "CadQuery/OCP is unavailable in the worker environment",
            detail={"exception": f"{type(exc).__name__}: {exc}"},
        )

    try:
        load_started = time.perf_counter()
        namespace = runpy.run_path(str(source_path), run_name="__cad_agent_model__")
        timings["load_source"] = time.perf_counter() - load_started
    except Exception as exc:
        return _failure(
            request,
            "source_load_failed",
            f"failed to load generated source: {type(exc).__name__}: {exc}",
            detail={"traceback": traceback.format_exc(limit=12)},
            timings_s=timings,
        )

    build_model = namespace.get("build_model")
    if not callable(build_model):
        return _failure(
            request,
            "contract_missing",
            "generated source must define build_model(params)",
            timings_s=timings,
        )

    try:
        build_started = time.perf_counter()
        model = build_model(dict(request.parameters))
        timings["build_model"] = time.perf_counter() - build_started
        if model is None:
            raise TypeError("build_model(params) returned None")
    except Exception as exc:
        return _failure(
            request,
            "build_failed",
            f"build_model failed: {type(exc).__name__}: {exc}",
            detail={"traceback": traceback.format_exc(limit=12)},
            timings_s=timings,
        )

    expected_parts = _part_count(model, cq)
    try:
        export_started = time.perf_counter()
        _export_step(cq, model, step_path)
        timings["export_step"] = time.perf_counter() - export_started
    except Exception as exc:
        return _failure(
            request,
            "export_failed",
            f"STEP export failed: {type(exc).__name__}: {exc}",
            detail={"traceback": traceback.format_exc(limit=12)},
            timings_s=timings,
        )

    try:
        validate_started = time.perf_counter()
        validation = _validate_step(cq, step_path, expected_parts)
        timings["validate_step"] = time.perf_counter() - validate_started
    except StepValidationError as exc:
        result = _failure(
            request,
            exc.code,
            str(exc),
            detail={"traceback": traceback.format_exc(limit=12), "step_path": str(step_path)},
            timings_s=timings,
        )
        result.step_path = str(step_path)
        result.validation = exc.validation
        return result
    except Exception as exc:
        result = _failure(
            request,
            "step_readback_failed",
            f"STEP readback validation failed: {type(exc).__name__}: {exc}",
            detail={"traceback": traceback.format_exc(limit=12), "step_path": str(step_path)},
            timings_s=timings,
        )
        result.step_path = str(step_path)
        return result

    timings["total"] = time.perf_counter() - started
    return CompileResult(
        status="success",
        source_path=str(source_path),
        output_dir=str(output_dir),
        parameters=request.parameters,
        step_path=str(step_path),
        validation=validation,
        timings_s=timings,
    )


def _write_result(request: CompileRequest, response_path: Path, result: CompileResult) -> None:
    payload = result.to_dict()
    write_json(response_path, payload)
    write_json(Path(request.output_dir) / "manifest.json", payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile generated CadQuery code into a validated STEP file.")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args(argv)

    request = CompileRequest.from_dict(read_json(args.request))
    result = compile_request(request)
    _write_result(request, args.response, result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
