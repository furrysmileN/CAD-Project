#!/usr/bin/env python3
"""Prepare BenchCAD code_gen data for modified_cadrille Qwen JSONL training.

The pipeline is intentionally split into resumable stages:

1. materialize: download BenchCAD parquet shards and write one CadQuery .py file
   per sample.
2. export: execute each .py and export STEP plus STL.
3. assets: call build_step_assets.py to create point clouds and renders.
4. build-index: write the JSONL format consumed by modified_cadrille/qwen3vl_data.py.
5. validate: perform lightweight path and dataset-loader checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np
import trimesh
from tqdm import tqdm


DEFAULT_REPO_ID = "BenchCAD/BenchCAD"
DEFAULT_CONFIG = "code_gen"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/benchcad_codegen_qwen")
MATERIALIZED_MANIFEST = Path("index/materialized_manifest.jsonl")
EXPORT_MANIFEST = Path("index/export_manifest.jsonl")
EXPORT_FAILURES = Path("index/export_failures.jsonl")
BENCHCAD_INDEX = Path("index/benchcad_index.jsonl")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _rel(path: Path, base_dir: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), base_dir.resolve())
    except ValueError:
        return str(path.resolve())


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _sanitize(value: Any, fallback: str = "sample", max_len: int = 80) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    text = text.strip("._-") or fallback
    return text[:max_len]


def _stable_sample_id(row: dict[str, Any], global_index: int) -> str:
    parts = [
        row.get("stem"),
        row.get("family"),
        row.get("variant"),
        row.get("difficulty"),
        row.get("base_plane"),
        row.get("standard"),
    ]
    seed = "|".join(str(part) for part in parts if part not in (None, ""))
    digest = hashlib.sha1(
        (seed + "|" + str(global_index)).encode("utf-8", errors="replace")
    ).hexdigest()[:8]
    stem = _sanitize(row.get("stem") or row.get("family") or digest, fallback=digest)
    return f"benchcad_{global_index:06d}_{stem}_{digest}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _require_huggingface_hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "huggingface_hub is required. Install it with `pip install huggingface_hub`."
        ) from exc
    return HfApi, hf_hub_download


def _set_hf_endpoint(endpoint: Optional[str]) -> None:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")


def _list_parquet_files(repo_id: str, config: str, revision: str) -> list[str]:
    HfApi, _ = _require_huggingface_hub()
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    prefix = f"{config}/"
    candidates = [
        name
        for name in files
        if name.startswith(prefix) and name.lower().endswith(".parquet")
    ]
    candidates.sort()
    if not candidates:
        raise RuntimeError(f"No parquet files found under {prefix!r} in {repo_id}.")
    return candidates


def _download_parquet_files(
    *,
    repo_id: str,
    config: str,
    revision: str,
    hf_endpoint: Optional[str],
    output_root: Path,
) -> list[Path]:
    _set_hf_endpoint(hf_endpoint)
    _, hf_hub_download = _require_huggingface_hub()
    filenames = _list_parquet_files(repo_id, config, revision)
    local_root = output_root / "hf"
    paths: list[Path] = []
    for filename in tqdm(filenames, desc="download parquet"):
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=str(local_root),
        )
        paths.append(Path(path))
    return paths


def _local_parquet_files(local_data_root: Path, config: str) -> list[Path]:
    root = local_data_root.expanduser().resolve()
    candidates: list[Path] = []
    for directory in (root / config / "data", root / config, root):
        if directory.is_dir():
            candidates = sorted(directory.glob("*.parquet"))
            if candidates:
                break
    if not candidates:
        raise RuntimeError(
            f"No local parquet files found for config={config!r} under {root}"
        )
    return candidates


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except Exception:
        return []
    return list(pq.ParquetFile(path).schema_arrow.names)


def _iter_parquet_rows(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    wanted = ["stem", "family", "variant", "difficulty", "base_plane", "standard", "code"]
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        columns = [name for name in wanted if name in parquet_file.schema_arrow.names]
        if "code" not in columns:
            raise RuntimeError(f"Parquet file lacks required `code` column: {path}")
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                yield dict(row)
        return
    except ModuleNotFoundError:
        pass

    try:
        import pandas as pd

        columns = _parquet_columns(path) or wanted
        columns = [name for name in wanted if name in columns]
        frame = pd.read_parquet(path, columns=columns)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Unable to read parquet. Install a parquet backend such as `pyarrow` "
            "or install `datasets` and rerun the materialize stage."
        ) from exc
    if "code" not in frame.columns:
        raise RuntimeError(f"Parquet file lacks required `code` column: {path}")
    for row in frame.to_dict(orient="records"):
        yield dict(row)


def _materialized_record(
    *,
    output_root: Path,
    sample_id: str,
    source_file: Path,
    source_file_index: int,
    source_row: int,
    global_index: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    code = str(row.get("code") or "")
    code_path = output_root / "cadquery" / f"{sample_id}.py"
    if not code.endswith("\n"):
        code += "\n"
    code_path.parent.mkdir(parents=True, exist_ok=True)
    if not _nonempty_file(code_path):
        code_path.write_text(code, encoding="utf-8")

    return {
        "sample_id": sample_id,
        "source_index": int(global_index),
        "source_file_index": int(source_file_index),
        "source_row": int(source_row),
        "source_file": str(source_file),
        "cadquery_path": _rel(code_path, output_root),
        "code_sha256": _sha256_text(code),
        "stem": row.get("stem"),
        "family": row.get("family"),
        "variant": row.get("variant"),
        "difficulty": row.get("difficulty"),
        "base_plane": row.get("base_plane"),
        "standard": row.get("standard"),
    }


def materialize_benchcad(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / MATERIALIZED_MANIFEST
    temp_manifest = manifest_path.with_suffix(".jsonl.tmp")
    temp_manifest.unlink(missing_ok=True)

    if args.local_data_root is not None:
        parquet_paths = _local_parquet_files(args.local_data_root, args.config)
    else:
        parquet_paths = _download_parquet_files(
            repo_id=args.repo_id,
            config=args.config,
            revision=args.revision,
            hf_endpoint=args.hf_endpoint,
            output_root=output_root,
        )

    global_index = 0
    written = 0
    started = time.time()
    for file_index, parquet_path in enumerate(parquet_paths):
        for source_row, row in enumerate(_iter_parquet_rows(parquet_path, args.batch_size)):
            if global_index < args.start_index:
                global_index += 1
                continue
            if args.max_samples is not None and written >= args.max_samples:
                break
            code = row.get("code")
            if not isinstance(code, str) or not code.strip():
                global_index += 1
                continue
            sample_id = _stable_sample_id(row, global_index)
            record = _materialized_record(
                output_root=output_root,
                sample_id=sample_id,
                source_file=parquet_path,
                source_file_index=file_index,
                source_row=source_row,
                global_index=global_index,
                row=row,
            )
            _append_jsonl(temp_manifest, record)
            written += 1
            global_index += 1
        if args.max_samples is not None and written >= args.max_samples:
            break

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_manifest.replace(manifest_path)
    summary = {
        "stage": "materialize",
        "repo_id": args.repo_id,
        "config": args.config,
        "revision": args.revision,
        "local_data_root": str(args.local_data_root) if args.local_data_root else None,
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "num_records": written,
        "elapsed_s": round(time.time() - started, 2),
    }
    _write_json(output_root / "index/materialize_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


@dataclass(frozen=True)
class ExportJob:
    sample_id: str
    code_path: str
    step_path: str
    mesh_path: str
    output_root: str
    force: bool
    triangle_face_tol: float
    angle_tol_rads: float
    timeout_s: Optional[int]


def _shape_candidates(obj: Any) -> list[Any]:
    candidates: list[Any] = []
    if hasattr(obj, "vals"):
        try:
            candidates.extend(list(obj.vals()))
        except Exception:
            pass
    if hasattr(obj, "val"):
        try:
            candidates.append(obj.val())
        except Exception:
            pass
    if hasattr(obj, "wrapped"):
        try:
            candidates.append(obj.wrapped)
        except Exception:
            pass
    candidates.append(obj)

    unique: list[Any] = []
    seen: set[int] = set()
    for item in candidates:
        ident = id(item)
        if item is not None and ident not in seen:
            unique.append(item)
            seen.add(ident)
    return unique


def _export_step(cq_module: Any, cad_obj: Any, step_path: Path) -> None:
    step_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = step_path.with_name(f".{step_path.stem}.tmp.step")
    tmp_path.unlink(missing_ok=True)
    last_exc: Optional[BaseException] = None
    for candidate in _shape_candidates(cad_obj):
        try:
            cq_module.exporters.export(candidate, str(tmp_path))
            if _nonempty_file(tmp_path):
                tmp_path.replace(step_path)
                return
        except BaseException as exc:  # noqa: BLE001
            last_exc = exc
            tmp_path.unlink(missing_ok=True)
    raise RuntimeError(f"STEP export failed: {last_exc}")


def _mesh_from_cad_object(cad_obj: Any, triangle_face_tol: float, angle_tol_rads: float) -> trimesh.Trimesh:
    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0
    last_exc: Optional[BaseException] = None
    for shape in _shape_candidates(cad_obj):
        if not hasattr(shape, "tessellate"):
            continue
        try:
            try:
                vertices, faces = shape.tessellate(
                    triangle_face_tol,
                    angularTolerance=angle_tol_rads,
                )
            except TypeError:
                vertices, faces = shape.tessellate(triangle_face_tol, angle_tol_rads)
            if not vertices or not faces:
                continue
            verts = np.asarray([(v.x, v.y, v.z) for v in vertices], dtype=np.float64)
            tris = np.asarray(faces, dtype=np.int64) + vertex_offset
            all_vertices.append(verts)
            all_faces.append(tris)
            vertex_offset += len(verts)
        except BaseException as exc:  # noqa: BLE001
            last_exc = exc
    if not all_vertices or not all_faces:
        raise RuntimeError(f"empty tessellation: {last_exc}")
    mesh = trimesh.Trimesh(
        vertices=np.concatenate(all_vertices, axis=0),
        faces=np.concatenate(all_faces, axis=0),
        process=False,
    )
    if mesh.is_empty or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("empty mesh after tessellation")
    return mesh


def _export_mesh(cad_obj: Any, mesh_path: Path, triangle_face_tol: float, angle_tol_rads: float) -> None:
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = mesh_path.with_name(f".{mesh_path.stem}.tmp.stl")
    tmp_path.unlink(missing_ok=True)
    mesh = _mesh_from_cad_object(cad_obj, triangle_face_tol, angle_tol_rads)
    mesh.export(tmp_path)
    if not _nonempty_file(tmp_path):
        raise RuntimeError(f"STL export produced empty file: {tmp_path}")
    tmp_path.replace(mesh_path)


def _run_export_job(job: ExportJob) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    sample_id = job.sample_id
    code_path = Path(job.code_path)
    step_path = Path(job.step_path)
    mesh_path = Path(job.mesh_path)
    output_root = Path(job.output_root)

    def _on_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise TimeoutError(f"export timed out after {job.timeout_s}s")

    if (
        not job.force
        and _nonempty_file(step_path)
        and _nonempty_file(mesh_path)
    ):
        return (
            {
                "sample_id": sample_id,
                "status": "skipped_existing",
                "cadquery_path": _rel(code_path, output_root),
                "step_path": _rel(step_path, output_root),
                "mesh_path": _rel(mesh_path, output_root),
            },
            None,
        )

    try:
        previous_handler = None
        if job.timeout_s is not None and job.timeout_s > 0:
            previous_handler = signal.signal(signal.SIGALRM, _on_timeout)
            signal.alarm(int(job.timeout_s))
        import cadquery as cq

        shown_objects: list[Any] = []

        def show_object(obj: Any, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            shown_objects.append(obj)
            return obj

        namespace: dict[str, Any] = {
            "__name__": "__benchcad_sample__",
            "__file__": str(code_path),
            "show_object": show_object,
        }
        source = code_path.read_text(encoding="utf-8")
        exec(compile(source, str(code_path), "exec"), namespace)  # noqa: S102
        cad_obj = None
        for name in ("r", "result", "model", "shape", "solid", "part"):
            if name in namespace:
                cad_obj = namespace[name]
                break
        if cad_obj is None and shown_objects:
            cad_obj = shown_objects[-1]
        if cad_obj is None:
            raise RuntimeError(
                "CadQuery code did not define any supported output variable "
                "(`r`, `result`, `model`, `shape`, `solid`, `part`) and did not call show_object()."
            )
        _export_step(cq, cad_obj, step_path)
        _export_mesh(cad_obj, mesh_path, job.triangle_face_tol, job.angle_tol_rads)
        if job.timeout_s is not None and job.timeout_s > 0:
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
        return (
            {
                "sample_id": sample_id,
                "status": "success",
                "cadquery_path": _rel(code_path, output_root),
                "step_path": _rel(step_path, output_root),
                "mesh_path": _rel(mesh_path, output_root),
                "step_size": step_path.stat().st_size,
                "mesh_size": mesh_path.stat().st_size,
            },
            None,
        )
    except BaseException as exc:  # noqa: BLE001
        if job.timeout_s is not None and job.timeout_s > 0:
            signal.alarm(0)
            if "previous_handler" in locals() and previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
        step_path.unlink(missing_ok=True)
        mesh_path.unlink(missing_ok=True)
        return (
            None,
            {
                "sample_id": sample_id,
                "stage": "export",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "cadquery_path": _rel(code_path, output_root),
                "traceback": traceback.format_exc(limit=6),
            },
        )


def _load_materialized_records(output_root: Path, manifest: Optional[Path]) -> list[dict[str, Any]]:
    path = manifest or (output_root / MATERIALIZED_MANIFEST)
    path = path if path.is_absolute() else output_root / path
    records = list(_iter_jsonl(path))
    if not records:
        raise RuntimeError(f"No materialized records found in {path}")
    return records


def export_cadquery(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    records = _load_materialized_records(output_root, args.manifest)
    export_manifest = output_root / EXPORT_MANIFEST
    failures_path = output_root / EXPORT_FAILURES
    if args.rewrite_manifest:
        export_manifest.unlink(missing_ok=True)
        failures_path.unlink(missing_ok=True)

    jobs: list[ExportJob] = []
    for rec in records:
        sample_id = str(rec["sample_id"])
        code_path = output_root / str(rec["cadquery_path"])
        jobs.append(
            ExportJob(
                sample_id=sample_id,
                code_path=str(code_path),
                step_path=str(output_root / "step" / f"{sample_id}.step"),
                mesh_path=str(output_root / "mesh" / f"{sample_id}.stl"),
                output_root=str(output_root),
                force=bool(args.force),
                triangle_face_tol=float(args.triangle_face_tol),
                angle_tol_rads=float(args.angle_tol_rads),
                timeout_s=args.export_timeout_s,
            )
        )

    started = time.time()
    ok = 0
    failed = 0

    def consume(result: tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]) -> None:
        nonlocal ok, failed
        record, error = result
        if record is not None:
            _append_jsonl(export_manifest, record)
            ok += 1
        if error is not None:
            _append_jsonl(failures_path, error)
            failed += 1

    if args.num_processes <= 1:
        for job in tqdm(jobs, desc="export cadquery"):
            consume(_run_export_job(job))
    else:
        with Pool(processes=int(args.num_processes)) as pool:
            for result in tqdm(
                pool.imap_unordered(_run_export_job, jobs),
                total=len(jobs),
                desc="export cadquery",
            ):
                consume(result)

    summary = {
        "stage": "export",
        "output_root": str(output_root),
        "n_inputs": len(jobs),
        "n_ok": ok,
        "n_failed": failed,
        "manifest": str(export_manifest),
        "failures": str(failures_path),
        "elapsed_s": round(time.time() - started, 2),
    }
    _write_json(output_root / "index/export_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def refresh_export_manifest(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    records = _load_materialized_records(output_root, args.manifest)
    export_manifest = output_root / EXPORT_MANIFEST
    failures_path = output_root / EXPORT_FAILURES
    export_manifest.unlink(missing_ok=True)
    failures_path.unlink(missing_ok=True)

    ok = 0
    failed = 0
    for rec in records:
        sample_id = str(rec["sample_id"])
        code_path = output_root / str(rec["cadquery_path"])
        step_path = output_root / "step" / f"{sample_id}.step"
        mesh_path = output_root / "mesh" / f"{sample_id}.stl"
        if _nonempty_file(step_path) and _nonempty_file(mesh_path):
            _append_jsonl(
                export_manifest,
                {
                    "sample_id": sample_id,
                    "status": "success",
                    "cadquery_path": _rel(code_path, output_root),
                    "step_path": _rel(step_path, output_root),
                    "mesh_path": _rel(mesh_path, output_root),
                    "step_size": step_path.stat().st_size,
                    "mesh_size": mesh_path.stat().st_size,
                },
            )
            ok += 1
        else:
            _append_jsonl(
                failures_path,
                {
                    "sample_id": sample_id,
                    "stage": "export",
                    "error_type": "MissingOutput",
                    "error": "STEP or STL output missing after export stage",
                    "cadquery_path": _rel(code_path, output_root),
                    "step_path": _rel(step_path, output_root),
                    "mesh_path": _rel(mesh_path, output_root),
                },
            )
            failed += 1

    summary = {
        "stage": "refresh-export-manifest",
        "output_root": str(output_root),
        "n_inputs": len(records),
        "n_ok": ok,
        "n_failed": failed,
        "manifest": str(export_manifest),
        "failures": str(failures_path),
    }
    _write_json(output_root / "index/export_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def run_assets(args: argparse.Namespace) -> int:
    from . import build_step_assets

    output_root = args.output_root.resolve()
    step_dir = output_root / "step"
    if not step_dir.is_dir():
        raise RuntimeError(f"STEP directory does not exist: {step_dir}")
    argv = [
        "--input-dir",
        str(step_dir),
        "--output-dir",
        str(output_root),
        "--num-points",
        str(args.num_points),
        "--num-views",
        str(args.num_views),
        "--img-size",
        str(args.img_size),
        "--render-backend",
        str(args.render_backend),
        "--step-mesh-backend",
        str(args.step_mesh_backend),
        "--step-mesh-fallback-backends",
        str(args.step_mesh_fallback_backends),
        "--num-processes",
        str(args.num_processes),
        "--triangle-face-tol",
        str(args.triangle_face_tol),
        "--angle-tol-rads",
        str(args.angle_tol_rads),
    ]
    if args.blender_bin:
        argv.extend(["--blender-bin", str(args.blender_bin)])
    if args.blender_script:
        argv.extend(["--blender-script", str(args.blender_script)])
    if args.blender_engine:
        argv.extend(["--blender-engine", str(args.blender_engine)])
    if args.blender_samples is not None:
        argv.extend(["--blender-samples", str(args.blender_samples)])
    if args.blender_style:
        argv.extend(["--blender-style", str(args.blender_style)])
    if args.blender_device:
        argv.extend(["--blender-device", str(args.blender_device)])
    if args.visualization_root:
        argv.extend(["--visualization-root", str(args.visualization_root)])
    if args.freecad_cmd:
        argv.extend(["--freecad-cmd", str(args.freecad_cmd)])
    if args.step_mesh_timeout_s is not None:
        argv.extend(["--step-mesh-timeout-s", str(args.step_mesh_timeout_s)])
    if args.skip_existing:
        argv.append("--skip-existing")
    if args.resume_manifest:
        argv.append("--resume-manifest")
    if args.incremental_manifest:
        argv.append("--incremental-manifest")
    if getattr(args, "max_tasks", None) is not None:
        argv.extend(["--max-tasks", str(args.max_tasks)])
    return int(build_step_assets.main(argv))


def _export_success_by_sample(output_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for rec in _iter_jsonl(output_root / EXPORT_MANIFEST):
        status = rec.get("status")
        if status not in {"success", "skipped_existing"}:
            continue
        sample_id = str(rec.get("sample_id"))
        step_path = output_root / str(rec.get("step_path", ""))
        mesh_path = output_root / str(rec.get("mesh_path", ""))
        if _nonempty_file(step_path) and _nonempty_file(mesh_path):
            records[sample_id] = rec
    return records


def _split_for_sample(sample_id: str, val_ratio: float) -> str:
    ratio = max(0.0, min(float(val_ratio), 0.9))
    if ratio <= 0.0:
        return "train"
    bucket = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < ratio else "train"


def build_jsonl_index(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    materialized = {
        str(rec["sample_id"]): rec
        for rec in _load_materialized_records(output_root, args.materialized_manifest)
    }
    exported = _export_success_by_sample(output_root)
    output_path = args.output_path or (output_root / BENCHCAD_INDEX)
    output_path = output_path if output_path.is_absolute() else output_root / output_path
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for sample_id in sorted(exported):
        src = materialized.get(sample_id, {})
        exp = exported[sample_id]
        code_path = output_root / str(exp["cadquery_path"])
        mesh_path = output_root / str(exp["mesh_path"])
        step_path = output_root / str(exp["step_path"])
        if not (_nonempty_file(code_path) and _nonempty_file(mesh_path)):
            continue
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "split": _split_for_sample(sample_id, args.val_ratio),
            "cadquery_path": _rel(code_path, output_dir),
            "mesh_path": _rel(mesh_path, output_dir),
            "step_path": _rel(step_path, output_dir),
            "source_index": src.get("source_index"),
            "family": src.get("family"),
            "variant": src.get("variant"),
            "difficulty": src.get("difficulty"),
        }
        point_path = output_root / "points" / f"{sample_id}.npz"
        if _nonempty_file(point_path):
            record["pointcloud_path"] = _rel(point_path, output_dir)
        image_dir = output_root / "images" / sample_id
        image_paths = sorted(
            path
            for path in image_dir.glob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and _nonempty_file(path)
        )
        if image_paths:
            record["image_paths"] = [_rel(path, output_dir) for path in image_paths[: args.max_images]]
        if args.require_pointcloud and "pointcloud_path" not in record:
            continue
        if args.require_images and "image_paths" not in record:
            continue
        records.append(record)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "stage": "build-index",
        "output_path": str(output_path),
        "num_records": len(records),
        "num_train": sum(1 for rec in records if rec["split"] == "train"),
        "num_val": sum(1 for rec in records if rec["split"] == "val"),
        "num_with_pointcloud": sum(1 for rec in records if rec.get("pointcloud_path")),
        "num_with_images": sum(1 for rec in records if rec.get("image_paths")),
        "val_ratio": args.val_ratio,
    }
    _write_json(output_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def _resolve_record_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def validate_index(args: argparse.Namespace) -> dict[str, Any]:
    index_path = args.index_path
    output_root = args.output_root.resolve()
    if index_path is None:
        index_path = output_root / BENCHCAD_INDEX
    index_path = index_path if index_path.is_absolute() else output_root / index_path
    base_dir = index_path.parent
    records = list(_iter_jsonl(index_path))
    if not records:
        raise RuntimeError(f"No records found in index: {index_path}")

    missing: list[dict[str, Any]] = []
    checked_npz = 0
    checked_images = 0
    for rec in records[: args.max_path_checks]:
        sample_id = rec.get("sample_id")
        for key in ("cadquery_path", "mesh_path"):
            path = _resolve_record_path(str(rec.get(key, "")), base_dir)
            if not _nonempty_file(path):
                missing.append({"sample_id": sample_id, "field": key, "path": str(path)})
        if rec.get("pointcloud_path"):
            path = _resolve_record_path(str(rec["pointcloud_path"]), base_dir)
            if not _nonempty_file(path):
                missing.append({"sample_id": sample_id, "field": "pointcloud_path", "path": str(path)})
            else:
                data = np.load(path)
                if "points" not in data.files:
                    missing.append({"sample_id": sample_id, "field": "pointcloud_points", "path": str(path)})
                checked_npz += 1
        for image in rec.get("image_paths") or []:
            path = _resolve_record_path(str(image), base_dir)
            if not _nonempty_file(path):
                missing.append({"sample_id": sample_id, "field": "image_paths", "path": str(path)})
            checked_images += 1

    dataset_ok = False
    dataset_error: Optional[str] = None
    try:
        modified_root = args.modified_cadrille_root.resolve()
        sys.path.insert(0, str(modified_root))
        from qwen3vl_data import BenchCADIndexedDataset, build_pointcloud_adapter

        adapter = build_pointcloud_adapter("serialize", num_points=args.dataset_num_points, precision=4)
        for split in ("train", "val"):
            ds = BenchCADIndexedDataset(
                str(index_path),
                split,
                args.prompt,
                adapter,
                max_images=args.max_images,
            )
            if len(ds) > 0:
                _ = ds[0]
        dataset_ok = True
    except BaseException as exc:  # noqa: BLE001
        dataset_error = f"{type(exc).__name__}: {exc}"

    summary = {
        "stage": "validate",
        "index_path": str(index_path),
        "num_records": len(records),
        "path_check_limit": min(len(records), args.max_path_checks),
        "num_missing": len(missing),
        "missing_examples": missing[:20],
        "checked_npz": checked_npz,
        "checked_images": checked_images,
        "dataset_loader_ok": dataset_ok,
        "dataset_loader_error": dataset_error,
    }
    _write_json(index_path.with_suffix(".validate_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if missing or not dataset_ok:
        raise RuntimeError("Validation failed; see validate summary.")
    return summary


def _add_common_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("materialize", help="Download BenchCAD parquet and write .py files.")
    _add_common_root_arg(p)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--revision", default="main")
    p.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    p.add_argument("--local-data-root", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.set_defaults(func=materialize_benchcad)

    p = sub.add_parser("export", help="Execute CadQuery .py files and export STEP/STL.")
    _add_common_root_arg(p)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--num-processes", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p.add_argument("--triangle-face-tol", type=float, default=0.001)
    p.add_argument("--angle-tol-rads", type=float, default=0.1)
    p.add_argument("--export-timeout-s", type=int, default=300)
    p.add_argument("--force", action="store_true")
    p.add_argument("--rewrite-manifest", action="store_true")
    p.set_defaults(func=export_cadquery)

    p = sub.add_parser("refresh-export-manifest", help="Rebuild export manifest from existing STEP/STL files.")
    _add_common_root_arg(p)
    p.add_argument("--manifest", type=Path, default=None)
    p.set_defaults(func=refresh_export_manifest)

    p = sub.add_parser("assets", help="Generate point clouds and renders from STEP files.")
    _add_common_root_arg(p)
    p.add_argument("--num-points", type=int, default=8192)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--render-backend", default="blender-step", choices=("trimesh", "blender-step", "none"))
    p.add_argument("--step-mesh-backend", default="cadquery")
    p.add_argument("--step-mesh-fallback-backends", default="")
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--triangle-face-tol", type=float, default=0.01)
    p.add_argument("--angle-tol-rads", type=float, default=0.1)
    p.add_argument("--blender-bin", default="blender")
    p.add_argument("--blender-script", type=Path, default=None)
    p.add_argument("--blender-engine", default="CYCLES")
    p.add_argument("--blender-samples", type=int, default=64)
    p.add_argument("--blender-style", default="visualization")
    p.add_argument("--blender-device", default="AUTO")
    p.add_argument("--visualization-root", default=None)
    p.add_argument("--freecad-cmd", default="freecadcmd")
    p.add_argument("--step-mesh-timeout-s", type=float, default=None)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--resume-manifest", action="store_true", default=True)
    p.add_argument("--incremental-manifest", action="store_true", default=True)
    p.add_argument("--max-tasks", type=int, default=None)
    p.set_defaults(func=run_assets)

    p = sub.add_parser("build-index", help="Build modified_cadrille Qwen JSONL index.")
    _add_common_root_arg(p)
    p.add_argument("--materialized-manifest", type=Path, default=None)
    p.add_argument("--output-path", type=Path, default=None)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--max-images", type=int, default=4)
    p.add_argument("--require-pointcloud", action="store_true")
    p.add_argument("--require-images", action="store_true")
    p.set_defaults(func=build_jsonl_index)

    p = sub.add_parser("validate", help="Validate paths and qwen3vl_data loading.")
    _add_common_root_arg(p)
    p.add_argument("--index-path", type=Path, default=None)
    p.add_argument("--modified-cadrille-root", type=Path, default=Path("/root/autodl-tmp/modified_cadrille"))
    p.add_argument("--max-path-checks", type=int, default=100)
    p.add_argument("--dataset-num-points", type=int, default=256)
    p.add_argument("--max-images", type=int, default=4)
    p.add_argument("--prompt", default="Generate cadquery code")
    p.set_defaults(func=validate_index)

    p = sub.add_parser("run", help="Run materialize, export, assets, build-index, validate.")
    _add_common_root_arg(p)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--revision", default="main")
    p.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    p.add_argument("--local-data-root", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--export-processes", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p.add_argument("--asset-processes", type=int, default=1)
    p.add_argument("--render-backend", default="blender-step", choices=("trimesh", "blender-step", "none"))
    p.add_argument("--blender-bin", default="blender")
    p.add_argument("--continue-on-asset-failure", action="store_true")
    p.set_defaults(func=run_all)
    return parser.parse_args(argv)


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root
    materialize_benchcad(
        argparse.Namespace(
            output_root=root,
            repo_id=args.repo_id,
            config=args.config,
            revision=args.revision,
            hf_endpoint=args.hf_endpoint,
            local_data_root=args.local_data_root,
            batch_size=args.batch_size,
            start_index=0,
            max_samples=args.max_samples,
        )
    )
    export_cadquery(
        argparse.Namespace(
            output_root=root,
            manifest=None,
            num_processes=args.export_processes,
            triangle_face_tol=0.001,
            angle_tol_rads=0.1,
            export_timeout_s=300,
            force=False,
            rewrite_manifest=True,
        )
    )
    asset_rc = run_assets(
        argparse.Namespace(
            output_root=root,
            num_points=8192,
            num_views=4,
            img_size=128,
            render_backend=args.render_backend,
            step_mesh_backend="cadquery",
            step_mesh_fallback_backends="",
            num_processes=args.asset_processes,
            triangle_face_tol=0.01,
            angle_tol_rads=0.1,
            blender_bin=args.blender_bin,
            blender_script=None,
            blender_engine="CYCLES",
            blender_samples=64,
            blender_style="visualization",
            blender_device="AUTO",
            visualization_root=None,
            freecad_cmd="freecadcmd",
            step_mesh_timeout_s=None,
            skip_existing=True,
            resume_manifest=True,
            incremental_manifest=True,
            max_tasks=None,
        )
    )
    if asset_rc != 0 and not args.continue_on_asset_failure:
        raise RuntimeError(f"Asset generation failed with exit code {asset_rc}")
    index_summary = build_jsonl_index(
        argparse.Namespace(
            output_root=root,
            materialized_manifest=None,
            output_path=None,
            val_ratio=0.05,
            max_images=4,
            require_pointcloud=False,
            require_images=False,
        )
    )
    validate_summary = validate_index(
        argparse.Namespace(
            output_root=root,
            index_path=None,
            modified_cadrille_root=Path("/root/autodl-tmp/modified_cadrille"),
            max_path_checks=100,
            dataset_num_points=256,
            max_images=4,
            prompt="Generate cadquery code",
        )
    )
    return {"asset_rc": asset_rc, "index": index_summary, "validate": validate_summary}


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    result = args.func(args)
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
