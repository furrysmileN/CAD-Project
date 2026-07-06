from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import numpy as np
import trimesh

try:
    import cadquery as cq
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    cq = None


class StepMeshBackendError(RuntimeError):
    """STEP 转 mesh 后端的统一异常。"""

    def __init__(self, message: str, *, backend: str, detail: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.backend = backend
        self.detail = detail or {}


@dataclass(frozen=True)
class MeshLoadResult:
    mesh: trimesh.Trimesh
    tri_mapping: Optional[np.ndarray]
    loader: str
    mesh_path: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepMeshConfig:
    backend: str = "freecad"
    fallback_backends: tuple[str, ...] = ("cadquery",)
    work_dir: Optional[Path] = None
    keep_intermediate: bool = True
    mesh_format: str = "stl"
    freecad_cmd: str = "freecadcmd"
    timeout_s: Optional[float] = None
    triangle_face_tol: float = 0.01
    angle_tol_rads: float = 0.1


class StepMeshBackend(Protocol):
    name: str

    def check_available(self, config: StepMeshConfig) -> None:
        ...

    def load(self, step_path: Path, config: StepMeshConfig) -> MeshLoadResult:
        ...


def _validate_mesh(mesh: trimesh.Trimesh, *, backend: str, source: Path) -> trimesh.Trimesh:
    if mesh.is_empty or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise StepMeshBackendError(f"empty mesh loaded from {source}", backend=backend)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if not np.isfinite(bounds).all():
        raise StepMeshBackendError(f"mesh bounds are not finite: {source}", backend=backend)
    return mesh


def _load_mesh_file(path: Path, *, backend: str) -> trimesh.Trimesh:
    try:
        loaded = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:
        raise StepMeshBackendError(f"failed to load converted mesh {path}: {exc}", backend=backend) from exc
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise StepMeshBackendError(f"converted scene contains no mesh geometry: {path}", backend=backend)
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise StepMeshBackendError(f"converted file is not a Trimesh: {path}", backend=backend)
    return _validate_mesh(loaded, backend=backend, source=path)


class CadQueryBackend:
    name = "cadquery"

    def check_available(self, config: StepMeshConfig) -> None:
        del config
        if cq is None:
            raise StepMeshBackendError(
                "cadquery backend unavailable: install cadquery/OCP or choose --step-mesh-backend freecad",
                backend=self.name,
                detail={"install_hint": "conda install -c conda-forge cadquery"},
            )

    def load(self, step_path: Path, config: StepMeshConfig) -> MeshLoadResult:
        self.check_available(config)
        assert cq is not None
        workplane = cq.importers.importStep(str(step_path))
        if hasattr(workplane, "vals"):
            shapes = list(workplane.vals())
        elif hasattr(workplane, "val"):
            shapes = [workplane.val()]
        else:
            shapes = [workplane]

        all_verts: list[np.ndarray] = []
        all_tris: list[np.ndarray] = []
        tri_mapping: list[int] = []
        vert_offset = 0
        for shape_index, shape in enumerate(shapes):
            vertices, faces = shape.tessellate(
                config.triangle_face_tol,
                angularTolerance=config.angle_tol_rads,
            )
            if len(vertices) == 0 or len(faces) == 0:
                continue
            verts = np.asarray([(v.x, v.y, v.z) for v in vertices], dtype=np.float64)
            tris = np.asarray(faces, dtype=np.int64) + vert_offset
            all_verts.append(verts)
            all_tris.append(tris)
            tri_mapping.extend([shape_index] * len(tris))
            vert_offset += len(verts)
        if not all_verts or not all_tris:
            raise StepMeshBackendError(f"empty mesh after cadquery tessellation: {step_path}", backend=self.name)
        mesh = trimesh.Trimesh(
            vertices=np.concatenate(all_verts, axis=0),
            faces=np.concatenate(all_tris, axis=0),
            process=False,
        )
        return MeshLoadResult(
            mesh=_validate_mesh(mesh, backend=self.name, source=step_path),
            tri_mapping=np.asarray(tri_mapping, dtype=np.int32),
            loader=self.name,
            metadata={"has_brep_face_mapping": True},
        )


class FreeCADBackend:
    name = "freecad"

    def check_available(self, config: StepMeshConfig) -> None:
        if shutil.which(config.freecad_cmd) is None:
            raise StepMeshBackendError(
                f"freecad backend unavailable: `{config.freecad_cmd}` not found in PATH",
                backend=self.name,
                detail={
                    "check_command": f"which {config.freecad_cmd}",
                    "install_hint": "install FreeCAD and ensure freecadcmd is in PATH",
                },
            )

    def _output_path(self, step_path: Path, config: StepMeshConfig) -> Path:
        suffix = "." + config.mesh_format.lower().lstrip(".")
        if config.work_dir is not None:
            safe_stem = step_path.with_suffix("").name
            out_dir = config.work_dir / "meshes" / safe_stem
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir / f"{safe_stem}{suffix}"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        return Path(tmp.name)

    def load(self, step_path: Path, config: StepMeshConfig) -> MeshLoadResult:
        self.check_available(config)
        mesh_path = self._output_path(step_path, config)
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        script_file = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        script_file.write(
            "\n".join(
                [
                    "import sys",
                    "import FreeCAD",
                    "import Import",
                    "import Mesh",
                    "step_path = sys.argv[-2]",
                    "out_path = sys.argv[-1]",
                    "doc = FreeCAD.newDocument('step_to_mesh')",
                    "Import.insert(step_path, doc.Name)",
                    "doc.recompute()",
                    "objects = [obj for obj in doc.Objects if hasattr(obj, 'Shape')]",
                    "if not objects:",
                    "    raise RuntimeError('FreeCAD imported no shape objects')",
                    "Mesh.export(objects, out_path)",
                    "FreeCAD.closeDocument(doc.Name)",
                ]
            )
        )
        script_file.close()
        try:
            proc = subprocess.run(
                [config.freecad_cmd, script_file.name, str(step_path), str(mesh_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=config.timeout_s,
            )
        except FileNotFoundError as exc:
            raise StepMeshBackendError(
                f"freecad backend unavailable: `{config.freecad_cmd}` not found in PATH",
                backend=self.name,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise StepMeshBackendError(
                f"FreeCAD STEP conversion timed out after {config.timeout_s}s: {step_path}",
                backend=self.name,
                detail={"timeout_s": config.timeout_s},
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or f"exit code {exc.returncode}"
            raise StepMeshBackendError(
                f"FreeCAD STEP conversion failed for {step_path}: {detail}",
                backend=self.name,
                detail={
                    "return_code": exc.returncode,
                    "stdout": exc.stdout,
                    "stderr": exc.stderr,
                },
            ) from exc
        finally:
            Path(script_file.name).unlink(missing_ok=True)

        if not mesh_path.is_file() or mesh_path.stat().st_size == 0:
            raise StepMeshBackendError(
                f"FreeCAD did not produce a valid mesh: {mesh_path}",
                backend=self.name,
            )
        mesh = _load_mesh_file(mesh_path, backend=self.name)
        return MeshLoadResult(
            mesh=mesh,
            tri_mapping=None,
            loader=self.name,
            mesh_path=mesh_path if config.keep_intermediate else None,
            metadata={
                "mesh_path": str(mesh_path),
                "mesh_format": config.mesh_format,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            },
        )


_BACKENDS: dict[str, StepMeshBackend] = {
    "cadquery": CadQueryBackend(),
    "freecad": FreeCADBackend(),
}


def available_backend_names() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def get_backend(name: str) -> StepMeshBackend:
    key = name.strip().lower()
    if key not in _BACKENDS:
        raise StepMeshBackendError(
            f"unknown STEP mesh backend: {name}; available={', '.join(available_backend_names())}",
            backend=key or "unknown",
        )
    return _BACKENDS[key]


def _backend_chain(config: StepMeshConfig) -> list[str]:
    names: list[str] = []
    for name in (config.backend, *config.fallback_backends):
        key = str(name).strip().lower()
        if key and key not in names:
            names.append(key)
    return names
def load_step_mesh(step_path: str | Path, config: StepMeshConfig) -> MeshLoadResult:
    path = Path(step_path)
    errors: list[dict[str, Any]] = []
    for backend_name in _backend_chain(config):
        backend = get_backend(backend_name)
        try:
            return backend.load(path, config)
        except StepMeshBackendError as exc:
            errors.append({"backend": exc.backend, "error": str(exc), "detail": exc.detail})
        except Exception as exc:
            errors.append(
                {
                    "backend": backend_name,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
    raise StepMeshBackendError(
        f"all STEP mesh backends failed for {path}: "
        + "; ".join(f"{item['backend']}: {item['error']}" for item in errors),
        backend=config.backend,
        detail={"attempts": errors},
    )


def parse_fallback_backends(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = list(raw)
    return tuple(part.strip().lower() for part in parts if str(part).strip())
