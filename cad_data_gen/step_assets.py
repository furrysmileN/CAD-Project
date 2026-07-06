from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import trimesh

from .step_mesh_backends import StepMeshConfig, load_step_mesh


def iter_step_files(input_dir: str | Path, recursive: bool = False, filename_pattern: str = "*") -> Iterator[Path]:
    root = Path(input_dir)
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".step", ".stp"):
            continue
        if not fnmatch.fnmatch(path.name, filename_pattern):
            continue
        if path.stat().st_size == 0:
            continue
        yield path


def sample_id_from_relative_path(relative_step_path: str) -> str:
    rel = Path(relative_step_path).with_suffix("")
    safe_parts = []
    for part in rel.parts:
        if part in ("", "."):
            continue
        safe_parts.append("".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in part))
    return "__".join(safe_parts)


def load_step_as_mesh(
    step_path: str | Path,
    triangle_face_tol: float = 0.01,
    angle_tol_rads: float = 0.1,
    *,
    step_mesh_backend: str = "freecad",
    step_mesh_fallback_backends: Sequence[str] = ("cadquery",),
    step_mesh_work_dir: Optional[Path] = None,
    step_mesh_format: str = "stl",
    freecad_cmd: str = "freecadcmd",
    step_mesh_timeout_s: Optional[float] = None,
    keep_intermediate_mesh: bool = True,
) -> Tuple[trimesh.Trimesh, Optional[np.ndarray], str]:
    result = load_step_mesh(
        step_path,
        StepMeshConfig(
            backend=step_mesh_backend,
            fallback_backends=tuple(step_mesh_fallback_backends),
            work_dir=step_mesh_work_dir,
            keep_intermediate=keep_intermediate_mesh,
            mesh_format=step_mesh_format,
            freecad_cmd=freecad_cmd,
            timeout_s=step_mesh_timeout_s,
            triangle_face_tol=triangle_face_tol,
            angle_tol_rads=angle_tol_rads,
        ),
    )
    return result.mesh, result.tri_mapping, result.loader
