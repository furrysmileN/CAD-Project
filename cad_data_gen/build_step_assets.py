#!/usr/bin/env python3
"""
Build point-cloud and multi-view image assets from STEP files.

This module keeps the helper API that `build_occlusion_assets.py` depends on:
STEP loading, unit-cube normalization, deterministic camera fronts, a small CPU
renderer, and the Blender wrapper used for higher quality renders.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

from .pipeline_state import AssetPaths, SampleContext, StageError, StageStatus
from .step_mesh_backends import (
    MeshLoadResult,
    StepMeshBackendError,
    StepMeshConfig,
    load_step_mesh,
    parse_fallback_backends,
)

DEFAULT_VISUALIZATION_ROOT = Path("/root/autodl-tmp/visualization/visualization-tar")


@dataclass(frozen=True)
class ProcessArgs:
    input_dir: Path
    output_dir: Path
    recursive: bool
    filename_pattern: str
    num_points: int
    num_views: int
    img_size: int
    camera_distance: float
    camera_jitter_degrees: float
    camera_jitter_seed: int
    triangle_face_tol: float
    angle_tol_rads: float
    step_mesh_backend: str
    step_mesh_fallback_backends: tuple[str, ...]
    step_mesh_work_dir: Optional[Path]
    step_mesh_format: str
    freecad_cmd: str
    step_mesh_timeout_s: Optional[float]
    keep_intermediate_mesh: bool
    normalize: str
    render_backend: str
    blender_bin: str
    blender_script: Optional[Path]
    blender_engine: str
    blender_samples: int
    blender_style: str
    visualization_root: Optional[str]
    blender_device: str
    foreground_occluder: bool
    foreground_occluder_seed: int
    foreground_occluder_color: tuple[int, int, int]
    foreground_occluder_size_min: float
    foreground_occluder_size_max: float
    foreground_occluder_depth: float
    lighting_preset: str
    lighting_seed: int
    lighting_ambient: Optional[tuple[float, float]]
    lighting_directional: Optional[tuple[float, float]]
    lighting_jitter: float
    skip_existing: bool


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


def sample_id_from_relative_path(relative_step_path: str | Path) -> str:
    rel = Path(relative_step_path).with_suffix("")
    safe_parts: list[str] = []
    for part in rel.parts:
        if part in ("", "."):
            continue
        safe_parts.append("".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in part))
    return "__".join(safe_parts)


def load_step_as_mesh_result(
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
) -> MeshLoadResult:
    config = StepMeshConfig(
        backend=step_mesh_backend,
        fallback_backends=tuple(step_mesh_fallback_backends),
        work_dir=step_mesh_work_dir,
        keep_intermediate=keep_intermediate_mesh,
        mesh_format=step_mesh_format,
        freecad_cmd=freecad_cmd,
        timeout_s=step_mesh_timeout_s,
        triangle_face_tol=triangle_face_tol,
        angle_tol_rads=angle_tol_rads,
    )
    return load_step_mesh(step_path, config)


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
    result = load_step_as_mesh_result(
        step_path,
        triangle_face_tol=triangle_face_tol,
        angle_tol_rads=angle_tol_rads,
        step_mesh_backend=step_mesh_backend,
        step_mesh_fallback_backends=step_mesh_fallback_backends,
        step_mesh_work_dir=step_mesh_work_dir,
        step_mesh_format=step_mesh_format,
        freecad_cmd=freecad_cmd,
        step_mesh_timeout_s=step_mesh_timeout_s,
        keep_intermediate_mesh=keep_intermediate_mesh,
    )
    return result.mesh, result.tri_mapping, result.loader


def mesh_to_point_cloud(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface(mesh, int(num_points))
    return np.asarray(points, dtype=np.float32)


def mesh_to_point_cloud_with_normals(mesh: trimesh.Trimesh, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    points, face_indices = trimesh.sample.sample_surface(mesh, int(num_points))
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    normals = face_normals[np.asarray(face_indices, dtype=np.int64)]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)
    return np.asarray(points, dtype=np.float32), np.asarray(normals, dtype=np.float32)


def normalize_trimesh_unit_cube(
    mesh: trimesh.Trimesh,
    points: Optional[np.ndarray] = None,
) -> tuple[trimesh.Trimesh, Optional[np.ndarray], np.ndarray, float, np.ndarray]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if not np.isfinite(bounds).all():
        raise ValueError("mesh bounds are not finite")
    extents = bounds[1] - bounds[0]
    scale = 1.0 / max(float(extents.max()), 1e-12)
    raw_center = (bounds[0] + bounds[1]) * 0.5

    mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) - raw_center) * scale
    centered_points = None
    if points is not None:
        centered_points = (np.asarray(points, dtype=np.float64) - raw_center) * scale
    return mesh, centered_points, bounds, float(scale), raw_center.astype(np.float64)


def _round_array(values: np.ndarray, decimals: int = 6) -> list[float]:
    return [round(float(value), decimals) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def mesh_metrics_from_mesh(
    mesh: trimesh.Trimesh,
    tri_mapping: Optional[np.ndarray],
    loader: str,
) -> dict[str, Any]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    bbox_min = bounds[0]
    bbox_max = bounds[1]
    bbox_extent = bbox_max - bbox_min
    center = (bbox_min + bbox_max) / 2.0
    diagonal = float(np.linalg.norm(bbox_extent))

    metrics: dict[str, Any] = {
        "loader": loader,
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "bbox_min": _round_array(bbox_min),
        "bbox_max": _round_array(bbox_max),
        "bbox_extent": _round_array(bbox_extent),
        "bbox_center": _round_array(center),
        "bbox_diagonal": round(diagonal, 6),
        "surface_area": round(float(mesh.area), 6),
        "is_watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "volume": round(abs(float(mesh.volume)), 6) if mesh.is_watertight else None,
    }
    try:
        components = mesh.split(only_watertight=False)
        metrics["connected_components"] = int(len(components))
    except Exception:
        metrics["connected_components"] = None

    if tri_mapping is not None:
        unique_faces = np.unique(tri_mapping.astype(np.int64))
        metrics["brep_face_count_from_mapping"] = int(len(unique_faces))
        metrics["has_point_to_brep_face_mapping_source"] = True
    else:
        metrics["brep_face_count_from_mapping"] = None
        metrics["has_point_to_brep_face_mapping_source"] = False
    return metrics


def get_view_fronts(num_views: int) -> List[List[float]]:
    if num_views == 1:
        return [[1, 1, 1]]
    if num_views == 2:
        return [[1, 1, 1], [-1, -1, -1]]
    if num_views == 4:
        return [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
    if num_views == 6:
        return [
            [1, 1, 1],
            [-1, -1, -1],
            [-1, 1, -1],
            [1, -1, 1],
            [0, 1, 0],
            [0, -1, 0],
        ]

    # Deterministic fallback for arbitrary view counts.
    fronts: list[list[float]] = []
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(num_views):
        y = 1.0 - (2.0 * i + 1.0) / float(num_views)
        radius = np.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        fronts.append([float(np.cos(theta) * radius), float(y), float(np.sin(theta) * radius)])
    return fronts


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _view_basis(front: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = np.asarray(front, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("camera front must be non-zero")
    direction = direction / norm
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(up, direction)
    if float(np.linalg.norm(right)) < 1e-8:
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(up, direction)
    right = right / float(np.linalg.norm(right))
    true_up = np.cross(direction, right)
    true_up = true_up / float(np.linalg.norm(true_up))
    return right, true_up, direction


def jitter_camera_fronts(
    fronts: Sequence[Sequence[float]],
    sample_id: str,
    max_degrees: float,
    seed: int = 0,
) -> List[List[float]]:
    max_radians = np.radians(max(0.0, float(max_degrees)))
    if max_radians <= 0.0:
        return [[float(v) for v in front] for front in fronts]

    rng = np.random.default_rng(_stable_seed(sample_id, int(seed)))
    jittered: list[list[float]] = []
    for front in fronts:
        right, true_up, direction = _view_basis(front)
        radius = max_radians * np.sqrt(float(rng.uniform(0.0, 1.0)))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        candidate = direction + np.tan(radius) * (np.cos(theta) * right + np.sin(theta) * true_up)
        candidate = candidate / max(float(np.linalg.norm(candidate)), 1e-12)
        jittered.append([float(v) for v in candidate])
    return jittered


def _project_vertices(vertices: np.ndarray, front: Sequence[float], img_size: int) -> tuple[np.ndarray, np.ndarray]:
    right, true_up, direction = _view_basis(front)
    verts = np.asarray(vertices, dtype=np.float64)
    xy = np.stack([verts @ right, verts @ true_up], axis=1)
    span = float(np.max(np.ptp(xy, axis=0)))
    if span < 1e-8:
        span = 1.0
    pad = img_size * 0.12
    scale = (img_size - 2.0 * pad) / span
    pixels = np.empty_like(xy)
    pixels[:, 0] = xy[:, 0] * scale + img_size / 2.0
    pixels[:, 1] = img_size / 2.0 - xy[:, 1] * scale
    depth = verts @ direction
    return pixels, depth


def make_lighting_config(
    preset: str = "default",
    seed: int = 0,
    ambient: Optional[tuple[float, float]] = None,
    directional: Optional[tuple[float, float]] = None,
    jitter: float = 0.0,
) -> dict[str, Any]:
    presets: dict[str, dict[str, Any]] = {
        "default": {
            "ambient": [0.58, 0.58],
            "directional": [0.42, 0.42],
            "key_vector": [0.0, 0.0, 1.0],
            "fill": 28.0,
            "background": 245,
        },
        "fixed_blue_orange": {
            "ambient": [0.30, 0.30],
            "directional": [0.22, 0.22],
            "key_vector": [0.35, -0.25, 0.90],
            "fill": 12.0,
            "background": 220,
            "world": 0.18,
            "key_energy": 95.0,
            "fill_energy": 12.0,
            "exposure": -0.65,
        },
        "studio": {
            "ambient": [0.62, 0.76],
            "directional": [0.30, 0.50],
            "key_vector": [0.35, -0.25, 0.90],
            "fill": 34.0,
            "background": 246,
        },
        "high_contrast": {
            "ambient": [0.34, 0.50],
            "directional": [0.55, 0.82],
            "key_vector": [0.65, -0.55, 0.65],
            "fill": 18.0,
            "background": 238,
        },
        "low_key": {
            "ambient": [0.22, 0.38],
            "directional": [0.40, 0.65],
            "key_vector": [-0.45, -0.25, 0.85],
            "fill": 10.0,
            "background": 224,
        },
        "overcast": {
            "ambient": [0.72, 0.86],
            "directional": [0.08, 0.24],
            "key_vector": [0.0, 0.0, 1.0],
            "fill": 38.0,
            "background": 248,
        },
        "random": {
            "ambient": [0.28, 0.82],
            "directional": [0.12, 0.78],
            "key_vector": [0.35, -0.25, 0.90],
            "fill": 24.0,
            "background": 242,
        },
    }
    if preset not in presets:
        raise ValueError(f"unknown lighting preset: {preset}")
    config = dict(presets[preset])
    if ambient is not None:
        config["ambient"] = [float(ambient[0]), float(ambient[1])]
    if directional is not None:
        config["directional"] = [float(directional[0]), float(directional[1])]
    config["preset"] = preset
    config["seed"] = int(seed)
    config["jitter"] = float(jitter)
    return config


def _sample_range(rng: np.random.Generator, values: Sequence[float]) -> float:
    low = float(values[0])
    high = float(values[1]) if len(values) > 1 else low
    if high < low:
        low, high = high, low
    return float(rng.uniform(low, high)) if high > low else low


def sample_lighting_for_sample(lighting: dict[str, Any], sample_id: str) -> dict[str, Any]:
    rng = np.random.default_rng(_stable_seed(sample_id, int(lighting.get("seed", 0))))
    ambient = _sample_range(rng, lighting.get("ambient", [0.58, 0.58]))
    directional = _sample_range(rng, lighting.get("directional", [0.42, 0.42]))
    key_vector = np.asarray(lighting.get("key_vector", [0.35, -0.25, 0.90]), dtype=np.float64)
    jitter = float(lighting.get("jitter", 0.0))
    if jitter > 0.0:
        key_vector = key_vector + rng.normal(0.0, jitter, size=3)
    if float(np.linalg.norm(key_vector)) < 1e-12:
        key_vector = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    key_vector = key_vector / max(float(np.linalg.norm(key_vector)), 1e-12)

    sampled = dict(lighting)
    sampled["ambient"] = [ambient, ambient]
    sampled["directional"] = [directional, directional]
    sampled["key_vector"] = [float(v) for v in key_vector]
    sampled["fill"] = float(lighting.get("fill", 28.0))
    sampled["background"] = int(np.clip(float(lighting.get("background", 245)), 0.0, 255.0))
    sampled["world"] = float(lighting.get("world", max(0.08, ambient * 0.62)))
    sampled["key_energy"] = float(lighting.get("key_energy", 70.0 + directional * 220.0))
    sampled["fill_energy"] = float(lighting.get("fill_energy", 8.0 + ambient * 35.0))
    sampled["exposure"] = float(lighting.get("exposure", -0.35))
    sampled["sample_id"] = sample_id
    return sampled


def _lighting_for_view(front: Sequence[float], lighting: Optional[dict[str, Any]], view_index: int) -> dict[str, Any]:
    if lighting is None:
        lighting = make_lighting_config()
    rng = np.random.default_rng(int(lighting.get("seed", 0)) + int(view_index) * 1009)
    ambient = _sample_range(rng, lighting.get("ambient", [0.58, 0.58]))
    directional_strength = _sample_range(rng, lighting.get("directional", [0.42, 0.42]))
    fill = float(lighting.get("fill", 28.0))
    background = int(np.clip(float(lighting.get("background", 245)), 0.0, 255.0))

    _, _, direction = _view_basis(front)
    key_vector = np.asarray(lighting.get("key_vector", [0.0, 0.0, 1.0]), dtype=np.float64)
    if float(np.linalg.norm(key_vector)) < 1e-12:
        key_vector = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    jitter = float(lighting.get("jitter", 0.0))
    if jitter > 0.0:
        key_vector = key_vector + rng.normal(0.0, jitter, size=3)
    light = direction + key_vector
    if float(np.linalg.norm(light)) < 1e-12:
        light = direction
    light = light / max(float(np.linalg.norm(light)), 1e-12)
    return {
        "ambient": ambient,
        "directional": directional_strength,
        "fill": fill,
        "background": background,
        "light": light,
    }


def _rasterize_mesh(
    mesh: trimesh.Trimesh,
    front: Sequence[float],
    img_size: int,
    lighting: Optional[dict[str, Any]] = None,
    view_index: int = 0,
) -> Image.Image:
    pixels, depths = _project_vertices(np.asarray(mesh.vertices), front, img_size)
    z_buffer = np.full((img_size, img_size), -np.inf, dtype=np.float64)
    view_lighting = _lighting_for_view(front, lighting, view_index)
    rgb = np.full((img_size, img_size, 3), int(view_lighting["background"]), dtype=np.uint8)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    light = np.asarray(view_lighting["light"], dtype=np.float64)

    for face_index, tri in enumerate(np.asarray(mesh.faces, dtype=np.int64)):
        pts = pixels[tri]
        min_xy = np.floor(pts.min(axis=0)).astype(int)
        max_xy = np.ceil(pts.max(axis=0)).astype(int)
        min_x = max(0, int(min_xy[0]))
        min_y = max(0, int(min_xy[1]))
        max_x = min(img_size - 1, int(max_xy[0]))
        max_y = min(img_size - 1, int(max_xy[1]))
        if min_x > max_x or min_y > max_y:
            continue

        p0, p1, p2 = pts
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(float(denom)) < 1e-12:
            continue

        yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        px = xx + 0.5
        py = yy + 0.5
        w0 = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) / denom
        w1 = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue

        z_tri = depths[tri]
        z_pixels = w0 * z_tri[0] + w1 * z_tri[1] + w2 * z_tri[2]
        current = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
        update = inside & (z_pixels > current)
        if not update.any():
            continue
        normal = normals[face_index] if face_index < len(normals) else np.asarray([0.0, 0.0, 1.0])
        shade = float(view_lighting["ambient"]) + float(view_lighting["directional"]) * max(
            0.0,
            float(normal @ light),
        )
        color = np.asarray([150, 176, 205], dtype=np.float64) * shade
        color = np.clip(color + float(view_lighting["fill"]), 0.0, 255.0).astype(np.uint8)
        current[update] = z_pixels[update]
        patch = rgb[min_y : max_y + 1, min_x : max_x + 1]
        patch[update] = color

    return Image.fromarray(rgb, mode="RGB")


def render_views_to_png(
    mesh_centered: trimesh.Trimesh,
    image_dir: Path,
    fronts: List[List[float]],
    img_size: int,
    camera_distance: float = -0.9,
    render_width: int = 512,
    render_height: int = 512,
    render_backend: str = "trimesh",
    lighting: Optional[dict[str, Any]] = None,
) -> List[str]:
    del camera_distance, render_width, render_height, render_backend
    image_dir.mkdir(parents=True, exist_ok=True)
    rel_paths: list[str] = []
    for i, front in enumerate(fronts):
        image = _rasterize_mesh(mesh_centered, front, img_size, lighting=lighting, view_index=i)
        out_name = f"view_{i:03d}.png"
        image.save(image_dir / out_name)
        rel_paths.append(str(Path("images") / image_dir.name / out_name))
    return rel_paths


def _color_to_unit_rgba(color: Sequence[int | float]) -> list[float]:
    values = list(color)
    if len(values) == 3:
        values.append(255)
    return [float(v) / 255.0 if float(v) > 1.0 else float(v) for v in values[:4]]


def resolve_blender_bin(
    blender_bin: str,
    blender_style: str,
    visualization_root: Optional[str],
) -> str:
    if blender_bin != "blender":
        return blender_bin

    root: Optional[Path]
    if visualization_root is not None:
        root = Path(visualization_root).expanduser().resolve()
    elif blender_style == "visualization" and DEFAULT_VISUALIZATION_ROOT.is_dir():
        root = DEFAULT_VISUALIZATION_ROOT
    else:
        root = None

    candidates: list[Path] = []
    if root is not None:
        candidates.extend(
            [
                root / "blender" / "blender",
                root / "blender-3.6.0-linux-x64" / "blender",
                root / "blender-3.6.5-linux-x64" / "blender",
                root / "blender-4.0.0-linux-x64" / "blender",
            ]
        )
        candidates.extend(root.glob("blender*/blender"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return blender_bin


def make_foreground_occluder_config(
    num_views: int,
    seed: int,
    color: Sequence[int | float],
    size_min: float,
    size_max: float,
    depth: float = 0.45,
    shapes: Sequence[str] = ("rectangle", "ellipse", "triangle", "hexagon"),
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    views: list[dict[str, Any]] = []
    for _ in range(int(num_views)):
        width = float(rng.uniform(size_min, size_max))
        height = float(rng.uniform(size_min, size_max))
        max_x = max(0.0, 0.5 - width * 0.5)
        max_y = max(0.0, 0.5 - height * 0.5)
        views.append(
            {
                "offset_xy": [
                    float(rng.uniform(-max_x, max_x)),
                    float(rng.uniform(-max_y, max_y)),
                ],
                "size_xy": [width, height],
                "angle_degrees": float(rng.uniform(-25.0, 25.0)),
                "shape": str(rng.choice(list(shapes))),
                "depth": float(depth),
            }
        )
    return {
        "color": _color_to_unit_rgba(color),
        "depth": float(depth),
        "views": views,
        "placement": "image_plane_random",
    }


def render_step_views_with_blender(
    *,
    step_path: Path,
    mesh_path: Optional[Path],
    image_dir: Path,
    fronts: Sequence[Sequence[float]],
    img_size: int,
    raw_center: Sequence[float],
    scale: float,
    camera_distance: float,
    blender_bin: str,
    blender_script: Optional[Path],
    blender_engine: str,
    blender_samples: int,
    blender_style: str,
    visualization_root: Optional[str],
    blender_device: str,
    lighting: Optional[dict[str, Any]] = None,
    foreground_occluder: Optional[dict[str, Any]] = None,
) -> list[str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    script_path = blender_script or (Path(__file__).with_name("render_step_with_blender.py"))
    resolved_blender = resolve_blender_bin(blender_bin, blender_style, visualization_root)
    config = {
        "step_path": str(step_path),
        "mesh_path": str(mesh_path) if mesh_path is not None else None,
        "output_dir": str(image_dir),
        "fronts": [[float(v) for v in front] for front in fronts],
        "img_size": int(img_size),
        "raw_center": [float(v) for v in raw_center],
        "scale": float(scale),
        "camera_distance": float(camera_distance),
        "engine": blender_engine,
        "samples": int(blender_samples),
        "render_style": blender_style,
        "lighting": lighting or {},
        "foreground_occluder": foreground_occluder,
        "blender_device": blender_device,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)
        config_path = Path(f.name)
    try:
        cmd = [
            resolved_blender,
            "-b",
            "--python",
            str(script_path),
            "--",
            "--config",
            str(config_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    finally:
        config_path.unlink(missing_ok=True)
    return [str(Path("images") / image_dir.name / f"view_{i:03d}.png") for i in range(len(fronts))]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _existing_sample_done(paths: AssetPaths, num_views: int) -> bool:
    if not paths.point_path.exists():
        return False
    return all((paths.image_dir / f"view_{i:03d}.png").exists() for i in range(num_views))


def _process_step_file(step_path: Path, args: ProcessArgs) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    try:
        rel_step = step_path.resolve().relative_to(args.input_dir.resolve())
    except ValueError:
        rel_step = Path(step_path.name)
    sample_id = sample_id_from_relative_path(rel_step)
    paths = AssetPaths(args.output_dir, sample_id)
    context = SampleContext(step_path=step_path, rel_step=rel_step, sample_id=sample_id, paths=paths)

    status = StageStatus(sample_id=sample_id, status="running")
    if args.skip_existing and _existing_sample_done(paths, args.num_views):
        status.mark_success()
        return (
            {
                "sample_id": sample_id,
                "step_path": str(step_path),
                "relative_step_path": str(rel_step),
                "point_path": str(paths.point_path),
                "image_paths": [str(paths.image_dir / f"view_{i:03d}.png") for i in range(args.num_views)],
                "status": "skipped_existing",
            },
            None,
        )

    try:
        load_result = load_step_as_mesh_result(
            context.step_path,
            triangle_face_tol=args.triangle_face_tol,
            angle_tol_rads=args.angle_tol_rads,
            step_mesh_backend=args.step_mesh_backend,
            step_mesh_fallback_backends=args.step_mesh_fallback_backends,
            step_mesh_work_dir=args.step_mesh_work_dir,
            step_mesh_format=args.step_mesh_format,
            freecad_cmd=args.freecad_cmd,
            step_mesh_timeout_s=args.step_mesh_timeout_s,
            keep_intermediate_mesh=args.keep_intermediate_mesh,
        )
        raw_mesh = load_result.mesh
        raw_points, normals = mesh_to_point_cloud_with_normals(raw_mesh, args.num_points)
        mesh_centered, centered_points, raw_bounds, scale, raw_center = normalize_trimesh_unit_cube(
            raw_mesh.copy(),
            raw_points,
        )
        assert centered_points is not None
        paths.point_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            paths.point_path,
            points=centered_points.astype(np.float32),
            normals=normals.astype(np.float32),
            sample_id=sample_id,
            raw_center=raw_center.astype(np.float64),
            scale=np.asarray(scale, dtype=np.float64),
            raw_bounds=np.asarray(raw_bounds, dtype=np.float64),
        )
        status.mark_done("point_cloud", point_path=str(paths.point_path))

        fronts = jitter_camera_fronts(
            get_view_fronts(args.num_views),
            sample_id,
            args.camera_jitter_degrees,
            args.camera_jitter_seed,
        )
        lighting = sample_lighting_for_sample(
            make_lighting_config(
                args.lighting_preset,
                seed=args.lighting_seed,
                ambient=args.lighting_ambient,
                directional=args.lighting_directional,
                jitter=args.lighting_jitter,
            ),
            sample_id,
        )
        foreground = None
        if args.foreground_occluder:
            foreground = make_foreground_occluder_config(
                args.num_views,
                _stable_seed(sample_id, args.foreground_occluder_seed),
                args.foreground_occluder_color,
                args.foreground_occluder_size_min,
                args.foreground_occluder_size_max,
                args.foreground_occluder_depth,
            )

        if args.render_backend == "none":
            image_paths: list[str] = []
        elif args.render_backend == "blender-step":
            blender_mesh_path = load_result.mesh_path
            if blender_mesh_path is None:
                candidate_mesh_path = args.output_dir / "mesh" / f"{sample_id}.stl"
                if candidate_mesh_path.is_file() and candidate_mesh_path.stat().st_size > 0:
                    blender_mesh_path = candidate_mesh_path
            image_paths = render_step_views_with_blender(
                step_path=context.step_path,
                mesh_path=blender_mesh_path,
                image_dir=paths.image_dir,
                fronts=fronts,
                img_size=args.img_size,
                raw_center=raw_center,
                scale=scale,
                camera_distance=args.camera_distance,
                blender_bin=args.blender_bin,
                blender_script=args.blender_script,
                blender_engine=args.blender_engine,
                blender_samples=args.blender_samples,
                blender_style=args.blender_style,
                visualization_root=args.visualization_root,
                blender_device=args.blender_device,
                lighting=lighting,
                foreground_occluder=foreground,
            )
            image_paths = [str(paths.image_dir / Path(path).name) for path in image_paths]
        else:
            image_paths = render_views_to_png(
                mesh_centered,
                paths.image_dir,
                fronts,
                args.img_size,
                camera_distance=args.camera_distance,
                render_backend=args.render_backend,
                lighting=lighting,
            )
            image_paths = [str(paths.output_dir / rel_path) for rel_path in image_paths]
        status.mark_done("images", image_dir=str(paths.image_dir))
        status.mark_success()

        record = {
            "sample_id": sample_id,
            "step_path": str(step_path),
            "relative_step_path": str(rel_step),
            "point_path": str(paths.point_path),
            "image_paths": image_paths,
            "num_points": int(args.num_points),
            "num_views": int(args.num_views),
            "img_size": int(args.img_size),
            "render_backend": args.render_backend,
            "mesh_loader": load_result.loader,
            "mesh_metrics": mesh_metrics_from_mesh(raw_mesh, load_result.tri_mapping, load_result.loader),
            "conversion_metadata": load_result.metadata,
            "normalization": {
                "mode": args.normalize,
                "raw_bounds": np.asarray(raw_bounds).tolist(),
                "raw_center": raw_center.tolist(),
                "scale": float(scale),
            },
            "status": status.to_dict(),
        }
        return record, None
    except BaseException as exc:  # noqa: BLE001
        backend = getattr(exc, "backend", None)
        detail = getattr(exc, "detail", {})
        error = StageError(
            stage="assets",
            error_type=type(exc).__name__,
            error=str(exc),
            step_path=str(step_path),
            sample_id=sample_id,
            backend=backend,
            extra=detail if isinstance(detail, dict) else {},
        )
        return None, error.to_dict()


def _process_step_file_star(params: tuple[Path, ProcessArgs]) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    return _process_step_file(*params)


def _load_completed_sample_ids(manifest_path: Path) -> set[str]:
    completed: set[str] = set()
    if not manifest_path.exists():
        return completed
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = row.get("sample_id")
            if sample_id:
                completed.add(str(sample_id))
    return completed


def _iter_tasks(args: ProcessArgs, resume_manifest: bool) -> list[Path]:
    manifest_path = args.output_dir / "manifest.jsonl"
    completed = _load_completed_sample_ids(manifest_path) if resume_manifest else set()
    tasks: list[Path] = []
    for step_path in iter_step_files(args.input_dir, args.recursive, args.filename_pattern):
        try:
            rel_step = step_path.resolve().relative_to(args.input_dir.resolve())
        except ValueError:
            rel_step = Path(step_path.name)
        sample_id = sample_id_from_relative_path(rel_step)
        if sample_id in completed:
            continue
        tasks.append(step_path)
    return tasks


def _run_tasks(tasks: list[Path], args: ProcessArgs, num_processes: int) -> tuple[int, int]:
    manifest_path = args.output_dir / "manifest.jsonl"
    failures_path = args.output_dir / "failures.jsonl"
    ok = 0
    failed = 0

    def consume(result: tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]) -> None:
        nonlocal ok, failed
        record, error = result
        if record is not None:
            _append_jsonl(manifest_path, record)
            ok += 1
        if error is not None:
            _append_jsonl(failures_path, error)
            failed += 1

    if num_processes <= 1:
        for task in tqdm(tasks, desc="build assets"):
            consume(_process_step_file(task, args))
    else:
        with Pool(processes=num_processes, initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN)) as pool:
            task_iter = zip(tasks, repeat(args))
            for result in tqdm(pool.imap_unordered(_process_step_file_star, task_iter), total=len(tasks), desc="build assets"):
                consume(result)
    return ok, failed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point-cloud and image assets from STEP files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--filename-pattern", default="*")
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--camera-distance", type=float, default=-0.9)
    parser.add_argument("--camera-jitter-degrees", type=float, default=0.0)
    parser.add_argument("--camera-jitter-seed", type=int, default=0)
    parser.add_argument("--triangle-face-tol", type=float, default=0.01)
    parser.add_argument("--angle-tol-rads", type=float, default=0.1)
    parser.add_argument("--step-mesh-backend", default="freecad")
    parser.add_argument("--step-mesh-fallback-backends", default="cadquery")
    parser.add_argument("--step-mesh-work-dir", type=Path, default=None)
    parser.add_argument("--step-mesh-format", default="stl")
    parser.add_argument("--freecad-cmd", default="freecadcmd")
    parser.add_argument("--step-mesh-timeout-s", type=float, default=None)
    parser.add_argument("--discard-intermediate-mesh", action="store_true")
    parser.add_argument("--normalize", default="unit_cube")
    parser.add_argument("--render-backend", default="trimesh", choices=("trimesh", "blender-step", "none"))
    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument("--blender-script", type=Path, default=None)
    parser.add_argument("--blender-engine", default="CYCLES")
    parser.add_argument("--blender-samples", type=int, default=64)
    parser.add_argument("--blender-style", default="visualization")
    parser.add_argument("--visualization-root", default=None)
    parser.add_argument("--blender-device", default="AUTO")
    parser.add_argument("--foreground-occluder", action="store_true")
    parser.add_argument("--foreground-occluder-seed", type=int, default=0)
    parser.add_argument("--foreground-occluder-color", default="255,158,20")
    parser.add_argument("--foreground-occluder-size-min", type=float, default=0.20)
    parser.add_argument("--foreground-occluder-size-max", type=float, default=0.36)
    parser.add_argument("--foreground-occluder-depth", type=float, default=0.45)
    parser.add_argument("--lighting-preset", default="default")
    parser.add_argument("--lighting-seed", type=int, default=0)
    parser.add_argument("--lighting-ambient", default=None)
    parser.add_argument("--lighting-directional", default=None)
    parser.add_argument("--lighting-jitter", type=float, default=0.0)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--incremental-manifest", action="store_true")
    parser.add_argument("--resume-manifest", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser.parse_args(argv)


def _parse_float_pair(raw: Optional[str]) -> Optional[tuple[float, float]]:
    if raw is None:
        return None
    parts = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("expected two comma-separated floats")
    return (parts[0], parts[1])


def _parse_color(raw: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("expected RGB color as R,G,B")
    return tuple(int(np.clip(part, 0, 255)) for part in parts)


def main(argv: Optional[list[str]] = None) -> int:
    ns = parse_args(argv)
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    process_args = ProcessArgs(
        input_dir=ns.input_dir.resolve(),
        output_dir=ns.output_dir.resolve(),
        recursive=bool(ns.recursive),
        filename_pattern=ns.filename_pattern,
        num_points=int(ns.num_points),
        num_views=int(ns.num_views),
        img_size=int(ns.img_size),
        camera_distance=float(ns.camera_distance),
        camera_jitter_degrees=float(ns.camera_jitter_degrees),
        camera_jitter_seed=int(ns.camera_jitter_seed),
        triangle_face_tol=float(ns.triangle_face_tol),
        angle_tol_rads=float(ns.angle_tol_rads),
        step_mesh_backend=str(ns.step_mesh_backend),
        step_mesh_fallback_backends=parse_fallback_backends(ns.step_mesh_fallback_backends),
        step_mesh_work_dir=ns.step_mesh_work_dir,
        step_mesh_format=str(ns.step_mesh_format),
        freecad_cmd=str(ns.freecad_cmd),
        step_mesh_timeout_s=ns.step_mesh_timeout_s,
        keep_intermediate_mesh=not bool(ns.discard_intermediate_mesh),
        normalize=str(ns.normalize),
        render_backend=str(ns.render_backend),
        blender_bin=str(ns.blender_bin),
        blender_script=ns.blender_script,
        blender_engine=str(ns.blender_engine),
        blender_samples=int(ns.blender_samples),
        blender_style=str(ns.blender_style),
        visualization_root=ns.visualization_root,
        blender_device=str(ns.blender_device),
        foreground_occluder=bool(ns.foreground_occluder),
        foreground_occluder_seed=int(ns.foreground_occluder_seed),
        foreground_occluder_color=_parse_color(ns.foreground_occluder_color),
        foreground_occluder_size_min=float(ns.foreground_occluder_size_min),
        foreground_occluder_size_max=float(ns.foreground_occluder_size_max),
        foreground_occluder_depth=float(ns.foreground_occluder_depth),
        lighting_preset=str(ns.lighting_preset),
        lighting_seed=int(ns.lighting_seed),
        lighting_ambient=_parse_float_pair(ns.lighting_ambient),
        lighting_directional=_parse_float_pair(ns.lighting_directional),
        lighting_jitter=float(ns.lighting_jitter),
        skip_existing=bool(ns.skip_existing),
    )
    tasks = _iter_tasks(process_args, resume_manifest=bool(ns.resume_manifest))
    if ns.max_tasks is not None:
        tasks = tasks[: max(0, int(ns.max_tasks))]
    started = __import__("time").time()
    ok, failed = _run_tasks(tasks, process_args, max(1, int(ns.num_processes)))
    summary = {
        "n_inputs": len(tasks),
        "n_ok": ok,
        "n_failed": failed,
        "manifest": str(ns.output_dir / "manifest.jsonl"),
        "failures": str(ns.output_dir / "failures.jsonl"),
        "elapsed_s": round(__import__("time").time() - started, 2),
    }
    _write_json(ns.output_dir / "summary.json", summary)
    _write_json(ns.output_dir / "progress.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
