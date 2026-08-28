from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .common import atomic_write_json, sha256_file, sha256_json


CAMERAS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, -1.0, 0.0),
    "side": (1.0, 0.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "isometric": (1.0, -1.0, 1.0),
}


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"点云必须是非空 (N,3)，实际为 {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("点云包含 NaN 或 Inf")
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    scale = float(np.max(high - low))
    if scale <= 1e-12:
        raise ValueError("点云 bbox 尺度为零")
    normalized = (points - center) / scale
    return normalized, {"center": center.tolist(), "scale": scale}


def _camera_basis(direction: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = np.asarray(direction, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up_hint))) > 0.95:
        up_hint = np.array([0.0, 1.0, 0.0])
    right = np.cross(up_hint, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return right, up, forward


def render_depth_contour(
    normalized: np.ndarray,
    direction: tuple[float, float, float],
    *,
    resolution: int,
    padding: float,
) -> Image.Image:
    right, up, forward = _camera_basis(direction)
    xy = np.column_stack((normalized @ right, normalized @ up))
    depth = normalized @ forward
    extent = 0.5 + padding
    px = np.rint((xy[:, 0] + extent) / (2.0 * extent) * (resolution - 1)).astype(int)
    py = np.rint((extent - xy[:, 1]) / (2.0 * extent) * (resolution - 1)).astype(int)
    valid = (px >= 0) & (px < resolution) & (py >= 0) & (py < resolution)
    px, py, depth = px[valid], py[valid], depth[valid]

    zbuffer = np.full((resolution, resolution), -np.inf, dtype=np.float64)
    np.maximum.at(zbuffer, (py, px), depth)
    occupied = np.isfinite(zbuffer)
    # A deterministic one-pixel dilation makes a 2K cloud legible without interpolation.
    dilated = occupied.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        dilated[max(0, dy):resolution + min(0, dy), max(0, dx):resolution + min(0, dx)] |= occupied[
            max(0, -dy):resolution - max(0, dy), max(0, -dx):resolution - max(0, dx)
        ]
    filled_depth = zbuffer.copy()
    if occupied.any():
        dmin, dmax = float(depth.min()), float(depth.max())
        gray = np.zeros_like(zbuffer, dtype=np.uint8)
        gray[occupied] = np.rint(48 + 180 * (zbuffer[occupied] - dmin) / max(dmax - dmin, 1e-12)).astype(np.uint8)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            target = filled_depth[max(0, dy):resolution + min(0, dy), max(0, dx):resolution + min(0, dx)]
            source = zbuffer[max(0, -dy):resolution - max(0, dy), max(0, -dx):resolution - max(0, dx)]
            np.maximum(target, source, out=target)
        gray[dilated & ~occupied] = 36
    else:
        gray = np.zeros_like(zbuffer, dtype=np.uint8)

    interior = occupied.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = np.zeros_like(occupied)
        neighbor[max(0, dy):resolution + min(0, dy), max(0, dx):resolution + min(0, dx)] = occupied[
            max(0, -dy):resolution - max(0, dy), max(0, -dx):resolution - max(0, dx)
        ]
        interior &= neighbor
    contour = occupied & ~interior
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    rgb[contour] = np.array([40, 220, 255], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def encode_point_cloud(
    npy_path: Path,
    cache_dir: Path,
    *,
    views: list[str],
    resolution: int,
    padding: float,
    encoding_version: str,
    force: bool = False,
) -> dict[str, Any]:
    input_hash = sha256_file(npy_path)
    params = {
        "encoding_version": encoding_version,
        "views": views,
        "resolution": int(resolution),
        "padding": float(padding),
        "cameras": {name: CAMERAS[name] for name in views},
    }
    cache_key = sha256_json({"input_sha256": input_hash, "params": params})
    metadata_path = cache_dir / f"{npy_path.stem}.json"
    if metadata_path.is_file() and not force:
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == cache_key and all(Path(item["path"]).is_file() for item in cached["images"]):
            return cached

    points = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    normalized, normalization = normalize_points(points)
    cache_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for view in views:
        if view not in CAMERAS:
            raise ValueError(f"未知点云相机 {view!r}")
        output_path = cache_dir / f"{npy_path.stem}__{view}.png"
        image = render_depth_contour(normalized, CAMERAS[view], resolution=resolution, padding=padding)
        image.save(output_path, format="PNG", optimize=False, compress_level=9)
        images.append({"view": view, "path": str(output_path.resolve()), "sha256": sha256_file(output_path)})
    metadata = {
        "source_path": str(npy_path.resolve()),
        "source_sha256": input_hash,
        "cache_key": cache_key,
        "params": params,
        "normalization": normalization,
        "images": images,
    }
    atomic_write_json(metadata_path, metadata)
    return metadata
