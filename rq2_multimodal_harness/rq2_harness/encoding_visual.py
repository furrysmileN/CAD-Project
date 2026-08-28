"""Deterministic visual encodings for the 20 x 63 experiment.

This module deliberately does not import or alter ``pointcloud.py``: the old
experiment remains reproducible while P3 v2 uses the corrected fixed range.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .common import atomic_write_json, sha256_file, sha256_json


CAMERAS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, -1.0, 0.0),
    "side": (1.0, 0.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "isometric": (1.0, -1.0, 1.0),
}
P3_VERSION = "v2-fixed-range"
DEPTH_RANGE = (-0.5, 0.5)

# Explicit, frozen turbo-like RGB table. Spatial values are never interpolated.
TURBO_LIKE_LUT = np.asarray(
    [
        (48, 18, 59), (65, 67, 165), (57, 120, 220), (33, 173, 211),
        (42, 209, 159), (122, 229, 86), (202, 219, 54), (246, 177, 39),
        (239, 103, 26), (190, 43, 24), (122, 4, 3),
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class VisualEncodingConfig:
    resolution: int = 512
    padding: float = 0.06
    views: tuple[str, ...] = ("front", "side", "top", "isometric")

    def validate(self) -> None:
        if self.resolution <= 0:
            raise ValueError("resolution 必须为正整数")
        if not 0 <= self.padding < 0.5:
            raise ValueError("padding 必须在 [0, 0.5) 内")
        unknown = set(self.views) - set(CAMERAS)
        if unknown:
            raise ValueError(f"未知相机: {sorted(unknown)}")


def canonicalize_points(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"输入必须为非空 (N,3)，实际为 {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("输入包含 NaN 或 Inf")
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2.0
    longest = float(np.max(high - low))
    if longest <= 1e-12:
        raise ValueError("canonical bbox 尺度为零")
    return (points - center) / longest, {
        "source_center": center.tolist(),
        "source_longest_edge": longest,
        "canonical_center": [0.0, 0.0, 0.0],
        "canonical_longest_edge": 1.0,
    }


def camera_basis(direction: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-12:
        raise ValueError("相机方向不能为零")
    forward /= norm
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ up_hint)) > 0.95:
        up_hint = np.array([0.0, 1.0, 0.0])
    right = np.cross(up_hint, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return right, up, forward


def _project(
    points: np.ndarray, direction: Sequence[float], resolution: int, padding: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    right, up, forward = camera_basis(direction)
    extent = 0.5 + float(padding)
    x = points @ right
    y = points @ up
    depth = points @ forward
    px = np.rint((x + extent) / (2 * extent) * (resolution - 1)).astype(np.int64)
    py = np.rint((extent - y) / (2 * extent) * (resolution - 1)).astype(np.int64)
    valid = (px >= 0) & (px < resolution) & (py >= 0) & (py < resolution)
    return px, py, depth, valid


def render_p1(
    canonical_points: np.ndarray,
    direction: Sequence[float],
    *,
    resolution: int = 512,
    padding: float = 0.06,
) -> Image.Image:
    """White background, fixed 2x2 black point glyphs, no interpolation."""
    px, py, _, valid = _project(canonical_points, direction, resolution, padding)
    array = np.full((resolution, resolution, 3), 255, dtype=np.uint8)
    for x, y in zip(px[valid], py[valid]):
        array[y:min(y + 2, resolution), x:min(x + 2, resolution)] = 0
    return Image.fromarray(array, "RGB")


def render_p2(
    canonical_points: np.ndarray,
    direction: Sequence[float],
    *,
    resolution: int = 512,
    padding: float = 0.06,
) -> Image.Image:
    """Discrete depth-coloured points on white; no lines or hole filling."""
    px, py, depth, valid = _project(canonical_points, direction, resolution, padding)
    px, py, depth = px[valid], py[valid], depth[valid]
    image = np.full((resolution, resolution, 3), 255, dtype=np.uint8)
    if len(depth):
        indices = np.rint(
            np.clip((depth - DEPTH_RANGE[0]) / (DEPTH_RANGE[1] - DEPTH_RANGE[0]), 0, 1)
            * (len(TURBO_LIKE_LUT) - 1)
        ).astype(np.int64)
        # Resolve pixel collisions by the same deterministic nearest-camera z-buffer as P3.
        order = np.lexsort((np.arange(len(depth)), depth))
        image[py[order], px[order]] = TURBO_LIKE_LUT[indices[order]]
    return Image.fromarray(image, "RGB")


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    result = np.zeros_like(mask)
    result[max(0, dy):mask.shape[0] + min(0, dy), max(0, dx):mask.shape[1] + min(0, dx)] = mask[
        max(0, -dy):mask.shape[0] - max(0, dy), max(0, -dx):mask.shape[1] - max(0, dx)
    ]
    return result


def render_p3(
    canonical_points: np.ndarray,
    direction: Sequence[float],
    *,
    resolution: int = 512,
    padding: float = 0.06,
) -> Image.Image:
    """Legacy z-buffer/dilation/cyan contour morphology with fixed depth."""
    px, py, depth, valid = _project(canonical_points, direction, resolution, padding)
    px, py, depth = px[valid], py[valid], depth[valid]
    zbuffer = np.full((resolution, resolution), -np.inf, dtype=np.float64)
    np.maximum.at(zbuffer, (py, px), depth)
    occupied = np.isfinite(zbuffer)
    dilated = occupied.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        dilated |= _shift(occupied, dy, dx)
    gray = np.zeros((resolution, resolution), dtype=np.uint8)
    if occupied.any():
        normalized = np.clip(
            (zbuffer[occupied] - DEPTH_RANGE[0]) / (DEPTH_RANGE[1] - DEPTH_RANGE[0]), 0, 1
        )
        gray[occupied] = np.rint(48 + 180 * normalized).astype(np.uint8)
        gray[dilated & ~occupied] = 36
    interior = occupied.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        interior &= _shift(occupied, dy, dx)
    contour = occupied & ~interior
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    rgb[contour] = (40, 220, 255)
    return Image.fromarray(rgb, "RGB")


POINT_RENDERERS = {"P1": render_p1, "P2": render_p2, "P3": render_p3}


def image_qc(path_or_image: Path | Image.Image, *, expected_size: int | None = None) -> dict[str, Any]:
    close = not isinstance(path_or_image, Image.Image)
    image = Image.open(path_or_image) if close else path_or_image
    try:
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        corners = np.asarray(
            [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]],
            dtype=np.int16,
        )
        background = np.median(corners, axis=0)
        # Blender colour management/dithering can vary a nominally uniform
        # background by a few integer levels, so exact equality is not a
        # reliable foreground test.
        occupied = np.max(
            np.abs(rgb.astype(np.int16) - background),
            axis=2,
        ) > 8
        ys, xs = np.nonzero(occupied)
        bbox = None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        return {
            "size": [width, height],
            "size_ok": expected_size is None or (width == expected_size and height == expected_size),
            "nonempty": bool(len(xs)),
            "bbox": bbox,
            "occupied_pixels": int(occupied.sum()),
        }
    finally:
        if close:
            image.close()


def bbox_iou(a: Sequence[int] | None, b: Sequence[int] | None) -> float:
    if a is None or b is None:
        return 0.0
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _save_encoding(
    source: Path,
    cache_dir: Path,
    encoding: str,
    config: VisualEncodingConfig,
    *,
    force: bool,
) -> dict[str, Any]:
    source_hash = sha256_file(source)
    params = {
        "encoding": encoding,
        "version": P3_VERSION if encoding == "P3" else "v1",
        **asdict(config),
        "cameras": {view: CAMERAS[view] for view in config.views},
        "depth_range": list(DEPTH_RANGE) if encoding in {"P2", "P3"} else None,
    }
    key = sha256_json({"source_sha256": source_hash, "params": params})
    metadata_path = cache_dir / f"{source.stem}__{encoding}.json"
    if metadata_path.is_file() and not force:
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == key and all(Path(i["path"]).is_file() for i in cached["images"]):
            return cached
    points, normalization = canonicalize_points(np.load(source, mmap_mode="r", allow_pickle=False))
    cache_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for view in config.views:
        path = cache_dir / f"{source.stem}__{encoding}__{view}.png"
        image = POINT_RENDERERS[encoding](
            points, CAMERAS[view], resolution=config.resolution, padding=config.padding
        )
        image.save(path, "PNG", optimize=False, compress_level=9)
        images.append({
            "view": view, "path": str(path.resolve()), "sha256": sha256_file(path),
            "qc": image_qc(image, expected_size=config.resolution),
        })
    metadata = {
        "schema_version": "rq2.visual_encoding.v1",
        "source_path": str(source.resolve()), "source_sha256": source_hash,
        "cache_key": key, "params": params, "normalization": normalization, "images": images,
    }
    metadata["metadata_sha256"] = sha256_json(metadata)
    atomic_write_json(metadata_path, metadata)
    return metadata


def render_i3(
    step_path: Path,
    output_path: Path,
    direction: Sequence[float],
    *,
    resolution: int = 512,
    padding: float = 0.06,
    edge_samples: int = 96,
) -> dict[str, Any]:
    """Render exact BRep edges with triangle depth used only for visibility.

    Tessellation edges are never drawn. This is the documented fallback for
    OCP HLR API incompatibilities across CadQuery/OCP releases.
    """
    try:
        import cadquery as cq
    except ImportError as exc:
        raise RuntimeError("I3 需要 CadQuery/OCP") from exc
    shape = cq.importers.importStep(str(step_path)).val()
    vertices, faces = shape.tessellate(0.02)
    raw = np.asarray([[v.x, v.y, v.z] for v in vertices], dtype=np.float64)
    canonical, normalization = canonicalize_points(raw)
    px, py, depth, valid = _project(canonical, direction, resolution, padding)
    zbuffer = np.full((resolution, resolution), -np.inf)
    triangles = np.asarray(faces, dtype=np.int64)
    # Conservative software rasterization of triangle interiors for occlusion only.
    for tri in triangles:
        tx, ty, tz = px[tri], py[tri], depth[tri]
        if not valid[tri].all():
            continue
        minx, maxx = max(0, int(tx.min())), min(resolution - 1, int(tx.max()))
        miny, maxy = max(0, int(ty.min())), min(resolution - 1, int(ty.max()))
        denom = (ty[1] - ty[2]) * (tx[0] - tx[2]) + (tx[2] - tx[1]) * (ty[0] - ty[2])
        if abs(float(denom)) < 1e-12:
            continue
        yy, xx = np.mgrid[miny:maxy + 1, minx:maxx + 1]
        w0 = ((ty[1] - ty[2]) * (xx - tx[2]) + (tx[2] - tx[1]) * (yy - ty[2])) / denom
        w1 = ((ty[2] - ty[0]) * (xx - tx[2]) + (tx[0] - tx[2]) * (yy - ty[2])) / denom
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        local_z = w0 * tz[0] + w1 * tz[1] + w2 * tz[2]
        region = zbuffer[miny:maxy + 1, minx:maxx + 1]
        region[inside] = np.maximum(region[inside], local_z[inside])
    canvas = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(canvas)
    center = np.asarray(normalization["source_center"])
    scale = float(normalization["source_longest_edge"])
    drawn = 0
    for edge in shape.Edges():
        samples, _ = edge.sample(max(2, edge_samples))
        edge_points = (np.asarray([[p.x, p.y, p.z] for p in samples]) - center) / scale
        ex, ey, ez, ev = _project(edge_points, direction, resolution, padding)
        visible = ev & (ez >= zbuffer[np.clip(ey, 0, resolution - 1), np.clip(ex, 0, resolution - 1)] - 0.006)
        run: list[tuple[int, int]] = []
        for x, y, is_visible in zip(ex, ey, visible):
            if is_visible:
                run.append((int(x), int(y)))
            else:
                if len(run) > 1:
                    draw.line(run, fill=(20, 20, 20), width=1)
                    drawn += 1
                run = []
        if len(run) > 1:
            draw.line(run, fill=(20, 20, 20), width=1)
            drawn += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "PNG", optimize=False, compress_level=9)
    return {"normalization": normalization, "visible_edge_runs": drawn}


def _prepare_i3(
    source: Path, cache_dir: Path, config: VisualEncodingConfig, force: bool
) -> dict[str, Any]:
    source_hash = sha256_file(source)
    params = {"encoding": "I3", "version": "brep-edge-visibility-v1", **asdict(config),
              "cameras": {v: CAMERAS[v] for v in config.views}}
    key = sha256_json({"source_sha256": source_hash, "params": params})
    meta_path = cache_dir / f"{source.stem}__I3.json"
    if meta_path.is_file() and not force:
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == key and all(Path(i["path"]).is_file() for i in cached["images"]):
            return cached
    images = []
    for view in config.views:
        path = cache_dir / f"{source.stem}__I3__{view}.png"
        detail = render_i3(source, path, CAMERAS[view], resolution=config.resolution, padding=config.padding)
        images.append({"view": view, "path": str(path.resolve()), "sha256": sha256_file(path),
                       "qc": image_qc(path, expected_size=config.resolution), "detail": detail})
    meta = {"schema_version": "rq2.visual_encoding.v1", "source_path": str(source.resolve()),
            "source_sha256": source_hash, "cache_key": key, "params": params, "images": images}
    meta["metadata_sha256"] = sha256_json(meta)
    atomic_write_json(meta_path, meta)
    return meta


def _prepare_blender(
    obj_path: Path,
    cache_dir: Path,
    encoding: str,
    config: VisualEncodingConfig,
    blender_executable: str,
    blender_script: Path,
    force: bool,
) -> dict[str, Any]:
    source_hash = sha256_file(obj_path)
    params = {"encoding": encoding, "version": "blender-encoding-v1", **asdict(config),
              "cameras": {v: CAMERAS[v] for v in config.views}}
    key = sha256_json({"source_sha256": source_hash, "script_sha256": sha256_file(blender_script), "params": params})
    meta_path = cache_dir / f"{obj_path.stem}__{encoding}.json"
    if meta_path.is_file() and not force:
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == key and all(Path(i["path"]).is_file() for i in cached["images"]):
            return cached
    cache_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        blender_executable, "-b", "-P", str(blender_script), "--", "--obj", str(obj_path),
        "--outdir", str(cache_dir), "--encoding", encoding, "--size", str(config.resolution),
        "--padding", str(config.padding), "--views", ",".join(config.views),
    ], check=True)
    images = []
    for view in config.views:
        path = cache_dir / f"{obj_path.stem}__{encoding}__{view}.png"
        images.append({"view": view, "path": str(path.resolve()), "sha256": sha256_file(path),
                       "qc": image_qc(path, expected_size=config.resolution)})
    meta = {"schema_version": "rq2.visual_encoding.v1", "source_path": str(obj_path.resolve()),
            "source_sha256": source_hash, "cache_key": key, "params": params, "images": images}
    meta["metadata_sha256"] = sha256_json(meta)
    atomic_write_json(meta_path, meta)
    return meta


def prepare_visual_encodings(
    *,
    cache_dir: str | Path,
    pointcloud_path: str | Path | None = None,
    obj_path: str | Path | None = None,
    step_path: str | Path | None = None,
    encodings: Iterable[str] = ("P1", "P2", "P3", "I1", "I2", "I3"),
    views: Iterable[str] = ("front", "side", "top", "isometric"),
    resolution: int = 512,
    padding: float = 0.06,
    blender_executable: str | None = None,
    blender_script: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare selected encodings with content-addressed, idempotent caching."""
    selected = tuple(encodings)
    unknown = set(selected) - {"P1", "P2", "P3", "I1", "I2", "I3"}
    if unknown:
        raise ValueError(f"未知视觉编码: {sorted(unknown)}")
    config = VisualEncodingConfig(resolution, padding, tuple(views))
    config.validate()
    cache = Path(cache_dir)
    results: dict[str, Any] = {}
    for encoding in selected:
        if encoding.startswith("P"):
            if pointcloud_path is None:
                raise ValueError(f"{encoding} 需要 pointcloud_path")
            results[encoding] = _save_encoding(Path(pointcloud_path), cache, encoding, config, force=force)
        elif encoding == "I3":
            if step_path is None:
                raise ValueError("I3 需要 step_path")
            results[encoding] = _prepare_i3(Path(step_path), cache, config, force)
        else:
            if obj_path is None or not blender_executable:
                raise ValueError(f"{encoding} 需要 obj_path 和 blender_executable")
            script = Path(blender_script) if blender_script else Path(__file__).parents[1] / "scripts" / "blender_encoding_render.py"
            results[encoding] = _prepare_blender(
                Path(obj_path), cache, encoding, config, blender_executable, script, force
            )
    pair_qc = []
    for image_encoding in (name for name in selected if name.startswith("I")):
        for point_encoding in (name for name in selected if name.startswith("P")):
            image_by_view = {i["view"]: i for i in results[image_encoding]["images"]}
            point_by_view = {i["view"]: i for i in results[point_encoding]["images"]}
            for view in config.views:
                pair_qc.append({
                    "image_encoding": image_encoding, "point_encoding": point_encoding, "view": view,
                    "bbox_iou": bbox_iou(image_by_view[view]["qc"]["bbox"], point_by_view[view]["qc"]["bbox"]),
                })
    response = {"schema_version": "rq2.visual_bundle.v1", "config": asdict(config),
                "encodings": results, "image_point_bbox_iou": pair_qc}
    response["bundle_sha256"] = sha256_json(response)
    return response
