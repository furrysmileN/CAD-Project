"""Fixed four-view RGB renders from STEP. CadQuery tessellation + software shading (no browser)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .common import sha256_file
from .encoding_visual import CAMERAS, camera_basis, canonicalize_points

VIEWS = ("front", "side", "top", "isometric")


def _shade_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    direction: Sequence[float],
    *,
    resolution: int = 512,
    padding: float = 0.06,
) -> Image.Image:
    right, up, forward = camera_basis(direction)
    extent = 0.5 + float(padding)
    x = vertices @ right
    y = vertices @ up
    depth = vertices @ forward
    px = np.rint((x + extent) / (2 * extent) * (resolution - 1)).astype(np.int64)
    py = np.rint((extent - y) / (2 * extent) * (resolution - 1)).astype(np.int64)
    zbuffer = np.full((resolution, resolution), -np.inf)
    color = np.full((resolution, resolution, 3), 245, dtype=np.uint8)
    light = np.asarray(direction, dtype=np.float64)
    light = light / max(np.linalg.norm(light), 1e-9)
    for tri in faces:
        pts = vertices[tri]
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        lambert = float(max(0.18, abs(n @ light)))
        shade = np.array([40, 70, 110], dtype=np.float64) * lambert + 30
        tx, ty, tz = px[tri], py[tri], depth[tri]
        if not ((tx >= 0) & (tx < resolution) & (ty >= 0) & (ty < resolution)).all():
            minx, maxx = max(0, int(tx.min())), min(resolution - 1, int(tx.max()))
            miny, maxy = max(0, int(ty.min())), min(resolution - 1, int(ty.max()))
        else:
            minx, maxx = max(0, int(tx.min())), min(resolution - 1, int(tx.max()))
            miny, maxy = max(0, int(ty.min())), min(resolution - 1, int(ty.max()))
        if maxx < minx or maxy < miny:
            continue
        denom = (ty[1] - ty[2]) * (tx[0] - tx[2]) + (tx[2] - tx[1]) * (ty[0] - ty[2])
        if abs(float(denom)) < 1e-12:
            continue
        yy, xx = np.mgrid[miny : maxy + 1, minx : maxx + 1]
        w0 = ((ty[1] - ty[2]) * (xx - tx[2]) + (tx[2] - tx[1]) * (yy - ty[2])) / denom
        w1 = ((ty[2] - ty[0]) * (xx - tx[2]) + (tx[0] - tx[2]) * (yy - ty[2])) / denom
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        local_z = w0 * tz[0] + w1 * tz[1] + w2 * tz[2]
        region = zbuffer[miny : maxy + 1, minx : maxx + 1]
        better = inside & (local_z > region)
        region[better] = local_z[better]
        color[miny : maxy + 1, minx : maxx + 1][better] = shade.clip(0, 255).astype(np.uint8)
    return Image.fromarray(color, "RGB")


def render_step_views(
    step_path: str | Path,
    output_dir: str | Path,
    *,
    resolution: int = 512,
    padding: float = 0.06,
) -> list[dict[str, Any]]:
    import cadquery as cq

    step_path = Path(step_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = cq.importers.importStep(str(step_path)).val()
    if shape is None or not shape.isValid():
        raise ValueError(f"invalid STEP: {step_path}")
    vertices, faces = shape.tessellate(0.03)
    raw = np.asarray([[v.x, v.y, v.z] for v in vertices], dtype=np.float64)
    canonical, _ = canonicalize_points(raw)
    tris = np.asarray(faces, dtype=np.int64)
    views = []
    for name in VIEWS:
        path = output_dir / f"{name}.png"
        image = _shade_mesh(canonical, tris, CAMERAS[name], resolution=resolution, padding=padding)
        image.save(path, "PNG", optimize=False, compress_level=9)
        views.append({"view": name, "path": str(path.resolve()), "sha256": sha256_file(path)})
    return views
