from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

METRICS_VERSION = "rq2.geometry.v2"


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not len(a) or not len(b):
        raise ValueError("Chamfer 输入不能为空")
    return float(cKDTree(a).query(b, k=1)[0].mean() + cKDTree(b).query(a, k=1)[0].mean())


def f_score(pred_points: np.ndarray, gt_points: np.ndarray, tau: float = 0.01) -> dict[str, float]:
    """F-score：pred 中距 GT 小于 tau 的比例为 precision，GT 中距 pred 小于 tau 的比例为 recall。"""
    a = np.asarray(pred_points, dtype=np.float64)
    b = np.asarray(gt_points, dtype=np.float64)
    if not len(a) or not len(b):
        raise ValueError("F-score 输入不能为空")
    dist_a_to_b = cKDTree(b).query(a, k=1)[0]
    dist_b_to_a = cKDTree(a).query(b, k=1)[0]
    precision = float((dist_a_to_b <= tau).mean())
    recall = float((dist_b_to_a <= tau).mean())
    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom > 0 else 0.0
    return {"tau": float(tau), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def canonicalize_points(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2.0
    scale = float(np.max(high - low))
    if scale <= 1e-12:
        raise ValueError("零尺度几何")
    return (points - center) / scale, {"center": center.tolist(), "longest_edge": scale, "size": (high - low).tolist()}


def bbox_metrics(pred_points: np.ndarray, gt_points: np.ndarray) -> dict[str, Any]:
    pred_low, pred_high = pred_points.min(axis=0), pred_points.max(axis=0)
    gt_low, gt_high = gt_points.min(axis=0), gt_points.max(axis=0)
    pred_size, gt_size = pred_high - pred_low, gt_high - gt_low
    pred_long, gt_long = float(pred_size.max()), float(gt_size.max())
    pred_aspect = pred_size / max(pred_long, 1e-12)
    gt_aspect = gt_size / max(gt_long, 1e-12)
    pred_center, gt_center = (pred_low + pred_high) / 2.0, (gt_low + gt_high) / 2.0
    return {
        "pred_size": pred_size.tolist(),
        "gt_size": gt_size.tolist(),
        "aspect_l1": float(np.abs(pred_aspect - gt_aspect).mean()),
        "scale_ratio": pred_long / max(gt_long, 1e-12),
        "scale_log_abs": abs(math.log(max(pred_long / max(gt_long, 1e-12), 1e-12))),
        "center_distance": float(np.linalg.norm(pred_center - gt_center)),
        "center_offset": (pred_center - gt_center).tolist(),
    }


def _sample_shape(shape: Any, n: int, seed: int) -> np.ndarray:
    vertices, faces = shape.tessellate(0.05)
    verts = np.asarray([[vertex.x, vertex.y, vertex.z] for vertex in vertices], dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if not len(verts):
        raise ValueError("STEP tessellation 无顶点")
    rng = np.random.default_rng(seed)
    if not len(triangles):
        return verts[rng.integers(0, len(verts), size=n)]
    v0, v1, v2 = verts[triangles[:, 0]], verts[triangles[:, 1]], verts[triangles[:, 2]]
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) / 2
    if areas.sum() <= 1e-12:
        return verts[rng.integers(0, len(verts), size=n)]
    chosen = rng.choice(len(triangles), size=n, p=areas / areas.sum())
    root = np.sqrt(rng.random(n))
    other = rng.random(n)
    return (
        (1 - root)[:, None] * v0[chosen]
        + (root * (1 - other))[:, None] * v1[chosen]
        + (root * other)[:, None] * v2[chosen]
    )


def _shape_mesh(shape: Any, center: np.ndarray, scale: float):
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("dependency_unavailable:trimesh") from exc
    vertices, faces = shape.tessellate(0.05)
    verts = (np.asarray([[vertex.x, vertex.y, vertex.z] for vertex in vertices], dtype=np.float64) - center) / scale
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64), process=False)


def _voxel_iou(pred_mesh: Any, gt_mesh: Any, resolution: int) -> float:
    pitch = 1.0 / float(resolution)
    pred_voxels = pred_mesh.voxelized(pitch).fill().points
    gt_voxels = gt_mesh.voxelized(pitch).fill().points
    pred_set = {tuple(index) for index in np.rint(pred_voxels / pitch).astype(np.int32)}
    gt_set = {tuple(index) for index in np.rint(gt_voxels / pitch).astype(np.int32)}
    union = pred_set | gt_set
    return len(pred_set & gt_set) / len(union) if union else 0.0


def invalid_metrics(reason: str) -> dict[str, Any]:
    return {
        "valid": False,
        "failure": reason,
        "shape_only_cd": None,
        "common_frame_cd": None,
        "bbox": None,
        "voxel_iou": {"status": "not_computed", "value": None, "reason": "invalid_geometry"},
        "fscore_shape": None,
        "fscore_common": None,
        "shape_voxel_iou": {"status": "not_computed", "value": None, "reason": "invalid_geometry"},
        "joint_quality": 0.0,
        "metrics_version": METRICS_VERSION,
    }


def score_step_pair(
    pred_step: str | Path,
    gt_step: str | Path,
    *,
    n_points: int = 2048,
    seed: int = 42,
    voxel_resolution: int = 48,
    tau: float = 0.25,
    fscore_tau: float = 0.01,
) -> dict[str, Any]:
    try:
        import cadquery as cq

        pred_shape = cq.importers.importStep(str(pred_step)).val()
        gt_shape = cq.importers.importStep(str(gt_step)).val()
        if pred_shape is None or gt_shape is None or not pred_shape.isValid() or not gt_shape.isValid():
            return invalid_metrics("invalid_step_shape")
        pred_points = _sample_shape(pred_shape, n_points, seed)
        # Common random numbers make the identity case exactly zero while preserving determinism.
        gt_points = _sample_shape(gt_shape, n_points, seed)
        pred_shape_only, pred_meta = canonicalize_points(pred_points)
        gt_canonical, gt_meta = canonicalize_points(gt_points)
        shape_cd = chamfer_distance(pred_shape_only, gt_canonical)

        gt_center = np.asarray(gt_meta["center"])
        gt_scale = float(gt_meta["longest_edge"])
        # Harness plans are already expressed in the canonical GT frame.
        pred_common = pred_points
        common_cd = chamfer_distance(pred_common, gt_canonical)
        bbox = bbox_metrics(pred_common, gt_canonical)
        fscore_shape = f_score(pred_shape_only, gt_canonical, fscore_tau)
        fscore_common = f_score(pred_common, gt_canonical, fscore_tau)
        try:
            pred_mesh = _shape_mesh(pred_shape, np.zeros(3), 1.0)
            gt_mesh = _shape_mesh(gt_shape, gt_center, gt_scale)
            voxel_value = _voxel_iou(pred_mesh, gt_mesh, voxel_resolution)
            voxel = {"status": "ok", "value": voxel_value, "resolution": voxel_resolution}
        except Exception as exc:
            voxel_value = None
            voxel = {"status": "degraded", "value": None, "reason": str(exc)[:300]}
        try:
            pred_center = np.asarray(pred_meta["center"])
            pred_scale = float(pred_meta["longest_edge"])
            pred_shape_mesh = _shape_mesh(pred_shape, pred_center, pred_scale)
            gt_shape_mesh = _shape_mesh(gt_shape, gt_center, gt_scale)
            shape_voxel_value = _voxel_iou(pred_shape_mesh, gt_shape_mesh, voxel_resolution)
            shape_voxel = {"status": "ok", "value": shape_voxel_value, "resolution": voxel_resolution}
        except Exception as exc:
            shape_voxel = {"status": "degraded", "value": None, "reason": str(exc)[:300]}
        distance_quality = math.exp(-0.5 * (shape_cd + common_cd) / max(tau, 1e-12))
        joint = distance_quality * (0.5 + 0.5 * voxel_value) if voxel_value is not None else distance_quality
        return {
            "valid": True,
            "shape_only_cd": shape_cd,
            "common_frame_cd": common_cd,
            "bbox": bbox,
            "voxel_iou": voxel,
            "fscore_shape": fscore_shape,
            "fscore_common": fscore_common,
            "shape_voxel_iou": shape_voxel,
            "joint_quality": float(joint),
            "normalization": {"pred_shape_only": pred_meta, "gt_to_common": gt_meta},
            "failure_aware_tau": tau,
            "metrics_version": METRICS_VERSION,
        }
    except Exception as exc:
        return invalid_metrics(f"{type(exc).__name__}:{str(exc)[:300]}")
