from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_K = 16


def estimate_normals(points: np.ndarray, k: int = DEFAULT_K) -> np.ndarray:
    """k 邻域 PCA 法向估计，按远离质心方向定向（确定性，无需 seed）。

    返回 (N,3) 单位法向。
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError(f"点数过少无法估计法向: {len(points)}")
    k = min(int(k), len(points) - 1)
    tree = cKDTree(points)
    _, neighbors = tree.query(points, k=k + 1)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    centroid = points.mean(axis=0)
    normals = np.empty((len(points), 3), dtype=np.float64)
    for index in range(len(points)):
        local = points[neighbors[index]]
        local = local - local.mean(axis=0)
        _, _, vh = np.linalg.svd(local, full_matrices=False)
        normal = vh[-1]
        if np.dot(normal, points[index] - centroid) < 0:
            normal = -normal
        normals[index] = normal
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / np.where(norms > 0, norms, 1.0)[:, None]
    return normals


def normal_summary(normals: np.ndarray) -> dict[str, Any]:
    """主法向方向（聚类到 PCA 轴附近的占比）。"""
    normals = np.asarray(normals, dtype=np.float64)
    axes = np.eye(3)
    scores = {}
    for axis_index, axis in enumerate(axes):
        agreement = np.abs(normals @ axis) > 0.9
        scores[f"axis_{axis_index}"] = float(agreement.mean())
    dominant_axis = max(range(3), key=lambda index: scores[f"axis_{index}"])
    return {
        "available": True,
        "dominant_axis": dominant_axis,
        "dominant_axis_ratio": scores[f"axis_{dominant_axis}"],
        "axis_ratios": scores,
    }
