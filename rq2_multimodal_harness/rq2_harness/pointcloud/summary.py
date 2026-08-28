from __future__ import annotations

from typing import Any

import numpy as np

from .canonical import CanonicalTransform
from .io import PointCloudError


def summarize(points: np.ndarray, transform: CanonicalTransform) -> dict[str, Any]:
    """bbox / PCA 主轴 / 密度摘要（输入为 canonical 或 raw 均可，统一在 raw 帧描述）。

    返回字段均在 raw 帧；`canonical` 子块给出变换后的 bbox 尺寸（最长边 1）。
    """
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        raise PointCloudError("空点云")
    low = points.min(axis=0)
    high = points.max(axis=0)
    size = high - low
    center = (low + high) / 2.0
    longest = float(size.max())

    cov = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    principal_axes = eigenvectors[:, order].T  # 行向量，第一行为主轴
    extents = []
    for axis in principal_axes:
        projection = points @ axis
        extents.append(float(projection.max() - projection.min()))
    extents = np.asarray(extents)

    canonical = transform.forward(points)
    canonical_low = canonical.min(axis=0)
    canonical_high = canonical.max(axis=0)
    canonical_size = canonical_high - canonical_low

    tree = None
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    median_nn = float(np.median(distances[:, 1])) if len(points) > 1 else 0.0

    return {
        "point_count": int(len(points)),
        "bbox": {
            "min": low.tolist(),
            "max": high.tolist(),
            "center": center.tolist(),
            "size": size.tolist(),
            "longest_edge": longest,
        },
        "canonical_bbox": {
            "center": canonical_low.tolist(),
            "size": canonical_size.tolist(),
        },
        "principal_axes": principal_axes.tolist(),
        "axis_extents": extents.tolist(),
        "aspect_ratios": (extents / max(longest, 1e-12)).tolist(),
        "eigenvalue_ratios": (eigenvalues[order] / max(float(eigenvalues[order][0]), 1e-12)).tolist(),
        "median_nn_distance": median_nn,
    }
