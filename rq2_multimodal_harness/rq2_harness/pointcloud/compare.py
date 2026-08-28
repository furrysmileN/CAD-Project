from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .canonical import CanonicalTransform

DEFAULT_TAU = 0.02  # canonical 帧距离容差


def sample_step(step_path, n_points: int = 2048, seed: int = 42) -> np.ndarray:
    """从候选 STEP 面积加权采样点（复用 geometry._sample_shape 的逻辑）。"""
    try:
        import cadquery as cq
    except ImportError as exc:
        raise RuntimeError(f"dependency_unavailable:cadquery ({exc})") from exc
    from ..geometry import _sample_shape

    shape = cq.importers.importStep(str(step_path)).val()
    if shape is None or not shape.isValid():
        raise ValueError("invalid_step_shape")
    return np.asarray(_sample_shape(shape, int(n_points), int(seed)), dtype=np.float64)


def compare_cad_to_cloud(
    step_points_raw: np.ndarray,
    cloud_points_raw: np.ndarray,
    transform: CanonicalTransform,
    *,
    tau: float = DEFAULT_TAU,
    n_points: int = 2048,
    seed: int = 42,
) -> dict[str, Any]:
    """候选 STEP（raw 帧采样点）与输入点云（raw 帧）在 canonical 帧比较。

    输出（direction 均以「越大越好」重新表达，避免减法方向混淆）：
    source_coverage: 输入点云中距候选 ≤ tau 的比例（recall 侧）
    prediction_precision: 候选点中距输入点云 ≤ tau 的比例（precision 侧）
    fscore / chamfer / 双向距离分位数 / missing-extra 区域摘要。
    """
    pred = transform.forward(np.asarray(step_points_raw, dtype=np.float64))
    source = transform.forward(np.asarray(cloud_points_raw, dtype=np.float64))
    if not len(pred) or not len(source):
        return {"valid": False, "reason": "empty_input"}

    source_to_pred = cKDTree(pred).query(source, k=1)[0]
    pred_to_source = cKDTree(source).query(pred, k=1)[0]
    source_coverage = float((source_to_pred <= tau).mean())
    prediction_precision = float((pred_to_source <= tau).mean())
    f1 = (
        2.0 * source_coverage * prediction_precision / (source_coverage + prediction_precision)
        if source_coverage + prediction_precision > 0
        else 0.0
    )
    chamfer = float(source_to_pred.mean() + pred_to_source.mean())
    quantiles = [0.5, 0.9, 0.95]

    missing_mask = source_to_pred > tau
    extra_mask = pred_to_source > tau
    missing_ratio = float(missing_mask.mean())
    extra_ratio = float(extra_mask.mean())

    def _region_bbox(mask: np.ndarray, points: np.ndarray) -> dict[str, Any] | None:
        if not mask.any():
            return None
        low = points[mask].min(axis=0)
        high = points[mask].max(axis=0)
        return {
            "point_count": int(mask.sum()),
            "bbox_size": (high - low).tolist(),
            "center": ((low + high) / 2.0).tolist(),
        }

    return {
        "valid": True,
        "tau": tau,
        "source_coverage": source_coverage,
        "prediction_precision": prediction_precision,
        "fscore": f1,
        "chamfer": chamfer,
        "source_to_pred_quantiles": {
            f"p{int(percent * 100)}": float(np.quantile(source_to_pred, percent))
            for percent in quantiles
        },
        "pred_to_source_quantiles": {
            f"p{int(percent * 100)}": float(np.quantile(pred_to_source, percent))
            for percent in quantiles
        },
        "missing_region": {
            "ratio": missing_ratio,
            "bbox": _region_bbox(missing_mask, source),
        },
        "extra_region": {
            "ratio": extra_ratio,
            "bbox": _region_bbox(extra_mask, pred),
        },
    }


def localize_geometric_error(
    step_points_raw: np.ndarray,
    cloud_points_raw: np.ndarray,
    transform: CanonicalTransform,
    *,
    tau: float = DEFAULT_TAU,
    top_k: int = 3,
    n_points: int = 2048,
    seed: int = 42,
) -> dict[str, Any]:
    """missing/extra 区域的 top-k 聚类与局部 bbox（简单网格聚类）。"""
    pred = transform.forward(np.asarray(step_points_raw, dtype=np.float64))
    source = transform.forward(np.asarray(cloud_points_raw, dtype=np.float64))
    source_to_pred = cKDTree(pred).query(source, k=1)[0]
    pred_to_source = cKDTree(source).query(pred, k=1)[0]
    missing = source[source_to_pred > tau]
    extra = pred[pred_to_source > tau]

    def _cluster(points: np.ndarray) -> list[dict[str, Any]]:
        if not len(points):
            return []
        from scipy.spatial import cKDTree as Tree

        tree = Tree(points)
        _, neighbors = tree.query(points, k=min(8, len(points)))
        epsilon = float(np.median(np.abs(neighbors[:, 1:] - neighbors[:, :1]))) * 2.0
        parents = list(range(len(points)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        for pair in tree.query_pairs(r=max(epsilon, tau)):
            left, right = find(pair[0]), find(pair[1])
            if left != right:
                parents[left] = right
        groups: dict[int, list[int]] = {}
        for index in range(len(points)):
            groups.setdefault(find(index), []).append(index)
        clusters = []
        for indices in groups.values():
            if len(indices) < 4:
                continue
            local = points[indices]
            low = local.min(axis=0)
            high = local.max(axis=0)
            clusters.append(
                {
                    "point_count": len(indices),
                    "bbox_size": (high - low).tolist(),
                    "center": ((low + high) / 2.0).tolist(),
                }
            )
        clusters.sort(key=lambda item: item["point_count"], reverse=True)
        return clusters[:top_k]

    return {
        "tau": tau,
        "missing_clusters": _cluster(missing),
        "extra_clusters": _cluster(extra),
        "missing_total": int(len(missing)),
        "extra_total": int(len(extra)),
    }
