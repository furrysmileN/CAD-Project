from __future__ import annotations

import numpy as np


class PointCloudError(ValueError):
    """点云输入非法。"""


def load_point_cloud(path) -> np.ndarray:
    """读取 .npy 点云（mmap），返回 float64 (N,3)。"""
    try:
        points = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise PointCloudError(f"无法读取点云 {path}: {exc}") from exc
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise PointCloudError(f"点云必须是 (N,3) 数组，实际 shape={points.shape}")
    return np.ascontiguousarray(points[:, :3])


def clean_points(points: np.ndarray) -> tuple[np.ndarray, dict]:
    """清洗点云并报告质量：NaN/Inf 过滤、重复点去重、退化检查。

    返回 (cleaned_points, quality)，quality 字段：
    raw_count / valid_count / valid_ratio / duplicate_removed /
    finite_removed / degenerate（退化为点/线时为 True）。
    """
    points = np.asarray(points, dtype=np.float64)
    raw_count = len(points)
    finite = np.isfinite(points).all(axis=1)
    cleaned = points[finite]
    finite_removed = raw_count - int(finite.sum())
    if not len(cleaned):
        raise PointCloudError("点云没有有效点（全部为 NaN/Inf）")
    cleaned = np.unique(cleaned, axis=0)
    duplicate_removed = raw_count - finite_removed - len(cleaned)
    span = cleaned.max(axis=0) - cleaned.min(axis=0)
    span = span[span > 0]
    degenerate = len(span) < 3
    quality = {
        "raw_count": raw_count,
        "valid_count": int(len(cleaned)),
        "valid_ratio": float(len(cleaned) / raw_count) if raw_count else 0.0,
        "finite_removed": int(finite_removed),
        "duplicate_removed": int(duplicate_removed),
        "degenerate": bool(degenerate),
    }
    return cleaned, quality


def hash_points(points: np.ndarray) -> str:
    """点云内容的确定性哈希（排序后按字节），用于 cloud_id。"""
    import hashlib

    payload = np.ascontiguousarray(np.sort(points, axis=0).astype("<f8"))
    return hashlib.sha256(payload.tobytes()).hexdigest()
