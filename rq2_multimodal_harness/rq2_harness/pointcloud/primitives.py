from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_RANSAC_SEED = 42
DEFAULT_PLANE_TOLERANCE = 0.012  # canonical 帧下的距离容差
DEFAULT_MAX_ITERATIONS = 1500


def fit_planes(
    points: np.ndarray,
    *,
    tolerance: float = DEFAULT_PLANE_TOLERANCE,
    max_planes: int = 2,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    seed: int = DEFAULT_RANSAC_SEED,
) -> list[dict[str, Any]]:
    """RANSAC 主平面拟合（points 应在 canonical 帧）。

    返回按 support_ratio 降序的平面列表，每个：
    id / normal（单位向量）/ offset（沿法向到原点距离）/ support_ratio /
    confidence（迭代次数下可解释为内点比例）/ inlier_mask 由调用方按需重建。
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return []
    rng = np.random.default_rng(seed)
    best_by_round: list[dict[str, Any]] = []
    remaining = points
    for plane_index in range(max_planes):
        best: dict[str, Any] | None = None
        for _ in range(max_iterations):
            sample = remaining[rng.integers(0, len(remaining), size=3)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            offset = float(np.dot(normal, sample[0]))
            distances = np.abs(remaining @ normal - offset)
            inlier_ratio = float((distances <= tolerance).mean())
            if best is None or inlier_ratio > best["support_ratio"]:
                best = {
                    "normal": normal,
                    "offset": offset,
                    "support_ratio": inlier_ratio,
                }
        if best is None:
            break
        inliers = np.abs(remaining @ best["normal"] - best["offset"]) <= tolerance
        if best["support_ratio"] < 0.05:
            break
        best["inlier_count"] = int(inliers.sum())
        best["id"] = f"plane_{plane_index + 1:02d}"
        best["confidence"] = best["support_ratio"]
        best["normal"] = best["normal"].tolist()
        best_by_round.append(best)
        if not inliers.all():
            remaining = remaining[~inliers]
    return best_by_round


def dominant_plane(points: np.ndarray, **kwargs: Any) -> dict[str, Any] | None:
    planes = fit_planes(points, max_planes=1, **kwargs)
    return planes[0] if planes else None
