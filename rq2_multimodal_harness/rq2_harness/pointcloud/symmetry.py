from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .canonical import CanonicalTransform

DEFAULT_TOLERANCE = 0.015
DEFAULT_SAMPLE = 512
DEFAULT_SEED = 42


def detect_mirror_symmetry(
    points: np.ndarray,
    transform: CanonicalTransform | None = None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    sample: int = DEFAULT_SAMPLE,
    seed: int = DEFAULT_SEED,
    max_candidates: int = 3,
) -> list[dict[str, Any]]:
    """镜像面对称候选（points 在 canonical 帧）。

    候选法向：三个主轴 + 主平面法向（若存在）；对每个法向在若干偏移上计算
    反射一致性（反射后落在原云 tolerance 内的采样点比例），取最优偏移。

    返回按 support_ratio 降序的候选列表。
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 8:
        return []
    tree = cKDTree(points)
    rng = np.random.default_rng(seed)
    sampled = points[rng.choice(len(points), size=min(sample, len(points)), replace=False)]

    axes = np.eye(3)
    cov = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    candidate_normals = [eigenvectors[:, index] for index in order]
    try:
        from .primitives import dominant_plane

        plane = dominant_plane(points, seed=seed)
        if plane is not None and not any(
            abs(float(np.dot(np.asarray(plane["normal"]), normal))) > 0.99 for normal in candidate_normals
        ):
            candidate_normals.append(np.asarray(plane["normal"]))
    except Exception:
        pass

    candidates: list[dict[str, Any]] = []
    for normal in candidate_normals:
        normal = normal / np.linalg.norm(normal)
        offsets = [0.0, 0.05, -0.05, 0.1, -0.1]
        best: dict[str, Any] | None = None
        for offset in offsets:
            reflected = sampled - 2.0 * (sampled @ normal - offset)[:, None] * normal
            distances, _ = tree.query(reflected, k=1)
            ratio = float((distances <= tolerance).mean())
            if best is None or ratio > best["support_ratio"]:
                best = {"normal": normal.copy(), "offset": offset, "support_ratio": ratio}
        candidates.append(best)
    candidates.sort(key=lambda item: item["support_ratio"], reverse=True)
    for index, candidate in enumerate(candidates[:max_candidates]):
        candidate["id"] = f"sym_{index + 1:02d}"
        candidate["type"] = "mirror"
        candidate["normal"] = candidate["normal"].tolist()
        candidate["confidence"] = candidate["support_ratio"]
    return candidates[:max_candidates]
