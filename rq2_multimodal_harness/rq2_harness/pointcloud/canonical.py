from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .io import PointCloudError


@dataclass(frozen=True)
class CanonicalTransform:
    """raw → canonical 变换（与既有管线约定一致：bbox 中心移到原点、最长边归一为 1）。

    forward:  p' = (p - center) / scale
    inverse:  p  = p' * scale + center
    """

    center: np.ndarray  # (3,)
    scale: float  # 最长 bbox 边（>0）

    @classmethod
    def from_points(cls, points: np.ndarray) -> "CanonicalTransform":
        points = np.asarray(points, dtype=np.float64)
        if not len(points):
            raise PointCloudError("空点云无法计算 canonical 变换")
        low = points.min(axis=0)
        high = points.max(axis=0)
        center = (low + high) / 2.0
        scale = float(np.max(high - low))
        if scale <= 1e-12:
            raise PointCloudError("点云 bbox 尺度为零（退化为单点）")
        return cls(center=np.asarray(center, dtype=np.float64), scale=scale)

    def forward(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=np.float64) - self.center) / self.scale

    def inverse(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) * self.scale + self.center

    def to_matrix(self) -> np.ndarray:
        """行向量约定：p' = [p,1] @ M；M 为 4×4。"""
        s = self.scale
        matrix = np.eye(4)
        matrix[:3, :3] = np.eye(3) / s
        matrix[3, :3] = -self.center / s
        return matrix

    def inverse_matrix(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = np.eye(3) * self.scale
        matrix[3, :3] = self.center
        return matrix

    def to_dict(self) -> dict[str, Any]:
        return {"name": "canonical", "center": self.center.tolist(), "scale": self.scale}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalTransform":
        return cls(
            center=np.asarray(data["center"], dtype=np.float64),
            scale=float(data["scale"]),
        )

    def fingerprint(self) -> dict[str, Any]:
        return self.to_dict()
