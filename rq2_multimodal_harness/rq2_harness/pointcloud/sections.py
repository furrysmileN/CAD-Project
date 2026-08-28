from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

DEFAULT_THICKNESS = 0.01  # canonical 帧（旧默认；v1 证据改用自适应厚度）

_COMPONENT_FACTOR = 2.5


def adaptive_thickness(
    points: np.ndarray,
    *,
    min_thickness: float = 0.09,
    factor: float = 4.0,
    max_thickness: float = 0.15,
) -> float:
    """按点云密度自适应截面厚度（canonical 帧）。

    固定厚度在稀疏点云（2048 点）上往往只截到少量点，外环 bbox 不可靠；
    厚度取 4 × 中位最近邻距离并设下限 0.09（≈ 零件最长边 9%），保证带内
    点数与覆盖面稳定。语义为「平面附近的局部几何带」，外环 bbox 描述该带
    内零件的局部范围（离线审计按同一厚度做 band-to-band 对比）。
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return min_thickness
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(points).query(points, k=2)
    median_nn = float(np.median(distances[:, 1]))
    return float(min(max(min_thickness, factor * median_nn), max_thickness))


def _unit_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, up_hint))) > 0.95:
        up_hint = np.array([0.0, 1.0, 0.0])
    right = np.cross(up_hint, normal)
    right = right / np.linalg.norm(right)
    up = np.cross(normal, right)
    return right, up, normal


def slice_points(
    points: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """平面截面：返回 (slice_3d, slice_2d, basis)。"""
    points = np.asarray(points, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    right, up, forward = _unit_basis(normal)
    distances = np.abs((points - origin) @ forward)
    mask = distances <= thickness / 2.0
    slice_3d = points[mask]
    slice_2d = np.column_stack((slice_3d @ right, slice_3d @ up))
    return slice_3d, slice_2d, (right, up, forward)


def _components(points_2d: np.ndarray, epsilon: float) -> list[np.ndarray]:
    """按 2D 距离连通性聚类（阈值 epsilon 取 2.5 × 中位最近邻距离）。"""
    if len(points_2d) == 0:
        return []
    from scipy.spatial import cKDTree

    tree = cKDTree(points_2d)
    if len(points_2d) > 1:
        distances, _ = tree.query(points_2d, k=2)
        epsilon = max(epsilon, float(np.median(distances[:, 1])) * _COMPONENT_FACTOR)
    else:
        epsilon = max(epsilon, 1e-6)
    parents = list(range(len(points_2d)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    pairs = tree.query_pairs(r=epsilon)
    for left, right in pairs:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[root_left] = root_right
    groups: dict[int, list[int]] = {}
    for index in range(len(points_2d)):
        groups.setdefault(find(index), []).append(index)
    return [points_2d[indices] for indices in groups.values()]


def _fit_circle(points_2d: np.ndarray) -> dict[str, Any] | None:
    """Kasa 代数圆拟合 + 内点比例。"""
    if len(points_2d) < 5:
        return None
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    a_matrix = np.column_stack((2 * x, 2 * y, np.ones(len(x))))
    b_vector = x**2 + y**2
    try:
        solution, *_ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center_x, center_y, constant = solution
    radius_sq = center_x**2 + center_y**2 + constant
    if radius_sq <= 1e-12:
        return None
    radius = float(np.sqrt(max(radius_sq, 0.0)))
    center = np.array([center_x, center_y])
    distances = np.linalg.norm(points_2d - center, axis=1)
    tolerance = max(0.01, radius * 0.08)
    inlier_ratio = float((np.abs(distances - radius) <= tolerance).mean())
    return {"center": center.tolist(), "radius": radius, "inlier_ratio": inlier_ratio}


def _is_ring(points_2d: np.ndarray, circle: dict[str, Any] | None) -> bool:
    """环状（孔/轴）判定：圆内部（r < 0.6R）几乎没有点。"""
    if circle is None or len(points_2d) < 8:
        return False
    center = np.asarray(circle["center"])
    radius = circle["radius"]
    if radius <= 1e-9:
        return False
    inner = np.linalg.norm(points_2d - center, axis=1) < 0.6 * radius
    return float(inner.mean()) < 0.2


def query_cross_section(
    points: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
    thickness: float | None = None,
) -> dict[str, Any]:
    """截面查询：外环 bbox、圆孔候选、未分类区域统计。

    输入 canonical 帧；返回点均以截面 2D 局部坐标表示（right, up 基）。
    thickness 为 None 时按点云密度自适应（adaptive_thickness）。
    """
    if thickness is None:
        thickness = adaptive_thickness(points)
    slice_3d, slice_2d, (right, up, forward) = slice_points(
        points, origin, normal, thickness
    )
    result: dict[str, Any] = {
        "point_count": int(len(slice_2d)),
        "slice_ratio": float(len(slice_2d) / len(points)) if len(points) else 0.0,
        "normal": forward.tolist(),
        "thickness": thickness,
    }
    if len(slice_2d) < 4:
        result["outer"] = None
        result["loops"] = []
        result["holes"] = []
        return result

    components = _components(slice_2d, epsilon=thickness)
    loops = []
    holes = []
    outer = None
    outer_index: int | None = None
    for index, component in enumerate(components):
        if len(component) < 4:
            continue
        low = component.min(axis=0)
        high = component.max(axis=0)
        size = high - low
        area = float((high[0] - low[0]) * (high[1] - low[1]))
        if outer is None or area > outer["area"]:
            outer = {
                "type": "outer",
                "bbox_size": size.tolist(),
                "area": area,
                "closed": True,
                "point_count": int(len(component)),
            }
            outer_index = index
    for index, component in enumerate(components):
        if len(component) < 4 or index == outer_index:
            continue
        low = component.min(axis=0)
        high = component.max(axis=0)
        size = high - low
        circle = _fit_circle(component)
        ring = _is_ring(component, circle)
        loops.append(
            {
                "id": f"loop_{index + 1:02d}",
                "type": "circular_hole" if ring and circle is not None else "region",
                "bbox_size": size.tolist(),
                "point_count": int(len(component)),
            }
        )
        if ring and circle is not None and circle["inlier_ratio"] >= 0.6:
            holes.append(
                {
                    "id": f"hole_{index + 1:02d}",
                    "type": "circular_hole",
                    "center": circle["center"],
                    "radius": circle["radius"],
                    "point_count": int(len(component)),
                    "inlier_ratio": circle["inlier_ratio"],
                    "confidence": float(min(circle["inlier_ratio"], 1.0)),
                }
            )
    if outer is not None:
        outer.pop("area", None)
    result["outer"] = outer
    result["loops"] = loops
    result["holes"] = holes
    return result
