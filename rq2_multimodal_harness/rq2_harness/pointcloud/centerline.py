"""从规范化表面点云恢复可拒答的管状中心线证据。

第一版只确认两类高置信几何：单一直段，以及位于两个全局轴张成平面内的
单圆弧转角。判定只依赖连续截面、环心和拟合残差，不读取零件名。
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .sections import (
    _AXIS_NORMALS,
    _components,
    _fit_circle,
    _measure_annulus,
    adaptive_thickness,
    slice_points,
)

AXES = ("X", "Y", "Z")
AXIS_VECTORS = np.eye(3)
PROFILE_WORKPLANE = {"X": "YZ", "Y": "XZ", "Z": "XY"}


def _rounded(values: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(item), digits) for item in values]


def _circle_wire(radius: float) -> list[dict[str, Any]]:
    """闭合二维圆；只用 Plan v3 已支持的三点弧。"""
    r = round(float(radius), 4)
    return [
        {"kind": "move", "to": [r, 0.0]},
        {"kind": "three_point_arc", "through": [0.0, r], "to": [-r, 0.0]},
        {"kind": "three_point_arc", "through": [0.0, -r], "to": [r, 0.0]},
    ]


def _sample_annular_sections(
    points: np.ndarray,
    *,
    samples_per_axis: int = 25,
) -> list[dict[str, Any]]:
    thickness = adaptive_thickness(points)
    low = points.min(axis=0)
    high = points.max(axis=0)
    samples: list[dict[str, Any]] = []
    for axis_index, axis in enumerate(AXES):
        extent = float(high[axis_index] - low[axis_index])
        if extent <= 1e-9:
            continue
        margin = min(0.02 * extent, thickness / 4.0)
        offsets = np.linspace(
            float(low[axis_index] + margin),
            float(high[axis_index] - margin),
            samples_per_axis,
        )
        for offset in offsets:
            origin = np.zeros(3, dtype=np.float64)
            origin[axis_index] = offset
            _slice_3d, slice_2d, basis = slice_points(
                points, origin, _AXIS_NORMALS[axis], thickness
            )
            if len(slice_2d) < 16:
                continue
            components = _components(slice_2d, epsilon=thickness)
            if not components:
                continue
            component = max(components, key=len)
            measured = _measure_annulus(component)
            if measured is None:
                continue
            circle = _fit_circle(component)
            center_2d = (
                np.asarray(circle["center"], dtype=np.float64)
                if circle is not None
                else component.mean(axis=0)
            )
            right, up, _forward = basis
            center_3d = origin + center_2d[0] * right + center_2d[1] * up
            samples.append(
                {
                    "axis": axis,
                    "axis_index": axis_index,
                    "offset": float(offset),
                    "center": center_3d,
                    "inner_radius": float(measured["inner_radius"]),
                    "outer_radius": float(measured["outer_radius"]),
                    "point_count": int(len(component)),
                }
            )
    return samples


def _line_candidates(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for axis_index, axis in enumerate(AXES):
        group = [item for item in samples if item["axis"] == axis]
        if len(group) < 4:
            continue
        outer_median = float(np.median([item["outer_radius"] for item in group]))
        filtered = [
            item
            for item in group
            if abs(item["outer_radius"] - outer_median)
            <= max(0.025, 0.2 * outer_median)
        ]
        if len(filtered) < 4:
            continue
        offsets = np.asarray([item["offset"] for item in filtered])
        centers = np.asarray([item["center"] for item in filtered])
        other = [index for index in range(3) if index != axis_index]
        fixed = np.median(centers[:, other], axis=0)
        residual = np.linalg.norm(centers[:, other] - fixed, axis=1)
        stable = residual <= max(0.035, float(np.median(residual)) * 3.0)
        if int(stable.sum()) < 4:
            continue
        offsets = offsets[stable]
        centers = centers[stable]
        fixed = np.median(centers[:, other], axis=0)
        span = float(offsets.max() - offsets.min())
        if span < 0.2:
            continue
        anchor = np.zeros(3, dtype=np.float64)
        anchor[axis_index] = float(np.median(offsets))
        anchor[other] = fixed
        candidates.append(
            {
                "axis": axis,
                "axis_index": axis_index,
                "anchor": anchor,
                "offset_min": float(offsets.min()),
                "offset_max": float(offsets.max()),
                "offset_median": float(np.median(offsets)),
                "span": span,
                "inner_radius": float(
                    np.median([item["inner_radius"] for item in filtered])
                ),
                "outer_radius": float(
                    np.median([item["outer_radius"] for item in filtered])
                ),
                "support_slices": int(len(offsets)),
                "center_residual": float(np.median(residual[stable])),
            }
        )
    return candidates


def _line_path(candidate: dict[str, Any]) -> dict[str, Any]:
    axis_index = int(candidate["axis_index"])
    start = np.asarray(candidate["anchor"], dtype=np.float64).copy()
    end = start.copy()
    start[axis_index] = candidate["offset_min"]
    end[axis_index] = candidate["offset_max"]
    confidence = min(
        0.95,
        0.45
        + 0.04 * candidate["support_slices"]
        - 2.0 * candidate["center_residual"],
    )
    return {
        "id": "path_01",
        "kind": "line",
        "confidence": round(max(0.0, confidence), 4),
        "section_profile": {
            "kind": "annulus",
            "inner_radius": round(candidate["inner_radius"], 4),
            "outer_radius": round(candidate["outer_radius"], 4),
        },
        "segments": [
            {
                "kind": "line",
                "start": _rounded(start),
                "end": _rounded(end),
                "support_slices": candidate["support_slices"],
            }
        ],
        "path_wire": [
            {"kind": "move", "to": _rounded(start)},
            {"kind": "line", "to": _rounded(end)},
        ],
        "profile_workplane": PROFILE_WORKPLANE[candidate["axis"]],
        "profile_offset": _rounded(start),
    }


def _fit_single_bend(
    points: np.ndarray,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any] | None:
    """由两条正交、单侧延伸的等截面直腿拟合公共切圆弧。"""
    axis_a = int(left["axis_index"])
    axis_b = int(right["axis_index"])
    if axis_a == axis_b:
        return None
    common_axis = ({0, 1, 2} - {axis_a, axis_b}).pop()
    inner = float(np.median([left["inner_radius"], right["inner_radius"]]))
    outer = float(np.median([left["outer_radius"], right["outer_radius"]]))
    if outer <= inner or abs(left["outer_radius"] - right["outer_radius"]) > max(
        0.03, 0.18 * outer
    ):
        return None
    # 剩余轴应主要是截面直径，避免在桶、块、网格上误触发路径。
    extents = points.max(axis=0) - points.min(axis=0)
    if abs(float(extents[common_axis]) - 2.0 * outer) > 0.35 * max(
        float(extents[common_axis]), 2.0 * outer
    ):
        return None

    cross = np.zeros(3, dtype=np.float64)
    cross[axis_b] = float(left["anchor"][axis_b])
    cross[axis_a] = float(right["anchor"][axis_a])
    cross[common_axis] = float(
        np.median([left["anchor"][common_axis], right["anchor"][common_axis]])
    )
    sign_a = 1.0 if left["offset_median"] >= cross[axis_a] else -1.0
    sign_b = 1.0 if right["offset_median"] >= cross[axis_b] else -1.0

    def fit_from(initial_radius: float) -> tuple[float, np.ndarray, float, float] | None:
        radius = initial_radius
        subset = points
        for _iteration in range(2):
            center_a = cross[axis_a] + sign_a * radius
            center_b = cross[axis_b] + sign_b * radius
            far_a = (
                sign_a * (points[:, axis_a] - center_a) > 0.65 * outer
            ) & (np.abs(points[:, axis_b] - cross[axis_b]) < 1.4 * outer)
            far_b = (
                sign_b * (points[:, axis_b] - center_b) > 0.65 * outer
            ) & (np.abs(points[:, axis_a] - cross[axis_a]) < 1.4 * outer)
            subset = points[~(far_a | far_b)]
            if len(subset) < 128:
                return None

            def residual(value: np.ndarray) -> np.ndarray:
                r = float(value[0])
                ca = cross[axis_a] + sign_a * r
                cb = cross[axis_b] + sign_b * r
                radial = np.sqrt(
                    (subset[:, axis_a] - ca) ** 2
                    + (subset[:, axis_b] - cb) ** 2
                )
                distance = np.sqrt(
                    (radial - r) ** 2
                    + (subset[:, common_axis] - cross[common_axis]) ** 2
                )
                return np.minimum(
                    np.abs(distance - inner), np.abs(distance - outer)
                )

            upper = max(0.16, 0.8 * min(extents[axis_a], extents[axis_b]))
            result = least_squares(
                residual,
                [radius],
                bounds=([0.06], [upper]),
                loss="soft_l1",
                f_scale=0.02,
                max_nfev=300,
            )
            radius = float(result.x[0])
        errors = residual(np.asarray([radius]))
        return (
            radius,
            subset,
            float(np.median(errors)),
            float(np.percentile(errors, 90)),
        )

    starts = np.linspace(
        0.16 * min(extents[axis_a], extents[axis_b]),
        0.42 * min(extents[axis_a], extents[axis_b]),
        4,
    )
    fits = [item for item in (fit_from(float(start)) for start in starts) if item]
    if not fits:
        return None
    radius, subset, median_error, p90_error = min(
        fits, key=lambda item: (item[3], item[2])
    )
    if p90_error > max(0.025, 0.15 * outer) or len(subset) < 128:
        return None

    center = cross.copy()
    center[axis_a] += sign_a * radius
    center[axis_b] += sign_b * radius
    tangent_a = center.copy()
    tangent_a[axis_b] = cross[axis_b]
    tangent_b = center.copy()
    tangent_b[axis_a] = cross[axis_a]
    endpoint_a = tangent_a.copy()
    endpoint_b = tangent_b.copy()
    endpoint_a[axis_a] = (
        points[:, axis_a].max() if sign_a > 0 else points[:, axis_a].min()
    )
    endpoint_b[axis_b] = (
        points[:, axis_b].max() if sign_b > 0 else points[:, axis_b].min()
    )
    radial_a = tangent_a - center
    radial_b = tangent_b - center
    radial_mid = radial_a + radial_b
    norm_mid = float(np.linalg.norm(radial_mid))
    if norm_mid <= 1e-9:
        return None
    through = center + radius * radial_mid / norm_mid

    confidence = min(
        0.95,
        0.55
        + 0.02 * min(left["support_slices"], right["support_slices"])
        - 4.0 * p90_error,
    )
    path_wire = [
        {"kind": "move", "to": _rounded(endpoint_a)},
        {"kind": "line", "to": _rounded(tangent_a)},
        {
            "kind": "three_point_arc",
            "through": _rounded(through),
            "to": _rounded(tangent_b),
        },
        {"kind": "line", "to": _rounded(endpoint_b)},
    ]
    return {
        "id": "path_01",
        "kind": "line_arc_line",
        "confidence": round(max(0.0, confidence), 4),
        "plane_axes": [AXES[axis_a], AXES[axis_b]],
        "section_profile": {
            "kind": "annulus",
            "inner_radius": round(inner, 4),
            "outer_radius": round(outer, 4),
        },
        "segments": [
            {
                "kind": "line",
                "start": _rounded(endpoint_a),
                "end": _rounded(tangent_a),
                "support_slices": left["support_slices"],
            },
            {
                "kind": "arc",
                "start": _rounded(tangent_a),
                "through": _rounded(through),
                "end": _rounded(tangent_b),
                "center": _rounded(center),
                "radius": round(radius, 4),
                "normal": _rounded(AXIS_VECTORS[common_axis]),
                "angle_deg": 90.0,
                "fit_median_error": round(median_error, 4),
                "fit_p90_error": round(p90_error, 4),
                "support_points": int(len(subset)),
            },
            {
                "kind": "line",
                "start": _rounded(tangent_b),
                "end": _rounded(endpoint_b),
                "support_slices": right["support_slices"],
            },
        ],
        "junctions": [
            {
                "type": "bend",
                "position": _rounded(center),
                "angle_deg": 90.0,
                "confidence": round(max(0.0, confidence), 4),
            }
        ],
        "path_wire": path_wire,
        "profile_workplane": PROFILE_WORKPLANE[left["axis"]],
        "profile_offset": _rounded(endpoint_a),
    }


def infer_path_graph(points: np.ndarray) -> dict[str, Any]:
    """返回 path_graph；不能确认时显式拒答而不是猜路径。"""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) < 64:
        return {
            "status": "unresolved",
            "components": [],
            "uncertainties": ["insufficient_points"],
        }
    points = points[:, :3]
    samples = _sample_annular_sections(points)
    lines = _line_candidates(samples)
    if len(lines) == 1:
        component = _line_path(lines[0])
        return {
            "status": "resolved",
            "components": [component],
            "section_samples": len(samples),
            "uncertainties": [],
        }
    if len(lines) == 2:
        component = _fit_single_bend(points, lines[0], lines[1])
        if component is not None and component["confidence"] >= 0.55:
            return {
                "status": "resolved",
                "components": [component],
                "section_samples": len(samples),
                "uncertainties": [],
            }
    reason = "ambiguous_centerline"
    if not lines:
        reason = "no_stable_annular_axis"
    elif len(lines) > 2:
        reason = "multi_branch_not_resolved_v1"
    return {
        "status": "unresolved",
        "components": [],
        "section_samples": len(samples),
        "uncertainties": [reason],
    }


def bound_sweep_operations(
    component: dict[str, Any],
    *,
    operation_id: str,
    combine: str,
) -> list[dict[str, Any]]:
    """把高置信 path evidence 确定性展开为外扫掠 + 内切扫掠。"""
    profile = component.get("section_profile") or {}
    if component.get("confidence", 0.0) < 0.55 or profile.get("kind") != "annulus":
        raise ValueError("path evidence confidence is insufficient")
    outer = float(profile["outer_radius"])
    inner = float(profile["inner_radius"])
    common = {
        "op": "sweep_profile",
        "workplane": component["profile_workplane"],
        "path_wire": component["path_wire"],
        "sweep_mode": "frenet",
        "offset": component["profile_offset"],
    }
    return [
        {
            "id": f"{operation_id}_outer",
            "combine": combine,
            "wire": _circle_wire(outer),
            **common,
        },
        {
            "id": f"{operation_id}_inner",
            "combine": "cut",
            "wire": _circle_wire(inner),
            **common,
        },
    ]
