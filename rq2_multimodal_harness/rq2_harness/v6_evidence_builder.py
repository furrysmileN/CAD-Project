"""Build P_comp from a point cloud only. GT / latent_spec are forbidden in this path."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label

from .common import sha256_json
from .pointcloud.canonical import CanonicalTransform
from .pointcloud.evidence import build_evidence
from .pointcloud.sections import _unit_basis, query_cross_section

EVIDENCE_SCHEMA = "point_evidence.v6"
GENERATOR_VERSION = "rq2.v6.evidence.v2"
HOLE_SLICE_THICKNESSES = (0.03, 0.035, 0.045, 0.06, 0.075, 0.08)
HOLE_RADIUS_RANGE = (0.025, 0.16)


def _longest(size: list[float]) -> float:
    return max(float(x) for x in size) if size else 1.0


def _merge_holes(holes: list[dict[str, Any]], center_tol: float = 0.04) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for hole in holes:
        center = np.asarray(hole["center"][:2], dtype=float)
        radius = float(hole["radius"])
        matched = None
        for item in merged:
            other = np.asarray(item["center"][:2], dtype=float)
            if float(np.linalg.norm(center - other)) <= center_tol:
                matched = item
                break
        if matched is None:
            merged.append(
                {
                    "center": [round(float(center[0]), 4), round(float(center[1]), 4)],
                    "radius": round(radius, 4),
                    "point_count": int(hole.get("point_count") or 0),
                    "confidence": float(hole.get("confidence") or hole.get("inlier_ratio") or 0.0),
                }
            )
            continue
        count = int(matched.get("point_count") or 0) + int(hole.get("point_count") or 0)
        matched["radius"] = round((float(matched["radius"]) + radius) / 2.0, 4)
        matched["center"] = [
            round((float(matched["center"][0]) + float(center[0])) / 2.0, 4),
            round((float(matched["center"][1]) + float(center[1])) / 2.0, 4),
        ]
        matched["point_count"] = count
        matched["confidence"] = max(float(matched.get("confidence") or 0.0), float(hole.get("confidence") or 0.0))
    return merged


def _xy_circular_holes(canonical: np.ndarray) -> list[dict[str, Any]]:
    origin = np.zeros(3)
    normal = np.array([0.0, 0.0, 1.0])
    collected: list[dict[str, Any]] = []
    for thickness in HOLE_SLICE_THICKNESSES:
        section = query_cross_section(canonical, origin, normal, thickness=thickness)
        for hole in section.get("holes") or []:
            radius = float(hole.get("radius") or 0.0)
            if HOLE_RADIUS_RANGE[0] <= radius <= HOLE_RADIUS_RANGE[1]:
                collected.append(hole)
    return _merge_holes(collected)


def _hole_depth_and_type(canonical: np.ndarray, center_xy: list[float], radius: float) -> dict[str, Any]:
    xy = canonical[:, :2]
    dist = np.linalg.norm(xy - np.asarray(center_xy[:2], dtype=float), axis=1)
    interior = canonical[dist < radius * 0.7]
    z_min = float(canonical[:, 2].min())
    z_max = float(canonical[:, 2].max())
    thickness = z_max - z_min
    if len(interior) < 4:
        return {"through": True, "depth": round(thickness, 4), "kind": "through"}
    high = interior[interior[:, 2] > z_min + 0.02 * max(thickness, 0.05)]
    if len(high) < 2:
        return {"through": True, "depth": round(thickness, 4), "kind": "through"}
    floor = float(np.median(high[:, 2]))
    depth = z_max - floor
    if depth < 0.03 or depth >= 0.90 * thickness:
        return {"through": True, "depth": round(thickness, 4), "kind": "through"}
    return {"through": False, "depth": round(depth, 4), "kind": "blind"}


def _measure_holes(canonical: np.ndarray) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, hole in enumerate(_xy_circular_holes(canonical)):
        measured = _hole_depth_and_type(canonical, hole["center"], float(hole["radius"]))
        facts.append(
            {
                "fact_id": f"section_xy/hole_{index:02d}",
                "category": "through_vs_blind" if measured["through"] else "depth",
                "value": "through" if measured["through"] else measured["depth"],
                "through": measured["through"],
                "depth": measured["depth"],
                "radius": hole["radius"],
                "center": hole["center"],
                "confidence": hole.get("confidence"),
                "source": "point_cloud_section",
                "role": "measured",
            }
        )
    return facts


def _bin_max(points: np.ndarray, u: int, v: int, w: int, n_bins: int = 36) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    uu = points[:, u]
    vv = points[:, v]
    ww = points[:, w]
    u0, u1 = float(uu.min()), float(uu.max())
    v0, v1 = float(vv.min()), float(vv.max())
    ui = np.clip(((uu - u0) / max(u1 - u0, 1e-9) * n_bins).astype(int), 0, n_bins - 1)
    vi = np.clip(((vv - v0) / max(v1 - v0, 1e-9) * n_bins).astype(int), 0, n_bins - 1)
    max_w = np.full((n_bins, n_bins), np.nan)
    count = np.zeros((n_bins, n_bins), dtype=int)
    for i, j, value in zip(ui, vi, ww):
        count[i, j] += 1
        max_w[i, j] = value if np.isnan(max_w[i, j]) else max(max_w[i, j], value)
    return max_w, count, (u0, u1), (v0, v1)


def _face_depressions(
    points: np.ndarray,
    u: int,
    v: int,
    w: int,
    *,
    min_drop: float,
    max_drop: float,
    min_bins: int = 12,
) -> list[dict[str, Any]]:
    max_w, count, ur, vr = _bin_max(points, u, v, w)
    occupied = count >= 1
    if int(occupied.sum()) < 16:
        return []
    face = float(np.nanpercentile(max_w[occupied], 90))
    w_min = float(points[:, w].min())
    low = occupied & (max_w <= face - min_drop) & (max_w >= w_min + 0.03)
    labeled, n_comp = label(low)
    n_bins = max_w.shape[0]
    found: list[dict[str, Any]] = []
    for index in range(1, n_comp + 1):
        ys, xs = np.where(labeled == index)
        if len(ys) < min_bins:
            continue
        floor = float(np.median(max_w[ys, xs]))
        depth = face - floor
        if depth < min_drop or depth > max_drop:
            continue
        cu = ur[0] + (ur[1] - ur[0]) * (float(ys.mean()) + 0.5) / n_bins
        cv = vr[0] + (vr[1] - vr[0]) * (float(xs.mean()) + 0.5) / n_bins
        found.append(
            {
                "depth": round(depth, 4),
                "center": [round(cu, 4), round(cv, 4)],
                "n_bins": int(len(ys)),
                "face": round(face, 4),
            }
        )
    found.sort(key=lambda item: item["n_bins"], reverse=True)
    return found


def _pocket_depth(canonical: np.ndarray) -> dict[str, Any] | None:
    hits = _face_depressions(canonical, 0, 1, 2, min_drop=0.06, max_drop=0.30, min_bins=8)
    if not hits:
        return None
    best = max(hits, key=lambda item: (float(item["depth"]), int(item["n_bins"])))
    return {
        "fact_id": "top_face.pocket_depth",
        "category": "depth",
        "value": best["depth"],
        "center": best["center"],
        "source": "point_cloud_top_depression",
        "role": "measured",
        "through": False,
    }


def _hidden_presence(canonical: np.ndarray) -> dict[str, Any]:
    y = canonical[:, 1]
    y_max = float(y.max())
    skin = canonical[y >= y_max - 0.04]
    if len(skin) < 40:
        return _hidden_fact(False)
    size = skin.max(axis=0) - skin.min(axis=0)
    if float(size[0]) < 0.35 or float(size[2]) < 0.28:
        return _hidden_fact(False)
    x0, x1 = float(skin[:, 0].min()), float(skin[:, 0].max())
    z0, z1 = float(skin[:, 2].min()), float(skin[:, 2].max())
    normal = np.array([0.0, 1.0, 0.0])
    right, up, _ = _unit_basis(normal)
    found = []
    for dy in (0.03, 0.06, 0.09, 0.12, 0.15):
        origin = np.array([0.0, y_max - dy, 0.0])
        section = query_cross_section(canonical, origin, normal, thickness=0.05)
        for hole in section.get("holes") or []:
            radius = float(hole.get("radius") or 0.0)
            if not (0.045 <= radius <= 0.11):
                continue
            c2 = np.asarray(hole.get("center") or [0.0, 0.0], dtype=float)
            world = origin + c2[0] * right + c2[1] * up
            wx, wz = float(world[0]), float(world[2])
            if not (x0 + 0.12 < wx < x1 - 0.12 and z0 + 0.10 < wz < z1 - 0.10):
                continue
            found.append({"radius": round(radius, 4), "center": [round(wx, 4), round(wz, 4)], "dy": dy})
    return _hidden_fact(bool(found), extra={"candidates": found[:4]})


def _hidden_fact(present: bool, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    fact = {
        "fact_id": "back_face.hidden_presence",
        "category": "hidden_presence",
        "value": present,
        "role": "measured",
        "source": "point_cloud_back_skin",
    }
    if extra:
        fact.update(extra)
    return fact


def _bolt_spacing(hole_facts: list[dict[str, Any]]) -> dict[str, Any] | None:
    bolts = []
    for item in hole_facts:
        radius = item.get("radius")
        center = item.get("center")
        if radius is None or not center:
            continue
        if not (0.03 <= float(radius) <= 0.08):
            continue
        radial = float(np.hypot(float(center[0]), float(center[1])))
        if radial < 0.08:
            continue
        bolts.append([float(center[0]), float(center[1]), radial, float(radius)])
    if len(bolts) < 2:
        return None
    arr = np.asarray(bolts, dtype=float)
    median_r = float(np.median(arr[:, 2]))
    order = np.argsort(np.abs(arr[:, 2] - median_r))
    chosen = arr[order[: min(4, len(arr))], :2]
    spacing = float(max(float(np.ptp(chosen[:, 0])), float(np.ptp(chosen[:, 1]))))
    if spacing < 0.12 or spacing > 0.45:
        return None
    return {
        "fact_id": "hole_centers.spacing",
        "category": "offset_or_spacing",
        "value": round(spacing, 4),
        "role": "measured",
        "source": "point_cloud_section",
        "n_bolts": int(len(chosen)),
    }


def _principal_axis_and_radius(canonical: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    centered = canonical - canonical.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = np.asarray(vh[0], dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    if axis[2] < 0:
        axis = -axis
    radial = np.linalg.norm(centered - np.outer(centered @ axis, axis), axis=1)
    radius = float(np.percentile(radial, 97))
    axis_fact = {
        "fact_id": "frame.principal_axis",
        "category": "axis_or_symmetry",
        "value": [round(float(x), 4) for x in axis],
        "role": "measured",
        "source": "point_cloud_pca",
    }
    radius_fact = {
        "fact_id": "frame.outer_radius",
        "category": "radius_or_width",
        "value": round(radius, 4),
        "role": "measured",
        "source": "point_cloud_radial",
    }
    return axis_fact, radius_fact


def _bbox_fact(size: list[float]) -> dict[str, Any]:
    return {
        "fact_id": "frame.bbox_longest",
        "category": "bbox",
        "value": round(_longest(size), 4),
        "role": "measured",
        "source": "point_cloud_bbox",
    }


def build_p_comp(npy_path: str | Path, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    base = build_evidence(npy_path, config=config)
    points = np.asarray(np.load(str(npy_path)), dtype=np.float64)
    transform = CanonicalTransform.from_points(points)
    canonical = transform.forward(points)
    cad_facts = _measure_holes(canonical)
    pocket = _pocket_depth(canonical)
    if pocket is not None:
        cad_facts.append(pocket)
    cad_facts.append(_hidden_presence(canonical))
    spacing = _bolt_spacing(cad_facts)
    if spacing:
        cad_facts.append(spacing)
    axis_fact, radius_fact = _principal_axis_and_radius(canonical)
    cad_facts.extend([axis_fact, radius_fact])
    bbox = (base.get("frame") or {}).get("bbox_size") or [1.0, 1.0, 1.0]
    cad_facts.append(_bbox_fact(list(bbox)))
    evidence = {
        **base,
        "schema": EVIDENCE_SCHEMA,
        "cad_facts": cad_facts,
        "generator": "v6_evidence_builder.build_p_comp",
        "generator_version": GENERATOR_VERSION,
        "inputs": {"pointcloud_only": True, "reads_gt": False},
    }
    evidence["content_hash"] = sha256_json({key: value for key, value in evidence.items() if key != "content_hash"})
    return evidence


def _pick_primary(facts: list[dict[str, Any]], critical: dict[str, Any]) -> dict[str, Any] | None:
    category = critical.get("category")
    if category == "depth":
        blinds = [
            item
            for item in facts
            if item.get("category") == "depth"
            and item.get("source") != "point_cloud_bbox"
            and item.get("through") is not True
            and isinstance(item.get("value"), (int, float))
            and 0.05 <= float(item["value"]) <= 0.35
        ]
        if not blinds:
            return None
        section = [item for item in blinds if item.get("source") == "point_cloud_section"]
        pool = section or blinds
        return max(pool, key=lambda item: float(item.get("value") or 0.0))
    if category == "through_vs_blind":
        holes = [item for item in facts if "through" in item or item.get("category") == "through_vs_blind"]
        if any(item.get("through") is False or item.get("value") == "blind" for item in holes):
            value = "blind"
        elif any(
            item.get("category") == "depth"
            and isinstance(item.get("value"), (int, float))
            and 0.05 <= float(item["value"]) <= 0.16
            for item in facts
        ):
            value = "blind"
        elif any(item.get("through") is True or item.get("value") == "through" for item in holes):
            value = "through"
        else:
            return None
        return {
            "fact_id": str(critical.get("fact_id")),
            "category": "through_vs_blind",
            "value": value,
            "source": "point_cloud_inferred",
            "role": "measured",
        }
    if category == "offset_or_spacing":
        return next((item for item in facts if item.get("category") == "offset_or_spacing"), None)
    if category == "radius_or_width":
        return next((item for item in facts if item.get("source") == "point_cloud_radial"), None)
    if category == "hidden_presence":
        return next((item for item in facts if item.get("category") == "hidden_presence"), None)
    if category == "axis_or_symmetry":
        return next((item for item in facts if item.get("category") == "axis_or_symmetry"), None)
    return None


def attach_primary_critical(p_comp: dict[str, Any], critical: dict[str, Any]) -> dict[str, Any]:
    """Label a measured primary fact. Never copies GT numeric values. Never uses bbox as a stand-in."""
    facts = list(p_comp.get("cad_facts") or [])
    chosen = _pick_primary(facts, critical)
    if chosen is None:
        chosen = {
            "fact_id": critical.get("fact_id"),
            "category": critical.get("category"),
            "value": None,
            "source": "point_cloud_unresolved",
            "role": "primary_critical",
        }
        facts.append(chosen)
        p_comp.setdefault("uncertainties", []).append(
            {
                "target": critical.get("fact_id"),
                "issue": "critical_category_unresolved",
                "recommended_tool": None,
            }
        )
    else:
        chosen = dict(chosen)
        chosen["role"] = "primary_critical"
        chosen["fact_id"] = critical.get("fact_id")
        facts.append(chosen)
    p_comp["cad_facts"] = facts
    p_comp["uncertainties"] = [
        item
        for item in (p_comp.get("uncertainties") or [])
        if item.get("issue") not in {"critical_category_not_measured", "critical_category_weak_measurement"}
    ]
    p_comp["content_hash"] = sha256_json({key: value for key, value in p_comp.items() if key != "content_hash"})
    return p_comp
