from __future__ import annotations

from typing import Any

import numpy as np

from ..common import sha256_json
from .canonical import CanonicalTransform
from .io import clean_points, hash_points, load_point_cloud
from .normals import estimate_normals, normal_summary
from .primitives import fit_planes
from .sections import adaptive_thickness, query_cross_section
from .summary import summarize
from .symmetry import detect_mirror_symmetry

EVIDENCE_SCHEMA = "point_evidence.v1"
DEFAULT_SECTION_AXES = ("XY", "XZ", "YZ")


def _median_nn(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(points).query(points, k=2)
    return float(np.median(distances[:, 1]))

# 截面平面定义（canonical 帧，过中心）
_SECTION_PLANES = {
    "XY": {"normal": [0.0, 0.0, 1.0]},
    "XZ": {"normal": [0.0, 1.0, 0.0]},
    "YZ": {"normal": [1.0, 0.0, 0.0]},
}


def build_evidence(
    npy_path,
    *,
    normals_k: int = 16,
    plane_tolerance: float = 0.012,
    plane_max_planes: int = 2,
    ransac_seed: int = 42,
    section_axes: tuple[str, ...] = DEFAULT_SECTION_AXES,
    section_thickness: float | None = None,
    symmetry_tolerance: float = 0.015,
    symmetry_sample: int = 512,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """从 .npy 点云构建 PointEvidence v1（仅依赖输入点云，不访问 GT）。

    输出字段：schema / cloud_id / frame / quality / symmetry_candidates /
    primitive_candidates / sections / hypotheses / uncertainties /
    config / content_hash。

    section_thickness 为 None 时按点云密度自适应（4 × 中位最近邻距离，
    下限 0.09 上限 0.15），并在 sections 子块记录实际生效值。
    """
    raw = load_point_cloud(npy_path)
    points, quality = clean_points(raw)
    if quality["degenerate"]:
        return _degenerate_evidence(npy_path, quality, config)

    transform = CanonicalTransform.from_points(points)
    canonical = transform.forward(points)

    summary = summarize(points, transform)
    try:
        normals = estimate_normals(canonical, k=normals_k)
        normal_info = normal_summary(normals)
    except Exception as exc:
        normal_info = {"available": False, "error": str(exc)[:200]}

    planes = fit_planes(
        canonical,
        tolerance=plane_tolerance,
        max_planes=plane_max_planes,
        seed=ransac_seed,
    )
    symmetry = detect_mirror_symmetry(
        canonical,
        tolerance=symmetry_tolerance,
        sample=symmetry_sample,
        seed=ransac_seed,
    )

    sections = {}
    hypotheses: list[dict[str, Any]] = []
    uncertainties: list[dict[str, Any]] = []
    effective_thickness = (
        section_thickness if section_thickness is not None else adaptive_thickness(canonical)
    )
    for axis in section_axes:
        plane = _SECTION_PLANES[axis]
        section = query_cross_section(
            canonical,
            origin=np.zeros(3),
            normal=np.asarray(plane["normal"]),
            thickness=effective_thickness,
        )
        sections[axis] = {
            "normal": plane["normal"],
            "thickness": effective_thickness,
            "outer": section.get("outer"),
            "holes": section.get("holes") or [],
        }
        for hole in sections[axis]["holes"]:
            hypotheses.append(
                {
                    "target": f"{axis}/{hole['id']}",
                    "type": "circular_hole",
                    "radius": hole["radius"],
                    "center": hole["center"],
                    "confidence": hole["confidence"],
                    "evidence": f"section_{axis}",
                }
            )
            uncertainties.append(
                {
                    "target": f"{axis}/{hole['id']}",
                    "issue": "through_or_blind_unknown",
                    "recommended_tool": "query_cross_section",
                }
            )
        if sections[axis]["outer"] is None and len(canonical) > 4:
            uncertainties.append(
                {
                    "target": f"section_{axis}",
                    "issue": "outer_loop_unreliable",
                    "recommended_tool": "query_cross_section",
                }
            )

    plane_candidates = []
    for plane in planes:
        plane_candidates.append(
            {
                "id": plane["id"],
                "type": "plane",
                "normal": plane["normal"],
                "offset": plane["offset"],
                "support_ratio": plane["support_ratio"],
                "confidence": plane["confidence"],
            }
        )

    frame = {
        "name": "canonical",
        "center": summary["bbox"]["center"],
        "bbox_size": summary["canonical_bbox"]["size"],
        "principal_axes": summary["principal_axes"][:3],
        "eigenvalue_ratios": summary["eigenvalue_ratios"],
    }
    quality_block = {
        "point_count": summary["point_count"],
        "valid_ratio": quality["valid_ratio"],
        "normal_available": bool(normal_info.get("available", False)),
        "degenerate": False,
        "median_nn_distance": _median_nn(canonical),
    }

    effective_config = {
        "normals_k": normals_k,
        "plane_tolerance": plane_tolerance,
        "plane_max_planes": plane_max_planes,
        "ransac_seed": ransac_seed,
        "section_axes": list(section_axes),
        "section_thickness": section_thickness,
        "effective_section_thickness": effective_thickness,
        "symmetry_tolerance": symmetry_tolerance,
        "symmetry_sample": symmetry_sample,
    }
    if config is not None:
        effective_config = {**config, **effective_config}

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "cloud_id": "c_" + hash_points(points)[:16],
        "source_sha256": None,  # 由 prepare 阶段填充文件哈希（运行时不可见路径）
        "frame": frame,
        "quality": quality_block,
        "symmetry_candidates": symmetry,
        "primitive_candidates": plane_candidates,
        "sections": sections,
        "hypotheses": hypotheses,
        "uncertainties": uncertainties,
        "config": effective_config,
    }
    evidence["content_hash"] = sha256_json(evidence)
    return evidence


def _degenerate_evidence(npy_path, quality: dict[str, Any], config: dict[str, Any] | None) -> dict[str, Any]:
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "cloud_id": None,
        "source_sha256": None,
        "frame": None,
        "quality": {**quality, "normal_available": False},
        "symmetry_candidates": [],
        "primitive_candidates": [],
        "sections": {},
        "hypotheses": [],
        "uncertainties": [{"target": "cloud", "issue": "degenerate_input", "recommended_tool": None}],
        "config": config or {},
    }
    evidence["content_hash"] = sha256_json(evidence)
    return evidence
