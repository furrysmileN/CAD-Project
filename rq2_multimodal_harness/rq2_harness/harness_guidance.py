"""本地构造决策：只从当前任务点云证据推导，不读零件族 / stem / GT。"""
from __future__ import annotations

from typing import Any

THIN_RATIO = 0.08
AXIS_NAMES = ("X", "Y", "Z")
WORKPLANE_FOR_THIN = {"X": "YZ", "Y": "XZ", "Z": "XY"}
TWO_LONG_SIMILAR = 1.25
THIRD_CLEARLY_SHORT = 1.8
ROD_LONG_RATIO = 3.0
ROD_END_SIMILAR = 2.0
SECTION_SCALE_LOFT = 1.5

GENERATOR_EXTRUDE = "extrude_cut"
GENERATOR_REVOLVE_LONG = "revolve_along_long"
GENERATOR_SWEEP_PLANE = "sweep_or_revolve_in_long_plane"
GENERATOR_BLOCK = "block_or_loft"

CONSTRUCTION_LAWS = """[CONSTRUCTION_LAWS]
Answer only these four questions from measurements. Do not infer part names or classes.

1. Workplane: thin_axis = the smallest bbox edge; put the primary face on the other two axes.
   Never default a thin part onto XY just because plates are often drawn that way.
2. Generator from the three edge lengths (not from names):
   - one edge much thinner than the others → polygon_extrude or box, then cut
   - one edge much longer and the other two similar → cylinder or revolve_profile along the long axis
   - two long similar edges and a clearly shorter third → sweep_profile or a 90-degree revolve
     in the plane spanned by the two long axes
   - three similar edges → box, or loft_profiles if two sections have different outer scales
3. Profile topology: a reliable inner loop or an outer ring on an offset slice → hollow;
   otherwise solid. Do not fake a hollow by revolving a large solid rectangle.
4. Sizes: use only numbers listed under [DECISIONS]. If an inner radius is unmeasured,
   do not invent wall thickness; sweep or extrude the outer envelope.

A helix is sweep_profile whose path is helix. A taper is loft between two different outer scales.
"""


def _as_bbox(bbox_size: Any) -> list[float] | None:
    if not isinstance(bbox_size, (list, tuple)) or len(bbox_size) != 3:
        return None
    try:
        values = [float(item) for item in bbox_size]
    except (TypeError, ValueError):
        return None
    if any(item < 0 or item != item for item in values):  # NaN check
        return None
    return values


def _bbox_from_evidence(evidence: dict[str, Any] | None, bbox_size: Any = None) -> list[float] | None:
    if bbox_size is not None:
        parsed = _as_bbox(bbox_size)
        if parsed is not None:
            return parsed
    if not isinstance(evidence, dict):
        return None
    frame = evidence.get("frame") or {}
    return _as_bbox(frame.get("bbox_size"))


def infer_pose(bbox_size: Any) -> dict[str, Any] | None:
    """从规范包围盒推出薄轴、长轴、工作面与板/杆/块。"""
    sizes = _as_bbox(bbox_size)
    if sizes is None:
        return None
    ranked = sorted(zip(sizes, AXIS_NAMES), key=lambda item: item[0])
    min_s, thin_axis = ranked[0]
    mid_s, mid_axis = ranked[1]
    max_s, long_axis = ranked[2]
    thin_ratio = (min_s / max_s) if max_s > 0 else 0.0
    if thin_ratio < THIN_RATIO:
        shape_class = "plate"
    elif mid_s > 0 and max_s / mid_s >= ROD_LONG_RATIO and min_s > 0 and mid_s / min_s < ROD_END_SIMILAR:
        shape_class = "rod"
    else:
        shape_class = "block"
    return {
        "bbox_size": [round(item, 4) for item in sizes],
        "thin_axis": thin_axis,
        "long_axis": long_axis,
        "mid_axis": mid_axis,
        "workplane": WORKPLANE_FOR_THIN[thin_axis],
        "shape_class": shape_class,
        "thin_ratio": round(thin_ratio, 4),
        "max_over_mid": round(max_s / mid_s, 4) if mid_s > 0 else None,
        "mid_over_min": round(mid_s / min_s, 4) if min_s > 0 else None,
    }


def infer_generator(pose: dict[str, Any] | None) -> dict[str, Any]:
    """由三边长短关系选生成元，不按零件名。"""
    if not pose:
        return {
            "id": None,
            "verbs": [],
            "reason": "bbox unmeasured; do not guess a workplane or generator",
        }
    if pose["shape_class"] == "plate":
        return {
            "id": GENERATOR_EXTRUDE,
            "verbs": ["polygon_extrude", "box", "slot", "hole"],
            "reason": (
                f"one thin edge ({pose['thin_axis']}); extrude on {pose['workplane']} then cut"
            ),
        }
    max_over_mid = pose.get("max_over_mid")
    mid_over_min = pose.get("mid_over_min")
    if (
        isinstance(max_over_mid, (int, float))
        and isinstance(mid_over_min, (int, float))
        and max_over_mid <= TWO_LONG_SIMILAR
        and mid_over_min >= THIRD_CLEARLY_SHORT
    ):
        return {
            "id": GENERATOR_SWEEP_PLANE,
            "verbs": ["sweep_profile", "revolve_profile"],
            "reason": (
                f"two long similar edges, shorter {pose['thin_axis']}; "
                f"sweep or 90-degree revolve on {pose['workplane']}"
            ),
        }
    if pose["shape_class"] == "rod":
        return {
            "id": GENERATOR_REVOLVE_LONG,
            "verbs": ["cylinder", "revolve_profile"],
            "reason": f"one long edge ({pose['long_axis']}); cylinder or revolve along that axis",
        }
    return {
        "id": GENERATOR_BLOCK,
        "verbs": ["box", "loft_profiles"],
        "reason": "three comparable edges; box, or loft if section outer scales differ",
    }


def _hole_radii(evidence: dict[str, Any] | None) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if not isinstance(evidence, dict):
        return found
    for axis, block in (evidence.get("sections") or {}).items():
        if not isinstance(block, dict):
            continue
        for hole in block.get("holes") or []:
            if not isinstance(hole, dict):
                continue
            radius = hole.get("radius")
            if isinstance(radius, (int, float)) and radius > 0:
                found.append(
                    {
                        "section": axis,
                        "radius": round(float(radius), 4),
                        "confidence": hole.get("confidence"),
                    }
                )
    return found


def _section_outer_scales(evidence: dict[str, Any] | None) -> dict[str, float]:
    scales: dict[str, float] = {}
    if not isinstance(evidence, dict):
        return scales
    for axis, block in (evidence.get("sections") or {}).items():
        if not isinstance(block, dict):
            continue
        outer = block.get("outer") or {}
        bbox_2d = outer.get("bbox_size")
        if not isinstance(bbox_2d, (list, tuple)) or len(bbox_2d) != 2:
            continue
        try:
            width, height = abs(float(bbox_2d[0])), abs(float(bbox_2d[1]))
        except (TypeError, ValueError):
            continue
        scales[str(axis)] = round(max(width, height), 4)
    return scales


def _canonical_points(raw: Any):
    import numpy as np

    from .pointcloud.canonical import CanonicalTransform
    from .pointcloud.io import clean_points

    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3 or len(array) < 16:
        return None
    cleaned, quality = clean_points(array[:, :3])
    if quality.get("degenerate"):
        return None
    return CanonicalTransform.from_points(cleaned).forward(cleaned)


def _offset_rings(evidence: dict[str, Any] | None, points: Any = None) -> list[dict[str, Any]]:
    if isinstance(evidence, dict):
        stored = evidence.get("offset_rings")
        if isinstance(stored, list) and stored:
            return [item for item in stored if isinstance(item, dict)]
    if points is None:
        return []
    from .pointcloud.sections import probe_outer_rings

    canonical = _canonical_points(points)
    if canonical is None:
        return []
    bbox = _bbox_from_evidence(evidence)
    return probe_outer_rings(canonical, bbox)


def _can_measure_topology(evidence: dict[str, Any] | None, points: Any) -> bool:
    if points is not None:
        return True
    if not isinstance(evidence, dict):
        return False
    if evidence.get("offset_rings"):
        return True
    sections = evidence.get("sections")
    return isinstance(sections, dict) and bool(sections)


def _path_graph(evidence: dict[str, Any] | None, points: Any) -> dict[str, Any]:
    if isinstance(evidence, dict):
        stored = evidence.get("path_graph")
        if isinstance(stored, dict):
            return stored
    if points is None:
        return {
            "status": "unmeasured",
            "components": [],
            "uncertainties": ["no_point_cloud_in_condition"],
        }
    canonical = _canonical_points(points)
    if canonical is None:
        return {
            "status": "unresolved",
            "components": [],
            "uncertainties": ["invalid_point_cloud"],
        }
    from .pointcloud.centerline import infer_path_graph

    return infer_path_graph(canonical)


def infer_topology(
    evidence: dict[str, Any] | None,
    *,
    points: Any = None,
) -> dict[str, Any]:
    """中面内环或偏移面外轮廓圆环 → 空心；探过但没有环 → 实心；没探过 → 未测。"""
    holes = _hole_radii(evidence)
    if holes:
        return {
            "kind": "hollow",
            "inner_radius_known": True,
            "holes": holes,
            "rings": [],
            "inner_radius": holes[0]["radius"],
            "outer_radius": None,
            "note": "use a measured inner loop; do not invent a second radius",
        }
    rings = _offset_rings(evidence, points)
    if rings:
        inner = [float(item["inner_radius"]) for item in rings if item.get("inner_radius")]
        outer = [float(item["outer_radius"]) for item in rings if item.get("outer_radius")]
        inner.sort()
        outer.sort()
        mid = len(inner) // 2
        return {
            "kind": "hollow",
            "inner_radius_known": True,
            "holes": [],
            "rings": rings,
            "note": "offset slice shows an outer ring; use these radii, do not invent wall thickness",
            "inner_radius": round(inner[mid], 4) if inner else None,
            "outer_radius": round(outer[mid], 4) if outer else None,
        }
    if _can_measure_topology(evidence, points):
        return {
            "kind": "solid",
            "inner_radius_known": False,
            "holes": [],
            "rings": [],
            "note": "no inner loop or offset ring; do not invent wall thickness",
        }
    return {
        "kind": "unmeasured",
        "inner_radius_known": False,
        "holes": [],
        "rings": [],
        "note": "no point-cloud section in this condition; do not guess hollow or wall thickness",
    }


def infer_sizes(
    pose: dict[str, Any] | None,
    evidence: dict[str, Any] | None,
    topology: dict[str, Any],
) -> dict[str, Any]:
    scales = _section_outer_scales(evidence)
    loft_hint = False
    comparable = dict(scales)
    if pose and pose.get("shape_class") == "plate":
        comparable = {}
    elif (
        pose
        and isinstance(pose.get("max_over_mid"), (int, float))
        and isinstance(pose.get("mid_over_min"), (int, float))
        and pose["max_over_mid"] <= TWO_LONG_SIMILAR
        and pose["mid_over_min"] >= THIRD_CLEARLY_SHORT
    ):
        # 两长轴平面上的截面是路径包络，不能拿来和端截面比尺度
        comparable = {axis: value for axis, value in scales.items() if axis != pose.get("workplane")}
    if len(comparable) >= 2:
        values = list(comparable.values())
        lo, hi = min(values), max(values)
        loft_hint = lo > 0 and hi / lo >= SECTION_SCALE_LOFT
    hole_radii = [item["radius"] for item in topology.get("holes") or []]
    if not hole_radii and topology.get("inner_radius") is not None:
        hole_radii = [topology["inner_radius"]]
    allowed = {
        "bbox_size": None if pose is None else pose["bbox_size"],
        "section_outer_scales": scales,
        "hole_radii": hole_radii,
        "inner_radius": topology.get("inner_radius"),
        "outer_radius": topology.get("outer_radius"),
        "loft_if_section_scales_differ": loft_hint,
    }
    return allowed


def infer_decisions(
    evidence: dict[str, Any] | None = None,
    *,
    bbox_size: Any = None,
    points: Any = None,
) -> dict[str, Any]:
    pose = infer_pose(_bbox_from_evidence(evidence, bbox_size))
    generator = infer_generator(pose)
    topology = infer_topology(evidence, points=points)
    sizes = infer_sizes(pose, evidence, topology)
    path_graph = _path_graph(evidence, points)
    return {
        "pose": pose,
        "generator": generator,
        "topology": topology,
        "sizes": sizes,
        "path_graph": path_graph,
    }


def pose_block(pose: dict[str, Any] | None) -> str:
    if not pose:
        return ""
    return "\n".join(
        [
            "[POSE]",
            "Local measurements from the current-task point cloud (canonical frame, longest edge = 1).",
            f"- bbox_size: {pose['bbox_size']}",
            f"- thin_axis: {pose['thin_axis']} (smallest edge)",
            f"- long_axis: {pose['long_axis']} (largest edge)",
            f"- recommended_workplane: {pose['workplane']} (the two non-thin axes)",
            f"- extent_class: {pose['shape_class']} (plate if min/max < {THIN_RATIO:.2f}; else rod or block)",
            "Place the primary face on recommended_workplane.",
        ]
    )


def decisions_block(decisions: dict[str, Any]) -> str:
    pose = decisions.get("pose")
    generator = decisions.get("generator") or {}
    topology = decisions.get("topology") or {}
    sizes = decisions.get("sizes") or {}
    path_graph = decisions.get("path_graph") or {}
    lines = [
        "[DECISIONS]",
        "This-task answers. Use these instead of guessing a part class.",
        f"- workplane: {None if pose is None else pose['workplane']} (thin_axis={None if pose is None else pose['thin_axis']})",
        f"- generator: {generator.get('id')} → {generator.get('verbs')}",
        f"- generator_reason: {generator.get('reason')}",
        f"- topology: {topology.get('kind')}; inner_radius_known={topology.get('inner_radius_known')}",
        f"- topology_note: {topology.get('note')}",
        f"- allowed_sizes: bbox={sizes.get('bbox_size')} section_outer={sizes.get('section_outer_scales')} "
        f"inner_radius={sizes.get('inner_radius')} outer_radius={sizes.get('outer_radius')} "
        f"hole_radii={sizes.get('hole_radii')} loft_if_scales_differ={sizes.get('loft_if_section_scales_differ')}",
        f"- path_graph_status: {path_graph.get('status')}; "
        f"evidence_refs={[item.get('id') for item in path_graph.get('components') or []]}; "
        f"uncertainties={path_graph.get('uncertainties')}",
    ]
    for component in path_graph.get("components") or []:
        lines.append(
            f"- path_evidence {component.get('id')}: kind={component.get('kind')} "
            f"confidence={component.get('confidence')} segments={component.get('segments')} "
            f"section_profile={component.get('section_profile')}"
        )
    if path_graph.get("status") == "resolved":
        lines.append(
            "- For a target whose visible structure follows a resolved measured path, use "
            "sweep_path_ref with its evidence_ref. Numeric path/profile fields are bound "
            "deterministically after your response; do not copy or edit them."
        )
    return "\n".join(lines)


def laws_block() -> str:
    return CONSTRUCTION_LAWS.rstrip()


def build_guidance(
    evidence: dict[str, Any] | None = None,
    *,
    bbox_size: Any = None,
    points: Any = None,
) -> dict[str, Any]:
    """同源引导：prompt 文本与对照页共用。"""
    decisions = infer_decisions(evidence, bbox_size=bbox_size, points=points)
    pose = decisions["pose"]
    pose_text = pose_block(pose)
    laws_text = laws_block()
    # 没有本任务包围盒就不要写 [DECISIONS]：未测 ≠ 实心，更不能把点云半径写进仅照片条件
    decision_text = decisions_block(decisions) if pose is not None else ""
    parts = [item for item in (pose_text, laws_text, decision_text) if item]
    compare = None
    if pose is not None:
        compare = {
            "thin_axis": pose["thin_axis"],
            "long_axis": pose["long_axis"],
            "workplane": pose["workplane"],
            "extent_class": pose["shape_class"],
            "thin_ratio": pose["thin_ratio"],
            "bbox_size": pose["bbox_size"],
            "generator": decisions["generator"].get("id"),
            "generator_verbs": decisions["generator"].get("verbs"),
            "topology": decisions["topology"].get("kind"),
            "inner_radius_known": decisions["topology"].get("inner_radius_known"),
            "inner_radius": decisions["sizes"].get("inner_radius"),
            "outer_radius": decisions["sizes"].get("outer_radius"),
            "loft_if_section_scales_differ": decisions["sizes"].get("loft_if_section_scales_differ"),
            "path_graph_status": decisions["path_graph"].get("status"),
            "path_components": decisions["path_graph"].get("components"),
        }
    return {
        "pose": pose,
        "decisions": decisions,
        "pose_block": pose_text,
        "laws_block": laws_text,
        "decisions_block": decision_text,
        "prompt_block": "\n\n".join(parts),
        "compare": compare,
    }
