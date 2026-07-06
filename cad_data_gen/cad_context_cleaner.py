#!/usr/bin/env python3
"""Clean and compact CAD technical context for LLM shape description."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Optional


SURFACE_PRIORITY = [
    "PLANE",
    "CYLINDRICAL_SURFACE",
    "CONICAL_SURFACE",
    "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "B_SPLINE_SURFACE_WITH_KNOTS",
    "B_SPLINE_SURFACE",
]

ENTITY_PRIORITY = [
    "MANIFOLD_SOLID_BREP",
    "SHELL_BASED_SURFACE_MODEL",
    "CLOSED_SHELL",
    "ADVANCED_FACE",
    "PLANE",
    "CYLINDRICAL_SURFACE",
    "CONICAL_SURFACE",
    "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "B_SPLINE_SURFACE_WITH_KNOTS",
    "CIRCLE",
    "ELLIPSE",
    "LINE",
    "EDGE_CURVE",
    "ORIENTED_EDGE",
    "VERTEX_POINT",
]

CORE_FEATURE_TYPES = {
    "newSketch",
    "extrude",
    "revolve",
    "fillet",
    "chamfer",
    "shell",
    "booleanBodies",
    "boolean",
    "loft",
    "sweep",
    "hole",
    "mirror",
    "linearPattern",
    "circularPattern",
    "draft",
}

HIGH_VALUE_FEATURE_TYPES = {
    "revolve": "旋转体/轴对称结构",
    "loft": "放样过渡或复杂截面变化",
    "sweep": "扫掠管道或沿路径成形结构",
    "shell": "薄壁/抽壳结构",
    "booleanBodies": "布尔合并或切除",
    "boolean": "布尔操作",
    "hole": "孔结构",
    "linearPattern": "线性阵列",
    "circularPattern": "圆周阵列",
    "mirror": "镜像对称",
    "fillet": "圆角过渡",
    "chamfer": "倒角边界",
    "draft": "拔模斜面",
}


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _round_number(value: Any, digits: int = 6) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        rounded = round(value, digits)
        return 0.0 if abs(rounded) < 10 ** (-digits) else rounded
    return value


def _compact_value(value: Any, *, depth: int = 2, max_list: int = 8, max_dict_items: int = 12) -> Any:
    """Recursively shrink arbitrary JSON-like values without changing semantics."""
    if depth <= 0:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return _round_number(value)
        if isinstance(value, list):
            return f"list[{len(value)}]"
        if isinstance(value, dict):
            return f"dict[{len(value)}]"
        return str(value)
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_dict_items:
                compact["__truncated__"] = len(value) - max_dict_items
                break
            if _is_empty(item):
                continue
            compact[str(key)] = _compact_value(
                item,
                depth=depth - 1,
                max_list=max_list,
                max_dict_items=max_dict_items,
            )
        return compact
    if isinstance(value, list):
        compact_list = [
            _compact_value(item, depth=depth - 1, max_list=max_list, max_dict_items=max_dict_items)
            for item in value[:max_list]
            if not _is_empty(item)
        ]
        if len(value) > max_list:
            compact_list.append({"__truncated__": len(value) - max_list})
        return compact_list
    return _round_number(value)


def _filter_counts(counts: Any, priority: list[str], *, top_n: int = 16) -> dict[str, int]:
    if not isinstance(counts, dict):
        return {}
    result: dict[str, int] = {}
    for key in priority:
        value = counts.get(key)
        if value:
            result[key] = int(value)
    if len(result) >= top_n:
        return result
    remaining = sorted(

e and str(key) not in result),
        key=lambda item: item[1],
        reverse=True,
    )
    for key, value in remaining[: max(0, top_n - len(result))]:
        result[key] = value
    return result


def _sample_list(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []return [_compact_value(item, depth=2, max_list=6, max_dict_items=8) f
or item in value[:limit]]

def compact_mesh_metrics(mesh_metrics: Optional[dict[str, Any]], mesh_err
or: Optional[str], *, mode: str) -> Optional[dict[str, Any]]:
    if mode == "minimal":
        return None
    if not isinstance(mesh_metrics, dict):
        return {"error": mesh_error} if mesh_error else None
    keys = [
        "vertices",
        "triangles",
        "bbox_extent",
        "bbox_diagonal",
        "surface_area",
        "volume",
        "is_watertight",
        "euler_number",
        "connected_components",
        "brep_face_count_from_mapping",
    ]compact = {key: _compact_value(mesh_metrics.get(key), depth=1) for ke
y in keys if not _is_empty(mesh_metrics.get(key))}
    if mesh_error:
        compact["error"] = mesh_error
    return compact or None


def compact_step_summary(
    step_stats: dict[str, Any],
    mesh_metrics: Optional[dict[str, Any]] = None,
    mesh_error: Optional[str] = None,
    *,
    mode: str = "balanced",
    max_circle_samples: int = 4,
    max_ellipse_samples: int = 3,
    max_surface_parameter_samples: int = 4,
) -> dict[str, Any]:
    """Keep only final-geometry signals useful for shape description."""features = step_stats.get("model_features", {}) if isinstance(step_st
ats, dict) else {}
    if not isinstance(features, dict):
        features = {}

    surface_counts = features.get("surface_counts") or {}
    face_surface_counts = features.get("face_surface_counts") or {}
    important_counts = step_stats.get("important_entity_counts") or {}

    summary: dict[str, Any] = {
        "schema": step_stats.get("schema"),
        "units": _sample_list(step_stats.get("units"), 4),"component_names": _sample_list(features.get("product_names") or
features.get("solid_names") or [], 8),"solid_names": _sample_list(features.get("solid_names") or [],
8),"surface_types_present": [key for key in SURFACE_PRIORITY if surf
ace_counts.get(key)],"face_surface_types_present": [key for key in SURFACE_PRIORITY i
f face_surface_counts.get(key)],"surface_counts_core": _filter_counts(surface_counts, SURFACE_PRI
ORITY, top_n=10),"face_surface_counts_core": _filter_counts(face_surface_counts, S
URFACE_PRIORITY, top_n=10),
        "circle_radius_count": features.get("circle_radius_count"),"circle_radii_samples": _sample_list(features.get("circle_radii_s
amples"), max_circle_samples),"ellipse_parameter_samples": _sample_list(features.get("ellipse_p
arameter_samples"), max_ellipse_samples),"modeling_hints_from_step": _sample_list(features.get("modeling_h
ints_from_step"), 8),"important_entity_counts_core": _filter_counts(important_counts,
ENTITY_PRIORITY, top_n=14),
    }
    if mode == "full_compact":
        summary["surface_parameter_samples"] = _sample_list(
            features.get("surface_parameter_samples"),
            max_surface_parameter_samples,
        )summary["top_entity_counts"] = _sample_list(step_stats.get("top_e
ntity_counts"), 12)mesh_compact = compact_mesh_metrics(mesh_metrics, mesh_error, mode=mo
de)
    if mesh_compact:
        summary["mesh_metrics_core"] = mesh_compactreturn {key: value for key, value in summary.items() if not _is_empty
(value)}


def _feature_operation_type(feature: dict[str, Any]) -> Optional[Any]:
    scalars = feature.get("scalar_parameters")
    if isinstance(scalars, dict) and scalars.get("operationType"):
        return scalars.get("operationType")
    parameters = feature.get("parameters")
    if isinstance(parameters, list):
        for param in parameters:if isinstance(param, dict) and param.get("parameter_id") ==
"operationType":
                return param.get("value")
    return None

def _compact_feature(feature: dict[str, Any], *, max_scalar_params: int
= 10) -> dict[str, Any]:
    scalar_parameters = feature.get("scalar_parameters")
    if isinstance(scalar_parameters, dict):
        scalar_parameters = {str(key): _compact_value(value, depth=1, max_list=5, max_dict
_items=6)for key, value in list(scalar_parameters.items())[:max_scalar
_params]
            if not _is_empty(value)
        }
    else:scalar_parameters = _compact_value(scalar_parameters, depth=1, ma
x_list=max_scalar_params, max_dict_items=6)

    compact = {
        "index": feature.get("index"),
        "feature_type": feature.get("feature_type"),
        "operation_label": feature.get("operation_label"),
        "name": feature.get("name"),
        "operation_type": _feature_operation_type(feature),
        "suppressed": feature.get("suppressed"),
        "status": feature.get("status"),
        "scalar_parameters": scalar_parameters,"inputs_summary": _compact_value(feature.get("inputs"), depth=1,
max_list=6, max_dict_items=8),"resolved_inputs_summary": _compact_value(feature.get("resolved_i
nputs"), depth=1, max_list=6, max_dict_items=8),
        "sketch_entity_summary": _compact_value(feature.get("sketch_entity_summary"), depth=2, max_list=8, max_dict_items=8),
        "constraint_summary": _compact_value(feature.get("constraint_summ
ary"), depth=1, max_list=6, max_dict_items=8),}
    return {key: value for key, value in compact.items() if not _is_empty
(value)}

def _ordered_core_features(ofs_summary: dict[str, Any]) -> list[dict[st
r, Any]]:
    raw_sequence = ofs_summary.get("feature_sequence") or []
    if not isinstance(raw_sequence, list):
        return []
    return [
        featurefor feature in raw_sequence
        if isinstance(feature, dict) and str(feature.get("feature_typ
e")) in CORE_FEATURE_TYPES
    ]


def compact_ofs_summary(
    ofs_summary: Optional[dict[str, Any]],
    ofs_error: Optional[str] = None,
    *,
    max_features: int = 24,
    max_scalar_params: int = 10,
) -> Optional[dict[str, Any]]:
    """Keep modeling operations and remove raw OFS payload details."""
    if not isinstance(ofs_summary, dict):
        return {"error": ofs_error} if ofs_error else None

    core_features = _ordered_core_features(ofs_summary)
    compact_features = [
        _compact_feature(feature, max_scalar_params=max_scalar_params)
        for feature in core_features[:max_features]]
    type_sequence = [str(feature.get("feature_type")) for feature in core
_features if feature.get("feature_type")]
    high_value = []
    seen = set()for feature_type in type_sequence:
        if feature_type in HIGH_VALUE_FEATURE_TYPES and feature_type not in seen:
            high_value.append({"feature_type": feature_type, "signal": HI
GH_VALUE_FEATURE_TYPES[feature_type]})
            seen.add(feature_type)
    compact: dict[str, Any] = {
        "ofs_path": Path(str(ofs_summary.get("ofs_path"))).name if ofs_su
mmary.get("ofs_path") else None,"feature_count_total": ofs_summary.get("feature_count_total"),
        "feature_type_counts": _filter_counts(ofs_summary.get("feature_type_counts") or {}, list(CORE_FEATURE_TYPES), top_n=18),
        "operation_type_counts": _compact_value(ofs_summary.get("operatio
n_type_counts"), depth=1, max_list=8, max_dict_items=12),
        "ordered_feature_types": type_sequence[:max_features],
        "high_value_modeling_signals": high_value[:10],"feature_sequence_compact": compact_features,
        "truncated_feature_sequence": max(0, len(core_features) - len(com
pact_features)),
        "error": ofs_error,}
    return {key: value for key, value in compact.items() if not _is_empty
(value)}

def compact_aux_summary(value: Optional[dict[str, Any]], error: Optional
[str], *, max_items: int = 12) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):return {"error": error} if error else None
    compact = _compact_value(value, depth=2, max_list=8, max_dict_items=m
ax_items)
    if isinstance(compact, dict) and error:
        compact["error"] = error
    return compact if compact else None


def build_compact_technical_context(
    *,
    sample_id: str,
    relative_step_path: str,
    render_image_count: int,
    point_path: Optional[Path],
    step_stats: dict[str, Any],
    mesh_metrics: Optional[dict[str, Any]],
    mesh_error: Optional[str],
    ofs_summary: Optional[dict[str, Any]],
    ofs_error: Optional[str],
    feat_summary: Optional[dict[str, Any]] = None,
    feat_error: Optional[str] = None,
    meta_summary: Optional[dict[str, Any]] = None,
    meta_error: Optional[str] = None,
    mode: str = "balanced",
    max_ofs_features: int = 24,
) -> dict[str, Any]:
    if mode not in {"minimal", "balanced", "full_compact"}:
        raise ValueError(f"Unsupported compact context mode: {mode}")

    context: dict[str, Any] = {
        "sample_id": sample_id,
        "step_path": relative_step_path,"point_cloud_available": point_path is not None,
        "point_cloud_file": Path(point_path).name if point_path is not No
ne else None,
        "render_image_count": render_image_count,
        "context_mode": mode,
        "step_geometry_core": compact_step_summary(
            step_stats,
            mesh_metrics,
            mesh_error,
            mode=mode,
        ),
        "ofs_modeling_core": compact_ofs_summary(
            ofs_summary,
            ofs_error,
            max_features=max_ofs_features,
        ),
    }if mode == "full_compact":
        context["feat_summary_core"] = compact_aux_summary(feat_summary, feat_error)
        context["metadata_summary_core"] = compact_aux_summary(meta_summa
ry, meta_error)
    elif feat_error:
        context["feat_error"] = feat_error
    elif meta_error:
        context["metadata_error"] = meta_error
    return {key: value for key, value in context.items() if not _is_empty
(value)}

def count_feature_types(compact_context: dict[str, Any]) -> dict[str, in
t]:
    ofs_core = compact_context.get("ofs_modeling_core")
    if not isinstance(ofs_core, dict):
        return {}
    counts = ofs_core.get("feature_type_counts")if isinstance(counts, dict):
        return {str(key): int(value) for key, value in counts.items() if
value}sequence = ofs_core.get("ordered_feature_types") or []
    return dict(Counter(str(item) for item in sequence))
