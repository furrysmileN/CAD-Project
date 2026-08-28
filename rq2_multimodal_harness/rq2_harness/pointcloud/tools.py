from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .canonical import CanonicalTransform
from .compare import compare_cad_to_cloud, localize_geometric_error, sample_step
from .primitives import fit_planes
from .sections import adaptive_thickness, query_cross_section
from .summary import summarize
from .symmetry import detect_mirror_symmetry

TOOL_PROTOCOL_VERSION = "pointcloud.tools.v1"

# 工具注册表：名称 → (描述, 参数 schema)
# 参数 schema 为 dict：{name: {type, required, description}}
TOOL_SPECS: dict[str, dict[str, Any]] = {
    "get_pointcloud_summary": {
        "description": "返回该点云的全局摘要（bbox、主轴、平面候选、对称候选、法向分布）。",
        "params": {
            "cloud_id": {"type": "string", "required": True, "description": "当前任务绑定的点云 id（与 Prompt 中相同）"},
        },
    },
    "query_cross_section": {
        "description": "用给定平面（origin + normal，canonical 帧，厚度 thickness）截取点云，返回外环 bbox、圆孔候选与区域统计。",
        "params": {
            "cloud_id": {"type": "string", "required": True, "description": "当前任务绑定的点云 id"},
            "origin": {"type": "vec3", "required": True, "description": "平面经过的点（canonical 帧）"},
            "normal": {"type": "vec3", "required": True, "description": "平面法向（canonical 帧）"},
            "thickness": {"type": "number", "required": False, "description": "截面厚度（canonical 帧，默认按点云密度自适应，下限 0.09 上限 0.15）"},
        },
    },
    "detect_symmetry": {
        "description": "重新检测镜像面对称候选（法向 + 偏移 + 支持比例）。",
        "params": {
            "cloud_id": {"type": "string", "required": True, "description": "当前任务绑定的点云 id"},
        },
    },
    "fit_primitives": {
        "description": "在给定区域（region_bbox，可选）拟合平面基元；v1 仅支持 plane。",
        "params": {
            "cloud_id": {"type": "string", "required": True, "description": "当前任务绑定的点云 id"},
            "primitive_types": {"type": "list", "required": True, "description": "如 [\"plane\"]；\"cylinder\" v1 不支持"},
            "region_bbox": {"type": "vec6", "required": False, "description": "[xmin,ymin,zmin,xmax,ymax,zmax]（canonical 帧）"},
        },
    },
    "measure_pointcloud": {
        "description": "测量点云尺寸/中心/主轴长度。",
        "params": {
            "cloud_id": {"type": "string", "required": True, "description": "当前任务绑定的点云 id"},
            "measurement_type": {
                "type": "string",
                "required": True,
                "description": "bbox_size | center | axis_extents",
            },
        },
    },
    "compare_cad_to_cloud": {
        "description": "把当前候选 STEP 与输入点云比较（coverage / precision / fscore / missing-extra 比例）。",
        "params": {
            "cloud_id": {"type": "string", "required": True, "description": "当前任务绑定的点云 id"},
            "candidate_step_id": {"type": "string", "required": True, "description": "当前任务候选 STEP 的 id"},
        },
    },
    "localize_geometric_error": {
        "description": "定位候选 STEP 相对输入点云的 missing/extra 区域（top-k 聚类与局部 bbox）。",
        "params": {
            "cloud_id": {"type": "string", "required": True, "description": "当前任务绑定的点云 id"},
            "candidate_step_id": {"type": "string", "required": True, "description": "当前任务候选 STEP 的 id"},
            "top_k": {"type": "integer", "required": False, "description": "返回聚类数（默认 3）"},
        },
    },
}

SUPPORTED_MEASUREMENTS = {"bbox_size", "center", "axis_extents"}


@dataclass
class PointCloudSession:
    """一次任务内的点云工具执行上下文。cloud_id 隔离：模型不能访问未绑定点云。"""

    cloud_id: str
    points_canonical: np.ndarray
    transform: CanonicalTransform
    candidate_steps: dict[str, Path] = field(default_factory=dict)
    _step_cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    n_points: int = 2048
    seed: int = 42

    def _check_cloud(self, request: dict[str, Any]) -> None:
        if request.get("cloud_id") != self.cloud_id:
            raise PermissionError(
                f"cloud_id 不匹配：请求 {request.get('cloud_id')!r}，当前任务绑定 {self.cloud_id!r}"
            )

    def _step_points(self, step_id: str) -> np.ndarray:
        if step_id not in self.candidate_steps:
            raise ValueError(f"未知 candidate_step_id {step_id!r}")
        if step_id not in self._step_cache:
            self._step_cache[step_id] = sample_step(
                self.candidate_steps[step_id], n_points=self.n_points, seed=self.seed
            )
        return self._step_cache[step_id]


def _validate_params(spec: dict[str, Any], params: Any) -> tuple[dict[str, Any], str | None]:
    if not isinstance(params, dict):
        return params, "params 必须是 JSON object"
    errors = []
    for name, schema in spec["params"].items():
        if schema["required"] and name not in params:
            errors.append(f"缺少必需参数 {name}")
    for name, value in params.items():
        schema = spec["params"].get(name)
        if schema is None:
            continue
        kind = schema["type"]
        if kind == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{name} 必须是数字")
        elif kind == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"{name} 必须是整数")
        elif kind == "vec3":
            if not isinstance(value, list) or len(value) != 3 or not all(
                isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
            ):
                errors.append(f"{name} 必须是 [x,y,z]")
        elif kind == "vec6":
            if not isinstance(value, list) or len(value) != 6 or not all(
                isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
            ):
                errors.append(f"{name} 必须是 [xmin,ymin,zmin,xmax,ymax,zmax]")
        elif kind == "list":
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                errors.append(f"{name} 必须是字符串列表")
        elif kind == "string":
            if not isinstance(value, str):
                errors.append(f"{name} 必须是字符串")
    if errors:
        return params, "; ".join(errors[:8])
    return params, None


def execute_tool(session: PointCloudSession, request: dict[str, Any]) -> dict[str, Any]:
    """执行一次工具调用。返回 {"ok", "result"|"error"}。任何异常都结构化返回。"""
    try:
        if not isinstance(request, dict):
            raise ValueError("query_request 必须是 JSON object")
        name = request.get("tool")
        if not isinstance(name, str) or name not in TOOL_SPECS:
            raise ValueError(f"未知工具 {name!r}，可用: {', '.join(sorted(TOOL_SPECS))}")
        params, error = _validate_params(TOOL_SPECS[name], request.get("params"))
        if error is not None:
            raise ValueError(f"参数错误: {error}")
        session._check_cloud(params)
        result = _dispatch(session, name, params)
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": {"code": type(exc).__name__, "message": str(exc)[:500]}}


def _dispatch(session: PointCloudSession, name: str, params: dict[str, Any]) -> dict[str, Any]:
    points = session.points_canonical
    if name == "get_pointcloud_summary":
        summary = summarize(session.transform.inverse(points), session.transform)
        return {
            "point_count": int(len(points)),
            "canonical_bbox_size": summary["canonical_bbox"]["size"],
            "principal_axes": summary["principal_axes"],
            "axis_extents": summary["axis_extents"],
            "aspect_ratios": summary["aspect_ratios"],
            "median_nn_distance": summary["median_nn_distance"],
        }
    if name == "query_cross_section":
        origin = np.asarray(params["origin"], dtype=np.float64)
        normal = np.asarray(params["normal"], dtype=np.float64)
        thickness = params.get("thickness")
        thickness_value = adaptive_thickness(points) if thickness is None else float(thickness)
        if not np.isfinite(origin).all() or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 1e-12:
            raise ValueError("origin/normal 必须有限且 normal 非零")
        if thickness_value <= 0:
            raise ValueError("thickness 必须 > 0")
        result = query_cross_section(points, origin, normal, thickness_value)
        return {
            "point_count": result["point_count"],
            "slice_ratio": result["slice_ratio"],
            "outer": result.get("outer"),
            "holes": result.get("holes") or [],
            "loops": result.get("loops") or [],
        }
    if name == "detect_symmetry":
        return {"symmetry_candidates": detect_mirror_symmetry(points, seed=session.seed)}
    if name == "fit_primitives":
        types = params.get("primitive_types") or []
        if "cylinder" in types or "sphere" in types:
            raise ValueError(f"primitive_types 中 {[t for t in types if t not in ('plane',)]} v1 不支持")
        return {"planes": fit_planes(points, seed=session.seed)}
    if name == "measure_pointcloud":
        measurement = params.get("measurement_type")
        if measurement not in SUPPORTED_MEASUREMENTS:
            raise ValueError(f"measurement_type 必须是 {sorted(SUPPORTED_MEASUREMENTS)}")
        summary = summarize(session.transform.inverse(points), session.transform)
        if measurement == "bbox_size":
            return {"bbox_size": summary["canonical_bbox"]["size"]}
        if measurement == "center":
            return {"center": summary["bbox"]["center"]}
        return {"axis_extents": summary["axis_extents"]}
    if name == "compare_cad_to_cloud":
        step_points = session._step_points(params["candidate_step_id"])
        return compare_cad_to_cloud(
            step_points,
            session.transform.inverse(points),
            session.transform,
            n_points=session.n_points,
            seed=session.seed,
        )
    if name == "localize_geometric_error":
        step_points = session._step_points(params["candidate_step_id"])
        return localize_geometric_error(
            step_points,
            session.transform.inverse(points),
            session.transform,
            top_k=int(params.get("top_k", 3)),
            n_points=session.n_points,
            seed=session.seed,
        )
    raise ValueError(f"工具 {name!r} 未实现")


def parse_query_request(text: str) -> dict[str, Any]:
    """解析模型输出：是 query_request JSON（含 tool+params）还是其它。"""
    from ..prompting import _extract_json_candidate

    candidate = _extract_json_candidate(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {"kind": "unknown", "request": None, "raw": text}
    if isinstance(parsed, dict) and isinstance(parsed.get("query_request"), dict):
        inner = parsed["query_request"]
        if isinstance(inner.get("tool"), str):
            return {"kind": "query_request", "request": inner, "raw": text}
    if isinstance(parsed, dict) and isinstance(parsed.get("tool"), str):
        return {"kind": "query_request", "request": parsed, "raw": text}
    return {"kind": "plan", "request": parsed, "raw": text}


def tool_manifest() -> str:
    """给模型的工具说明文本（紧凑）。"""
    lines = ["可用点云工具（canonical 帧；每次调用返回 JSON）："]
    for name, spec in sorted(TOOL_SPECS.items()):
        params = ", ".join(
            f"{pname}{'*' if schema.get('required') else ''}:{schema['type']}"
            for pname, schema in spec["params"].items()
        )
        lines.append(f"- {name}({params}) — {spec['description']}")
    return "\n".join(lines)
