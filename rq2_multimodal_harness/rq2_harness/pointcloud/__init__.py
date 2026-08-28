"""本地点云几何证据服务（PointCloud Service v1）。

从 .npy 点云计算 PointEvidence（三维几何摘要 + 假设 + 不确定项），并提供
按需查询工具（有限状态机协议），供外部 VLM API 的 P_geom 条件使用。

同时再导出同目录 `pointcloud.py` 的深度剪影编码 API，避免包名遮蔽后
`prepare.py` / `test_core.py` 找不到 `encode_point_cloud`。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .canonical import CanonicalTransform
from .evidence import EVIDENCE_SCHEMA, build_evidence
from .io import PointCloudError, clean_points, hash_points, load_point_cloud
from .service import PointCloudService
from .tools import (
    TOOL_PROTOCOL_VERSION,
    TOOL_SPECS,
    PointCloudSession,
    execute_tool,
    parse_query_request,
    tool_manifest,
)

_legacy_path = Path(__file__).resolve().parent.parent / "pointcloud.py"
_legacy_spec = importlib.util.spec_from_file_location("rq2_harness._pc_encode_legacy", _legacy_path)
if _legacy_spec is None or _legacy_spec.loader is None:
    raise ImportError(f"无法加载点云编码模块 {_legacy_path}")
_legacy = importlib.util.module_from_spec(_legacy_spec)
_legacy_spec.loader.exec_module(_legacy)
CAMERAS = _legacy.CAMERAS
encode_point_cloud = _legacy.encode_point_cloud
normalize_points = _legacy.normalize_points
render_depth_contour = _legacy.render_depth_contour

__all__ = [
    "CAMERAS",
    "CanonicalTransform",
    "EVIDENCE_SCHEMA",
    "PointCloudError",
    "PointCloudService",
    "PointCloudSession",
    "TOOL_PROTOCOL_VERSION",
    "TOOL_SPECS",
    "build_evidence",
    "clean_points",
    "encode_point_cloud",
    "execute_tool",
    "hash_points",
    "load_point_cloud",
    "normalize_points",
    "parse_query_request",
    "render_depth_contour",
    "tool_manifest",
]
