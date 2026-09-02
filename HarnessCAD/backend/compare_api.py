"""Serve experiment cases for GT / input / prediction comparison."""

from __future__ import annotations

import csv
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/compare", tags=["compare"])


def _detect_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in here.parents:
        if (parent / "rq2_multimodal_harness").is_dir():
            return parent
        if (parent / "experiments" / "rq2_multimodal_harness").is_dir():
            return parent
    return here.parents[1]


PROJECT_ROOT = _detect_project_root()
_RQ2_ROOT = PROJECT_ROOT / "rq2_multimodal_harness"
if not _RQ2_ROOT.is_dir():
    _RQ2_ROOT = PROJECT_ROOT / "experiments" / "rq2_multimodal_harness"
if str(_RQ2_ROOT) not in sys.path:
    sys.path.insert(0, str(_RQ2_ROOT))
from rq2_harness.harness_guidance import build_guidance  # noqa: E402
from rq2_harness.pc_conditions import parse_condition  # noqa: E402

V5_ROOT = _RQ2_ROOT / "outputs/v5_complementarity"
V5_STATE = V5_ROOT / "repeats/state"
V5_METRICS = V5_ROOT / "repeats/analysis/primary_metrics.csv"
RENDERS = PROJECT_ROOT / "processed/renders/benchcad"
MODELS = PROJECT_ROOT / "processed/models/benchcad"
POINTCLOUDS = PROJECT_ROOT / "processed/point_clouds/benchcad/2048"
TEXT_JSONL = PROJECT_ROOT / "processed/text_descriptions/benchcad/code_gen_hdv3.jsonl"
EVIDENCE_DIR = V5_ROOT / "evidence"
PC_VIEWS = V5_ROOT / "pointcloud_views"
STL_CACHE = _RQ2_ROOT / "outputs/_case_gallery/stl_cache"

FEATURED = (
    ("i_beam_000097_s20260505", "接近贴合的棱柱件"),
    ("slotted_plate_000582_s20260505", "立板：仅照片会躺平"),
    ("pipe_elbow_000924_s20260505", "建得出来但不像的弯管"),
)
DEFAULT_CONDITION = "I1P_geom"
IMAGE_VIEWS = ("view_0", "view_2", "view_4", "view_6")
PC_VIEWS_ORDER = ("front", "side", "top", "isometric")


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round(item, digits) for item in value]
    return value


@lru_cache(maxsize=1)
def _text_index() -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    if not TEXT_JSONL.is_file():
        return index
    with TEXT_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            stem = row.get("stem")
            if not stem or row.get("lang") not in {None, "en"}:
                continue
            bucket = index.setdefault(stem, {})
            level = row.get("level")
            if level in {"L1", "L3"}:
                bucket[level] = row.get("text") or ""
    return index


@lru_cache(maxsize=1)
def _metric_rows() -> list[dict[str, str]]:
    if not V5_METRICS.is_file():
        return []
    with V5_METRICS.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _state_path(stem: str, condition: str) -> Path:
    return V5_STATE / stem / condition / "r01.json"


def _load_state(stem: str, condition: str) -> dict[str, Any]:
    path = _state_path(stem, condition)
    if not path.is_file():
        raise HTTPException(404, f"没有找到 {stem} / {condition} 的实验记录")
    return json.loads(path.read_text(encoding="utf-8"))


def _step_to_stl(step_path: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_mtime >= step_path.stat().st_mtime:
        return dest
    import cadquery as cq

    shape = cq.importers.importStep(str(step_path))
    if shape is None or shape.val() is None:
        raise HTTPException(500, f"无法读取 STEP：{step_path.name}")
    cq.exporters.export(shape, str(dest))
    return dest


def _quality_label(joint: float | None) -> str:
    if joint is None:
        return "无法评分"
    if joint >= 0.85:
        return "几乎贴合真值"
    if joint >= 0.55:
        return "大外形对，细节有偏差"
    if joint >= 0.25:
        return "能看出同类零件，差得较远"
    return "几乎对不上"


def _evidence_brief(evidence: dict[str, Any] | None) -> dict[str, Any]:
    if not evidence:
        return {}
    frame = evidence.get("frame") or {}
    quality = evidence.get("quality") or {}
    sections = {}
    for axis, block in (evidence.get("sections") or {}).items():
        outer = (block or {}).get("outer") or {}
        sections[axis] = {
            "bbox_size": _round(outer.get("bbox_size")),
            "point_count": outer.get("point_count"),
            "closed": outer.get("closed"),
        }
    return {
        "bbox_size": _round(frame.get("bbox_size")),
        "point_count": quality.get("point_count"),
        "sections": sections,
        "symmetry": [
            {
                "type": item.get("type"),
                "confidence": _round(item.get("confidence")),
            }
            for item in (evidence.get("symmetry_candidates") or [])[:3]
        ],
    }


@router.get("/cases")
def list_cases(condition: str = DEFAULT_CONDITION) -> dict[str, Any]:
    featured_ids = {item[0] for item in FEATURED}
    rows = [
        row
        for row in _metric_rows()
        if row.get("condition") == condition and row.get("repeat_id") == "1"
    ]
    items = []
    for row in rows:
        try:
            joint = float(row["joint_quality"])
        except (KeyError, TypeError, ValueError):
            joint = None
        items.append(
            {
                "stem": row["sample_id"],
                "family": row.get("family"),
                "difficulty": row.get("difficulty"),
                "condition": condition,
                "joint_quality": joint,
                "status": row.get("status"),
                "featured": row["sample_id"] in featured_ids,
                "label": _quality_label(joint),
            }
        )
    items.sort(key=lambda item: (-int(item["featured"]), -(item["joint_quality"] or -1)))
    return {"condition": condition, "count": len(items), "cases": items}


@router.get("/cases/{stem}")
def get_case(stem: str, condition: str = DEFAULT_CONDITION) -> dict[str, Any]:
    state = _load_state(stem, condition)
    geometry = state.get("geometry") or {}
    prompt = state.get("prompt_audit") or {}
    text = _text_index().get(stem, {})
    evidence_path = EVIDENCE_DIR / f"{stem}.point_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else None
    try:
        spec = parse_condition(condition)
        allow_cloud = bool(spec.point_geom)
    except ValueError:
        allow_cloud = condition != "I1"
    npy_path = POINTCLOUDS / f"{stem}.npy"
    points = None
    if allow_cloud and npy_path.is_file():
        loaded = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        if getattr(loaded, "ndim", 0) == 2 and loaded.shape[1] >= 3:
            points = np.asarray(loaded[:, :3], dtype=np.float64)
    evidence_for_guidance = evidence if allow_cloud else None
    guidance = build_guidance(evidence_for_guidance, points=points)
    pred_step = Path(state.get("result_step_path") or "")
    gt_step = MODELS / f"{stem}.step"
    sent = set(prompt.get("allowed_modalities") or [])
    joint = geometry.get("joint_quality")
    voxel = ((geometry.get("voxel_iou") or geometry.get("shape_voxel_iou") or {}).get("value"))
    return {
        "stem": stem,
        "family": stem.rsplit("_", 2)[0] if "_" in stem else stem,
        "condition": condition,
        "condition_note": "V5 照片 + 点云几何描述（C 臂）" if condition == "I1P_geom" else condition,
        "sent_modalities": sorted(sent),
        "status": (state.get("stage") or {}).get("episode_status") or state.get("status"),
        "distance": {
            "joint_quality": joint,
            "shape_only_cd": geometry.get("shape_only_cd"),
            "common_frame_cd": geometry.get("common_frame_cd"),
            "voxel_iou": voxel,
            "label": _quality_label(joint if isinstance(joint, (int, float)) else None),
            "gt_size": (geometry.get("bbox") or {}).get("gt_size"),
            "pred_size": (geometry.get("bbox") or {}).get("pred_size"),
        },
        "inputs": {
            "images_sent": "images" in sent,
            "point_geom_sent": "point_geom" in sent,
            "text_sent": "text" in sent,
            "image_views": list(IMAGE_VIEWS),
            "pointcloud_views": list(PC_VIEWS_ORDER),
            "l1": text.get("L1") or "",
            "l3": text.get("L3") or "",
            "point_evidence": _evidence_brief(evidence),
            "local_guidance": guidance.get("compare"),
        },
        "assets": {
            "gt_stl": f"/api/compare/stl/{stem}?kind=gt",
            "pred_stl": f"/api/compare/stl/{stem}?kind=pred&condition={condition}" if pred_step.is_file() else None,
            "pointcloud": f"/api/compare/pointcloud/{stem}",
            "images": [f"/api/compare/image/{stem}/{view}" for view in IMAGE_VIEWS],
            "pointcloud_images": [f"/api/compare/pc-image/{stem}/{view}" for view in PC_VIEWS_ORDER],
        },
        "available": {
            "gt_step": gt_step.is_file(),
            "pred_step": pred_step.is_file(),
            "pointcloud": (POINTCLOUDS / f"{stem}.npy").is_file(),
        },
    }


@router.get("/stl/{stem}")
def get_stl(stem: str, kind: str = "gt", condition: str = DEFAULT_CONDITION) -> FileResponse:
    if kind == "gt":
        step = MODELS / f"{stem}.step"
        dest = STL_CACHE / f"{stem}__gt.stl"
    elif kind == "pred":
        state = _load_state(stem, condition)
        step = Path(state.get("result_step_path") or "")
        dest = STL_CACHE / f"{stem}__{condition}__pred.stl"
    else:
        raise HTTPException(400, "kind 只能是 gt 或 pred")
    if not step.is_file():
        raise HTTPException(404, f"没有 {kind} STEP")
    try:
        _step_to_stl(step, dest)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"STEP 转 STL 失败：{exc}") from exc
    return FileResponse(dest, media_type="model/stl", filename=dest.name)


@router.get("/image/{stem}/{view}")
def get_image(stem: str, view: str) -> FileResponse:
    if view not in IMAGE_VIEWS:
        raise HTTPException(400, "未知图像视角")
    path = RENDERS / stem / f"{view}.png"
    if not path.is_file():
        raise HTTPException(404, f"没有输入图 {view}")
    return FileResponse(path, media_type="image/png")


@router.get("/pc-image/{stem}/{view}")
def get_pc_image(stem: str, view: str) -> FileResponse:
    if view not in PC_VIEWS_ORDER:
        raise HTTPException(400, "未知点云视角")
    path = PC_VIEWS / f"{stem}__{view}.png"
    if not path.is_file():
        raise HTTPException(404, f"没有点云投影 {view}")
    return FileResponse(path, media_type="image/png")


@router.get("/pointcloud/{stem}")
def get_pointcloud(stem: str) -> dict[str, Any]:
    path = POINTCLOUDS / f"{stem}.npy"
    if not path.is_file():
        raise HTTPException(404, "没有点云文件")
    points = np.load(path, mmap_mode="r", allow_pickle=False)
    if points.ndim != 2 or points.shape[1] < 3:
        raise HTTPException(500, "点云形状无效")
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    center = xyz.mean(axis=0)
    shifted = xyz - center
    scale = float(np.max(np.ptp(shifted, axis=0))) or 1.0
    canonical = (shifted / scale).tolist()
    return {"count": len(canonical), "points": canonical}
