"""P_geom 区块化 prompt 组装。

区块顺序固定为：
[TEXT_INTENT] → [IMAGE_OBSERVATION] → [POINT_OBSERVATION] → [POINT_HYPOTHESIS]
→ [POINT_TOOLS]（仅工具条件）→ plan constraints。

System prompt 默认 prompting.SYSTEM_PROMPTS["v3"]（已冻 V5/V8）。
新栈用 plan_prompt_version="v5"（Plan v3 + 本地姿态/配方）。
不把真实文件名、GT STEP/code、原始 XYZ 坐标列表写入 prompt。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .common import sha256_json
from .harness_guidance import build_guidance
from .pc_conditions import PCConditionSpec, parse_condition
from .pointcloud.io import PointCloudError, load_point_cloud
from .pointcloud.tools import tool_manifest
from .prompting import PLAN_TEMPLATES, SYSTEM_PROMPTS, _image_data_url


PROMPT_VERSION = "rq2.pc_geom.prompt.v1"
MAX_EVIDENCE_TOKENS = 2000
_IMAGE_CAPTION = (
    "[IMAGE_OBSERVATION]\n"
    "Surface-render inputs: four orthographic RGB surface renders in fixed "
    "front, side, top, and isometric order. Colors and lighting describe appearance only. "
    "Do not infer part names, provenance, or classes from appearance."
)
_PROJ_CAPTION = (
    "[POINT_OBSERVATION]\n"
    "Point-sampled inputs: four deterministic depth-contour encodings of the same "
    "2048-point cloud in fixed front, side, top, and isometric order. Grayscale encodes "
    "canonical camera depth using the same fixed range [-0.5, +0.5] for every image; "
    "cyan marks the projected contour. These are 2D projections, not native 3D measurements."
)


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _round_number(value: Any, digits: int = 4) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return [_round_number(item, digits) for item in value]
    if isinstance(value, dict):
        return {key: _round_number(item, digits) for key, item in value.items()}
    return value


def _quality_block(evidence: dict[str, Any]) -> dict[str, Any]:
    quality = evidence.get("quality") or {}
    return {
        "point_count": quality.get("point_count"),
        "valid_ratio": _round_number(quality.get("valid_ratio")),
        "degenerate": bool(quality.get("degenerate")),
        "median_nn_distance": _round_number(quality.get("median_nn_distance")),
    }


def _frame_for_profile(frame: dict[str, Any], profile: str) -> dict[str, Any]:
    if profile == "bbox":
        return {"center": _round_number(frame.get("center")), "bbox_size": _round_number(frame.get("bbox_size"))}
    return _round_number(frame)


def apply_evidence_profile(evidence: dict[str, Any], profile: str | None) -> dict[str, Any]:
    """按消融档过滤 PointEvidence（不重跑点云算法）。"""
    resolved = profile or "full"
    frame = evidence.get("frame") or {}
    compact: dict[str, Any] = {
        "schema": evidence.get("schema"),
        "cloud_id": evidence.get("cloud_id"),
        "frame": _frame_for_profile(frame, resolved),
        "quality": _quality_block(evidence),
        "evidence_profile": resolved,
    }
    if resolved in {"axes", "sym", "full", "partial"}:
        compact["frame"] = _frame_for_profile(frame, "axes")
    if resolved in {"sym", "full", "partial"}:
        compact["symmetry_candidates"] = _round_number(evidence.get("symmetry_candidates") or [])[:3]
    if resolved in {"full", "partial"}:
        compact["primitive_candidates"] = _round_number(evidence.get("primitive_candidates") or [])[:3]
        compact["sections"] = _round_number(evidence.get("sections") or {})
        compact["hypotheses"] = _round_number(evidence.get("hypotheses") or [])[:8]
        compact["uncertainties"] = _round_number(evidence.get("uncertainties") or [])[:8]
        compact["effective_section_thickness"] = _round_number(
            (evidence.get("config") or {}).get("effective_section_thickness")
        )
    if resolved == "partial":
        compact["sections"] = {
            axis: {
                "normal": block.get("normal"),
                "thickness": block.get("thickness"),
                "outer": block.get("outer"),
                "holes": [],
                "holes_hidden": True,
            }
            for axis, block in (compact.get("sections") or {}).items()
            if isinstance(block, dict)
        }
        compact["hypotheses"] = [
            item for item in compact.get("hypotheses") or [] if item.get("type") != "circular_hole"
        ]
        extra = list(compact.get("uncertainties") or [])
        extra.append(
            {
                "code": "section_holes_hidden",
                "message": "Hole and slot candidates were withheld. Query query_cross_section if needed.",
                "recommended_tool": "query_cross_section",
            }
        )
        compact["uncertainties"] = extra[:8]
    return compact


def compact_evidence_for_prompt(
    evidence: dict[str, Any],
    *,
    max_tokens: int = MAX_EVIDENCE_TOKENS,
    profile: str | None = None,
) -> dict[str, Any]:
    """把 PointEvidence 压成 prompt 可见块：去掉路径/内部配置，四位小数，限 token。"""
    compact = apply_evidence_profile(evidence, profile)
    payload = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if estimate_tokens(payload) <= max_tokens:
        return compact
    if "hypotheses" in compact:
        compact["hypotheses"] = compact["hypotheses"][:2]
    if "uncertainties" in compact:
        compact["uncertainties"] = compact["uncertainties"][:2]
    if isinstance(compact.get("sections"), dict):
        compact["sections"] = {
            axis: {
                "normal": block.get("normal"),
                "thickness": block.get("thickness"),
                "outer": block.get("outer"),
                "holes": (block.get("holes") or [])[:2],
            }
            for axis, block in compact["sections"].items()
            if isinstance(block, dict)
        }
    compact["truncated"] = True
    return compact


def evidence_observation_block(compact: dict[str, Any]) -> str:
    observation = {
        "schema": compact.get("schema"),
        "cloud_id": compact.get("cloud_id"),
        "evidence_profile": compact.get("evidence_profile"),
        "frame": compact.get("frame"),
        "quality": compact.get("quality"),
    }
    for key in ("primitive_candidates", "symmetry_candidates", "sections", "effective_section_thickness"):
        if key in compact:
            observation[key] = compact.get(key)
    return (
        "[POINT_OBSERVATION]\n"
        "Native 3D point-cloud measurements in the canonical frame "
        "(bbox center at origin, longest bbox edge = 1). "
        "These are computed locally from the input cloud. "
        "Do not invent precise sizes, hole diameters, or axis directions that are absent here.\n"
        + json.dumps(observation, ensure_ascii=False, indent=2)
    )


def evidence_hypothesis_block(compact: dict[str, Any]) -> str:
    hypotheses = compact.get("hypotheses") or []
    uncertainties = compact.get("uncertainties") or []
    return (
        "[POINT_HYPOTHESIS]\n"
        "The following items are hypotheses or uncertainties, not confirmed measurements. "
        "Treat them as suggestions to verify with a tool or to approximate conservatively.\n"
        + json.dumps(
            {"hypotheses": hypotheses, "uncertainties": uncertainties},
            ensure_ascii=False,
            indent=2,
        )
    )


_ORTHO_VIEWS = ("front", "side", "top", "isometric")
_LEGACY_RGB_VIEWS = ("view_0", "view_2", "view_4", "view_6")
_IMAGE_CAPTION_LEGACY = (
    "[IMAGE_OBSERVATION]\n"
    "Surface-render inputs: four RGB surface renders in the frozen view_0, view_2, "
    "view_4, and view_6 order (same images as the RQ2b I condition). "
    "Colors and lighting describe appearance only. "
    "Do not infer part names, provenance, or classes from appearance."
)


def _entry_images(images: Any, *, expected: tuple[str, ...] | None = _ORTHO_VIEWS) -> list[dict[str, Any]]:
    if not isinstance(images, list) or len(images) != 4:
        raise ValueError("每种视觉输入必须恰好包含 4 张图片")
    actual = [str(item.get("view")) for item in images if isinstance(item, dict)]
    if expected is not None and actual != list(expected):
        raise ValueError(f"视觉顺序必须为 {list(expected)}，实际为 {actual}")
    if expected is None and actual not in (list(_ORTHO_VIEWS), list(_LEGACY_RGB_VIEWS)):
        raise ValueError(f"视觉 view 标签无法识别: {actual}")
    for item in images:
        path = Path(str(item.get("path", "")))
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"视觉文件不存在或为空: {path.name}")
    return images


def _i1_images(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    encodings = row.get("render_encodings") or {}
    entry = encodings.get("I1") if isinstance(encodings, dict) else None
    if isinstance(entry, dict) and entry.get("images"):
        return _entry_images(entry["images"], expected=_ORTHO_VIEWS), _IMAGE_CAPTION
    images = row.get("images")
    if images:
        return _entry_images(images, expected=_LEGACY_RGB_VIEWS), _IMAGE_CAPTION_LEGACY
    raise KeyError("manifest 缺少 render_encodings.I1.images 或 images")


def _proj_images(row: dict[str, Any]) -> list[dict[str, Any]]:
    # 与 confirm C 臂 P 条件同源：pilot_v2 的 depth-contour 四视图。
    encoded = (row.get("point_cloud") or {}).get("encoding") or {}
    images = encoded.get("images")
    if images:
        return _entry_images(images)
    encodings = row.get("point_encodings") or {}
    for key in ("P3", "P2", "P1"):
        entry = encodings.get(key) if isinstance(encodings, dict) else None
        if isinstance(entry, dict) and entry.get("images"):
            return _entry_images(entry["images"])
    raise KeyError("manifest 缺少点云投影视图")


def _t1_text(row: dict[str, Any]) -> str:
    encodings = row.get("text_encodings") or {}
    entry = encodings.get("T1") if isinstance(encodings, dict) else None
    if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"].strip():
        return entry["text"].strip()
    text = ((row.get("text") or {}).get("L1") or "").strip()
    if not text:
        raise KeyError("manifest 缺少 text.L1 / text_encodings.T1")
    return text


def _t2_text(row: dict[str, Any]) -> str:
    encodings = row.get("text_encodings") or {}
    entry = encodings.get("T2") if isinstance(encodings, dict) else None
    if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"].strip():
        return entry["text"].strip()
    text = ((row.get("text") or {}).get("L3") or "").strip()
    if not text:
        raise KeyError("manifest 缺少 text.L3 / text_encodings.T2")
    return text


def _text_for_spec(row: dict[str, Any], spec: PCConditionSpec) -> tuple[str, str]:
    level = spec.text_level or "T1"
    if level == "T2":
        return _t2_text(row), "T2"
    return _t1_text(row), "T1"


def _append_images(content: list[dict[str, Any]], images: list[dict[str, Any]], image_max_edge: int) -> list[str]:
    hashes = []
    for item in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(Path(item["path"]), image_max_edge)},
            }
        )
        hashes.append(str(item.get("sha256") or ""))
    return hashes


def tool_instruction_block(
    spec: PCConditionSpec,
    *,
    cloud_id: str | None,
    max_pre_queries: int,
    max_post_queries: int,
) -> str:
    if not spec.tools:
        return ""
    return (
        "[POINT_TOOLS]\n"
        "You may output either (a) one query_request JSON or (b) the final Plan JSON. "
        "A query_request must be exactly: "
        '{"tool": "<name>", "params": {...}, "reason": "<short why>"}. '
        f"Bound cloud_id is {json.dumps(cloud_id)}. "
        "Do not query any other cloud_id. "
        f"Budget: at most {max_pre_queries} queries before the Plan, "
        f"and at most {max_post_queries} query after a candidate STEP exists "
        "(tools compare_cad_to_cloud / localize_geometric_error only then; "
        "candidate_step_id will be provided as cand_0).\n"
        + tool_manifest()
        + "\nIf evidence is sufficient, skip tools and output the Plan JSON now."
    )


def query_or_submit_block(
    spec: PCConditionSpec,
    *,
    cloud_id: str | None,
    max_queries: int,
    forced: bool,
) -> str:
    schema = (
        '{"action":"query","query":{"tool":"<name>","arguments":{}},"plan":null} '
        'or {"action":"submit_plan","query":null,"plan":{...}}'
    )
    force_line = (
        "You MUST issue action=query first with tool query_cross_section before any plan.\n"
        if forced
        else "Use action=query only when a withheld measurement is needed.\n"
    )
    return (
        "[POINT_TOOLS]\n"
        "First-round output must be a query_or_submit object: "
        f"{schema}. "
        f"Bound cloud_id is {json.dumps(cloud_id)}. "
        f"At most {max_queries} queries. Query history is accumulated. "
        + force_line
        + tool_manifest()
    )


def _row_points(row: dict[str, Any]):
    path = (row.get("point_cloud") or {}).get("path")
    if not path:
        return None
    npy = Path(str(path))
    if not npy.is_file():
        return None
    try:
        return load_point_cloud(npy)
    except (PointCloudError, OSError, ValueError):
        return None


def build_pc_messages(
    row: dict[str, Any],
    spec: PCConditionSpec | str,
    *,
    evidence: dict[str, Any] | None = None,
    image_max_edge: int = 1024,
    plan_prompt_version: str = "v3",
    max_pre_queries: int = 3,
    max_post_queries: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if plan_prompt_version not in SYSTEM_PROMPTS or plan_prompt_version not in PLAN_TEMPLATES:
        raise ValueError(f"plan_prompt_version 仅支持 {sorted(SYSTEM_PROMPTS)}")
    spec = spec if isinstance(spec, PCConditionSpec) else parse_condition(spec)
    prompt_sample_id = "s_" + hashlib.sha256(str(row["sample_id"]).encode("utf-8")).hexdigest()[:16]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Create a CAD plan for sample_id {json.dumps(prompt_sample_id)}. "
                "Use only the observations supplied below. Do not infer names, provenance, "
                "classes, or metadata. Do not invent precise dimensions that are not present "
                "in PointEvidence or tool results."
            ),
        }
    ]
    audit: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "plan_prompt_version": plan_prompt_version,
        "prompt_sample_id": prompt_sample_id,
        "condition_id": spec.condition_id,
        "slot_order": ["task", "text", "images", "point_proj", "point_geom", "guidance", "tools", "plan_constraints"],
        "modality_hashes": {},
        "allowed_modalities": sorted(spec.modalities),
    }

    if spec.text:
        text, text_level = _text_for_spec(row, spec)
        label = "Detailed construction description" if text_level == "T2" else "Short natural-language description"
        content.append({"type": "text", "text": f"[TEXT_INTENT]\n{label}:\n{text}"})
        audit["modality_hashes"]["text"] = sha256_json({text_level: text})
        audit["text_level"] = text_level

    if spec.images:
        images, image_caption = _i1_images(row)
        content.append({"type": "text", "text": image_caption})
        hashes = _append_images(content, images, image_max_edge)
        audit["modality_hashes"]["images"] = hashes
        audit["image_views"] = [item.get("view") for item in images]

    if spec.point_proj:
        images = _proj_images(row)
        content.append({"type": "text", "text": _PROJ_CAPTION})
        hashes = _append_images(content, images, image_max_edge)
        source = ((row.get("point_cloud") or {}).get("sha256")) or ""
        audit["modality_hashes"]["point_proj"] = {"source": source, "images": hashes}

    compact: dict[str, Any] | None = None
    if spec.point_geom:
        if not isinstance(evidence, dict):
            raise ValueError(f"条件 {spec.condition_id} 需要 PointEvidence")
        compact = compact_evidence_for_prompt(evidence, profile=spec.resolved_profile)
        payload = json.dumps(compact, ensure_ascii=False)
        audit["evidence_profile"] = spec.resolved_profile
        audit["shuffle"] = bool(spec.shuffle)
        audit["evidence_tokens"] = estimate_tokens(payload)
        audit["evidence_hash"] = evidence.get("content_hash")
        audit["cloud_id"] = compact.get("cloud_id")
        content.append({"type": "text", "text": evidence_observation_block(compact)})
        if compact.get("hypotheses") or compact.get("uncertainties"):
            content.append({"type": "text", "text": evidence_hypothesis_block(compact)})
        audit["modality_hashes"]["point_geom"] = {
            "content_hash": evidence.get("content_hash"),
            "cloud_id": compact.get("cloud_id"),
            "tokens": audit["evidence_tokens"],
        }

    if plan_prompt_version in {"v5", "v6"}:
        # 仅照片 / 仅投影条件不得读取 npy；空心半径只允许出现在 point_geom 条件
        guidance = build_guidance(
            evidence if isinstance(evidence, dict) else None,
            points=_row_points(row) if spec.point_geom else None,
        )
        if guidance["prompt_block"]:
            content.append({"type": "text", "text": guidance["prompt_block"]})
        audit["guidance"] = {
            "pose": guidance.get("pose"),
            "decisions": {
                "generator": ((guidance.get("decisions") or {}).get("generator") or {}).get("id"),
                "topology": ((guidance.get("decisions") or {}).get("topology") or {}).get("kind"),
                "path_graph": ((guidance.get("decisions") or {}).get("path_graph") or {}),
            },
        }

    if spec.tool_protocol in {"query_or_submit", "forced_query"}:
        tools_block = query_or_submit_block(
            spec,
            cloud_id=None if compact is None else compact.get("cloud_id"),
            max_queries=max_pre_queries,
            forced=spec.tool_protocol == "forced_query",
        )
    else:
        tools_block = tool_instruction_block(
            spec,
            cloud_id=None if compact is None else compact.get("cloud_id"),
            max_pre_queries=max_pre_queries,
            max_post_queries=max_post_queries,
        )
    if tools_block:
        content.append({"type": "text", "text": tools_block})
        audit["tools"] = True

    content.append(
        {
            "type": "text",
            "text": (
                "Required JSON shape (example values are illustrative only):\n"
                + json.dumps(PLAN_TEMPLATES[plan_prompt_version], ensure_ascii=False, indent=2)
                + "\nDo not copy the example dimensions. Every operation must include all "
                "fields required by the system schema. Prefer a simpler valid approximation "
                "when observations are ambiguous. Output JSON only."
            ),
        }
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[plan_prompt_version]},
        {"role": "user", "content": content},
    ]
    audit["prompt_sha256"] = sha256_json(messages)
    return messages, audit


def format_tool_user_message(result: dict[str, Any], *, remaining_pre: int, remaining_post: int, has_candidate: bool) -> str:
    body = json.dumps(result, ensure_ascii=False, indent=2)[:6000]
    if remaining_pre <= 0 and not has_candidate:
        closer = "Query budget before the Plan is exhausted. Output the complete Plan JSON now."
    elif has_candidate and remaining_post <= 0:
        closer = "Post-generation query budget is exhausted. Output the complete Plan JSON now."
    else:
        closer = (
            f"Remaining pre-generation queries: {remaining_pre}. "
            f"Remaining post-generation queries: {remaining_post}. "
            "If evidence is sufficient, output the Plan JSON; otherwise issue another query_request."
        )
    return f"[POINT_TOOL_RESULT]\n{body}\n{closer}"


def format_geometry_feedback(compare_result: dict[str, Any], *, candidate_step_id: str) -> str:
    compact = {
        "candidate_step_id": candidate_step_id,
        "source_coverage": compare_result.get("source_coverage"),
        "prediction_precision": compare_result.get("prediction_precision"),
        "fscore": compare_result.get("fscore") or compare_result.get("f1"),
        "chamfer": compare_result.get("chamfer"),
        "missing_ratio": compare_result.get("missing_ratio"),
        "extra_ratio": compare_result.get("extra_ratio"),
    }
    return (
        "[GEOMETRY_FEEDBACK]\n"
        f"A candidate STEP is registered as {candidate_step_id}. "
        "You may issue one compare_cad_to_cloud / localize_geometric_error query, "
        "or output a revised complete Plan JSON.\n"
        + json.dumps(_round_number(compact), ensure_ascii=False, indent=2)
    )


FORBIDDEN_PROMPT_MARKERS = (
    ".npy",
    ".step",
    ".stp",
    "gt_code",
    "processed\\",
    "processed/",
    "gt-secret",
)


def audit_payload(
    messages: list[dict[str, Any]],
    spec: PCConditionSpec,
    *,
    sample_id: str | None = None,
    family: str | None = None,
    gt_hash: str | None = None,
    max_evidence_tokens: int = MAX_EVIDENCE_TOKENS,
) -> dict[str, Any]:
    """Dry-run / 泄漏审计：不把结论写入 prompt，只返回检查结果。"""
    serialized = json.dumps(messages, ensure_ascii=False)
    image_count = serialized.count("data:image/png;base64,")
    expected_images = (4 if spec.images else 0) + (4 if spec.point_proj else 0)
    issues: list[str] = []
    if image_count != expected_images:
        issues.append(f"image_count {image_count} != expected {expected_images}")
    if spec.text != ("[TEXT_INTENT]" in serialized):
        issues.append("TEXT_INTENT 与 text 模态不一致")
    if spec.images != ("[IMAGE_OBSERVATION]" in serialized):
        issues.append("IMAGE_OBSERVATION 与 images 模态不一致")
    if spec.point_proj != ("cyan marks the projected contour" in serialized):
        issues.append("P_proj 投影说明与条件不一致")
    if spec.point_geom != ("[POINT_OBSERVATION]" in serialized and "Native 3D" in serialized):
        issues.append("POINT_OBSERVATION 与 point_geom 模态不一致")
    if spec.tools != ("[POINT_TOOLS]" in serialized):
        issues.append("POINT_TOOLS 与 tools 开关不一致")
    if spec.point_geom and spec.point_proj:
        issues.append("同一条件同时含 P_proj 与 P_geom")
    profile = spec.resolved_profile
    if spec.point_geom and profile == "bbox":
        if '"symmetry_candidates"' in serialized or '"sections"' in serialized:
            issues.append("bbox profile 含 symmetry/sections")
    if spec.point_geom and profile == "axes":
        if '"symmetry_candidates"' in serialized or '"sections"' in serialized:
            issues.append("axes profile 含 symmetry/sections")
    if spec.point_geom and profile == "sym":
        if '"sections"' in serialized:
            issues.append("sym profile 含 sections")
    for marker in FORBIDDEN_PROMPT_MARKERS:
        if marker.lower() in serialized.lower():
            issues.append(f"prompt 含禁止标记 {marker}")
    if sample_id and sample_id in serialized:
        issues.append("prompt 含真实 sample_id")
    if (
        family
        and not spec.text
        and re.search(rf"(?<![A-Za-z0-9_]){re.escape(family)}(?![A-Za-z0-9_])", serialized)
    ):
        issues.append("prompt 含 family 名")
    if gt_hash and gt_hash in serialized:
        issues.append("prompt 含 GT hash")
    # 不允许把大量 XYZ 原样塞进 prompt
    if serialized.count("[0.") > 80 and "operations" not in serialized[-2000:]:
        issues.append("疑似大量原始坐标被写入 prompt")
    evidence_tokens = 0
    if spec.point_geom:
        start = serialized.find("[POINT_OBSERVATION]")
        end = serialized.find("[POINT_HYPOTHESIS]")
        if start >= 0 and end > start:
            evidence_tokens = estimate_tokens(serialized[start:end])
        if evidence_tokens > max_evidence_tokens:
            issues.append(f"evidence tokens {evidence_tokens} > {max_evidence_tokens}")
    return {
        "ok": not issues,
        "issues": issues,
        "image_count": image_count,
        "expected_images": expected_images,
        "chars": len(serialized),
        "evidence_tokens": evidence_tokens,
    }
