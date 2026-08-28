from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .common import sha256_json
from .prompting import PLAN_TEMPLATES, SYSTEM_PROMPTS, _image_data_url


PROMPT_VERSION = "rq2.encoding_screen.prompt.v1"

_TEXT_LABELS = {
    "T1": "Short natural-language description",
    "T2": "Detailed procedural description",
    "T3": "Structured CAD brief",
}

_RENDER_CAPTIONS = {
    "I1": (
        "Surface-render inputs: four orthographic RGB surface renders in fixed "
        "front, side, top, and isometric order. Colors and lighting describe appearance only."
    ),
    "I2": (
        "Surface-render inputs: four orthographic object-space normal maps in fixed "
        "front, side, top, and isometric order. RGB encodes XYZ normal components as "
        "(component + 1) / 2."
    ),
    "I3": (
        "Surface-render inputs: four orthographic engineering line drawings in fixed "
        "front, side, top, and isometric order. Black lines are visible silhouettes and "
        "mechanical feature edges after hidden-line removal; white is background."
    ),
}

_POINT_CAPTIONS = {
    "P1": (
        "Point-sampled inputs: four orthographic projections of the same 2048-point cloud "
        "in fixed front, side, top, and isometric order. Black marks are discrete points; "
        "there is no depth color, interpolation, or reconstructed surface."
    ),
    "P2": (
        "Point-sampled inputs: four orthographic projections of the same 2048-point cloud "
        "in fixed front, side, top, and isometric order. Point color encodes canonical "
        "camera depth with one fixed mapping shared by every image: -0.5 is blue, 0 is "
        "green, and +0.5 is red. Points are not connected or interpolated."
    ),
    "P3": (
        "Point-sampled inputs: four deterministic depth-contour encodings of the same "
        "2048-point cloud in fixed front, side, top, and isometric order. Grayscale encodes "
        "canonical camera depth using the same fixed range [-0.5, +0.5] for every image; "
        "cyan marks the projected contour."
    ),
}


def _spec_value(spec: Any, name: str) -> str | None:
    if isinstance(spec, dict):
        value = spec.get(name)
    else:
        value = getattr(spec, name)
    return None if value is None else str(value)


def _encoding_entry(row: dict[str, Any], group: str, encoding: str) -> dict[str, Any]:
    entries = row.get(group)
    if not isinstance(entries, dict) or not isinstance(entries.get(encoding), dict):
        raise KeyError(f"manifest 缺少 {group}.{encoding}")
    return entries[encoding]


def _entry_text(entry: dict[str, Any]) -> str:
    value = entry.get("text")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("文本编码为空")
    return value.strip()


def _entry_images(entry: dict[str, Any]) -> list[dict[str, Any]]:
    images = entry.get("images")
    if not isinstance(images, list) or len(images) != 4:
        raise ValueError("每种视觉编码必须恰好包含4张图片")
    expected = ["front", "side", "top", "isometric"]
    actual = [str(item.get("view")) for item in images if isinstance(item, dict)]
    if actual != expected:
        raise ValueError(f"视觉顺序必须为 {expected}，实际为 {actual}")
    for item in images:
        path = Path(str(item.get("path", "")))
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"视觉编码文件不存在或为空: {path}")
    return images


def build_encoding_messages(
    row: dict[str, Any],
    spec: Any,
    *,
    image_max_edge: int = 1024,
    plan_prompt_version: str = "v2",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if plan_prompt_version not in SYSTEM_PROMPTS or plan_prompt_version not in PLAN_TEMPLATES:
        raise ValueError(f"plan_prompt_version 仅支持 {sorted(SYSTEM_PROMPTS)}")
    text_encoding = _spec_value(spec, "text")
    render_encoding = _spec_value(spec, "render")
    point_encoding = _spec_value(spec, "point")
    if text_encoding is None and render_encoding is None and point_encoding is None:
        raise ValueError("三种模态不能同时缺失")

    prompt_sample_id = "s_" + hashlib.sha256(str(row["sample_id"]).encode("utf-8")).hexdigest()[:16]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Create a CAD plan for sample_id {json.dumps(prompt_sample_id)}. "
                "Use only the observations supplied below. Do not infer names, provenance, "
                "classes, or metadata."
            ),
        }
    ]
    audit: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "plan_prompt_version": plan_prompt_version,
        "prompt_sample_id": prompt_sample_id,
        "text_encoding": text_encoding,
        "render_encoding": render_encoding,
        "point_encoding": point_encoding,
        "slot_order": ["task", "text", "render", "point_cloud", "plan_constraints"],
        "modality_hashes": {},
    }

    if text_encoding is not None:
        if text_encoding not in _TEXT_LABELS:
            raise ValueError(f"未知文本编码 {text_encoding}")
        entry = _encoding_entry(row, "text_encodings", text_encoding)
        text = _entry_text(entry)
        content.append({"type": "text", "text": f"{_TEXT_LABELS[text_encoding]}:\n{text}"})
        audit["modality_hashes"]["text"] = entry.get("sha256") or sha256_json({"text": text})

    if render_encoding is not None:
        if render_encoding not in _RENDER_CAPTIONS:
            raise ValueError(f"未知渲染编码 {render_encoding}")
        entry = _encoding_entry(row, "render_encodings", render_encoding)
        images = _entry_images(entry)
        content.append({"type": "text", "text": _RENDER_CAPTIONS[render_encoding]})
        for item in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_data_url(Path(item["path"]), image_max_edge),
                    },
                }
            )
        audit["modality_hashes"]["render"] = {
            "images": [item["sha256"] for item in images],
            "params": entry.get("params"),
        }

    if point_encoding is not None:
        if point_encoding not in _POINT_CAPTIONS:
            raise ValueError(f"未知点云编码 {point_encoding}")
        entry = _encoding_entry(row, "point_encodings", point_encoding)
        images = _entry_images(entry)
        content.append({"type": "text", "text": _POINT_CAPTIONS[point_encoding]})
        for item in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_data_url(Path(item["path"]), image_max_edge),
                    },
                }
            )
        audit["modality_hashes"]["point_cloud"] = {
            "source": entry.get("source_sha256"),
            "images": [item["sha256"] for item in images],
            "params": entry.get("params"),
        }

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
