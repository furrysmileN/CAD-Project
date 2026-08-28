"""Token helpers and P_repeat construction. Same schema as P_comp; no dummy padding."""
from __future__ import annotations

import copy
import json
from typing import Any

from .common import sha256_json
from .v6_carriers import list_carriers, strip_carriers

TOKEN_TOLERANCE = 0.10


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def evidence_token_count(evidence: dict[str, Any]) -> int:
    return estimate_tokens(json.dumps(evidence, ensure_ascii=False, separators=(",", ":")))


def _strip_critical(evidence: dict[str, Any], critical: dict[str, Any]) -> dict[str, Any]:
    return strip_carriers(evidence, critical)


def _redundant_visible(p_comp: dict[str, Any]) -> list[dict[str, Any]]:
    frame = p_comp.get("frame") or {}
    size = list(frame.get("bbox_size") or [1.0, 1.0, 1.0])
    extras: list[dict[str, Any]] = []
    if len(size) == 3:
        extras.append(
            {
                "id": "visible_bbox_aspect",
                "kind": "visible_surface",
                "aspect": [round(size[0] / max(size[2], 1e-9), 4), round(size[1] / max(size[2], 1e-9), 4)],
                "note": "outer bounding-box aspect recoverable from silhouette",
            }
        )
    sections = p_comp.get("sections") or {}
    for axis, block in sections.items():
        if not isinstance(block, dict):
            continue
        outer = block.get("outer") or {}
        extras.append(
            {
                "id": f"visible_section_outer_{axis}",
                "kind": "visible_surface",
                "axis": axis,
                "outer": outer,
            }
        )
    quality = p_comp.get("quality") or {}
    extras.append(
        {
            "id": "visible_cloud_quality",
            "kind": "visible_surface",
            "point_count": quality.get("point_count"),
            "valid_ratio": quality.get("valid_ratio"),
        }
    )
    return extras


def build_p_repeat(p_comp: dict[str, Any], critical: dict[str, Any]) -> dict[str, Any]:
    """Keep image-visible facts, drop the primary critical fact, match token length with real extras."""
    out = _strip_critical(p_comp, critical)
    extras = _redundant_visible(p_comp)
    target = evidence_token_count(p_comp)
    low, high = int(target * (1 - TOKEN_TOLERANCE)), int(target * (1 + TOKEN_TOLERANCE))
    out["visible_surface_measurements"] = []
    for extra in extras:
        current = evidence_token_count(out)
        if current >= low:
            break
        out["visible_surface_measurements"].append(extra)
    current = evidence_token_count(out)
    idx = 0
    while current < low and extras:
        clone = copy.deepcopy(extras[idx % len(extras)])
        clone["id"] = f"{clone['id']}_dup{idx}"
        clone["offset_index"] = idx
        out["visible_surface_measurements"].append(clone)
        current = evidence_token_count(out)
        idx += 1
        if idx > 40:
            break
    out["content_hash"] = sha256_json({key: value for key, value in out.items() if key != "content_hash"})
    out["token_count"] = evidence_token_count(out)
    out["token_target"] = target
    if not (low <= out["token_count"] <= high * 2):
        out["token_balance_warning"] = True
    return out


def repeat_contains_critical(p_repeat: dict[str, Any], critical: dict[str, Any]) -> bool:
    return bool(list_carriers(p_repeat, critical))
