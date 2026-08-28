"""Deterministic single-field corruption for P_wrong."""
from __future__ import annotations

import copy
import hashlib
from typing import Any

from .common import sha256_json

CORRUPTION_VERSION = "rq2.v6.corruption.v1"


def _seed_bits(sample_id: str, fact_id: str) -> int:
    digest = hashlib.sha256(f"{CORRUPTION_VERSION}:{sample_id}:{fact_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _scale(bits: int) -> float:
    return 0.75 if bits % 2 == 0 else 1.25


def _mutate_value(category: str, value: Any, bits: int, bbox_longest: float) -> tuple[Any, dict[str, Any]]:
    if category == "depth" or category == "radius_or_width":
        scale = _scale(bits)
        if isinstance(value, (int, float)):
            return round(float(value) * scale, 6), {"type": "scale", "factor": scale}
        raise TypeError("数值类事实需要标量")
    if category == "through_vs_blind":
        flipped = "through" if value == "blind" else "blind"
        return flipped, {"type": "flip", "from": value, "to": flipped}
    if category == "hidden_presence":
        flipped = not bool(value)
        return flipped, {"type": "flip", "from": bool(value), "to": flipped}
    if category == "offset_or_spacing":
        delta = round(0.10 * float(bbox_longest), 6)
        sign = 1.0 if bits % 2 == 0 else -1.0
        if isinstance(value, (int, float)):
            return round(float(value) + sign * delta, 6), {"type": "offset", "delta": sign * delta}
        raise TypeError("spacing 需要标量")
    if category == "axis_or_symmetry":
        axes = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        replacement = list(axes[bits % 3])
        return replacement, {"type": "axis_replace", "to": replacement}
    raise ValueError(f"未知 critical 类别: {category}")


def _replace_in(obj: Any, *, fact_id: str, old: Any, new: Any, budget: list[int]) -> Any:
    if budget[0] <= 0:
        return obj
    if obj == old and budget[0] > 0:
        budget[0] -= 1
        return copy.deepcopy(new)
    if isinstance(obj, dict):
        if obj.get("fact_id") == fact_id and "value" in obj:
            updated = dict(obj)
            updated["value"] = copy.deepcopy(new)
            budget[0] -= 1
            return updated
        return {key: _replace_in(value, fact_id=fact_id, old=old, new=new, budget=budget) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_replace_in(item, fact_id=fact_id, old=old, new=new, budget=budget) for item in obj]
    return obj


def build_p_wrong(p_comp: dict[str, Any], critical: dict[str, Any], *, sample_id: str) -> dict[str, Any]:
    bits = _seed_bits(sample_id, str(critical.get("fact_id")))
    frame = p_comp.get("frame") or {}
    size = frame.get("bbox_size") or [1.0, 1.0, 1.0]
    longest = max(float(x) for x in size) if size else 1.0
    new_value, spec = _mutate_value(str(critical["category"]), critical.get("value"), bits, longest)
    out = copy.deepcopy(p_comp)
    facts = list(out.get("cad_facts") or [])
    replaced = False
    for item in facts:
        if item.get("role") == "primary_critical" or item.get("fact_id") == critical.get("fact_id"):
            item["value"] = copy.deepcopy(new_value)
            replaced = True
            break
    if not replaced:
        facts.append(
            {
                "fact_id": critical.get("fact_id"),
                "category": critical.get("category"),
                "value": new_value,
                "role": "primary_critical",
            }
        )
    out["cad_facts"] = facts
    out["corruption"] = {
        "version": CORRUPTION_VERSION,
        "seed_bits": bits,
        "fact_id": critical.get("fact_id"),
        "category": critical.get("category"),
        **spec,
        "new_value": new_value,
    }
    out["content_hash"] = sha256_json({key: value for key, value in out.items() if key != "content_hash"})
    return out


def unchanged_except_critical(p_comp: dict[str, Any], p_wrong: dict[str, Any], critical: dict[str, Any]) -> bool:
    def _clean(blob: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(blob)
        out.pop("content_hash", None)
        out.pop("corruption", None)
        fact_id = critical.get("fact_id")
        out["cad_facts"] = [
            item
            for item in (out.get("cad_facts") or [])
            if item.get("role") != "primary_critical" and item.get("fact_id") != fact_id
        ]
        return out

    return json_equal(_clean(p_comp), _clean(p_wrong))


def json_equal(a: Any, b: Any) -> bool:
    import json

    return json.dumps(a, ensure_ascii=False, sort_keys=True) == json.dumps(b, ensure_ascii=False, sort_keys=True)
