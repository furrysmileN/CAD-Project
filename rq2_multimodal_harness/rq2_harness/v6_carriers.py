"""Fact carrier graph: every copy of a critical fact in PointEvidence.

P_repeat must delete all carriers, not just the primary cad_facts row.
P_counterfactual (later) must replace the same set with a self-consistent value.
"""
from __future__ import annotations

import copy
import json
import math
from typing import Any

from .common import sha256_json

CARRIER_VERSION = "rq2.v6.carrier.v1"


def _same_scalar(a: Any, b: Any, *, abs_tol: float = 1e-3) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, bool) and isinstance(b, bool):
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    try:
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            import numpy as np

            va = np.asarray(a, dtype=float)
            vb = np.asarray(b, dtype=float)
            va = va / max(float(np.linalg.norm(va)), 1e-9)
            vb = vb / max(float(np.linalg.norm(vb)), 1e-9)
            return abs(float(va @ vb)) >= 0.999
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=abs_tol)
    except (TypeError, ValueError):
        return False


def _primary_value(evidence: dict[str, Any], critical: dict[str, Any]) -> Any:
    fact_id = str(critical.get("fact_id") or "")
    for item in evidence.get("cad_facts") or []:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "primary_critical" or (fact_id and item.get("fact_id") == fact_id):
            return item.get("value")
    return None


def list_carriers(evidence: dict[str, Any], critical: dict[str, Any], *, values: list[Any] | None = None) -> list[str]:
    """Return stable path strings for every copy of the critical fact."""
    fact_id = str(critical.get("fact_id") or "")
    category = str(critical.get("category") or "")
    targets = [item for item in (values or [_primary_value(evidence, critical)]) if item is not None]
    hits: list[str] = []
    for index, item in enumerate(evidence.get("cad_facts") or []):
        if not isinstance(item, dict):
            continue
        if fact_id and item.get("fact_id") == fact_id:
            hits.append(f"cad_facts[{index}]:{fact_id}")
            continue
        if item.get("category") == category and any(_same_scalar(item.get("value"), target) for target in targets):
            hits.append(f"cad_facts[{index}]:{item.get('fact_id')}")
        if category == "depth" and any(_same_scalar(item.get("depth"), target) for target in targets):
            hits.append(f"cad_facts[{index}].depth")
    sections = evidence.get("sections") or {}
    if isinstance(sections, dict):
        for axis, block in sections.items():
            if not isinstance(block, dict):
                continue
            for hole_i, hole in enumerate(block.get("holes") or []):
                if not isinstance(hole, dict):
                    continue
                if category == "through_vs_blind":
                    mapped = "through" if hole.get("through") else "blind"
                    if any(_same_scalar(mapped, target) for target in targets):
                        hits.append(f"sections.{axis}.holes[{hole_i}]")
                if category in {"depth", "radius_or_width", "offset_or_spacing"}:
                    for key in ("depth", "radius", "diameter"):
                        if any(_same_scalar(hole.get(key), target) for target in targets):
                            hits.append(f"sections.{axis}.holes[{hole_i}]")
    for index, item in enumerate(evidence.get("hypotheses") or []):
        blob = json.dumps(item, ensure_ascii=False)
        if fact_id and fact_id in blob:
            hits.append(f"hypotheses[{index}]")
            continue
        if any(target is not None and str(target) in blob for target in targets):
            hits.append(f"hypotheses[{index}]")
    return sorted(set(hits))


def strip_carriers(evidence: dict[str, Any], critical: dict[str, Any]) -> dict[str, Any]:
    """Delete every carrier of the critical fact. Does not copy GT."""
    out = copy.deepcopy(evidence)
    fact_id = str(critical.get("fact_id") or "")
    category = str(critical.get("category") or "")
    values = [_primary_value(evidence, critical)]
    drop_paths = set(list_carriers(evidence, critical, values=values))

    kept_facts = []
    for index, item in enumerate(out.get("cad_facts") or []):
        path_id = f"cad_facts[{index}]:{item.get('fact_id')}"
        path_depth = f"cad_facts[{index}].depth"
        if path_id in drop_paths or path_depth in drop_paths:
            continue
        if item.get("role") == "primary_critical":
            continue
        if fact_id and item.get("fact_id") == fact_id:
            continue
        if item.get("category") == category:
            continue
        kept_facts.append(item)
    out["cad_facts"] = kept_facts

    sections = out.get("sections")
    if isinstance(sections, dict):
        for axis, block in list(sections.items()):
            if not isinstance(block, dict):
                continue
            holes = []
            for hole_i, hole in enumerate(block.get("holes") or []):
                if f"sections.{axis}.holes[{hole_i}]" in drop_paths:
                    continue
                holes.append(hole)
            block["holes"] = holes
    if isinstance(out.get("hypotheses"), list):
        out["hypotheses"] = [
            item
            for index, item in enumerate(out["hypotheses"])
            if f"hypotheses[{index}]" not in drop_paths
        ]
    out.pop("critical_fact_withheld", None)
    out["carrier_graph"] = {
        "version": CARRIER_VERSION,
        "stripped_fact_id": fact_id,
        "stripped_category": category,
        "stripped_paths": sorted(drop_paths),
    }
    out["content_hash"] = sha256_json({key: value for key, value in out.items() if key != "content_hash"})
    return out
