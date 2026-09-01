"""BenchCAD GT 操作阶层与 FeaturePlan v3 表达力覆盖。"""
from __future__ import annotations

from typing import Any, Iterable

from .audit_expressivity import cadquery_operations

ADVANCED_GT_OPS = frozenset(
    {
        "sweep",
        "loft",
        "makehelix",
        "twist",
        "twistextrude",
        "threepointarc",
    }
)
DIRECT_V3_OPS = frozenset({"sweep", "loft", "makehelix", "threepointarc"})
GEAR_FAMILIES = frozenset(
    {
        "bevel_gear",
        "helical_gear",
        "spur_gear",
        "double_simplex_sprocket",
        "sprocket",
        "ratchet_sector",
    }
)
PRIORITY_FAMILIES = (
    "coil_spring",
    "helical_gear",
    "bevel_gear",
    "twisted_drill",
    "impeller",
)
V3_APPROXIMABLE = frozenset(
    {
        "box",
        "cylinder",
        "sphere",
        "union",
        "cut",
        "intersect",
        "fuse",
        "polyline",
        "moveto",
        "lineto",
        "close",
        "extrude",
        "revolve",
        "hole",
        "slot2d",
        "fillet",
        "chamfer",
        "translate",
        "rotate",
        "workplane",
        "center",
        "vector",
        "rect",
        "circle",
        "polygon",
        "cutthruall",
        "cutblind",
        "pushpoints",
        "rarray",
        "transformed",
        "faces",
        "edges",
        "sweep",
        "loft",
        "makehelix",
        "threepointarc",
        "threepointarcto",
    }
)


def classify_code(code: str, family: str = "") -> dict[str, Any]:
    counts = cadquery_operations(code)
    advanced_hits = sorted(name for name in counts if name in ADVANCED_GT_OPS)
    gear = str(family or "") in GEAR_FAMILIES
    stratum = "advanced" if advanced_hits or gear else "standard"
    present = set(counts)
    uncovered = sorted(present - V3_APPROXIMABLE - {"<syntax_error>"})
    direct = [name for name in advanced_hits if name in DIRECT_V3_OPS]
    expressible = (not advanced_hits) or bool(direct) or gear
    if advanced_hits and not direct and not gear:
        expressible = False
    return {
        "ops": dict(counts),
        "advanced_ops": advanced_hits,
        "gear_family": gear,
        "stratum": stratum,
        "v3_expressible": expressible,
        "uncovered_ops": uncovered,
    }


def coverage_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    advanced = [row for row in items if row.get("stratum") == "advanced"]
    n_adv = len(advanced)
    n_ok = sum(1 for row in advanced if row.get("v3_expressible"))
    rate = (n_ok / n_adv) if n_adv else 0.0
    return {
        "n": len(items),
        "n_advanced": n_adv,
        "n_advanced_expressible": n_ok,
        "advanced_expressible_rate": rate,
        "pass_gate": n_adv > 0 and rate >= 0.70,
        "threshold": 0.70,
    }
