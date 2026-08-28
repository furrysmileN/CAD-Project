# -*- coding: utf-8 -*-
"""Remeasure confirm100 P_comp from existing STEP files. Does not call the live API.

Does not overwrite pilot/live. Writes p_comp_v2.json next to confirm inputs
and a measurement audit JSON.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.common import atomic_write_json
from rq2_harness.v6_evidence_builder import attach_primary_critical, build_p_comp
from rq2_harness.v6_fact_masks import build_p_repeat
from rq2_harness.v6_feature_scorer import DEFAULT_TOLERANCE
from rq2_harness.v6_sample_pointcloud import sample_pointcloud

OUT = ROOT / "outputs" / "v6_information_complementarity"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _meas_ok(measured: Any, gt: Any, category: str) -> bool:
    if measured is None or gt is None:
        return False
    if category == "through_vs_blind":
        return str(measured) == str(gt)
    if category == "hidden_presence":
        return bool(measured) == bool(gt)
    if category == "axis_or_symmetry":
        try:
            import numpy as np

            a = np.asarray(measured, dtype=float)
            b = np.asarray(gt, dtype=float)
            a = a / max(float(np.linalg.norm(a)), 1e-9)
            b = b / max(float(np.linalg.norm(b)), 1e-9)
            return abs(float(a @ b)) >= 0.85
        except Exception:
            return False
    try:
        return abs(float(measured) - float(gt)) <= float(DEFAULT_TOLERANCE.get(category, 0.04))
    except (TypeError, ValueError):
        return False


def main() -> int:
    latent_dir = OUT / "latent_specs"
    n_points = 4096
    rows: list[dict[str, Any]] = []
    by_cat: dict[str, Counter] = defaultdict(Counter)
    by_fam: dict[str, Counter] = defaultdict(Counter)
    cache = ROOT.parent.parent / "_tmp_pc"
    cache.mkdir(parents=True, exist_ok=True)

    for path in sorted(latent_dir.glob("v6_confirm_*.json")):
        latent = _load(path)
        sample_id = latent["sample_id"]
        family = latent["family"]
        critical = latent.get("critical_fact") or {}
        category = str(critical.get("category") or "")
        gt = critical.get("value")
        step = OUT / "targets" / sample_id / "target.step"
        npy = cache / f"{sample_id}.npy"
        if not npy.is_file():
            sample_pointcloud(step, npy, n_points=n_points, seed=42)
        p_comp = attach_primary_critical(build_p_comp(npy), critical)
        p_repeat = build_p_repeat(p_comp, critical)
        dest = OUT / "inputs" / sample_id / "p_comp_v2.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(dest, p_comp)
        atomic_write_json(OUT / "inputs" / sample_id / "p_repeat_v2.json", p_repeat)
        primary = next((item for item in p_comp.get("cad_facts") or [] if item.get("role") == "primary_critical"), None)
        measured = None if primary is None else primary.get("value")
        ok = _meas_ok(measured, gt, category)
        row = {
            "sample_id": sample_id,
            "family": family,
            "category": category,
            "gt": gt,
            "measured": measured,
            "source": None if primary is None else primary.get("source"),
            "ok": ok,
        }
        rows.append(row)
        by_cat[category]["n"] += 1
        by_cat[category]["ok"] += int(ok)
        by_fam[family]["n"] += 1
        by_fam[family]["ok"] += int(ok)
        print(f"{sample_id} {family} {category} ok={ok} gt={gt} meas={measured} src={row['source']}")

    n = len(rows)
    n_ok = sum(1 for item in rows if item["ok"])
    summary = {
        "n": n,
        "ok": n_ok,
        "rate": round(n_ok / n, 4) if n else 0,
        "threshold": 0.90,
        "pass": (n_ok / n) >= 0.90 if n else False,
        "n_points": n_points,
        "by_category": {key: dict(val) for key, val in by_cat.items()},
        "by_family": {key: dict(val) for key, val in by_fam.items()},
        "unresolved": sum(1 for item in rows if item["measured"] is None),
        "bbox_fallback": sum(1 for item in rows if item["source"] == "point_cloud_bbox"),
    }
    out = OUT / "audits" / "v6b_measurement_remeasure_confirm100.json"
    atomic_write_json(out, {"summary": summary, "samples": rows})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
