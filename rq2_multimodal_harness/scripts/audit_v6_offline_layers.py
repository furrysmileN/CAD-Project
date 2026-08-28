# -*- coding: utf-8 -*-
"""Offline four-layer audit for V6 confirm100 + Oracle feature scorer.

Does not call the live API. Does not modify V6 pilot state.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.common import atomic_write_json
from rq2_harness.prompting import validate_plan
from rq2_harness.v6_corruptions import unchanged_except_critical
from rq2_harness.v6_fact_masks import repeat_contains_critical
from rq2_harness.v6_feature_scorer import DEFAULT_TOLERANCE, score_critical_fact

OUT = ROOT / "outputs" / "v6_information_complementarity"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _meas_ok(measured: Any, gt: Any, category: str) -> bool:
    if measured is None or gt is None:
        return False
    if category in {"through_vs_blind"}:
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


def _same_scalar(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, bool) and isinstance(b, bool):
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


def _residual_carriers(blob: dict[str, Any], *, category: str, fact_id: str, values: list[Any], skip_primary: bool) -> list[str]:
    hits: list[str] = []
    targets = [item for item in values if item is not None]
    for index, item in enumerate(blob.get("cad_facts") or []):
        if not isinstance(item, dict):
            continue
        if skip_primary and item.get("role") == "primary_critical":
            continue
        if fact_id and item.get("fact_id") == fact_id:
            hits.append(f"cad_facts[{index}]:{fact_id}")
            continue
        if item.get("category") == category and any(_same_scalar(item.get("value"), target) for target in targets):
            hits.append(f"cad_facts[{index}]:{item.get('fact_id')}")
    sections = blob.get("sections") or {}
    if isinstance(sections, dict):
        for axis, block in sections.items():
            if not isinstance(block, dict):
                continue
            for hole_i, hole in enumerate(block.get("holes") or []):
                if not isinstance(hole, dict):
                    continue
                if category == "through_vs_blind" and "through" in hole:
                    mapped = "through" if hole.get("through") else "blind"
                    if any(_same_scalar(mapped, target) for target in targets):
                        hits.append(f"sections.{axis}.holes[{hole_i}].through")
                if category in {"depth", "radius_or_width"}:
                    for key in ("depth", "radius", "diameter"):
                        if any(_same_scalar(hole.get(key), target) for target in targets):
                            hits.append(f"sections.{axis}.holes[{hole_i}].{key}")
    for index, item in enumerate(blob.get("hypotheses") or []):
        if fact_id and fact_id in json.dumps(item, ensure_ascii=False):
            hits.append(f"hypotheses[{index}]")
    return sorted(set(hits))


def _mutate_counterfactual_plan(gt_plan: dict[str, Any], latent: dict[str, Any]) -> dict[str, Any]:
    plan = copy.deepcopy(gt_plan)
    critical = latent.get("critical_fact") or {}
    category = str(critical.get("category") or "")
    op_id = critical.get("operation_id")
    gt = critical.get("value")
    ops = plan.get("operations") or []
    matched = next((op for op in ops if isinstance(op, dict) and op.get("id") == op_id), None)
    if category == "depth":
        if matched and "depth" in matched:
            matched["depth"] = round(float(matched["depth"]) + 0.15, 4)
        elif matched and matched.get("op") == "box" and isinstance(matched.get("size"), list) and len(matched["size"]) == 3:
            matched["size"][2] = round(float(matched["size"][2]) + 0.15, 4)
    elif category == "through_vs_blind" and matched and "depth" in matched:
        matched["depth"] = 0.40
    elif category == "hidden_presence":
        plan["operations"] = [op for op in ops if not (isinstance(op, dict) and op.get("id") == "back_hole")]
    elif category == "offset_or_spacing":
        for op in ops:
            if not isinstance(op, dict) or not str(op.get("id") or "").startswith("bolt_"):
                continue
            center = op.get("center")
            if isinstance(center, list) and len(center) >= 2:
                center[0] = round(float(center[0]) * 1.5, 4)
                center[1] = round(float(center[1]) * 1.5, 4)
    elif category == "radius_or_width":
        if matched and "diameter" in matched:
            matched["diameter"] = round(float(matched["diameter"]) + 0.08, 4)
        elif matched and "radius" in matched:
            matched["radius"] = round(float(matched["radius"]) + 0.08, 4)
        else:
            target = None
            try:
                target = float(gt)
            except (TypeError, ValueError):
                target = None
            for op in ops:
                if not isinstance(op, dict) or op.get("op") != "revolve_profile":
                    continue
                for point in op.get("profile") or []:
                    if isinstance(point, list) and target is not None and abs(float(point[0]) - target) <= 1e-4:
                        point[0] = round(float(point[0]) + 0.08, 4)
    elif category == "axis_or_symmetry":
        changed = False
        for op in ops:
            if not isinstance(op, dict):
                continue
            if op.get("op") == "cylinder" and isinstance(op.get("axis"), list):
                op["axis"] = [1.0, 0.0, 0.0]
                changed = True
            if op.get("op") == "revolve_profile":
                op["workplane"] = "YZ"
                changed = True
        if not changed:
            plan["_mutation_note"] = "no axis field to mutate"
    return plan


def main() -> int:
    latent_dir = OUT / "latent_specs"
    inputs = OUT / "inputs"
    samples: list[dict[str, Any]] = []
    by_cat: dict[str, Counter] = defaultdict(Counter)
    by_fam: dict[str, Counter] = defaultdict(Counter)

    for path in sorted(latent_dir.glob("v6_confirm_*.json")):
        latent = _load(path)
        sample_id = latent["sample_id"]
        family = latent["family"]
        critical = latent.get("critical_fact") or {}
        category = str(critical.get("category") or "")
        gt_value = critical.get("value")
        sample_dir = inputs / sample_id
        p_comp_path = sample_dir / "p_comp_v2.json"
        if not p_comp_path.is_file():
            p_comp_path = sample_dir / "p_comp.json"
        p_comp = _load(p_comp_path)
        p_repeat_path = sample_dir / "p_repeat_v2.json"
        if not p_repeat_path.is_file():
            p_repeat_path = sample_dir / "p_repeat.json"
        p_repeat = _load(p_repeat_path)
        p_wrong = _load(sample_dir / "p_wrong.json")
        primary = next(
            (item for item in (p_comp.get("cad_facts") or []) if item.get("role") == "primary_critical"),
            None,
        )
        measured = None if primary is None else primary.get("value")
        meas_ok = _meas_ok(measured, gt_value, category)

        repeat_id = repeat_contains_critical(p_repeat, critical)
        residual_repeat = _residual_carriers(
            p_repeat,
            category=category,
            fact_id=str(critical.get("fact_id") or ""),
            values=[measured, gt_value],
            skip_primary=False,
        )
        corruption = p_wrong.get("corruption") or {}
        wrong_value = corruption.get("new_value")
        if wrong_value is None and isinstance(p_wrong.get("cad_facts"), list):
            wp = next((item for item in p_wrong["cad_facts"] if item.get("role") == "primary_critical"), None)
            wrong_value = None if wp is None else wp.get("value")
        collision = _meas_ok(wrong_value, gt_value, category)
        residual_wrong = _residual_carriers(
            p_wrong,
            category=category,
            fact_id=str(critical.get("fact_id") or ""),
            values=[measured, gt_value],
            skip_primary=True,
        )
        manip_ok = (
            (not repeat_id)
            and (not residual_repeat)
            and (not collision)
            and (not residual_wrong)
            and unchanged_except_critical(p_comp, p_wrong, critical)
        )

        gt_plan = latent.get("gt_plan") or {"operations": latent.get("operations")}
        cf_plan = _mutate_counterfactual_plan(gt_plan, latent)
        gt_issues = validate_plan(gt_plan, plan_version="v2")
        cf_issues = validate_plan(cf_plan, plan_version="v2")
        gt_score = score_critical_fact(gt_plan, latent)
        cf_score = score_critical_fact(cf_plan, latent)
        oracle_changed = bool(gt_score.get("exact")) and not bool(cf_score.get("exact"))

        row = {
            "sample_id": sample_id,
            "family": family,
            "category": category,
            "gt": gt_value,
            "measured": measured,
            "measured_source": None if primary is None else primary.get("source"),
            "meas_ok": meas_ok,
            "repeat_still_has_primary": repeat_id,
            "repeat_residual": residual_repeat,
            "wrong_value": wrong_value,
            "wrong_collision": collision,
            "wrong_residual": residual_wrong,
            "manip_ok": manip_ok,
            "gt_schema_issues": [item.get("code") for item in gt_issues],
            "cf_schema_issues": [item.get("code") for item in cf_issues],
            "gt_exact": gt_score.get("exact"),
            "gt_within": gt_score.get("within_tolerance"),
            "gt_pred": gt_score.get("pred_value"),
            "cf_exact": cf_score.get("exact"),
            "cf_pred": cf_score.get("pred_value"),
            "oracle_pred_changed": oracle_changed,
        }
        samples.append(row)
        by_cat[category]["n"] += 1
        by_cat[category]["meas_ok"] += int(meas_ok)
        by_cat[category]["gt_exact"] += int(bool(gt_score.get("exact")))
        by_cat[category]["oracle_changed"] += int(oracle_changed)
        by_cat[category]["manip_ok"] += int(manip_ok)
        by_fam[family]["n"] += 1
        by_fam[family]["meas_ok"] += int(meas_ok)
        by_fam[family]["gt_exact"] += int(bool(gt_score.get("exact")))
        by_fam[family]["oracle_changed"] += int(oracle_changed)

    n = len(samples)
    n_meas = sum(1 for item in samples if item["meas_ok"])
    n_manip = sum(1 for item in samples if item["manip_ok"])
    n_gt_schema = sum(1 for item in samples if not item["gt_schema_issues"])
    n_cf_schema = sum(1 for item in samples if not item["cf_schema_issues"])
    n_gt_exact = sum(1 for item in samples if item["gt_exact"])
    n_oracle = sum(1 for item in samples if item["oracle_pred_changed"])
    summary = {
        "split": "confirm100",
        "n": n,
        "measurement": {
            "ok": n_meas,
            "rate": round(n_meas / n, 4) if n else 0,
            "threshold": 0.90,
            "pass": (n_meas / n) >= 0.90 if n else False,
            "by_category": {key: dict(val) for key, val in by_cat.items()},
            "by_family": {key: dict(val) for key, val in by_fam.items()},
        },
        "manipulation": {
            "ok": n_manip,
            "rate": round(n_manip / n, 4) if n else 0,
            "threshold": 1.0,
            "pass": n_manip == n,
            "repeat_has_primary": sum(1 for item in samples if item["repeat_still_has_primary"]),
            "repeat_residual": sum(1 for item in samples if item["repeat_residual"]),
            "wrong_collision": sum(1 for item in samples if item["wrong_collision"]),
            "wrong_residual": sum(1 for item in samples if item["wrong_residual"]),
        },
        "plan_schema": {
            "gt_ok": n_gt_schema,
            "cf_ok": n_cf_schema,
            "gt_issue_codes": dict(Counter(code for item in samples for code in item["gt_schema_issues"])),
            "pass_gt": n_gt_schema == n,
        },
        "oracle_scorer": {
            "gt_exact": n_gt_exact,
            "gt_exact_rate": round(n_gt_exact / n, 4) if n else 0,
            "pred_changed": n_oracle,
            "pred_changed_rate": round(n_oracle / n, 4) if n else 0,
            "pass_gt_exact_near_100": (n_gt_exact / n) >= 0.95 if n else False,
            "pass_counterfactual_detected": n_oracle == n,
            "by_category": {key: dict(val) for key, val in by_cat.items()},
        },
        "verdict": (
            "先修 scorer，不要测模型。"
            if (n_gt_exact / n if n else 0) < 0.95 or n_oracle != n
            else "评分层已过；测量层未过（且旧 P_wrong 操纵未过），不要进入 V6b live。"
            if (n_meas / n if n else 0) < 0.90
            else "Oracle 与测量达到进入探针的离线资格。"
        ),
    }
    dest = OUT / "audits" / "v6b_offline_layers_confirm100.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest, {"summary": summary, "samples": samples})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
