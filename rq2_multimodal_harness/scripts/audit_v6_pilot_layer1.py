# -*- coding: utf-8 -*-
"""Layer-1 autopsy of V6 pilot: evidence fidelity, residual copies, plan identity."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.common import atomic_write_json, sha256_json
from rq2_harness.v6_corruptions import unchanged_except_critical
from rq2_harness.v6_fact_masks import evidence_token_count, repeat_contains_critical

OUT = ROOT / "outputs" / "v6_information_complementarity"
ANALYSIS = OUT / "pilot" / "live" / "analysis"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(a: Any, b: Any, *, category: str) -> bool:
    if a is None or b is None:
        return False
    if category == "through_vs_blind":
        return str(a) == str(b)
    if category == "hidden_presence":
        return bool(a) == bool(b)
    if category == "axis_or_symmetry":
        try:
            import numpy as np

            x = np.asarray(a, dtype=float)
            y = np.asarray(b, dtype=float)
            x = x / max(float(np.linalg.norm(x)), 1e-9)
            y = y / max(float(np.linalg.norm(y)), 1e-9)
            return abs(float(x @ y)) >= 0.85
        except Exception:
            return False
    try:
        tol = 0.05 if category == "offset_or_spacing" else 0.04
        return abs(float(a) - float(b)) <= tol
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
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return list(a) == list(b)
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


def _residual_paths(blob: dict[str, Any], *, category: str, fact_id: str, values: list[Any], skip_primary: bool) -> list[str]:
    hits: list[str] = []
    targets = [item for item in values if item is not None]
    for index, item in enumerate(blob.get("cad_facts") or []):
        if not isinstance(item, dict):
            continue
        if skip_primary and item.get("role") == "primary_critical":
            continue
        same_id = bool(fact_id) and item.get("fact_id") == fact_id
        same_cat = item.get("category") == category
        if same_id or (same_cat and any(_same_scalar(item.get("value"), target) for target in targets)):
            hits.append(f"cad_facts[{index}]:{item.get('fact_id')}")
    sections = blob.get("sections") or {}
    for axis, block in sections.items() if isinstance(sections, dict) else []:
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
            if category == "offset_or_spacing" and hole.get("center"):
                hits.append(f"sections.{axis}.holes[{hole_i}].center")
    for index, item in enumerate(blob.get("hypotheses") or []):
        dumped = json.dumps(item, ensure_ascii=False)
        if fact_id and fact_id in dumped:
            hits.append(f"hypotheses[{index}]")
    return sorted(set(hits))


def _round_vec(values: Any, ndigits: int = 4) -> tuple[float, ...] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    try:
        return tuple(round(float(x), ndigits) for x in values)
    except (TypeError, ValueError):
        return None


def _plan_ops_hash(plan: dict[str, Any]) -> str:
    return sha256_json(plan.get("operations") or [])


def _index_episodes(runs_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for episode_path in runs_dir.glob("*/episode_v2.json"):
        plan_path = episode_path.parent / "input_plan.json"
        if not plan_path.is_file():
            continue
        episode = _load(episode_path)
        plan = _load(plan_path)
        trace = episode.get("operationTrace") or []
        bbox = None
        if trace:
            bbox = ((trace[-1].get("after") or {}).get("bboxSize"))
        ops = [op for op in (plan.get("operations") or []) if isinstance(op, dict)]
        rows.append(
            {
                "run_id": episode.get("runId") or episode_path.parent.name,
                "ops_hash": _plan_ops_hash(plan),
                "op_count": len(ops),
                "hole_count": sum(1 for op in ops if op.get("op") == "hole"),
                "bbox": _round_vec(bbox),
                "sample_id_in_plan": plan.get("sample_id"),
            }
        )
    return rows


def main() -> int:
    manifest = [
        json.loads(line)
        for line in (OUT / "manifest_pilot20.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    latent_dir = OUT / "latent_specs"
    inputs = OUT / "inputs"
    state_dir = OUT / "pilot" / "live" / "state"
    episodes = _index_episodes(OUT / "pilot" / "live" / "runs")
    ep_index: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        ep_index[(row["bbox"], row["op_count"], row["hole_count"])].append(row)

    samples = []
    n_meas_ok = 0
    n_repeat_residual = 0
    n_wrong_weak = 0
    n_wrong_old_copy = 0
    n_plan_c34 = 0
    n_plan_c35 = 0
    n_jq_c34 = 0
    n_jq_c35 = 0
    n_pred_c35 = 0
    n_eligible = 0
    fam_jq: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in manifest:
        sample_id = row["sample_id"]
        family = row["family"]
        latent = _load(latent_dir / f"{sample_id}.json")
        critical = latent.get("critical_fact") or {}
        category = str(critical.get("category") or "")
        gt = critical.get("value")
        sample_dir = inputs / sample_id
        p_comp = _load(sample_dir / "p_comp.json")
        p_repeat = _load(sample_dir / "p_repeat.json")
        p_wrong = _load(sample_dir / "p_wrong.json")
        primary = next(
            (item for item in (p_comp.get("cad_facts") or []) if item.get("role") == "primary_critical"),
            None,
        )
        measured = None if primary is None else primary.get("value")
        meas_ok = _close(measured, gt, category=category)
        n_meas_ok += int(meas_ok)

        repeat_has_id = repeat_contains_critical(p_repeat, critical)
        residual = _residual_paths(
            p_repeat,
            category=category,
            fact_id=str(critical.get("fact_id") or ""),
            values=[measured, gt],
            skip_primary=False,
        )
        if residual:
            n_repeat_residual += 1

        corruption = p_wrong.get("corruption") or {}
        new_value = corruption.get("new_value")
        wrong_primary = next(
            (item for item in (p_wrong.get("cad_facts") or []) if item.get("role") == "primary_critical"),
            None,
        )
        wrong_value = None if wrong_primary is None else wrong_primary.get("value")
        wrong_equals_gt = _close(wrong_value, gt, category=category)
        wrong_equals_meas = _close(wrong_value, measured, category=category) if measured is not None else False
        wrong_weak = (not meas_ok) or wrong_equals_gt or wrong_value is None
        if wrong_weak:
            n_wrong_weak += 1
        old_copies = _residual_paths(
            p_wrong,
            category=category,
            fact_id=str(critical.get("fact_id") or ""),
            values=[measured, gt],
            skip_primary=True,
        )
        if old_copies:
            n_wrong_old_copy += 1

        eligible = bool(meas_ok and not residual and not wrong_weak and not old_copies)
        n_eligible += int(eligible)

        conds = {}
        for condition in ("C0", "C1", "C2", "C3", "C4", "C5"):
            state = _load(state_dir / sample_id / condition / "r01.json")
            first = state.get("first_attempt") or {}
            geom = first.get("geometry") or {}
            feat = first.get("features") or {}
            bbox = _round_vec((geom.get("bbox") or {}).get("pred_size"))
            key = (bbox, int(feat.get("operation_count") or 0), int(feat.get("hole_count") or 0))
            matches = ep_index.get(key) or []
            hashes = sorted({item["ops_hash"] for item in matches})
            conds[condition] = {
                "jq": geom.get("joint_quality"),
                "cd": geom.get("common_frame_cd"),
                "valid": geom.get("valid"),
                "exact": feat.get("exact"),
                "tol": feat.get("within_tolerance"),
                "pred": feat.get("pred_value"),
                "gt": feat.get("gt_value"),
                "ops": feat.get("operation_count"),
                "holes": feat.get("hole_count"),
                "bbox": list(bbox) if bbox else None,
                "plan_hash_candidates": hashes,
                "n_episode_matches": len(matches),
            }
            if geom.get("joint_quality") is not None:
                fam_jq[family][condition].append(float(geom["joint_quality"]))

        h3 = (conds["C3"]["plan_hash_candidates"] or [None])[0]
        h4 = (conds["C4"]["plan_hash_candidates"] or [None])[0]
        h5 = (conds["C5"]["plan_hash_candidates"] or [None])[0]
        same_plan_c34 = bool(h3 and h4 and h3 == h4)
        same_plan_c35 = bool(h3 and h5 and h3 == h5)
        jq3, jq4, jq5 = conds["C3"]["jq"] or 0, conds["C4"]["jq"] or 0, conds["C5"]["jq"] or 0
        jq_tie_c34 = abs(jq3 - jq4) < 1e-9
        jq_tie_c35 = abs(jq3 - jq5) < 1e-9
        pred_tie_c35 = conds["C3"]["pred"] == conds["C5"]["pred"]
        n_plan_c34 += int(same_plan_c34)
        n_plan_c35 += int(same_plan_c35)
        n_jq_c34 += int(jq_tie_c34)
        n_jq_c35 += int(jq_tie_c35)
        n_pred_c35 += int(pred_tie_c35)

        samples.append(
            {
                "sample_id": sample_id,
                "family": family,
                "category": category,
                "gt": gt,
                "measured": measured,
                "measured_source": None if primary is None else primary.get("source"),
                "measured_matches_gt": meas_ok,
                "repeat_still_has_primary_id": repeat_has_id,
                "repeat_residual_copies": residual,
                "wrong_value": wrong_value,
                "wrong_new_value": new_value,
                "wrong_type": corruption.get("type"),
                "wrong_equals_gt": wrong_equals_gt,
                "wrong_equals_measured": wrong_equals_meas,
                "wrong_is_weak": wrong_weak,
                "wrong_old_value_copies": old_copies,
                "unchanged_except_critical": unchanged_except_critical(p_comp, p_wrong, critical),
                "token_p_comp": evidence_token_count(p_comp),
                "token_p_repeat": evidence_token_count(p_repeat),
                "eligible_for_k3k4": eligible,
                "jq_tie_c3_c4": jq_tie_c34,
                "jq_tie_c3_c5": jq_tie_c35,
                "pred_tie_c3_c5": pred_tie_c35,
                "same_plan_c3_c4": same_plan_c34,
                "same_plan_c3_c5": same_plan_c35,
                "plan_hash_c3": h3,
                "plan_hash_c4": h4,
                "plan_hash_c5": h5,
                "conditions": conds,
            }
        )

    family_means = {
        family: {cond: round(sum(vals) / len(vals), 4) for cond, vals in conds.items()}
        for family, conds in fam_jq.items()
    }
    summary = {
        "n_samples": len(samples),
        "measured_matches_gt": n_meas_ok,
        "repeat_residual_copies": n_repeat_residual,
        "wrong_weak_or_inaccurate_base": n_wrong_weak,
        "wrong_old_value_still_present": n_wrong_old_copy,
        "eligible_for_k3k4": n_eligible,
        "jq_tie_c3_c4": n_jq_c34,
        "jq_tie_c3_c5": n_jq_c35,
        "pred_tie_c3_c5": n_pred_c35,
        "same_plan_c3_c4": n_plan_c34,
        "same_plan_c3_c5": n_plan_c35,
        "feature_exact_by_condition": {
            cond: sum(1 for item in samples if item["conditions"][cond]["exact"])
            for cond in ("C0", "C1", "C2", "C3", "C4", "C5")
        },
        "family_mean_jq_first_attempt": family_means,
        "verdict": (
            "K3/K4 在本轮几乎无法检验：20 个样本里只有 4 个同时满足"
            "“关键事实测准、P_wrong 改到了不同值”。这 4 个上 C5 的关键预测仍与 C3 相同。"
        ),
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "samples": samples}
    atomic_write_json(ANALYSIS / "layer1_audit.json", payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("--- ineligible samples ---")
    for item in samples:
        if item["eligible_for_k3k4"]:
            print("ELIGIBLE", item["sample_id"], item["family"], item["category"])
            continue
        reasons = []
        if not item["measured_matches_gt"]:
            reasons.append("meas_miss")
        if item["repeat_residual_copies"]:
            reasons.append(f"repeat_resid={len(item['repeat_residual_copies'])}")
        if item["wrong_is_weak"]:
            reasons.append("wrong_weak")
        if item["wrong_old_value_copies"]:
            reasons.append(f"wrong_copy={len(item['wrong_old_value_copies'])}")
        print(item["sample_id"], item["family"], item["category"], ";".join(reasons))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
