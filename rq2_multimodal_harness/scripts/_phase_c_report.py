"""Phase C live 报告：官方 C1–C8 + 剔除欠费失败的敏感性分析。"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import load_config, project_path
from rq2_harness.pc_analysis import condition_summary, load_task_rows
from rq2_harness.pc_conditions import V5_CONFIRM_IDS, V5_CONTROL_IDS
from rq2_harness.v5_analysis import C_CONTRASTS, aggregate_repeats, analyze_v5_confirm, _pairs
from rq2_harness.v5_stats import apply_holm, paired_continuous, paired_success

BILLING_STATUS = "task_failed"
CONDITION_IDS = V5_CONFIRM_IDS + V5_CONTROL_IDS


def _contrasts(aggregated):
    contrasts = []
    for cid, left, right in C_CONTRASTS:
        deltas, left_ok, right_ok = _pairs(aggregated, left, right)
        if not deltas:
            continue
        stats = paired_continuous(deltas)
        success = paired_success(left_ok, right_ok)
        contrasts.append(
            {
                "id": cid,
                "contrast": f"{left}-{right}",
                "left": left,
                "right": right,
                **stats,
                "mcnemar_p": success.get("p_value"),
            }
        )
    by = {(row["sample_id"], row["condition"]): row for row in aggregated}
    samples = sorted({row["sample_id"] for row in aggregated})
    c6_deltas = []
    for sample_id in samples:
        t1p = by.get((sample_id, "T1I1P_geom"))
        t1 = by.get((sample_id, "T1I1"))
        t2p = by.get((sample_id, "T2I1P_geom"))
        t2 = by.get((sample_id, "T2I1"))
        if None in (t1p, t1, t2p, t2):
            continue
        weak = float(t1p["joint_quality"]) - float(t1["joint_quality"])
        strong = float(t2p["joint_quality"]) - float(t2["joint_quality"])
        c6_deltas.append(weak - strong)
    if c6_deltas:
        contrasts.append(
            {
                "id": "C6",
                "contrast": "text_strength_moderation",
                "left": "weak_gain",
                "right": "strong_gain",
                **paired_continuous(c6_deltas),
            }
        )
    return apply_holm([row for row in contrasts if row.get("wilcoxon_p") is not None])


def _round_row(row):
    out = {}
    for key, value in row.items():
        if isinstance(value, float):
            out[key] = round(value, 6)
        else:
            out[key] = value
    return out


def complementarity(aggregated):
    by = {(row["sample_id"], row["condition"]): row for row in aggregated}
    samples = sorted({row["sample_id"] for row in aggregated})
    both = 0
    only_image = 0
    only_pc = 0
    neither = 0
    n = 0
    for sample_id in samples:
        i1p = by.get((sample_id, "I1P_geom"))
        i1 = by.get((sample_id, "I1"))
        p = by.get((sample_id, "P_geom"))
        if None in (i1p, i1, p):
            continue
        n += 1
        jq_i1p = float(i1p["joint_quality"])
        over_i1 = jq_i1p > float(i1["joint_quality"])
        over_p = jq_i1p > float(p["joint_quality"])
        if over_i1 and over_p:
            both += 1
        elif over_i1:
            only_image += 1
        elif over_p:
            only_pc += 1
        else:
            neither += 1
    return {
        "n_samples": n,
        "bidirectional": both,
        "unidirectional_over_image_only": only_image,
        "unidirectional_over_pc_only": only_pc,
        "neither": neither,
        "bidirectional_rate": (both / n) if n else None,
    }


def cd_summary(rows, condition_ids):
    by = defaultdict(list)
    for row in rows:
        if row.get("common_frame_cd") is None:
            continue
        if row.get("status") != "completed":
            continue
        by[row["condition"]].append(float(row["common_frame_cd"]))
    out = []
    for condition in condition_ids:
        values = by.get(condition) or []
        out.append(
            {
                "condition": condition,
                "n_with_cd": len(values),
                "mean_common_frame_cd": mean(values) if values else None,
                "median_common_frame_cd": median(values) if values else None,
            }
        )
    return out


def main() -> int:
    config = load_config(
        str(Path(__file__).resolve().parents[1] / "configs" / "v5_phase_c_confirm.yaml")
    )
    output_dir = project_path(config["paths"]["output_root"])
    manifest = project_path(config["paths"]["manifest"])
    official_meta = analyze_v5_confirm(output_dir, manifest, condition_ids=CONDITION_IDS)
    rows = load_task_rows(output_dir, manifest, condition_ids=CONDITION_IDS)
    status = Counter(row["status"] for row in rows)
    billing = [row for row in rows if row["status"] == BILLING_STATUS]
    usable = [row for row in rows if row["status"] != BILLING_STATUS]
    official_agg = aggregate_repeats(rows)
    usable_agg = aggregate_repeats(usable)
    official_contrasts = _contrasts(official_agg)
    usable_contrasts = _contrasts(usable_agg)
    payload = {
        "schema_version": "rq2.v5.phase_c.live_report.v1",
        "run_counts": dict(status),
        "n_rows": len(rows),
        "n_billing_failed": len(billing),
        "billing_by_condition": dict(Counter(row["condition"] for row in billing)),
        "official": {
            "n_rows": official_meta["n_rows"],
            "n_aggregated": official_meta["n_aggregated"],
            "condition_summary": [_round_row(item) for item in condition_summary(rows, CONDITION_IDS)],
            "contrasts_holm": [_round_row(item) for item in official_contrasts],
            "complementarity": complementarity(official_agg),
            "common_frame_cd": [_round_row(item) for item in cd_summary(rows, CONDITION_IDS)],
        },
        "exclude_billing_failures": {
            "n_rows": len(usable),
            "n_aggregated": len(usable_agg),
            "condition_summary": [_round_row(item) for item in condition_summary(usable, CONDITION_IDS)],
            "contrasts_holm": [_round_row(item) for item in usable_contrasts],
            "complementarity": complementarity(usable_agg),
            "common_frame_cd": [_round_row(item) for item in cd_summary(usable, CONDITION_IDS)],
        },
        "notes": [
            "task_failed 全部为 DashScope 欠费 400，不是模型失败。",
            "官方分析把失败记 joint_quality=0；敏感性分析剔除 task_failed。",
            "双向互补：样本级 I1P_geom > I1 且 I1P_geom > P_geom。",
        ],
    }
    out_path = output_dir / "analysis" / "v5_phase_c_live_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "status": dict(status)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
