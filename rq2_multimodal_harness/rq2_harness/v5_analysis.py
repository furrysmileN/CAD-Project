"""V5 Phase C 预注册分析：C1–C8 + Holm，重复先按 sample×condition 取均值。"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .pc_analysis import _write_csv, load_task_rows
from .v5_stats import apply_holm, paired_continuous, paired_success

C_CONTRASTS = (
    ("C1", "P_geom", "P_proj"),
    ("C2", "I1P_geom", "I1"),
    ("C3", "I1P_geom", "P_geom"),
    ("C4", "T1I1P_geom", "T1I1"),
    ("C5", "T2I1P_geom", "T2I1"),
    ("C7", "I1P_geom", "I1P_proj"),
    ("C8", "I1P_geom", "I1P_shuffle"),
)


def aggregate_repeats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["sample_id"], row["condition"])].append(row)
    aggregated = []
    for (sample_id, condition), items in groups.items():
        jq = mean(float(item.get("joint_quality") or 0.0) for item in items)
        ok = mean(1.0 if item.get("completed") else 0.0 for item in items)
        cd = [item.get("common_frame_cd") for item in items if item.get("common_frame_cd") is not None]
        aggregated.append(
            {
                **items[0],
                "joint_quality": jq,
                "completed": ok >= 0.5,
                "n_repeats": len(items),
                "success_rate_repeats": ok,
                "common_frame_cd": mean(float(v) for v in cd) if cd else None,
            }
        )
    return aggregated


def _pairs(rows: list[dict[str, Any]], left: str, right: str) -> tuple[list[float], list[bool], list[bool]]:
    by = {(row["sample_id"], row["condition"]): row for row in rows}
    samples = {row["sample_id"] for row in rows}
    deltas: list[float] = []
    left_ok: list[bool] = []
    right_ok: list[bool] = []
    for sample_id in samples:
        a = by.get((sample_id, left))
        b = by.get((sample_id, right))
        if a is None or b is None:
            continue
        deltas.append(float(a["joint_quality"]) - float(b["joint_quality"]))
        left_ok.append(bool(a["completed"]))
        right_ok.append(bool(b["completed"]))
    return deltas, left_ok, right_ok


def analyze_v5_confirm(output_dir: Path, manifest_path: Path, condition_ids: tuple[str, ...]) -> dict[str, Any]:
    rows = load_task_rows(output_dir, manifest_path, condition_ids=condition_ids)
    aggregated = aggregate_repeats(rows)
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
    c6_deltas: list[float] = []
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
    holm_rows = [row for row in contrasts if row.get("wilcoxon_p") is not None]
    holm = apply_holm(holm_rows)
    analysis_dir = output_dir / "analysis"
    _write_csv(analysis_dir / "primary_metrics.csv", aggregated)
    _write_csv(analysis_dir / "contrasts_raw.csv", contrasts)
    _write_csv(analysis_dir / "contrasts_holm.csv", holm)
    return {"n_rows": len(rows), "n_aggregated": len(aggregated), "n_contrasts": len(contrasts)}
