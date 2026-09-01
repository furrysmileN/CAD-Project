"""V8 Cut 2 描述性分析：新臂 vs V5 锚点。n=20，不做 Holm。"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .common import atomic_write_json, project_path, read_jsonl
from .pc_analysis import _geometry_values, _write_csv, load_task_rows
from .pc_conditions import V8_I1_ABLATION_IDS
from .v5_stats import paired_continuous
from .v8_autopsy import load_primary_metrics

ANCHORS = ("I1", "I1P_geom", "I1P_proj", "I1P_shuffle")
NEW_IDS = tuple(V8_I1_ABLATION_IDS)


def _mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return mean(clean) if clean else None


def load_v8_rows(output_dir: Path, manifest_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = load_task_rows(output_dir, manifest_path, condition_ids=NEW_IDS)
    by: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        by[(row["sample_id"], row["condition"])] = row
    return by


def load_v5_anchors(metrics_path: Path, sample_ids: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    by = {}
    for row in load_primary_metrics(metrics_path):
        if row["sample_id"] in sample_ids and row["condition"] in ANCHORS:
            by[(row["sample_id"], row["condition"])] = row
    return by


def _deltas(by: dict[tuple[str, str], dict[str, Any]], samples: list[str], left: str, right: str) -> list[float]:
    out = []
    for sample_id in samples:
        a = by.get((sample_id, left))
        b = by.get((sample_id, right))
        if a is None or b is None:
            continue
        out.append(float(a.get("joint_quality") or 0.0) - float(b.get("joint_quality") or 0.0))
    return out


def analyze_cut2(
    *,
    v8_output: Path,
    manifest_path: Path,
    v5_metrics: Path,
    dest: Path,
) -> dict[str, Any]:
    sample_ids = [row["sample_id"] for row in read_jsonl(manifest_path)]
    wanted = set(sample_ids)
    merged = dict(load_v5_anchors(v5_metrics, wanted))
    merged.update(load_v8_rows(v8_output, manifest_path))
    # I1P_full alias
    for sample_id in sample_ids:
        full = merged.get((sample_id, "I1P_geom"))
        if full is not None:
            merged[(sample_id, "I1P_full")] = full

    contrasts = [
        ("H-B0", "I1P_full", "I1P_shuffle"),
        ("H-B1", "I1P_full", "I1P_bbox"),
        ("H-B2a", "I1P_full", "I1P_sym"),
        ("H-B2b", "I1P_sym", "I1P_axes"),
        ("H-B3", "I1P_full", "I1P_sym"),
        ("bbox-I1", "I1P_bbox", "I1"),
        ("axes-bbox", "I1P_axes", "I1P_bbox"),
        ("sym-axes", "I1P_sym", "I1P_axes"),
        ("full-sym", "I1P_full", "I1P_sym"),
    ]
    contrast_rows = []
    stats_by: dict[str, dict[str, Any]] = {}
    for cid, left, right in contrasts:
        deltas = _deltas(merged, sample_ids, left, right)
        stats = paired_continuous(deltas)
        row = {"id": cid, "left": left, "right": right, "contrast": f"{left}-{right}", **stats}
        contrast_rows.append(row)
        stats_by[cid] = row

    full_sym = float(stats_by["H-B3"].get("mean_delta") or 0.0)
    abs_full_sym = abs(full_sym)
    sym_axes = float(stats_by["H-B2b"].get("mean_delta") or 0.0)
    h_b2 = abs_full_sym < 0.02 and sym_axes > 0.03
    h_b3 = full_sym >= 0.03
    h_b1 = float(stats_by["H-B1"].get("mean_delta") or 0.0) > 0
    h_b0 = float(stats_by["H-B0"].get("mean_delta") or 0.0) > 0

    means = {}
    for condition in list(ANCHORS) + list(NEW_IDS) + ["I1P_full"]:
        values = [float(merged[sid, condition]["joint_quality"]) for sid in sample_ids if (sid, condition) in merged]
        means[condition] = mean(values) if values else None

    verdict = {
        "H-B0": h_b0,
        "H-B1": h_b1,
        "H-B2": h_b2,
        "H-B3": h_b3,
        "next": "cut3" if h_b3 else ("stop_shuffle" if not h_b0 else ("cut4_or_stop" if not h_b1 else "cut4_or_describe")),
    }
    if h_b3:
        verdict["next"] = "cut3_instrument"
    elif not h_b0:
        verdict["next"] = "stop_boilerplate_prior"
    elif not h_b1:
        verdict["next"] = "skip_cut3_bbox_explains"
    elif h_b2:
        verdict["next"] = "skip_cut3_symmetry_bundle"
    else:
        verdict["next"] = "no_expand_ambiguous"

    dest.mkdir(parents=True, exist_ok=True)
    per_sample = []
    for sample_id in sample_ids:
        item = {"sample_id": sample_id}
        for condition in ("I1", "I1P_bbox", "I1P_axes", "I1P_sym", "I1P_full", "I1P_shuffle", "I1P_proj"):
            row = merged.get((sample_id, condition))
            item[f"jq_{condition}"] = None if row is None else float(row.get("joint_quality") or 0.0)
        per_sample.append(item)
    _write_csv(dest / "cut2_per_sample.csv", per_sample)
    _write_csv(dest / "cut2_contrasts.csv", contrast_rows)
    payload = {
        "schema_version": "rq2.v8.cut2.analysis.v1",
        "n": len(sample_ids),
        "means": means,
        "contrasts": contrast_rows,
        "hypotheses": verdict,
        "note": "n=20 描述性，非正式显著。I1P_full 复用 V5 I1P_geom 三次重复均值。",
    }
    atomic_write_json(dest / "cut2_descriptive.json", payload)
    lines = [
        "# V8 Cut 2 描述性结果（n=20）",
        "",
        "主指标 `joint_quality`。`I1P_full` = V5 `I1P_geom` 锚点。不做 Holm。",
        "",
        "## 均值",
        "",
        "| 条件 | 均值 jq |",
        "|---|---:|",
    ]
    for condition, value in means.items():
        if value is None:
            continue
        lines.append(f"| `{condition}` | {value:.3f} |")
    lines += [
        "",
        "## 假设",
        "",
        f"- H-B0 full > shuffle：{'成立' if h_b0 else '不成立'}（Δ={stats_by['H-B0'].get('mean_delta'):.3f}）",
        f"- H-B1 full > bbox：{'成立' if h_b1 else '不成立'}（Δ={stats_by['H-B1'].get('mean_delta'):.3f}，CI 是否跨 0 见 JSON）",
        f"- H-B2 对称束解释剩余：{'成立' if h_b2 else '不成立'}（|full-sym|={abs_full_sym:.3f}，sym-axes={sym_axes:.3f}）",
        f"- H-B3 截面/细特征跳变：{'成立' if h_b3 else '不成立'}（full-sym Δ={full_sym:.3f}）",
        "",
        f"**下一刀：** `{verdict['next']}`",
        "",
        "同场嵌套（本次 live，可直接比）：`I1P_bbox` → `I1P_axes` → `I1P_sym`。",
        "`I1` / `I1P_full` / shuffle 是 V5 锚点，跨会话，只能作方向对照。",
        "H-B3 未过，**不进入 Cut 3**，不扩 100。",
        "",
    ]
    (dest / "CUT2_ANALYSIS_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
