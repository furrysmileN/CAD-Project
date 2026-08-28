"""Phase A：从 V4 确认 state 排除筛选 20，在 held-out 80 上重算预注册对比。零 API。"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .common import atomic_write_json, read_jsonl
from .v5_stats import apply_holm, paired_continuous, paired_success

HELDOUT_CONTRASTS = (
    ("P_geom_tool", "P_proj", "P_geom - P_proj"),
    ("I1P_geom", "I1", "I1P_geom - I1"),
    ("I1P_geom", "P_geom_tool", "I1P_geom - P_geom"),
    ("T1I1P_geom", "T1I1", "T1I1P_geom - T1I1"),
    ("T1I1P_geom", "I1P_geom", "T1I1P_geom - I1P_geom"),
)

CORE_FOR_HOLM = HELDOUT_CONTRASTS


def load_screen_ids(selection_summary: Path) -> set[str]:
    payload = json.loads(selection_summary.read_text(encoding="utf-8"))
    ids = payload.get("sample_ids") or []
    return {str(item) for item in ids}


def _geom(state: dict[str, Any]) -> dict[str, Any]:
    geometry = state.get("geometry") or {}
    completed = state.get("status") == "completed"
    voxel = geometry.get("voxel_iou") or {}
    fscore_shape = geometry.get("fscore_shape") or {}
    fscore_common = geometry.get("fscore_common") or {}
    return {
        "completed": completed,
        "joint_quality": float(geometry.get("joint_quality") or 0.0) if completed else 0.0,
        "shape_only_cd": geometry.get("shape_only_cd"),
        "common_frame_cd": geometry.get("common_frame_cd"),
        "voxel_iou": voxel.get("value") if isinstance(voxel, dict) else voxel,
        "f1_shape": fscore_shape.get("f1") if isinstance(fscore_shape, dict) else fscore_shape,
        "f1_common": fscore_common.get("f1") if isinstance(fscore_common, dict) else fscore_common,
    }


def load_confirm_rows(
    state_dir: Path,
    manifest_path: Path,
    *,
    exclude_ids: set[str],
) -> list[dict[str, Any]]:
    manifest = {row["sample_id"]: row for row in read_jsonl(manifest_path)}
    rows: list[dict[str, Any]] = []
    for path in sorted(state_dir.glob("*/*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(state.get("sample_id") or path.parent.name)
        if sample_id in exclude_ids:
            continue
        if sample_id not in manifest:
            continue
        if state.get("status") in {"dry_run", "running"}:
            continue
        sample = manifest[sample_id]
        geom = _geom(state)
        rows.append(
            {
                "sample_id": sample_id,
                "condition": state.get("condition_id") or state.get("condition") or path.stem,
                "status": state.get("status"),
                "family": sample.get("family"),
                "difficulty": sample.get("difficulty"),
                "complexity_bin": sample.get("complexity_bin"),
                **geom,
            }
        )
    return rows


def _by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["sample_id"]][row["condition"]] = row
    return grouped


def condition_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    result = []
    for condition, items in sorted(by_cond.items()):
        n = len(items)
        ok = [item for item in items if item["completed"]]
        jq = [float(item["joint_quality"]) for item in items]
        jq_ok = [float(item["joint_quality"]) for item in ok]
        result.append(
            {
                "condition": condition,
                "n": n,
                "n_valid": len(ok),
                "valid_ratio": (len(ok) / n) if n else None,
                "mean_joint_quality": mean(jq) if jq else None,
                "mean_joint_quality_valid": mean(jq_ok) if jq_ok else None,
                "p_valid_times_eq_valid": (len(ok) / n) * mean(jq_ok) if n and jq_ok else 0.0,
                "mean_common_frame_cd_valid": mean(
                    [float(item["common_frame_cd"]) for item in ok if item.get("common_frame_cd") is not None]
                )
                if any(item.get("common_frame_cd") is not None for item in ok)
                else None,
                "mean_f1_common_valid": mean(
                    [float(item["f1_common"]) for item in ok if item.get("f1_common") is not None]
                )
                if any(item.get("f1_common") is not None for item in ok)
                else None,
            }
        )
    return result


def _numeric_deltas(
    grouped: dict[str, dict[str, dict[str, Any]]],
    left: str,
    right: str,
    field: str,
    *,
    valid_only: bool,
) -> list[float]:
    deltas: list[float] = []
    for sample in grouped.values():
        a = sample.get(left)
        b = sample.get(right)
        if a is None or b is None:
            continue
        if valid_only and not (a["completed"] and b["completed"]):
            continue
        if valid_only:
            av = a.get(field)
            bv = b.get(field)
            if av is None or bv is None:
                continue
            deltas.append(float(av) - float(bv))
        else:
            if field == "joint_quality":
                deltas.append(float(a["joint_quality"]) - float(b["joint_quality"]))
            else:
                av = a.get(field) if a["completed"] else None
                bv = b.get(field) if b["completed"] else None
                if av is None or bv is None:
                    continue
                deltas.append(float(av) - float(bv))
    return deltas


def contrast_table(rows: list[dict[str, Any]], *, valid_only: bool) -> list[dict[str, Any]]:
    grouped = _by_sample(rows)
    result = []
    for left, right, label in HELDOUT_CONTRASTS:
        jq_deltas = _numeric_deltas(grouped, left, right, "joint_quality", valid_only=valid_only)
        stats = paired_continuous(jq_deltas)
        left_ok: list[bool] = []
        right_ok: list[bool] = []
        for sample in grouped.values():
            a = sample.get(left)
            b = sample.get(right)
            if a is None or b is None:
                continue
            if valid_only and not (a["completed"] and b["completed"]):
                continue
            left_ok.append(bool(a["completed"]))
            right_ok.append(bool(b["completed"]))
        success = paired_success(left_ok, right_ok) if left_ok else {}
        cd = _numeric_deltas(grouped, left, right, "common_frame_cd", valid_only=True)
        iou = _numeric_deltas(grouped, left, right, "voxel_iou", valid_only=True)
        f1 = _numeric_deltas(grouped, left, right, "f1_common", valid_only=True)
        result.append(
            {
                "contrast": f"{left}-{right}",
                "label": label,
                "left": left,
                "right": right,
                "valid_only": valid_only,
                **stats,
                "mcnemar_p": success.get("p_value"),
                "mcnemar_only_left": success.get("only_a_completed"),
                "mcnemar_only_right": success.get("only_b_completed"),
                "mean_delta_common_frame_cd_valid": paired_continuous(cd)["mean_delta"] if cd else None,
                "mean_delta_voxel_iou_valid": paired_continuous(iou)["mean_delta"] if iou else None,
                "mean_delta_f1_common_valid": paired_continuous(f1)["mean_delta"] if f1 else None,
            }
        )
    return result


def stratified(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _by_sample(rows)
    result = []
    for field in ("difficulty",):
        levels = sorted({row.get(field) for row in rows if row.get(field) is not None}, key=str)
        for level in levels:
            sample_ids = {row["sample_id"] for row in rows if row.get(field) == level}
            subset_grouped = {sid: grouped[sid] for sid in sample_ids if sid in grouped}
            subset_rows = [row for row in rows if row["sample_id"] in sample_ids]
            for left, right, label in HELDOUT_CONTRASTS:
                deltas = _numeric_deltas(subset_grouped, left, right, "joint_quality", valid_only=False)
                stats = paired_continuous(deltas)
                result.append(
                    {
                        "stratum": field,
                        "level": level,
                        "n_samples": len(sample_ids),
                        "contrast": f"{left}-{right}",
                        "label": label,
                        "n_pairs": stats["n_pairs"],
                        "mean_delta": stats["mean_delta"],
                        "ci95_low": stats["ci95_low"],
                        "ci95_high": stats["ci95_high"],
                    }
                )
            _ = subset_rows
    return result


def gate_decision(contrasts: list[dict[str, Any]]) -> dict[str, Any]:
    target = next((row for row in contrasts if row["contrast"] == "P_geom_tool-P_proj" and not row["valid_only"]), None)
    if target is None:
        return {"proceed": False, "reason": "missing_P_geom_minus_P_proj"}
    mean_delta = target.get("mean_delta")
    low = target.get("ci95_low")
    if mean_delta is None or low is None:
        return {"proceed": False, "reason": "empty_contrast"}
    flipped = float(mean_delta) < 0
    large_negative = float(low) < -0.05 and float(mean_delta) <= 0
    proceed = not flipped and not large_negative
    return {
        "proceed": proceed,
        "mean_delta": mean_delta,
        "ci95_low": low,
        "ci95_high": target.get("ci95_high"),
        "flipped": flipped,
        "large_negative_ci": large_negative,
        "reason": "ok" if proceed else ("direction_flipped" if flipped else "ci_crosses_large_negative"),
    }


def write_report(
    output_dir: Path,
    *,
    n_excluded: int,
    n_kept: int,
    metrics: list[dict[str, Any]],
    contrasts_all: list[dict[str, Any]],
    contrasts_valid: list[dict[str, Any]],
    holm: list[dict[str, Any]],
    strata: list[dict[str, Any]],
    gate: dict[str, Any],
) -> Path:
    lines = [
        "# V4 held-out 80 重分析（Phase A，零 API）",
        "",
        f"- 排除筛选样本：{n_excluded}",
        f"- 保留确认样本：{n_kept}",
        "- 配对探索性数字同时给出 Wilcoxon / McNemar / Holm；门槛只看 `P_geom − P_proj` 方向。",
        "",
        "## 条件汇总",
        "",
        "| 条件 | n | 有效率 | mean JQ | 有效子集 mean JQ | common-frame CD | F1 common |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['condition']} | {row['n']} | {_fmt(row['valid_ratio'])} | "
            f"{_fmt(row['mean_joint_quality'])} | {_fmt(row['mean_joint_quality_valid'])} | "
            f"{_fmt(row['mean_common_frame_cd_valid'])} | {_fmt(row['mean_f1_common_valid'])} |"
        )
    lines += ["", "## 全样本配对（失败记 0）", ""]
    lines += [
        "| 对比 | n | mean ΔJQ | median Δ | 95% CI | 赢/输/平 | Wilcoxon p | Holm p |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    holm_by = {row["contrast"]: row for row in holm}
    for row in contrasts_all:
        adj = holm_by.get(row["contrast"], {})
        lines.append(
            f"| {row['label']} | {row['n_pairs']} | {_fmt(row['mean_delta'])} | {_fmt(row['median_delta'])} | "
            f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}] | "
            f"{row['n_left_better']}/{row['n_right_better']}/{row['n_tie']} | "
            f"{_fmt(row['wilcoxon_p'])} | {_fmt(adj.get('p_holm'))} |"
        )
    lines += ["", "## paired-valid-only 几何", ""]
    lines += [
        "| 对比 | n | ΔJQ | Δ common-frame CD | Δ IoU | Δ F1 common |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in contrasts_valid:
        lines.append(
            f"| {row['label']} | {row['n_pairs']} | {_fmt(row['mean_delta'])} | "
            f"{_fmt(row['mean_delta_common_frame_cd_valid'])} | "
            f"{_fmt(row['mean_delta_voxel_iou_valid'])} | {_fmt(row['mean_delta_f1_common_valid'])} |"
        )
    lines += ["", "## 难度分层 mean ΔJQ", ""]
    lines += ["| 难度 | 对比 | n | mean Δ | 95% CI |", "|---|---|---:|---:|---|"]
    for row in strata:
        lines.append(
            f"| {row['level']} | {row['label']} | {row['n_pairs']} | {_fmt(row['mean_delta'])} | "
            f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}] |"
        )
    lines += [
        "",
        "## 门槛",
        "",
        f"- `P_geom − P_proj` mean Δ = {_fmt(gate.get('mean_delta'))}，CI "
        f"[{_fmt(gate.get('ci95_low'))}, {_fmt(gate.get('ci95_high'))}]",
        f"- 判定：{'进入 Phase B' if gate.get('proceed') else '暂停扩 API'}（{gate.get('reason')}）",
        "",
    ]
    path = output_dir / "V4_HELDOUT80_REAUDIT_ZH.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def run_heldout80(
    *,
    state_dir: Path,
    manifest_path: Path,
    selection_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    exclude = load_screen_ids(selection_summary)
    rows = load_confirm_rows(state_dir, manifest_path, exclude_ids=exclude)
    kept = {row["sample_id"] for row in rows}
    metrics = condition_metrics(rows)
    contrasts_all = contrast_table(rows, valid_only=False)
    contrasts_valid = contrast_table(rows, valid_only=True)
    holm = apply_holm(contrasts_all, "wilcoxon_p")
    strata = stratified(rows)
    gate = gate_decision(contrasts_all)
    output_dir.mkdir(parents=True, exist_ok=True)
    from .pc_analysis import _write_csv

    _write_csv(output_dir / "heldout80_metrics.csv", metrics)
    _write_csv(output_dir / "heldout80_paired_contrasts.csv", contrasts_all + contrasts_valid)
    _write_csv(output_dir / "heldout80_holm.csv", holm)
    _write_csv(output_dir / "heldout80_strata.csv", strata)
    report = write_report(
        output_dir,
        n_excluded=len(exclude),
        n_kept=len(kept),
        metrics=metrics,
        contrasts_all=contrasts_all,
        contrasts_valid=contrasts_valid,
        holm=holm,
        strata=strata,
        gate=gate,
    )
    payload = {
        "n_excluded": len(exclude),
        "n_kept": len(kept),
        "n_rows": len(rows),
        "gate": gate,
        "report": str(report),
    }
    atomic_write_json(output_dir / "heldout80_summary.json", payload)
    return payload
