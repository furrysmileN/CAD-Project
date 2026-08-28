"""阶段 0：confirm C 臂零成本重分析（不调用 API）。

目的：在格式与执行失败基本被控制（C 臂成功率 97.6%）之后，重算旧 P_proj
在七条件矩阵中的互补性，确认 Harness 固定后旧点云投影视图是否仍有真实边际
价值，作为 P_geom（原生点云证据）实验的对照基准。

口径：
- 成功 = state.status == "completed"（Episode success / success_with_warnings）。
- joint_quality 失败固定 0（与 geometry.py / confirm_analysis 一致）。
- 互补增益 = 组合条件 − 同一样本上最佳组成单模态（与 analysis.complement_gain 一致）。
- paired-valid-only：只保留两个条件都成功的 sample，再比较几何指标。
- 贡献分解：恢复（单模态无效 → 组合有效）与改善（双方有效且组合几何更优）。
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from .analysis import bootstrap_ci
from .common import read_jsonl

CONFIRM_CONDITIONS = ("T", "I", "P", "TI", "TP", "IP", "TIP")
COMBO_COMPONENTS = {
    "TI": ("T", "I"),
    "TP": ("T", "P"),
    "IP": ("I", "P"),
    "TIP": ("T", "I", "P"),
}
CONFIRM_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "confirm_n100"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "outputs" / "pilot_v2" / "manifest.jsonl"

GEOMETRY_FIELDS = (
    ("shape_only_cd", "shape_only_cd"),
    ("common_frame_cd", "common_frame_cd"),
    ("voxel_iou", "voxel_iou"),
    ("f1_shape", "fscore_shape"),
    ("f1_common", "fscore_common"),
    ("joint_quality", "joint_quality"),
)


def _load_states(state_dir: Path) -> list[dict[str, Any]]:
    states = []
    if not state_dir.is_dir():
        return states
    for path in sorted(state_dir.glob("*/*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("status") in {"dry_run", "running"}:
            continue
        condition = state.get("condition")
        if condition not in CONFIRM_CONDITIONS:
            continue
        states.append(state)
    return states


def _geometry_values(state: dict[str, Any]) -> dict[str, float | None]:
    geometry = state.get("geometry") or {}
    values: dict[str, float | None] = {}
    for key, source in GEOMETRY_FIELDS:
        if source == "joint_quality":
            value = float(geometry.get("joint_quality") or 0.0)
            values[key] = value if state.get("status") == "completed" else 0.0
        elif source == "fscore_shape":
            inner = geometry.get("fscore_shape") or {}
            values[key] = inner.get("f1")
        elif source == "fscore_common":
            inner = geometry.get("fscore_common") or {}
            values[key] = inner.get("f1")
        elif source == "voxel_iou":
            inner = geometry.get("voxel_iou") or {}
            values[key] = inner.get("value")
        else:
            values[key] = geometry.get(source)
    return values


def _pairs_valid_rows(rows: list[dict[str, Any]], left: str, right: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_sample: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][row["condition"]] = row
    return [
        (left_row, right_row)
        for scores in by_sample.values()
        if (left_row := scores.get(left)) is not None
        and (right_row := scores.get(right)) is not None
        and left_row.get("status") == "completed"
        and right_row.get("status") == "completed"
    ]


def _mean_non_null(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return mean(clean) if clean else None


def analyze_c_arm(
    *,
    arm: str = "C",
    seed: int = 20260816,
    bootstrap_repeats: int = 5000,
    state_dir: Path | None = None,
    manifest_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    state_root = state_dir or CONFIRM_ROOT / "arms" / arm / "state"
    rows = _load_states(state_root)
    if not rows:
        raise RuntimeError(f"{state_root} 下没有可分析的任务 state（先运行 confirm --arm {arm}）")
    manifest = (
        {row["sample_id"]: row for row in read_jsonl(manifest_path or MANIFEST_PATH)}
        if (manifest_path or MANIFEST_PATH).is_file()
        else {}
    )
    for row in rows:
        meta = manifest.get(row["sample_id"]) or {}
        row["_family"] = meta.get("family")
        row["_difficulty"] = meta.get("difficulty")
        row["_complexity_bin"] = meta.get("complexity_bin")

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    # 1. 条件汇总
    summaries = []
    for condition in CONFIRM_CONDITIONS:
        subset = by_condition.get(condition, [])
        qualities = [_geometry_values(row)["joint_quality"] or 0.0 for row in subset]
        valid = [row for row in subset if row["status"] == "completed"]
        ci = bootstrap_ci(qualities, seed + CONFIRM_CONDITIONS.index(condition), bootstrap_repeats)
        summaries.append(
            {
                "condition": condition,
                "n": len(subset),
                "success_rate": mean([row["status"] == "completed" for row in subset]) if subset else None,
                "valid_rate": mean([row["status"] == "completed" for row in subset]) if subset else None,
                "joint_quality_mean": ci["mean"],
                "joint_quality_median": median(qualities) if qualities else None,
                "joint_quality_ci_low": ci["low"],
                "joint_quality_ci_high": ci["high"],
                "joint_quality_completed_mean": mean(
                    [_geometry_values(row)["joint_quality"] or 0.0 for row in valid]
                ) if valid else None,
                "shape_only_cd_mean": _mean_non_null(
                    [_geometry_values(row)["shape_only_cd"] for row in valid]
                ),
                "common_frame_cd_mean": _mean_non_null(
                    [_geometry_values(row)["common_frame_cd"] for row in valid]
                ),
                "voxel_iou_mean": _mean_non_null([_geometry_values(row)["voxel_iou"] for row in valid]),
                "f1_shape_mean": _mean_non_null([_geometry_values(row)["f1_shape"] for row in valid]),
            }
        )

    # 2. 互补增益（failure-aware joint quality）
    score_matrix: dict[str, dict[str, float]] = defaultdict(dict)
    valid_matrix: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        score_matrix[row["sample_id"]][row["condition"]] = _geometry_values(row)["joint_quality"] or 0.0
        valid_matrix[row["sample_id"]][row["condition"]] = row["status"] == "completed"

    gains: list[dict[str, Any]] = []
    for condition, components in COMBO_COMPONENTS.items():
        values: list[float] = []
        per_sample: list[dict[str, Any]] = []
        for sample, scores in score_matrix.items():
            needed = set(components) | {condition}
            if needed <= set(scores):
                gain = scores[condition] - max(scores[name] for name in components)
                values.append(gain)
                per_sample.append(
                    {
                        "sample_id": sample,
                        "condition": condition,
                        "gain": gain,
                        "best_single": max(scores[name] for name in components),
                    }
                )
        ci = bootstrap_ci(values, seed + 100 + list(COMBO_COMPONENTS).index(condition), bootstrap_repeats)
        wins = sum(1 for value in values if value > 1e-9)
        losses = sum(1 for value in values if value < -1e-9)
        gains.append(
            {
                "condition": condition,
                "components": list(components),
                "n": len(values),
                "mean": ci["mean"],
                "ci_low": ci["low"],
                "ci_high": ci["high"],
                "win_count": wins,
                "loss_count": losses,
                "tie_count": len(values) - wins - losses,
                "per_sample": per_sample,
            }
        )

    # 3. paired-valid-only 几何表：只保留两条件都成功的 sample
    paired_valid: list[dict[str, Any]] = []
    for left, right in (
        ("TI", "T"), ("TI", "I"), ("TP", "T"), ("TP", "P"),
        ("IP", "I"), ("IP", "P"), ("TIP", "TI"), ("TIP", "TP"), ("TIP", "IP"),
    ):
        pairs = _pairs_valid_rows(rows, left, right)
        if not pairs:
            continue
        left_values = {field: _mean_non_null([_geometry_values(a)[field] for a, _ in pairs]) for field, _ in GEOMETRY_FIELDS}
        right_values = {field: _mean_non_null([_geometry_values(b)[field] for _, b in pairs]) for field, _ in GEOMETRY_FIELDS}
        paired_valid.append(
            {
                "left": left,
                "right": right,
                "n_paired_valid": len(pairs),
                **{
                    field: {"left": left_values[field], "right": right_values[field]}
                    for field, _ in GEOMETRY_FIELDS
                },
            }
        )

    # 4. 贡献分解：恢复无效任务 vs 改善有效模型几何
    contribution: list[dict[str, Any]] = []
    for condition, components in COMBO_COMPONENTS.items():
        recovered = []
        improved = []
        regressed = []
        for sample, valid_scores in valid_matrix.items():
            if condition not in valid_scores or not all(name in valid_scores for name in components):
                continue
            combo_valid = valid_scores[condition]
            singles_valid = [valid_scores[name] for name in components]
            combo_q = score_matrix[sample][condition]
            best_single_q = max(score_matrix[sample][name] for name in components)
            if not combo_valid and not any(singles_valid):
                continue
            if combo_valid and not any(singles_valid):
                recovered.append(sample)
            elif combo_valid and any(singles_valid):
                if combo_q > best_single_q + 1e-9:
                    improved.append(sample)
                elif combo_q < best_single_q - 1e-9:
                    regressed.append(sample)
        contribution.append(
            {
                "condition": condition,
                "recovered_count": len(recovered),
                "recovered_samples": recovered,
                "improved_count": len(improved),
                "improved_samples": improved,
                "regressed_count": len(regressed),
                "regressed_samples": regressed,
            }
        )

    # 5. 分层（difficulty / family）
    stratified: list[dict[str, Any]] = []
    for field in ("_difficulty", "_family", "_complexity_bin"):
        groups: dict[tuple[Any, str], list[float]] = defaultdict(list)
        for row in rows:
            value = row.get(field)
            if value is None:
                continue
            groups[(value, row["condition"])].append(_geometry_values(row)["joint_quality"] or 0.0)
        for (value, condition), values in sorted(groups.items(), key=lambda item: (str(item[0][0]), item[0][1])):
            subset = [row for row in rows if row.get(field) == value and row["condition"] == condition]
            stratified.append(
                {
                    "stratum": field.lstrip("_"),
                    "value": value,
                    "condition": condition,
                    "n": len(values),
                    "success_rate": mean([row["status"] == "completed" for row in subset]) if subset else None,
                    "joint_quality_mean": mean(values),
                }
            )

    report = {
        "schema_version": "rq2.p0.reanalysis.v1",
        "arm": arm,
        "n_samples": len({row["sample_id"] for row in rows}),
        "condition_summary": summaries,
        "complement_gains": [{key: item[key] for key in ("condition", "components", "n", "mean", "ci_low", "ci_high", "win_count", "loss_count", "tie_count")} for item in gains],
        "paired_valid_only": paired_valid,
        "contribution_breakdown": contribution,
        "stratified": stratified,
        "bootstrap": {"unit": "sample", "repeats": bootstrap_repeats, "ci": 0.95, "seed": seed},
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "p0_analysis.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_csv(output_dir / "p0_condition_summary.csv", summaries)
        _write_csv(output_dir / "p0_complement_gains.csv", report["complement_gains"])
        _write_csv(
            output_dir / "p0_paired_valid_only.csv",
            [
                {
                    "left": item["left"],
                    "right": item["right"],
                    "n_paired_valid": item["n_paired_valid"],
                    **{f"{field}_{side}": value for field, _ in GEOMETRY_FIELDS for side in ("left", "right") if (value := item[field][side]) is not None},
                }
                for item in paired_valid
            ],
        )
        _write_csv(
            output_dir / "p0_contribution.csv",
            [{key: item[key] for key in ("condition", "recovered_count", "improved_count", "regressed_count")} for item in contribution],
        )
        (output_dir / "P0_CARM_REANALYSIS_ZH.md").write_text(
            _render_report(report), encoding="utf-8"
        )
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# P0：confirm C 臂重分析——旧点云投影视图（P_proj）在 Harness 固定后的互补性",
        "",
        "> 零 API 成本；数据来源 `outputs/confirm_n100/arms/C/state`（qwen3.8-max，v3 prompt + R4 + 2 轮反馈）。",
        "> 本报告是 P_geom（原生点云证据）实验的基线对照，不构成显著性宣称。",
        "",
        f"- 样本数：{report['n_samples']}；成功率口径：`status == completed`。",
        "",
        "## 条件汇总",
        "",
        "| 条件 | n | 成功率 | joint quality 均值 | 95% CI | 中位数 | 成功者几何均值 | shape-only CD | common CD | voxel IoU | F1@0.01 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["condition_summary"]:
        lines.append(
            f"| {item['condition']} | {item['n']} | {_fmt(item['success_rate'])} | {_fmt(item['joint_quality_mean'])} | "
            f"[{_fmt(item['joint_quality_ci_low'])}, {_fmt(item['joint_quality_ci_high'])}] | {_fmt(item['joint_quality_median'])} | "
            f"{_fmt(item['joint_quality_completed_mean'])} | {_fmt(item['shape_only_cd_mean'])} | "
            f"{_fmt(item['common_frame_cd_mean'])} | {_fmt(item['voxel_iou_mean'])} | {_fmt(item['f1_shape_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## 互补增益（组合 − 最佳组成单模态，failure-aware joint quality）",
            "",
            "| 组合 | 组成 | n | 均值增益 | 95% CI | 赢/平/输 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in report["complement_gains"]:
        lines.append(
            f"| {item['condition']} | {'+'.join(item['components'])} | {item['n']} | {_fmt(item['mean'])} | "
            f"[{_fmt(item['ci_low'])}, {_fmt(item['ci_high'])}] | {item['win_count']}/{item['tie_count']}/{item['loss_count']} |"
        )
    lines.extend(
        [
            "",
            "## paired-valid-only 几何表（只保留两条件都成功的 sample×task）",
            "",
            "| 对比 | n | joint quality 左/右 | shape-only CD 左/右 | common CD 左/右 | voxel IoU 左/右 |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for item in report["paired_valid_only"]:
        lines.append(
            f"| {item['left']} vs {item['right']} | {item['n_paired_valid']} | "
            f"{_fmt(item['joint_quality']['left'])} / {_fmt(item['joint_quality']['right'])} | "
            f"{_fmt(item['shape_only_cd']['left'])} / {_fmt(item['shape_only_cd']['right'])} | "
            f"{_fmt(item['common_frame_cd']['left'])} / {_fmt(item['common_frame_cd']['right'])} | "
            f"{_fmt(item['voxel_iou']['left'])} / {_fmt(item['voxel_iou']['right'])} |"
        )
    lines.extend(
        [
            "",
            "## 贡献分解（组合相对最佳单模态）",
            "",
            "| 组合 | 恢复无效任务 | 改善有效几何 | 有效但变差 |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in report["contribution_breakdown"]:
        lines.append(
            f"| {item['condition']} | {item['recovered_count']} | {item['improved_count']} | {item['regressed_count']} |"
        )
    lines.extend(
        [
            "",
            "## 统计说明",
            "",
            "- 配对单位：sample×condition；bootstrap 以样本为单位重采样。",
            "- paired-valid-only 表的均值只包含两条件均完成的样本，用于区分「多生成有效模型」与「有效模型本身更准确」。",
            "- 分层结果（difficulty/family/complexity_bin）见 `p0_analysis.json`。",
        ]
    )
    return "\n".join(lines) + "\n"
