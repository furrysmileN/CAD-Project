"""RQ2b 确认实验分析：A0（qwen3.8-max 新基线）vs C（v3+repair+反馈），
外加 pilot_v2（qwen3.7-plus）作为模型/运行方差的历史参照。

口径与 encoding_analysis 保持一致：失败类别映射、token 统计、修复率定义均复用
feedback.failure_kind_from_code 与相同的轮次语义；配对检验按 sample×condition 对齐，
bootstrap 以样本为单位重采样（保留样本内 7 条件相关性）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import yaml

from .common import read_jsonl
from .feedback import failure_kind_from_code

CONFIRM_CONDITIONS = ("T", "I", "P", "TI", "TP", "IP", "TIP")

# 分析臂：<arm> → (state 目录, 模型, 描述)。pilot 为历史参照（同 pipeline，仅模型不同）。
ARM_DEFS: dict[str, dict[str, Any]] = {
    "A0": {
        "state_dir": "experiments/rq2_multimodal_harness/outputs/confirm_n100/arms/A0/state",
        "model": "qwen3.8-max",
        "label": "A0（新基线）",
        "description": "v2 prompt · 无 repair · 无反馈",
    },
    "C": {
        "state_dir": "experiments/rq2_multimodal_harness/outputs/confirm_n100/arms/C/state",
        "model": "qwen3.8-max",
        "label": "C",
        "description": "v3 prompt · repair R4 · 2 轮反馈",
    },
    "pilot": {
        "state_dir": "experiments/rq2_multimodal_harness/outputs/pilot_v2/state",
        "model": "qwen3.7-plus",
        "label": "pilot_v2（历史）",
        "description": "v2 prompt · 无 repair · 无反馈",
    },
}


def _round_failure_kind(round_record: Any) -> str | None:
    if not isinstance(round_record, dict):
        return None
    failure = round_record.get("failure")
    if not isinstance(failure, dict):
        return None
    kind = str(failure.get("kind") or "")
    if kind == "schema":
        return "format"
    if kind == "execution":
        inner = failure.get("failure")
        code = inner.get("code") if isinstance(inner, dict) else None
        return failure_kind_from_code(code) or "execution"
    return kind or None


def _usage_tokens(usage: Any) -> dict[str, float | None]:
    if not isinstance(usage, dict):
        return {"input": None, "output": None, "total": None}
    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens")
    return {
        "input": float(input_tokens) if input_tokens is not None else None,
        "output": float(output_tokens) if output_tokens is not None else None,
        "total": float(usage.get("total_tokens") or 0.0)
        if usage.get("total_tokens") is not None
        else None,
    }


def _sum_token_field(usage_rows: list[dict[str, float | None]], field: str) -> float | None:
    values = [row[field] for row in usage_rows]
    if not values or all(value is None for value in values):
        return None
    return float(sum(value or 0.0 for value in values))


def task_row(state: dict[str, Any], *, arm: str) -> dict[str, Any]:
    feedback = state.get("feedback") or {}
    rounds = feedback.get("rounds")
    if not isinstance(rounds, list):
        rounds = []
    status = str(state.get("status") or "")
    completed = status == "completed"
    geometry = state.get("geometry") or {}
    joint_quality = float(geometry.get("joint_quality") or 0.0) if completed else 0.0
    kinds = [_round_failure_kind(record) for record in rounds]
    if not rounds:
        # 无 feedback 块的旧 state（pilot_v2 复用）：按终态推断单轮失败类型
        if completed:
            kinds = [None]
        elif status == "parse_failed":
            kinds = ["format"]
        elif status == "episode_failed":
            failure_code = (
                (((state.get("episode") or {}).get("response") or {}).get("failure") or {}).get("code")
            )
            kinds = [failure_kind_from_code(failure_code) or "execution"]
        else:
            kinds = ["other"]
    n_rounds = len(kinds)
    kept_round = feedback.get("kept_round")
    fixed = bool(completed and n_rounds >= 2)
    fixed_at_round = int(kept_round) if fixed and kept_round is not None else None
    new_error_kind = any(
        kinds[index] and kinds[index - 1] and kinds[index] != kinds[index - 1]
        for index in range(1, len(kinds))
    )
    usage_rows: list[dict[str, float | None]] = []
    for record in rounds:
        api = record.get("api") if isinstance(record, dict) else None
        usage_rows.append(_usage_tokens(api.get("usage") if isinstance(api, dict) else None))
    if not usage_rows:
        usage_rows = [_usage_tokens((state.get("api") or {}).get("usage"))]
    round0 = usage_rows[0]
    total_input = _sum_token_field(usage_rows, "input")
    total_output = _sum_token_field(usage_rows, "output")
    total_tokens = _sum_token_field(usage_rows, "total")
    feedback_input = (
        total_input - round0["input"]
        if total_input is not None and round0["input"] is not None
        else None
    )
    feedback_output = (
        total_output - round0["output"]
        if total_output is not None and round0["output"] is not None
        else None
    )
    return {
        "arm": arm,
        "sample_id": str(state.get("sample_id") or ""),
        "condition": str(state.get("condition_id") or state.get("condition") or ""),
        "status": status,
        "completed": completed,
        "joint_quality": joint_quality,
        "n_rounds": n_rounds,
        "kept_round": kept_round,
        "fixed": fixed,
        "fixed_at_round": fixed_at_round,
        "round0_failure_kind": kinds[0] if kinds else None,
        "round1_failure_kind": kinds[1] if len(kinds) > 1 else None,
        "round2_failure_kind": kinds[2] if len(kinds) > 2 else None,
        "new_error_kind": new_error_kind,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_tokens,
        "feedback_input_tokens": feedback_input,
        "feedback_output_tokens": feedback_output,
    }


def load_task_rows(state_dir: Path, arm: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    state_root = Path(state_dir)
    subdir = state_root / "state" if (state_root / "state").is_dir() else state_root
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(subdir.glob("*/*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(state.get("sample_id") or path.parent.name)
        condition = str(state.get("condition_id") or state.get("condition") or path.stem)
        if condition not in CONFIRM_CONDITIONS:
            continue
        key = (sample_id, condition)
        if key in seen:
            raise ValueError(f"重复 task state: {sample_id}/{condition}")
        seen.add(key)
        if sample_id not in manifest:
            raise ValueError(f"state 样本不在清单中: {sample_id}")
        row = task_row(state, arm=arm)
        for field in ("family", "difficulty", "complexity_bin"):
            row[field] = manifest[sample_id].get(field)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"{state_dir} 中没有可分析的确认实验 state")
    return rows


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    completed = [row for row in rows if row["completed"]]
    round0_failed = [row for row in rows if row["round0_failure_kind"]]
    format_failed = [row for row in round0_failed if row["round0_failure_kind"] == "format"]
    execution_failed = [row for row in round0_failed if row["round0_failure_kind"] == "execution"]
    fixed = [row for row in round0_failed if row["fixed"]]
    multi = [row for row in rows if row["n_rounds"] >= 2]
    new_error = [row for row in multi if row["new_error_kind"]]
    fixed_at_1 = sum(1 for row in fixed if row["fixed_at_round"] == 1)
    fixed_at_2 = sum(1 for row in fixed if row["fixed_at_round"] == 2)
    quality_values = [row["joint_quality"] for row in rows]
    return {
        "arm": rows[0]["arm"] if rows else None,
        "n": n,
        "completed": len(completed),
        "success_rate": _ratio(len(completed), n),
        "joint_quality_mean": mean(quality_values) if quality_values else None,
        "joint_quality_median": median(quality_values) if quality_values else None,
        "round0_failed": len(round0_failed),
        "format_failed": len(format_failed),
        "execution_failed": len(execution_failed),
        "format_fix_rate": _ratio(len([r for r in format_failed if r["fixed"]]), len(format_failed)),
        "execution_fix_rate": _ratio(
            len([r for r in execution_failed if r["fixed"]]), len(execution_failed)
        ),
        "overall_fix_rate": _ratio(len(fixed), len(round0_failed)),
        "fixed_at_round1": fixed_at_1,
        "fixed_at_round2": fixed_at_2,
        "multi_round_tasks": len(multi),
        "new_error_kind_tasks": len(new_error),
        "new_error_kind_rate": _ratio(len(new_error), len(multi)),
        "input_tokens_mean": _mean_field(rows, "input_tokens"),
        "output_tokens_mean": _mean_field(rows, "output_tokens"),
        "total_tokens_mean": _mean_field(rows, "total_tokens"),
        "feedback_input_tokens_mean": _mean_field(rows, "feedback_input_tokens"),
        "feedback_output_tokens_mean": _mean_field(rows, "feedback_output_tokens"),
    }


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row[field] for row in rows if row[field] is not None]
    return mean(values) if values else None


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    result: dict[str, dict[str, Any]] = {}
    for condition in CONFIRM_CONDITIONS:
        group = by_condition.get(condition, [])
        result[condition] = arm_summary(group) if group else {"n": 0, "success_rate": None}
    return result


def mcnemar_exact(a_completed: np.ndarray, b_completed: np.ndarray) -> dict[str, Any]:
    """配对二分类 McNemar 精确检验（双边，二项分布）。a、b 为等长 bool 数组。"""
    both = int(np.sum(a_completed & b_completed))
    only_a = int(np.sum(a_completed & ~b_completed))
    only_b = int(np.sum(~a_completed & b_completed))
    neither = int(np.sum(~a_completed & ~b_completed))
    total = int(len(a_completed))
    disc = only_a + only_b
    if disc == 0:
        p_value = 1.0
    else:
        k = only_a
        # 双边 p = 2 * min(P(X<=k), P(X>=k)) 截断到 1
        p_lower = sum(
            math.comb(disc, i) * (0.5 ** disc) for i in range(k + 1)
        )
        p_upper = sum(
            math.comb(disc, i) * (0.5 ** disc) for i in range(k, disc + 1)
        )
        p_value = min(1.0, 2.0 * min(p_lower, p_upper))
    return {
        "n": total,
        "both_completed": both,
        "only_a_completed": only_a,
        "only_b_completed": only_b,
        "neither": neither,
        "discordant": disc,
        "p_value": p_value,
        "direction": "a_better" if only_a > only_b else ("b_better" if only_b > only_a else "tie"),
    }


def wilcoxon_signed_rank(diffs: np.ndarray) -> dict[str, Any]:
    """配对 Wilcoxon 符号秩检验（正态近似 + 平局修正），返回 stat、p、n 非零差。"""
    values = np.asarray(diffs, dtype=float)
    nonzero = values[values != 0.0]
    n = len(nonzero)
    if n == 0:
        return {"n": 0, "stat": 0.0, "p_value": 1.0, "mean_diff": float(np.mean(values)) if len(values) else 0.0}
    ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1.0
    # 平局修正：对相同 |d| 取平均秩
    abs_vals = np.abs(nonzero)
    _, inverse, counts = np.unique(abs_vals, return_inverse=True, return_counts=True)
    avg_rank = {}
    order = np.argsort(abs_vals)
    for value in np.unique(abs_vals):
        indices = np.where(abs_vals == value)[0]
        avg_rank[value] = float(np.mean(ranks[indices]))
    rank_values = np.array([avg_rank[v] for v in abs_vals])
    w_plus = float(np.sum(rank_values[nonzero > 0]))
    w_minus = float(np.sum(rank_values[nonzero < 0]))
    stat = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    tie_correction = 0.0
    for count in counts:
        if count > 1:
            tie_correction += count * (count - 1) * (count + 1)
    if tie_correction:
        var_w -= tie_correction / 48.0
    z = (stat - mean_w) / math.sqrt(var_w) if var_w > 0 else 0.0
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return {
        "n": n,
        "stat": stat,
        "z": z,
        "p_value": p_value,
        "mean_diff": float(np.mean(values)),
        "median_diff": float(np.median(values)),
    }


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bootstrap_paired_delta(
    sample_ids: list[str],
    deltas: dict[tuple[str, str], float],
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """样本级重采样：对 100 个样本有放回抽样，重算配对差均值（样本内 7 条件取均值）。"""
    rng = np.random.default_rng(seed)
    per_sample = defaultdict(list)
    for (sample_id, _condition), delta in deltas.items():
        per_sample[sample_id].append(delta)
    sample_means = {sample_id: float(np.mean(values)) for sample_id, values in per_sample.items()}
    ids = list(sample_means.keys())
    boot_means = []
    for _ in range(repeats):
        picked = rng.choice(len(ids), size=len(ids), replace=True)
        boot_means.append(float(np.mean([sample_means[ids[i]] for i in picked])))
    arr = np.asarray(boot_means)
    return {
        "repeats": repeats,
        "seed": seed,
        "observed_mean": float(np.mean([sample_means[i] for i in ids])),
        "mean": float(np.mean(arr)),
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "std": float(np.std(arr, ddof=1)) if repeats > 1 else 0.0,
    }


def _paired(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index_b = {(row["sample_id"], row["condition"]): row for row in rows_b}
    paired_a: list[dict[str, Any]] = []
    paired_b: list[dict[str, Any]] = []
    for row in rows_a:
        match = index_b.get((row["sample_id"], row["condition"]))
        if match is None:
            raise ValueError(
                f"配对缺失：{row['arm']} 有 {row['sample_id']}/{row['condition']}，另一臂无"
            )
        paired_a.append(row)
        paired_b.append(match)
    return paired_a, paired_b


def _stratify(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "unknown")].append(row)
    return {key: arm_summary(group) for key, group in sorted(groups.items())}


def _fmt(value: Any, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _pct(value: Any) -> str:
    return "NA" if value is None else f"{value:.1%}"


def write_report(
    output_dir: Path,
    arm_summaries: dict[str, dict[str, Any]],
    condition_summaries: dict[str, dict[str, dict[str, Any]]],
    paired: dict[str, dict[str, Any]],
    stratified: dict[str, dict[str, dict[str, Any]]],
    figure: str | None = None,
) -> None:
    lines = [
        "# RQ2b 确认实验报告",
        "",
        "## 实验设计",
        "",
        "- 冻结臂 × 全量 100 样本 × 7 条件（T/I/P/TI/TP/IP/TIP），共 700 任务/臂，按",
        "  sample×condition 配对比较；样本与输入复用 pilot_v2 冻结 manifest。",
    ]
    for arm in ("A0", "C", "pilot"):
        if arm not in arm_summaries:
            continue
        spec = ARM_DEFS[arm]
        lines.append(f"- {spec['label']}（{spec['model']}）：{spec['description']}。")
    lines.extend(["", "## 各臂总体结果", ""])
    header = ["指标"] + [ARM_DEFS[arm]["label"] for arm in ("A0", "C", "pilot") if arm in arm_summaries]
    rows = [
        ("任务数", lambda s: str(s["n"])),
        ("成功率", lambda s: _pct(s.get("success_rate"))),
        ("joint quality 均值", lambda s: _fmt(s.get("joint_quality_mean"))),
        ("joint quality 中位数", lambda s: _fmt(s.get("joint_quality_median"))),
        ("round0 失败数", lambda s: str(s.get("round0_failed"))),
        ("格式类失败数", lambda s: str(s.get("format_failed"))),
        ("执行类失败数", lambda s: str(s.get("execution_failed"))),
        ("总体修复率", lambda s: _pct(s.get("overall_fix_rate"))),
        ("格式类修复率", lambda s: _pct(s.get("format_fix_rate"))),
        ("执行类修复率", lambda s: _pct(s.get("execution_fix_rate"))),
        ("多轮任务数", lambda s: str(s.get("multi_round_tasks"))),
        ("引入新错误率", lambda s: _pct(s.get("new_error_kind_rate"))),
        ("每任务 input tokens 均值", lambda s: _fmt(s.get("input_tokens_mean"), 1)),
        ("每任务 output tokens 均值", lambda s: _fmt(s.get("output_tokens_mean"), 1)),
        ("每任务总 tokens 均值", lambda s: _fmt(s.get("total_tokens_mean"), 1)),
        ("反馈轮 input tokens 增量均值", lambda s: _fmt(s.get("feedback_input_tokens_mean"), 1)),
        ("反馈轮 output tokens 增量均值", lambda s: _fmt(s.get("feedback_output_tokens_mean"), 1)),
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for label, getter in rows:
        values = []
        for arm in ("A0", "C", "pilot"):
            if arm in arm_summaries:
                values.append(getter(arm_summaries[arm]))
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.extend(["", "## 逐条件结果（成功率 / joint quality 均值）", ""])
    cond_header = ["条件"] + [ARM_DEFS[arm]["label"] for arm in ("A0", "C", "pilot") if arm in arm_summaries]
    lines.append("| " + " | ".join(cond_header) + " |")
    lines.append("|" + "---|" * len(cond_header))
    for condition in CONFIRM_CONDITIONS:
        values = []
        for arm in ("A0", "C", "pilot"):
            if arm in condition_summaries and condition in condition_summaries[arm]:
                summary = condition_summaries[arm][condition]
                values.append(
                    f"{_pct(summary.get('success_rate'))} / {_fmt(summary.get('joint_quality_mean'))}"
                )
            elif arm in condition_summaries:
                values.append("–")
        lines.append(f"| {condition} | " + " | ".join(values) + " |")

    lines.extend(["", "## 配对检验（A0 新基线 vs C，同模型 qwen3.8-max）", ""])
    for key, label in (
        ("a0_vs_c", "A0 vs C"),
        ("a0_vs_pilot", "A0 vs pilot_v2（模型/运行方差对照）"),
    ):
        if key not in paired:
            continue
        result = paired[key]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- McNemar（完成率）：n={result['mcnemar']['n']}，",
                f"仅 A0 完成 {result['mcnemar']['only_a_completed']} / 仅对手完成 "
                f"{result['mcnemar']['only_b_completed']}，p={result['mcnemar']['p_value']:.4f}，"
                f"方向 {result['mcnemar']['direction']}。",
                f"- Wilcoxon 符号秩（joint quality，配对）：n={result['wilcoxon']['n']}，"
                f"平均差 {result['wilcoxon']['mean_diff']:+.4f}，"
                f"中位差 {result['wilcoxon']['median_diff']:+.4f}，p={result['wilcoxon']['p_value']:.4f}。",
                f"- 样本级 bootstrap（{result['bootstrap']['repeats']} 次，seed="
                f"{result['bootstrap']['seed']}）：观测均值 "
                f"{result['bootstrap']['observed_mean']:+.4f}，"
                f"95% CI [{result['bootstrap']['ci95'][0]:+.4f}, "
                f"{result['bootstrap']['ci95'][1]:+.4f}]。",
                "",
            ]
        )

    lines.extend(["## 分层结果（成功率，A0 vs C）", ""])
    for field, label in (("difficulty", "难度"), ("complexity_bin", "复杂度 bin"), ("family", "族")):
        if field not in stratified:
            continue
        lines.append(f"### {label}")
        lines.append("| 分组 | A0 n | A0 成功率 | C n | C 成功率 |")
        lines.append("|---:|---:|---:|---:|---:|")
        groups = sorted({group for arm_data in stratified[field].values() for group in arm_data})
        for group in groups:
            a = stratified[field].get("A0", {}).get(group)
            c = stratified[field].get("C", {}).get(group)
            if a is None and c is None:
                continue
            lines.append(
                f"| {group} | {a['n'] if a else '–'} | {_pct(a.get('success_rate')) if a else '–'} | "
                f"{c['n'] if c else '–'} | {_pct(c.get('success_rate')) if c else '–'} |"
            )
        lines.append("")

    lines.append("## 统计说明")
    lines.extend(
        [
            "",
            "- McNemar 为配对二分类精确检验（二项双边）；Wilcoxon 为正态近似 + 平局修正。",
            "- bootstrap 以样本为单位重采样（样本内 7 条件差取均值），95% CI 为 2.5/97.5 百分位。",
            "- 格式类 = JSON 解析失败、client 侧 schema 校验失败与后端 plan_validation_failed；",
            "  执行类 = CadQuery 执行异常。失败类别映射与 runner 共用 failure_kind_from_code。",
            "- 修复率分母为 round0 失败的同类任务；引入新错误率分母为多轮任务数。",
            "- A0 新基线 vs pilot_v2 的差异同时包含模型（3.8-max vs 3.7-plus）与运行方差，"
            "仅作方向性参照，不作处理效应归因。",
        ]
    )
    if figure:
        lines.extend(["", "## 图表", "", f"- `{figure}`"])
    (output_dir / "CONFIRM_REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_confirmation(
    arm_state_dirs: dict[str, Path],
    output_dir: Path,
    manifest_path: Path,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    manifest_rows = list(read_jsonl(manifest_path))
    manifest = {str(row["sample_id"]): row for row in manifest_rows}
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    for arm in ("A0", "C", "pilot"):
        if arm not in arm_state_dirs:
            continue
        arm_rows[arm] = load_task_rows(arm_state_dirs[arm], arm, manifest)
    arm_summaries = {arm: arm_summary(rows) for arm, rows in arm_rows.items()}
    condition_summaries = {arm: condition_summary(rows) for arm, rows in arm_rows.items()}

    paired: dict[str, dict[str, Any]] = {}
    if "A0" in arm_rows and "C" in arm_rows:
        paired_a, paired_c = _paired(arm_rows["A0"], arm_rows["C"])
        a_done = np.array([row["completed"] for row in paired_a])
        c_done = np.array([row["completed"] for row in paired_c])
        a_jq = np.array([row["joint_quality"] for row in paired_a])
        c_jq = np.array([row["joint_quality"] for row in paired_c])
        deltas = {
            (row_a["sample_id"], row_a["condition"]): row_c["joint_quality"] - row_a["joint_quality"]
            for row_a, row_c in zip(paired_a, paired_c)
        }
        sample_ids = sorted({row["sample_id"] for row in paired_a})
        paired["a0_vs_c"] = {
            "mcnemar": mcnemar_exact(a_done, c_done),
            "wilcoxon": wilcoxon_signed_rank(c_jq - a_jq),
            "bootstrap": bootstrap_paired_delta(
                sample_ids, deltas, repeats=bootstrap_repeats, seed=seed
            ),
        }
    if "A0" in arm_rows and "pilot" in arm_rows:
        paired_a, paired_p = _paired(arm_rows["A0"], arm_rows["pilot"])
        a_done = np.array([row["completed"] for row in paired_a])
        p_done = np.array([row["completed"] for row in paired_p])
        a_jq = np.array([row["joint_quality"] for row in paired_a])
        p_jq = np.array([row["joint_quality"] for row in paired_p])
        deltas = {
            (row_a["sample_id"], row_a["condition"]): row_a["joint_quality"] - row_p["joint_quality"]
            for row_a, row_p in zip(paired_a, paired_p)
        }
        sample_ids = sorted({row["sample_id"] for row in paired_a})
        paired["a0_vs_pilot"] = {
            "mcnemar": mcnemar_exact(p_done, a_done),
            "wilcoxon": wilcoxon_signed_rank(a_jq - p_jq),
            "bootstrap": bootstrap_paired_delta(
                sample_ids, deltas, repeats=bootstrap_repeats, seed=seed + 1
            ),
        }

    stratified: dict[str, dict[str, dict[str, Any]]] = {}
    if "A0" in arm_rows and "C" in arm_rows:
        for field in ("difficulty", "complexity_bin", "family"):
            stratified[field] = {}
            for arm in ("A0", "C"):
                stratified[field][arm] = _stratify(arm_rows[arm], field)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "confirm_analysis.json").write_text(
        json.dumps(
            {
                "arms": {arm: ARM_DEFS[arm] for arm in arm_summaries},
                "arm_summary": arm_summaries,
                "condition_summary": condition_summaries,
                "paired": paired,
                "stratified": stratified,
                "bootstrap": {"repeats": bootstrap_repeats, "seed": seed, "ci": 0.95},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "confirm_arm_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        fields = [
            "arm", "n", "completed", "success_rate", "joint_quality_mean", "joint_quality_median",
            "round0_failed", "format_failed", "execution_failed", "format_fix_rate",
            "execution_fix_rate", "overall_fix_rate", "fixed_at_round1", "fixed_at_round2",
            "multi_round_tasks", "new_error_kind_tasks", "new_error_kind_rate",
            "input_tokens_mean", "output_tokens_mean", "total_tokens_mean",
            "feedback_input_tokens_mean", "feedback_output_tokens_mean",
        ]
        writer.writerow(fields)
        for arm in ("A0", "C", "pilot"):
            if arm not in arm_summaries:
                continue
            summary = arm_summaries[arm]
            writer.writerow([summary.get(field) for field in fields])
    with (output_dir / "confirm_task_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        fields = [
            "arm", "sample_id", "condition", "family", "difficulty", "complexity_bin",
            "status", "completed", "joint_quality", "n_rounds", "kept_round", "fixed",
            "fixed_at_round", "round0_failure_kind", "round1_failure_kind", "round2_failure_kind",
            "new_error_kind", "input_tokens", "output_tokens", "total_tokens",
            "feedback_input_tokens", "feedback_output_tokens",
        ]
        writer.writerow(fields)
        for arm in ("A0", "C", "pilot"):
            for row in arm_rows.get(arm, []):
                writer.writerow([row.get(field) for field in fields])

    write_report(output_dir, arm_summaries, condition_summaries, paired, stratified)
    return {
        "arm_summary": arm_summaries,
        "paired": paired,
    }


def feedback_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分析 RQ2b 确认实验（A0 新基线 vs C + pilot 参照）")
    harness_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", default=str(harness_dir / "outputs" / "confirm_n100" / "analysis"))
    parser.add_argument(
        "--manifest",
        default=str(harness_dir / "outputs" / "pilot_v2" / "manifest.jsonl"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args(argv)
    arm_state_dirs: dict[str, Path] = {}
    for arm, spec in ARM_DEFS.items():
        candidate = harness_dir.parents[1] / spec["state_dir"]
        if (candidate / "state").is_dir() or candidate.is_dir():
            arm_state_dirs[arm] = candidate
    result = analyze_confirmation(
        arm_state_dirs,
        Path(args.output),
        Path(args.manifest),
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(feedback_main())
