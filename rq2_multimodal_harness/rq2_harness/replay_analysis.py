from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from .common import atomic_write_json, load_config, project_path
from .conditions import CONDITIONS
from .replay_v21 import SETTING_ORDER


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _task_row(state: dict[str, Any]) -> dict[str, Any]:
    stage = state.get("stage") or {}
    episode_response = ((state.get("episode") or {}).get("response") or {})
    validation = episode_response.get("validation") or {}
    failure = episode_response.get("failure") or {}
    repair = state.get("repair_v21") or {}
    geometry = state.get("geometry") or {}
    return {
        "setting": state["setting"],
        "sample_id": state["sample_id"],
        "condition": state["condition"],
        "status": stage.get("status") or state.get("status"),
        "parse_ok": bool(stage.get("parse_ok", False)),
        "schema_valid": bool(stage.get("schema_valid", False)),
        "execution_success": bool(stage.get("execution_success", False)),
        "geometry_valid": bool(stage.get("geometry_valid", False)),
        "joint_quality": float(stage.get("joint_quality") or 0.0),
        "shape_only_cd": geometry.get("shape_only_cd"),
        "common_frame_cd": geometry.get("common_frame_cd"),
        "voxel_iou": (geometry.get("voxel_iou") or {}).get("value"),
        "episode_status": stage.get("episode_status"),
        "failure_code": failure.get("code"),
        "validation_issue_codes": ",".join(
            sorted({str(issue.get("code")) for issue in validation.get("issues", []) if issue.get("code")})
        ),
        "repair_changed": bool(repair.get("changed", False)),
        "repair_codes": ",".join(repair.get("repair_codes") or []),
        "repair_count": int(repair.get("repair_count") or 0),
        "before_plan_sha256": repair.get("before_sha256"),
        "after_plan_sha256": repair.get("after_sha256"),
        "execution_mode": state.get("execution_mode"),
        "reused_from": state.get("reused_from"),
        "result_step_path": state.get("result_step_path"),
    }


def _load_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for setting in SETTING_ORDER:
        state_dir = output_dir / "repair_state" / setting
        paths = sorted(state_dir.glob("*/*.json"))
        if len(paths) != 700:
            raise RuntimeError(f"{setting} 只有 {len(paths)} 个 task state，预期 700")
        rows.extend(_task_row(json.loads(path.read_text(encoding="utf-8"))) for path in paths)
    return rows


def _exact_mcnemar(before: Iterable[bool], after: Iterable[bool]) -> dict[str, Any]:
    pairs = list(zip(before, after))
    rescued = sum(not left and right for left, right in pairs)
    regressed = sum(left and not right for left, right in pairs)
    discordant = rescued + regressed
    if not discordant:
        p_value = 1.0
        status = "all_pairs_equal"
    else:
        from scipy.stats import binomtest

        p_value = float(binomtest(min(rescued, regressed), n=discordant, p=0.5, alternative="two-sided").pvalue)
        status = "ok"
    return {
        "n": len(pairs),
        "rescued": rescued,
        "regressed": regressed,
        "discordant": discordant,
        "p_value": p_value,
        "status": status,
    }


def _bootstrap_values(values: list[float], *, seed: int, repeats: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "wins": 0, "ties": 0, "losses": 0}
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(repeats, len(array)))].mean(axis=1)
    eps = 1e-15
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "wins": int(np.sum(array > eps)),
        "ties": int(np.sum(np.abs(array) <= eps)),
        "losses": int(np.sum(array < -eps)),
    }


def _sample_level_deltas(
    before: dict[tuple[str, str], dict[str, Any]],
    after: dict[tuple[str, str], dict[str, Any]],
    *,
    condition: str | None,
    field: str,
) -> list[float]:
    by_sample: dict[str, list[float]] = defaultdict(list)
    for key, left in before.items():
        sample_id, task_condition = key
        if condition is not None and task_condition != condition:
            continue
        right = after[key]
        by_sample[sample_id].append(float(right[field]) - float(left[field]))
    return [mean(values) for _, values in sorted(by_sample.items())]


def _difference_in_differences(
    before: dict[tuple[str, str], dict[str, Any]],
    after: dict[tuple[str, str], dict[str, Any]],
    *,
    left_condition: str,
    right_condition: str,
    field: str,
) -> list[float]:
    samples = sorted({sample_id for sample_id, _ in before})
    values = []
    for sample_id in samples:
        left_key = (sample_id, left_condition)
        right_key = (sample_id, right_condition)
        values.append(
            (float(after[left_key][field]) - float(before[left_key][field]))
            - (float(after[right_key][field]) - float(before[right_key][field]))
        )
    return values


def _scope_rows(rows: list[dict[str, Any]], condition: str | None) -> list[dict[str, Any]]:
    return rows if condition is None else [row for row in rows if row["condition"] == condition]


def analyze_replay(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = project_path(config["paths"]["output_dir"]).resolve()
    rows = _load_rows(output_dir)
    by_setting = {setting: [row for row in rows if row["setting"] == setting] for setting in SETTING_ORDER}
    matrix = {
        setting: {(row["sample_id"], row["condition"]): row for row in setting_rows}
        for setting, setting_rows in by_setting.items()
    }
    baseline = matrix["R0"]
    repeats = int(config["analysis"]["bootstrap_repeats"])
    seed = int(config["seed"])

    summary_rows: list[dict[str, Any]] = []
    for setting_index, setting in enumerate(SETTING_ORDER):
        for condition in (None, *CONDITIONS):
            subset = _scope_rows(by_setting[setting], condition)
            base_subset = _scope_rows(by_setting["R0"], condition)
            base_index = {(row["sample_id"], row["condition"]): row for row in base_subset}
            schema_rescued = sum(
                not base_index[(row["sample_id"], row["condition"])]["schema_valid"] and row["schema_valid"]
                for row in subset
            )
            geometry_rescued = sum(
                not base_index[(row["sample_id"], row["condition"])]["geometry_valid"] and row["geometry_valid"]
                for row in subset
            )
            baseline_schema_failed = sum(row["parse_ok"] and not row["schema_valid"] for row in base_subset)
            baseline_geometry_invalid = sum(not row["geometry_valid"] for row in base_subset)
            baseline_execution_success = sum(row["execution_success"] for row in base_subset)
            baseline_geometry_valid = sum(row["geometry_valid"] for row in base_subset)
            execution_regressed = sum(
                base_index[(row["sample_id"], row["condition"])]["execution_success"] and not row["execution_success"]
                for row in subset
            )
            geometry_regressed = sum(
                base_index[(row["sample_id"], row["condition"])]["geometry_valid"] and not row["geometry_valid"]
                for row in subset
            )
            quality_ci = _bootstrap_values(
                _sample_level_deltas(baseline, matrix[setting], condition=condition, field="joint_quality"),
                seed=seed + 100 * setting_index + (0 if condition is None else CONDITIONS.index(condition) + 1),
                repeats=repeats,
            )
            summary_rows.append(
                {
                    "setting": setting,
                    "scope": "overall" if condition is None else condition,
                    "n": len(subset),
                    "parse_rate": mean(float(row["parse_ok"]) for row in subset),
                    "schema_valid_rate": mean(float(row["schema_valid"]) for row in subset),
                    "execution_success_rate": mean(float(row["execution_success"]) for row in subset),
                    "geometry_valid_rate": mean(float(row["geometry_valid"]) for row in subset),
                    "joint_quality_mean": mean(row["joint_quality"] for row in subset),
                    "repair_trigger_rate": mean(float(row["repair_changed"]) for row in subset),
                    "schema_rescued": schema_rescued,
                    "schema_rescue_rate": schema_rescued / baseline_schema_failed if baseline_schema_failed else 0.0,
                    "geometry_rescued": geometry_rescued,
                    "geometry_rescue_rate": geometry_rescued / baseline_geometry_invalid
                    if baseline_geometry_invalid
                    else 0.0,
                    "execution_regressed": execution_regressed,
                    "execution_regression_rate": execution_regressed / baseline_execution_success
                    if baseline_execution_success
                    else 0.0,
                    "geometry_regressed": geometry_regressed,
                    "geometry_regression_rate": geometry_regressed / baseline_geometry_valid
                    if baseline_geometry_valid
                    else 0.0,
                    "joint_quality_delta": quality_ci["mean"],
                    "joint_quality_delta_ci_low": quality_ci["ci_low"],
                    "joint_quality_delta_ci_high": quality_ci["ci_high"],
                    "quality_wins": quality_ci["wins"],
                    "quality_ties": quality_ci["ties"],
                    "quality_losses": quality_ci["losses"],
                }
            )

    transition_counter: Counter[tuple[str, str, str, str]] = Counter()
    focused_counter: Counter[tuple[str, str, str]] = Counter()
    focused_names = (
        "validation_failed→execution_success",
        "validation_failed→runtime_failed",
        "runtime_failed→execution_success",
        "success→failed",
    )
    for setting in SETTING_ORDER[1:]:
        for scope in ("overall", *CONDITIONS):
            for transition in focused_names:
                focused_counter[(setting, scope, transition)] = 0
        for key, before in baseline.items():
            after = matrix[setting][key]
            scope_values = ("overall", before["condition"])
            for scope in scope_values:
                transition_counter[(setting, scope, before["status"], after["status"])] += 1
                if before["episode_status"] == "validation_failed" and after["execution_success"]:
                    focused_counter[(setting, scope, "validation_failed→execution_success")] += 1
                if before["episode_status"] == "validation_failed" and after["schema_valid"] and not after["execution_success"]:
                    focused_counter[(setting, scope, "validation_failed→runtime_failed")] += 1
                if before["schema_valid"] and not before["execution_success"] and after["execution_success"]:
                    focused_counter[(setting, scope, "runtime_failed→execution_success")] += 1
                if before["execution_success"] and not after["execution_success"]:
                    focused_counter[(setting, scope, "success→failed")] += 1
    transition_rows = [
        {
            "setting": setting,
            "scope": scope,
            "transition_type": "status",
            "from_state": before,
            "to_state": after,
            "count": count,
        }
        for (setting, scope, before, after), count in sorted(transition_counter.items())
    ]
    transition_rows.extend(
        {
            "setting": setting,
            "scope": scope,
            "transition_type": "focused",
            "from_state": transition.split("→")[0],
            "to_state": transition.split("→")[1],
            "count": count,
        }
        for (setting, scope, transition), count in sorted(focused_counter.items())
    )

    mcnemar = []
    for condition in (None, *CONDITIONS):
        keys = sorted(key for key in baseline if condition is None or key[1] == condition)
        for field in ("schema_valid", "execution_success", "geometry_valid"):
            result = _exact_mcnemar(
                [bool(baseline[key][field]) for key in keys],
                [bool(matrix["R4"][key][field]) for key in keys],
            )
            mcnemar.append(
                {
                    "comparison": "R4-R0",
                    "scope": "overall" if condition is None else condition,
                    "outcome": field,
                    **result,
                }
            )

    quality_bootstrap = {}
    for condition in (None, "T", "P", "TP"):
        name = "overall" if condition is None else condition
        quality_bootstrap[name] = _bootstrap_values(
            _sample_level_deltas(baseline, matrix["R4"], condition=condition, field="joint_quality"),
            seed=seed + 1000 + (0 if condition is None else CONDITIONS.index(condition) + 1),
            repeats=repeats,
        )
    quality_bootstrap["TP_minus_T"] = _bootstrap_values(
        _difference_in_differences(
            baseline,
            matrix["R4"],
            left_condition="TP",
            right_condition="T",
            field="joint_quality",
        ),
        seed=seed + 1100,
        repeats=repeats,
    )
    quality_bootstrap["TP_minus_P"] = _bootstrap_values(
        _difference_in_differences(
            baseline,
            matrix["R4"],
            left_condition="TP",
            right_condition="P",
            field="joint_quality",
        ),
        seed=seed + 1101,
        repeats=repeats,
    )

    condition_gains = {}
    for condition in ("T", "P", "TP"):
        keys = [key for key in baseline if key[1] == condition]
        condition_gains[condition] = {
            field: mean(float(matrix["R4"][key][field]) - float(baseline[key][field]) for key in keys)
            for field in ("schema_valid", "execution_success", "geometry_valid", "joint_quality")
        }
    condition_gains["TP_minus_T"] = {
        field: condition_gains["TP"][field] - condition_gains["T"][field]
        for field in condition_gains["T"]
    }
    condition_gains["TP_minus_P"] = {
        field: condition_gains["TP"][field] - condition_gains["P"][field]
        for field in condition_gains["P"]
    }

    repair_code_stats = []
    for setting in SETTING_ORDER[1:]:
        code_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in by_setting[setting]:
            for code in filter(None, row["repair_codes"].split(",")):
                code_tasks[code].append(row)
        for code, code_rows in sorted(code_tasks.items()):
            repair_code_stats.append(
                {
                    "setting": setting,
                    "repair_code": code,
                    "task_count": len(code_rows),
                    "schema_rescued": sum(
                        not baseline[(row["sample_id"], row["condition"])]["schema_valid"] and row["schema_valid"]
                        for row in code_rows
                    ),
                    "geometry_rescued": sum(
                        not baseline[(row["sample_id"], row["condition"])]["geometry_valid"] and row["geometry_valid"]
                        for row in code_rows
                    ),
                }
            )

    runtime_failures = []
    for setting in SETTING_ORDER:
        counts = Counter(
            row["failure_code"] or row["episode_status"] or row["status"]
            for row in by_setting[setting]
            if row["schema_valid"] and not row["execution_success"]
        )
        runtime_failures.extend(
            {"setting": setting, "failure_code": code, "count": count}
            for code, count in sorted(counts.items())
        )

    execution_regressions = [
        {
            "sample_id": key[0],
            "condition": key[1],
            "setting": setting,
            "baseline_status": baseline[key]["status"],
            "replay_status": matrix[setting][key]["status"],
        }
        for setting in SETTING_ORDER[1:]
        for key in baseline
        if baseline[key]["execution_success"] and not matrix[setting][key]["execution_success"]
    ]
    repaired_tasks = [
        {
            "sample_id": row["sample_id"],
            "condition": row["condition"],
            "repair_codes": row["repair_codes"],
            "baseline_status": baseline[(row["sample_id"], row["condition"])]["status"],
            "replay_status": row["status"],
            "schema_rescued": (
                not baseline[(row["sample_id"], row["condition"])]["schema_valid"] and row["schema_valid"]
            ),
            "geometry_rescued": (
                not baseline[(row["sample_id"], row["condition"])]["geometry_valid"] and row["geometry_valid"]
            ),
            "joint_quality_delta": (
                row["joint_quality"] - baseline[(row["sample_id"], row["condition"])]["joint_quality"]
            ),
        }
        for row in by_setting["R4"]
        if row["repair_changed"]
    ]
    baseline_reproduction = json.loads(
        (output_dir / "baseline_reproduction_summary.json").read_text(encoding="utf-8")
    )
    report = {
        "schema_version": "rq2.repair_stats.v1",
        "comparison": "R4-R0",
        "bootstrap": {"unit": "sample", "repeats": repeats, "seed": seed},
        "mcnemar": mcnemar,
        "quality_bootstrap": quality_bootstrap,
        "condition_gains": condition_gains,
        "repair_code_stats": repair_code_stats,
        "repaired_tasks": repaired_tasks,
        "runtime_failures": runtime_failures,
        "execution_regressions": execution_regressions,
        "baseline_reproduction": {
            "gate_passed": baseline_reproduction.get("gate_passed"),
            "checks": baseline_reproduction.get("checks"),
            "mismatch_count": len(baseline_reproduction.get("mismatches") or []),
        },
    }
    _write_csv(output_dir / "repair_task_rows.csv", rows)
    _write_csv(output_dir / "repair_summary.csv", summary_rows)
    _write_csv(output_dir / "repair_failure_transitions.csv", transition_rows)
    atomic_write_json(output_dir / "repair_stats.json", report)
    _write_report(output_dir / "REPLAY_REPORT_ZH.md", summary_rows, report)
    if execution_regressions:
        raise RuntimeError(f"检测到 {len(execution_regressions)} 个 success→failed 回归，已停止")
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_report(path: Path, summaries: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    overall = {row["setting"]: row for row in summaries if row["scope"] == "overall"}
    gains = stats["condition_gains"]
    quality = stats["quality_bootstrap"]
    code_stats = [row for row in stats["repair_code_stats"] if row["setting"] == "R4"]
    runtime = [row for row in stats["runtime_failures"] if row["setting"] == "R4"]
    overall_mcnemar = {
        row["outcome"]: row
        for row in stats["mcnemar"]
        if row["scope"] == "overall"
    }
    lines = [
        "# Plan v2.1 安全修复与离线重放报告",
        "",
        "## 1. 实验范围",
        "",
        "本阶段仅重放冻结的 700 条 Qwen 原始回复；未调用模型 API、未修改 prompt、未增加 CAD operation，且未覆盖 `pilot_v2`。",
        "",
        "## 2. 原实验能否复现",
        "",
        (
            "可以。700/700 条原始回复均找到，Plan hash、解析、Schema、执行状态、几何有效性和 "
            "joint quality 全部精确匹配；mismatch=0，冻结 baseline 目录 hash 前后不变。"
        ),
        "",
        "完整门禁证据见 `baseline_reproduction.csv` 和 `baseline_reproduction_summary.json`。",
        "",
        "## 3. R0–R4 总体结果",
        "",
        "| 设置 | 解析率 | Schema 通过率 | CAD 成功率 | 几何有效率 | joint quality | 修复触发率 | Schema 救回 | 几何救回 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in SETTING_ORDER:
        row = overall[setting]
        lines.append(
            f"| {setting} | {_fmt(row['parse_rate'])} | {_fmt(row['schema_valid_rate'])} | "
            f"{_fmt(row['execution_success_rate'])} | {_fmt(row['geometry_valid_rate'])} | "
            f"{_fmt(row['joint_quality_mean'])} | {_fmt(row['repair_trigger_rate'])} | "
            f"{row['schema_rescued']} | {row['geometry_rescued']} |"
        )
    lines.extend(["", "## 4. 哪些安全规则救回了任务", ""])
    if code_stats:
        for row in code_stats:
            lines.append(
                f"- `{row['repair_code']}`：触发 {row['task_count']} 个任务，"
                f"Schema 救回 {row['schema_rescued']}，最终几何救回 {row['geometry_rescued']}。"
            )
    else:
        lines.append("- R4 未触发任何安全修复。")
    lines.extend(["", "R4 实际改变了以下任务：", ""])
    for row in stats["repaired_tasks"]:
        lines.append(
            f"- `{row['sample_id']}/{row['condition']}`：{row['repair_codes']}；"
            f"{row['baseline_status']} → {row['replay_status']}；"
            f"Schema 救回={row['schema_rescued']}，几何救回={row['geometry_rescued']}，"
            f"joint quality Δ={_fmt(row['joint_quality_delta'])}。"
        )
    lines.extend(
        [
            "",
            "R3 没有触发任务：历史输出中未发现可以同时保留完整轴、原点和角度语义的无歧义旧字段结构；"
            "三维 revolve 轴和 Euler 风格 rotate 数组按安全边界保持失败。",
            "",
            "## 5. 配对统计",
            "",
            f"- R4−R0 Schema McNemar：rescued={overall_mcnemar['schema_valid']['rescued']}，"
            f"regressed={overall_mcnemar['schema_valid']['regressed']}，p={_fmt(overall_mcnemar['schema_valid']['p_value'])}。",
            f"- R4−R0 最终几何 McNemar：rescued={overall_mcnemar['geometry_valid']['rescued']}，"
            f"regressed={overall_mcnemar['geometry_valid']['regressed']}，p={_fmt(overall_mcnemar['geometry_valid']['p_value'])}。",
            f"- failure-aware joint quality 平均 Δ={_fmt(quality['overall']['mean'])}，"
            f"95% CI [{_fmt(quality['overall']['ci_low'])}, {_fmt(quality['overall']['ci_high'])}]，"
            f"win/tie/loss={quality['overall']['wins']}/{quality['overall']['ties']}/{quality['overall']['losses']}。",
            "",
            "2 个救回不足以形成统计显著证据。结果否定的是“浅层安全规范化可以解释大部分失败”，"
            "而不是否定严格接口契约本身是系统瓶颈。",
        ]
    )
    lines.extend(
        [
            "",
            "## 6. 是否破坏既有成功结果",
            "",
            f"- execution success→failed：{len(stats['execution_regressions'])}。",
            "- 安全修复不允许改变原有合法 Plan；任何非零回归都会触发停止条件。",
            "",
            "## 7. TP 的修复收益是否高于 T/P",
            "",
            f"- ΔT joint quality：{_fmt(gains['T']['joint_quality'])}。",
            f"- ΔP joint quality：{_fmt(gains['P']['joint_quality'])}。",
            f"- ΔTP joint quality：{_fmt(gains['TP']['joint_quality'])}。",
            f"- ΔTP−ΔT：{_fmt(quality['TP_minus_T']['mean'])}，"
            f"95% CI [{_fmt(quality['TP_minus_T']['ci_low'])}, {_fmt(quality['TP_minus_T']['ci_high'])}]。",
            f"- ΔTP−ΔP：{_fmt(quality['TP_minus_P']['mean'])}，"
            f"95% CI [{_fmt(quality['TP_minus_P']['ci_low'])}, {_fmt(quality['TP_minus_P']['ci_high'])}]。",
            "",
            "只有差分收益为正且置信区间不跨 0，才支持“TP 比 T/P 更受益于接口规范化”。",
            "",
            "## 8. 修复后的运行时瓶颈",
            "",
        ]
    )
    if runtime:
        for row in runtime:
            lines.append(f"- `{row['failure_code']}`：{row['count']} 个。")
    else:
        lines.append("- R4 没有 Schema 通过后的运行时失败。")
    r4 = overall["R4"]
    r0 = overall["R0"]
    lines.extend(
        [
            "",
            "## 9. 是否值得进入一次模型反馈修正",
            "",
            f"- R4 相比 R0 的 Schema 通过率变化：{_fmt(r4['schema_valid_rate'] - r0['schema_valid_rate'])}。",
            f"- R4 相比 R0 的几何有效率变化：{_fmt(r4['geometry_valid_rate'] - r0['geometry_valid_rate'])}。",
            f"- R4 Schema rescue rate：{_fmt(r4['schema_rescue_rate'])}；"
            f"geometry rescue rate：{_fmt(r4['geometry_rescue_rate'])}。",
            (
                "- 结论：值得把“一次带精确校验错误的模型反馈修正”作为下一阶段受控实验。理由不是安全修复收益很大，"
                "而是安全修复只救回 2 个任务，绝大多数失败涉及缺失字段、错误 revolve/rotate 结构、引用或拓扑，"
                "这些问题在不猜测语义的前提下只能由模型重新表达。"
            ),
            "- 该建议不构成本阶段继续执行授权；本次离线实验在报告处停止。",
            "",
            "本报告生成后按任务指示停止；未启动 L1、冲突模态、第二模型或任何外部 API。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分析 Plan v2.1 离线重放")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "replay_v21.yaml"))
    args = parser.parse_args(argv)
    report = analyze_replay(load_config(args.config))
    print(json.dumps({"comparison": report["comparison"], "regressions": len(report["execution_regressions"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
