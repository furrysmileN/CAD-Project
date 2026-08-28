from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np

from .common import atomic_write_json, load_config, project_path, read_jsonl
from .conditions import CONDITIONS


def complement_gain(scores: dict[str, float], condition: str) -> float:
    components = {"TI": ("T", "I"), "TP": ("T", "P"), "IP": ("I", "P"), "TIP": ("T", "I", "P")}
    if condition not in components:
        raise ValueError(f"{condition} 不是组合条件")
    return float(scores[condition] - max(scores[name] for name in components[condition]))


def bootstrap_ci(values: list[float], seed: int, repeats: int = 5000) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "low": None, "high": None, "n": 0}
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repeats, len(array)))
    estimates = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "low": float(np.quantile(estimates, 0.025)),
        "high": float(np.quantile(estimates, 0.975)),
        "n": len(values),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def _wilcoxon(a: list[float], b: list[float]) -> tuple[float | None, float | None, str]:
    if len(a) != len(b):
        return None, None, "length_mismatch"
    if not a:
        return None, None, "no_pairs"
    try:
        from scipy.stats import wilcoxon

        differences = np.asarray(a) - np.asarray(b)
        if not np.any(np.abs(differences) > 1e-15):
            return 0.0, 1.0, "all_differences_zero"
        result = wilcoxon(a, b, alternative="two-sided", zero_method="pratt")
        return float(result.statistic), float(result.pvalue), "ok"
    except Exception as exc:
        return None, None, f"degraded:{type(exc).__name__}:{str(exc)[:200]}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(output_dir: Path) -> list[dict[str, Any]]:
    manifest = {row["sample_id"]: row for row in read_jsonl(output_dir / "manifest.jsonl")}
    rows = []
    for path in sorted((output_dir / "state").glob("*/*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("status") in {"dry_run", "running"}:
            continue
        sample = manifest.get(state.get("sample_id"))
        if sample is None or state.get("condition") not in CONDITIONS:
            continue
        geometry = state.get("geometry") or {}
        voxel = geometry.get("voxel_iou") or {}
        fscore_shape = geometry.get("fscore_shape") or {}
        fscore_common = geometry.get("fscore_common") or {}
        shape_voxel = geometry.get("shape_voxel_iou") or {}
        parse = state.get("parse") or {}
        episode_response = ((state.get("episode") or {}).get("response") or {})
        validation = episode_response.get("validation") or {}
        failure = episode_response.get("failure") or {}
        api = state.get("api") or {}
        usage = api.get("usage") or {}
        rows.append(
            {
                "sample_id": state["sample_id"],
                "condition": state["condition"],
                "status": state.get("status"),
                "valid": bool(geometry.get("valid", False)),
                "joint_quality": float(geometry.get("joint_quality") or 0.0),
                "shape_only_cd": geometry.get("shape_only_cd"),
                "common_frame_cd": geometry.get("common_frame_cd"),
                "voxel_iou": voxel.get("value"),
                "f1_shape": fscore_shape.get("f1"),
                "f1_common": fscore_common.get("f1"),
                "shape_iou": shape_voxel.get("value"),
                "family": sample["family"],
                "difficulty": sample["difficulty"],
                "complexity": sample["complexity"],
                "complexity_bin": sample["complexity_bin"],
                "parse_ok": bool(parse.get("ok", False)),
                "repair_kind": (parse.get("repair") or {}).get("kind"),
                "schema_valid": bool(validation.get("valid", False)),
                "episode_status": episode_response.get("status"),
                "failure_code": failure.get("code"),
                "validation_issue_codes": ",".join(
                    sorted({str(item.get("code")) for item in validation.get("issues", []) if item.get("code")})
                ),
                "warning_codes": ",".join(
                    sorted({str(item.get("code")) for item in episode_response.get("warnings", []) if item.get("code")})
                ),
                "total_tokens": usage.get("total_tokens"),
                "api_latency_sec": api.get("latency_sec"),
            }
        )
    return rows


def analyze(config: dict[str, Any], output: Path | None = None, bootstrap_repeats: int = 5000) -> dict[str, Any]:
    experiment_dir = project_path(config["paths"]["output_dir"])
    report_dir = output or experiment_dir / "analysis"
    rows = _load_rows(experiment_dir)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    summaries = []
    pipeline_summaries = []
    usage_summaries = []
    for condition in config["conditions"]:
        subset = by_condition.get(condition, [])
        qualities = [row["joint_quality"] for row in subset]
        ci = bootstrap_ci(qualities, int(config["seed"]) + CONDITIONS.index(condition), bootstrap_repeats)
        summaries.append(
            {
                "condition": condition,
                "n": len(subset),
                "valid_rate": mean([float(row["valid"]) for row in subset]) if subset else None,
                "joint_quality_mean": ci["mean"],
                "joint_quality_ci_low": ci["low"],
                "joint_quality_ci_high": ci["high"],
                "shape_only_cd_mean": mean([row["shape_only_cd"] for row in subset if row["shape_only_cd"] is not None])
                if any(row["shape_only_cd"] is not None for row in subset) else None,
                "common_frame_cd_mean": mean([row["common_frame_cd"] for row in subset if row["common_frame_cd"] is not None])
                if any(row["common_frame_cd"] is not None for row in subset) else None,
                "voxel_iou_mean": mean([row["voxel_iou"] for row in subset if row["voxel_iou"] is not None])
                if any(row["voxel_iou"] is not None for row in subset) else None,
                "f1_shape_mean": mean([row["f1_shape"] for row in subset if row["f1_shape"] is not None])
                if any(row["f1_shape"] is not None for row in subset) else None,
                "shape_iou_mean": mean([row["shape_iou"] for row in subset if row["shape_iou"] is not None])
                if any(row["shape_iou"] is not None for row in subset) else None,
                "invalid_ratio": 1.0 - mean([float(row["valid"]) for row in subset]) if subset else None,
            }
        )
        pipeline_summaries.append(
            {
                "condition": condition,
                "n": len(subset),
                "parse_rate": mean([float(row["parse_ok"]) for row in subset]) if subset else None,
                "schema_valid_rate": mean([float(row["schema_valid"]) for row in subset]) if subset else None,
                "execution_success_rate": mean([float(row["status"] == "completed") for row in subset])
                if subset else None,
                "parse_failed": sum(row["status"] == "parse_failed" for row in subset),
                "validation_failed": sum(row["episode_status"] == "validation_failed" for row in subset),
                "runtime_failed": sum(
                    row["episode_status"] in {"failed", "execution_timeout", "execution_failed"} for row in subset
                ),
            }
        )
        token_values = [float(row["total_tokens"]) for row in subset if row["total_tokens"] is not None]
        latency_values = [float(row["api_latency_sec"]) for row in subset if row["api_latency_sec"] is not None]
        usage_summaries.append(
            {
                "condition": condition,
                "n_with_usage": len(token_values),
                "total_tokens": int(sum(token_values)),
                "mean_tokens": mean(token_values) if token_values else None,
                "mean_api_latency_sec": mean(latency_values) if latency_values else None,
            }
        )

    failure_groups: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        if row["status"] == "parse_failed":
            failure_groups[(row["condition"], "parse", "parse_failed")] += 1
        elif row["status"] == "episode_failed":
            code = row["failure_code"] or row["validation_issue_codes"] or "unknown_episode_failure"
            stage = "validation" if row["episode_status"] == "validation_failed" else "runtime"
            failure_groups[(row["condition"], stage, code)] += 1
    failure_summary = [
        {"condition": condition, "stage": stage, "code": code, "count": count}
        for (condition, stage, code), count in sorted(failure_groups.items())
    ]

    score_matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        score_matrix[row["sample_id"]][row["condition"]] = row["joint_quality"]
    gains = []
    for index, condition in enumerate(("TI", "TP", "IP", "TIP")):
        required = set({"TI": "TI", "TP": "TP", "IP": "IP", "TIP": "TIP"}[condition])
        values = []
        for scores in score_matrix.values():
            needed = required | {condition}
            if needed <= set(scores):
                values.append(complement_gain(scores, condition))
        gains.append({"condition": condition, **bootstrap_ci(values, int(config["seed"]) + 100 + index, bootstrap_repeats)})

    pairwise = []
    for left, right in itertools.combinations(config["conditions"], 2):
        paired = [(scores[left], scores[right]) for scores in score_matrix.values() if left in scores and right in scores]
        statistic, p_value, status = _wilcoxon([item[0] for item in paired], [item[1] for item in paired])
        pairwise.append({"left": left, "right": right, "n": len(paired), "statistic": statistic, "p_value": p_value, "status": status})
    valid_indices = [index for index, item in enumerate(pairwise) if item["p_value"] is not None]
    adjusted = holm_adjust([pairwise[index]["p_value"] for index in valid_indices])
    for index, value in zip(valid_indices, adjusted):
        pairwise[index]["p_holm"] = value

    strata_rows = []
    for field in ("family", "difficulty", "complexity_bin"):
        groups: dict[tuple[Any, str], list[float]] = defaultdict(list)
        for row in rows:
            groups[(row[field], row["condition"])].append(row["joint_quality"])
        for (value, condition), values in sorted(groups.items(), key=lambda item: (str(item[0][0]), item[0][1])):
            strata_rows.append(
                {"stratum": field, "value": value, "condition": condition, "n": len(values), "joint_quality_mean": mean(values)}
            )

    report = {
        "schema_version": "rq2.analysis.v1",
        "condition_summary": summaries,
        "pipeline_summary": pipeline_summaries,
        "failure_summary": failure_summary,
        "usage_summary": usage_summaries,
        "complement_gains": gains,
        "paired_wilcoxon_holm": pairwise,
        "stratified_summary": strata_rows,
        "bootstrap": {"unit": "sample", "repeats": bootstrap_repeats, "ci": 0.95, "seed": int(config["seed"])},
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_dir / "analysis.json", report)
    _write_csv(report_dir / "condition_summary.csv", summaries)
    _write_csv(report_dir / "pipeline_summary.csv", pipeline_summaries)
    _write_csv(report_dir / "failure_summary.csv", failure_summary)
    _write_csv(report_dir / "usage_summary.csv", usage_summaries)
    _write_csv(report_dir / "complement_gains.csv", gains)
    _write_csv(report_dir / "paired_wilcoxon_holm.csv", pairwise)
    _write_csv(report_dir / "stratified_summary.csv", strata_rows)
    _write_csv(report_dir / "task_rows.csv", rows)
    lines = [
        "# RQ2 多模态 Harness 实验分析",
        "",
        "## 条件汇总",
        "",
        "| 条件 | n | 有效率 | IR 无效率 | 联合质量均值 | 95% CI | shape-only CD | common-frame CD | 体素 IoU | shape IoU | F1@0.01 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        fmt: Callable[[Any], str] = lambda value: "NA" if value is None else f"{value:.4f}"
        lines.append(
            f"| {item['condition']} | {item['n']} | {fmt(item['valid_rate'])} | {fmt(item['invalid_ratio'])} | {fmt(item['joint_quality_mean'])} | "
            f"[{fmt(item['joint_quality_ci_low'])}, {fmt(item['joint_quality_ci_high'])}] | "
            f"{fmt(item['shape_only_cd_mean'])} | {fmt(item['common_frame_cd_mean'])} | {fmt(item['voxel_iou_mean'])} | "
            f"{fmt(item['shape_iou_mean'])} | {fmt(item['f1_shape_mean'])} |"
        )
    lines.extend(["", "## 互补增益", "", "增益定义为组合条件减去同一样本上最佳组成单模态；TIP 与 T/I/P 中最佳者比较。", ""])
    for item in gains:
        lines.append(f"- {item['condition']}: {item['mean'] if item['mean'] is not None else 'NA'}，95% CI [{item['low']}, {item['high']}]，n={item['n']}")
    lines.extend(
        [
            "",
            "## 执行流水线",
            "",
            "| 条件 | 解析率 | Schema 通过率 | CAD 执行成功率 | 解析失败 | 校验失败 | 运行时失败 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in pipeline_summaries:
        lines.append(
            f"| {item['condition']} | {fmt(item['parse_rate'])} | {fmt(item['schema_valid_rate'])} | "
            f"{fmt(item['execution_success_rate'])} | {item['parse_failed']} | "
            f"{item['validation_failed']} | {item['runtime_failed']} |"
        )
    lines.extend(["", "## API 用量", ""])
    for item in usage_summaries:
        lines.append(
            f"- {item['condition']}: total_tokens={item['total_tokens']}，"
            f"mean_tokens={fmt(item['mean_tokens'])}，mean_latency={fmt(item['mean_api_latency_sec'])} 秒"
        )
    lines.extend(
        [
            "",
            "失败明细见 `failure_summary.csv`；逐任务的修复类型、校验问题、运行失败码与警告见 `task_rows.csv`。",
        ]
    )
    lines.extend(
        [
            "",
            "## 统计说明",
            "",
            f"- 以样本为 bootstrap 单位，重复 {bootstrap_repeats} 次，seed={config['seed']}。",
            "- 条件间使用配对 Wilcoxon signed-rank 检验，并对全部条件对执行 Holm 校正。",
            "- 无效/缺失几何的联合质量固定为 0；体素依赖降级不会把无效样本误记为有效。",
            "- IR 无效率 = 1 − 几何有效率，即未形成有效可评分几何的任务比例。",
            "- F1@0.01 与 shape IoU 在各自归一化（shape-only）空间计算，对齐常见 text-to-CAD 文献口径。",
            "- family、difficulty、complexity_bin 的完整分层结果见 `stratified_summary.csv`。",
        ]
    )
    (report_dir / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分析 RQ2 多模态 Harness 实验")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"))
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    report = analyze(config, Path(args.output).resolve() if args.output else None, args.bootstrap_repeats)
    print(json.dumps({"conditions": len(report["condition_summary"]), "output": args.output or "default"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
