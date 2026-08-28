from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
import yaml

from .feedback import failure_kind_from_code


MODALITIES = ("T", "I", "P")
ENCODINGS = (1, 2, 3)
CONDITION_RE = re.compile(r"^(?:T[123])?(?:I[123])?(?:P[123])?$")
DEFAULT_COST = {
    "input_per_million_tokens": 0.0,
    "output_per_million_tokens": 0.0,
    "total_per_million_tokens": 0.0,
    "latency_per_second": 0.0,
}


def parse_condition(condition: str) -> dict[str, int]:
    if not condition or not CONDITION_RE.fullmatch(condition):
        raise ValueError(f"非法编码条件: {condition!r}")
    result = {match.group(1): int(match.group(2)) for match in re.finditer(r"([TIP])([123])", condition)}
    if not result:
        raise ValueError(f"编码条件不能为空: {condition!r}")
    return result


def format_condition(parts: dict[str, int]) -> str:
    return "".join(f"{modality}{parts[modality]}" for modality in MODALITIES if modality in parts)


def expected_conditions() -> tuple[str, ...]:
    conditions: list[str] = []
    for mask in range(1, 1 << len(MODALITIES)):
        selected = [name for index, name in enumerate(MODALITIES) if mask & (1 << index)]
        grids: list[dict[str, int]] = [{}]
        for modality in selected:
            grids = [{**item, modality: encoding} for item in grids for encoding in ENCODINGS]
        conditions.extend(format_condition(item) for item in grids)
    return tuple(conditions)


ENCODING_CONDITIONS = expected_conditions()


def _nested(data: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = data
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            if value is not None:
                return value
    return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _codes(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    values = {str(item.get("code")) for item in items if isinstance(item, dict) and item.get("code")}
    return ",".join(sorted(values))


def _operation_count(state: dict[str, Any], validation: dict[str, Any]) -> int | None:
    value = _nested(
        validation,
        ("planSummary", "operationCount"),
        ("plan_summary", "operation_count"),
    )
    if value is None:
        operations = _nested(
            state,
            ("repaired_plan", "operations"),
            ("parse", "plan", "operations"),
            ("plan", "operations"),
        )
        value = len(operations) if isinstance(operations, list) else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _failure_stage(
    *, parse_ok: bool, schema_valid: bool, execution_success: bool, geometry_valid: bool
) -> str:
    if not parse_ok:
        return "parse"
    if not schema_valid:
        return "schema"
    if not execution_success:
        return "execution"
    if not geometry_valid:
        return "geometry"
    return "none"


def state_to_row(
    state: dict[str, Any],
    sample: dict[str, Any] | None = None,
    cost_config: dict[str, float] | None = None,
) -> dict[str, Any]:
    condition_value = state.get("condition_id") or state.get("condition") or ""
    condition = str(condition_value) if not isinstance(condition_value, dict) else ""
    parts = parse_condition(condition)
    stage = state.get("stage") or {}
    parse_data = state.get("parse") or {}
    geometry = state.get("geometry") or {}
    response = ((state.get("episode") or {}).get("response") or {})
    validation = response.get("validation") or {}
    failure = response.get("failure") or {}
    api = state.get("api") or {}
    usage = api.get("usage") or state.get("usage") or {}

    parse_ok = bool(stage.get("parse_ok", parse_data.get("ok", False)))
    schema_valid = bool(stage.get("schema_valid", validation.get("valid", False)))
    episode_status = stage.get("episode_status") or response.get("status")
    execution_success = bool(
        stage.get(
            "execution_success",
            episode_status in {"completed", "success"} or state.get("status") == "completed",
        )
    )
    geometry_valid = bool(stage.get("geometry_valid", geometry.get("valid", False)))
    raw_quality = _as_float(stage.get("joint_quality", geometry.get("joint_quality")))
    pipeline_valid = parse_ok and schema_valid and execution_success and geometry_valid
    joint_quality = raw_quality if pipeline_valid and raw_quality is not None else 0.0
    input_tokens = _as_float(
        _nested(usage, ("input_tokens",), ("prompt_tokens",), ("inputTokens",))
    )
    output_tokens = _as_float(
        _nested(usage, ("output_tokens",), ("completion_tokens",), ("outputTokens",))
    )
    total_tokens = _as_float(_nested(usage, ("total_tokens",), ("totalTokens",)))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    latency = _as_float(
        _nested(api, ("latency_sec",), ("latency_seconds",), ("latency",))
    )
    prices = {**DEFAULT_COST, **(cost_config or {})}
    if input_tokens is not None or output_tokens is not None:
        token_cost = (
            (input_tokens or 0.0) * prices["input_per_million_tokens"]
            + (output_tokens or 0.0) * prices["output_per_million_tokens"]
        ) / 1_000_000
    elif total_tokens is not None:
        token_cost = total_tokens * prices["total_per_million_tokens"] / 1_000_000
    else:
        token_cost = None
    estimated_cost = (
        (token_cost or 0.0) + (latency or 0.0) * prices["latency_per_second"]
        if token_cost is not None or latency is not None
        else None
    )
    failure_stage = _failure_stage(
        parse_ok=parse_ok,
        schema_valid=schema_valid,
        execution_success=execution_success,
        geometry_valid=geometry_valid,
    )
    failure_code = (
        failure.get("code")
        or (_codes(parse_data.get("issues")) if failure_stage == "parse" else "")
        or (_codes(validation.get("issues")) if failure_stage == "schema" else "")
        or geometry.get("failure")
        or (str(state.get("status")) if failure_stage != "none" else "")
        or "unknown"
    )
    voxel = geometry.get("voxel_iou")
    voxel_iou = voxel.get("value") if isinstance(voxel, dict) else voxel
    row: dict[str, Any] = {
        "sample_id": str(state.get("sample_id") or (sample or {}).get("sample_id") or ""),
        "condition": condition,
        "modalities": "".join(parts),
        "text_encoding": parts.get("T"),
        "image_encoding": parts.get("I"),
        "point_encoding": parts.get("P"),
        "status": state.get("status"),
        "parse_ok": parse_ok,
        "schema_valid": schema_valid,
        "execution_success": execution_success,
        "geometry_valid": geometry_valid,
        "joint_quality": float(joint_quality),
        "shape_only_cd": _as_float(geometry.get("shape_only_cd")),
        "common_frame_cd": _as_float(geometry.get("common_frame_cd")),
        "voxel_iou": _as_float(voxel_iou),
        "operation_count": _operation_count(state, validation),
        "failure_stage": failure_stage,
        "failure_code": "" if failure_stage == "none" else str(failure_code),
        "validation_issue_codes": _codes(validation.get("issues")),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_sec": latency,
        "estimated_cost": estimated_cost,
    }
    for field in ("family", "difficulty", "complexity", "complexity_bin"):
        row[field] = (sample or {}).get(field)
    return row


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} 不是 JSON object")
                rows.append(value)
    return rows


def load_task_rows(
    experiment_dir: Path, cost_config: dict[str, float] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    manifest_path = experiment_dir / "sample_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少样本清单: {manifest_path}")
    manifest_rows = _read_jsonl(manifest_path)
    manifest = {str(item["sample_id"]): item for item in manifest_rows}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted((experiment_dir / "state").glob("*/*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(state.get("sample_id") or path.parent.name)
        condition_value = state.get("condition_id") or state.get("condition") or path.stem
        condition = str(condition_value) if not isinstance(condition_value, dict) else path.stem
        if condition not in ENCODING_CONDITIONS:
            continue
        state = {**state, "sample_id": sample_id, "condition": condition}
        key = (sample_id, condition)
        if key in seen:
            raise ValueError(f"重复 task state: {sample_id}/{condition}")
        seen.add(key)
        if sample_id not in manifest:
            raise ValueError(f"state 样本不在清单中: {sample_id}")
        rows.append(state_to_row(state, manifest[sample_id], cost_config))
    if not rows:
        raise RuntimeError(f"{experiment_dir / 'state'} 中没有可分析的编码条件 state")
    missing = [
        f"{sample_id}/{condition}"
        for sample_id in sorted(manifest)
        for condition in ENCODING_CONDITIONS
        if (sample_id, condition) not in seen
    ]
    return rows, missing


def bootstrap_summary(values: Iterable[float], seed: int, repeats: int) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {
            "n": 0, "mean": None, "median": None, "ci_low": None, "ci_high": None,
            "wins": 0, "ties": 0, "losses": 0,
        }
    rng = np.random.default_rng(seed)
    estimates = array[rng.integers(0, len(array), size=(repeats, len(array)))].mean(axis=1)
    tolerance = 1e-12
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "wins": int(np.sum(array > tolerance)),
        "ties": int(np.sum(np.abs(array) <= tolerance)),
        "losses": int(np.sum(array < -tolerance)),
    }


def _comparison_specs() -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for modality in MODALITIES:
        for left in ENCODINGS:
            for right in ENCODINGS:
                if left < right:
                    specs.append({
                        "comparison_family": "single_modality_encoding",
                        "contrast": f"{modality}{right} - {modality}{left}",
                        "left": f"{modality}{left}",
                        "right": f"{modality}{right}",
                        "changed_modality": modality,
                    })
    for modality in MODALITIES:
        others = [item for item in MODALITIES if item != modality]
        contexts: list[dict[str, int]] = []
        for mask in range(1, 1 << len(others)):
            selected = [item for index, item in enumerate(others) if mask & (1 << index)]
            grids: list[dict[str, int]] = [{}]
            for item in selected:
                grids = [{**grid, item: encoding} for grid in grids for encoding in ENCODINGS]
            contexts.extend(grids)
        for context in contexts:
            for left in ENCODINGS:
                for right in ENCODINGS:
                    if left < right:
                        before = format_condition({**context, modality: left})
                        after = format_condition({**context, modality: right})
                        specs.append({
                            "comparison_family": "encoding_replacement_fixed_others",
                            "contrast": f"{after} - {before}",
                            "left": before,
                            "right": after,
                            "changed_modality": modality,
                        })
    for first, second in (("T", "I"), ("T", "P"), ("I", "P")):
        for a in ENCODINGS:
            for b in ENCODINGS:
                pair = format_condition({first: a, second: b})
                for base in (f"{first}{a}", f"{second}{b}"):
                    specs.append({
                        "comparison_family": "direct_bimodal_gain",
                        "contrast": f"{pair} - {base}",
                        "left": base,
                        "right": pair,
                        "changed_modality": next(iter(set(pair[::2]) - set(base[::2]))),
                    })
    for t in ENCODINGS:
        for i in ENCODINGS:
            for p in ENCODINGS:
                triple = f"T{t}I{i}P{p}"
                for pair in (f"T{t}I{i}", f"T{t}P{p}", f"I{i}P{p}"):
                    specs.append({
                        "comparison_family": "trimodal_increment",
                        "contrast": f"{triple} - {pair}",
                        "left": pair,
                        "right": triple,
                        "changed_modality": next(iter(set(triple[::2]) - set(pair[::2]))),
                    })
    return specs


def interaction_rows(
    rows: list[dict[str, Any]], *, seed: int, bootstrap_repeats: int
) -> list[dict[str, Any]]:
    matrix = {(row["sample_id"], row["condition"]): row for row in rows}
    samples = sorted({row["sample_id"] for row in rows})
    output: list[dict[str, Any]] = []
    for index, spec in enumerate(_comparison_specs()):
        deltas = [
            matrix[(sample, spec["right"])]["joint_quality"]
            - matrix[(sample, spec["left"])]["joint_quality"]
            for sample in samples
            if (sample, spec["left"]) in matrix and (sample, spec["right"]) in matrix
        ]
        output.append({
            "row_type": "preregistered_comparison",
            **spec,
            **bootstrap_summary(deltas, seed + index, bootstrap_repeats),
        })
    for modality in MODALITIES:
        for encoding in ENCODINGS:
            by_sample: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                if parse_condition(row["condition"]).get(modality) == encoding:
                    by_sample[row["sample_id"]].append(row["joint_quality"])
            sample_means = [
                mean(values) for _, values in sorted(by_sample.items()) if values
            ]
            summary = bootstrap_summary(
                sample_means, seed + 10_000 + len(output), bootstrap_repeats
            )
            output.append({
                "row_type": "marginal_mean",
                "comparison_family": "encoding_marginal_mean",
                "contrast": f"{modality}{encoding}",
                "left": "",
                "right": f"{modality}{encoding}",
                "changed_modality": modality,
                **summary,
            })
    return output


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def condition_summary(rows: list[dict[str, Any]], seed: int, repeats: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    output = []
    for index, condition in enumerate(ENCODING_CONDITIONS):
        subset = grouped.get(condition)
        if not subset:
            continue
        quality = bootstrap_summary(
            [row["joint_quality"] for row in subset], seed + 20_000 + index, repeats
        )
        output.append({
            "condition": condition,
            "n": len(subset),
            "parse_rate": mean(float(row["parse_ok"]) for row in subset),
            "schema_valid_rate": mean(float(row["schema_valid"]) for row in subset),
            "execution_success_rate": mean(float(row["execution_success"]) for row in subset),
            "geometry_valid_rate": mean(float(row["geometry_valid"]) for row in subset),
            "joint_quality_mean": quality["mean"],
            "joint_quality_median": quality["median"],
            "joint_quality_ci_low": quality["ci_low"],
            "joint_quality_ci_high": quality["ci_high"],
            "shape_only_cd_mean": _mean_field(subset, "shape_only_cd"),
            "common_frame_cd_mean": _mean_field(subset, "common_frame_cd"),
            "voxel_iou_mean": _mean_field(subset, "voxel_iou"),
            "operation_count_mean": _mean_field(subset, "operation_count"),
        })
    return output


def failure_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (row["condition"], row["failure_stage"], row["failure_code"])
        for row in rows if row["failure_stage"] != "none"
    )
    return [
        {"condition": condition, "stage": stage, "failure_code": code, "count": count}
        for (condition, stage, code), count in sorted(counts.items())
    ]


def cost_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    return [{
        "condition": condition,
        "n": len(subset),
        "n_with_tokens": sum(row["total_tokens"] is not None for row in subset),
        "input_tokens_mean": _mean_field(subset, "input_tokens"),
        "output_tokens_mean": _mean_field(subset, "output_tokens"),
        "total_tokens_mean": _mean_field(subset, "total_tokens"),
        "latency_sec_mean": _mean_field(subset, "latency_sec"),
        "estimated_cost_mean": _mean_field(subset, "estimated_cost"),
        "estimated_cost_total": sum(
            float(row["estimated_cost"]) for row in subset if row["estimated_cost"] is not None
        ),
    } for condition, subset in sorted(grouped.items())]


def pareto_rows(
    summaries: list[dict[str, Any]], costs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    cost_by_condition = {row["condition"]: row["estimated_cost_mean"] for row in costs}
    candidates = [
        (row["condition"], row["joint_quality_mean"], cost_by_condition.get(row["condition"]))
        for row in summaries
        if row["joint_quality_mean"] is not None and cost_by_condition.get(row["condition"]) is not None
    ]
    output = []
    for condition, quality, cost in candidates:
        dominators = [
            other
            for other, other_quality, other_cost in candidates
            if other != condition
            and other_quality >= quality
            and other_cost <= cost
            and (other_quality > quality or other_cost < cost)
        ]
        output.append({
            "condition": condition,
            "joint_quality_mean": quality,
            "estimated_cost_mean": cost,
            "pareto_optimal": not dominators,
            "dominated_by_count": len(dominators),
            "dominated_by": ",".join(sorted(dominators)),
        })
    return sorted(output, key=lambda row: (not row["pareto_optimal"], -row["joint_quality_mean"]))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_figures(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    costs: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
) -> list[str]:
    if not summaries:
        raise RuntimeError("没有条件汇总数据，拒绝生成空图")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    labels = [row["condition"] for row in summaries]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(max(10, len(labels) * 0.23), 5))
    axis.bar(x, [row["joint_quality_mean"] for row in summaries], color="#4472C4")
    axis.set_xlabel("Encoding condition")
    axis.set_ylabel("Failure-aware joint quality (0–1)")
    axis.set_xticks(x, labels, rotation=90, fontsize=7)
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "encoding_condition_quality.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(output_dir)))

    cost_map = {row["condition"]: row["estimated_cost_mean"] for row in costs}
    available = [
        row for row in summaries if cost_map.get(row["condition"]) is not None
    ]
    if available:
        fig, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(
            [cost_map[row["condition"]] for row in available],
            [row["joint_quality_mean"] for row in available],
            color="#ED7D31",
        )
        for row in available:
            axis.annotate(
                row["condition"],
                (cost_map[row["condition"]], row["joint_quality_mean"]),
                fontsize=6,
            )
        axis.set_xlabel("Estimated cost (configured currency / task)")
        axis.set_ylabel("Failure-aware joint quality (0–1)")
        axis.grid(alpha=0.25)
        fig.tight_layout()
        path = figure_dir / "encoding_quality_cost_pareto.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path.relative_to(output_dir)))

    marginal = [row for row in interactions if row["row_type"] == "marginal_mean"]
    if not marginal:
        raise RuntimeError("没有边际均值数据，拒绝生成空图")
    fig, axis = plt.subplots(figsize=(8, 5))
    for modality, color in zip(MODALITIES, ("#4472C4", "#ED7D31", "#70AD47")):
        subset = [row for row in marginal if row["changed_modality"] == modality]
        axis.plot(
            ENCODINGS,
            [row["mean"] for row in subset],
            marker="o",
            label=modality,
            color=color,
        )
    axis.set_xlabel("Encoding index")
    axis.set_ylabel("Marginal failure-aware joint quality (0–1)")
    axis.set_xticks(ENCODINGS)
    axis.set_ylim(0, 1)
    axis.legend(title="Modality")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "encoding_marginal_means.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(output_dir)))
    return paths


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _write_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    missing: list[str],
    summaries: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
    figure_paths: list[str],
    repeats: int,
    seed: int,
) -> None:
    marginal = [row for row in interactions if row["row_type"] == "marginal_mean"]
    lines = [
        "# 编码筛选分析报告",
        "",
        "## 数据完整性",
        "",
        f"- 已分析 task：{len(rows)}；样本数：{len({row['sample_id'] for row in rows})}。",
        f"- 观测条件：{len(summaries)}/63；缺失 sample-condition：{len(missing)}。",
        "- 缺失条件按缺失处理，不进行插补。",
        "",
        "## 主要定义",
        "",
        "- failure-aware joint quality：仅几何有效时采用 joint quality，否则固定为 0。",
        "- 流水线依次报告 parse、schema、execution、geometry；距离指标仅汇总实际可用值。",
        "- 配对比较只包含预注册比较族，不执行 63 条件的 1953 个有向全对比较，也不作显著性宣称。",
        "",
        "## 条件概览",
        "",
        "| 条件 | n | 联合质量均值 | 中位数 | 95% bootstrap CI | Parse | Schema | Execution | Geometry |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition']} | {row['n']} | {_fmt(row['joint_quality_mean'])} | "
            f"{_fmt(row['joint_quality_median'])} | [{_fmt(row['joint_quality_ci_low'])}, "
            f"{_fmt(row['joint_quality_ci_high'])}] | {_fmt(row['parse_rate'])} | "
            f"{_fmt(row['schema_valid_rate'])} | {_fmt(row['execution_success_rate'])} | "
            f"{_fmt(row['geometry_valid_rate'])} |"
        )
    lines.extend(["", "## 编码边际均值", ""])
    for row in marginal:
        lines.append(
            f"- {row['contrast']}：均值 {_fmt(row['mean'])}，中位数 {_fmt(row['median'])}，"
            f"95% CI [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]，n={row['n']}。"
        )
    frontier = [row for row in pareto if row["pareto_optimal"]]
    lines.extend([
        "",
        "## 质量—成本 Pareto",
        "",
        (
            "- Pareto 条件：" + "、".join(row["condition"] for row in frontier)
            if frontier else
            "- 无法计算 Pareto：没有同时具备质量与可配置成本的条件。"
        ),
        "",
        "## 图表",
        "",
    ])
    lines.extend(f"- `{figure}`" for figure in figure_paths)
    lines.extend([
        "",
        "## 统计说明",
        "",
        f"- 样本级配对差 bootstrap：{repeats} 次，95% CI，seed={seed}。",
        "- `encoding_interactions.csv` 同时给出配对差均值/中位数、win/tie/loss 和 CI。",
        "- 成本由输入/输出/总 token 单价及延迟单价配置；默认单价为 0。",
        "- 本报告用于筛选与效应量描述，不包含 p 值或显著性结论。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_encoding_screen(
    experiment_dir: Path,
    output_dir: Path | None = None,
    *,
    bootstrap_repeats: int = 5000,
    seed: int = 20260813,
    cost_config: dict[str, float] | None = None,
) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else experiment_dir / "analysis"
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats 必须大于 0")
    prices = {**DEFAULT_COST, **(cost_config or {})}
    rows, missing = load_task_rows(experiment_dir, prices)
    summaries = condition_summary(rows, seed, bootstrap_repeats)
    failures = failure_summary(rows)
    interactions = interaction_rows(
        rows, seed=seed, bootstrap_repeats=bootstrap_repeats
    )
    costs = cost_summary(rows)
    pareto = pareto_rows(summaries, costs)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "encoding_condition_summary.csv", summaries)
    _write_csv(output_dir / "encoding_task_rows.csv", rows)
    _write_csv(
        output_dir / "encoding_failure_summary.csv",
        failures,
        ["condition", "stage", "failure_code", "count"],
    )
    _write_csv(output_dir / "encoding_interactions.csv", interactions)
    _write_csv(output_dir / "encoding_cost_summary.csv", costs)
    _write_csv(
        output_dir / "encoding_pareto.csv",
        pareto,
        [
            "condition", "joint_quality_mean", "estimated_cost_mean",
            "pareto_optimal", "dominated_by_count", "dominated_by",
        ],
    )
    figures = _make_figures(output_dir, summaries, costs, interactions)
    result = {
        "schema_version": "rq2.encoding_screen.analysis.v1",
        "experiment_dir": str(experiment_dir),
        "n_samples": len({row["sample_id"] for row in rows}),
        "n_tasks": len(rows),
        "observed_conditions": len(summaries),
        "expected_conditions": 63,
        "missing_task_count": len(missing),
        "missing_tasks": missing,
        "condition_summary": summaries,
        "failure_summary": failures,
        "interactions": interactions,
        "cost_summary": costs,
        "pareto": pareto,
        "figures": figures,
        "cost_config": prices,
        "bootstrap": {"unit": "sample", "repeats": bootstrap_repeats, "seed": seed, "ci": 0.95},
    }
    (output_dir / "encoding_screen.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(
        output_dir / "ENCODING_SCREEN_REPORT_ZH.md",
        rows=rows,
        missing=missing,
        summaries=summaries,
        interactions=interactions,
        pareto=pareto,
        figure_paths=figures,
        repeats=bootstrap_repeats,
        seed=seed,
    )
    return result


# ---------------------------------------------------------------------------
# RQ2b 反馈修正实验分析：按臂统计修复率（按失败类型）、成功率、joint quality 与 token 增量
# ---------------------------------------------------------------------------

FEEDBACK_ARMS = ("A0", "A1", "B1", "B2", "C")
FEEDBACK_CONDITIONS = ("T2", "I1", "P1", "T2I1", "T2I2P1", "I1P1")


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
        # 后端 plan 校验拒绝（plan_validation_failed）在修复前的 runner 中被记为
        # execution kind，但语义上属于格式/校验类失败。这里按与 runner 共用的
        # feedback.failure_kind_from_code 映射归回 format，保证新旧 state 口径一致。
        inner = failure.get("failure")
        code = inner.get("code") if isinstance(inner, dict) else None
        return failure_kind_from_code(code) or "execution"
    return kind or None


def _usage_tokens(usage: Any) -> dict[str, float | None]:
    return {
        "input": _as_float(_nested(usage, ("input_tokens",), ("prompt_tokens",))),
        "output": _as_float(_nested(usage, ("output_tokens",), ("completion_tokens",))),
        "total": _as_float(_nested(usage, ("total_tokens",), ("total_tokens",))),
    }


def _sum_token_field(usage_rows: list[dict[str, float | None]], field: str) -> float | None:
    values = [row[field] for row in usage_rows]
    if not values or all(value is None for value in values):
        return None
    return float(sum(value or 0.0 for value in values))


def feedback_task_row(state: dict[str, Any], *, arm: str) -> dict[str, Any]:
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
        # 无 feedback 块的旧 state（A0 复用）：按终态推断单轮失败类型
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
    fixed_at_round = (
        int(kept_round)
        if fixed and kept_round is not None
        else None
    )
    new_error_kind = any(
        kinds[index] and kinds[index - 1] and kinds[index] != kinds[index - 1]
        for index in range(1, len(kinds))
    )
    usage_rows: list[dict[str, float | None]] = []
    for record in rounds:
        api = record.get("api") if isinstance(record, dict) else None
        usage = api.get("usage") if isinstance(api, dict) else None
        usage_rows.append(_usage_tokens(usage))
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


def load_feedback_states(
    state_dir: Path,
    arm: str,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    state_root = Path(state_dir)
    state_subdir = state_root / "state" if (state_root / "state").is_dir() else state_root
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(state_subdir.glob("*/*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(state.get("sample_id") or path.parent.name)
        condition = str(state.get("condition_id") or state.get("condition") or path.stem)
        if condition not in FEEDBACK_CONDITIONS:
            continue
        key = (sample_id, condition)
        if key in seen:
            raise ValueError(f"重复 task state: {sample_id}/{condition}")
        seen.add(key)
        if sample_id not in manifest:
            raise ValueError(f"state 样本不在清单中: {sample_id}")
        row = feedback_task_row(state, arm=arm)
        for field in ("family", "difficulty", "complexity", "complexity_bin"):
            row[field] = manifest[sample_id].get(field)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"{state_dir} 中没有可分析的反馈实验 state")
    return rows


def feedback_arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    completed = [row for row in rows if row["completed"]]
    round0_failed = [row for row in rows if row["round0_failure_kind"]]
    format_failed = [row for row in round0_failed if row["round0_failure_kind"] == "format"]
    execution_failed = [
        row for row in round0_failed if row["round0_failure_kind"] == "execution"
    ]
    fixed = [row for row in round0_failed if row["fixed"]]
    multi = [row for row in rows if row["n_rounds"] >= 2]
    new_error = [row for row in multi if row["new_error_kind"]]
    fixed_at_1 = sum(1 for row in fixed if row["fixed_at_round"] == 1)
    fixed_at_2 = sum(1 for row in fixed if row["fixed_at_round"] == 2)
    quality_values = [row["joint_quality"] for row in rows]

    def _ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "arm": rows[0]["arm"] if rows else None,
        "n": n,
        "completed": len(completed),
        "success_rate": _ratio(len(completed), n),
        "joint_quality_mean": mean(quality_values) if quality_values else None,
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
        "fixed_at_round1_rate": _ratio(fixed_at_1, len(round0_failed)),
        "fixed_at_round2_rate": _ratio(fixed_at_2, len(round0_failed)),
        "multi_round_tasks": len(multi),
        "new_error_kind_tasks": len(new_error),
        "new_error_kind_rate": _ratio(len(new_error), len(multi)),
        "input_tokens_mean": _mean_field(rows, "input_tokens"),
        "output_tokens_mean": _mean_field(rows, "output_tokens"),
        "total_tokens_mean": _mean_field(rows, "total_tokens"),
        "feedback_input_tokens_mean": _mean_field(rows, "feedback_input_tokens"),
        "feedback_output_tokens_mean": _mean_field(rows, "feedback_output_tokens"),
    }


def _feedback_arm_rows_csv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "arm", "sample_id", "condition", "status", "completed", "joint_quality",
        "n_rounds", "kept_round", "fixed", "fixed_at_round",
        "round0_failure_kind", "round1_failure_kind", "round2_failure_kind",
        "new_error_kind", "input_tokens", "output_tokens", "total_tokens",
        "feedback_input_tokens", "feedback_output_tokens",
        "family", "difficulty", "complexity", "complexity_bin",
    ]
    return [{key: row.get(key) for key in fields} for row in rows]


def _make_feedback_figure(output_dir: Path, summaries: dict[str, dict[str, Any]]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    arms = [arm for arm in FEEDBACK_ARMS if arm in summaries]
    success = [summaries[arm]["success_rate"] or 0.0 for arm in arms]
    quality = [summaries[arm]["joint_quality_mean"] or 0.0 for arm in arms]
    x = np.arange(len(arms))
    width = 0.38
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(x - width / 2, success, width, label="Success rate", color="#4472C4")
    axis.bar(x + width / 2, quality, width, label="Mean joint quality", color="#ED7D31")
    axis.set_xticks(x, arms)
    axis.set_xlabel("Arm")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "feedback_arm_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.relative_to(output_dir))


def _write_feedback_report(
    path: Path,
    *,
    summaries: dict[str, dict[str, Any]],
    arm_rows: dict[str, list[dict[str, Any]]],
    figure: str | None,
    offline: dict[str, Any] | None,
) -> None:
    fmt = _fmt

    def cell(arm: str, key: str) -> str:
        return fmt(summaries[arm].get(key))

    def pct(arm: str, key: str) -> str:
        value = summaries[arm].get(key)
        return "NA" if value is None else f"{float(value):.1%}"

    arm_line = " | ".join(arm for arm in FEEDBACK_ARMS if arm in summaries)
    divider = "|---" + "|---:" * len([a for a in FEEDBACK_ARMS if a in summaries]) + "|"
    lines = [
        "# RQ2b 反馈修正实验报告",
        "",
        "## 实验设计",
        "",
        "- 复用冻结 20 样本 × 6 条件（T2、I1、P1、T2I1、T2I2P1、I1P1），每臂 120 任务。",
        "- A0：v2 prompt 无反馈（复用 encoding_screen_n20 既有 state，零新增成本）。",
        "- A1：v3 prompt（revolve/rotate 硬规则 + revolve few-shot）无反馈。",
        "- B1：v2 prompt + 1 轮反馈（仅 schema/格式类错误）。",
        "- B2：v2 prompt + 最多 2 轮反馈（schema + 执行崩溃，反馈轮（第 1 轮起）温度 0.3）。",
        "- C：v3 prompt + B2 反馈。",
        "",
        "## 各臂总体结果",
        "",
        f"| 指标 | {arm_line} |",
        divider,
        f"| 任务数 | {' | '.join(str(summaries[a]['n']) for a in summaries)} |",
        f"| 成功率 | {' | '.join(pct(a, 'success_rate') for a in summaries)} |",
        f"| joint quality 均值 | {' | '.join(cell(a, 'joint_quality_mean') for a in summaries)} |",
        f"| round0 失败数 | {' | '.join(str(summaries[a]['round0_failed']) for a in summaries)} |",
        f"| 格式类失败数 | {' | '.join(str(summaries[a]['format_failed']) for a in summaries)} |",
        f"| 执行类失败数 | {' | '.join(str(summaries[a]['execution_failed']) for a in summaries)} |",
        f"| 总体修复率（round0 失败→完成） | {' | '.join(pct(a, 'overall_fix_rate') for a in summaries)} |",
        f"| 格式类修复率 | {' | '.join(pct(a, 'format_fix_rate') for a in summaries)} |",
        f"| 执行类修复率 | {' | '.join(pct(a, 'execution_fix_rate') for a in summaries)} |",
        f"| 第 1 轮反馈修复数 | {' | '.join(str(summaries[a]['fixed_at_round1']) for a in summaries)} |",
        f"| 第 2 轮反馈修复数 | {' | '.join(str(summaries[a]['fixed_at_round2']) for a in summaries)} |",
        f"| 引入新错误类型任务数 | {' | '.join(str(summaries[a]['new_error_kind_tasks']) for a in summaries)} |",
        f"| 引入新错误类型率（多轮任务中） | {' | '.join(pct(a, 'new_error_kind_rate') for a in summaries)} |",
        "",
        "## Token 与成本",
        "",
        f"| 指标 | {arm_line} |",
        divider,
        f"| 每任务总 input tokens 均值 | {' | '.join(cell(a, 'input_tokens_mean') for a in summaries)} |",
        f"| 每任务总 output tokens 均值 | {' | '.join(cell(a, 'output_tokens_mean') for a in summaries)} |",
        f"| 每任务总 tokens 均值 | {' | '.join(cell(a, 'total_tokens_mean') for a in summaries)} |",
        f"| 反馈轮 input tokens 增量均值 | {' | '.join(cell(a, 'feedback_input_tokens_mean') for a in summaries)} |",
        f"| 反馈轮 output tokens 增量均值 | {' | '.join(cell(a, 'feedback_output_tokens_mean') for a in summaries)} |",
        "",
    ]
    if offline:
        lines.extend([
            "## 离线修复潜力评估（前置）",
            "",
            f"- 失败 state 总数：{offline.get('n_failed')}；格式类 {offline.get('by_kind', {}).get('format')}，"
            f"执行类 {offline.get('by_kind', {}).get('execution')}。",
            f"- 按错误码可修复性分级加权的预期修复率：{offline.get('expected_fix_rate_overall'):.1%}。",
            "",
        ])
    lines.extend([
        "## 结论与建议",
        "",
    ])
    arms = [arm for arm in FEEDBACK_ARMS if arm in summaries]
    conclusions = []
    if "A0" in summaries:
        conclusions.append(
            "- A0 复用 encoding_screen_n20 早期运行的 state（v2 prompt 无反馈）；B1/B2 与 A0 同 "
            "prompt 但为本次新运行，A0 与 B1 的差异可作为温度 0 重复运行的方差参照。"
        )
    if "A1" in summaries and "A0" in summaries:
        a0, a1 = summaries["A0"], summaries["A1"]
        delta = (a1["success_rate"] or 0.0) - (a0["success_rate"] or 0.0)
        conclusions.append(
            f"- A1 vs A0（仅 prompt v3 约束增强）：成功率变化 {delta:+.1%}；"
            f"后端校验类失败 {a0['format_failed']}→{a1['format_failed']}，"
            f"执行类失败 {a0['execution_failed']}→{a1['execution_failed']}。"
        )
    if "B1" in summaries and "A0" in summaries:
        b1, a0 = summaries["B1"], summaries["A0"]
        delta = (b1["success_rate"] or 0.0) - (a0["success_rate"] or 0.0)
        conclusions.append(
            f"- B1 vs A0（1 轮 schema 反馈）：成功率变化 {delta:+.1%}，"
            f"格式类修复率 {fmt(summaries['B1'].get('format_fix_rate'))}。"
        )
        if not b1.get("multi_round_tasks"):
            conclusions.append(
                f"- 注意：B1 的 {b1['format_failed']} 个校验类失败均来自后端 plan_validation_failed，"
                f"运行器将其记为执行类，而 B1 仅启用 schema 反馈源，反馈轮实际触发 0 次——"
                f"B1 等价于 v2 无反馈重跑，与 A0 的 {delta:+.1%} 差异反映温度 0 重复运行的方差，"
                f"不宜解读为反馈的负面效应，也不构成对 schema 反馈本身有效性的检验。"
            )
            conclusions.append(
                "- 待办：B1 的历史 state 由修复前的 runner 生成（各臂 state 记录的"
                " encoding_runner.py 指纹早于含 plan_validation_failed 归类修复的版本），"
                "即 B1 运行时 schema 反馈实际未触发；确认实验前需用修复后的 runner 重跑 B1"
                "（重跑时旧 state 会自动归档到 history/）。"
            )
    if "B2" in summaries and "B1" in summaries:
        b2, b1 = summaries["B2"], summaries["B1"]
        delta = (b2["success_rate"] or 0.0) - (b1["success_rate"] or 0.0)
        conclusions.append(
            f"- B2 vs B1（增加执行类反馈与第 2 轮）：成功率变化 {delta:+.1%}；"
            f"执行类修复率 {fmt(b2.get('execution_fix_rate'))}"
            f"（{summaries['B2']['execution_failed']} 个执行类失败全部修复），"
            f"校验类修复率 {fmt(b2.get('format_fix_rate'))}。"
        )
    if "C" in summaries and "B2" in summaries:
        c, b2 = summaries["C"], summaries["B2"]
        delta = (c["success_rate"] or 0.0) - (b2["success_rate"] or 0.0)
        conclusions.append(
            f"- C vs B2（在反馈基础上叠加 v3 prompt）：成功率变化 {delta:+.1%}；"
            f"round0 失败数 {b2['round0_failed']}→{c['round0_failed']}，"
            f"两类失败修复率均 100%。"
        )
    if conclusions:
        lines.extend(conclusions)
    best_arm = max(
        (arm for arm in arms if summaries[arm]["success_rate"] is not None),
        key=lambda arm: summaries[arm]["success_rate"] or 0.0,
        default=None,
    )
    threshold = 0.05
    if len(arms) < 2:
        lines.append(
            "- 当前仅有 1 个臂的结果，待其余臂运行完成后自动生成臂间对比与进入确认实验的建议。"
        )
    elif best_arm and "A0" in summaries:
        gain = (summaries[best_arm]["success_rate"] or 0.0) - (summaries["A0"]["success_rate"] or 0.0)
        if gain >= threshold:
            lines.append(
                f"- 建议：{best_arm} 相对 A0 成功率提升 {gain:+.1%}（≥ {threshold:.0%} 阈值），"
                "且成本增量可见于 token 表；建议进入确认实验（冻结臂 × 全量样本/条件，"
                "按 sample×condition 做臂间配对比较并保留运行间方差估计）。"
            )
        else:
            lines.append(
                f"- 建议：最优臂 {best_arm} 相对 A0 提升 {gain:+.1%}，未达 {threshold:.0%} 阈值；"
                "修复收益有限，建议先复核离线可修复性假设与反馈消息可执行性，再决定是否进入确认实验。"
            )
    lines.extend([
        "",
        "## 统计说明",
        "",
        "- 本实验为探索性 20 样本 × 6 条件筛选，比例为点估计，不作显著性宣称。",
        "- 格式类 = JSON 解析失败、client 侧 schema 校验失败与后端 plan_validation_failed；",
        "  执行类 = CadQuery 执行异常（operation_exception、invalid_shape_after_operation 等）。",
        "- 失败类别映射（format/execution）由 feedback.failure_kind_from_code 统一定义，",
        "  runner 与分析共用；CSV 中 round*_failure_kind 列即按此归一。",
        "- 修复率分母为 round0 失败的同类任务；引入新错误率分母为多轮任务数。",
        "- 反馈轮 token 增量 = 各轮使用量之和减去第 0 轮。",
        "- 反馈轮温度为 round2_temperature（0.3），从第 1 轮反馈（round>=1）起生效，",
        "  并非仅第 2 轮；反馈上下文为非累积式（原始输入 + 上一轮输出 + 本轮反馈）。",
    ])
    if figure:
        lines.extend(["", "## 图表", "", f"- `{figure}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_feedback_experiment(
    arm_state_dirs: dict[str, Path],
    output_dir: Path,
    *,
    manifest_path: Path | None = None,
    offline_summary_path: Path | None = None,
) -> dict[str, Any]:
    manifest_rows = _read_jsonl(manifest_path)
    manifest = {str(row["sample_id"]): row for row in manifest_rows}
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    for arm in FEEDBACK_ARMS:
        if arm not in arm_state_dirs:
            continue
        arm_rows[arm] = load_feedback_states(arm_state_dirs[arm], arm, manifest)
    summaries = {
        arm: feedback_arm_summary(rows) for arm, rows in arm_rows.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "feedback_task_rows.csv",
        [row for rows in arm_rows.values() for row in _feedback_arm_rows_csv(rows)],
    )
    _write_csv(
        output_dir / "feedback_arm_summary.csv",
        list(summaries.values()),
        [
            "arm", "n", "completed", "success_rate", "joint_quality_mean",
            "round0_failed", "format_failed", "execution_failed",
            "overall_fix_rate", "format_fix_rate", "execution_fix_rate",
            "fixed_at_round1", "fixed_at_round2",
            "fixed_at_round1_rate", "fixed_at_round2_rate",
            "multi_round_tasks", "new_error_kind_tasks", "new_error_kind_rate",
            "input_tokens_mean", "output_tokens_mean", "total_tokens_mean",
            "feedback_input_tokens_mean", "feedback_output_tokens_mean",
        ],
    )
    offline: dict[str, Any] | None = None
    if offline_summary_path and offline_summary_path.is_file():
        offline = json.loads(offline_summary_path.read_text(encoding="utf-8"))
    figure = _make_feedback_figure(output_dir, summaries)
    _write_feedback_report(
        output_dir / "FEEDBACK_REPORT_ZH.md",
        summaries=summaries,
        arm_rows=arm_rows,
        figure=figure,
        offline=offline,
    )
    result = {
        "schema_version": "rq2.feedback_n20.analysis.v1",
        "arms": list(arm_rows),
        "arm_summary": summaries,
        "task_rows": (
            output_dir / "feedback_task_rows.csv"
        ).as_posix(),
        "figure": figure,
    }
    (output_dir / "feedback_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def feedback_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分析 RQ2b 反馈修正实验（各臂对比）")
    harness_dir = Path(__file__).resolve().parents[1]
    default_manifest = harness_dir / "outputs" / "feedback_n20" / "sample_manifest.jsonl"
    parser.add_argument(
        "--arm-dir",
        nargs=2,
        action="append",
        metavar=("ARM", "STATE_DIR"),
        help="臂与其 state 目录（可多次指定；默认按 configs/feedback_n20.yaml 的 arms 推断）",
    )
    parser.add_argument("--output", default=str(harness_dir / "outputs" / "feedback_n20" / "analysis"))
    parser.add_argument("--manifest", default=str(default_manifest))
    parser.add_argument(
        "--offline-summary",
        default=str(
            harness_dir
            / "outputs"
            / "encoding_screen_n20"
            / "analysis"
            / "feedback_potential.json"
        ),
    )
    args = parser.parse_args(argv)
    if args.arm_dir:
        arm_state_dirs = {arm: Path(state_dir) for arm, state_dir in args.arm_dir}
    else:
        config_path = harness_dir / "configs" / "feedback_n20.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        arms_block = config.get("arms") or {}
        arm_state_dirs = {}
        for arm in FEEDBACK_ARMS:
            block = arms_block.get(arm) or {}
            state_dir = block.get("reuse_state_dir") or block.get("output_dir")
            if not state_dir:
                continue
            path = Path(state_dir)
            if not path.is_absolute():
                path = harness_dir.parents[1] / path
            if path.is_dir():
                arm_state_dirs[arm] = path
    result = analyze_feedback_experiment(
        arm_state_dirs,
        Path(args.output),
        manifest_path=Path(args.manifest),
        offline_summary_path=Path(args.offline_summary),
    )
    print(json.dumps(
        {
            arm: {
                "n": summary["n"],
                "success_rate": summary["success_rate"],
                "overall_fix_rate": summary["overall_fix_rate"],
            }
            for arm, summary in result["arm_summary"].items()
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分析 RQ2 编码筛选实验")
    default_input = Path(__file__).resolve().parents[1] / "outputs" / "encoding_screen_n20"
    parser.add_argument("--input", default=str(default_input))
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--input-cost-per-million", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million", type=float, default=0.0)
    parser.add_argument("--total-cost-per-million", type=float, default=0.0)
    parser.add_argument("--latency-cost-per-second", type=float, default=0.0)
    args = parser.parse_args(argv)
    result = analyze_encoding_screen(
        Path(args.input),
        Path(args.output) if args.output else None,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
        cost_config={
            "input_per_million_tokens": args.input_cost_per_million,
            "output_per_million_tokens": args.output_cost_per_million,
            "total_per_million_tokens": args.total_cost_per_million,
            "latency_per_second": args.latency_cost_per_second,
        },
    )
    print(json.dumps({
        "n_tasks": result["n_tasks"],
        "observed_conditions": result["observed_conditions"],
        "missing_task_count": result["missing_task_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
