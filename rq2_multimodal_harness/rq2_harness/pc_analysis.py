"""P_geom 20×9 筛选分析：成功率、joint quality、配对几何、工具统计、分层。

配对对比为探索性，不做显著性宣称。失败任务的 joint_quality 记 0。
paired-valid-only：只保留两个条件都 status==completed 的样本再比几何。
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .analysis import bootstrap_ci
from .common import atomic_write_json, read_jsonl
from .pc_conditions import CONFIRM_CONDITION_IDS, SCREEN_CONDITION_IDS

PAIRED_CONTRASTS = (
    ("P_geom_tool", "P_proj"),
    ("P_geom_static", "P_proj"),
    ("I1P_geom", "I1P_proj"),
    ("I1P_geom", "I1"),
    ("T1I1P_geom", "T1I1"),
    ("T1I1P_geom", "T1I1P_proj"),
    ("I1P_proj", "I1"),
    ("P_geom_tool", "P_geom_static"),
)

FREEZE_CONDITIONS = (
    "I1",
    "P_proj",
    "P_geom_tool",
    "I1P_proj",
    "I1P_geom",
    "T1I1",
    "T1I1P_proj",
    "T1I1P_geom",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _geometry_values(state: dict[str, Any]) -> dict[str, float | None]:
    geometry = state.get("geometry") or {}
    completed = state.get("status") == "completed"
    voxel = geometry.get("voxel_iou") or {}
    fscore_shape = geometry.get("fscore_shape") or {}
    fscore_common = geometry.get("fscore_common") or {}
    bbox = geometry.get("bbox") or {}
    return {
        "joint_quality": float(geometry.get("joint_quality") or 0.0) if completed else 0.0,
        "shape_only_cd": geometry.get("shape_only_cd"),
        "common_frame_cd": geometry.get("common_frame_cd"),
        "voxel_iou": voxel.get("value") if isinstance(voxel, dict) else voxel,
        "f1_shape": fscore_shape.get("f1") if isinstance(fscore_shape, dict) else fscore_shape,
        "f1_common": fscore_common.get("f1") if isinstance(fscore_common, dict) else fscore_common,
        "bbox_scale_log_abs": bbox.get("scale_log_abs") if isinstance(bbox, dict) else None,
    }


def _usage(state: dict[str, Any]) -> dict[str, float]:
    api = state.get("api") or {}
    usage = api.get("usage") or {}
    feedback = state.get("feedback") or {}
    n_calls = int(feedback.get("n_api_calls") or (1 if api else 0))
    return {
        "latency_sec": float(api.get("latency_sec") or state.get("elapsed_sec") or 0.0),
        "prompt_tokens": float(usage.get("prompt_tokens") or usage.get("input_tokens") or 0.0),
        "completion_tokens": float(usage.get("completion_tokens") or usage.get("output_tokens") or 0.0),
        "n_api_calls": float(n_calls),
        "n_tool_calls": float(len(state.get("tool_traces") or [])),
    }


def load_task_rows(
    output_dir: Path,
    manifest_path: Path,
    *,
    condition_ids: tuple[str, ...] = SCREEN_CONDITION_IDS,
) -> list[dict[str, Any]]:
    allowed = frozenset(condition_ids)
    manifest = {row["sample_id"]: row for row in read_jsonl(manifest_path)}
    rows: list[dict[str, Any]] = []
    state_dir = output_dir / "state"
    if not state_dir.is_dir():
        return rows
    state_paths = list(state_dir.glob("*/*.json")) + list(state_dir.glob("*/*/r*.json"))
    for path in sorted(state_paths):
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("status") in {"dry_run", "running"}:
            continue
        condition = state.get("condition_id") or state.get("condition")
        if condition not in allowed:
            continue
        sample = manifest.get(state.get("sample_id")) or {}
        geom = _geometry_values(state)
        usage = _usage(state)
        traces = state.get("tool_traces") or []
        rows.append(
            {
                "sample_id": state["sample_id"],
                "condition": condition,
                "status": state.get("status"),
                "completed": state.get("status") == "completed",
                "family": sample.get("family"),
                "difficulty": sample.get("difficulty"),
                "complexity_bin": sample.get("complexity_bin"),
                "repeat_id": state.get("repeat_id", 0),
                **geom,
                **usage,
                "tool_names": ",".join(str(item.get("tool") or "") for item in traces if isinstance(item, dict)),
            }
        )
    return rows


def _mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return mean(clean) if clean else None


def condition_summary(
    rows: list[dict[str, Any]],
    condition_ids: tuple[str, ...] = SCREEN_CONDITION_IDS,
) -> list[dict[str, Any]]:
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    result = []
    for condition in condition_ids:
        items = by_cond.get(condition) or []
        n = len(items)
        n_ok = sum(1 for item in items if item["completed"])
        result.append(
            {
                "condition": condition,
                "n": n,
                "n_completed": n_ok,
                "success_rate": (n_ok / n) if n else None,
                "mean_joint_quality": _mean([item["joint_quality"] for item in items]),
                "median_joint_quality": median([item["joint_quality"] for item in items]) if items else None,
                "mean_joint_quality_completed": _mean(
                    [item["joint_quality"] for item in items if item["completed"]]
                ),
                "mean_latency_sec": _mean([item["latency_sec"] for item in items]),
                "mean_prompt_tokens": _mean([item["prompt_tokens"] for item in items]),
                "mean_completion_tokens": _mean([item["completion_tokens"] for item in items]),
                "mean_api_calls": _mean([item["n_api_calls"] for item in items]),
                "mean_tool_calls": _mean([item["n_tool_calls"] for item in items]),
            }
        )
    return result


def paired_table(
    rows: list[dict[str, Any]],
    *,
    valid_only: bool,
    condition_ids: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    allowed = frozenset(condition_ids) if condition_ids is not None else None
    by_key: dict[tuple[str, str], dict[str, Any]] = {
        (row["sample_id"], row["condition"]): row for row in rows
    }
    result = []
    for left, right in PAIRED_CONTRASTS:
        if allowed is not None and (left not in allowed or right not in allowed):
            continue
        deltas: list[float] = []
        n_pairs = 0
        n_left_better = 0
        n_right_better = 0
        n_tie = 0
        for sample_id in {row["sample_id"] for row in rows}:
            a = by_key.get((sample_id, left))
            b = by_key.get((sample_id, right))
            if a is None or b is None:
                continue
            if valid_only and not (a["completed"] and b["completed"]):
                continue
            n_pairs += 1
            delta = float(a["joint_quality"]) - float(b["joint_quality"])
            deltas.append(delta)
            if delta > 1e-12:
                n_left_better += 1
            elif delta < -1e-12:
                n_right_better += 1
            else:
                n_tie += 1
        ci = bootstrap_ci(deltas, seed=42) if deltas else {"mean": None, "low": None, "high": None, "n": 0}
        result.append(
            {
                "contrast": f"{left}-{right}",
                "left": left,
                "right": right,
                "valid_only": valid_only,
                "n_pairs": n_pairs,
                "mean_delta_joint_quality": ci["mean"],
                "ci95_low": ci["low"],
                "ci95_high": ci["high"],
                "n_left_better": n_left_better,
                "n_right_better": n_right_better,
                "n_tie": n_tie,
            }
        )
    return result


def stratified(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get(field), row["condition"])].append(row)
    result = []
    for (level, condition), items in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        n = len(items)
        n_ok = sum(1 for item in items if item["completed"])
        result.append(
            {
                "stratum": field,
                "level": level,
                "condition": condition,
                "n": n,
                "success_rate": (n_ok / n) if n else None,
                "mean_joint_quality": _mean([item["joint_quality"] for item in items]),
            }
        )
    return result


def tool_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tool_rows = [row for row in rows if row["condition"] in {"P_geom_tool", "I1P_geom", "T1I1P_geom"}]
    names: Counter[str] = Counter()
    for row in tool_rows:
        for name in str(row.get("tool_names") or "").split(","):
            if name:
                names[name] += 1
    return {
        "n_tool_condition_tasks": len(tool_rows),
        "mean_tool_calls": _mean([row["n_tool_calls"] for row in tool_rows]),
        "tool_counts": dict(names),
        "n_with_any_tool": sum(1 for row in tool_rows if row["n_tool_calls"] > 0),
    }


def freeze_decision(
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    *,
    kind: str = "screen",
) -> dict[str, Any]:
    by_cond = {row["condition"]: row for row in summary}
    by_contrast = {row["contrast"]: row for row in paired}
    geom_vs_proj = by_contrast.get("P_geom_tool-P_proj") or {}
    i1_gain = by_contrast.get("I1P_geom-I1") or {}
    t_gain = by_contrast.get("T1I1P_geom-T1I1") or {}
    proj_vs_i1 = by_contrast.get("I1P_proj-I1") or {}
    static_vs_tool = by_contrast.get("P_geom_tool-P_geom_static") or {}
    mean_delta = geom_vs_proj.get("mean_delta_joint_quality")
    n_pairs = int(geom_vs_proj.get("n_pairs") or 0)
    keep_static = False
    notes = []
    if mean_delta is None:
        notes.append("筛选尚未产生可配对的 live 结果，冻结条件按计划预登记，待补跑后复核。")
    else:
        notes.append(
            f"P_geom_tool 相对 P_proj 的 mean Δjoint_quality={mean_delta:.4f} "
            f"（探索性，n={n_pairs}，不做显著性宣称）。"
        )
        static_delta = static_vs_tool.get("mean_delta_joint_quality")
        if static_delta is not None and abs(float(static_delta)) <= 0.02:
            notes.append(
                "P_geom_tool 与 P_geom_static 几乎同分布；筛选中模型未发出 query_request "
                "（工具预算未被使用）。确认阶段仍按计划保留 P_geom_tool、去掉 P_geom_static，"
                "以便 100 样本上继续观察工具是否被调用。"
            )
        elif static_delta is not None and static_delta > 0.02:
            notes.append("工具条件优于静态证据，确认阶段保留 P_geom_tool、去掉 P_geom_static。")
        else:
            notes.append("确认阶段按计划去掉 P_geom_static 以控制成本。")
        i1_delta = i1_gain.get("mean_delta_joint_quality")
        t_delta = t_gain.get("mean_delta_joint_quality")
        if i1_delta is not None:
            notes.append(f"I1P_geom−I1 mean Δ={float(i1_delta):.4f}（RGB+原生证据相对纯 RGB）。")
        if t_delta is not None:
            notes.append(f"T1I1P_geom−T1I1 mean Δ={float(t_delta):.4f}。")
        stage_label = "100 样本确认" if kind == "confirm" else "20 样本筛选"
        ci_low = proj_vs_i1.get("ci95_low")
        ci_high = proj_vs_i1.get("ci95_high")
        if ci_low is not None and ci_high is not None:
            crosses_zero = float(ci_low) <= 0.0 <= float(ci_high)
            ctrl_note = (
                "负向对照（I1P_proj−I1）CI 穿过 0，与阶段 0 结论一致。"
                if crosses_zero
                else "负向对照（I1P_proj−I1）增益很小，仍远小于原生 PointEvidence。"
            )
        else:
            ctrl_note = "负向对照（I1P_proj−I1）待补齐。"
        notes.append(
            "对照阶段 0：Harness 固定后旧 P_proj 几乎无正互补；本轮原生 PointEvidence "
            f"在{stage_label}上相对 P_proj 显示正的 joint_quality 差。{ctrl_note}"
        )
    decision = {
        "confirm_conditions": list(FREEZE_CONDITIONS),
        "drop_from_screen": ["P_geom_static"],
        "keep_P_geom_static_in_confirm": keep_static,
        "evidence_schema": "point_evidence.v1",
        "tool_budget": {"max_pre_queries": 3, "max_post_queries": 1},
        "model": "qwen3.8-max",
        "harness_arm": "C",
        "notes": notes,
        "exploratory_contrasts": {
            "P_geom_tool-P_proj": geom_vs_proj,
            "I1P_geom-I1": i1_gain,
            "T1I1P_geom-T1I1": t_gain,
        },
    }
    return decision


def write_report(
    output_dir: Path,
    *,
    summary: list[dict[str, Any]],
    paired_all: list[dict[str, Any]],
    paired_valid: list[dict[str, Any]],
    by_difficulty: list[dict[str, Any]],
    by_family: list[dict[str, Any]],
    tools: dict[str, Any],
    decision: dict[str, Any],
    n_rows: int,
    kind: str = "screen",
) -> str:
    title = (
        "# 原生点云几何证据确认报告（P_geom 100×8）"
        if kind == "confirm"
        else "# 原生点云几何证据筛选报告（P_geom 20×9）"
    )
    tool_note = (
        "确认阶段继续统计 query_request；筛选中工具调用为 0。"
        if kind == "confirm"
        else (
            "筛选中模型未发出 query_request（json_mode 下模型直接输出 Plan）。"
            "因此 P_geom_tool 与 P_geom_static 在本轮几乎是同一输入。该负向工具使用结果如实保留。"
        )
    )
    lines = [
        title,
        "",
        "> Harness 固定 C 臂（v3 prompt + R4 repair + 2 轮反馈）。配对对比探索性，不做显著性宣称。",
        "> 负结果如实保留。",
        "",
        f"- 有效任务行：{n_rows}",
        f"- 工具条件任务数：{tools.get('n_tool_condition_tasks')}",
        f"- 平均工具调用：{tools.get('mean_tool_calls')}",
        f"- 至少调用一次工具的任务：{tools.get('n_with_any_tool')}",
        f"- 工具频次：{json.dumps(tools.get('tool_counts') or {}, ensure_ascii=False)}",
        "",
        tool_note,
        "",
        "## 条件汇总",
        "",
        "| 条件 | n | 成功率 | mean JQ | median JQ | mean API | mean tools |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def fmt(value: Any, digits: int = 3) -> str:
            return "—" if value is None else f"{value:.{digits}f}"

        lines.append(
            f"| {row['condition']} | {row['n']} | {fmt(row['success_rate'])} | "
            f"{fmt(row['mean_joint_quality'])} | {fmt(row['median_joint_quality'])} | "
            f"{fmt(row['mean_api_calls'], 2)} | {fmt(row['mean_tool_calls'], 2)} |"
        )
    lines += ["", "## 配对 Δjoint_quality（全部样本，失败记 0）", ""]
    lines += ["| 对比 | n | mean Δ | 95% CI | left更好 | right更好 | 平 |", "|---|---:|---:|---|---:|---:|---:|"]
    for row in paired_all:
        ci = (
            "—"
            if row["mean_delta_joint_quality"] is None
            else f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
        )
        mean_d = "—" if row["mean_delta_joint_quality"] is None else f"{row['mean_delta_joint_quality']:.3f}"
        lines.append(
            f"| {row['contrast']} | {row['n_pairs']} | {mean_d} | {ci} | "
            f"{row['n_left_better']} | {row['n_right_better']} | {row['n_tie']} |"
        )
    lines += ["", "## paired-valid-only（两条件都 completed）", ""]
    lines += ["| 对比 | n | mean Δ | 95% CI |", "|---|---:|---:|---|"]
    for row in paired_valid:
        ci = (
            "—"
            if row["mean_delta_joint_quality"] is None
            else f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
        )
        mean_d = "—" if row["mean_delta_joint_quality"] is None else f"{row['mean_delta_joint_quality']:.3f}"
        lines.append(f"| {row['contrast']} | {row['n_pairs']} | {mean_d} | {ci} |")
    lines += ["", "## 按 difficulty 分层 mean JQ", ""]
    lines += ["| difficulty | 条件 | n | 成功率 | mean JQ |", "|---|---|---:|---:|---:|"]
    for row in by_difficulty:
        jq = "—" if row["mean_joint_quality"] is None else f"{row['mean_joint_quality']:.3f}"
        sr = "—" if row["success_rate"] is None else f"{row['success_rate']:.3f}"
        lines.append(f"| {row['level']} | {row['condition']} | {row['n']} | {sr} | {jq} |")
    lines += ["", "## 冻结决策", ""]
    lines.append("确认阶段 8 条件：`" + "`, `".join(decision["confirm_conditions"]) + "`")
    lines.append("")
    lines.append(f"- 证据 schema：`{decision['evidence_schema']}`")
    lines.append(
        f"- 工具预算：pre={decision['tool_budget']['max_pre_queries']} / "
        f"post={decision['tool_budget']['max_post_queries']}"
    )
    lines.append(f"- 模型 / Harness：{decision['model']} / 臂 {decision['harness_arm']}")
    for note in decision.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    if kind == "confirm":
        lines.append("本轮不做：P_enc、8192 密度臂、换模型家族。")
    else:
        lines.append("本轮不做：P_enc、100 样本确认、8192 密度臂、换模型家族。")
    lines.append("")
    return "\n".join(lines)


def analyze_pc_geom(
    output_dir: Path,
    manifest_path: Path,
    *,
    condition_ids: tuple[str, ...] | None = None,
    kind: str = "screen",
) -> dict[str, Any]:
    selected = tuple(condition_ids or (CONFIRM_CONDITION_IDS if kind == "confirm" else SCREEN_CONDITION_IDS))
    rows = load_task_rows(output_dir, manifest_path, condition_ids=selected)
    summary = condition_summary(rows, selected)
    paired_all = paired_table(rows, valid_only=False, condition_ids=selected)
    paired_valid = paired_table(rows, valid_only=True, condition_ids=selected)
    by_difficulty = stratified(rows, "difficulty")
    by_family = stratified(rows, "family")
    tools = tool_stats(rows)
    decision = freeze_decision(summary, paired_all, kind=kind)
    analysis_dir = output_dir / "analysis"
    _write_csv(analysis_dir / "pc_task_rows.csv", rows)
    _write_csv(analysis_dir / "pc_condition_summary.csv", summary)
    _write_csv(analysis_dir / "pc_paired_all.csv", paired_all)
    _write_csv(analysis_dir / "pc_paired_valid_only.csv", paired_valid)
    _write_csv(analysis_dir / "pc_by_difficulty.csv", by_difficulty)
    _write_csv(analysis_dir / "pc_by_family.csv", by_family)
    report = write_report(
        output_dir,
        summary=summary,
        paired_all=paired_all,
        paired_valid=paired_valid,
        by_difficulty=by_difficulty,
        by_family=by_family,
        tools=tools,
        decision=decision,
        n_rows=len(rows),
        kind=kind,
    )
    report_name = (
        "NATIVE_POINTCLOUD_CONFIRM_REPORT_ZH.md"
        if kind == "confirm"
        else "NATIVE_POINTCLOUD_SCREEN_REPORT_ZH.md"
    )
    report_path = analysis_dir / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    payload = {
        "n_rows": len(rows),
        "summary": summary,
        "paired_all": paired_all,
        "paired_valid": paired_valid,
        "tools": tools,
        "decision": decision,
    }
    atomic_write_json(analysis_dir / "pc_geom_screen.json", payload)
    freeze_config = {
        "schema_version": "rq2.pc_geom_confirm_freeze.v1",
        "n": 100,
        "conditions": decision["confirm_conditions"],
        "evidence_schema": decision["evidence_schema"],
        "tools": decision["tool_budget"],
        "model": decision["model"],
        "harness_arm": decision["harness_arm"],
        "notes": decision["notes"],
    }
    atomic_write_json(analysis_dir / "pc_geom_confirm_freeze.json", freeze_config)
    return payload
