# -*- coding: utf-8 -*-
"""离线修复潜力评估：扫描 encoding_screen_n20 全部失败 state，
统计错误类型分布、可修复性分级，并抽样输出将要发回模型的反馈消息。

只读脚本，不调用 API、不修改 state。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "outputs" / "encoding_screen_n20" / "state"
OUT_DIR = ROOT / "outputs" / "encoding_screen_n20" / "analysis"

# 可修复性分级假设（基于错误码语义，用于校准预期修复率区间）
FIXABILITY = {
    # 格式类：错误定位精确、修法明确
    "invalid_revolve_axis": ("high", "轴两点重合/非法：改选两个不同的 2D 点"),
    "invalid_rotate": ("high", "rotate 轴/原点非法：改为单位向量+合法原点"),
    "axis_not_unit_length": ("high", "轴未归一化：换成单位向量"),
    "invalid_vector": ("high", "矢量格式非法：换成 3 数字列表"),
    "invalid_number": ("high", "数值非法：换成有限数字"),
    "polygon_not_closed": ("high", "多边形未闭合：补上首点"),
    "duplicate_polygon_vertex": ("high", "连续重复顶点：删除重复点"),
    "invalid_combine": ("high", "combine 取值非法：按位置规则改正"),
    "invalid_sample_id": ("high", "sample_id 非字符串"),
    "sample_id_mismatch": ("high", "sample_id 写错：改回给定的 id"),
    "field_set_mismatch": ("high", "必填键缺失/多余键"),
    "extra_field": ("high", "多余字段：删除"),
    "invalid_schema_version": ("high", "schema_version 写错"),
    "invalid_coordinate_system": ("high", "坐标系块不标准"),
    "noncanonical_coordinate_system": ("high", "坐标系块不标准"),
    "invalid_operations": ("high", "operations 数量/类型非法"),
    "unsupported_operation": ("high", "用了不存在的 op"),
    "unsupported_primitive": ("high", "用了不存在的 primitive"),
    "not_object": ("high", "操作不是对象"),
    "invalid_or_duplicate_id": ("high", "id 重复/非法"),
    "empty_transform": ("high", "transform 缺 translate/rotate"),
    "modifier_cannot_be_first": ("high", "fillet/chamfer 不能放第一个"),
    "invalid_vec3": ("high", "中心点不是 3 数字"),
    "invalid_json": ("high", "JSON 语法错误：重写为合法 JSON"),
    "invalid_polygon": ("medium", "多边形语义非法：简化/重新采样轮廓"),
    "degenerate_polygon": ("medium", "退化多边形：去除共线重复点"),
    "self_intersecting_polygon": ("medium", "边自相交：简化轮廓避开交叉"),
    "profile_crosses_axis": ("medium", "旋转轮廓穿过轴线：把轮廓移到轴一侧"),
    "invalid_reference": ("medium", "引用了不存在的 op id：改为已有 id"),
    "invalid_workplane": ("medium", "工作平面非法：改为 XY/XZ/YZ"),
    "empty_after_operation": ("medium", "布尔后为空：调整尺寸/位置使其相交"),
    # 执行类：错误消息多为 CadQuery/OCP 内部异常，较晦涩
    "operation_exception": ("low", "CadQuery 执行异常：按 operationId 定位后简化/替换该操作"),
    "invalid_shape_after_operation": ("low", "操作后形状非法：简化该操作几何"),
    "invalid_step_shape": ("low", "STEP 无效：通常需回退最后操作"),
    "step_export_failed": ("low", "STEP 导出失败"),
    "execution_timeout": ("low", "执行超时：减少操作数"),
}
EXPECTED_FIX_RATE = {"high": 0.55, "medium": 0.30, "low": 0.10}  # 预期修复率假设


def _failure_kind(state: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    status = str(state.get("status"))
    if status == "parse_failed":
        parse = state.get("parse") or {}
        issues = parse.get("issues") or state.get("post_repair_issues") or []
        if issues:
            return "format", issues
        return "parse_json", [{"code": "invalid_json", "path": "$"}]
    if status == "episode_failed":
        response = (state.get("episode") or {}).get("response") or {}
        failure = response.get("failure") or {}
        if failure.get("code") == "plan_validation_failed":
            issues = (response.get("validation") or {}).get("issues") or []
            return "format", issues
        if failure:
            issue = {
                "code": str(failure.get("code") or "operation_exception"),
                "path": f"$.operations[{failure.get('operationIndex')}]",
                "message": str(failure.get("message"))[:200],
                "operationId": failure.get("operationId"),
            }
            return "execution", [issue]
    return "other", [{"code": str(status)}]


def _feedback_message(state: dict[str, Any], kind: str, issues: list[dict[str, Any]]) -> str:
    lines = ["Your previous plan failed validation/execution. Fix ONLY the reported problems and return the corrected JSON plan.", "", "Reported errors:"]
    for issue in issues:
        code = issue.get("code")
        path = issue.get("path")
        message = issue.get("message")
        parts = [f"- [{code}] at {path}"]
        if message:
            parts.append(f"  message: {message}")
        lines.extend(parts)
    lines.extend([
        "",
        "Requirements: keep all other operations unchanged unless required; output one JSON object only.",
    ])
    return "\n".join(lines)


def main() -> None:
    states = sorted(STATE_DIR.glob("*/*.json"))
    failed = [s for s in states if json.loads(s.read_text(encoding="utf-8")).get("status") != "completed"]
    print(f"state 总数={len(states)}  失败数={len(failed)}")

    by_kind: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()
    code_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    kind_codes: dict[str, Counter[str]] = defaultdict(Counter)

    for path in failed:
        state = json.loads(path.read_text(encoding="utf-8"))
        kind, issues = _failure_kind(state)
        by_kind[kind] += 1
        for issue in issues:
            code = str(issue.get("code") or "unknown")
            code_counts[code] += 1
            kind_codes[kind][code] += 1
            if len(code_examples[code]) < 3:
                code_examples[code].append(
                    {
                        "sample": state.get("sample_id"),
                        "condition": state.get("condition_id"),
                        "path": issue.get("path"),
                        "message": issue.get("message"),
                    }
                )

    print()
    print("=== 失败大类 ===")
    for kind, count in by_kind.most_common():
        print(f"  {kind:12s} {count}")
    print()
    print("=== 错误码分布（count | fixability | 说明 | 示例）===")
    weighted_expected = 0.0
    total = sum(code_counts.values())
    rows = []
    for code, count in code_counts.most_common():
        fixability, note = FIXABILITY.get(code, ("low", "未分类"))
        rows.append((code, count, fixability, note))
        weighted_expected += count * EXPECTED_FIX_RATE[fixability]
    for code, count, fixability, note in sorted(rows, key=lambda r: -r[1]):
        example = code_examples[code][0]
        print(f"  {code:32s} {count:4d}  [{fixability:6s}] {note}")
        print(f"    {'':32s} e.g. {example['condition']}/{example['sample']} path={example['path']} msg={example['message']}")
    print()
    print(f"  预期修复率（按分级假设加权，全体错误码出现次数）: {weighted_expected / total:.2%}")

    # 按首个错误码计每个失败任务的预期修复
    per_task_expected = 0.0
    for path in failed:
        state = json.loads(path.read_text(encoding="utf-8"))
        kind, issues = _failure_kind(state)
        if not issues:
            continue
        fixability = FIXABILITY.get(str(issues[0].get("code")), ("low",))[0]
        per_task_expected += EXPECTED_FIX_RATE[fixability]
    print(f"  每失败任务按首个错误码计的预期可救回数: {per_task_expected:.1f} / {len(failed)} ({per_task_expected / len(failed):.1%})")

    # 按失败大类分组：格式类 vs 执行类的预期可救回
    per_kind_expected: dict[str, float] = defaultdict(float)
    per_kind_n: Counter[str] = Counter()
    for path in failed:
        state = json.loads(path.read_text(encoding="utf-8"))
        kind, issues = _failure_kind(state)
        per_kind_n[kind] += 1
        if not issues:
            continue
        fixability = FIXABILITY.get(str(issues[0].get("code")), ("low",))[0]
        per_kind_expected[kind] += EXPECTED_FIX_RATE[fixability]
    print()
    print("=== 按失败大类的预期可救回（首个错误码分级加权） ===")
    for kind, n in per_kind_n.most_common():
        expected = per_kind_expected.get(kind, 0.0)
        print(f"  {kind:12s} 失败 {n:4d} 个，预期可救回 {expected:5.1f} 个（{expected / n:.1%}）")

    # 抽样 30 个反馈消息供人工检查
    samples = failed[:: max(1, len(failed) // 30)][:30]
    sample_lines = []
    for path in samples:
        state = json.loads(path.read_text(encoding="utf-8"))
        kind, issues = _failure_kind(state)
        message = _feedback_message(state, kind, issues)
        sample_lines.append(
            "=" * 70
            + f"\n[{state.get('sample_id')} / {state.get('condition_id')} / {kind}]\n"
            + message
            + "\n"
        )
    sample_text = "\n".join(sample_lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "feedback_potential_samples.txt").write_text(sample_text, encoding="utf-8")
    summary = {
        "n_states": len(states),
        "n_failed": len(failed),
        "by_kind": dict(by_kind),
        "code_counts": dict(code_counts),
        "kind_codes": {kind: dict(counter) for kind, counter in kind_codes.items()},
        "expected_fix_rate_overall": weighted_expected / total if total else 0.0,
        "expected_recoverable_tasks": round(per_task_expected, 1),
        "fixability": FIXABILITY,
        "expected_fix_rate_by_class": EXPECTED_FIX_RATE,
    }
    (OUT_DIR / "feedback_potential.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print()
    print(f"已写入: {OUT_DIR / 'feedback_potential.json'}")
    print(f"已写入: {OUT_DIR / 'feedback_potential_samples.txt'}（30 个反馈消息抽样，供人工检查可执行性）")


if __name__ == "__main__":
    main()
