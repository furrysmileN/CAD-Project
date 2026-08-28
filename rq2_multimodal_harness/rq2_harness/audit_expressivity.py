from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import atomic_write_json, load_config, project_path, read_jsonl
from .geometry import score_step_pair
from .prompting import validate_plan


V1_DIRECT = {"box", "cylinder", "sphere", "union", "cut", "intersect", "fuse"}
V1_APPROXIMABLE = {"circle", "extrude", "hole", "translate", "workplane", "center"}
V2_DIRECT = V1_DIRECT | {
    "polyline",
    "moveto",
    "lineto",
    "close",
    "extrude",
    "revolve",
    "hole",
    "slot2d",
    "fillet",
    "chamfer",
    "translate",
    "rotate",
    "workplane",
    "center",
    "vector",
}
V2_APPROXIMABLE = {
    "rect",
    "circle",
    "polygon",
    "cutthruall",
    "cutblind",
    "pushpoints",
    "rarray",
    "transformed",
    "faces",
    "edges",
}
IGNORED_CALLS = {"show_object"}


def cadquery_operations(code: str) -> Counter[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return Counter({"<syntax_error>": 1})
    result: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Attribute):
                name = function.attr.lower()
                if name not in IGNORED_CALLS:
                    result[name] += 1
            elif isinstance(function, ast.Name):
                name = function.id.lower()
                if name not in IGNORED_CALLS:
                    result[name] += 1
    return result


def _coverage(counts: Counter[str], supported: set[str]) -> float:
    denominator = sum(counts.values())
    return sum(count for name, count in counts.items() if name in supported) / denominator if denominator else 0.0


def audit(config: dict[str, Any], oracle_plan_dir: Path | None = None, output: Path | None = None) -> dict[str, Any]:
    output_dir = project_path(config["paths"]["output_dir"])
    manifest = list(read_jsonl(output_dir / "manifest.jsonl"))
    total = Counter()
    samples = []
    for row in manifest:
        code = Path(row["gt_code"]["path"]).read_text(encoding="utf-8")
        counts = cadquery_operations(code)
        total.update(counts)
        samples.append(
            {
                "sample_id": row["sample_id"],
                "operation_count": sum(counts.values()),
                "v1_direct_coverage": _coverage(counts, V1_DIRECT),
                "v1_estimated_coverage": _coverage(counts, V1_DIRECT | V1_APPROXIMABLE),
                "v2_direct_coverage": _coverage(counts, V2_DIRECT),
                "v2_estimated_coverage": _coverage(counts, V2_DIRECT | V2_APPROXIMABLE),
                "v1_fully_representable_estimate": set(counts) <= V1_DIRECT | V1_APPROXIMABLE,
                "v2_fully_representable_estimate": set(counts) <= V2_DIRECT | V2_APPROXIMABLE,
                "v2_unsupported": dict(
                    counts - Counter({key: counts[key] for key in V2_DIRECT | V2_APPROXIMABLE})
                ),
            }
        )
    denominator = sum(total.values())
    report: dict[str, Any] = {
        "schema_version": "rq2.expressivity_audit.v1",
        "sample_count": len(manifest),
        "operation_count": denominator,
        "common_operations": total.most_common(50),
        "coverage": {
            "plan_v1_direct": _coverage(total, V1_DIRECT),
            "plan_v1_with_simple_approximations": _coverage(total, V1_DIRECT | V1_APPROXIMABLE),
            "plan_v2_direct": _coverage(total, V2_DIRECT),
            "plan_v2_with_simple_approximations": _coverage(total, V2_DIRECT | V2_APPROXIMABLE),
            "plan_v1_fully_representable_samples": sum(
                bool(item["v1_fully_representable_estimate"]) for item in samples
            ),
            "plan_v2_fully_representable_samples": sum(
                bool(item["v2_fully_representable_estimate"]) for item in samples
            ),
            "note": "静态调用覆盖是表达上限代理，不等价于可无损转换；oracle Plan 几何评分仍是最终依据。",
        },
        "samples": samples,
        "oracle_plans": [],
    }
    if oracle_plan_dir:
        by_id = {row["sample_id"]: row for row in manifest}
        for plan_path in sorted(oracle_plan_dir.rglob("*.json")):
            try:
                value = json.loads(plan_path.read_text(encoding="utf-8"))
                plan = value.get("plan", value) if isinstance(value, dict) else value
                issues = validate_plan(plan)
                sample_id = plan.get("sample_id") if isinstance(plan, dict) else None
                item: dict[str, Any] = {
                    "path": str(plan_path.resolve()),
                    "sample_id": sample_id,
                    "valid": not issues,
                    "issues": issues,
                    "score": float(not issues),
                }
                result_step = plan_path.with_name("result.step")
                if not result_step.is_file():
                    result_step = plan_path.with_suffix(".step")
                if not issues and sample_id in by_id and result_step.is_file():
                    item["geometry"] = score_step_pair(
                        result_step,
                        by_id[sample_id]["step"]["path"],
                        n_points=int(config["scoring"]["point_samples"]),
                        seed=int(config["seed"]),
                        voxel_resolution=int(config["scoring"]["voxel_resolution"]),
                        tau=float(config["scoring"]["failure_aware_tau"]),
                    )
                    item["score"] = item["geometry"]["joint_quality"]
                report["oracle_plans"].append(item)
            except Exception as exc:
                report["oracle_plans"].append({"path": str(plan_path), "valid": False, "score": 0.0, "error": str(exc)[:300]})
    target = output or output_dir / "audits" / "expressivity.json"
    atomic_write_json(target, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="审计 GT CadQuery 对 Harness Plan 的表达性")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"))
    parser.add_argument("--oracle-plan-dir")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = audit(
        load_config(args.config),
        Path(args.oracle_plan_dir).resolve() if args.oracle_plan_dir else None,
        Path(args.output).resolve() if args.output else None,
    )
    print(json.dumps({"sample_count": report["sample_count"], "coverage": report["coverage"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
