"""离线回填文献对齐指标（F1@0.01、shape-only voxel IoU、IR 口径）。

对现有 state 中已完成的任务，用当时保存的预测 STEP 与 manifest 中的 GT STEP
重新执行确定性评分，把新增指标写回 state["geometry"]。整个过程不调用模型 API。

安全策略：
- 旧指标（shape_only_cd / common_frame_cd / voxel_iou / joint_quality）必须能
  逐位复现，任何不一致的任务只记录、不覆盖；
- 写回使用原子写入，支持断点续跑（已有 metrics_version 的任务默认跳过）；
- --check-only 只核对可复现性，不修改任何文件。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import atomic_write_json, load_config, project_path, read_jsonl
from rq2_harness.geometry import score_step_pair

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"
COMPARED_FIELDS = ("shape_only_cd", "common_frame_cd", "joint_quality", "voxel_iou")


def _is_close(a, b, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def _compare_geometry(old: dict, new: dict) -> list[str]:
    mismatches = []
    for key in COMPARED_FIELDS[:3]:
        if not _is_close(old.get(key), new.get(key)):
            mismatches.append(key)
    if not _is_close((old.get("voxel_iou") or {}).get("value"), (new.get("voxel_iou") or {}).get("value")):
        mismatches.append("voxel_iou")
    return mismatches


def _result_step(state: dict) -> str | None:
    path = state.get("result_step_path")
    if not path:
        path = (state.get("episode") or {}).get("result_step_path")
    return path or None


def _failure_patch(reason: str) -> dict:
    return {
        "fscore_shape": None,
        "fscore_common": None,
        "shape_voxel_iou": {"status": "not_computed", "value": None, "reason": reason},
        "metrics_version": "rq2.geometry.v2",
    }


def backfill(
    config: dict,
    *,
    check_only: bool = False,
    limit: int | None = None,
    conditions: list[str] | None = None,
    force: bool = False,
) -> dict:
    output_dir = project_path(config["paths"]["output_dir"])
    state_dir = output_dir / "state"
    manifest_path = output_dir / "manifest.jsonl"
    gt_steps = {row["sample_id"]: row["step"]["path"] for row in read_jsonl(manifest_path)}
    selected = tuple(conditions or config["conditions"])
    n_points = int(config["scoring"]["point_samples"])
    seed = int(config["seed"])
    voxel_resolution = int(config["scoring"]["voxel_resolution"])
    tau = float(config["scoring"]["failure_aware_tau"])
    fscore_tau = float(config["scoring"].get("fscore_tau", 0.01))

    counts = {
        "checked": 0,
        "updated": 0,
        "patched_failure": 0,
        "skipped_uptodate": 0,
        "skipped_mismatch": 0,
        "skipped_no_gt_step": 0,
        "skipped_not_selected": 0,
    }
    mismatches: list[dict] = []

    paths = sorted(state_dir.glob("*/*.json"))
    if limit is not None:
        paths = paths[: int(limit)]
    for path in paths:
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("status") in {"dry_run", "running"}:
            continue
        condition = state.get("condition")
        if condition not in selected:
            counts["skipped_not_selected"] += 1
            continue
        geometry = state.get("geometry") or {}
        gt_step = gt_steps.get(state.get("sample_id"))
        if not gt_step:
            counts["skipped_no_gt_step"] += 1
            continue
        pred_step = _result_step(state)
        if pred_step and Path(pred_step).exists() and Path(gt_step).exists():
            scoring = score_step_pair(
                pred_step,
                gt_step,
                n_points=n_points,
                seed=seed,
                voxel_resolution=voxel_resolution,
                tau=tau,
                fscore_tau=fscore_tau,
            )
            old_valid = bool(geometry.get("valid"))
            new_valid = bool(scoring["valid"])
            if old_valid and new_valid:
                counts["checked"] += 1
                fields = _compare_geometry(geometry, scoring)
                if fields:
                    counts["skipped_mismatch"] += 1
                    mismatches.append(
                        {"sample_id": state["sample_id"], "condition": condition, "fields": fields}
                    )
                    continue
            elif old_valid != new_valid:
                counts["skipped_mismatch"] += 1
                mismatches.append(
                    {
                        "sample_id": state["sample_id"],
                        "condition": condition,
                        "fields": [f"valid_flip:{old_valid}->{new_valid}"],
                    }
                )
                continue
            if check_only:
                continue
            if force or geometry.get("metrics_version") != scoring.get("metrics_version"):
                state["geometry"] = scoring
                atomic_write_json(path, state)
                counts["updated"] += 1
            else:
                counts["skipped_uptodate"] += 1
        else:
            if check_only:
                continue
            if geometry.get("metrics_version") and not force:
                counts["skipped_uptodate"] += 1
                continue
            if geometry.get("valid"):
                counts["skipped_mismatch"] += 1
                mismatches.append(
                    {
                        "sample_id": state["sample_id"],
                        "condition": condition,
                        "fields": ["missing_step_file_for_valid_geometry"],
                    }
                )
                continue
            reason = "no_result_step" if not pred_step else "missing_step_file"
            state["geometry"] = {**geometry, **_failure_patch(reason)}
            atomic_write_json(path, state)
            counts["patched_failure"] += 1
    return {"counts": counts, "mismatches": mismatches}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线回填 F1@0.01 / shape-only IoU / IR 口径指标")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--conditions", help="空格或逗号分隔的条件，默认取配置全部")
    parser.add_argument("--limit", type=int, help="只处理 state 目录前 N 个任务文件")
    parser.add_argument("--check-only", action="store_true", help="只核对旧指标可复现性，不写文件")
    parser.add_argument("--force", action="store_true", help="即使已有 metrics_version 也重写")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    conditions = None
    if args.conditions:
        conditions = [item.strip() for item in args.conditions.replace(",", " ").split() if item.strip()]
    result = backfill(
        config,
        check_only=args.check_only,
        limit=args.limit,
        conditions=conditions,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result["mismatches"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
