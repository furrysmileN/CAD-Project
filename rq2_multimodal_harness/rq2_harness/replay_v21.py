from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .backend import run_episode
from .common import (
    atomic_write_json,
    load_config,
    project_path,
    read_jsonl,
    safe_id,
    sha256_file,
    sha256_json,
    state_path,
)
from .geometry import invalid_metrics, score_step_pair
from .prompting import parse_plan_response, validate_plan
from .repair_v21 import REPAIR_VERSION, repair_plan_v21


SUCCESS_STATUSES = {"success", "success_with_warnings"}
SETTING_ORDER = ("R0", "R1", "R2", "R3", "R4")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tree_snapshot(state_dir: Path) -> dict[str, Any]:
    files = sorted(state_dir.glob("*/*.json"))
    hashes = {path.relative_to(state_dir).as_posix(): sha256_file(path) for path in files}
    return {"count": len(files), "sha256": sha256_json(hashes), "files": hashes}


def _code_fingerprint(config: dict[str, Any]) -> str:
    backend_root = project_path(config["backend"]["root"])
    paths = [
        Path(__file__),
        Path(__file__).with_name("repair_v21.py"),
        Path(__file__).with_name("prompting.py"),
        Path(__file__).with_name("geometry.py"),
        Path(__file__).with_name("backend.py"),
        backend_root / "backend" / "harness_api_v2.py",
        backend_root / "backend" / "plan_v2_schema.py",
        backend_root / "backend" / "plan_v2_compiler.py",
        Path(config["_config_path"]),
    ]
    return sha256_json({str(path.resolve()): sha256_file(path) for path in paths})


def _scoring_fingerprint(config: dict[str, Any]) -> str:
    return sha256_json(
        {
            "backend": config["backend"],
            "scoring": config["scoring"],
            "seed": config["seed"],
            "repair_version": REPAIR_VERSION,
        }
    )


def _compact_episode(episode: dict[str, Any]) -> dict[str, Any]:
    response = episode.get("response") or {}
    keep = {
        "traceVersion",
        "runId",
        "createdAtUtc",
        "status",
        "failure",
        "warnings",
        "validation",
        "metrics",
        "provenance",
        "totalDurationSec",
        "artifactManifest",
        "error",
    }
    return {
        "backend_version": episode.get("backend_version"),
        "run_dir": episode.get("run_dir"),
        "episode_path": episode.get("episode_path"),
        "result_step_path": episode.get("result_step_path"),
        "response": {key: response.get(key) for key in keep if key in response},
    }


def _classify(
    parsed: dict[str, Any],
    *,
    expected_sample_id: str,
    episode: dict[str, Any] | None,
    geometry: dict[str, Any] | None,
) -> dict[str, Any]:
    parse_ok = bool(parsed.get("ok", False))
    sample_id_match = bool(parse_ok and (parsed.get("plan") or {}).get("sample_id") == expected_sample_id)
    response = (episode or {}).get("response") or {}
    validation = response.get("validation") or {}
    episode_status = response.get("status")
    execution_success = episode_status in SUCCESS_STATUSES
    geometry_valid = bool((geometry or {}).get("valid", False))
    if not parse_ok or not sample_id_match:
        status = "parse_failed"
    elif execution_success:
        status = "completed"
    else:
        status = "episode_failed"
    return {
        "status": status,
        "parse_ok": parse_ok and sample_id_match,
        "sample_id_match": sample_id_match,
        "schema_valid": bool(validation.get("valid", False)),
        "execution_success": execution_success,
        "geometry_valid": geometry_valid,
        "episode_status": episode_status,
        "joint_quality": float((geometry or {}).get("joint_quality") or 0.0),
    }


def _execute_from_parsed(
    parsed: dict[str, Any],
    *,
    expected_sample_id: str,
    manifest_row: dict[str, Any],
    config: dict[str, Any],
    run_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not parsed.get("ok") or (parsed.get("plan") or {}).get("sample_id") != expected_sample_id:
        return None, invalid_metrics("parse_or_sample_id_failed")
    episode = run_episode(parsed["plan"], config["backend"], run_root=run_root)
    result_step = episode.get("result_step_path")
    if not result_step:
        return _compact_episode(episode), invalid_metrics("backend_no_result_step")
    scoring = config["scoring"]
    geometry = score_step_pair(
        result_step,
        manifest_row["step"]["path"],
        n_points=int(scoring["point_samples"]),
        seed=int(config["seed"]),
        voxel_resolution=int(scoring["voxel_resolution"]),
        tau=float(scoring["failure_aware_tau"]),
    )
    return _compact_episode(episode), geometry


def _original_stage(state: dict[str, Any]) -> dict[str, Any]:
    parse = state.get("parse") or {}
    episode = state.get("episode")
    geometry = state.get("geometry")
    expected = ((state.get("prompt_audit") or {}).get("prompt_sample_id") or "")
    return _classify(parse, expected_sample_id=expected, episode=episode, geometry=geometry)


def _task_fingerprint(
    *,
    baseline_state_sha256: str,
    raw_response_sha256: str,
    setting: str,
    plan_sha256: str | None,
    code_fingerprint: str,
    scoring_fingerprint: str,
) -> str:
    return sha256_json(
        {
            "baseline_state_sha256": baseline_state_sha256,
            "raw_response_sha256": raw_response_sha256,
            "setting": setting,
            "plan_sha256": plan_sha256,
            "code_fingerprint": code_fingerprint,
            "scoring_fingerprint": scoring_fingerprint,
        }
    )


def _empty_repair_log(rules: Iterable[str]) -> dict[str, Any]:
    digest = sha256_json(None)
    return {
        "repair_version": REPAIR_VERSION,
        "rules": list(rules),
        "changed": False,
        "before_sha256": digest,
        "after_sha256": digest,
        "repair_codes": [],
        "changed_paths": [],
        "repair_count": 0,
    }


def _parse_for_setting(raw_response: str, rules: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    base = parse_plan_response(raw_response, plan_version="v2")
    plan = base.get("plan")
    if not isinstance(plan, dict):
        return base, _empty_repair_log(rules)
    repaired, repair_log = repair_plan_v21(plan, rules)
    issues = validate_plan(repaired, plan_version="v2")
    parsed = {
        "ok": not issues,
        "plan": repaired,
        "issues": issues,
        "repair": base.get("repair"),
    }
    return parsed, repair_log


def _compare_reproduction(
    original: dict[str, Any],
    reproduced: dict[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    original_stage = _original_stage(original)
    reproduced_stage = reproduced["stage"]
    original_plan = (original.get("parse") or {}).get("plan")
    reproduced_plan = (reproduced.get("parse") or {}).get("plan")
    original_plan_sha = sha256_json(original_plan) if isinstance(original_plan, dict) else None
    reproduced_plan_sha = sha256_json(reproduced_plan) if isinstance(reproduced_plan, dict) else None
    quality_delta = reproduced_stage["joint_quality"] - original_stage["joint_quality"]
    return {
        "sample_id": original["sample_id"],
        "condition": original["condition"],
        "original_state": original_stage["status"],
        "reproduced_state": reproduced_stage["status"],
        "status_match": original_stage["status"] == reproduced_stage["status"],
        "parse_match": original_stage["parse_ok"] == reproduced_stage["parse_ok"],
        "schema_match": original_stage["schema_valid"] == reproduced_stage["schema_valid"],
        "execution_match": original_stage["execution_success"] == reproduced_stage["execution_success"],
        "geometry_valid_match": original_stage["geometry_valid"] == reproduced_stage["geometry_valid"],
        "joint_quality_match": abs(quality_delta) <= tolerance,
        "joint_quality_delta": quality_delta,
        "raw_response_sha256": reproduced["raw_response_sha256"],
        "parsed_plan_sha256": reproduced_plan_sha,
        "original_parsed_plan_sha256": original_plan_sha,
        "plan_hash_match": original_plan_sha == reproduced_plan_sha,
        "original_episode_path": ((original.get("episode") or {}).get("episode_path")),
        "reproduced_episode_path": ((reproduced.get("episode") or {}).get("episode_path")),
    }


def reproduce_baseline(
    config: dict[str, Any],
    *,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    baseline_state_dir = project_path(config["baseline"]["state_dir"]).resolve()
    baseline_manifest = project_path(config["baseline"]["manifest"]).resolve()
    output_dir = project_path(config["paths"]["output_dir"]).resolve()
    baseline_output = project_path(config["baseline"]["output_dir"]).resolve()
    if output_dir == baseline_output or baseline_output in output_dir.parents:
        raise ValueError("离线重放输出目录不得等于或位于冻结 baseline 目录内")
    output_dir.mkdir(parents=True, exist_ok=True)

    before_snapshot = _tree_snapshot(baseline_state_dir)
    expected = int(config["baseline"]["expected_tasks"])
    if limit is None and before_snapshot["count"] != expected:
        raise RuntimeError(f"冻结 baseline state 数量为 {before_snapshot['count']}，预期 {expected}")
    manifest = {row["sample_id"]: row for row in read_jsonl(baseline_manifest)}
    paths = sorted(baseline_state_dir.glob("*/*.json"))
    if limit is not None:
        paths = paths[:limit]
    code_fp = _code_fingerprint(config)
    scoring_fp = _scoring_fingerprint(config)
    reproduction_dir = output_dir / "reproduction_state"
    run_root = output_dir / "runs" / "baseline_reproduction"
    rows: list[dict[str, Any]] = []

    for index, source_path in enumerate(paths):
        original = json.loads(source_path.read_text(encoding="utf-8"))
        sample_id = str(original["sample_id"])
        condition = str(original["condition"])
        if sample_id not in manifest:
            raise RuntimeError(f"manifest 缺少 baseline 样本 {sample_id}")
        raw = original.get("raw_response")
        if not isinstance(raw, str):
            raise RuntimeError(f"{source_path} 缺少 raw_response")
        baseline_sha = sha256_file(source_path)
        raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        parsed, repair_log = _parse_for_setting(raw, [])
        plan = parsed.get("plan")
        plan_sha = sha256_json(plan) if isinstance(plan, dict) else None
        fingerprint = _task_fingerprint(
            baseline_state_sha256=baseline_sha,
            raw_response_sha256=raw_sha,
            setting="baseline_reproduction",
            plan_sha256=plan_sha,
            code_fingerprint=code_fp,
            scoring_fingerprint=scoring_fp,
        )
        destination = state_path(reproduction_dir, sample_id, condition)
        reproduced: dict[str, Any]
        if destination.is_file() and not force:
            cached = json.loads(destination.read_text(encoding="utf-8"))
            if cached.get("task_fingerprint") == fingerprint and cached.get("status") in {
                "completed",
                "parse_failed",
                "episode_failed",
            }:
                reproduced = cached
            else:
                reproduced = {}
        else:
            reproduced = {}
        if not reproduced:
            episode, geometry = _execute_from_parsed(
                parsed,
                expected_sample_id=(original.get("prompt_audit") or {}).get("prompt_sample_id", ""),
                manifest_row=manifest[sample_id],
                config=config,
                run_root=run_root,
            )
            stage = _classify(
                parsed,
                expected_sample_id=(original.get("prompt_audit") or {}).get("prompt_sample_id", ""),
                episode=episode,
                geometry=geometry,
            )
            reproduced = {
                "schema_version": "rq2.replay_task.v1",
                "mode": "baseline_reproduction",
                "task_index": index,
                "sample_id": sample_id,
                "condition": condition,
                "status": stage["status"],
                "baseline_state_path": str(source_path.resolve()),
                "baseline_state_sha256": baseline_sha,
                "raw_response_sha256": raw_sha,
                "parsed_plan_sha256": plan_sha,
                "task_fingerprint": fingerprint,
                "parse": parsed,
                "repair_v21": repair_log,
                "episode": episode,
                "geometry": geometry,
                "result_step_path": (episode or {}).get("result_step_path"),
                "stage": stage,
            }
            atomic_write_json(destination, reproduced)
        rows.append(
            _compare_reproduction(
                original,
                reproduced,
                tolerance=float(config["reproduction"]["joint_quality_abs_tolerance"]),
            )
        )

    after_snapshot = _tree_snapshot(baseline_state_dir)
    frozen_unchanged = before_snapshot["sha256"] == after_snapshot["sha256"]
    checks = {
        "task_count": len(rows),
        "all_raw_responses_found": len(rows) == len(paths),
        "status_match_count": sum(bool(row["status_match"]) for row in rows),
        "parse_match_count": sum(bool(row["parse_match"]) for row in rows),
        "schema_match_count": sum(bool(row["schema_match"]) for row in rows),
        "execution_match_count": sum(bool(row["execution_match"]) for row in rows),
        "geometry_valid_match_count": sum(bool(row["geometry_valid_match"]) for row in rows),
        "plan_hash_match_count": sum(bool(row["plan_hash_match"]) for row in rows),
        "joint_quality_match_count": sum(bool(row["joint_quality_match"]) for row in rows),
        "baseline_frozen_unchanged": frozen_unchanged,
    }
    required_count = len(rows)
    gate_passed = bool(
        frozen_unchanged
        and checks["all_raw_responses_found"]
        and checks["status_match_count"] == required_count
        and checks["parse_match_count"] == required_count
        and checks["schema_match_count"] == required_count
        and checks["execution_match_count"] == required_count
        and checks["geometry_valid_match_count"] == required_count
        and checks["plan_hash_match_count"] == required_count
        and checks["joint_quality_match_count"] == required_count
    )
    summary = {
        "schema_version": "rq2.baseline_reproduction.v1",
        "gate_passed": gate_passed,
        "limited": limit is not None,
        "checks": checks,
        "baseline_snapshot_before": {
            "count": before_snapshot["count"],
            "sha256": before_snapshot["sha256"],
        },
        "baseline_snapshot_after": {
            "count": after_snapshot["count"],
            "sha256": after_snapshot["sha256"],
        },
        "code_fingerprint": code_fp,
        "scoring_fingerprint": scoring_fp,
        "mismatches": [
            row
            for row in rows
            if not all(
                row[key]
                for key in (
                    "status_match",
                    "parse_match",
                    "schema_match",
                    "execution_match",
                    "geometry_valid_match",
                    "plan_hash_match",
                    "joint_quality_match",
                )
            )
        ],
    }
    _write_csv(output_dir / "baseline_reproduction.csv", rows)
    atomic_write_json(output_dir / "baseline_reproduction_summary.json", summary)
    atomic_write_json(
        output_dir / "replay_meta.json",
        {
            "schema_version": "rq2.replay_meta.v1",
            "repair_version": REPAIR_VERSION,
            "baseline_output": str(baseline_output),
            "baseline_manifest_sha256": sha256_file(baseline_manifest),
            "baseline_state_snapshot_sha256": before_snapshot["sha256"],
            "code_fingerprint": code_fp,
            "scoring_fingerprint": scoring_fp,
            "python_version": sys.version,
        },
    )
    return summary


def _load_reproduction_index(output_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for path in sorted((output_dir / "reproduction_state").glob("*/*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        result[(state["sample_id"], state["condition"])] = state
    return result


def _reuse_state(
    source: dict[str, Any],
    *,
    parsed: dict[str, Any],
    repair_log: dict[str, Any],
    setting: str,
    source_label: str,
    task_fingerprint: str,
    baseline_state_path: Path,
    baseline_state_sha256: str,
    raw_sha: str,
    plan_sha: str | None,
) -> dict[str, Any]:
    episode = source.get("episode")
    geometry = source.get("geometry") or invalid_metrics("reused_invalid_geometry")
    expected = ((source.get("prompt_audit") or {}).get("prompt_sample_id")) or (
        ((json.loads(baseline_state_path.read_text(encoding="utf-8")).get("prompt_audit") or {}).get("prompt_sample_id"))
    )
    stage = _classify(parsed, expected_sample_id=expected or "", episode=episode, geometry=geometry)
    return {
        "schema_version": "rq2.replay_task.v1",
        "mode": "repair_ablation",
        "setting": setting,
        "sample_id": source["sample_id"],
        "condition": source["condition"],
        "status": stage["status"],
        "baseline_state_path": str(baseline_state_path.resolve()),
        "baseline_state_sha256": baseline_state_sha256,
        "raw_response_sha256": raw_sha,
        "parsed_plan_sha256": plan_sha,
        "task_fingerprint": task_fingerprint,
        "parse": parsed,
        "repair_v21": repair_log,
        "episode": episode,
        "geometry": geometry,
        "result_step_path": source.get("result_step_path") or (episode or {}).get("result_step_path"),
        "stage": stage,
        "execution_mode": "reused",
        "reused_from": source_label,
    }


def run_ablation(
    config: dict[str, Any],
    *,
    settings: Iterable[str] = SETTING_ORDER,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir = project_path(config["paths"]["output_dir"]).resolve()
    reproduction_summary_path = output_dir / "baseline_reproduction_summary.json"
    if not reproduction_summary_path.is_file():
        raise RuntimeError("缺少 baseline_reproduction_summary.json，请先运行基线复现")
    reproduction_summary = json.loads(reproduction_summary_path.read_text(encoding="utf-8"))
    if not reproduction_summary.get("gate_passed"):
        raise RuntimeError("基线复现门禁未通过，禁止执行 R1-R4")
    baseline_state_dir = project_path(config["baseline"]["state_dir"]).resolve()
    manifest = {row["sample_id"]: row for row in read_jsonl(project_path(config["baseline"]["manifest"]))}
    paths = sorted(baseline_state_dir.glob("*/*.json"))
    if limit is not None:
        paths = paths[:limit]
    selected = tuple(dict.fromkeys(settings))
    invalid_settings = set(selected) - set(SETTING_ORDER)
    if invalid_settings:
        raise ValueError(f"未知设置: {sorted(invalid_settings)}")
    reproduction = _load_reproduction_index(output_dir)
    code_fp = _code_fingerprint(config)
    scoring_fp = _scoring_fingerprint(config)
    plan_cache: dict[tuple[str, str, str | None, bool], tuple[dict[str, Any], str]] = {}
    counts: dict[str, Counter[str]] = {setting: Counter() for setting in selected}

    for setting in selected:
        rules = list(config["repair"]["settings"][setting])
        for source_path in paths:
            baseline = json.loads(source_path.read_text(encoding="utf-8"))
            sample_id = str(baseline["sample_id"])
            condition = str(baseline["condition"])
            raw = baseline["raw_response"]
            raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            baseline_sha = sha256_file(source_path)
            parsed, repair_log = _parse_for_setting(raw, rules)
            plan = parsed.get("plan")
            plan_sha = sha256_json(plan) if isinstance(plan, dict) else None
            fingerprint = _task_fingerprint(
                baseline_state_sha256=baseline_sha,
                raw_response_sha256=raw_sha,
                setting=setting,
                plan_sha256=plan_sha,
                code_fingerprint=code_fp,
                scoring_fingerprint=scoring_fp,
            )
            destination = state_path(output_dir / "repair_state" / setting, sample_id, condition)
            replay_state: dict[str, Any] | None = None
            if destination.is_file() and not force:
                cached = json.loads(destination.read_text(encoding="utf-8"))
                if cached.get("task_fingerprint") == fingerprint and cached.get("status") in {
                    "completed",
                    "parse_failed",
                    "episode_failed",
                }:
                    replay_state = cached
            if replay_state is None:
                base_reproduction = reproduction[(sample_id, condition)]
                base_plan_sha = base_reproduction.get("parsed_plan_sha256")
                if plan_sha == base_plan_sha and parsed.get("ok") == (base_reproduction.get("parse") or {}).get("ok"):
                    replay_state = _reuse_state(
                        base_reproduction,
                        parsed=parsed,
                        repair_log=repair_log,
                        setting=setting,
                        source_label="baseline_reproduction",
                        task_fingerprint=fingerprint,
                        baseline_state_path=source_path,
                        baseline_state_sha256=baseline_sha,
                        raw_sha=raw_sha,
                        plan_sha=plan_sha,
                    )
                else:
                    cache_key = (sample_id, condition, plan_sha, bool(parsed.get("ok")))
                    if cache_key in plan_cache:
                        cached_source, label = plan_cache[cache_key]
                        replay_state = _reuse_state(
                            cached_source,
                            parsed=parsed,
                            repair_log=repair_log,
                            setting=setting,
                            source_label=label,
                            task_fingerprint=fingerprint,
                            baseline_state_path=source_path,
                            baseline_state_sha256=baseline_sha,
                            raw_sha=raw_sha,
                            plan_sha=plan_sha,
                        )
                    else:
                        episode, geometry = _execute_from_parsed(
                            parsed,
                            expected_sample_id=(baseline.get("prompt_audit") or {}).get("prompt_sample_id", ""),
                            manifest_row=manifest[sample_id],
                            config=config,
                            run_root=output_dir / "runs" / setting,
                        )
                        stage = _classify(
                            parsed,
                            expected_sample_id=(baseline.get("prompt_audit") or {}).get("prompt_sample_id", ""),
                            episode=episode,
                            geometry=geometry,
                        )
                        replay_state = {
                            "schema_version": "rq2.replay_task.v1",
                            "mode": "repair_ablation",
                            "setting": setting,
                            "sample_id": sample_id,
                            "condition": condition,
                            "status": stage["status"],
                            "baseline_state_path": str(source_path.resolve()),
                            "baseline_state_sha256": baseline_sha,
                            "raw_response_sha256": raw_sha,
                            "parsed_plan_sha256": plan_sha,
                            "task_fingerprint": fingerprint,
                            "parse": parsed,
                            "repair_v21": repair_log,
                            "episode": episode,
                            "geometry": geometry,
                            "result_step_path": (episode or {}).get("result_step_path"),
                            "stage": stage,
                            "execution_mode": "executed",
                            "reused_from": None,
                        }
                        plan_cache[cache_key] = (replay_state, setting)
                atomic_write_json(destination, replay_state)
            counts[setting][replay_state["status"]] += 1

    final_snapshot = _tree_snapshot(baseline_state_dir)
    meta = json.loads((output_dir / "replay_meta.json").read_text(encoding="utf-8"))
    if final_snapshot["sha256"] != meta["baseline_state_snapshot_sha256"]:
        raise RuntimeError("冻结 baseline state 在离线重放期间发生变化")
    summary = {
        "schema_version": "rq2.repair_ablation_run.v1",
        "repair_version": REPAIR_VERSION,
        "settings": list(selected),
        "tasks_per_setting": len(paths),
        "limited": limit is not None,
        "counts": {setting: dict(counts[setting]) for setting in selected},
        "baseline_snapshot_unchanged": True,
        "api_calls": 0,
    }
    atomic_write_json(output_dir / "run_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan v2.1 安全修复与离线重放（不调用模型 API）")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "replay_v21.yaml"))
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--settings", nargs="*", default=list(SETTING_ORDER))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    started = time.perf_counter()
    if not args.skip_baseline:
        reproduction = reproduce_baseline(config, force=args.force, limit=args.limit)
        print(json.dumps({"baseline": reproduction["checks"], "gate_passed": reproduction["gate_passed"]}, ensure_ascii=False))
        if not reproduction["gate_passed"]:
            return 2
    if args.baseline_only:
        return 0
    result = run_ablation(
        config,
        settings=args.settings,
        force=args.force,
        limit=args.limit,
    )
    result["elapsed_sec"] = time.perf_counter() - started
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
