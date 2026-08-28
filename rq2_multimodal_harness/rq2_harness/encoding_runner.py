from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .api_client import APISettings, FatalAPIError, chat_completion
from .backend import run_episode
from .common import (
    atomic_write_json,
    load_config,
    project_path,
    read_jsonl,
    safe_id,
    sha256_file,
    sha256_json,
)
from .encoding_conditions import CONDITIONS, ConditionSpec, parse_condition
from .encoding_prompting import build_encoding_messages
from .feedback import (
    EXECUTION_SOURCE,
    FEEDBACK_VERSION,
    SCHEMA_SOURCE,
    build_execution_feedback,
    build_schema_feedback,
    failure_kind_from_code,
    feedback_turn,
    resolve_feedback_config,
)
from .geometry import score_step_pair
from .prompting import parse_plan_response, validate_plan
from .repair_v21 import REPAIR_VERSION, repair_plan_v21


STATE_SCHEMA = "rq2.encoding_screen.task_state.v1"
RUN_SCHEMA = "rq2.encoding_screen.run_summary.v1"
TERMINAL_STATUSES = {
    "completed",
    "parse_failed",
    "episode_failed",
}


def _condition_id(spec: ConditionSpec) -> str:
    value = getattr(spec, "condition_id")
    return str(value() if callable(value) else value)


def _condition_dict(spec: ConditionSpec) -> dict[str, str | None]:
    return {
        "text": getattr(spec, "text"),
        "render": getattr(spec, "render"),
        "point": getattr(spec, "point"),
    }


def _resolve_conditions(raw: Iterable[str] | None) -> tuple[ConditionSpec, ...]:
    if not raw:
        return tuple(CONDITIONS)
    values: list[str] = []
    for item in raw:
        values.extend(part.strip() for part in item.split(",") if part.strip())
    specs = tuple(parse_condition(value) for value in values)
    ids = [_condition_id(spec) for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("conditions 包含重复项")
    return specs


def _task_order(
    rows: list[dict[str, Any]],
    conditions: tuple[ConditionSpec, ...],
    seed: int,
) -> list[tuple[dict[str, Any], ConditionSpec]]:
    tasks = [(row, condition) for row in rows for condition in conditions]
    tasks.sort(
        key=lambda task: hashlib.sha256(
            f"{seed}:{task[0]['sample_id']}:{_condition_id(task[1])}".encode("utf-8")
        ).hexdigest()
    )
    return tasks


def _code_fingerprint() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    names = (
        "encoding_conditions.py",
        "encoding_prompting.py",
        "encoding_runner.py",
        "feedback.py",
        "repair_v21.py",
        "geometry.py",
        "backend.py",
    )
    files = {
        name: sha256_file(root / name)
        for name in names
        if (root / name).is_file()
    }
    return {"files": files, "sha256": sha256_json(files)}


def _condition_input_hash(
    row: dict[str, Any],
    spec: ConditionSpec,
    prompt_audit: dict[str, Any],
) -> str:
    return sha256_json(
        {
            "sample_id": row["sample_id"],
            "condition": _condition_dict(spec),
            "modalities": prompt_audit["modality_hashes"],
            "gt_step": (row.get("step") or {}).get("sha256"),
        }
    )


def _state_path(output_dir: Path, sample_id: str, condition_id: str) -> Path:
    return output_dir / "state" / safe_id(sample_id) / f"{safe_id(condition_id)}.json"


def _should_skip(previous: dict[str, Any], task_fingerprint: str, dry_run: bool) -> bool:
    if previous.get("task_fingerprint") != task_fingerprint:
        return False
    if dry_run:
        return previous.get("status") == "dry_run"
    return (
        previous.get("status") in TERMINAL_STATUSES
        and previous.get("status") != "dry_run"
    )


def _archive_previous(
    output_dir: Path,
    previous: dict[str, Any],
    sample_id: str,
    condition_id: str,
) -> None:
    target = (
        output_dir
        / "history"
        / safe_id(sample_id)
        / safe_id(condition_id)
        / (
            f"{time.time_ns()}__"
            f"{str(previous.get('task_fingerprint') or 'nohash')[:12]}__"
            f"{safe_id(str(previous.get('status') or 'unknown'))}.json"
        )
    )
    atomic_write_json(target, previous)


def run_encoding_screen(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    limit: int | None = None,
    conditions: tuple[ConditionSpec, ...] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = project_path(config["paths"]["output_dir"])
    manifest_path = output_dir / "sample_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"缺少冻结样本清单 {manifest_path}；请先运行 prepare_encoding_screen"
        )
    rows = list(read_jsonl(manifest_path))
    expected_n = int(config["n"])
    if len(rows) != expected_n:
        raise RuntimeError(f"sample_manifest 应有 {expected_n} 行，实际 {len(rows)}")
    if limit is not None:
        rows = rows[:limit]
    selected = conditions or tuple(CONDITIONS)
    if not selected:
        raise ValueError("至少需要一个条件")

    api_settings = APISettings.from_config(config["api"])
    if not dry_run:
        api_settings.resolved(require_key=True)
    seed = int(config["seed"])
    image_max_edge = int(config["modalities"]["image_max_edge"])
    repair_rules = tuple(config.get("repair", {}).get("rules") or ())
    feedback = resolve_feedback_config(config)
    plan_prompt_version = str(feedback.get("plan_prompt_version") or "v2")
    code_fingerprint = _code_fingerprint()
    scoring_fingerprint = sha256_json(config["scoring"])
    feedback_fingerprint = sha256_json(
        {
            "plan_prompt_version": plan_prompt_version,
            "feedback": {
                key: feedback.get(key)
                for key in (
                    "arm",
                    "enabled",
                    "max_rounds",
                    "sources",
                    "round2_temperature",
                    "keep_best",
                )
            },
        }
    )
    tasks = _task_order(rows, selected, seed)
    task_manifest = [
        {
            "order": index,
            "sample_id": row["sample_id"],
            "condition_id": _condition_id(spec),
            "condition": _condition_dict(spec),
        }
        for index, (row, spec) in enumerate(tasks)
    ]
    atomic_write_json(
        output_dir / "task_order.json",
        {
            "schema_version": "rq2.encoding_screen.task_order.v1",
            "seed": seed,
            "expected_tasks": len(rows) * len(selected),
            "conditions": [_condition_id(spec) for spec in selected],
            "tasks": task_manifest,
        },
    )

    counts: Counter[str] = Counter()
    api_calls = 0
    started_run = time.perf_counter()
    for task_index, (row, spec) in enumerate(tasks):
        condition_id = _condition_id(spec)
        path = _state_path(output_dir, row["sample_id"], condition_id)
        started = time.perf_counter()
        messages, prompt_audit = build_encoding_messages(
            row,
            spec,
            image_max_edge=image_max_edge,
            plan_prompt_version=plan_prompt_version,
        )
        prompt_hash = sha256_json(messages)
        input_hash = _condition_input_hash(row, spec, prompt_audit)
        task_fingerprint = sha256_json(
            {
                "prompt": prompt_hash,
                "input": input_hash,
                "code": code_fingerprint["sha256"],
                "scoring": scoring_fingerprint,
                "repair_version": REPAIR_VERSION,
                "repair_rules": repair_rules,
                "feedback": feedback_fingerprint,
            }
        )
        if path.is_file() and not force:
            previous = json.loads(path.read_text(encoding="utf-8"))
            if _should_skip(previous, task_fingerprint, dry_run):
                counts["skipped"] += 1
                continue
            if not dry_run:
                _archive_previous(
                    output_dir,
                    previous,
                    row["sample_id"],
                    condition_id,
                )

        base_state: dict[str, Any] = {
            "schema_version": STATE_SCHEMA,
            "task_index": task_index,
            "sample_id": row["sample_id"],
            "condition_id": condition_id,
            "condition": _condition_dict(spec),
            "status": "running",
            "input_sha256": input_hash,
            "prompt_sha256": prompt_hash,
            "prompt_audit": prompt_audit,
            "task_fingerprint": task_fingerprint,
            "code_fingerprint": code_fingerprint,
            "scoring_fingerprint": scoring_fingerprint,
            "repair": {"version": REPAIR_VERSION, "rules": list(repair_rules)},
        }
        atomic_write_json(path, base_state)
        if dry_run:
            base_state.update(
                {
                    "status": "dry_run",
                    "elapsed_sec": time.perf_counter() - started,
                }
            )
            atomic_write_json(path, base_state)
            counts["dry_run"] += 1
            continue

        try:
            feedback_state: dict[str, Any] = {
                "version": FEEDBACK_VERSION,
                "arm": feedback.get("arm"),
                "enabled": bool(feedback["enabled"]),
                "max_rounds": int(feedback["max_rounds"]),
                "sources": list(feedback["sources"]),
                "round2_temperature": float(feedback["round2_temperature"]),
                "keep_best": bool(feedback["keep_best"]),
                "rounds": [],
            }
            outcomes: list[dict[str, Any]] = []
            current_messages = messages
            status = "task_failed"
            for round_index in range(int(feedback["max_rounds"]) + 1):
                round_temperature = (
                    api_settings.temperature
                    if round_index == 0
                    else float(feedback["round2_temperature"])
                )
                round_settings = (
                    api_settings
                    if round_temperature == api_settings.temperature
                    else replace(api_settings, temperature=round_temperature)
                )
                round_record: dict[str, Any] = {
                    "round": round_index,
                    "temperature": round_temperature,
                }
                api_result = chat_completion(current_messages, round_settings)
                api_calls += int(api_result.get("attempt") or 1)
                raw_response = api_result["text"]
                round_record["api"] = {
                    key: value for key, value in api_result.items() if key != "text"
                }
                parsed = parse_plan_response(raw_response, plan_version="v2")
                outcome: dict[str, Any] = {
                    "round": round_index,
                    "raw_response": raw_response,
                    "raw_response_sha256": hashlib.sha256(
                        raw_response.encode("utf-8")
                    ).hexdigest(),
                    "api": round_record["api"],
                    "parse": parsed,
                    "joint_quality": 0.0,
                }
                original_plan = parsed.get("plan")
                if not isinstance(original_plan, dict):
                    issues = parsed.get("issues") or []
                    round_record["failure"] = {"kind": SCHEMA_SOURCE, "issues": issues}
                    outcome["status"] = "parse_failed"
                    outcomes.append(outcome)
                    feedback_state["rounds"].append(round_record)
                    status = "parse_failed"
                    if (
                        SCHEMA_SOURCE in feedback["sources"]
                        and round_index < int(feedback["max_rounds"])
                    ):
                        current_messages = feedback_turn(
                            messages,
                            raw_response,
                            build_schema_feedback(issues, {}, round_index),
                        )
                        continue
                    break

                repaired_plan, repair_log = repair_plan_v21(
                    original_plan,
                    rules=repair_rules,
                )
                repaired_issues = validate_plan(repaired_plan, plan_version="v2")
                if repaired_plan.get("sample_id") != prompt_audit["prompt_sample_id"]:
                    repaired_issues = list(repaired_issues) + [
                        {"path": "$.sample_id", "code": "sample_id_mismatch"}
                    ]
                outcome["parsed_plan_sha256"] = sha256_json(original_plan)
                outcome["repaired_plan"] = repaired_plan
                outcome["repaired_plan_sha256"] = sha256_json(repaired_plan)
                outcome["repair_log"] = repair_log
                outcome["post_repair_issues"] = repaired_issues
                if repaired_issues:
                    round_record["failure"] = {
                        "kind": SCHEMA_SOURCE,
                        "issues": repaired_issues,
                    }
                    outcome["status"] = "parse_failed"
                    outcomes.append(outcome)
                    feedback_state["rounds"].append(round_record)
                    status = "parse_failed"
                    if (
                        SCHEMA_SOURCE in feedback["sources"]
                        and round_index < int(feedback["max_rounds"])
                    ):
                        current_messages = feedback_turn(
                            messages,
                            raw_response,
                            build_schema_feedback(
                                repaired_issues,
                                repaired_plan,
                                round_index,
                            ),
                        )
                        continue
                    break

                episode = run_episode(
                    repaired_plan,
                    config["backend"],
                    run_root=output_dir / "runs",
                )
                result_step = episode["result_step_path"]
                if result_step:
                    geometry = score_step_pair(
                        result_step,
                        row["step"]["path"],
                        n_points=int(config["scoring"]["point_samples"]),
                        seed=seed,
                        voxel_resolution=int(config["scoring"]["voxel_resolution"]),
                        tau=float(config["scoring"]["failure_aware_tau"]),
                    )
                else:
                    geometry = {
                        "valid": False,
                        "failure": "backend_no_result_step",
                        "joint_quality": 0.0,
                    }
                response_status = str(episode["response"].get("status"))
                execution_success = response_status in {
                    "success",
                    "success_with_warnings",
                }
                outcome["episode"] = episode
                outcome["geometry"] = geometry
                outcome["result_step_path"] = result_step
                outcome["operation_count"] = len(
                    repaired_plan.get("operations") or []
                )
                outcome["stage"] = {
                    "parse_ok": True,
                    "schema_valid": True,
                    "episode_status": response_status,
                    "execution_success": execution_success,
                    "geometry_valid": bool(geometry.get("valid", False)),
                }
                if execution_success:
                    outcome["status"] = "completed"
                    outcome["joint_quality"] = float(
                        geometry.get("joint_quality") or 0.0
                    )
                    outcomes.append(outcome)
                    feedback_state["rounds"].append(round_record)
                    status = "completed"
                    break
                failure = episode["response"].get("failure") or {
                    "code": response_status,
                    "message": str(episode["response"].get("error") or ""),
                }
                if failure_kind_from_code(failure.get("code")) == "format":
                    # 后端计划校验拒绝属于格式/校验类失败：记为 schema kind 并触发
                    # schema 反馈，而不是按"执行失败"处理（否则仅启用 schema 源的
                    # 臂会漏掉反馈机会，重复 RQ2b B1 的 0 触发问题）。
                    # 失败码→类别的映射与 encoding_analysis 共用
                    # feedback.failure_kind_from_code，保证 runner 与分析口径一致。
                    issues = (
                        (episode["response"].get("validation") or {}).get("issues")
                        or []
                    )
                    round_record["failure"] = {
                        "kind": SCHEMA_SOURCE,
                        "issues": issues,
                        "failure": failure,
                    }
                    outcome["status"] = "episode_failed"
                    outcomes.append(outcome)
                    feedback_state["rounds"].append(round_record)
                    status = "episode_failed"
                    if (
                        SCHEMA_SOURCE in feedback["sources"]
                        and round_index < int(feedback["max_rounds"])
                    ):
                        current_messages = feedback_turn(
                            messages,
                            raw_response,
                            build_schema_feedback(issues, repaired_plan, round_index),
                        )
                        continue
                    break
                round_record["failure"] = {
                    "kind": EXECUTION_SOURCE,
                    "failure": failure,
                }
                outcome["status"] = "episode_failed"
                outcomes.append(outcome)
                feedback_state["rounds"].append(round_record)
                status = "episode_failed"
                if (
                    EXECUTION_SOURCE in feedback["sources"]
                    and round_index < int(feedback["max_rounds"])
                ):
                    current_messages = feedback_turn(
                        messages,
                        raw_response,
                        build_execution_feedback(failure, repaired_plan, round_index),
                    )
                    continue
                break

            final_outcome = outcomes[-1] if outcomes else {}
            if feedback["keep_best"] and outcomes:
                best_outcome = max(
                    outcomes,
                    key=lambda item: float(item.get("joint_quality") or 0.0),
                )
                if float(best_outcome.get("joint_quality") or 0.0) > float(
                    final_outcome.get("joint_quality") or 0.0
                ):
                    final_outcome = best_outcome
            kept_round = int(final_outcome.get("round") or 0)
            parse = final_outcome.get("parse") or {
                "ok": False,
                "plan": None,
                "issues": [],
            }
            stage = dict(final_outcome.get("stage") or {})
            stage.setdefault("parse_ok", isinstance(parse.get("plan"), dict))
            stage.setdefault("schema_valid", False)
            state = {
                **base_state,
                "raw_response": final_outcome.get("raw_response"),
                "raw_response_sha256": final_outcome.get("raw_response_sha256"),
                "api": final_outcome.get("api"),
                "parse": parse,
                "stage": stage,
                "parsed_plan_sha256": final_outcome.get("parsed_plan_sha256"),
                "repaired_plan": final_outcome.get("repaired_plan"),
                "repaired_plan_sha256": final_outcome.get("repaired_plan_sha256"),
                "repair_log": final_outcome.get("repair_log"),
                "post_repair_issues": final_outcome.get("post_repair_issues"),
                "episode": final_outcome.get("episode"),
                "geometry": final_outcome.get("geometry"),
                "result_step_path": final_outcome.get("result_step_path"),
                "operation_count": final_outcome.get("operation_count"),
                "feedback": {
                    **feedback_state,
                    "final_round": (
                        feedback_state["rounds"][-1]["round"]
                        if feedback_state["rounds"]
                        else 0
                    ),
                    "kept_round": kept_round,
                    "n_api_calls": len(feedback_state["rounds"]),
                },
                "status": final_outcome.get("status") or status,
                "elapsed_sec": time.perf_counter() - started,
            }
            atomic_write_json(path, state)
            counts[state["status"]] += 1
        except FatalAPIError:
            base_state.update(
                {
                    "status": "fatal_api_error",
                    "elapsed_sec": time.perf_counter() - started,
                }
            )
            atomic_write_json(path, base_state)
            counts["fatal_api_error"] += 1
            raise
        except Exception as exc:
            base_state.update(
                {
                    "status": "task_failed",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc)[:1000],
                    },
                    "elapsed_sec": time.perf_counter() - started,
                }
            )
            atomic_write_json(path, base_state)
            counts["task_failed"] += 1
        finally:
            completed_tasks = task_index + 1
            if completed_tasks % 25 == 0 or completed_tasks == len(tasks):
                print(
                    "ENCODING_PROGRESS "
                    f"{completed_tasks}/{len(tasks)} "
                    f"api_calls={api_calls} "
                    f"counts={dict(sorted(counts.items()))}",
                    flush=True,
                )

    summary = {
        "schema_version": RUN_SCHEMA,
        "samples": len(rows),
        "conditions": len(selected),
        "expected_tasks": len(rows) * len(selected),
        "counts": dict(counts),
        "api_calls": api_calls,
        "dry_run": dry_run,
        "elapsed_sec": time.perf_counter() - started_run,
        "code_fingerprint": code_fingerprint,
        "scoring_fingerprint": scoring_fingerprint,
        "feedback": {
            "arm": feedback.get("arm"),
            "plan_prompt_version": plan_prompt_version,
            "enabled": bool(feedback["enabled"]),
            "max_rounds": int(feedback["max_rounds"]),
            "sources": list(feedback["sources"]),
        },
    }
    atomic_write_json(output_dir / "run_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行20×63编码全因子筛选实验")
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "encoding_screen_n20.yaml"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conditions", nargs="*")
    parser.add_argument(
        "--arm",
        choices=["A0", "A1", "B1", "B2", "C", "custom"],
        help="覆盖 config 中 feedback.arm（A0/A1/B1/B2/C 或 custom）",
    )
    parser.add_argument(
        "--output-dir",
        help="覆盖 config 的 paths.output_dir（默认按 arms.<arm>.output_dir 选择）",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.arm:
        config.setdefault("feedback", {})["arm"] = args.arm
        arm_block = (config.get("arms") or {}).get(args.arm) or {}
        if args.arm == "A0" and not args.output_dir:
            raise SystemExit(
                "A0 不重跑：直接复用 arms.A0.reuse_state_dir 的既有 state，"
                "分析脚本会自动读取。如需强制重跑 A0 请显式指定 --output-dir。"
            )
        if not args.output_dir and arm_block.get("output_dir"):
            config.setdefault("paths", {})["output_dir"] = arm_block["output_dir"]
    if args.output_dir:
        config.setdefault("paths", {})["output_dir"] = args.output_dir
    if args.conditions is None:
        subset = (config.get("conditions") or {}).get("subset")
        conditions = _resolve_conditions(subset) if subset else None
    else:
        conditions = _resolve_conditions(args.conditions)
    summary = run_encoding_screen(
        config,
        dry_run=args.dry_run,
        limit=args.limit,
        conditions=conditions,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0

