from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from .api_client import APISettings, FatalAPIError, MissingAPIKeyError, chat_completion
from .backend import run_episode
from .common import atomic_write_json, load_config, project_path, read_jsonl, safe_id, sha256_json, state_path
from .conditions import validate_conditions
from .geometry import score_step_pair
from .prepare import prepare
from .prompting import build_messages, parse_plan_response


def _parse_conditions(raw: list[str] | None, defaults: list[str]) -> tuple[str, ...]:
    if not raw:
        return validate_conditions(defaults)
    values = []
    for item in raw:
        values.extend(part.strip() for part in item.split(",") if part.strip())
    return validate_conditions(values)


def _task_order(rows: list[dict[str, Any]], conditions: tuple[str, ...], seed: int) -> list[tuple[dict[str, Any], str]]:
    tasks = [(row, condition) for row in rows for condition in conditions]
    tasks.sort(
        key=lambda task: hashlib.sha256(
            f"{seed}:{task[0]['sample_id']}:{task[1]}".encode("utf-8")
        ).hexdigest()
    )
    return tasks


def _should_skip_state(status: str | None, *, dry_run: bool) -> bool:
    if dry_run:
        return True
    return status in {"completed", "parse_failed", "episode_failed"}


def run(
    config: dict[str, Any],
    *,
    prepare_only: bool,
    dry_run: bool,
    limit: int | None,
    conditions: tuple[str, ...],
    force: bool,
) -> dict[str, Any]:
    output_dir = project_path(config["paths"]["output_dir"])
    manifest_path = output_dir / "manifest.jsonl"
    if force or not manifest_path.is_file():
        prepare_result = prepare(config, force=force)
    else:
        prepare_result = {"status": "cached", "manifest": str(manifest_path)}
    if prepare_only:
        return {"prepare": prepare_result}

    rows = list(read_jsonl(manifest_path))
    if limit is not None:
        rows = rows[:limit]
    api_settings = APISettings.from_config(config["api"])
    plan_version = str(config["prompt"]["plan_version"]).lower()
    if plan_version not in {"v1", "v2"}:
        raise ValueError("prompt.plan_version 仅支持 v1 或 v2")
    if not dry_run:
        api_settings.resolved(require_key=True)

    state_dir = output_dir / "state"
    tasks = _task_order(rows, conditions, int(config["seed"]))
    counts = {"completed": 0, "skipped": 0, "failed": 0, "dry_run": 0}
    task_manifest = [
        {"order": index, "sample_id": row["sample_id"], "condition": condition}
        for index, (row, condition) in enumerate(tasks)
    ]
    atomic_write_json(
        output_dir / "task_order.json",
        {"seed": int(config["seed"]), "conditions": list(conditions), "tasks": task_manifest},
    )

    for task_index, (row, condition) in enumerate(tasks):
        path = state_path(state_dir, row["sample_id"], condition)
        started = time.perf_counter()
        messages, prompt_audit = build_messages(
            row,
            condition,
            image_max_edge=int(config["modalities"]["images"]["max_edge"]),
            plan_version=plan_version,
        )
        prompt_hash = sha256_json(messages)
        if path.is_file() and not force:
            previous = json.loads(path.read_text(encoding="utf-8"))
            same_revision = (
                previous.get("prompt_sha256") == prompt_hash
                and previous.get("input_sha256") == row["input_sha256"]
            )
            if same_revision and _should_skip_state(previous.get("status"), dry_run=dry_run):
                counts["skipped"] += 1
                continue
            if not dry_run:
                archive = (
                    output_dir
                    / "history"
                    / safe_id(row["sample_id"])
                    / safe_id(condition)
                    / (
                        f"{time.time_ns()}__"
                        f"{str(previous.get('prompt_sha256') or 'nohash')[:12]}__"
                        f"{safe_id(str(previous.get('status') or 'unknown'))}.json"
                    )
                )
                atomic_write_json(archive, previous)
        base_state = {
            "schema_version": "rq2.task_state.v1",
            "task_index": task_index,
            "sample_id": row["sample_id"],
            "condition": condition,
            "status": "running",
            "input_sha256": row["input_sha256"],
            "input_hashes": {
                "step": row["step"]["sha256"],
                "images": [item["sha256"] for item in row["images"]],
                "point_cloud": row["point_cloud"]["sha256"],
                "gt_code": row["gt_code"]["sha256"],
            },
            "prompt_sha256": prompt_hash,
            "prompt_audit": prompt_audit,
        }
        atomic_write_json(path, base_state)
        if dry_run:
            base_state.update({"status": "dry_run", "elapsed_sec": time.perf_counter() - started})
            atomic_write_json(path, base_state)
            counts["dry_run"] += 1
            continue
        try:
            api_result = chat_completion(messages, api_settings)
            parsed = parse_plan_response(api_result["text"], plan_version=plan_version)
            state = {
                **base_state,
                "raw_response": api_result["text"],
                "api": {key: value for key, value in api_result.items() if key != "text"},
                "parse": parsed,
            }
            if not parsed["ok"]:
                state.update({"status": "parse_failed", "elapsed_sec": time.perf_counter() - started})
                atomic_write_json(path, state)
                counts["failed"] += 1
                continue
            if parsed["plan"].get("sample_id") != prompt_audit["prompt_sample_id"]:
                state["parse"]["issues"] = [{"path": "$.sample_id", "code": "sample_id_mismatch"}]
                state.update({"status": "parse_failed", "elapsed_sec": time.perf_counter() - started})
                atomic_write_json(path, state)
                counts["failed"] += 1
                continue
            episode = run_episode(parsed["plan"], config["backend"])
            state["episode"] = episode
            result_step = episode["result_step_path"]
            if result_step:
                state["geometry"] = score_step_pair(
                    result_step,
                    row["step"]["path"],
                    n_points=int(config["scoring"]["point_samples"]),
                    seed=int(config["seed"]),
                    voxel_resolution=int(config["scoring"]["voxel_resolution"]),
                    tau=float(config["scoring"]["failure_aware_tau"]),
                )
            else:
                state["geometry"] = {
                    "valid": False,
                    "failure": "backend_no_result_step",
                    "joint_quality": 0.0,
                }
            response_status = str(episode["response"].get("status"))
            state["status"] = "completed" if response_status in {"success", "success_with_warnings"} else "episode_failed"
            state["result_step_path"] = result_step
            state["elapsed_sec"] = time.perf_counter() - started
            atomic_write_json(path, state)
            counts["completed" if state["status"] == "completed" else "failed"] += 1
        except FatalAPIError:
            base_state.update({"status": "fatal_api_error", "elapsed_sec": time.perf_counter() - started})
            atomic_write_json(path, base_state)
            raise
        except Exception as exc:
            base_state.update(
                {
                    "status": "task_failed",
                    "error": {"type": type(exc).__name__, "message": str(exc)[:1000]},
                    "elapsed_sec": time.perf_counter() - started,
                }
            )
            atomic_write_json(path, base_state)
            counts["failed"] += 1
    summary = {"tasks": len(tasks), "counts": counts, "conditions": list(conditions), "dry_run": dry_run}
    atomic_write_json(output_dir / "run_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ2 多模态 HarnessCAD 实验运行器")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conditions", nargs="*")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    conditions = _parse_conditions(args.conditions, list(config["conditions"]))
    try:
        result = run(
            config,
            prepare_only=args.prepare_only,
            dry_run=args.dry_run,
            limit=args.limit,
            conditions=conditions,
            force=args.force,
        )
    except (MissingAPIKeyError, FatalAPIError) as exc:
        print(f"停止：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
