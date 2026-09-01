"""Harness vs CADrille Cut 2 运行器。"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from .api_client import APISettings, FatalAPIError, chat_completion
from .backend import run_episode
from .common import atomic_write_json, load_config, project_path, read_jsonl, safe_id
from .cq_sandbox import extract_cadquery_source, run_cadquery_sandbox
from .feedback import build_execution_feedback, build_schema_feedback, feedback_turn, resolve_feedback_config
from .geometry import score_step_pair
from .harness_guidance import _canonical_points, build_guidance
from .prompting import RAW_CADQUERY_SYSTEM, build_messages, parse_plan_response, validate_plan
from .repair_v21 import repair_plan_v21


ARMS = ("cadrille_rl", "qwen_raw", "qwen_harness")


def _state_path(root: Path, sample_id: str, arm: str) -> Path:
    return root / "state" / safe_id(sample_id) / f"{arm}.json"


def _score(pred: Path | None, gt: Path, scoring: dict[str, Any]) -> dict[str, Any]:
    if pred is None or not Path(pred).is_file():
        return {"joint_quality": 0.0, "valid": False, "success": False}
    try:
        geometry = score_step_pair(
            pred,
            gt,
            n_points=int(scoring.get("point_samples", 2048)),
            voxel_resolution=int(scoring.get("voxel_resolution", 48)),
            tau=float(scoring.get("failure_aware_tau", 0.25)),
        )
    except Exception as exc:
        return {"joint_quality": 0.0, "valid": False, "success": False, "error": str(exc)[:400]}
    geometry["success"] = True
    return geometry


def _guidance_for_row(row: dict[str, Any]) -> dict[str, Any]:
    path = (row.get("point_cloud") or {}).get("path")
    if not path or not Path(path).is_file():
        return build_guidance(None)
    import numpy as np

    points = np.load(path)
    canonical = _canonical_points(points)
    bbox = None
    if canonical is not None:
        bbox = (canonical.max(axis=0) - canonical.min(axis=0)).tolist()
    return build_guidance(None, bbox_size=bbox, points=points)


def _harness_messages(
    row: dict[str, Any],
    *,
    image_max_edge: int,
    plan_version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    messages, audit = build_messages(row, "I", image_max_edge=image_max_edge, plan_version=plan_version)
    if plan_version == "v5":
        guidance = _guidance_for_row(row)
        block = guidance.get("prompt_block") or ""
        if block:
            content = list(messages[1]["content"])
            content.insert(-1, {"type": "text", "text": block})
            messages = [messages[0], {**messages[1], "content": content}]
        audit["guidance"] = {
            "pose": guidance.get("pose"),
            "generator": ((guidance.get("decisions") or {}).get("generator") or {}).get("id"),
        }
    return messages, audit


def _run_harness_arm(
    row: dict[str, Any],
    *,
    config: dict[str, Any],
    api_settings: APISettings,
    scoring: dict[str, Any],
    image_max_edge: int,
    plan_version: str,
    repair_rules: tuple[str, ...],
    output_dir: Path,
) -> tuple[dict[str, Any], int]:
    messages, _audit = _harness_messages(row, image_max_edge=image_max_edge, plan_version=plan_version)
    feedback = resolve_feedback_config({"feedback": (config.get("harness") or {}).get("feedback") or {"arm": "C"}})
    max_rounds = int(feedback["max_rounds"]) if feedback.get("enabled") else 0
    current = messages
    api_calls = 0
    best: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    for round_index in range(max_rounds + 1):
        settings = api_settings if round_index == 0 else replace(api_settings, temperature=float(feedback["round2_temperature"]))
        response = chat_completion(current, settings)
        api_calls += 1
        raw = str(response.get("text") or "")
        parsed = parse_plan_response(raw, plan_version=plan_version)
        plan = parsed.get("plan") if isinstance(parsed.get("plan"), dict) else None
        issues = list(parsed.get("issues") or [])
        repaired = None
        if isinstance(plan, dict):
            repaired, _log = repair_plan_v21(plan, repair_rules)
            issues = validate_plan(repaired, plan_version=plan_version)
        episode = None
        pred = None
        if repaired and not issues:
            episode = run_episode(repaired, config["backend"], run_root=output_dir / "episodes" / row["sample_id"] / "qwen_harness" / f"r{round_index}")
            pred = episode.get("result_step_path")
        geometry = _score(Path(pred) if pred else None, Path(row["step"]["path"]), scoring)
        record = {
            "round": round_index,
            "parse_ok": isinstance(plan, dict),
            "schema_ok": not issues,
            "issues": issues[:12],
            "episode_status": (episode or {}).get("response", {}).get("status") if episode else None,
            "geometry": geometry,
        }
        last = record
        if best is None or float(geometry.get("joint_quality") or 0.0) > float((best.get("geometry") or {}).get("joint_quality") or 0.0):
            best = record
        exec_ok = record["episode_status"] in {"success", "success_with_warnings"}
        need = (issues or plan is None or not exec_ok) and round_index < max_rounds
        if need:
            if issues or plan is None:
                current = feedback_turn(messages, raw, build_schema_feedback(issues, repaired or {}, round_index))
            else:
                failure = ((episode or {}).get("response") or {}).get("failure") or {"message": "execution_failed"}
                current = feedback_turn(messages, raw, build_execution_feedback(failure, repaired or {}, round_index))
            continue
        break
    kept = best if feedback.get("keep_best") and best is not None else last
    geometry = (kept or {}).get("geometry") or {"joint_quality": 0.0, "success": False}
    if kept and kept.get("episode_status") in {"success", "success_with_warnings"}:
        status = "completed"
    elif kept and not kept.get("parse_ok"):
        status = "parse_failed"
    else:
        status = "episode_failed"
    return (
        {
            "status": status,
            "geometry": geometry,
            "feedback": {"kept_round": (kept or {}).get("round"), "last": last, "best": best},
            "issues": (kept or {}).get("issues"),
            "episode_status": (kept or {}).get("episode_status"),
        },
        api_calls,
    )


def _cadrille_source(pred_dir: Path, sample_id: str) -> str | None:
    if not pred_dir.is_dir():
        return None
    for name in (f"{sample_id}.py", f"{sample_id}.txt"):
        path = pred_dir / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def run_hvc(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    limit: int | None = None,
    force: bool = False,
    arms: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = project_path(config["paths"]["output_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(project_path(config["paths"]["manifest"])))
    if limit is not None:
        rows = rows[:limit]
    api_settings = APISettings.from_config(config["api"])
    if not dry_run:
        api_settings.resolved(require_key=True)
    raw_settings = replace(api_settings, json_mode=False)
    scoring = config["scoring"]
    sandbox_timeout = float((config.get("sandbox") or {}).get("timeout_sec", 30))
    image_max_edge = int((config.get("modalities") or {}).get("image_max_edge", 1024))
    plan_version = str((config.get("harness") or {}).get("plan_version") or "v4")
    repair_rules = tuple(((config.get("harness") or {}).get("repair") or {}).get("rules") or ("number", "rotate_revolve", "unit_axis", "polygon"))
    cadrille_dir = project_path(config["paths"]["cadrille_predictions"])
    counts: Counter[str] = Counter()
    api_calls = 0
    for row in rows:
        for arm in arms or config.get("arms") or ARMS:
            path = _state_path(output_dir, row["sample_id"], arm)
            if path.is_file() and not force:
                previous = json.loads(path.read_text(encoding="utf-8"))
                stale_harness = arm == "qwen_harness" and previous.get("plan_version") != plan_version
                if previous.get("status") not in {"dry_run", "running"} and not stale_harness:
                    counts[previous.get("status") or "cached"] += 1
                    continue
            started = time.perf_counter()
            state: dict[str, Any] = {
                "schema_version": "rq2.hvc.task.v1",
                "sample_id": row["sample_id"],
                "family": row.get("family"),
                "stratum": row.get("stratum"),
                "arm": arm,
                "status": "running",
                "plan_version": plan_version if arm == "qwen_harness" else None,
            }
            if not dry_run:
                print(f"[hvc] start {row['sample_id']} {arm}", flush=True)
            if dry_run:
                if arm == "qwen_harness":
                    _harness_messages(row, image_max_edge=image_max_edge, plan_version=plan_version)
                state["status"] = "dry_run"
                state["elapsed_sec"] = time.perf_counter() - started
                atomic_write_json(path, state)
                counts["dry_run"] += 1
                continue
            try:
                if arm == "cadrille_rl":
                    source = _cadrille_source(cadrille_dir, row["sample_id"])
                    if source is None:
                        state["status"] = "pending_gpu"
                        state["geometry"] = {"joint_quality": 0.0, "success": False}
                    else:
                        step_path = output_dir / "steps" / row["sample_id"] / f"{arm}.step"
                        sandbox = run_cadquery_sandbox(source, step_path, timeout_sec=sandbox_timeout)
                        state["sandbox"] = {key: sandbox[key] for key in ("ok", "issues", "returncode")}
                        state["geometry"] = _score(Path(sandbox["step_path"]) if sandbox["ok"] else None, Path(row["step"]["path"]), scoring)
                        state["status"] = "completed" if sandbox["ok"] else "episode_failed"
                elif arm == "qwen_raw":
                    messages, _audit = build_messages(row, "I", image_max_edge=image_max_edge, plan_version="v3")
                    messages[0]["content"] = RAW_CADQUERY_SYSTEM
                    messages[1]["content"] = [
                        item for item in messages[1]["content"] if item.get("type") == "image_url"
                    ] + [{"type": "text", "text": "Write CadQuery that reconstructs this part. Bind the solid to result."}]
                    response = chat_completion(messages, raw_settings)
                    api_calls += 1
                    source = extract_cadquery_source(str(response.get("text") or ""))
                    step_path = output_dir / "steps" / row["sample_id"] / f"{arm}.step"
                    sandbox = run_cadquery_sandbox(source, step_path, timeout_sec=sandbox_timeout)
                    state["sandbox"] = {key: sandbox[key] for key in ("ok", "issues", "returncode")}
                    state["geometry"] = _score(Path(sandbox["step_path"]) if sandbox["ok"] else None, Path(row["step"]["path"]), scoring)
                    state["status"] = "completed" if sandbox["ok"] else "episode_failed"
                else:
                    harness, n_calls = _run_harness_arm(
                        row,
                        config=config,
                        api_settings=api_settings,
                        scoring=scoring,
                        image_max_edge=image_max_edge,
                        plan_version=plan_version,
                        repair_rules=repair_rules,
                        output_dir=output_dir,
                    )
                    api_calls += n_calls
                    state.update(harness)
            except FatalAPIError:
                raise
            except Exception as exc:
                state["status"] = "task_failed"
                state["error"] = str(exc)[:500]
                state["geometry"] = {"joint_quality": 0.0, "success": False}
            state["elapsed_sec"] = time.perf_counter() - started
            atomic_write_json(path, state)
            counts[state["status"]] += 1
            if not dry_run:
                jq = float((state.get("geometry") or {}).get("joint_quality") or 0.0)
                print(
                    f"[hvc] {row['sample_id']} {arm} {state['status']} jq={jq:.3f} {state['elapsed_sec']:.1f}s",
                    flush=True,
                )
    summary = {
        "schema_version": "rq2.hvc.run.v1",
        "n_rows": len(rows),
        "counts": dict(counts),
        "api_calls": api_calls,
        "dry_run": dry_run,
        "output_dir": str(output_dir),
    }
    atomic_write_json(output_dir / "run_summary.json", summary)
    return summary
