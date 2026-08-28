"""V6 runner: dry-run payloads and live first_attempt / final_delivery endpoints.

Does not modify V5 state. Tools are disabled. Neutral condition IDs only.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backend import run_episode
from .common import EXPERIMENT_DIR, atomic_write_json, load_config, project_path, sha256_file, sha256_json
from .feedback import (
    EXECUTION_SOURCE,
    SCHEMA_SOURCE,
    build_execution_feedback,
    build_schema_feedback,
    feedback_turn,
    resolve_feedback_config,
)
from .geometry import score_step_pair
from .prompting import parse_plan_response, validate_plan
from .repair_v21 import repair_plan_v21
from .v6_conditions import NEUTRAL_IDS, audit_v6_payload, build_v6_messages, parse_condition
from .v6_feature_scorer import empty_feature_scores, score_critical_fact
from .v6_manifest import attach_evidence_payloads, iter_tasks, read_manifest

SCHEMA_VERSION = "rq2.v6.task_state.v1"


def _code_fingerprint() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    names = [
        "v6_conditions.py",
        "v6_runner.py",
        "v6_evidence_builder.py",
        "v6_fact_masks.py",
        "v6_corruptions.py",
        "v6_feature_scorer.py",
        "feedback.py",
        "repair_v21.py",
        "geometry.py",
        "prompting.py",
        "backend.py",
    ]
    files = {name: sha256_file(root / name) for name in names if (root / name).is_file()}
    return {"files": files, "sha256": sha256_json(files)}


def _state_path(state_dir: Path, sample_id: str, condition: str, repeat_id: int) -> Path:
    return state_dir / sample_id / condition / f"r{repeat_id:02d}.json"


def _payload_path(audit_dir: Path, sample_id: str, condition: str, repeat_id: int) -> Path:
    return audit_dir / sample_id / condition / f"r{repeat_id:02d}.json"


def _redact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    redacted = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    new_content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,<redacted>"}})
                else:
                    new_content.append(block)
            redacted.append({**message, "content": new_content})
        else:
            redacted.append(message)
    return redacted


def _score(pred_step: str | None, gt_step: str, scoring: dict[str, Any], plan: dict[str, Any] | None, latent: dict[str, Any]) -> dict[str, Any]:
    if not pred_step:
        geometry = {
            "valid": False,
            "joint_quality": 0.0,
            "common_frame_cd": None,
            "failure": "missing_pred_step",
        }
    else:
        geometry = score_step_pair(
            pred_step,
            gt_step,
            n_points=int(scoring.get("point_samples", 2048)),
            seed=int(scoring.get("seed", 42)),
            voxel_resolution=int(scoring.get("voxel_resolution", 48)),
            tau=float(scoring.get("failure_aware_tau", 0.25)),
        )
    features = score_critical_fact(plan, latent) if plan else empty_feature_scores()
    return {"geometry": geometry, "features": features}


def _resolve_latent_path(row: dict[str, Any], config: dict[str, Any]) -> Path:
    spec = row.get("latent_spec")
    if isinstance(spec, dict) and spec.get("path"):
        path = Path(spec["path"])
        if path.is_file():
            return path
    latent_dir = (config.get("paths") or {}).get("latent_dir")
    if latent_dir and row.get("sample_id"):
        path = project_path(latent_dir) / f"{row['sample_id']}.json"
        if path.is_file():
            return path
    raise KeyError(f"manifest 缺少 latent_spec: {row.get('sample_id')}")


def _fingerprint(config: dict[str, Any], row: dict[str, Any], spec, messages, repeat_id: int, code_fp: dict[str, Any]) -> str:
    evidence_key = spec.evidence_key
    evidence_hash = ""
    if evidence_key:
        blob = (row.get("evidence") or {}).get(evidence_key) or {}
        evidence_hash = str(blob.get("content_hash") or sha256_json(blob))
    images = ((row.get("images") or {}).get("views")) or []
    return sha256_json(
        {
            "latent": row.get("latent_spec_sha256"),
            "step": (row.get("target") or {}).get("step_sha256"),
            "images": [item.get("sha256") for item in images],
            "pointcloud": (row.get("pointcloud") or {}).get("sha256"),
            "evidence": evidence_hash,
            "condition": spec.condition_id,
            "repeat_id": repeat_id,
            "model": (config.get("api") or {}).get("default_model"),
            "prompt": sha256_json(messages),
            "prompt_version": "rq2.v6.prompt.v2",
            "harness": code_fp["sha256"],
            "scoring": sha256_json(config.get("scoring") or {}),
            "feedback": sha256_json(resolve_feedback_config(config.get("arms", {}).get("C", {}))),
        }
    )


def run_v6(config_path: str | Path, *, dry_run: bool = True, limit: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    output_root = project_path(config["paths"]["output_root"])
    split = str(config.get("split") or "pilot")
    repeats = list(config.get("repeat_ids") or [1])
    manifest = read_manifest(project_path(config["paths"]["manifest"]))
    if config.get("eligible_only"):
        manifest = [row for row in manifest if row.get("eligible") is True]
    conditions = list(config.get("conditions") or list(NEUTRAL_IDS))
    state_dir = output_root / split / ("dryrun" if dry_run else "live") / "state"
    audit_dir = output_root / "audits" / "payload_audit"
    runs_dir = output_root / split / ("dryrun" if dry_run else "live") / "runs"
    state_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    code_fp = _code_fingerprint()
    counts: dict[str, int] = {}
    n = 0
    for task in iter_tasks(manifest, conditions, repeats):
        if limit is not None and n >= limit:
            break
        n += 1
        row = attach_evidence_payloads(task["row"])
        spec = parse_condition(task["condition"])
        repeat_id = int(task["repeat_id"])
        messages = build_v6_messages(row, spec)
        payload_audit = audit_v6_payload(messages, spec, sample_id=row.get("sample_id"), family=row.get("family"))
        fp = _fingerprint(config, row, spec, messages, repeat_id, code_fp)
        path = _state_path(state_dir, row["sample_id"], spec.condition_id, repeat_id)
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("task_fingerprint") == fp and previous.get("status") not in {None, "task_failed"}:
                counts[previous.get("status") or "skipped"] = counts.get(previous.get("status") or "skipped", 0) + 1
                continue
        redacted = _redact_messages(messages)
        atomic_write_json(
            _payload_path(audit_dir, row["sample_id"], spec.condition_id, repeat_id),
            {"messages": redacted, "audit": payload_audit, "condition_id": spec.condition_id},
        )
        base = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": row["sample_id"],
            "condition": spec.condition_id,
            "semantic_condition": spec.semantic,
            "repeat_id": repeat_id,
            "task_fingerprint": fp,
            "payload_audit": payload_audit,
            "code_fingerprint": code_fp,
            "dry_run": dry_run,
        }
        if dry_run:
            status = "dry_run_completed" if payload_audit["ok"] else "payload_audit_failed"
            atomic_write_json(path, {**base, "status": status, "first_attempt": None, "final_delivery": None})
            counts[status] = counts.get(status, 0) + 1
            continue
        if not payload_audit["ok"]:
            atomic_write_json(path, {**base, "status": "payload_audit_failed", "first_attempt": None, "final_delivery": None})
            counts["payload_audit_failed"] = counts.get("payload_audit_failed", 0) + 1
            continue
        from .api_client import APISettings, FatalAPIError, chat_completion

        api_settings = APISettings.from_config(config["api"])
        arm_block = (config.get("arms") or {}).get("C") or {"feedback": {"arm": "C"}}
        feedback = resolve_feedback_config(arm_block)
        latent = json.loads(_resolve_latent_path(row, config).read_text(encoding="utf-8"))
        gt_step = row["target"]["step"]
        scoring = config["scoring"]
        first_attempt = None
        final_delivery = None
        current_messages = messages
        started = time.perf_counter()
        repair_rules = list(((arm_block.get("repair") or {}).get("rules")) or ["number", "rotate_revolve", "unit_axis", "polygon"])
        print(f"[v6-live] {row['sample_id']} {spec.condition_id} r{repeat_id:02d}", flush=True)
        try:
            for c_round in range(int(feedback["max_rounds"]) + 1):
                api_result = chat_completion(current_messages, api_settings)
                raw = api_result["text"]
                parsed = parse_plan_response(raw, plan_version="v2")
                # C-arm matches V5: keep a parsed dict even when schema is initially invalid, then R4.
                plan = parsed.get("plan") if isinstance(parsed.get("plan"), dict) else None
                repaired = None
                issues = list(parsed.get("issues") or [])
                if isinstance(plan, dict):
                    repaired, _repair_log = repair_plan_v21(plan, rules=repair_rules)
                    if not isinstance(repaired.get("sample_id"), str):
                        repaired["sample_id"] = "part"
                    issues = validate_plan(repaired, plan_version="v2")
                episode = None
                pred_step = None
                if repaired and not issues:
                    episode = run_episode(repaired, config["backend"], run_root=runs_dir)
                    pred_step = episode.get("result_step_path")
                scored = _score(pred_step, gt_step, scoring, repaired, latent)
                episode_status = ((episode or {}).get("response") or {}).get("status") if episode else None
                record = {
                    "round": c_round,
                    "parse_ok": isinstance(plan, dict),
                    "schema_ok": not issues,
                    "issues": issues[:12],
                    "episode_status": episode_status,
                    "plan_sha256": sha256_json(repaired) if isinstance(repaired, dict) else None,
                    "plan": repaired,
                    "api": {key: value for key, value in api_result.items() if key != "text"},
                    **scored,
                }
                if c_round == 0:
                    first_attempt = record
                final_delivery = record
                exec_ok = episode_status in {"success", "success_with_warnings"}
                need_feedback = (issues or not isinstance(plan, dict) or not exec_ok) and c_round < int(
                    feedback["max_rounds"]
                )
                if need_feedback:
                    if issues or not isinstance(plan, dict):
                        current_messages = feedback_turn(
                            messages, raw, build_schema_feedback(issues, repaired or {}, c_round)
                        )
                    else:
                        failure = ((episode or {}).get("response") or {}).get("failure") or {"message": "execution_failed"}
                        current_messages = feedback_turn(
                            messages, raw, build_execution_feedback(failure, repaired or {}, c_round)
                        )
                    continue
                break
            geom_ok = bool((final_delivery or {}).get("geometry", {}).get("valid"))
            status = "completed" if geom_ok else "execution_failed"
            if final_delivery and not final_delivery.get("parse_ok") and not geom_ok:
                status = "parse_failed"
            elif final_delivery and not final_delivery.get("schema_ok") and not geom_ok:
                status = "schema_failed"
        except FatalAPIError:
            atomic_write_json(
                path,
                {
                    **base,
                    "status": "fatal_api_error",
                    "first_attempt": first_attempt,
                    "final_delivery": final_delivery,
                    "elapsed_sec": time.perf_counter() - started,
                },
            )
            counts["fatal_api_error"] = counts.get("fatal_api_error", 0) + 1
            raise
        except Exception as exc:
            atomic_write_json(
                path,
                {
                    **base,
                    "status": "task_failed",
                    "error": {"type": type(exc).__name__, "message": str(exc)[:1000]},
                    "first_attempt": first_attempt,
                    "final_delivery": final_delivery,
                    "elapsed_sec": time.perf_counter() - started,
                },
            )
            counts["task_failed"] = counts.get("task_failed", 0) + 1
            print(f"[v6-live] FAIL {row['sample_id']} {spec.condition_id}: {type(exc).__name__}: {exc}", flush=True)
            continue
        atomic_write_json(
            path,
            {
                **base,
                "status": status,
                "first_attempt": first_attempt,
                "final_delivery": final_delivery,
                "elapsed_sec": time.perf_counter() - started,
            },
        )
        counts[status] = counts.get(status, 0) + 1
        jq = ((final_delivery or {}).get("geometry") or {}).get("joint_quality")
        feat = ((first_attempt or {}).get("features") or {})
        print(
            f"[v6-live] -> {status} jq={jq} exact={feat.get('exact')} pred={feat.get('pred_value')}",
            flush=True,
        )
        continue
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "n_tasks": n,
        "counts": counts,
        "state_dir": str(state_dir),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_dir = output_root / split / ("dryrun" if dry_run else "live")
    atomic_write_json(summary_dir / f"run_summary_{stamp}.json", summary)
    atomic_write_json(summary_dir / "run_summary_latest.json", summary)
    return summary
