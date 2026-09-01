"""V6b C2B / TB diagnostic runner. Writes to split=diag_c2b|diag_tb, never probe/live."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import atomic_write_json, load_config, project_path, sha256_file, sha256_json
from .feedback import build_execution_feedback, build_schema_feedback, feedback_turn, resolve_feedback_config
from .prompting import parse_plan_response, validate_plan
from .repair_v21 import repair_plan_v21
from .v6_manifest import attach_evidence_payloads, iter_tasks, read_manifest
from .v6_runner import (
    SCHEMA_VERSION,
    _fingerprint,
    _payload_path,
    _redact_messages,
    _resolve_latent_path,
    _score,
    _state_path,
)
from .v6b_diag_conditions import audit_diag_payload, build_diag_messages, fact_sentence, parse_diag_condition
from .backend import run_episode


def _diag_code_fingerprint() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    names = [
        "v6b_diag_conditions.py",
        "v6b_diag_runner.py",
        "v6_feature_scorer.py",
        "feedback.py",
        "repair_v21.py",
        "geometry.py",
        "prompting.py",
        "backend.py",
    ]
    files = {name: sha256_file(root / name) for name in names if (root / name).is_file()}
    return {"files": files, "sha256": sha256_json(files)}


def _with_text_fact(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if row.get("text_fact"):
        return row
    mate_id = row.get("mate_sample_id")
    sample_id = str(row.get("sample_id") or "")
    if not mate_id and sample_id.endswith("a"):
        mate_id = sample_id[:-1] + "b"
    kind = str(row.get("kind") or "")
    latent_dir = (config.get("paths") or {}).get("latent_dir")
    if not mate_id or not latent_dir:
        raise KeyError("TB 需要 mate_sample_id 与 latent_dir")
    spec_b = json.loads((project_path(latent_dir) / f"{mate_id}.json").read_text(encoding="utf-8"))
    out = dict(row)
    out["text_fact"] = fact_sentence(kind, spec_b)
    return out


def run_v6b_diag(config_path: str | Path, *, dry_run: bool = True, limit: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    output_root = project_path(config["paths"]["output_root"])
    split = str(config.get("split") or "diag_c2b")
    if split in {"probe", "pilot", "confirm"} or split.startswith("pilot"):
        raise ValueError(f"诊断不得写入冻结 split: {split}")
    repeats = list(config.get("repeat_ids") or [1, 2])
    manifest = read_manifest(project_path(config["paths"]["manifest"]))
    if config.get("eligible_only"):
        manifest = [row for row in manifest if row.get("eligible") is True]
    conditions = list(config.get("conditions") or [])
    if not set(conditions) <= {"C2B", "TB"}:
        raise ValueError(f"诊断条件只允许 C2B/TB: {conditions}")
    state_dir = output_root / split / ("dryrun" if dry_run else "live") / "state"
    audit_dir = output_root / split / "audits" / "payload_audit"
    runs_dir = output_root / split / ("dryrun" if dry_run else "live") / "runs"
    state_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    code_fp = _diag_code_fingerprint()
    counts: dict[str, int] = {}
    n = 0
    for task in iter_tasks(manifest, conditions, repeats):
        if limit is not None and n >= limit:
            break
        n += 1
        row = attach_evidence_payloads(task["row"])
        spec = parse_diag_condition(task["condition"])
        if spec.text_fact:
            row = _with_text_fact(row, config)
        repeat_id = int(task["repeat_id"])
        messages = build_diag_messages(row, spec)
        payload_audit = audit_diag_payload(messages, spec, sample_id=row.get("sample_id"), family=row.get("family"))
        fp = _fingerprint(config, row, spec, messages, repeat_id, code_fp)
        path = _state_path(state_dir, row["sample_id"], spec.condition_id, repeat_id)
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("task_fingerprint") == fp and previous.get("status") not in {None, "task_failed"}:
                counts[previous.get("status") or "skipped"] = counts.get(previous.get("status") or "skipped", 0) + 1
                continue
        atomic_write_json(
            _payload_path(audit_dir, row["sample_id"], spec.condition_id, repeat_id),
            {"messages": _redact_messages(messages), "audit": payload_audit, "condition_id": spec.condition_id},
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
            "diagnostic": True,
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
        print(f"[v6b-diag] {row['sample_id']} {spec.condition_id} r{repeat_id:02d}", flush=True)
        try:
            for c_round in range(int(feedback["max_rounds"]) + 1):
                api_result = chat_completion(current_messages, api_settings)
                raw = api_result["text"]
                parsed = parse_plan_response(raw, plan_version="v2")
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
                need_feedback = (issues or not isinstance(plan, dict) or not exec_ok) and c_round < int(feedback["max_rounds"])
                if need_feedback:
                    if issues or not isinstance(plan, dict):
                        current_messages = feedback_turn(messages, raw, build_schema_feedback(issues, repaired or {}, c_round))
                    else:
                        failure = ((episode or {}).get("response") or {}).get("failure") or {"message": "execution_failed"}
                        current_messages = feedback_turn(messages, raw, build_execution_feedback(failure, repaired or {}, c_round))
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
            print(f"[v6b-diag] FAIL {row['sample_id']} {spec.condition_id}: {type(exc).__name__}: {exc}", flush=True)
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
        feat = ((first_attempt or {}).get("features") or {})
        print(f"[v6b-diag] -> {status} pred={feat.get('pred_value')}", flush=True)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "n_tasks": n,
        "counts": counts,
        "state_dir": str(state_dir),
        "diagnostic": True,
        "split": split,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_dir = output_root / split / ("dryrun" if dry_run else "live")
    atomic_write_json(summary_dir / f"run_summary_{stamp}.json", summary)
    atomic_write_json(summary_dir / "run_summary_latest.json", summary)
    return summary
