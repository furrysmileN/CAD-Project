"""原生点云几何证据（P_geom）20×9 筛选运行器。

Harness 固定为 C 臂：默认 v3 prompt（Plan v2）+ repair R4 + 最多 2 轮 schema/execution 反馈。
yaml 写 arms.C.plan_version: v5 时切到 Plan v3 + 本地姿态/配方；反馈轮用同一 schema。
点云侧：PointEvidence + 可选 FSM 工具循环。断点续跑 / --force 归档与 confirm_runner 同构。
"""
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
from .measurement_binder import BindingError, bind_evidence_references
from .pc_conditions import SCREEN_CONDITION_IDS, PCConditionSpec, validate_conditions
from .pc_fsm import (
    ToolLoopConfig,
    ToolLoopState,
    apply_query,
    classify_model_output,
    register_candidate,
    _timed_execute,
)
from .pc_prompting import audit_payload, build_pc_messages
from .pc_tool_fsm import append_query_turn, parse_query_or_submit, run_forced_query
from .v5_shuffle import load_shuffle_mapping
from .pointcloud.service import PointCloudService
from .prompting import coerce_parse_version, parse_plan_response, validate_plan
from .repair_v21 import REPAIR_VERSION, repair_plan_v21


STATE_SCHEMA = "rq2.pc_geom.task_state.v1"
RUN_SCHEMA = "rq2.pc_geom.run_summary.v1"
TERMINAL_STATUSES = {"completed", "parse_failed", "episode_failed"}


def _resolve_conditions(raw: Iterable[str] | None, defaults: list[str]) -> tuple[PCConditionSpec, ...]:
    if not raw:
        return validate_conditions(defaults)
    values: list[str] = []
    for item in raw:
        values.extend(part.strip() for part in item.split(",") if part.strip())
    return validate_conditions(values)


def _task_order(
    rows: list[dict[str, Any]],
    conditions: tuple[PCConditionSpec, ...],
    seed: int,
    *,
    repeat_ids: tuple[int, ...] = (0,),
    control_ids: frozenset[str] = frozenset(),
    control_repeat: int = 1,
) -> list[tuple[dict[str, Any], PCConditionSpec, int]]:
    tasks: list[tuple[dict[str, Any], PCConditionSpec, int]] = []
    for row in rows:
        for spec in conditions:
            allowed = (control_repeat,) if spec.condition_id in control_ids else repeat_ids
            for repeat_id in allowed:
                tasks.append((row, spec, int(repeat_id)))
    tasks.sort(
        key=lambda task: hashlib.sha256(
            f"{seed}:{task[0]['sample_id']}:{task[1].condition_id}:{task[2]}".encode("utf-8")
        ).hexdigest()
    )
    return tasks


def _code_fingerprint() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    names = (
        "pc_runner.py",
        "pc_conditions.py",
        "pc_prompting.py",
        "pc_fsm.py",
        "pc_tool_fsm.py",
        "v5_shuffle.py",
        "prompting.py",
        "feedback.py",
        "repair_v21.py",
        "geometry.py",
        "backend.py",
    )
    files = {name: sha256_file(root / name) for name in names if (root / name).is_file()}
    pc_dir = root / "pointcloud"
    if pc_dir.is_dir():
        for path in sorted(pc_dir.glob("*.py")):
            files[f"pointcloud/{path.name}"] = sha256_file(path)
    return {"files": files, "sha256": sha256_json(files)}


def _state_path(output_dir: Path, sample_id: str, condition_id: str, repeat_id: int | None = None) -> Path:
    if repeat_id in (None, 0):
        return output_dir / "state" / safe_id(sample_id) / f"{safe_id(condition_id)}.json"
    return output_dir / "state" / safe_id(sample_id) / safe_id(condition_id) / f"r{int(repeat_id):02d}.json"


def _trace_path(output_dir: Path, sample_id: str, condition_id: str, repeat_id: int | None = None) -> Path:
    if repeat_id in (None, 0):
        return output_dir / "tool_traces" / safe_id(sample_id) / f"{safe_id(condition_id)}.json"
    return output_dir / "tool_traces" / safe_id(sample_id) / safe_id(condition_id) / f"r{int(repeat_id):02d}.json"


def _should_skip(previous: dict[str, Any], task_fingerprint: str, dry_run: bool) -> bool:
    if previous.get("task_fingerprint") != task_fingerprint:
        return False
    if dry_run:
        return previous.get("status") == "dry_run"
    return previous.get("status") in TERMINAL_STATUSES and previous.get("status") != "dry_run"


def _archive_previous(output_dir: Path, previous: dict[str, Any], sample_id: str, condition_id: str) -> None:
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


def _npy_path(row: dict[str, Any], config: dict[str, Any]) -> Path:
    raw = (row.get("point_cloud") or {}).get("path")
    if raw:
        path = Path(raw)
        if path.is_file():
            return path
    root = project_path(config.get("paths", {}).get("pointcloud_root") or "processed/point_clouds/benchcad")
    density = int((config.get("pointcloud") or {}).get("density") or 2048)
    return root / str(density) / f"{row['sample_id']}.npy"


def _service_from_config(config: dict[str, Any]) -> PointCloudService:
    block = dict(config.get("pointcloud") or {})
    return PointCloudService(
        normals_k=int(block.get("normals_k", 16)),
        plane_tolerance=float(block.get("plane_tolerance", 0.012)),
        plane_max_planes=int(block.get("plane_max_planes", 2)),
        ransac_seed=int(block.get("seed", 42)),
        section_thickness=block.get("section_thickness"),
        symmetry_tolerance=float(block.get("symmetry_tolerance", 0.015)),
        symmetry_sample=int(block.get("symmetry_sample", 512)),
        n_points=int(block.get("density", 2048)),
        seed=int(block.get("seed", 42)),
    )


def _wrap_kernel_feedback(text: str, *, candidate_id: str | None, tools_enabled: bool) -> str:
    prefix = "[KERNEL_FEEDBACK]\n"
    if tools_enabled and candidate_id:
        prefix += (
            f"A candidate STEP is registered as {candidate_id}. "
            "You may issue one query_request (compare_cad_to_cloud / localize_geometric_error) "
            "or output a corrected Plan JSON.\n\n"
        )
    return prefix + text


def run_pc_geom(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    limit: int | None = None,
    conditions: tuple[PCConditionSpec, ...] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = project_path(config["paths"]["output_root"])
    evidence_dir = project_path(config["paths"].get("evidence_dir") or (Path(config["paths"]["output_root"]) / "evidence"))
    manifest_path = project_path(config["paths"]["manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 manifest {manifest_path}")
    all_rows = list(read_jsonl(manifest_path))
    sample_filter = [str(item) for item in (config.get("sample_ids") or [])]
    if sample_filter:
        wanted = set(sample_filter)
        all_rows = [row for row in all_rows if row.get("sample_id") in wanted]
        missing = wanted - {row.get("sample_id") for row in all_rows}
        if missing:
            raise RuntimeError(f"manifest 缺少 sample_ids: {sorted(missing)}")
        if len(all_rows) != len(sample_filter):
            raise RuntimeError(f"sample_ids 过滤后行数 {len(all_rows)} != {len(sample_filter)}")
    else:
        expected_n = int(config["n"])
        if len(all_rows) != expected_n:
            raise RuntimeError(f"manifest 应有 {expected_n} 行，实际 {len(all_rows)}")
    rows = all_rows[:limit] if limit is not None else all_rows

    arm_block = dict((config.get("arms") or {}).get("C") or {})
    plan_version = str(arm_block.get("plan_version") or "v3")
    repair_rules = tuple((arm_block.get("repair") or {}).get("rules") or ("number", "rotate_revolve", "unit_axis", "polygon"))
    feedback = resolve_feedback_config({**config, "feedback": arm_block.get("feedback") or {"arm": "C"}})
    feedback["arm"] = "C"
    feedback["plan_prompt_version"] = plan_version
    parse_version = coerce_parse_version(plan_version)

    api_settings = APISettings.from_config(config["api"])
    if not dry_run:
        api_settings.resolved(require_key=True)
    seed = int(config["seed"])
    image_max_edge = int(config.get("modalities", {}).get("image_max_edge", 1024))
    tools_block = dict(config.get("tools") or {})
    tool_config = ToolLoopConfig(
        max_pre_queries=int(tools_block.get("max_pre_queries", 3)),
        max_post_queries=int(tools_block.get("max_post_queries", 1)),
        timeout_sec=float(tools_block.get("timeout_sec", 8)),
    )
    service = _service_from_config(config)
    evidence_config_hash = sha256_json(service.config())
    code_fingerprint = _code_fingerprint()
    scoring_fingerprint = sha256_json(config["scoring"])
    feedback_fingerprint = sha256_json(
        {
            "plan_prompt_version": plan_version,
            "feedback": {
                key: feedback.get(key)
                for key in ("arm", "enabled", "max_rounds", "sources", "round2_temperature", "keep_best")
            },
            "tools": {
                "max_pre_queries": tool_config.max_pre_queries,
                "max_post_queries": tool_config.max_post_queries,
            },
            "evidence": evidence_config_hash,
        }
    )
    selected = conditions or validate_conditions(list(config.get("conditions") or SCREEN_CONDITION_IDS))
    repeats = int(config.get("repeats") or 1)
    repeat_ids = tuple(int(item) for item in (config.get("repeat_ids") or list(range(1, repeats + 1))))
    if repeats <= 1 and not config.get("repeat_ids"):
        repeat_ids = (0,)
    control_ids = frozenset(str(item) for item in (config.get("control_conditions") or []))
    control_repeat = int(config.get("control_repeat") or (repeat_ids[0] if repeat_ids else 1))
    shuffle_path = config.get("paths", {}).get("shuffle_mapping")
    shuffle_map = load_shuffle_mapping(project_path(shuffle_path)) if shuffle_path else {}
    shuffle_hash = sha256_json(shuffle_map) if shuffle_map else ""
    tasks = _task_order(
        rows,
        selected,
        seed,
        repeat_ids=repeat_ids,
        control_ids=control_ids,
        control_repeat=control_repeat,
    )
    atomic_write_json(
        output_dir / "task_order.json",
        {
            "schema_version": "rq2.pc_geom.task_order.v2",
            "seed": seed,
            "expected_tasks": len(tasks),
            "arm": "C",
            "plan_version": plan_version,
            "repeat_ids": list(repeat_ids),
            "control_conditions": sorted(control_ids),
            "shuffle_mapping_sha256": shuffle_hash,
            "conditions": [spec.condition_id for spec in selected],
            "tasks": [
                {
                    "order": index,
                    "sample_id": row["sample_id"],
                    "condition_id": spec.condition_id,
                    "repeat_id": repeat_id,
                }
                for index, (row, spec, repeat_id) in enumerate(tasks)
            ],
        },
    )

    counts: Counter[str] = Counter()
    api_calls = 0
    dry_run_issues: list[dict[str, Any]] = []
    started_run = time.perf_counter()
    for task_index, (row, spec, repeat_id) in enumerate(tasks):
        condition_id = spec.condition_id
        path = _state_path(output_dir, row["sample_id"], condition_id, repeat_id)
        started = time.perf_counter()
        npy_path = _npy_path(row, config)
        evidence = None
        evidence_owner = row["sample_id"]
        if spec.point_geom:
            if spec.shuffle:
                evidence_owner = shuffle_map.get(row["sample_id"])
                if not evidence_owner:
                    raise RuntimeError(f"条件 {condition_id} 缺少 {row['sample_id']} 的 shuffle 映射")
                owner_row = next(item for item in all_rows if item["sample_id"] == evidence_owner)
                npy_path = _npy_path(owner_row, config)
            if not npy_path.is_file():
                raise FileNotFoundError(f"缺少点云 {npy_path}")
            evidence = service.prepare_evidence(npy_path, evidence_dir, evidence_owner)
        messages, prompt_audit = build_pc_messages(
            row,
            spec,
            evidence=evidence,
            image_max_edge=image_max_edge,
            plan_prompt_version=plan_version,
            max_pre_queries=tool_config.max_pre_queries,
            max_post_queries=tool_config.max_post_queries,
        )
        prompt_hash = sha256_json(messages)
        input_hash = sha256_json(
            {
                "sample_id": row["sample_id"],
                "condition": condition_id,
                "repeat_id": repeat_id,
                "modalities": prompt_audit.get("modality_hashes"),
                "evidence": None if evidence is None else evidence.get("content_hash"),
                "evidence_profile": spec.resolved_profile,
                "evidence_owner": evidence_owner if spec.point_geom else None,
                "shuffle": bool(spec.shuffle),
                "shuffle_mapping": shuffle_hash,
            }
        )
        task_fingerprint = sha256_json(
            {
                "prompt": prompt_hash,
                "input": input_hash,
                "code": code_fingerprint["sha256"],
                "scoring": scoring_fingerprint,
                "repair_version": REPAIR_VERSION if repair_rules else "none",
                "repair_rules": list(repair_rules),
                "feedback": feedback_fingerprint,
                "evidence_config": evidence_config_hash,
                "repeat_id": repeat_id,
            }
        )
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if not force and _should_skip(previous, task_fingerprint, dry_run):
                counts["skipped"] += 1
                continue
            if not dry_run:
                _archive_previous(output_dir, previous, row["sample_id"], condition_id)

        payload_audit = audit_payload(
            messages,
            spec,
            sample_id=str(row["sample_id"]),
            family=str(row.get("family") or ""),
            gt_hash=((row.get("gt_code") or {}).get("sha256")),
        )
        base_state: dict[str, Any] = {
            "schema_version": STATE_SCHEMA,
            "task_index": task_index,
            "sample_id": row["sample_id"],
            "condition_id": condition_id,
            "condition": condition_id,
            "repeat_id": repeat_id,
            "status": "running",
            "input_sha256": input_hash,
            "prompt_sha256": prompt_hash,
            "prompt_audit": prompt_audit,
            "payload_audit": payload_audit,
            "task_fingerprint": task_fingerprint,
            "code_fingerprint": code_fingerprint,
            "scoring_fingerprint": scoring_fingerprint,
            "plan_version": plan_version,
            "repair": {"version": REPAIR_VERSION if repair_rules else "none", "rules": list(repair_rules)},
            "pointcloud_evidence": None
            if evidence is None
            else {
                "schema": evidence.get("schema"),
                "cloud_id": evidence.get("cloud_id"),
                "content_hash": evidence.get("content_hash"),
                "config_hash": evidence_config_hash,
            },
            "tool_traces": [],
        }
        atomic_write_json(path, base_state)
        if dry_run:
            if not payload_audit["ok"]:
                dry_run_issues.append(
                    {"sample_id": row["sample_id"], "condition_id": condition_id, "issues": payload_audit["issues"]}
                )
            base_state.update({"status": "dry_run", "elapsed_sec": time.perf_counter() - started})
            atomic_write_json(path, base_state)
            counts["dry_run"] += 1
            continue

        session = service.session(npy_path, evidence) if spec.point_geom else None
        tool_state = ToolLoopState()
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
            task_api_calls = 0
            c_round = 0
            max_c = int(feedback["max_rounds"])
            max_iters = tool_config.max_pre_queries + tool_config.max_post_queries + max_c + 4
            if spec.tool_protocol == "forced_query" and session is not None and evidence is not None:
                forced_result = run_forced_query(
                    session,
                    cloud_id=str(evidence.get("cloud_id") or ""),
                    config=tool_config,
                    state=tool_state,
                )
                current_messages = append_query_turn(
                    current_messages,
                    request={"tool": "query_cross_section", "arguments": {"origin": [0, 0, 0], "normal": [0, 0, 1]}},
                    result=forced_result,
                )
            for _iter in range(max_iters):
                round_temperature = (
                    api_settings.temperature if c_round == 0 else float(feedback["round2_temperature"])
                )
                round_settings = (
                    api_settings
                    if round_temperature == api_settings.temperature
                    else replace(api_settings, temperature=round_temperature)
                )
                api_result = chat_completion(current_messages, round_settings)
                n_attempts = int(api_result.get("attempt") or 1)
                api_calls += n_attempts
                task_api_calls += n_attempts
                raw_response = api_result["text"]
                if spec.tool_protocol in {"query_or_submit", "forced_query"}:
                    classified = parse_query_or_submit(raw_response)
                    if classified.get("kind") == "query" and spec.tools and session is not None:
                        if tool_state.pre_queries >= tool_config.max_pre_queries:
                            classified = {"kind": "invalid", "error": "query_budget_exhausted", "raw": raw_response}
                        else:
                            request = dict(classified.get("request") or {})
                            params = dict(request.get("params") or {})
                            if evidence is not None and not params.get("cloud_id"):
                                params["cloud_id"] = evidence.get("cloud_id")
                            request["params"] = params
                            result = _timed_execute(session, request, tool_config.timeout_sec)
                            tool_state.pre_queries += 1
                            tool_state.traces.append(
                                {"tool": request.get("tool"), "params": params, "result": result}
                            )
                            current_messages = append_query_turn(
                                current_messages,
                                request=request,
                                result=result,
                            )
                            continue
                    if classified.get("kind") == "submit_plan" and isinstance(classified.get("plan"), dict):
                        raw_response = json.dumps(classified["plan"], ensure_ascii=False)
                        classified = {"kind": "unknown", "raw": raw_response}
                else:
                    classified = classify_model_output(raw_response)
                if classified.get("kind") == "query_request" and spec.tools and session is not None:
                    current_messages, _trace = apply_query(
                        base_messages=messages,
                        raw_response=raw_response,
                        request=classified.get("request") or {},
                        session=session,
                        state=tool_state,
                        config=tool_config,
                    )
                    continue
                if classified.get("kind") == "query_request":
                    # 非工具条件把 query 当非法 JSON plan
                    classified = {"kind": "unknown", "request": classified.get("request"), "raw": raw_response}

                round_record: dict[str, Any] = {"round": c_round, "temperature": round_temperature}
                parsed = parse_plan_response(raw_response, plan_version=parse_version)
                outcome: dict[str, Any] = {
                    "round": c_round,
                    "raw_response": raw_response,
                    "raw_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
                    "api": {key: value for key, value in api_result.items() if key != "text"},
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
                    if SCHEMA_SOURCE in feedback["sources"] and c_round < max_c:
                        current_messages = feedback_turn(
                            messages,
                            raw_response,
                            _wrap_kernel_feedback(
                                build_schema_feedback(issues, {}, c_round),
                                candidate_id="cand_0" if tool_state.has_candidate else None,
                                tools_enabled=spec.tools,
                            ),
                        )
                        c_round += 1
                        continue
                    break

                if parse_version == "v6":
                    path_graph = (
                        ((prompt_audit.get("guidance") or {}).get("decisions") or {}).get(
                            "path_graph"
                        )
                    )
                    semantic_plan = original_plan
                    try:
                        original_plan, binding_log = bind_evidence_references(
                            semantic_plan,
                            path_graph if isinstance(path_graph, dict) else None,
                        )
                    except BindingError as exc:
                        issues = [exc.issue]
                        round_record["failure"] = {
                            "kind": SCHEMA_SOURCE,
                            "issues": issues,
                        }
                        outcome["status"] = "binding_failed"
                        outcome["binding_issues"] = issues
                        outcomes.append(outcome)
                        feedback_state["rounds"].append(round_record)
                        status = "parse_failed"
                        if SCHEMA_SOURCE in feedback["sources"] and c_round < max_c:
                            current_messages = feedback_turn(
                                messages,
                                raw_response,
                                _wrap_kernel_feedback(
                                    build_schema_feedback(issues, semantic_plan, c_round),
                                    candidate_id="cand_0" if tool_state.has_candidate else None,
                                    tools_enabled=spec.tools,
                                ),
                            )
                            c_round += 1
                            continue
                        break
                    outcome["semantic_plan_sha256"] = sha256_json(semantic_plan)
                    outcome["binding_log"] = binding_log

                repaired_plan, repair_log = repair_plan_v21(original_plan, rules=repair_rules)
                repaired_issues = validate_plan(repaired_plan, plan_version=parse_version)
                if repaired_plan.get("sample_id") != prompt_audit["prompt_sample_id"]:
                    repaired_issues = list(repaired_issues) + [{"path": "$.sample_id", "code": "sample_id_mismatch"}]
                outcome["parsed_plan_sha256"] = sha256_json(original_plan)
                outcome["repaired_plan"] = repaired_plan
                outcome["repaired_plan_sha256"] = sha256_json(repaired_plan)
                outcome["repair_log"] = repair_log
                outcome["post_repair_issues"] = repaired_issues
                round_record["api"] = outcome["api"]
                if repaired_issues:
                    round_record["failure"] = {"kind": SCHEMA_SOURCE, "issues": repaired_issues}
                    outcome["status"] = "parse_failed"
                    outcomes.append(outcome)
                    feedback_state["rounds"].append(round_record)
                    status = "parse_failed"
                    if SCHEMA_SOURCE in feedback["sources"] and c_round < max_c:
                        current_messages = feedback_turn(
                            messages,
                            raw_response,
                            _wrap_kernel_feedback(
                                build_schema_feedback(repaired_issues, repaired_plan, c_round),
                                candidate_id="cand_0" if tool_state.has_candidate else None,
                                tools_enabled=spec.tools,
                            ),
                        )
                        c_round += 1
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
                    geometry = {"valid": False, "failure": "backend_no_result_step", "joint_quality": 0.0}
                response_status = str(episode["response"].get("status"))
                execution_success = response_status in {"success", "success_with_warnings"}
                outcome["episode"] = episode
                outcome["geometry"] = geometry
                outcome["result_step_path"] = result_step
                outcome["operation_count"] = len(repaired_plan.get("operations") or [])
                outcome["stage"] = {
                    "parse_ok": True,
                    "schema_valid": True,
                    "episode_status": response_status,
                    "execution_success": execution_success,
                    "geometry_valid": bool(geometry.get("valid", False)),
                }
                if spec.tools and session is not None and result_step:
                    try:
                        register_candidate(session, result_step, candidate_step_id="cand_0")
                        tool_state.has_candidate = True
                    except Exception:
                        pass
                if execution_success:
                    outcome["status"] = "completed"
                    outcome["joint_quality"] = float(geometry.get("joint_quality") or 0.0)
                    outcomes.append(outcome)
                    feedback_state["rounds"].append(round_record)
                    status = "completed"
                    break
                failure = episode["response"].get("failure") or {
                    "code": response_status,
                    "message": str(episode["response"].get("error") or ""),
                }
                if failure_kind_from_code(failure.get("code")) == "format":
                    issues = (episode["response"].get("validation") or {}).get("issues") or []
                    round_record["failure"] = {"kind": SCHEMA_SOURCE, "issues": issues, "failure": failure}
                    outcome["status"] = "episode_failed"
                    outcomes.append(outcome)
                    feedback_state["rounds"].append(round_record)
                    status = "episode_failed"
                    if SCHEMA_SOURCE in feedback["sources"] and c_round < max_c:
                        current_messages = feedback_turn(
                            messages,
                            raw_response,
                            _wrap_kernel_feedback(
                                build_schema_feedback(issues, repaired_plan, c_round),
                                candidate_id="cand_0" if tool_state.has_candidate else None,
                                tools_enabled=spec.tools,
                            ),
                        )
                        c_round += 1
                        continue
                    break
                round_record["failure"] = {"kind": EXECUTION_SOURCE, "failure": failure}
                outcome["status"] = "episode_failed"
                outcomes.append(outcome)
                feedback_state["rounds"].append(round_record)
                status = "episode_failed"
                if EXECUTION_SOURCE in feedback["sources"] and c_round < max_c:
                    current_messages = feedback_turn(
                        messages,
                        raw_response,
                        _wrap_kernel_feedback(
                            build_execution_feedback(failure, repaired_plan, c_round),
                            candidate_id="cand_0" if tool_state.has_candidate else None,
                            tools_enabled=spec.tools,
                        ),
                    )
                    c_round += 1
                    continue
                break

            final_outcome = outcomes[-1] if outcomes else {}
            if feedback["keep_best"] and outcomes:
                best_outcome = max(outcomes, key=lambda item: float(item.get("joint_quality") or 0.0))
                if float(best_outcome.get("joint_quality") or 0.0) > float(final_outcome.get("joint_quality") or 0.0):
                    final_outcome = best_outcome
            kept_round = int(final_outcome.get("round") or 0)
            parse = final_outcome.get("parse") or {"ok": False, "plan": None, "issues": []}
            stage = dict(final_outcome.get("stage") or {})
            stage.setdefault("parse_ok", isinstance(parse.get("plan"), dict))
            stage.setdefault("schema_valid", False)
            traces = list(tool_state.traces)
            atomic_write_json(
                _trace_path(output_dir, row["sample_id"], condition_id, repeat_id),
                {
                    "schema_version": "rq2.pc_geom.tool_trace.v1",
                    "sample_id": row["sample_id"],
                    "condition_id": condition_id,
                    "pre_queries": tool_state.pre_queries,
                    "post_queries": tool_state.post_queries,
                    "traces": traces,
                },
            )
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
                    "final_round": feedback_state["rounds"][-1]["round"] if feedback_state["rounds"] else 0,
                    "kept_round": kept_round,
                    "n_api_calls": task_api_calls,
                },
                "tool_traces": traces,
                "status": final_outcome.get("status") or status,
                "elapsed_sec": time.perf_counter() - started,
            }
            atomic_write_json(path, state)
            counts[state["status"]] += 1
        except FatalAPIError:
            base_state.update(
                {
                    "status": "fatal_api_error",
                    "tool_traces": tool_state.traces,
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
                    "error": {"type": type(exc).__name__, "message": str(exc)[:1000]},
                    "tool_traces": tool_state.traces,
                    "elapsed_sec": time.perf_counter() - started,
                }
            )
            atomic_write_json(path, base_state)
            counts["task_failed"] += 1
        finally:
            completed_tasks = task_index + 1
            if completed_tasks % 10 == 0 or completed_tasks == len(tasks):
                print(
                    "PC_GEOM_PROGRESS "
                    f"{completed_tasks}/{len(tasks)} "
                    f"api_calls={api_calls} "
                    f"counts={dict(sorted(counts.items()))}",
                    flush=True,
                )

    summary = {
        "schema_version": RUN_SCHEMA,
        "arm": "C",
        "plan_version": plan_version,
        "samples": len(rows),
        "conditions": [spec.condition_id for spec in selected],
        "expected_tasks": len(tasks),
        "repeats": list(repeat_ids),
        "counts": dict(counts),
        "api_calls": api_calls,
        "dry_run": dry_run,
        "dry_run_issues": dry_run_issues,
        "elapsed_sec": time.perf_counter() - started_run,
        "code_fingerprint": code_fingerprint,
        "scoring_fingerprint": scoring_fingerprint,
        "evidence_config_hash": evidence_config_hash,
        "feedback": {
            "arm": feedback.get("arm"),
            "plan_prompt_version": plan_version,
            "enabled": bool(feedback["enabled"]),
            "max_rounds": int(feedback["max_rounds"]),
            "sources": list(feedback["sources"]),
        },
    }
    atomic_write_json(output_dir / "run_summary.json", summary)
    if dry_run:
        atomic_write_json(
            output_dir / "analysis" / "phase5" / "dryrun_audit.json",
            {
                "ok": not dry_run_issues,
                "n_tasks": int(counts.get("dry_run", 0) + counts.get("skipped", 0)),
                "issues": dry_run_issues,
            },
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 P_geom 20×9 筛选（C 臂 Harness）")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "pc_geom_screen.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conditions", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    conditions = _resolve_conditions(args.conditions, list(config.get("conditions") or SCREEN_CONDITION_IDS))
    summary = run_pc_geom(
        config,
        dry_run=args.dry_run,
        limit=args.limit,
        conditions=conditions,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("dry_run") and summary.get("dry_run_issues"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
