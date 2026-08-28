"""V6b evidence readback diagnostic. Independent API calls; not the CAD probe.

Feeds the same POINT_* serialization as C2–C5. Asks the model to copy
hole_depth / through_or_blind / pocket_depth / hidden_feature_present /
evidence_source / confidence. Informal; does not write CAD Plans.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .api_client import APISettings, FatalAPIError, chat_completion
from .common import atomic_write_json, load_config, project_path, sha256_json
from .prompting import _extract_json_candidate
from .v6_conditions import FORBIDDEN_MARKERS, point_observation_blocks
from .v6_feature_scorer import DEFAULT_TOLERANCE
from .v6_manifest import attach_evidence_payloads, read_manifest

PROMPT_VERSION = "rq2.v6b.readback.v1"
EVIDENCE_KEYS = ("p_full", "p_repeat", "p_counterfactual")
KIND_FIELD = {
    "blind_depth": "hole_depth",
    "through_vs_blind": "through_or_blind",
    "pocket_depth": "pocket_depth",
    "hidden_presence": "hidden_feature_present",
}
READBACK_SCHEMA = {
    "hole_depth": None,
    "through_or_blind": None,
    "pocket_depth": None,
    "hidden_feature_present": None,
    "evidence_source": None,
    "confidence": None,
}
SYSTEM_PROMPT = (
    "You extract geometric facts from point-cloud-derived structured evidence. "
    "Return exactly one JSON object and no commentary. "
    "Do not generate a CAD plan, Python, or extra keys. "
    "Copy values that are explicitly stated. If a field is not stated, use null. "
    "Do not invent measurements."
)
READBACK_INSTRUCTIONS = (
    "[READBACK_CONSTRAINTS]\n"
    "Return exactly one JSON object with these keys:\n"
    + json.dumps(READBACK_SCHEMA, ensure_ascii=False, indent=2)
    + "\nRules:\n"
    "- hole_depth: depth of a non-through hole. Do not use the thickness of a through hole.\n"
    "- through_or_blind: \"through\" or \"blind\" if stated; otherwise null.\n"
    "- pocket_depth: depth of a top-face pocket or depression if stated; otherwise null.\n"
    "- hidden_feature_present: true/false if a back-face or hidden hole presence is stated; otherwise null.\n"
    "- evidence_source: source string of the cad_fact you used, or null.\n"
    "- confidence: confidence of that cad_fact if stated, else null.\n"
    "- If a cad_fact has role primary_critical, use that row for the matching field.\n"
    "- If the corresponding measurement is missing, the field must be null.\n"
    "Output JSON only."
)


def _primary_fact(evidence: dict[str, Any], critical: dict[str, Any] | None = None) -> dict[str, Any] | None:
    facts = [item for item in (evidence.get("cad_facts") or []) if isinstance(item, dict)]
    for item in facts:
        if item.get("role") == "primary_critical":
            return item
    fact_id = str((critical or {}).get("fact_id") or "")
    if fact_id:
        for item in facts:
            if item.get("fact_id") == fact_id:
                return item
    return None


def expected_from_evidence(
    evidence: dict[str, Any],
    *,
    kind: str,
    critical: dict[str, Any] | None = None,
) -> dict[str, Any]:
    field = KIND_FIELD[kind]
    primary = _primary_fact(evidence, critical)
    if primary is None or primary.get("value") is None:
        return {
            "present": False,
            "field": field,
            "value": None,
            "source": None,
            "confidence": None,
            "category": (critical or {}).get("category"),
        }
    return {
        "present": True,
        "field": field,
        "value": primary.get("value"),
        "source": primary.get("source"),
        "confidence": primary.get("confidence"),
        "category": primary.get("category") or (critical or {}).get("category"),
    }


def _norm_through(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"through", "thru", "through_hole"}:
        return "through"
    if text in {"blind", "blind_hole"}:
        return "blind"
    return None


def _norm_hidden(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "present", "1"}:
        return True
    if text in {"false", "no", "absent", "0"}:
        return False
    return None


def _norm_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def predicted_value(parsed: dict[str, Any] | None, kind: str) -> Any:
    if not isinstance(parsed, dict):
        return None
    field = KIND_FIELD[kind]
    raw = parsed.get(field)
    if raw is None and kind in {"blind_depth", "pocket_depth"}:
        other = "pocket_depth" if field == "hole_depth" else "hole_depth"
        raw = parsed.get(other)
    if kind == "through_vs_blind":
        return _norm_through(raw)
    if kind == "hidden_presence":
        return _norm_hidden(raw)
    return _norm_number(raw)


def values_match(pred: Any, expected: Any, category: str | None) -> bool:
    if pred is None or expected is None:
        return False
    if category == "through_vs_blind":
        return _norm_through(pred) == _norm_through(expected)
    if category == "hidden_presence":
        return _norm_hidden(pred) is not None and _norm_hidden(pred) == _norm_hidden(expected)
    try:
        tol = float(DEFAULT_TOLERANCE.get(category or "depth", 0.04))
        return abs(float(pred) - float(expected)) <= tol
    except (TypeError, ValueError):
        return False


def score_readback(
    parsed: dict[str, Any] | None,
    expected: dict[str, Any],
    *,
    kind: str,
    foil_value: Any = None,
) -> dict[str, Any]:
    pred = predicted_value(parsed, kind)
    category = str(expected.get("category") or "")
    if not expected.get("present"):
        reports_missing = pred is None
        leaked_foil = values_match(pred, foil_value, category) if foil_value is not None else False
        ok = reports_missing and not leaked_foil
        return {
            "ok": ok,
            "match": ok,
            "follow_counterfactual": False,
            "predicted": pred,
            "expected": None,
            "reports_missing": reports_missing,
            "leaked_foil": leaked_foil,
        }
    match = values_match(pred, expected.get("value"), category)
    follow = match and not values_match(pred, foil_value, category)
    return {
        "ok": match,
        "match": match,
        "follow_counterfactual": follow,
        "predicted": pred,
        "expected": expected.get("value"),
        "reports_missing": pred is None,
        "leaked_foil": values_match(pred, foil_value, category) if foil_value is not None else False,
    }


def parse_readback_response(text: str) -> dict[str, Any]:
    candidate = _extract_json_candidate(text or "")
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return {"ok": False, "parsed": None, "issues": [f"invalid_json: {exc}"]}
    if not isinstance(parsed, dict):
        return {"ok": False, "parsed": None, "issues": ["not_an_object"]}
    return {"ok": True, "parsed": parsed, "issues": []}


def build_readback_messages(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    user_content = [{"type": "text", "text": block} for block in point_observation_blocks(evidence)]
    user_content.append({"type": "text", "text": READBACK_INSTRUCTIONS})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def audit_readback_payload(
    messages: list[dict[str, Any]],
    *,
    sample_id: str | None = None,
    family: str | None = None,
    pair_id: str | None = None,
) -> dict[str, Any]:
    serialized = json.dumps(messages, ensure_ascii=False)
    issues: list[str] = []
    if "data:image/png;base64," in serialized:
        issues.append("readback 不得含图像")
    if "[PLAN_CONSTRAINTS]" in serialized or "harnesscad.plan" in serialized.lower():
        issues.append("readback 不得含 CAD Plan 约束")
    if "[POINT_OBSERVATION]" not in serialized:
        issues.append("缺少 POINT_OBSERVATION")
    if "[READBACK_CONSTRAINTS]" not in serialized:
        issues.append("缺少 READBACK_CONSTRAINTS")
    for marker in FORBIDDEN_MARKERS:
        if marker.lower() in serialized.lower():
            issues.append(f"prompt 含禁止标记 {marker}")
    for label in (sample_id, family, pair_id):
        if label and label in serialized:
            issues.append(f"prompt 含 {label}")
    if "api_key" in serialized.lower() or "sk-" in serialized:
        issues.append("prompt 疑似含密钥")
    return {
        "ok": not issues,
        "issues": issues,
        "prompt_sha256": sha256_json(messages),
        "prompt_version": PROMPT_VERSION,
    }


def _resolve_evidence(row: dict[str, Any], key: str) -> dict[str, Any]:
    evidence = row.get("evidence") or {}
    blob = evidence.get(key)
    if key == "p_full" and not isinstance(blob, dict):
        blob = evidence.get("p_comp")
    if key == "p_counterfactual" and not isinstance(blob, dict):
        blob = evidence.get("p_wrong")
    if not isinstance(blob, dict):
        raise KeyError(f"缺少 evidence.{key}")
    return blob


def _state_path(state_dir: Path, pair_id: str, evidence_key: str) -> Path:
    return state_dir / pair_id / f"{evidence_key}.json"


def _summarize(records: list[dict[str, Any]], *, p_full_min: float, follow_min: float) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = {key: [] for key in EVIDENCE_KEYS}
    for record in records:
        by_key[record["evidence_key"]].append(record)

    def _rate(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
        n = len(rows)
        hits = sum(1 for row in rows if (row.get("score") or {}).get(field))
        return {"n": n, "hits": hits, "rate": (hits / n if n else 0.0)}

    p_full = _rate(by_key["p_full"], "match")
    p_repeat = _rate(by_key["p_repeat"], "match")
    p_cf = _rate(by_key["p_counterfactual"], "match")
    follow = _rate(by_key["p_counterfactual"], "follow_counterfactual")
    by_kind: dict[str, dict[str, Any]] = {}
    for record in records:
        kind = str(record.get("kind") or "")
        bucket = by_kind.setdefault(kind, {key: {"n": 0, "hits": 0} for key in EVIDENCE_KEYS})
        key = record["evidence_key"]
        bucket[key]["n"] += 1
        if (record.get("score") or {}).get("match"):
            bucket[key]["hits"] += 1
    for kind, bucket in by_kind.items():
        for key, stats in bucket.items():
            stats["rate"] = stats["hits"] / stats["n"] if stats["n"] else 0.0
    gates = {
        "p_full_ge_80": p_full["rate"] >= p_full_min,
        "p_counterfactual_follow_ge_80": follow["rate"] >= follow_min,
    }
    gates["pass"] = bool(gates["p_full_ge_80"] and gates["p_counterfactual_follow_ge_80"])
    if not records:
        verdict = "无合格对，不能做 readback。"
    elif gates["pass"]:
        verdict = "readback 通过：模型能从 P_full 读回关键事实，且 P_counterfactual 跟随反事实。可以进入 V6b 小探针。"
    elif not gates["p_full_ge_80"]:
        verdict = "P_full readback 未达 80%。先修序列化/Prompt，不要跑 CAD 探针。"
    else:
        verdict = "P_counterfactual 未明显跟随反事实。先修序列化/Prompt，不要跑 CAD 探针。"
    return {
        "prompt_version": PROMPT_VERSION,
        "n_records": len(records),
        "p_full": p_full,
        "p_repeat": p_repeat,
        "p_counterfactual": p_cf,
        "p_counterfactual_follow": follow,
        "by_kind": by_kind,
        "gates": gates,
        "verdict": verdict,
    }


def run_v6b_readback(config_path: str | Path, *, dry_run: bool = True, limit: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    output_root = project_path(config["paths"]["output_root"])
    manifest = read_manifest(project_path(config["paths"]["manifest"]))
    eligible = [row for row in manifest if row.get("eligible")]
    state_dir = output_root / "readback" / ("dryrun" if dry_run else "live") / "state"
    audit_dir = output_root / "readback" / ("dryrun" if dry_run else "live") / "payload_audit"
    state_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    p_full_min = float((config.get("gates") or {}).get("p_full_min", 0.80))
    follow_min = float((config.get("gates") or {}).get("p_counterfactual_follow_min", 0.80))
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    n = 0
    api_settings = None if dry_run else APISettings.from_config(config["api"])
    for row in eligible:
        attached = attach_evidence_payloads(row)
        pair_id = str(row.get("pair_id") or row.get("sample_id"))
        kind = str(row["kind"])
        critical = row.get("critical_fact") or {}
        expected_full = expected_from_evidence(_resolve_evidence(attached, "p_full"), kind=kind, critical=critical)
        expected_cf = expected_from_evidence(
            _resolve_evidence(attached, "p_counterfactual"), kind=kind, critical=critical
        )
        for key in EVIDENCE_KEYS:
            if limit is not None and n >= limit:
                break
            n += 1
            evidence = _resolve_evidence(attached, key)
            messages = build_readback_messages(evidence)
            payload_audit = audit_readback_payload(
                messages,
                sample_id=row.get("sample_id"),
                family=row.get("family"),
                pair_id=pair_id,
            )
            expected = {
                "p_full": expected_full,
                "p_repeat": expected_from_evidence(evidence, kind=kind, critical=critical),
                "p_counterfactual": expected_cf,
            }[key]
            foil = {
                "p_full": expected_cf.get("value"),
                "p_repeat": expected_full.get("value"),
                "p_counterfactual": expected_full.get("value"),
            }[key]
            fingerprint = sha256_json(
                {
                    "prompt_version": PROMPT_VERSION,
                    "evidence_key": key,
                    "evidence": evidence.get("content_hash") or sha256_json(evidence),
                    "model": (config.get("api") or {}).get("default_model"),
                }
            )
            path = _state_path(state_dir, pair_id, key)
            audit_path = audit_dir / pair_id / f"{key}.json"
            atomic_write_json(audit_path, {"audit": payload_audit, "evidence_key": key, "pair_id": pair_id})
            base = {
                "pair_id": pair_id,
                "sample_id": row.get("sample_id"),
                "kind": kind,
                "evidence_key": key,
                "task_fingerprint": fingerprint,
                "payload_audit": payload_audit,
                "expected": expected,
                "dry_run": dry_run,
            }
            if path.is_file():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous.get("task_fingerprint") == fingerprint and previous.get("status") not in {
                    None,
                    "task_failed",
                }:
                    records.append(previous)
                    counts[previous.get("status") or "skipped"] = counts.get(previous.get("status") or "skipped", 0) + 1
                    continue
            if not payload_audit["ok"]:
                record = {**base, "status": "payload_audit_failed", "score": None, "raw_text": None}
                atomic_write_json(path, record)
                records.append(record)
                counts["payload_audit_failed"] = counts.get("payload_audit_failed", 0) + 1
                continue
            if dry_run:
                record = {**base, "status": "dry_run_completed", "score": None, "raw_text": None}
                atomic_write_json(path, record)
                records.append(record)
                counts["dry_run_completed"] = counts.get("dry_run_completed", 0) + 1
                continue
            print(f"[v6b-readback] {pair_id} {key}", flush=True)
            try:
                assert api_settings is not None
                started = time.perf_counter()
                api_result = chat_completion(messages, api_settings)
                parsed_wrap = parse_readback_response(api_result["text"])
                score = score_readback(parsed_wrap.get("parsed"), expected, kind=kind, foil_value=foil)
                record = {
                    **base,
                    "status": "completed" if parsed_wrap["ok"] else "parse_failed",
                    "score": score,
                    "parsed": parsed_wrap.get("parsed"),
                    "parse_issues": parsed_wrap.get("issues"),
                    "raw_text": api_result["text"],
                    "usage": api_result.get("usage"),
                    "latency_sec": time.perf_counter() - started,
                    "model": api_result.get("model"),
                }
            except FatalAPIError as exc:
                record = {**base, "status": "fatal_api_error", "error": str(exc)[:500], "score": None}
                atomic_write_json(path, record)
                counts["fatal_api_error"] = counts.get("fatal_api_error", 0) + 1
                summary = _summarize(records, p_full_min=p_full_min, follow_min=follow_min)
                summary["counts"] = counts
                summary["fatal"] = str(exc)[:500]
                return summary
            except Exception as exc:
                record = {**base, "status": "task_failed", "error": str(exc)[:500], "score": None}
            atomic_write_json(path, record)
            records.append(record)
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        if limit is not None and n >= limit:
            break
    scored = [row for row in records if isinstance(row.get("score"), dict)]
    summary = _summarize(scored if not dry_run else [], p_full_min=p_full_min, follow_min=follow_min)
    if dry_run:
        summary["verdict"] = (
            "dry-run 完成，未调用 API。"
            if counts.get("payload_audit_failed", 0) == 0
            else "dry-run payload 审计失败。"
        )
        summary["gates"]["pass"] = counts.get("payload_audit_failed", 0) == 0
    summary["counts"] = counts
    summary["n_eligible_pairs"] = len(eligible)
    summary["n_tasks"] = n
    atomic_write_json(output_root / "readback" / ("dryrun" if dry_run else "live") / "summary.json", summary)
    return summary
