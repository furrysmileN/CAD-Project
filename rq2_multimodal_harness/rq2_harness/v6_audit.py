"""Leakage, manipulation, token, and coordinate audits. GT is allowed only here."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import sha256_json
from .v6_conditions import FORBIDDEN_MARKERS, audit_v6_payload, build_v6_messages, parse_condition
from .v6_corruptions import unchanged_except_critical
from .v6_fact_masks import TOKEN_TOLERANCE, evidence_token_count, repeat_contains_critical


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_evidence_files(sample_dir: Path, latent: dict[str, Any]) -> dict[str, Any]:
    p_comp = _load_json(sample_dir / "p_comp.json")
    p_repeat = _load_json(sample_dir / "p_repeat.json")
    p_wrong = _load_json(sample_dir / "p_wrong.json")
    critical = latent.get("critical_fact") or {}
    issues: list[str] = []
    for blob, name in ((p_comp, "p_comp"), (p_repeat, "p_repeat"), (p_wrong, "p_wrong")):
        if blob.get("inputs", {}).get("reads_gt"):
            issues.append(f"{name} 声明读取了 GT")
        dumped = json.dumps(blob)
        if '"latent_spec"' in dumped or '"gt_plan"' in dumped:
            issues.append(f"{name} JSON 含 GT 标记")
    if not any(item.get("role") == "primary_critical" for item in p_comp.get("cad_facts") or []):
        issues.append("P_comp 缺少 primary_critical 测量")
    if repeat_contains_critical(p_repeat, critical):
        issues.append("P_repeat 仍含 critical fact")
    target = evidence_token_count(p_comp)
    got = evidence_token_count(p_repeat)
    if not (target * (1 - TOKEN_TOLERANCE) <= got <= target * (1 + TOKEN_TOLERANCE * 3)):
        issues.append(f"token 不平衡: P_repeat {got} vs P_comp {target}")
    if not unchanged_except_critical(p_comp, p_wrong, critical):
        issues.append("P_wrong 改动了非目标字段")
    if "corruption" not in p_wrong:
        issues.append("P_wrong 缺少 corruption 记录")
    measured = next((item for item in p_comp.get("cad_facts") or [] if item.get("role") == "primary_critical"), None)
    gt_value = critical.get("value")
    measured_ok = measured is not None
    return {
        "ok": not issues,
        "issues": issues,
        "token_p_comp": target,
        "token_p_repeat": got,
        "measured_primary": measured,
        "gt_critical": {"fact_id": critical.get("fact_id"), "value": gt_value, "category": critical.get("category")},
        "measured_present": measured_ok,
        "content_hash": {
            "p_comp": p_comp.get("content_hash"),
            "p_repeat": p_repeat.get("content_hash"),
            "p_wrong": p_wrong.get("content_hash"),
        },
    }


def audit_row_payloads(row: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for condition_id in ("C0", "C1", "C2", "C3", "C4", "C5"):
        spec = parse_condition(condition_id)
        messages = build_v6_messages(row, spec)
        result = audit_v6_payload(messages, spec, sample_id=row.get("sample_id"), family=row.get("family"))
        result["prompt_sha256"] = result["prompt_sha256"]
        results.append(result)
    return results


def leakage_scan_text(text: str) -> list[str]:
    issues = []
    lowered = text.lower()
    for marker in FORBIDDEN_MARKERS:
        if marker.lower() in lowered:
            issues.append(marker)
    return issues


def summarize_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "n_ok": sum(1 for row in rows if row.get("ok")),
        "sha256": sha256_json(rows),
    }
