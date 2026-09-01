"""V6b diagnostic payloads: C2B (P_B only) and TB (fact-equivalent text).

Reuses frozen SYSTEM_PROMPTS / PLAN_TEMPLATES / T0_TEXT / POINT_* serialization.
Does not modify v6_conditions.py (keeps C0–C5 fingerprints).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import sha256_json
from .prompting import PLAN_TEMPLATES, SYSTEM_PROMPTS
from .v6_conditions import FORBIDDEN_MARKERS, FILE_EXTS, T0_TEXT, point_observation_blocks
from .v6_fact_masks import estimate_tokens

DIAG_IDS = ("C2B", "TB")
SEMANTIC_IDS = {
    "C2B": "DIAG_PB_ONLY",
    "TB": "DIAG_TEXT_FACT",
}
DIAG_FORBIDDEN = FORBIDDEN_MARKERS + ("DIAG_PB_ONLY", "DIAG_TEXT_FACT")


@dataclass(frozen=True, slots=True)
class DiagConditionSpec:
    condition_id: str
    images: bool
    evidence_key: str | None
    text_fact: bool

    @property
    def semantic(self) -> str:
        return SEMANTIC_IDS[self.condition_id]


def parse_diag_condition(condition_id: str) -> DiagConditionSpec:
    if condition_id not in DIAG_IDS:
        raise ValueError(f"未知诊断条件: {condition_id}")
    return DiagConditionSpec(
        condition_id=condition_id,
        images=False,
        evidence_key="p_counterfactual" if condition_id == "C2B" else None,
        text_fact=condition_id == "TB",
    )


def fact_sentence(kind: str, spec_b: dict[str, Any]) -> str:
    """Deterministic English sentence for B's critical fact. No family / sample_id."""
    critical = spec_b.get("critical_fact") or {}
    value = critical.get("value")
    if kind == "pocket_depth":
        return f"The top-face pocket has a normalized depth of {value}."
    if kind == "blind_depth":
        return f"There is a blind hole with a normalized depth of {value}."
    if kind == "through_vs_blind":
        depth = critical.get("depth")
        if depth is None:
            op_id = critical.get("operation_id")
            for operation in spec_b.get("operations") or []:
                if operation.get("id") == op_id and "depth" in operation:
                    depth = operation.get("depth")
                    break
        if str(value) == "blind":
            return f"The hole is blind, with a normalized depth of {depth}."
        return "The hole is a through hole."
    if kind == "hidden_presence":
        if not value:
            return "There is no hole on the back face."
        diameter = critical.get("diameter")
        depth = critical.get("depth")
        radius = None
        try:
            radius = round(float(diameter) / 2.0, 4)
        except (TypeError, ValueError):
            radius = None
        if radius is not None and depth is not None:
            return (
                f"A hole is present on the back face, with a normalized radius of {radius} "
                f"and a normalized depth of {depth}."
            )
        return "A hole is present on the back face."
    raise ValueError(f"未知 kind: {kind}")


def _plan_constraints() -> str:
    return (
        "[PLAN_CONSTRAINTS]\n"
        "Return exactly one harnesscad.plan.v2 JSON object. "
        "Canonical frame: bbox center [0,0,0], longest edge 1.0. "
        "No commentary and no extra keys.\n"
        "Required JSON shape (example values are illustrative only):\n"
        + json.dumps(PLAN_TEMPLATES["v3"], ensure_ascii=False, indent=2)
        + "\nDo not copy the example dimensions. Every operation must include all "
        "fields required by the system schema. Prefer a simpler valid approximation "
        "when observations are ambiguous. Output JSON only."
    )


def build_diag_messages(row: dict[str, Any], spec: DiagConditionSpec) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": f"[TEXT_INTENT]\n{T0_TEXT}"},
    ]
    if spec.evidence_key:
        evidence = (row.get("evidence") or {}).get(spec.evidence_key)
        if not isinstance(evidence, dict):
            raise KeyError(f"manifest 缺少 evidence.{spec.evidence_key}")
        for block in point_observation_blocks(evidence):
            user_content.append({"type": "text", "text": block})
    if spec.text_fact:
        sentence = str(row.get("text_fact") or "").strip()
        if not sentence:
            raise KeyError("缺少 text_fact")
        user_content.append({"type": "text", "text": f"[TEXT_FACT]\n{sentence}"})
    user_content.append({"type": "text", "text": _plan_constraints()})
    return [
        {"role": "system", "content": SYSTEM_PROMPTS["v3"]},
        {"role": "user", "content": user_content},
    ]


def audit_diag_payload(
    messages: list[dict[str, Any]],
    spec: DiagConditionSpec,
    *,
    sample_id: str | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    serialized = json.dumps(messages, ensure_ascii=False)
    issues: list[str] = []
    image_count = serialized.count("data:image/png;base64,")
    if image_count != 0:
        issues.append(f"image_count {image_count} != expected 0")
    if "[TEXT_INTENT]" not in serialized:
        issues.append("缺少 TEXT_INTENT")
    if "[IMAGE_OBSERVATION]" in serialized:
        issues.append("诊断条件不得含 IMAGE_OBSERVATION")
    needs_point = spec.evidence_key is not None
    if needs_point != ("[POINT_OBSERVATION]" in serialized):
        issues.append("POINT_OBSERVATION 与条件不一致")
    if needs_point and "[POINT_HYPOTHESIS]" not in serialized:
        issues.append("缺少 POINT_HYPOTHESIS")
    if spec.text_fact != ("[TEXT_FACT]" in serialized):
        issues.append("TEXT_FACT 与条件不一致")
    if spec.text_fact and "[POINT_OBSERVATION]" in serialized:
        issues.append("TB 不得含 POINT_OBSERVATION")
    if spec.semantic in serialized:
        issues.append("prompt 含语义条件名")
    for marker in DIAG_FORBIDDEN:
        if marker.lower() in serialized.lower():
            issues.append(f"prompt 含禁止标记 {marker}")
    lowered = serialized.lower()
    for ext in FILE_EXTS:
        if any(token in lowered for token in (f"{ext}/", f"{ext}\\", f'{ext}"', f"{ext}'", f"target{ext}")):
            issues.append(f"prompt 含禁止标记 {ext}")
    if sample_id and sample_id in serialized:
        issues.append("prompt 含 sample_id")
    if family and family in serialized:
        issues.append("prompt 含 family")
    if "api_key" in serialized.lower() or "sk-" in serialized:
        issues.append("prompt 疑似含密钥")
    return {
        "ok": not issues,
        "issues": issues,
        "prompt_sha256": sha256_json(messages),
        "approx_tokens": estimate_tokens(serialized),
        "image_count": image_count,
        "condition_id": spec.condition_id,
    }
