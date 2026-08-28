"""V6 C0–C5 payload assembly. Neutral IDs only; semantic names stay off-prompt."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import sha256_file, sha256_json
from .prompting import PLAN_TEMPLATES, SYSTEM_PROMPTS, _image_data_url
from .v6_fact_masks import estimate_tokens

PROMPT_VERSION = "rq2.v6.prompt.v2"
T0_TEXT = (
    "Generate a valid HarnessCAD Plan for a general mechanical part from the observations below. "
    "Do not assume a named part family, hole count, depth, or exact dimensions that are not present "
    "in the observations."
)
NEUTRAL_IDS = ("C0", "C1", "C2", "C3", "C4", "C5")
SEMANTIC_IDS = {
    "C0": "C0_BASE",
    "C1": "C1_IMAGE",
    "C2": "C2_POINT",
    "C3": "C3_COMPLEMENT",
    "C4": "C4_REPEAT",
    "C5": "C5_WRONG",
}
EVIDENCE_KEY = {"C2": "p_comp", "C3": "p_comp", "C4": "p_repeat", "C5": "p_wrong"}
FORBIDDEN_MARKERS = (
    "gt_code",
    "latent_spec",
    "critical_fact",
    "C0_BASE",
    "C1_IMAGE",
    "C3_COMPLEMENT",
    "C4_REPEAT",
    "C5_WRONG",
    "P_comp",
    "P_repeat",
    "P_wrong",
    "processed\\",
    "processed/",
    "v6_pilot_",
    "v6_confirm_",
    "v6b_probe_",
    "v6b_pair_",
)
FILE_EXTS = (".npy", ".step", ".stp", ".stl")


@dataclass(frozen=True, slots=True)
class V6ConditionSpec:
    condition_id: str
    images: bool
    evidence_key: str | None

    @property
    def semantic(self) -> str:
        return SEMANTIC_IDS[self.condition_id]


def parse_condition(condition_id: str) -> V6ConditionSpec:
    if condition_id not in NEUTRAL_IDS:
        raise ValueError(f"未知 V6 条件: {condition_id}")
    return V6ConditionSpec(
        condition_id=condition_id,
        images=condition_id in {"C1", "C3", "C4", "C5"},
        evidence_key=EVIDENCE_KEY.get(condition_id),
    )


def _image_block(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    caption = (
        "[IMAGE_OBSERVATION]\n"
        "Surface-render inputs: four RGB views in fixed front, side, top, and isometric order. "
        "No cross-section and no dimension annotations. Colors and lighting describe appearance only."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": caption}]
    for item in images:
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(item["path"]), 1024)}})
    return content


def point_observation_blocks(evidence: dict[str, Any]) -> list[str]:
    """Same POINT_* serialization used by C2–C5. Readback must reuse this path."""
    return _evidence_blocks(evidence)


def _evidence_blocks(evidence: dict[str, Any]) -> list[str]:
    observation = {
        "schema": evidence.get("schema"),
        "cloud_id": evidence.get("cloud_id"),
        "frame": evidence.get("frame"),
        "quality": evidence.get("quality"),
        "symmetry_candidates": evidence.get("symmetry_candidates"),
        "primitive_candidates": evidence.get("primitive_candidates"),
        "sections": evidence.get("sections"),
        "cad_facts": evidence.get("cad_facts"),
        "visible_surface_measurements": evidence.get("visible_surface_measurements"),
    }
    hypothesis = {
        "hypotheses": evidence.get("hypotheses") or [],
        "uncertainties": evidence.get("uncertainties") or [],
    }
    return [
        "[POINT_OBSERVATION]\n"
        "point-cloud-derived structured geometric evidence in the canonical frame "
        "(bbox center at origin, longest bbox edge = 1). These are local measurements, "
        "not native point-cloud tokens or embeddings.\n"
        + json.dumps(observation, ensure_ascii=False, indent=2),
        "[POINT_HYPOTHESIS]\n"
        "Hypotheses, not confirmed measurements.\n"
        + json.dumps({"hypotheses": hypothesis["hypotheses"]}, ensure_ascii=False, indent=2),
        "[POINT_UNCERTAINTY]\n"
        "Uncertainty and missing-measurement notes.\n"
        + json.dumps({"uncertainties": hypothesis["uncertainties"]}, ensure_ascii=False, indent=2),
    ]


def build_v6_messages(row: dict[str, Any], spec: V6ConditionSpec) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": f"[TEXT_INTENT]\n{T0_TEXT}"},
    ]
    if spec.images:
        images = ((row.get("images") or {}).get("views")) or row.get("image_views")
        if not images:
            raise KeyError("manifest 缺少四视图图像")
        user_content.extend(_image_block(images))
    if spec.evidence_key:
        evidence = (row.get("evidence") or {}).get(spec.evidence_key)
        if not isinstance(evidence, dict):
            raise KeyError(f"manifest 缺少 evidence.{spec.evidence_key}")
        for block in _evidence_blocks(evidence):
            user_content.append({"type": "text", "text": block})
    user_content.append(
        {
            "type": "text",
            "text": (
                "[PLAN_CONSTRAINTS]\n"
                "Return exactly one harnesscad.plan.v2 JSON object. "
                "Canonical frame: bbox center [0,0,0], longest edge 1.0. "
                "No commentary and no extra keys.\n"
                "Required JSON shape (example values are illustrative only):\n"
                + json.dumps(PLAN_TEMPLATES["v3"], ensure_ascii=False, indent=2)
                + "\nDo not copy the example dimensions. Every operation must include all "
                "fields required by the system schema. Prefer a simpler valid approximation "
                "when observations are ambiguous. Output JSON only."
            ),
        }
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPTS["v3"]},
        {"role": "user", "content": user_content},
    ]


def audit_v6_payload(
    messages: list[dict[str, Any]],
    spec: V6ConditionSpec,
    *,
    sample_id: str | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    serialized = json.dumps(messages, ensure_ascii=False)
    issues: list[str] = []
    image_count = serialized.count("data:image/png;base64,")
    expected_images = 4 if spec.images else 0
    if image_count != expected_images:
        issues.append(f"image_count {image_count} != expected {expected_images}")
    if "[TEXT_INTENT]" not in serialized:
        issues.append("缺少 TEXT_INTENT")
    if spec.images != ("[IMAGE_OBSERVATION]" in serialized):
        issues.append("IMAGE_OBSERVATION 与条件不一致")
    needs_point = spec.evidence_key is not None
    if needs_point != ("[POINT_OBSERVATION]" in serialized):
        issues.append("POINT_OBSERVATION 与条件不一致")
    if needs_point and "[POINT_HYPOTHESIS]" not in serialized:
        issues.append("缺少 POINT_HYPOTHESIS")
    if needs_point and "[POINT_UNCERTAINTY]" not in serialized:
        issues.append("缺少 POINT_UNCERTAINTY")
    if spec.semantic in serialized or spec.semantic.lower() in serialized.lower():
        issues.append("prompt 含语义条件名")
    for marker in FORBIDDEN_MARKERS:
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


def image_bundle_hash(views: list[dict[str, Any]]) -> str:
    payload = [{"view": item.get("view"), "sha256": item.get("sha256") or sha256_file(Path(item["path"]))} for item in views]
    return sha256_json(payload)
