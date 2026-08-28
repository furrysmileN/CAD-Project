"""L3-only, auditable T3 text encoding.

No call is made at import time. Tests and dry runs inject a mock compatible
with :func:`rq2_harness.api_client.chat_completion`.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable

from .api_client import APISettings, chat_completion
from .common import atomic_write_json, sha256_json


T3_FIELDS = (
    "object_type",
    "overall_shape",
    "primary_features",
    "secondary_features",
    "spatial_relations",
    "dimensions_and_units",
    "uncertainties",
)

T3_SYSTEM_PROMPT = """You transform one L3 CAD description into a conservative structured summary.
Use only facts explicitly present in the supplied L3 text. Do not infer or add any number, count,
dimension, angle, tolerance, unit, material, manufacturing process, feature, or relation.
Return one JSON object with exactly these seven string fields and no other fields:
object_type, overall_shape, primary_features, secondary_features, spatial_relations,
dimensions_and_units, uncertainties. Use an empty string when the source does not state a field.
Do not output markdown or commentary."""


def build_t3_messages(l3: str) -> list[dict[str, Any]]:
    if not isinstance(l3, str) or not l3.strip():
        raise ValueError("T3 仅接受非空 L3 文本")
    return [
        {"role": "system", "content": T3_SYSTEM_PROMPT},
        {"role": "user", "content": "L3 SOURCE (the only evidence):\n" + l3.strip()},
    ]


def _extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.IGNORECASE | re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("T3 响应必须为 JSON object")
    if set(value) != set(T3_FIELDS):
        raise ValueError(f"T3 字段必须恰为 {list(T3_FIELDS)}")
    if any(not isinstance(value[field], str) for field in T3_FIELDS):
        raise ValueError("T3 七字段的值必须全部是字符串")
    return {field: value[field].strip() for field in T3_FIELDS}


_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*\d+(?:\.\d+)?)?(?![\w.])")
_UNIT_RE = re.compile(
    r"(?<![A-Za-z])(?:mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|m|met(?:er|re)s?|"
    r"in(?:ch(?:es)?)?|ft|feet|foot|deg(?:ree)?s?|°|rad(?:ian)?s?|%|µm|um)(?![A-Za-z])",
    re.IGNORECASE,
)


def _normalize_number(value: str) -> str:
    compact = re.sub(r"\s+", "", value)
    if "/" in compact:
        return "/".join(_normalize_number(part) for part in compact.split("/", 1))
    try:
        number = float(compact)
    except ValueError:
        return compact.lower()
    return format(number, ".15g")


def extract_numbers_and_units(text: str) -> dict[str, list[str]]:
    return {
        "numbers": [_normalize_number(match.group(0)) for match in _NUMBER_RE.finditer(text)],
        "units": [match.group(0).lower() for match in _UNIT_RE.finditer(text)],
    }


def check_no_added_numbers_or_units(source_l3: str, encoded: dict[str, str] | str) -> dict[str, Any]:
    output_text = encoded if isinstance(encoded, str) else " ".join(encoded[field] for field in T3_FIELDS)
    source = extract_numbers_and_units(source_l3)
    output = extract_numbers_and_units(output_text)
    source_numbers = set(source["numbers"])
    source_units = set(source["units"])
    added_numbers = sorted(set(output["numbers"]) - source_numbers)
    added_units = sorted(set(output["units"]) - source_units)
    return {
        "ok": not added_numbers and not added_units,
        "source": source,
        "output": output,
        "added_numbers": added_numbers,
        "added_units": added_units,
    }


def _fixed_settings(settings: APISettings) -> APISettings:
    extra = dict(settings.extra_body)
    extra["enable_thinking"] = False
    return replace(settings, temperature=0.0, extra_body=extra)


def encode_t3(
    l3: str,
    cache_dir: str | Path,
    settings: APISettings,
    *,
    source_id: str = "l3",
    force: bool = False,
    api_call: Callable[[list[dict[str, Any]], APISettings], dict[str, Any]] = chat_completion,
) -> dict[str, Any]:
    """Encode one L3 description; cache identity is source + prompt + model."""
    messages = build_t3_messages(l3)
    fixed_settings = _fixed_settings(settings)
    resolved = fixed_settings.resolved(require_key=False)
    source_hash = sha256_json({"L3": l3})
    prompt_hash = sha256_json(messages)
    cache_key = sha256_json({
        "source": source_hash,
        "prompt": prompt_hash,
        "model": resolved["model"],
    })
    cache_path = Path(cache_dir) / f"{re.sub(r'[^A-Za-z0-9._-]', '_', source_id)}__t3.json"
    if cache_path.is_file() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == cache_key:
            return cached

    api_result = api_call(messages, fixed_settings)
    raw_response = str(api_result.get("text", ""))
    after = _extract_json(raw_response)
    preservation = check_no_added_numbers_or_units(l3, after)
    state = {
        "schema_version": "rq2.t3.v1",
        "source_id": source_id,
        "source_sha256": source_hash,
        "prompt_sha256": prompt_hash,
        "model": resolved["model"],
        "cache_key": cache_key,
        "request": {"temperature": 0.0, "thinking": False, "input_level": "L3"},
        "before": l3,
        "after": after,
        "raw_response": raw_response,
        "before_sha256": source_hash,
        "after_sha256": sha256_json(after),
        "raw_response_sha256": sha256_json({"text": raw_response}),
        "usage": dict(api_result.get("usage") or {}),
        "number_unit_preservation": preservation,
    }
    state["record_sha256"] = sha256_json(state)
    atomic_write_json(cache_path, state)
    return state


# Descriptive alias for callers that use the encoding name in their API.
prepare_t3_encoding = encode_t3


def generate_t3_audit_checklist(
    records: Iterable[dict[str, Any]],
    output_path: str | Path | None = None,
    *,
    sample_count: int = 12,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select a deterministic 12-record human audit and optionally save it."""
    rows = list(records)
    if sample_count < 0:
        raise ValueError("sample_count 不能为负")
    ordered = sorted(rows, key=lambda row: (str(row.get("source_id", "")), str(row.get("cache_key", ""))))
    random.Random(seed).shuffle(ordered)
    selected = ordered[:sample_count]
    checklist = [
        {
            "audit_index": index + 1,
            "source_id": row.get("source_id"),
            "cache_key": row.get("cache_key"),
            "source_sha256": row.get("source_sha256"),
            "before": row.get("before"),
            "after": row.get("after"),
            "automatic_number_unit_check": row.get("number_unit_preservation"),
            "human_checks": {
                "only_l3_evidence": None,
                "exactly_seven_fields": None,
                "no_added_number_or_unit": None,
                "no_added_geometry_or_semantics": None,
                "uncertainty_preserved": None,
                "accept": None,
                "notes": "",
            },
        }
        for index, row in enumerate(selected)
    ]
    if output_path is not None:
        payload = {
            "schema_version": "rq2.t3_audit.v1",
            "requested_samples": sample_count,
            "available_records": len(rows),
            "seed": seed,
            "checklist": checklist,
        }
        payload["audit_sha256"] = sha256_json(payload)
        atomic_write_json(Path(output_path), payload)
    return checklist
