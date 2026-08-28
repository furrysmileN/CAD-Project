"""确定性 size-matched PointEvidence 错配（derangement）。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .common import atomic_write_json, sha256_json


def _aspect(row: dict[str, Any]) -> tuple[float, float, float]:
    frame = ((row.get("evidence_frame") or row.get("frame") or {}))
    size = frame.get("bbox_size") or []
    if len(size) != 3:
        return (0.0, 0.0, 0.0)
    longest = max(float(size[0]), float(size[1]), float(size[2]), 1e-12)
    return (float(size[0]) / longest, float(size[1]) / longest, float(size[2]) / longest)


def _score(src: dict[str, Any], dst: dict[str, Any], *, seed: int) -> tuple:
    aspect_src = _aspect(src)
    aspect_dst = _aspect(dst)
    aspect_l1 = sum(abs(a - b) for a, b in zip(aspect_src, aspect_dst))
    same_diff = src.get("difficulty") == dst.get("difficulty")
    same_bin = src.get("complexity_bin") == dst.get("complexity_bin")
    same_family = src.get("family") == dst.get("family")
    tie = hashlib.sha256(
        f"{seed}:{src.get('sample_id')}:{dst.get('sample_id')}".encode("utf-8")
    ).hexdigest()
    return (
        0 if same_diff else 1,
        0 if same_bin else 1,
        aspect_l1,
        1 if same_family else 0,
        tie,
    )


def build_shuffle_mapping(
    rows: list[dict[str, Any]],
    *,
    seed: int = 20260818,
) -> dict[str, Any]:
    """每个样本配到另一个样本的完整证据。保证 derangement。"""
    ids = [str(row["sample_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("shuffle 映射要求 sample_id 唯一")
    if len(ids) < 2:
        raise ValueError("shuffle 至少需要 2 个样本")
    by_id = {str(row["sample_id"]): row for row in rows}
    remaining = set(ids)
    mapping: dict[str, str] = {}
    for src_id in sorted(ids):
        src = by_id[src_id]
        candidates = [other for other in remaining if other != src_id]
        if not candidates:
            # 最后一个只剩自己：与前一个交换
            prev = next(iter(mapping))
            mapping[src_id] = mapping[prev]
            mapping[prev] = src_id
            remaining.discard(src_id)
            break
        ranked = sorted(candidates, key=lambda other: _score(src, by_id[other], seed=seed))
        chosen = ranked[0]
        mapping[src_id] = chosen
        remaining.discard(chosen)
    if set(mapping.values()) != set(ids) or any(src == dst for src, dst in mapping.items()):
        # 兜底旋转
        rotated = ids[1:] + ids[:1]
        mapping = dict(zip(ids, rotated))
    payload = {
        "schema_version": "rq2.v5.shuffle.v1",
        "seed": seed,
        "n": len(mapping),
        "mapping": mapping,
        "sha256": "",
    }
    payload["sha256"] = sha256_json({"seed": seed, "mapping": mapping})
    return payload


def write_shuffle_mapping(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def load_shuffle_mapping(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("mapping") or {}
    return {str(key): str(value) for key, value in mapping.items()}
