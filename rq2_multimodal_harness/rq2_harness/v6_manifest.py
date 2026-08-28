"""Load and write V6 manifests. Latent paths stay out of prompt-facing fields."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .common import sha256_file, write_jsonl


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def attach_evidence_payloads(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    evidence = {}
    for key in ("p_comp", "p_repeat", "p_wrong", "p_counterfactual", "p_full"):
        item = (row.get("inputs") or {}).get(key) or (row.get(key) if key in row else None)
        path = None
        if isinstance(item, dict):
            path = item.get("path")
        elif isinstance(item, str):
            path = item
        if path:
            evidence[key] = json.loads(Path(path).read_text(encoding="utf-8"))
    if "p_counterfactual" in evidence and "p_wrong" not in evidence:
        evidence["p_wrong"] = evidence["p_counterfactual"]
    if "p_full" in evidence and "p_comp" not in evidence:
        evidence["p_comp"] = evidence["p_full"]
    images = row.get("images") or {}
    if "views" not in images and row.get("image_dir"):
        views = []
        root = Path(row["image_dir"])
        for view in ("front", "side", "top", "isometric"):
            path = root / f"{view}.png"
            views.append({"view": view, "path": str(path), "sha256": sha256_file(path) if path.is_file() else ""})
        images = {"views": views}
    out["evidence"] = evidence
    out["images"] = images
    return out


def iter_tasks(rows: list[dict[str, Any]], conditions: list[str], repeats: list[int]) -> Iterator[dict[str, Any]]:
    for row in rows:
        for condition in conditions:
            for repeat_id in repeats:
                yield {"row": row, "condition": condition, "repeat_id": repeat_id}
