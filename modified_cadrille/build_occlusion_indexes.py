#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_relative(path: Path, index_dir: Path) -> str:
    if not path.is_absolute():
        return str(path)
    return os.path.relpath(str(path.resolve()), str(index_dir.resolve()))


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _build_occlusion_lookup(
    occlusion_manifest: Path,
    index_dir: Path,
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    with occlusion_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") != "done":
                continue
            source_id = row.get("source_sample_id")
            image_paths = row.get("image_paths", [])
            if not source_id or not image_paths:
                continue
            rel_images = [
                _to_relative(Path(p).resolve(), index_dir) for p in image_paths
            ]
            rel_masks = [
                _to_relative(Path(p).resolve(), index_dir)
                for p in row.get("mask_paths", [])
            ]
            point_path = row.get("point_path")
            entry = {
                "variant_sample_id": row.get("sample_id", f"{source_id}__occ_000"),
                "image_paths": rel_images,
                "mask_paths": rel_masks,
                "pointcloud_path": _to_relative(Path(point_path).resolve(), index_dir)
                if point_path
                else None,
            }
            table[source_id] = entry
    return table


def _with_occluded_images(
    rec: dict[str, Any],
    occ: dict[str, Any],
    *,
    keep_sample_id: bool,
) -> dict[str, Any]:
    out = dict(rec)
    if not keep_sample_id:
        out["sample_id"] = occ["variant_sample_id"]
    out["image_paths"] = list(occ["image_paths"])
    out["images"] = list(occ["image_paths"])
    out["occlusion_mask_paths"] = list(occ["mask_paths"])
    out["occlusion_mode"] = "occluder"
    out["occlusion_variant"] = "occ_000"
    out["is_occluded"] = True
    # Keep point cloud path unchanged for occluder mode.
    return out


def build_indexes(
    clean_index: Path,
    occlusion_manifest: Path,
    occluded_output: Path,
    mix50_output: Path,
) -> dict[str, Any]:
    index_dir = clean_index.resolve().parent
    clean_rows = _load_jsonl(clean_index.resolve())
    occlusion_lookup = _build_occlusion_lookup(occlusion_manifest.resolve(), index_dir)

    occluded_rows: list[dict[str, Any]] = []
    mix_rows: list[dict[str, Any]] = []
    train_clean = 0
    train_occ = 0
    val_clean = 0
    missing = 0

    for rec in clean_rows:
        sample_id = rec.get("sample_id")
        split = rec.get("split")
        occ = occlusion_lookup.get(sample_id)
        if occ is None:
            missing += 1

        if occ is not None:
            occ_row_same_id = _with_occluded_images(rec, occ, keep_sample_id=True)
            occluded_rows.append(occ_row_same_id)
        else:
            fallback = dict(rec)
            fallback["is_occluded"] = False
            fallback["occlusion_mode"] = "none"
            occluded_rows.append(fallback)

        base_row = dict(rec)
        base_row["is_occluded"] = False
        base_row["occlusion_mode"] = "none"
        mix_rows.append(base_row)
        if split == "train":
            train_clean += 1
            if occ is not None:
                mix_rows.append(_with_occluded_images(rec, occ, keep_sample_id=False))
                train_occ += 1
        elif split == "val":
            val_clean += 1

    _save_jsonl(occluded_output.resolve(), occluded_rows)
    _save_jsonl(mix50_output.resolve(), mix_rows)

    summary = {
        "clean_index": str(clean_index.resolve()),
        "occlusion_manifest": str(occlusion_manifest.resolve()),
        "occluded_output": str(occluded_output.resolve()),
        "mix50_output": str(mix50_output.resolve()),
        "clean_rows": len(clean_rows),
        "occluded_rows": len(occluded_rows),
        "mix50_rows": len(mix_rows),
        "train_clean_rows": train_clean,
        "train_occluded_rows_added": train_occ,
        "val_clean_rows": val_clean,
        "missing_occlusion_rows": missing,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build occluded and mix50 index files.")
    parser.add_argument("--clean-index", type=Path, required=True)
    parser.add_argument("--occlusion-manifest", type=Path, required=True)
    parser.add_argument("--occluded-output", type=Path, required=True)
    parser.add_argument("--mix50-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_indexes(
        clean_index=args.clean_index,
        occlusion_manifest=args.occlusion_manifest,
        occluded_output=args.occluded_output,
        mix50_output=args.mix50_output,
    )
    summary_path = args.summary_output
    if summary_path is None:
        summary_path = args.mix50_output.with_suffix(".build_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
