#!/usr/bin/env python3
"""Build compact STEP/OFS contexts before LLM description generation."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from cad_data_gen.cad_context_cleaner import build_compact_technical_context
from cad_data_gen.describe_step_with_deepseek import (
    extract_mesh_metrics,
    parse_step_statistics,
    summarize_feat_file,
    summarize_meta_file,
    summarize_ofs_features,
)
from cad_data_gen.describe_step_with_qwen import (
    StepRecord,
    build_ofs_index,
    find_render_images,
    load_asset_manifest_records,
    load_done_ids,
    resolve_ofs_path,
    resolve_sibling_data_path,
    scan_step_records,
)


def _path_or_none(path: Optional[Path]) -> Optional[str]:
    return str(path) if path is not None else None


def _render_images_for_record(record: StepRecord, max_images: int) -> list[Path]:
    images = list(record.image_paths) or find_render_images(record.render_dir, record.sample_id)
    return [path for path in images if path.is_file()][:max_images]


def clean_record_context(
    record: StepRecord,
    args: argparse.Namespace,
    ofs_index: dict[str, Path],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    if not record.step_path.is_file():
        return None, {
            "sample_id": record.sample_id,
            "stage": "locate_step",
            "error": f"STEP file not found: {record.step_path}",
        }

    render_images = _render_images_for_record(record, args.max_images)
    if not render_images and args.require_images:
        return None, {
            "sample_id": record.sample_id,
            "stage": "locate_render_images",
            "error": f"No render images found for {record.sample_id}",
        }

    try:
        step_stats = parse_step_statistics(record.step_path, args.max_step_chars)
    except Exception as exc:
        return None, {
            "sample_id": record.sample_id,
            "stage": "parse_step",
            "error": str(exc),
        }

    mesh_metrics, mesh_error = (record.mesh_metrics, record.mesh_metrics_error)
    mesh_metrics_source = "asset_manifest" if isinstance(mesh_metrics, dict) else None
    if mesh_metrics is None and not args.skip_mesh_metrics:
        mesh_metrics, mesh_error = extract_mesh_metrics(
            record.step_path,
            triangle_face_tol=args.triangle_face_tol,
            angle_tol_rads=args.angle_tol_rads,
        )
        mesh_metrics_source = "recomputed_from_step" if isinstance(mesh_metrics, dict) else None
    elif mesh_metrics is None and args.skip_mesh_metrics:
        mesh_metrics_source = "skipped"

    ofs_path = resolve_ofs_path(record, ofs_index)
    ofs_summary, ofs_error = (None, None)
    if ofs_path is not None:

tures=args.raw_max_ofs_features)
    elif args.ofs_dir:
        ofs_error = "No matching OFS file found"
    ofs_status = "matched" if ofs_path is not None else "missing"
    description_context_level = ("visual_geometry_with_ofs" if ofs_path is not None else "visual_g
eometry_only"
    )
    feat_path = record.feat_path or resolve_sibling_data_path(record, "_f
eatures_")meta_path = record.meta_path or resolve_sibling_data_path(record, "_m
etadata_")
    feat_summary, feat_error = (None, None)
    meta_summary, meta_error = (None, None)
    if args.context_mode == "full_compact":
        feat_summary, feat_error = summarize_feat_file(feat_path)
        meta_summary, meta_error = summarize_meta_file(meta_path)

    compact_context = build_compact_technical_context(
        sample_id=record.sample_id,
        relative_step_path=record.relative_step_path,
        render_image_count=len(render_images),
        point_path=record.point_path,
        step_stats=step_stats,
        mesh_metrics=mesh_metrics,
        mesh_error=mesh_error,
        ofs_summary=ofs_summary,
        ofs_error=ofs_error,
        feat_summary=feat_summary,
        feat_error=feat_error,
        meta_summary=meta_summary,
        meta_error=meta_error,
        mode=args.context_mode,
        max_ofs_features=args.compact_max_ofs_features,
    )

    result = {
        "sample_id": record.sample_id,
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step_path": record.relative_step_path,
        "absolute_step_path": str(record.step_path),
        "ofs_path": _path_or_none(ofs_path),
        "ofs_status": ofs_status,
        "has_ofs": ofs_path is not None,
        "description_context_level": description_context_level,
        "point_path": _path_or_none(record.point_path),
        "mesh_path": _path_or_none(record.mesh_path),
        "conversion_backend": record.conversion_backend,
        "conversion_metadata": record.conversion_metadata,
        "render_image_paths": [str(path) for path in render_images],
        "compact_context": compact_context,
        "context_mode": args.context_mode,
        "raw_stats": {
            "step_total_chars": step_stats.get("step_total_chars"),"step_entity_count_total": step_stats.get("entity_count_tota
l"),"raw_ofs_feature_count_total": (ofs_summary or {}).get("featu
re_count_total") if isinstance(ofs_summary, dict) else None,
            "mesh_metrics_source": mesh_metrics_source,"compact_context_chars": len(json.dumps(compact_context, ensu
re_ascii=False, separators=(",", ":"))),
        },
    }
    if args.include_raw_summaries:result["raw_step_statistics"] = {key: value for key, value in ste
p_stats.items() if key != "step_excerpt"}
        result["raw_ofs_summary"] = ofs_summary
        result["raw_ofs_error"] = ofs_error
        result["raw_mesh_metrics"] = mesh_metrics
        result["raw_mesh_error"] = mesh_error
    return result, None


def _load_records(args: argparse.Namespace) -> list[StepRecord]:
    if args.manifest:records = load_asset_manifest_records(Path(args.manifest).resolve
(), args.input_dir, args.render_root)
    else:records = scan_step_records(args.input_dir, args.recursive, args.
filename_pattern)
        if args.render_dir:
            records = [
                StepRecord(
                    sample_id=record.sample_id,
                    step_path=record.step_path,
                    relative_step_path=record.relative_step_path,
                    source_index=record.source_index,
                    dataset_key=record.dataset_key,
                    render_dir=args.render_dir,
                )
                for record in records
            ]
    if args.offset:
        records = records[args.offset:]
    if args.limit is not None:
        records = records[: args.limit]
    return records


def parse_args() -> argparse.Namespace:parser = argparse.ArgumentParser(description="Clean STEP/OFS context
for CAD shape description prompts")parser.add_argument("--input-dir", required=True, help="Input directo
ry containing STEP files")parser.add_argument("--output", required=True, help="Output compact c
ontext JSONL path")parser.add_argument("--manifest", help="Asset manifest JSONL from poi
nt cloud/render generation")parser.add_argument("--render-root", help="Root directory for relativ
e image/point paths in manifest")parser.add_argument("--render-dir", help="Directory containing rende
r images when no manifest image paths are provided")parser.add_argument("--ofs-dir", help="Directory containing OFS Featu
reScript YAML files")parser.add_argument("--context-mode", choices=["minimal", "balance
d", "full_compact"], default="balanced")parser.add_argument("--compact-max-ofs-features", type=int, default=2
4, help="Max compact OFS operations kept per sample")parser.add_argument("--raw-max-ofs-features", type=int, default=120,
help="Max raw OFS operations parsed before compaction")parser.add_argument("--max-images", type=int, default=4, help="Max re
nder paths recorded per sample")
    parser.add_argument("--max-step-chars", type=int, default=0)
    parser.add_argument("--triangle-face-tol", type=float, default=0.01)
    parser.add_argument("--angle-tol-rads", type=float, default=0.1)parser.add_argument("--skip-mesh-metrics", action="store_true", help
="Skip CAD tessellation metrics for faster cleaning")parser.add_argument("--require-images", action="store_true", help="Fa
il a sample if render images are missing")parser.add_argument("--include-raw-summaries", action="store_true", h
elp="Debug only: include raw STEP/OFS summaries in output")parser.add_argument("--limit", type=int, default=None, help="Process
at most N records")parser.add_argument("--offset", type=int, default=0, help="Skip firs
t N records")parser.add_argument("--resume", action="store_true", help="Skip sampl
e_ids already present in output JSONL")parser.add_argument("--recursive", action="store_true", help="Scan in
put-dir recursively when no manifest is given")
    parser.add_argument("--filename-pattern", default="*.step")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input_dir = Path(args.input_dir).resolve()
    args.output = Path(args.output).resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.render_root:
        args.render_root = Path(args.render_root).resolve()
    if args.render_dir:
        args.render_dir = Path(args.render_dir).resolve()
    if args.ofs_dir:
        args.ofs_dir = Path(args.ofs_dir).resolve()

    records = _load_records(args)
    done_ids = load_done_ids(args.output) if args.resume else set()
    if done_ids:records = [record for record in records if record.sample_id not i
n done_ids]

    ofs_index = build_ofs_index(args.ofs_dir) if args.ofs_dir else {}failures_path = args.output.with_suffix(args.output.suffix + ".failur
es.jsonl")
    write_mode = "a" if args.resume else "w"
    n_ok = n_fail = 0with args.output.open(write_mode, encoding="utf-8") as out_f, failure
s_path.open(write_mode, encoding="utf-8") as fail_f:
        for record in tqdm(records, desc="Clean CAD contexts"):result, failure = clean_record_context(record, args, ofs_inde
x)
            if result is not None:out_f.write(json.dumps(result, ensure_ascii=False) +
"\n")
                out_f.flush()
                n_ok += 1
            if failure is not None:fail_f.write(json.dumps(failure, ensure_ascii=False) +
"\n")
                fail_f.flush()
                n_fail += 1print(f"Done. ok={n_ok} failed={n_fail} output={args.output} failures
={failures_path}")


if __name__ == "__main__":()
