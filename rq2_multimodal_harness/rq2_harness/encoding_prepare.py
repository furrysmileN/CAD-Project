from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from .api_client import APISettings
from .common import (
    atomic_write_json,
    load_config,
    project_path,
    read_jsonl,
    sha256_json,
    write_jsonl,
)
from .encoding_selection import eligible_v2_sample_ids, freeze_selection
from .encoding_t3 import T3_FIELDS, encode_t3, generate_t3_audit_checklist
from .encoding_visual import prepare_visual_encodings


PREPARE_SCHEMA = "rq2.encoding_screen.prepare.v1"


def _find_blender() -> str:
    executable = shutil.which("blender")
    if executable:
        return executable
    root = Path(r"C:\Program Files\Blender Foundation")
    if root.is_dir():
        candidates = sorted(root.rglob("blender.exe"), reverse=True)
        if candidates:
            return str(candidates[0])
    raise FileNotFoundError("未找到 Blender executable")


def _format_t3(after: dict[str, str]) -> str:
    if set(after) != set(T3_FIELDS):
        raise ValueError("T3 缓存字段不完整")
    labels = {
        "object_type": "Part type",
        "overall_shape": "Overall geometry",
        "primary_features": "Primary features",
        "secondary_features": "Secondary features",
        "spatial_relations": "Spatial relations",
        "dimensions_and_units": "Dimensions and units",
        "uncertainties": "Uncertain information",
    }
    return "\n".join(f"{labels[field]}: {after[field]}" for field in T3_FIELDS)


def _text_entry(text: str, version: str, source_hash: str | None = None) -> dict[str, Any]:
    return {
        "text": text.strip(),
        "version": version,
        "source_sha256": source_hash,
        "sha256": sha256_json({"text": text.strip(), "version": version}),
    }


def _load_or_select(
    config: dict[str, Any],
    output_dir: Path,
    *,
    force: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = output_dir / "selection_manifest.jsonl"
    summary_path = output_dir / "selection_summary.json"
    if manifest_path.is_file() and summary_path.is_file() and not force:
        return (
            list(read_jsonl(manifest_path)),
            json.loads(summary_path.read_text(encoding="utf-8")),
        )
    selection = config["selection"]
    source_manifest = project_path(selection["source_manifest"])
    expressivity_path = project_path(selection["expressivity_audit"])
    rows = list(read_jsonl(source_manifest))
    expressivity = json.loads(expressivity_path.read_text(encoding="utf-8"))
    quotas = {
        str(key): int(value)
        for key, value in selection["difficulty_quotas"].items()
    }
    selected, summary = freeze_selection(
        rows,
        expressivity,
        seed=int(config["seed"]),
        difficulty_quotas=quotas,
    )
    if len(selected) != int(config["n"]):
        raise RuntimeError(f"冻结样本数应为 {config['n']}，实际 {len(selected)}")
    summary["eligible_ids_sha256"] = sha256_json(
        sorted(eligible_v2_sample_ids(expressivity))
    )
    write_jsonl(manifest_path, selected)
    atomic_write_json(summary_path, summary)
    return selected, summary


def _t3_settings(config: dict[str, Any]) -> APISettings:
    raw = config["modalities"]["text"]["encodings"]["T3"]["transform_api"]
    return APISettings.from_config(raw)


def prepare_encoding_screen(
    config: dict[str, Any],
    *,
    selection_only: bool = False,
    convert_t3: bool = False,
    prepare_visuals: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    selected, selection_summary = _load_or_select(
        config,
        output_dir,
        force=force,
    )
    if selection_only:
        report = {
            "schema_version": PREPARE_SCHEMA,
            "status": "selection_ready",
            "selected": len(selected),
            "selection": selection_summary,
        }
        atomic_write_json(output_dir / "prepare_summary.json", report)
        return report

    t3_dir = output_dir / "assets" / "t3"
    visual_dir = output_dir / "assets" / "visual"
    prepared_dir = output_dir / "prepared_samples"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    visual_config = config["modalities"]["point_cloud"]
    views = tuple(visual_config["cameras"])
    resolution = int(visual_config["resolution"])
    padding = float(visual_config["padding"])
    blender_executable = _find_blender() if prepare_visuals else None
    blender_script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "blender_encoding_render.py"
    )
    t3_settings = _t3_settings(config)
    t3_records: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    for source_row in selected:
        sample_id = str(source_row["sample_id"])
        prepared_path = prepared_dir / f"{sample_id}.json"
        if prepared_path.is_file() and not force:
            cached = json.loads(prepared_path.read_text(encoding="utf-8"))
            final_rows.append(cached)
            t3_record_path = t3_dir / f"{sample_id}__t3.json"
            if t3_record_path.is_file():
                t3_records.append(
                    json.loads(t3_record_path.read_text(encoding="utf-8"))
                )
            continue

        row = deepcopy(source_row)
        texts = row.get("text") or {}
        l1 = str(texts.get("L1") or "").strip()
        l3 = str(texts.get("L3") or "").strip()
        if not l1 or not l3:
            raise ValueError(f"{sample_id} 缺少 L1/L3")
        text_encodings = {
            "T1": _text_entry(l1, "rq2.text.l1.v1"),
            "T2": _text_entry(l3, "rq2.text.l3.v1"),
        }
        t3_path = t3_dir / f"{sample_id}__t3.json"
        if convert_t3:
            t3_record = encode_t3(
                l3,
                t3_dir,
                t3_settings,
                source_id=sample_id,
                force=force,
            )
        elif t3_path.is_file():
            t3_record = json.loads(t3_path.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(
                f"缺少 {sample_id} 的 T3 缓存；请使用 --convert-t3"
            )
        if not t3_record["number_unit_preservation"]["ok"]:
            raise RuntimeError(f"{sample_id} T3 新增了数字或单位")
        t3_records.append(t3_record)
        text_encodings["T3"] = _text_entry(
            _format_t3(t3_record["after"]),
            "rq2.text.l3_transform.v1",
            source_hash=t3_record["source_sha256"],
        )

        if not prepare_visuals:
            raise ValueError("完整 manifest 需要 --prepare-visuals")
        step_path = Path(row["step"]["path"])
        obj_path = step_path.with_suffix(".obj")
        point_path = Path(row["point_cloud"]["path"])
        for required in (step_path, obj_path, point_path):
            if not required.is_file() or required.stat().st_size == 0:
                raise FileNotFoundError(f"缺少视觉源资产: {required}")
        visual = prepare_visual_encodings(
            cache_dir=visual_dir,
            pointcloud_path=point_path,
            obj_path=obj_path,
            step_path=step_path,
            encodings=("P1", "P2", "P3", "I1", "I2", "I3"),
            views=views,
            resolution=resolution,
            padding=padding,
            blender_executable=blender_executable,
            blender_script=blender_script,
            force=force,
        )
        minimum_iou = float(
            config["modalities"].get("min_cross_modal_bbox_iou", 0.45)
        )
        bad_iou = [
            item
            for item in visual["image_point_bbox_iou"]
            if float(item["bbox_iou"]) < minimum_iou
        ]
        qc_rows.append(
            {
                "sample_id": sample_id,
                "bundle_sha256": visual["bundle_sha256"],
                "minimum_bbox_iou": min(
                    float(item["bbox_iou"])
                    for item in visual["image_point_bbox_iou"]
                ),
                "threshold": minimum_iou,
                "passed": not bad_iou,
                "failures": bad_iou,
            }
        )
        if bad_iou:
            raise RuntimeError(
                f"{sample_id} 有 {len(bad_iou)} 个跨模态 bbox IoU 低于 {minimum_iou}"
            )

        row["schema_version"] = "rq2.encoding_screen.sample_manifest.v1"
        row["text_encodings"] = text_encodings
        row["render_encodings"] = {
            key: visual["encodings"][key] for key in ("I1", "I2", "I3")
        }
        row["point_encodings"] = {
            key: visual["encodings"][key] for key in ("P1", "P2", "P3")
        }
        row["encoding_input_sha256"] = sha256_json(
            {
                "sample_id": sample_id,
                "text": {
                    key: value["sha256"] for key, value in text_encodings.items()
                },
                "render": {
                    key: value["cache_key"]
                    for key, value in row["render_encodings"].items()
                },
                "point": {
                    key: value["cache_key"]
                    for key, value in row["point_encodings"].items()
                },
                "step": row["step"]["sha256"],
            }
        )
        atomic_write_json(prepared_path, row)
        final_rows.append(row)

    audit_path = output_dir / "t3_audit.json"
    generate_t3_audit_checklist(
        t3_records,
        audit_path,
        sample_count=min(12, len(t3_records)),
        seed=int(config["seed"]),
    )
    write_jsonl(output_dir / "sample_manifest.jsonl", final_rows)
    atomic_write_json(
        output_dir / "visual_qc.json",
        {
            "schema_version": "rq2.encoding_screen.visual_qc.v1",
            "threshold": config["modalities"].get(
                "min_cross_modal_bbox_iou",
                0.45,
            ),
            "samples": qc_rows,
            "passed": all(item["passed"] for item in qc_rows),
        },
    )
    report = {
        "schema_version": PREPARE_SCHEMA,
        "status": "ready",
        "selected": len(selected),
        "prepared": len(final_rows),
        "visual_images": len(final_rows) * 6 * 4,
        "t3_records": len(t3_records),
        "t3_api_attempts": sum(
            int((record.get("usage") or {}).get("attempt") or 1)
            for record in t3_records
        ),
        "selection": selection_summary,
        "sample_manifest_sha256": sha256_json(
            [row["encoding_input_sha256"] for row in final_rows]
        ),
    }
    atomic_write_json(output_dir / "prepare_summary.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="准备20×63编码筛选资产")
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "encoding_screen_n20.yaml"
        ),
    )
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--convert-t3", action="store_true")
    parser.add_argument("--prepare-visuals", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    report = prepare_encoding_screen(
        load_config(args.config),
        selection_only=args.selection_only,
        convert_t3=args.convert_t3,
        prepare_visuals=args.prepare_visuals,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
