# -*- coding: utf-8 -*-
"""Generate V6 controlled CAD inputs: latent, STEP, images, point cloud, evidence."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.backend import run_episode
from rq2_harness.common import PROJECT_ROOT, atomic_write_json, sha256_file, sha256_json
from rq2_harness.v6_audit import audit_evidence_files, audit_row_payloads
from rq2_harness.v6_conditions import T0_TEXT
from rq2_harness.v6_corruptions import build_p_wrong
from rq2_harness.v6_evidence_builder import attach_primary_critical, build_p_comp
from rq2_harness.v6_fact_masks import build_p_repeat
from rq2_harness.v6_latent_generator import generate_split, parameter_signature
from rq2_harness.v6_manifest import write_manifest
from rq2_harness.v6_render_inputs import render_step_views
from rq2_harness.v6_sample_pointcloud import sample_pointcloud

BACKEND = {
    "episode_version": "v2",
    "root": str(PROJECT_ROOT / "HarnessCAD" / "HarnessCAD"),
    "timeout_sec": 30,
}


def _dump(path: Path, value) -> None:
    atomic_write_json(path, value)


def prepare_split(split: str, n: int, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    specs = generate_split(split, n)
    signatures = [parameter_signature(spec) for spec in specs]
    if len(set(signatures)) != len(signatures):
        raise RuntimeError("split 内部参数组合重复")
    other = "confirm" if split == "pilot" else "pilot"
    other_manifest = output_root / f"manifest_{other}{20 if other == 'pilot' else 100}.jsonl"
    if other_manifest.is_file():
        other_sigs = set()
        for line in other_manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                other_sigs.add(row.get("parameter_signature"))
        overlap = set(signatures) & other_sigs
        if overlap:
            raise RuntimeError(f"pilot/confirm 参数组合重叠: {len(overlap)}")

    (output_root / "inputs").mkdir(parents=True, exist_ok=True)
    t0_path = output_root / "inputs" / "text_t0.txt"
    t0_path.write_text(T0_TEXT + "\n", encoding="utf-8")
    rows = []
    audits = []
    excluded = []
    harness_root = output_root / "targets" / "_harness_runs"
    for spec in specs:
        sample_id = spec["sample_id"]
        try:
            latent_path = output_root / "latent_specs" / f"{sample_id}.json"
            _dump(latent_path, spec)
            episode = run_episode(spec["gt_plan"], BACKEND, run_root=harness_root)
            step_src = episode.get("result_step_path")
            if not step_src:
                raise RuntimeError(f"Harness 未导出 STEP: {episode.get('response', {}).get('status')}")
            target_dir = output_root / "targets" / sample_id
            target_dir.mkdir(parents=True, exist_ok=True)
            step_path = target_dir / "target.step"
            shutil.copy2(step_src, step_path)
            stl_src = Path(episode["run_dir"]) / "result.stl"
            stl_path = target_dir / "target.stl"
            if stl_src.is_file():
                shutil.copy2(stl_src, stl_path)
            input_dir = output_root / "inputs" / sample_id
            image_dir = input_dir / "images"
            views = render_step_views(step_path, image_dir)
            pc = sample_pointcloud(step_path, input_dir / "pointcloud.npy", n_points=4096, seed=42)
            p_comp = attach_primary_critical(build_p_comp(pc["path"]), spec["critical_fact"])
            p_repeat = build_p_repeat(p_comp, spec["critical_fact"])
            p_wrong = build_p_wrong(p_comp, spec["critical_fact"], sample_id=sample_id)
            _dump(input_dir / "p_comp.json", p_comp)
            _dump(input_dir / "p_repeat.json", p_repeat)
            _dump(input_dir / "p_wrong.json", p_wrong)
            evidence_audit = audit_evidence_files(input_dir, spec)
            row = {
                "sample_id": sample_id,
                "family": spec["family"],
                "split": split,
                "difficulty": spec["difficulty"],
                "parameter_signature": parameter_signature(spec),
                "latent_spec": {"path": str(latent_path), "sha256": sha256_file(latent_path)},
                "latent_spec_sha256": sha256_file(latent_path),
                "target": {
                    "step": str(step_path),
                    "stl": str(stl_path) if stl_path.is_file() else None,
                    "step_sha256": sha256_file(step_path),
                },
                "text_t0": {"path": str(t0_path), "sha256": sha256_file(t0_path)},
                "images": {"views": views, "dir": str(image_dir)},
                "image_dir": str(image_dir),
                "pointcloud": pc,
                "inputs": {
                    "p_comp": {"path": str(input_dir / "p_comp.json"), "sha256": sha256_file(input_dir / "p_comp.json")},
                    "p_repeat": {"path": str(input_dir / "p_repeat.json"), "sha256": sha256_file(input_dir / "p_repeat.json")},
                    "p_wrong": {"path": str(input_dir / "p_wrong.json"), "sha256": sha256_file(input_dir / "p_wrong.json")},
                },
                "critical_fact": {
                    "fact_id": spec["critical_fact"]["fact_id"],
                    "category": spec["critical_fact"]["category"],
                    "visibility_in_images": spec["critical_fact"]["visibility_in_images"],
                },
                "evidence_audit": evidence_audit,
                "episode_status": (episode.get("response") or {}).get("status"),
            }
            from rq2_harness.v6_manifest import attach_evidence_payloads

            payload_audits = audit_row_payloads(attach_evidence_payloads(row))
            row["payload_audits"] = [{"condition": item["condition_id"], "ok": item["ok"], "issues": item["issues"]} for item in payload_audits]
            if not evidence_audit["ok"] or not all(item["ok"] for item in payload_audits):
                excluded.append({"sample_id": sample_id, "evidence": evidence_audit, "payload": row["payload_audits"]})
            rows.append(row)
            audits.append({"sample_id": sample_id, **evidence_audit})
        except Exception as exc:
            excluded.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {sample_id}: {exc}")

    n_label = { "pilot": 20, "confirm": 100 }.get(split, n)
    manifest_path = output_root / f"manifest_{split}{n_label}.jsonl"
    write_manifest(manifest_path, rows)
    _dump(output_root / "audits" / f"evidence_manipulation_{split}.json", audits)
    _dump(output_root / "audits" / f"excluded_{split}.json", excluded)
    summary = {
        "split": split,
        "requested": n,
        "written": len(rows),
        "excluded": len(excluded),
        "manifest": str(manifest_path),
        "n_payload_ok": sum(1 for row in rows if all(item["ok"] for item in row.get("payload_audits") or [])),
        "sha256": sha256_json([row["sample_id"] for row in rows]),
    }
    _dump(output_root / "audits" / f"prepare_{split}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("pilot", "confirm"), default="pilot")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "outputs" / "v6_information_complementarity"),
    )
    args = parser.parse_args()
    prepare_split(args.split, args.n, Path(args.output_root))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
