# -*- coding: utf-8 -*-
"""Build V6b minimal-pair STEP / images / point clouds / evidence. No live API."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.backend import run_episode
from rq2_harness.common import PROJECT_ROOT, atomic_write_json, sha256_file, sha256_json
from rq2_harness.prompting import validate_plan
from rq2_harness.v6_carriers import list_carriers
from rq2_harness.v6_conditions import T0_TEXT
from rq2_harness.v6_evidence_builder import attach_primary_critical, build_p_comp
from rq2_harness.v6_fact_masks import build_p_repeat
from rq2_harness.v6_feature_scorer import DEFAULT_TOLERANCE, score_critical_fact
from rq2_harness.v6_manifest import write_manifest
from rq2_harness.v6_render_inputs import render_step_views
from rq2_harness.v6_sample_pointcloud import sample_pointcloud
from rq2_harness.v6b_pair_generator import generate_pairs, operations_differ_only_in_critical

BACKEND = {
    "episode_version": "v2",
    "root": str(PROJECT_ROOT / "HarnessCAD" / "HarnessCAD"),
    "timeout_sec": 30,
}


def _dump(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _meas_ok(measured: Any, gt: Any, category: str) -> bool:
    if measured is None or gt is None:
        return False
    if category == "through_vs_blind":
        return str(measured) == str(gt)
    if category == "hidden_presence":
        return bool(measured) == bool(gt)
    try:
        return abs(float(measured) - float(gt)) <= float(DEFAULT_TOLERANCE.get(category, 0.04))
    except (TypeError, ValueError):
        return False


def _primary(p_comp: dict[str, Any]) -> dict[str, Any] | None:
    return next((item for item in (p_comp.get("cad_facts") or []) if item.get("role") == "primary_critical"), None)


def _image_l1(path_a: Path, path_b: Path) -> float:
    a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        return 1.0
    return float(np.mean(np.abs(a - b)) / 255.0)


def _export_variant(spec: dict[str, Any], output_root: Path, harness_root: Path) -> dict[str, Any]:
    sample_id = spec["sample_id"]
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
    input_dir.mkdir(parents=True, exist_ok=True)
    views = render_step_views(step_path, input_dir / "images")
    pc = sample_pointcloud(step_path, input_dir / "pointcloud.npy", n_points=4096, seed=42)
    p_comp = attach_primary_critical(build_p_comp(pc["path"]), spec["critical_fact"])
    _dump(input_dir / "p_comp.json", p_comp)
    return {
        "sample_id": sample_id,
        "step": str(step_path),
        "stl": str(stl_path) if stl_path.is_file() else None,
        "views": views,
        "image_dir": str(input_dir / "images"),
        "pointcloud": pc,
        "p_comp": p_comp,
        "p_comp_path": str(input_dir / "p_comp.json"),
        "episode_status": (episode.get("response") or {}).get("status"),
        "schema_issues": validate_plan(spec["gt_plan"], plan_version="v2"),
    }


def prepare_pairs(n: int, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "inputs").mkdir(parents=True, exist_ok=True)
    t0_path = output_root / "inputs" / "text_t0.txt"
    t0_path.write_text(T0_TEXT + "\n", encoding="utf-8")
    harness_root = output_root / "targets" / "_harness_runs"
    pairs = generate_pairs(n)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        try:
            if not operations_differ_only_in_critical(pair):
                raise RuntimeError("A/B 不只差关键事实")
            built_a = _export_variant(pair["spec_a"], output_root, harness_root)
            built_b = _export_variant(pair["spec_b"], output_root, harness_root)
            critical_a = pair["spec_a"]["critical_fact"]
            critical_b = pair["spec_b"]["critical_fact"]
            p_a = built_a["p_comp"]
            p_b = built_b["p_comp"]
            p_repeat = build_p_repeat(p_a, critical_a)
            pair_dir = output_root / "inputs" / pair_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            _dump(pair_dir / "p_full.json", p_a)
            _dump(pair_dir / "p_repeat.json", p_repeat)
            _dump(pair_dir / "p_counterfactual.json", p_b)
            shutil.copy2(pair_dir / "p_full.json", pair_dir / "p_comp.json")
            primary_a = _primary(p_a)
            primary_b = _primary(p_b)
            meas_a = None if primary_a is None else primary_a.get("value")
            meas_b = None if primary_b is None else primary_b.get("value")
            category = str(critical_a.get("category") or "")
            gt_a = critical_a.get("value")
            gt_b = critical_b.get("value")
            score_a = score_critical_fact(pair["spec_a"]["gt_plan"], pair["spec_a"])
            score_ba = score_critical_fact(pair["spec_b"]["gt_plan"], pair["spec_a"])
            leakage = {
                item["view"]: round(_image_l1(Path(item["path"]), Path(other["path"])), 4)
                for item, other in zip(built_a["views"], built_b["views"])
            }
            meas_ok_a = _meas_ok(meas_a, gt_a, category)
            meas_ok_b = _meas_ok(meas_b, gt_b, category)
            collision = _meas_ok(meas_b, gt_a, category)
            repeat_hits = list_carriers(p_repeat, critical_a)
            eligible = bool(
                meas_ok_a
                and meas_ok_b
                and not collision
                and not repeat_hits
                and not built_a["schema_issues"]
                and not built_b["schema_issues"]
                and score_a.get("exact")
                and not score_ba.get("exact")
            )
            audit = {
                "pair_id": pair_id,
                "kind": pair["kind"],
                "family": pair["family"],
                "gt_a": gt_a,
                "gt_b": gt_b,
                "meas_a": meas_a,
                "meas_b": meas_b,
                "meas_ok_a": meas_ok_a,
                "meas_ok_b": meas_ok_b,
                "collision": collision,
                "repeat_residual": repeat_hits,
                "oracle_a_exact": bool(score_a.get("exact")),
                "oracle_b_changes_a": bool(score_a.get("exact")) and not bool(score_ba.get("exact")),
                "schema_ok": not built_a["schema_issues"] and not built_b["schema_issues"],
                "image_l1": leakage,
                "eligible": eligible,
            }
            row = {
                "sample_id": pair["spec_a"]["sample_id"],
                "pair_id": pair_id,
                "kind": pair["kind"],
                "family": pair["family"],
                "split": "probe",
                "variant": "A",
                "mate_sample_id": pair["spec_b"]["sample_id"],
                "eligible": eligible,
                "critical_fact": {
                    "fact_id": critical_a["fact_id"],
                    "category": category,
                    "visibility_in_images": critical_a.get("visibility_in_images"),
                },
                "text_t0": {"path": str(t0_path), "sha256": sha256_file(t0_path)},
                "images": {"views": built_a["views"], "dir": built_a["image_dir"]},
                "image_dir": built_a["image_dir"],
                "pointcloud": built_a["pointcloud"],
                "mate_pointcloud": built_b["pointcloud"],
                "inputs": {
                    "p_comp": {"path": str(pair_dir / "p_comp.json"), "sha256": sha256_file(pair_dir / "p_comp.json")},
                    "p_repeat": {"path": str(pair_dir / "p_repeat.json"), "sha256": sha256_file(pair_dir / "p_repeat.json")},
                    "p_counterfactual": {
                        "path": str(pair_dir / "p_counterfactual.json"),
                        "sha256": sha256_file(pair_dir / "p_counterfactual.json"),
                    },
                },
                "target": {"step": built_a["step"], "stl": built_a["stl"]},
                "mate_target": {"step": built_b["step"], "stl": built_b["stl"]},
                "offline_audit": audit,
            }
            rows.append(row)
            audits.append(audit)
            print(json.dumps({"pair_id": pair_id, "kind": pair["kind"], "eligible": eligible, "meas_a": meas_a, "meas_b": meas_b}, ensure_ascii=False))
        except Exception as exc:
            audits.append({"pair_id": pair_id, "kind": pair["kind"], "error": f"{type(exc).__name__}: {exc}", "eligible": False})
            print(f"FAIL {pair_id}: {exc}")

    n_eligible = sum(1 for row in rows if row.get("eligible"))
    by_kind: dict[str, int] = {}
    for row in rows:
        if row.get("eligible"):
            by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
    summary = {
        "n_requested": n,
        "n_written": len(rows),
        "n_eligible": n_eligible,
        "eligible_by_kind": by_kind,
        "kinds_covered": sorted(by_kind),
        "pass_probe_gate": n_eligible >= 8 and len(by_kind) >= 4,
        "manifest": str(output_root / "manifest_probe.jsonl"),
    }
    write_manifest(output_root / "manifest_probe.jsonl", rows)
    _dump(output_root / "audits" / "v6b_pairs_offline.json", {"summary": summary, "pairs": audits})
    _dump(output_root / "audits" / "prepare_probe.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "outputs" / "v6_information_complementarity" / "pilot_v2_minimal_pairs"),
    )
    args = parser.parse_args()
    summary = prepare_pairs(args.n, Path(args.output_root))
    return 0 if summary.get("pass_probe_gate") else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
