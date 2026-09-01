"""Cut 2 描述性分析与 Cut 3 门控。"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .common import atomic_write_json, project_path, read_jsonl
from .hvc_runner import ARMS, _state_path
from .hvc_sample import SAMPLE_SEED, load_exclude_ids, load_pool_candidates, sample_n40


def _load_states(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in output_dir.glob("state/*/*.json"):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def analyze_hvc(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = project_path(config["paths"]["output_root"])
    manifest = {row["sample_id"]: row for row in read_jsonl(project_path(config["paths"]["manifest"]))}
    states = _load_states(output_dir)
    by_arm: dict[str, list[float]] = defaultdict(list)
    by_stratum: dict[tuple[str, str], list[float]] = defaultdict(list)
    success: dict[str, list[int]] = defaultdict(list)
    pending_gpu = 0
    for state in states:
        arm = str(state.get("arm"))
        jq = float(((state.get("geometry") or {}).get("joint_quality")) or 0.0)
        ok = bool((state.get("geometry") or {}).get("success"))
        stratum = str(state.get("stratum") or manifest.get(state.get("sample_id"), {}).get("stratum") or "unknown")
        if state.get("status") == "pending_gpu":
            pending_gpu += 1
            continue
        if state.get("status") in {"dry_run"}:
            continue
        by_arm[arm].append(jq)
        by_stratum[(arm, stratum)].append(jq)
        success[arm].append(1 if ok else 0)
    means = {arm: mean(values) if values else None for arm, values in by_arm.items()}
    harness = means.get("qwen_harness")
    raw = means.get("qwen_raw")
    cadrille = means.get("cadrille_rl")
    delta_hr = None if harness is None or raw is None else harness - raw
    delta_hc = None if harness is None or cadrille is None else harness - cadrille
    gate = float((config.get("cut3") or {}).get("jq_delta_gate") or 0.03)
    suc_h = mean(success["qwen_harness"]) if success["qwen_harness"] else 0.0
    suc_r = mean(success["qwen_raw"]) if success["qwen_raw"] else 0.0
    proceed = bool(delta_hr is not None and (delta_hr >= gate or suc_h > suc_r + 1e-9))
    if pending_gpu and not any(state.get("arm") == "cadrille_rl" and state.get("status") == "completed" for state in states):
        cadrille_note = "cadrille_pending_gpu"
    else:
        cadrille_note = "scored"
    report = {
        "n_states": len(states),
        "means": means,
        "success_rate": {arm: mean(values) if values else None for arm, values in success.items()},
        "by_stratum": {f"{arm}|{stratum}": mean(values) for (arm, stratum), values in sorted(by_stratum.items()) if values},
        "delta_harness_raw": delta_hr,
        "delta_harness_cadrille": delta_hc,
        "pending_gpu": pending_gpu,
        "cadrille_note": cadrille_note,
        "gates": {
            "jq_delta_gate": gate,
            "proceed_cut3": proceed,
            "expand_100": proceed,
            "second_qwen": proceed,
            "cadrille_pc_appendix": proceed and pending_gpu == 0,
        },
    }
    dest = output_dir / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest / "cut2_descriptive.json", report)
    lines = [
        "# Harness vs CADrille Cut 2",
        "",
        f"- 状态条数：{report['n_states']}",
        f"- 均值 jq：`{json.dumps(means, ensure_ascii=False)}`",
        f"- 成功率：`{json.dumps(report['success_rate'], ensure_ascii=False)}`",
        f"- Harness − raw：{delta_hr}",
        f"- Harness − CADrille：{delta_hc}",
        f"- CADrille：{cadrille_note}（pending_gpu={pending_gpu}）",
        f"- 进入 Cut 3：{proceed}",
        "",
    ]
    (dest / "CUT2_ANALYSIS_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def prepare_cut3_expand100(config: dict[str, Any]) -> dict[str, Any]:
    analysis = analyze_hvc(config)
    dest = project_path("experiments/rq2_multimodal_harness/outputs/harness_vs_cadrille")
    if not analysis["gates"]["expand_100"]:
        atomic_write_json(dest / "cut3_status.json", {"status": "blocked", "reason": "cut2_gate_failed", "gates": analysis["gates"]})
        return {"status": "blocked", "gates": analysis["gates"]}
    exclude = load_exclude_ids()
    existing = {row["sample_id"] for row in read_jsonl(dest / "manifest_n40.jsonl")}
    exclude |= existing
    candidates = [row for row in load_pool_candidates() if row["stem"] not in exclude]
    sampled = sample_n40(candidates, seed=SAMPLE_SEED + 100, n=60)
    atomic_write_json(
        dest / "cut3_expand60.json",
        {"n": sampled["n"], "sample_ids": sampled["sample_ids"], "n_advanced": sampled["n_advanced"]},
    )
    atomic_write_json(dest / "cut3_status.json", {"status": "ready", "expand_ids": sampled["sample_ids"], "second_model": (config.get("cut3") or {}).get("second_model")})
    return {"status": "ready", "n_extra": sampled["n"]}
