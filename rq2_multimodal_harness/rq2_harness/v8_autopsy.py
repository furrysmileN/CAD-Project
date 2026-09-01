"""V8 Cut 1 尸检与 Cut 2 分层抽 20。零 API。"""
from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .common import PROJECT_ROOT, atomic_write_json, read_jsonl, write_jsonl
from .pc_analysis import _write_csv
from .pc_runner import _code_fingerprint, _state_path

SAMPLE_SEED = 20260828
SAMPLE_N = 20
PC_HELPS_IMAGE = 0.10
IMAGE_HELPS_PC = 0.08
BOTH_STILL_BAD = 0.35
ALREADY_GOOD = 0.75
SCALE_BBOX = 0.05
SHAPE_CD = 0.12
VOXEL_LOW = 0.30
JACCARD_LOW = 0.5

CORE_CONDITIONS = ("I1", "P_geom", "I1P_geom")


def _float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_primary_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for key in (
                "joint_quality",
                "common_frame_cd",
                "shape_only_cd",
                "voxel_iou",
                "bbox_scale_log_abs",
                "f1_common",
                "success_rate_repeats",
            ):
                parsed[key] = _float(row.get(key))
            parsed["completed"] = str(row.get("completed", "")).lower() in {"true", "1"}
            parsed["complexity_bin"] = int(row["complexity_bin"]) if str(row.get("complexity_bin", "")).isdigit() else row.get("complexity_bin")
            rows.append(parsed)
        return rows


def index_metrics(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["sample_id"], row["condition"]): row for row in rows}


def classify_regimes(i1p: float, i1: float, p_geom: float) -> dict[str, bool | str]:
    pc_helps = (i1p - i1) >= PC_HELPS_IMAGE
    image_helps = (i1p - p_geom) >= IMAGE_HELPS_PC
    both_bad = i1p < BOTH_STILL_BAD
    already_good = i1p >= ALREADY_GOOD
    labels = []
    if pc_helps:
        labels.append("pc_helps_image")
    if image_helps:
        labels.append("image_helps_pc")
    if both_bad:
        level = "both_still_bad"
    elif already_good:
        level = "already_good"
    else:
        level = "mid"
    if not pc_helps and not image_helps and level == "mid":
        combo = "weak_or_mixed"
    else:
        combo = "+".join(labels) if labels else "weak_or_mixed"
    return {
        "pc_helps_image": pc_helps,
        "image_helps_pc": image_helps,
        "both_still_bad": both_bad,
        "already_good": already_good,
        "jq_level": level,
        "regime": combo,
    }


def _ops(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(plan, dict):
        return []
    ops = plan.get("operations")
    return ops if isinstance(ops, list) else []


def _plan_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    repaired = state.get("repaired_plan")
    if isinstance(repaired, dict):
        return repaired
    parse = state.get("parse") or {}
    plan = parse.get("plan")
    return plan if isinstance(plan, dict) else None


def _load_state(state_root: Path, sample_id: str, condition: str, repeat_id: int) -> dict[str, Any] | None:
    path = _state_path(state_root, sample_id, condition, repeat_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_plans(state_root: Path, sample_id: str, condition: str, repeats: Iterable[int] = (1, 2, 3)) -> dict[str, Any]:
    hole_counts: list[int] = []
    cut_counts: list[int] = []
    op_sets: list[set[str]] = []
    thru_like = 0
    n_loaded = 0
    for repeat_id in repeats:
        state = _load_state(state_root, sample_id, condition, repeat_id)
        if not state:
            continue
        plan = _plan_from_state(state)
        ops = _ops(plan)
        if not ops:
            continue
        n_loaded += 1
        kinds = {str(op.get("op")) for op in ops if isinstance(op, dict) and op.get("op")}
        op_sets.append(kinds)
        holes = [op for op in ops if isinstance(op, dict) and op.get("op") == "hole"]
        cuts = [op for op in ops if isinstance(op, dict) and op.get("combine") == "cut"]
        hole_counts.append(len(holes))
        cut_counts.append(len(cuts))
        for op in holes:
            depth = _float(op.get("depth"))
            if depth is not None and depth >= 0.9:
                thru_like += 1
    union: set[str] = set()
    for item in op_sets:
        union |= item
    return {
        "n_plans": n_loaded,
        "mean_holes": mean(hole_counts) if hole_counts else 0.0,
        "mean_cuts": mean(cut_counts) if cut_counts else 0.0,
        "op_kinds": sorted(union),
        "thru_like_holes": thru_like,
    }


def gt_cadquery_heuristics(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"n_holes": 0, "n_thru": 0, "back_or_cavity_suspect": False, "ok": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    n_holes = len(re.findall(r"\.(?:hole|cboreHole|cskHole)\s*\(", text))
    n_thru = len(re.findall(r"cutThruAll\s*\(", text))
    suspect = bool(
        re.search(r'faces\s*\(\s*["\']<[XYZ]', text)
        or re.search(r"Workplane\s*\(\s*[\"'](?:XZ|YZ)", text)
        or "cutBlind" in text
    )
    return {
        "n_holes": n_holes,
        "n_thru": n_thru,
        "back_or_cavity_suspect": suspect,
        "ok": True,
    }


def evidence_hole_count(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    total = 0
    for block in (payload.get("sections") or {}).values():
        if isinstance(block, dict):
            total += len(block.get("holes") or [])
    hyps = payload.get("hypotheses") or []
    total += sum(1 for item in hyps if isinstance(item, dict) and item.get("type") == "circular_hole")
    return total


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def classify_residual(
    *,
    bbox_scale: float | None,
    shape_cd: float | None,
    common_cd: float | None,
    voxel: float | None,
    i1p_jq: float,
    plan_holes: float,
    gt_holes: int,
    evidence_holes: int,
    gt_thru: int,
    plan_thru_like: int,
    jaccard_i1: float,
    jaccard_p: float,
) -> dict[str, Any]:
    flags: list[str] = []
    scale = bbox_scale is not None and bbox_scale >= SCALE_BBOX and (
        shape_cd is None or common_cd is None or shape_cd < 0.5 * common_cd
    )
    global_shape = (shape_cd is not None and shape_cd >= SHAPE_CD) or (voxel is not None and voxel < VOXEL_LOW)
    hole_like = abs(plan_holes - gt_holes) >= 1.0 or (evidence_holes > 0 and plan_holes < 0.5)
    depth_or_blind = plan_holes >= 0.5 and ((gt_thru > 0) != (plan_thru_like > 0))
    topology = min(jaccard_i1, jaccard_p) < JACCARD_LOW and not scale
    if scale:
        flags.append("scale")
    if global_shape:
        flags.append("global_shape")
    if hole_like:
        flags.append("hole_like")
    if depth_or_blind:
        flags.append("depth_or_blind")
    if topology:
        flags.append("topology_order")
    if not flags and i1p_jq < 0.55:
        flags.append("unresolved")
    primary = flags[0] if flags else "none"
    return {"residual_primary": primary, "residual_flags": flags}


def hamilton_quotas(stratum_sizes: dict[tuple[Any, Any], int], n: int) -> dict[tuple[Any, Any], int]:
    keys = sorted(stratum_sizes, key=lambda item: (str(item[0]), str(item[1])))
    total = sum(stratum_sizes[key] for key in keys)
    if total <= 0 or n <= 0:
        return {key: 0 for key in keys}
    exact = {key: n * stratum_sizes[key] / total for key in keys}
    floors = {key: int(exact[key]) for key in keys}
    rem = n - sum(floors.values())
    remainders = sorted(
        keys,
        key=lambda key: (exact[key] - floors[key], stratum_sizes[key], str(key[0]), str(key[1])),
        reverse=True,
    )
    quotas = dict(floors)
    index = 0
    guard = 0
    while rem > 0 and remainders and guard < n * 20:
        key = remainders[index % len(remainders)]
        if quotas[key] < stratum_sizes[key]:
            quotas[key] += 1
            rem -= 1
        index += 1
        guard += 1
    return quotas


def sample_n20(rows: list[dict[str, Any]], *, seed: int = SAMPLE_SEED, n: int = SAMPLE_N) -> dict[str, Any]:
    by_stratum: dict[tuple[Any, Any], list[str]] = defaultdict(list)
    seen: set[str] = set()
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            continue
        seen.add(sample_id)
        key = (row.get("difficulty"), row.get("complexity_bin"))
        by_stratum[key].append(sample_id)
        meta[sample_id] = {"difficulty": row.get("difficulty"), "complexity_bin": row.get("complexity_bin"), "family": row.get("family")}
    for key in by_stratum:
        by_stratum[key] = sorted(set(by_stratum[key]))
    sizes = {key: len(ids) for key, ids in by_stratum.items()}
    quotas = hamilton_quotas(sizes, n)
    rng = random.Random(seed)
    chosen: list[str] = []
    leftover: list[str] = []
    for key, ids in sorted(by_stratum.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        k = min(int(quotas.get(key) or 0), len(ids))
        pick = rng.sample(ids, k) if k else []
        chosen.extend(pick)
        leftover.extend(item for item in ids if item not in pick)
    leftover_sorted = sorted(leftover)
    while len(chosen) < n and leftover_sorted:
        extra = rng.sample(leftover_sorted, 1)[0]
        leftover_sorted.remove(extra)
        chosen.append(extra)
    chosen = sorted(chosen)[:n]
    return {
        "seed": seed,
        "n": len(chosen),
        "sample_ids": chosen,
        "quotas": {f"{key[0]}|{key[1]}": value for key, value in sorted(quotas.items())},
        "stratum_sizes": {f"{key[0]}|{key[1]}": value for key, value in sorted(sizes.items())},
        "rows": [meta[sample_id] | {"sample_id": sample_id} for sample_id in chosen],
    }


def compare_fingerprint(state_root: Path) -> dict[str, Any]:
    current = _code_fingerprint()
    probe = next(state_root.glob("state/*/I1P_geom/r01.json"), None)
    if probe is None:
        probe = next(state_root.glob("state/*/I1P_geom/r02.json"), None)
    if probe is None or not probe.is_file():
        return {"ok": False, "reason": "missing_v5_state", "match": False}
    previous = json.loads(probe.read_text(encoding="utf-8")).get("code_fingerprint") or {}
    return {
        "ok": True,
        "match": previous.get("sha256") == current.get("sha256"),
        "v5_sha256": previous.get("sha256"),
        "current_sha256": current.get("sha256"),
        "sample_state": str(probe),
    }


def run_autopsy(
    *,
    metrics_path: Path,
    manifest_path: Path,
    state_root: Path,
    evidence_dir: Path,
    evidence_audit_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    metrics = load_primary_metrics(metrics_path)
    by = index_metrics(metrics)
    manifest = {row["sample_id"]: row for row in read_jsonl(manifest_path)}
    audit_rows = {}
    if evidence_audit_path.is_file():
        payload = json.loads(evidence_audit_path.read_text(encoding="utf-8"))
        audit_rows = {row["sample_id"]: row for row in payload.get("rows") or []}
    per_sample: list[dict[str, Any]] = []
    for sample_id, row in sorted(manifest.items()):
        i1 = by.get((sample_id, "I1"))
        p_geom = by.get((sample_id, "P_geom"))
        i1p = by.get((sample_id, "I1P_geom"))
        if i1 is None or p_geom is None or i1p is None:
            continue
        jq_i1 = float(i1.get("joint_quality") or 0.0)
        jq_p = float(p_geom.get("joint_quality") or 0.0)
        jq_i1p = float(i1p.get("joint_quality") or 0.0)
        regimes = classify_regimes(jq_i1p, jq_i1, jq_p)
        i1_plan = summarize_plans(state_root, sample_id, "I1")
        p_plan = summarize_plans(state_root, sample_id, "P_geom")
        i1p_plan = summarize_plans(state_root, sample_id, "I1P_geom")
        gt_path = Path((row.get("gt_code") or {}).get("path") or "")
        if not gt_path.is_absolute():
            gt_path = PROJECT_ROOT / gt_path
        gt = gt_cadquery_heuristics(gt_path if gt_path.parts else None)
        evidence_path = evidence_dir / f"{sample_id}.point_evidence.json"
        n_ev_holes = evidence_hole_count(evidence_path)
        residual = classify_residual(
            bbox_scale=i1p.get("bbox_scale_log_abs"),
            shape_cd=i1p.get("shape_only_cd"),
            common_cd=i1p.get("common_frame_cd"),
            voxel=i1p.get("voxel_iou"),
            i1p_jq=jq_i1p,
            plan_holes=float(i1p_plan["mean_holes"]),
            gt_holes=int(gt["n_holes"]),
            evidence_holes=n_ev_holes,
            gt_thru=int(gt["n_thru"]),
            plan_thru_like=int(i1p_plan["thru_like_holes"]),
            jaccard_i1=_jaccard(i1p_plan["op_kinds"], i1_plan["op_kinds"]),
            jaccard_p=_jaccard(i1p_plan["op_kinds"], p_plan["op_kinds"]),
        )
        section_pass = (audit_rows.get(sample_id) or {}).get("section_pass")
        per_sample.append(
            {
                "sample_id": sample_id,
                "family": row.get("family"),
                "difficulty": row.get("difficulty"),
                "complexity_bin": row.get("complexity_bin"),
                "jq_I1": jq_i1,
                "jq_P_geom": jq_p,
                "jq_I1P_geom": jq_i1p,
                "delta_C2": jq_i1p - jq_i1,
                "delta_C3": jq_i1p - jq_p,
                "bbox_scale_log_abs": i1p.get("bbox_scale_log_abs"),
                "shape_only_cd": i1p.get("shape_only_cd"),
                "common_frame_cd": i1p.get("common_frame_cd"),
                "voxel_iou": i1p.get("voxel_iou"),
                "section_pass": section_pass,
                "gt_holes": gt["n_holes"],
                "gt_thru": gt["n_thru"],
                "evidence_holes": n_ev_holes,
                "plan_holes_I1P": i1p_plan["mean_holes"],
                "back_or_cavity_suspect": gt["back_or_cavity_suspect"],
                "jaccard_I1P_vs_I1": _jaccard(i1p_plan["op_kinds"], i1_plan["op_kinds"]),
                "jaccard_I1P_vs_P": _jaccard(i1p_plan["op_kinds"], p_plan["op_kinds"]),
                **regimes,
                **residual,
            }
        )
    regime_counts = Counter(item["regime"] for item in per_sample)
    level_counts = Counter(item["jq_level"] for item in per_sample)
    residual_counts = Counter(item["residual_primary"] for item in per_sample)
    n_section_fail = sum(1 for item in per_sample if item.get("section_pass") is False)
    n_section_fail_bad = sum(
        1 for item in per_sample if item.get("section_pass") is False and item.get("both_still_bad")
    )
    fingerprint = compare_fingerprint(state_root)
    already_good_n = sum(1 for item in per_sample if item["already_good"])
    go = {
        "proceed_cut2": len(per_sample) >= 90 and already_good_n < 80,
        "already_good_n": already_good_n,
        "n_samples": len(per_sample),
        "fingerprint_match": fingerprint.get("match"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "autopsy_per_sample.csv", per_sample)
    crosstab = {
        "schema_version": "rq2.v8.cut1.autopsy.v1",
        "n": len(per_sample),
        "regime_counts": dict(regime_counts),
        "jq_level_counts": dict(level_counts),
        "n_pc_helps_image": sum(1 for item in per_sample if item["pc_helps_image"]),
        "n_image_helps_pc": sum(1 for item in per_sample if item["image_helps_pc"]),
        "n_bidirectional": sum(1 for item in per_sample if item["pc_helps_image"] and item["image_helps_pc"]),
        "section_fail": n_section_fail,
        "section_fail_and_both_still_bad": n_section_fail_bad,
        "fingerprint": fingerprint,
        "go": go,
        "thresholds": {
            "pc_helps_image": PC_HELPS_IMAGE,
            "image_helps_pc": IMAGE_HELPS_PC,
            "both_still_bad": BOTH_STILL_BAD,
            "already_good": ALREADY_GOOD,
        },
    }
    residual_crosstab = {
        "residual_primary": dict(residual_counts),
        "by_jq_level": {},
    }
    by_level: dict[str, Counter[str]] = defaultdict(Counter)
    for item in per_sample:
        by_level[str(item["jq_level"])][str(item["residual_primary"])] += 1
    residual_crosstab["by_jq_level"] = {key: dict(value) for key, value in by_level.items()}
    atomic_write_json(output_dir / "regime_crosstab.json", crosstab)
    atomic_write_json(output_dir / "residual_crosstab.json", residual_crosstab)
    report = _render_cut1_report(crosstab, residual_crosstab, per_sample)
    (output_dir / "CUT1_AUTOPSY_ZH.md").write_text(report, encoding="utf-8")
    return {"n": len(per_sample), "go": go, "output_dir": str(output_dir)}


def _render_cut1_report(
    crosstab: dict[str, Any], residual: dict[str, Any], rows: list[dict[str, Any]]
) -> str:
    go = crosstab.get("go") or {}
    lines = [
        "# V8 Cut 1 尸检",
        "",
        f"- 零件数：{crosstab.get('n')}",
        f"- 点云补图像（Δ≥{PC_HELPS_IMAGE}）：{crosstab.get('n_pc_helps_image')}",
        f"- 图像补点云（Δ≥{IMAGE_HELPS_PC}）：{crosstab.get('n_image_helps_pc')}",
        f"- 双向（两阈值同时）：{crosstab.get('n_bidirectional')}",
        f"- jq 水平：`{json.dumps(crosstab.get('jq_level_counts'), ensure_ascii=False)}`",
        f"- 体制交叉：`{json.dumps(crosstab.get('regime_counts'), ensure_ascii=False)}`",
        f"- 残差主标签：`{json.dumps(residual.get('residual_primary'), ensure_ascii=False)}`",
        f"- 截面审计失败：{crosstab.get('section_fail')}，其中 both_still_bad：{crosstab.get('section_fail_and_both_still_bad')}",
        f"- V5 fingerprint 一致：{go.get('fingerprint_match')}",
        f"- 进入 Cut 2：{go.get('proceed_cut2')}（already_good={go.get('already_good_n')}）",
        "",
        "抽样本身不看 jq。Cut 1 不跳过 Cut 2。",
        "",
    ]
    worst = sorted(rows, key=lambda item: float(item.get("jq_I1P_geom") or 0.0))[:8]
    lines.append("## I1P_geom 最低的 8 件")
    lines.append("")
    lines.append("| sample_id | family | jq | regime | residual |")
    lines.append("|---|---|---:|---|---|")
    for item in worst:
        lines.append(
            f"| `{item['sample_id']}` | {item.get('family')} | {item['jq_I1P_geom']:.3f} | "
            f"{item.get('regime')} | {item.get('residual_primary')} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_n20_manifest(
    parent_manifest: Path,
    sample_ids: list[str],
    dest: Path,
) -> list[dict[str, Any]]:
    wanted = set(sample_ids)
    rows = [row for row in read_jsonl(parent_manifest) if row["sample_id"] in wanted]
    rows.sort(key=lambda row: str(row["sample_id"]))
    if len(rows) != len(wanted):
        missing = wanted - {row["sample_id"] for row in rows}
        raise RuntimeError(f"manifest 缺样本: {sorted(missing)}")
    write_jsonl(dest, rows)
    return rows


def patch_preregistration(path: Path, sample_ids: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    block = "```text\nsample_ids:\n" + "".join(f"  - {item}\n" for item in sample_ids) + "```"
    pattern = re.compile(r"```text\nsample_ids: \[\][^\n]*\n```", re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(block, text, count=1)
    else:
        text = text.replace("```text\nsample_ids: []   # TODO Cut 1 后填入 20 个 ID\n```", block)
    path.write_text(text, encoding="utf-8")
