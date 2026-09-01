"""Cut 0：从 BenchCAD 1000 池分层抽 40，排除已用清单。"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import PROJECT_ROOT, atomic_write_json, project_path, read_jsonl, sha256_file, write_jsonl
from .hvc_ops import PRIORITY_FAMILIES, classify_code, coverage_report
from .prepare import assign_complexity_bins, code_complexity
from .v8_autopsy import hamilton_quotas

SAMPLE_SEED = 20260829
SAMPLE_N = 40
ADVANCED_N = 20
VIEW_IDS = ("view_0", "view_2", "view_4", "view_6")
EXCLUDE_MANIFESTS = (
    "experiments/rq2_multimodal_harness/outputs/v5_complementarity/manifest_new100.jsonl",
    "experiments/rq2_multimodal_harness/outputs/v8_residual_complementarity/manifest_n20.jsonl",
    "experiments/rq2_multimodal_harness/outputs/pilot_v2/manifest.jsonl",
    "experiments/rq2_multimodal_harness/outputs/encoding_screen_n20/sample_manifest.jsonl",
)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_exclude_ids() -> set[str]:
    exclude: set[str] = set()
    for rel in EXCLUDE_MANIFESTS:
        path = project_path(rel)
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            exclude.add(str(row.get("sample_id") or row.get("stem") or ""))
    exclude.discard("")
    return exclude


def _load_texts() -> dict[str, dict[str, str]]:
    path = project_path("processed/text_descriptions/benchcad/code_gen_hdv3.jsonl")
    texts: dict[str, dict[str, str]] = defaultdict(dict)
    if not path.is_file():
        return {}
    for row in read_jsonl(path):
        if str(row.get("lang") or "en") != "en":
            continue
        stem = str(row.get("stem") or "")
        level = str(row.get("level") or "")
        if stem and level in {"L1", "L3"}:
            texts[stem][level] = str(row.get("text") or "")
    return texts


def load_pool_candidates() -> list[dict[str, Any]]:
    from datasets import load_from_disk

    meta_path = project_path("processed/point_clouds/benchcad/pointcloud_metadata.jsonl")
    aligned = [
        row
        for row in read_jsonl(meta_path)
        if row.get("status") == "success" and int(row.get("sample_idx", 9999)) < 1000
    ]
    dataset = load_from_disk(str(project_path("data/benchcad/code_gen/dataset")))
    image_columns = [name for name in dataset.column_names if name.endswith("_png")]
    metadata = dataset.remove_columns(image_columns)
    exclude = load_exclude_ids()
    texts = _load_texts()
    models_dir = project_path("processed/models/benchcad")
    renders_root = project_path("processed/renders/benchcad")
    pc_root = project_path("processed/point_clouds/benchcad/2048")
    candidates: list[dict[str, Any]] = []
    for item in aligned:
        stem = str(item["stem"])
        if stem in exclude:
            continue
        sample = metadata[int(item["sample_idx"])]
        if str(sample.get("stem")) != stem:
            continue
        code = sample.get("code", "")
        if isinstance(code, bytes):
            code = code.decode("utf-8", errors="strict")
        code = str(code)
        complexity = code_complexity(code)
        if complexity < 0 or not code.strip():
            continue
        step_path = models_dir / f"{stem}.step"
        point_path = pc_root / f"{stem}.npy"
        image_paths = [renders_root / stem / f"{view}.png" for view in VIEW_IDS]
        if any(not path.is_file() or path.stat().st_size == 0 for path in [step_path, point_path, *image_paths]):
            continue
        sample_texts = texts.get(stem, {})
        if set(sample_texts) != {"L1", "L3"}:
            continue
        family = str(sample.get("family") or item.get("family") or "unknown")
        classified = classify_code(code, family)
        candidates.append(
            {
                "sample_idx": int(item["sample_idx"]),
                "stem": stem,
                "sample_id": stem,
                "family": family,
                "difficulty": str(sample.get("difficulty") or "unknown"),
                "complexity": complexity,
                "gt_code": code,
                "step_path": step_path,
                "point_path": point_path,
                "image_paths": image_paths,
                "texts": sample_texts,
                **classified,
            }
        )
    assign_complexity_bins(candidates, 3)
    return candidates


def _sample_stratum(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    by_stratum: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[(row.get("difficulty"), row.get("complexity_bin"))].append(row)
    for key, bucket in by_stratum.items():
        bucket.sort(key=lambda item: str(item["stem"]))
    sizes = {key: len(bucket) for key, bucket in by_stratum.items()}
    quotas = hamilton_quotas(sizes, n)
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    leftover: list[dict[str, Any]] = []
    for key, bucket in sorted(by_stratum.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        k = min(int(quotas.get(key) or 0), len(bucket))
        pick = rng.sample(bucket, k) if k else []
        chosen.extend(pick)
        leftover.extend(item for item in bucket if item not in pick)
    leftover.sort(key=lambda item: str(item["stem"]))
    while len(chosen) < n and leftover:
        extra = rng.sample(leftover, 1)[0]
        leftover.remove(extra)
        chosen.append(extra)
    chosen.sort(key=lambda item: str(item["stem"]))
    return chosen[:n]


def sample_n40(candidates: list[dict[str, Any]], *, seed: int = SAMPLE_SEED, n: int = SAMPLE_N) -> dict[str, Any]:
    advanced = [row for row in candidates if row["stratum"] == "advanced"]
    standard = [row for row in candidates if row["stratum"] == "standard"]
    forced: list[dict[str, Any]] = []
    used_families: set[str] = set()
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in advanced:
        by_family[str(row["family"])].append(row)
    rng = random.Random(seed)
    for family in PRIORITY_FAMILIES:
        bucket = sorted(by_family.get(family, []), key=lambda item: str(item["stem"]))
        if not bucket:
            continue
        pick = rng.choice(bucket)
        forced.append(pick)
        used_families.add(family)
    forced_ids = {row["stem"] for row in forced}
    remaining_adv = [row for row in advanced if row["stem"] not in forced_ids]
    advanced_quota = n // 2 if n != SAMPLE_N else ADVANCED_N
    need_adv = max(0, advanced_quota - len(forced))
    extra_adv = _sample_stratum(remaining_adv, need_adv, seed + 1)
    standard_pick = _sample_stratum(standard, n - advanced_quota, seed + 2)
    chosen = forced + extra_adv + standard_pick
    chosen.sort(key=lambda item: str(item["stem"]))
    if len(chosen) < n:
        leftover = [row for row in candidates if row["stem"] not in {item["stem"] for item in chosen}]
        leftover.sort(key=lambda item: str(item["stem"]))
        chosen.extend(leftover[: n - len(chosen)])
    chosen = chosen[:n]
    return {
        "seed": seed,
        "n": len(chosen),
        "sample_ids": [row["stem"] for row in chosen],
        "n_advanced": sum(1 for row in chosen if row["stratum"] == "advanced"),
        "n_standard": sum(1 for row in chosen if row["stratum"] == "standard"),
        "priority_families_present": sorted(used_families),
        "rows": chosen,
    }


def build_manifest_row(candidate: dict[str, Any], gt_dir: Path) -> dict[str, Any]:
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_code_path = gt_dir / f"{candidate['stem']}.py"
    text = candidate["gt_code"].rstrip() + "\n"
    gt_code_path.write_text(text, encoding="utf-8", newline="\n")
    step_path = Path(candidate["step_path"])
    point_path = Path(candidate["point_path"])
    import numpy as np

    row = {
        "schema_version": "rq2.manifest.v1",
        "sample_id": candidate["stem"],
        "sample_idx": candidate["sample_idx"],
        "family": candidate["family"],
        "difficulty": candidate["difficulty"],
        "complexity": candidate["complexity"],
        "complexity_bin": candidate["complexity_bin"],
        "stratum": candidate["stratum"],
        "advanced_ops": candidate["advanced_ops"],
        "v3_expressible": candidate["v3_expressible"],
        "text": {"L1": candidate["texts"]["L1"], "L3": candidate["texts"]["L3"]},
        "text_encodings": {
            "T1": {"text": candidate["texts"]["L1"], "version": "rq2.text.l1.v1"},
            "T2": {"text": candidate["texts"]["L3"], "version": "rq2.text.l3.v1"},
        },
        "step": {"path": str(step_path.resolve()), "sha256": sha256_file(step_path)},
        "gt_code": {"path": str(gt_code_path.resolve()), "sha256": _hash_text(text)},
        "images": [
            {"view": view, "path": str(path.resolve()), "sha256": sha256_file(path)}
            for view, path in zip(VIEW_IDS, candidate["image_paths"])
        ],
        "point_cloud": {
            "path": str(point_path.resolve()),
            "sha256": sha256_file(point_path),
            "shape": [int(value) for value in np.load(point_path, mmap_mode="r").shape],
        },
    }
    row["input_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "text": row["text"],
                "step": row["step"]["sha256"],
                "gt_code": row["gt_code"]["sha256"],
                "images": [item["sha256"] for item in row["images"]],
                "point_cloud": row["point_cloud"]["sha256"],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return row


def run_cut0(output_dir: Path | None = None) -> dict[str, Any]:
    dest = output_dir or project_path("experiments/rq2_multimodal_harness/outputs/harness_vs_cadrille")
    dest.mkdir(parents=True, exist_ok=True)
    candidates = load_pool_candidates()
    sampled = sample_n40(candidates)
    rows = [build_manifest_row(item, dest / "gt_code") for item in sampled["rows"]]
    write_jsonl(dest / "manifest_n40.jsonl", rows)
    overlap = {
        "n_new": len(rows),
        "exclude_manifests": list(EXCLUDE_MANIFESTS),
        "overlap": [],
        "overlap_empty": True,
    }
    exclude = load_exclude_ids()
    hit = sorted({row["sample_id"] for row in rows} & exclude)
    overlap["overlap"] = hit
    overlap["overlap_empty"] = not hit
    atomic_write_json(dest / "overlap_audit.json", overlap)
    slim_rows = [
        {
            "sample_id": row["sample_id"],
            "family": row["family"],
            "difficulty": row["difficulty"],
            "complexity_bin": row["complexity_bin"],
            "stratum": row["stratum"],
            "advanced_ops": row["advanced_ops"],
            "v3_expressible": row["v3_expressible"],
        }
        for row in rows
    ]
    coverage = coverage_report(slim_rows)
    prereg = dest / "preregistration.md"
    if prereg.is_file():
        text = prereg.read_text(encoding="utf-8")
        block = "```text\nsample_ids:\n" + "".join(f"  - {row['sample_id']}\n" for row in rows) + "```"
        text = text.replace("```text\nsample_ids: []\n```", block)
        prereg.write_text(text, encoding="utf-8")
    atomic_write_json(
        dest / "sample_n40.json",
        {
            "seed": SAMPLE_SEED,
            "n": len(rows),
            "n_advanced": sampled["n_advanced"],
            "n_standard": sampled["n_standard"],
            "priority_families_present": sampled["priority_families_present"],
            "sample_ids": [row["sample_id"] for row in rows],
            "rows": slim_rows,
            "coverage": coverage,
            "overlap": overlap,
            "pool_n": len(candidates),
            "project_root": str(PROJECT_ROOT),
        },
    )
    return {
        "n": len(rows),
        "overlap_empty": overlap["overlap_empty"],
        "coverage": coverage,
        "output_dir": str(dest),
    }
