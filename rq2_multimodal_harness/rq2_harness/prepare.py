from __future__ import annotations

import ast
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import PROJECT_ROOT, atomic_write_json, project_path, read_jsonl, sha256_file, write_jsonl
from .pointcloud import encode_point_cloud


def code_complexity(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return -1
    return sum(isinstance(node, (ast.Call, ast.For, ast.While, ast.If)) for node in ast.walk(tree))


def assign_complexity_bins(rows: list[dict[str, Any]], n_bins: int) -> None:
    values = sorted(row["complexity"] for row in rows)
    if not values:
        return
    cuts = [values[min(len(values) - 1, (len(values) * i) // n_bins)] for i in range(1, n_bins)]
    for row in rows:
        row["complexity_bin"] = sum(row["complexity"] >= cut for cut in cuts)


def deterministic_stratified_sample(
    rows: Iterable[dict[str, Any]],
    n: int,
    seed: int,
    strata: tuple[str, ...] = ("family", "difficulty", "complexity_bin"),
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in strata)].append(row)
    rng = random.Random(seed)
    for key, bucket in buckets.items():
        bucket.sort(key=lambda row: str(row["stem"]))
        random.Random(f"{seed}:{key!r}").shuffle(bucket)
    keys = sorted(buckets, key=lambda key: tuple("" if value is None else str(value) for value in key))
    rng.shuffle(keys)
    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in keys}
    while len(selected) < n:
        progressed = False
        for key in keys:
            offset = offsets[key]
            if offset < len(buckets[key]):
                selected.append(buckets[key][offset])
                offsets[key] += 1
                progressed = True
                if len(selected) == n:
                    break
        if not progressed:
            break
    return selected


def _load_texts(path: Path) -> dict[str, dict[str, str]]:
    texts: dict[str, dict[str, str]] = defaultdict(dict)
    for row in read_jsonl(path):
        if row.get("lang") == "en" and row.get("level") in {"L1", "L3"}:
            texts[str(row["stem"])][str(row["level"])] = str(row.get("text", "")).strip()
    return dict(texts)


def _load_aligned_metadata(path: Path, pool_size: int) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        if row.get("status") == "success" and int(row.get("sample_idx", pool_size)) < pool_size:
            rows.append(row)
    rows.sort(key=lambda row: (int(row["sample_idx"]), str(row["stem"])))
    return rows


def _step_is_valid(path: Path) -> bool:
    try:
        import cadquery as cq

        shape = cq.importers.importStep(str(path)).val()
        return shape is not None and bool(shape.isValid())
    except Exception:
        return False


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prepare(config: dict[str, Any], *, force: bool = False, keep_ids: list[str] | None = None) -> dict[str, Any]:
    from datasets import load_from_disk

    paths = config["paths"]
    modalities = config["modalities"]
    output_dir = project_path(paths["output_dir"])
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.is_file() and not force:
        return {"status": "cached", "manifest": str(manifest_path), "rows": sum(1 for _ in read_jsonl(manifest_path))}

    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    pc_meta_path = project_path(paths["point_cloud_metadata"])
    aligned = _load_aligned_metadata(pc_meta_path, int(config["sampling"]["aligned_pool_size"]))
    text_path = project_path(modalities["text"]["source"])
    texts = _load_texts(text_path)

    dataset_path = project_path(paths["dataset"])
    dataset = load_from_disk(str(dataset_path))
    image_columns = [name for name in dataset.column_names if name.endswith("_png")]
    metadata_dataset = dataset.remove_columns(image_columns)

    candidates: list[dict[str, Any]] = []
    for item in aligned:
        index = int(item["sample_idx"])
        sample = metadata_dataset[index]
        stem = str(item["stem"])
        if str(sample.get("stem")) != stem:
            failures.append({"stem": stem, "stage": "alignment", "error": f"dataset_stem={sample.get('stem')!r}"})
            continue
        code = sample.get("code", "")
        if isinstance(code, bytes):
            code = code.decode("utf-8", errors="strict")
        code = str(code)
        complexity = code_complexity(code)
        if complexity < 0 or not code.strip():
            failures.append({"stem": stem, "stage": "gt_code", "error": "empty_or_syntax_error"})
            continue
        candidates.append(
            {
                "sample_idx": index,
                "stem": stem,
                "family": str(sample.get("family") or item.get("family") or "unknown"),
                "difficulty": str(sample.get("difficulty") or "unknown"),
                "complexity": complexity,
                "gt_code": code,
            }
        )

    assign_complexity_bins(candidates, int(config["sampling"]["complexity_bins"]))
    exclude: set[str] = set()
    for item in config.get("sampling", {}).get("exclude_ids") or []:
        exclude.add(str(item))
    for rel in config.get("sampling", {}).get("exclude_manifests") or []:
        path = project_path(rel)
        if path.is_file():
            exclude.update(str(row.get("sample_id") or row.get("stem") or "") for row in read_jsonl(path))
    exclude_file = config.get("sampling", {}).get("exclude_file")
    if exclude_file:
        extra = project_path(str(exclude_file))
        if extra.is_file():
            exclude.update(
                line.strip()
                for line in extra.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
    if exclude:
        candidates = [row for row in candidates if str(row["stem"]) not in exclude]
    keep_list = [str(item) for item in (keep_ids or []) if str(item) not in exclude]
    keep_set = set(keep_list)
    if keep_list:
        by_stem = {row["stem"]: row for row in candidates}
        missing_keep = [stem for stem in keep_list if stem not in by_stem]
        if missing_keep:
            raise RuntimeError(f"keep_ids 不在候选池: {missing_keep[:8]}")
        rest = [row for row in candidates if row["stem"] not in keep_set]
        ordered_rest = deterministic_stratified_sample(
            rest,
            len(rest),
            int(config["seed"]),
            tuple(config["sampling"]["strata"]),
        )
        ordered = [by_stem[stem] for stem in keep_list] + ordered_rest
    else:
        ordered = deterministic_stratified_sample(
            candidates,
            len(candidates),
            int(config["seed"]),
            tuple(config["sampling"]["strata"]),
        )
    rows: list[dict[str, Any]] = []
    gt_dir = output_dir / "gt_code"
    pc_cache = output_dir / "pointcloud_views"
    models_dir = project_path(paths["models_dir"])
    renders_root = project_path(modalities["images"]["root"])
    pc_root = project_path(modalities["point_cloud"]["root"])
    view_ids = list(modalities["images"]["view_ids"])

    for candidate in ordered:
        if len(rows) >= int(config["n"]):
            break
        stem = candidate["stem"]
        step_path = models_dir / f"{stem}.step"
        point_path = pc_root / f"{stem}.npy"
        image_paths = [renders_root / stem / f"{view}.png" for view in view_ids]
        missing = [str(path) for path in [step_path, point_path, *image_paths] if not path.is_file() or path.stat().st_size == 0]
        sample_texts = texts.get(stem, {})
        if set(sample_texts) != {"L1", "L3"}:
            missing.append(f"text:L1/L3:{sorted(sample_texts)}")
        if missing:
            failures.append({"stem": stem, "stage": "verify", "error": "missing_or_empty", "paths": missing})
            continue
        if not _step_is_valid(step_path):
            failures.append({"stem": stem, "stage": "verify", "error": "invalid_step_shape", "paths": [str(step_path)]})
            continue
        try:
            pc_meta = encode_point_cloud(
                point_path,
                pc_cache,
                views=list(modalities["point_cloud"]["views"]),
                resolution=int(modalities["point_cloud"]["resolution"]),
                padding=float(modalities["point_cloud"]["padding"]),
                encoding_version=str(modalities["point_cloud"]["encoding_version"]),
                force=force,
            )
        except Exception as exc:
            failures.append({"stem": stem, "stage": "point_cloud_encode", "error": str(exc)[:500]})
            continue

        gt_dir.mkdir(parents=True, exist_ok=True)
        gt_code_path = gt_dir / f"{stem}.py"
        gt_code_path.write_text(candidate["gt_code"].rstrip() + "\n", encoding="utf-8", newline="\n")
        row = {
            "schema_version": "rq2.manifest.v1",
            "sample_id": stem,
            "sample_idx": candidate["sample_idx"],
            "family": candidate["family"],
            "difficulty": candidate["difficulty"],
            "complexity": candidate["complexity"],
            "complexity_bin": candidate["complexity_bin"],
            "text": {"L1": sample_texts["L1"], "L3": sample_texts["L3"]},
            "text_encodings": {
                "T1": {"text": sample_texts["L1"], "version": "rq2.text.l1.v1"},
                "T2": {"text": sample_texts["L3"], "version": "rq2.text.l3.v1"},
            },
            "step": {"path": str(step_path.resolve()), "sha256": sha256_file(step_path)},
            "gt_code": {"path": str(gt_code_path.resolve()), "sha256": _hash_text(candidate["gt_code"].rstrip() + "\n")},
            "images": [
                {"view": view, "path": str(path.resolve()), "sha256": sha256_file(path)}
                for view, path in zip(view_ids, image_paths)
            ],
            "point_cloud": {
                "path": str(point_path.resolve()),
                "sha256": pc_meta["source_sha256"],
                "shape": [int(value) for value in __import__("numpy").load(point_path, mmap_mode="r").shape],
                "encoding": pc_meta,
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
        rows.append(row)

    write_jsonl(manifest_path, rows)
    write_jsonl(output_dir / "prepare_failures.jsonl", failures)
    meta = {
        "schema_version": "rq2.prepare_meta.v1",
        "project_root": str(PROJECT_ROOT),
        "seed": int(config["seed"]),
        "requested_n": int(config["n"]),
        "prepared_n": len(rows),
        "aligned_candidates": len(aligned),
        "valid_candidates": len(candidates),
        "failure_count": len(failures),
        "source_hashes": {
            "point_cloud_metadata": sha256_file(pc_meta_path),
            "text_hdv3": sha256_file(text_path),
            "dataset_info": sha256_file(dataset_path / "dataset_info.json"),
        },
    }
    atomic_write_json(output_dir / "prepare_meta.json", meta)
    if len(rows) < int(config["n"]):
        raise RuntimeError(f"仅准备出 {len(rows)}/{config['n']} 个完整样本；详见 prepare_failures.jsonl")
    return {"status": "prepared", "manifest": str(manifest_path), "rows": len(rows), "failures": len(failures)}
