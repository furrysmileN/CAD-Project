from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_DIFFICULTY_QUOTAS = {"easy": 7, "medium": 7, "hard": 6}


def _sample_id(row: Mapping[str, Any]) -> str:
    value = row.get("sample_id")
    if not isinstance(value, str) or not value:
        raise ValueError("manifest 每行都必须有非空 sample_id")
    return value


def eligible_v2_sample_ids(expressivity: Mapping[str, Any]) -> frozenset[str]:
    samples = expressivity.get("samples")
    if not isinstance(samples, list):
        raise ValueError("expressivity audit 缺少 samples 数组")
    return frozenset(
        _sample_id(row)
        for row in samples
        if isinstance(row, Mapping) and row.get("v2_fully_representable_estimate") is True
    )


def select_encoding_samples(
    manifest_rows: Iterable[Mapping[str, Any]],
    expressivity: Mapping[str, Any],
    *,
    seed: int = 42,
    difficulty_quotas: Mapping[str, int] = DEFAULT_DIFFICULTY_QUOTAS,
) -> list[dict[str, Any]]:
    """Select a quota-balanced set, preferring three-bin coverage and unique families."""
    quotas = {str(key): int(value) for key, value in difficulty_quotas.items()}
    if not quotas or any(value < 0 for value in quotas.values()):
        raise ValueError("difficulty_quotas 必须是非负整数映射")

    eligible_ids = eligible_v2_sample_ids(expressivity)
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in manifest_rows:
        sample_id = _sample_id(row)
        if sample_id in rows_by_id:
            raise ValueError(f"manifest 包含重复 sample_id: {sample_id}")
        rows_by_id[sample_id] = row
    missing = eligible_ids - rows_by_id.keys()
    if missing:
        raise ValueError(f"audit 中有 {len(missing)} 个 fully representable 样本不在 manifest")

    candidates = [
        row
        for sample_id, row in rows_by_id.items()
        if sample_id in eligible_ids and str(row.get("difficulty")) in quotas
    ]
    available = Counter(str(row.get("difficulty")) for row in candidates)
    shortages = {key: quota - available[key] for key, quota in quotas.items() if available[key] < quota}
    if shortages:
        raise ValueError(f"fully representable 候选不足以满足 difficulty 配额: {shortages}")

    ordered_ids = sorted(_sample_id(row) for row in candidates)
    random.Random(seed).shuffle(ordered_ids)
    random_rank = {sample_id: rank for rank, sample_id in enumerate(ordered_ids)}
    remaining = dict(quotas)
    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    bin_counts: Counter[Any] = Counter()

    def choose(pool: Sequence[Mapping[str, Any]], *, prioritize_bin: bool = False) -> Mapping[str, Any]:
        if not pool:
            raise ValueError("没有候选可满足选择约束")

        def score(row: Mapping[str, Any]) -> tuple[Any, ...]:
            family = str(row.get("family") or "unknown")
            complexity_bin = row.get("complexity_bin")
            base = (family_counts[family], bin_counts[complexity_bin], random_rank[_sample_id(row)])
            return ((bin_counts[complexity_bin] > 0,) + base) if prioritize_bin else base

        return min(pool, key=score)

    def add(row: Mapping[str, Any]) -> None:
        sample_id = _sample_id(row)
        difficulty = str(row.get("difficulty"))
        selected.append(row)
        selected_ids.add(sample_id)
        remaining[difficulty] -= 1
        family_counts[str(row.get("family") or "unknown")] += 1
        bin_counts[row.get("complexity_bin")] += 1

    # Freeze coverage first. The normal 0/1/2 bins are discovered from data to keep the function testable.
    all_bins = sorted({row.get("complexity_bin") for row in candidates}, key=lambda value: str(value))
    for complexity_bin in all_bins:
        pool = [
            row
            for row in candidates
            if row.get("complexity_bin") == complexity_bin
            and _sample_id(row) not in selected_ids
            and remaining[str(row.get("difficulty"))] > 0
        ]
        add(choose(pool, prioritize_bin=True))

    difficulty_order = tuple(quotas)
    while any(remaining.values()):
        progressed = False
        for difficulty in difficulty_order:
            if remaining[difficulty] <= 0:
                continue
            pool = [
                row
                for row in candidates
                if str(row.get("difficulty")) == difficulty and _sample_id(row) not in selected_ids
            ]
            add(choose(pool))
            progressed = True
        if not progressed:
            raise AssertionError("选择算法未能推进")

    return [copy.deepcopy(dict(row)) for row in selected]


# Concise alias for callers that do not need experiment-specific naming.
select_samples = select_encoding_samples


def build_frozen_manifest(selected_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return detached JSON-ready rows for sample_manifest.jsonl without touching disk."""
    rows = [copy.deepcopy(dict(row)) for row in selected_rows]
    ids = [_sample_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("冻结 manifest 不能包含重复 sample_id")
    return rows


def build_selection_summary(
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    eligible_count: int | None = None,
) -> dict[str, Any]:
    ids = [_sample_id(row) for row in selected_rows]
    canonical = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": "rq2.encoding_selection.v1",
        "seed": seed,
        "selected_count": len(selected_rows),
        "eligible_v2_fully_representable_count": eligible_count,
        "difficulty_counts": dict(sorted(Counter(str(row.get("difficulty")) for row in selected_rows).items())),
        "complexity_bin_counts": {
            str(key): value
            for key, value in sorted(
                Counter(row.get("complexity_bin") for row in selected_rows).items(), key=lambda item: str(item[0])
            )
        },
        "unique_family_count": len({str(row.get("family") or "unknown") for row in selected_rows}),
        "sample_ids": ids,
        "sample_ids_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def freeze_selection(
    manifest_rows: Iterable[Mapping[str, Any]],
    expressivity: Mapping[str, Any],
    *,
    seed: int = 42,
    difficulty_quotas: Mapping[str, int] = DEFAULT_DIFFICULTY_QUOTAS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = select_encoding_samples(
        manifest_rows, expressivity, seed=seed, difficulty_quotas=difficulty_quotas
    )
    frozen = build_frozen_manifest(selected)
    summary = build_selection_summary(
        frozen, seed=seed, eligible_count=len(eligible_v2_sample_ids(expressivity))
    )
    return frozen, summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="冻结 20 样本编码筛选清单")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expressivity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    with args.expressivity.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    frozen, summary = freeze_selection(_read_jsonl(args.manifest), audit, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "sample_manifest.jsonl", frozen)
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
