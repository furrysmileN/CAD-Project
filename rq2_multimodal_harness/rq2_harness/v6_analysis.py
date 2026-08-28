"""K1–K6 analysis. Primary endpoint is first_attempt joint_quality."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .v5_stats import apply_holm, paired_continuous, paired_success

CONTRASTS = (
    ("K1", "C3", "C1", "P 对 I 的边际价值"),
    ("K2", "C3", "C2", "I 对 P 的边际价值"),
    ("K3", "C3", "C4", "独有事实 vs 重复事实"),
    ("K4", "C3", "C5", "正确事实 vs 局部错误事实"),
)


def _mean_repeats(rows: list[dict[str, Any]], *, endpoint: str, metric: str) -> dict[tuple[str, str], float]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["sample_id"]), str(row["condition"]))
        block = row.get(endpoint) or {}
        value = block.get(metric)
        if value is None:
            value = 0.0 if metric == "joint_quality" else None
        if value is None:
            continue
        buckets[key].append(float(value))
    return {key: sum(vals) / len(vals) for key, vals in buckets.items()}


def _paired_delta(
    means: dict[tuple[str, str], float],
    left: str,
    right: str,
    samples: list[str],
    *,
    invert: bool = False,
) -> list[float]:
    deltas = []
    for sample in samples:
        a = means.get((sample, left))
        b = means.get((sample, right))
        if a is None or b is None:
            continue
        delta = a - b
        deltas.append(-delta if invert else delta)
    return deltas


def analyze_v6(
    rows: list[dict[str, Any]],
    *,
    endpoint: str = "first_attempt",
    metric: str = "joint_quality",
    invert: bool = False,
) -> dict[str, Any]:
    samples = sorted({str(row["sample_id"]) for row in rows})
    means = _mean_repeats(rows, endpoint=endpoint, metric=metric)
    primary = []
    for kid, left, right, title in CONTRASTS:
        deltas = _paired_delta(means, left, right, samples, invert=invert)
        stats = paired_continuous(deltas)
        stats.update({"id": kid, "left": left, "right": right, "title": title, "endpoint": endpoint, "metric": metric})
        primary.append(stats)
    holm = apply_holm(primary, p_field="wilcoxon_p")
    # K5 is an interpretive restatement of K3.
    k3 = next(item for item in holm if item["id"] == "K3")
    k5 = {**k3, "id": "K5", "title": "信息缺口贡献（= C3-C4）"}
    synergy = []
    for sample in samples:
        c0 = means.get((sample, "C0"), 0.0)
        c1 = means.get((sample, "C1"))
        c2 = means.get((sample, "C2"))
        c3 = means.get((sample, "C3"))
        if None in (c1, c2, c3):
            continue
        value = c3 - c1 - c2 + c0
        synergy.append(-value if invert else value)
    k6 = paired_continuous(synergy)
    k6.update({"id": "K6", "title": "协同交互项 C3-C1-C2+C0", "endpoint": endpoint, "metric": metric})
    success_pairs = []
    for kid, left, right, _ in CONTRASTS:
        left_ok = []
        right_ok = []
        for sample in samples:
            lv = means.get((sample, left))
            rv = means.get((sample, right))
            if lv is None or rv is None:
                continue
            left_ok.append(lv > 0)
            right_ok.append(rv > 0)
        success_pairs.append({"id": kid, **paired_success(left_ok, right_ok)})
    return {
        "n_samples": len(samples),
        "endpoint": endpoint,
        "metric": metric,
        "contrasts_holm": holm,
        "k5": k5,
        "k6": k6,
        "mcnemar": success_pairs,
        "bidirectional": all(
            item["mean_delta"] is not None and item["mean_delta"] > 0 and (item.get("ci95_low") or 0) > 0
            for item in holm
            if item["id"] in {"K1", "K2"}
        ),
    }


def mock_rows(n: int = 20, repeats: int = 1) -> list[dict[str, Any]]:
    """Sanity-check fixture: C3 best, then C2, C4, C5, C1, C0. CD inverted."""
    rows = []
    base = {"C0": 0.10, "C1": 0.22, "C2": 0.40, "C3": 0.58, "C4": 0.28, "C5": 0.24}
    for i in range(n):
        jitter = (i % 5) * 0.01
        for cond, jq in base.items():
            for repeat in range(1, repeats + 1):
                quality = jq + jitter
                rows.append(
                    {
                        "sample_id": f"v6_mock_{i:04d}",
                        "condition": cond,
                        "repeat_id": repeat,
                        "first_attempt": {
                            "joint_quality": quality,
                            "common_frame_cd": max(0.02, 0.40 - quality),
                            "success": True,
                        },
                        "final_delivery": {
                            "joint_quality": min(1.0, quality + 0.02),
                            "common_frame_cd": max(0.01, 0.38 - quality),
                            "success": True,
                        },
                    }
                )
    return rows
