"""V5 共用统计：复用 confirm_analysis / analysis，不另写一套检验。"""
from __future__ import annotations

from statistics import median
from typing import Any

import numpy as np

from .analysis import bootstrap_ci, holm_adjust
from .confirm_analysis import mcnemar_exact, wilcoxon_signed_rank


def paired_win_counts(deltas: list[float], *, eps: float = 1e-12) -> dict[str, int]:
    n_left = sum(1 for value in deltas if value > eps)
    n_right = sum(1 for value in deltas if value < -eps)
    return {
        "n_left_better": n_left,
        "n_right_better": n_right,
        "n_tie": len(deltas) - n_left - n_right,
    }


def paired_continuous(deltas: list[float], *, seed: int = 42) -> dict[str, Any]:
    ci = bootstrap_ci(deltas, seed=seed) if deltas else {"mean": None, "low": None, "high": None, "n": 0}
    array = np.asarray(deltas, dtype=float)
    wilcox = wilcoxon_signed_rank(array) if len(array) else {"n": 0, "stat": 0.0, "p_value": 1.0}
    n = len(deltas)
    if n and wilcox.get("n"):
        z = abs(float(wilcox.get("z") or 0.0))
        r_rb = (2 * z) / (n ** 0.5) / 2.0 if n else 0.0
        effect = float(z / (n ** 0.5)) if n else 0.0
    else:
        r_rb = 0.0
        effect = 0.0
    wins = paired_win_counts(deltas)
    return {
        "n_pairs": n,
        "mean_delta": ci["mean"],
        "median_delta": float(median(deltas)) if deltas else None,
        "ci95_low": ci["low"],
        "ci95_high": ci["high"],
        "wilcoxon_stat": wilcox.get("stat"),
        "wilcoxon_p": wilcox.get("p_value"),
        "wilcoxon_z": wilcox.get("z"),
        "effect_r": effect,
        **wins,
    }


def paired_success(left_ok: list[bool], right_ok: list[bool]) -> dict[str, Any]:
    a = np.asarray(left_ok, dtype=bool)
    b = np.asarray(right_ok, dtype=bool)
    return mcnemar_exact(a, b)


def apply_holm(rows: list[dict[str, Any]], p_field: str = "wilcoxon_p") -> list[dict[str, Any]]:
    p_values = [float(row.get(p_field) or 1.0) for row in rows]
    adjusted = holm_adjust(p_values)
    out = []
    for row, p_adj in zip(rows, adjusted):
        item = dict(row)
        item["p_holm"] = p_adj
        out.append(item)
    return out
