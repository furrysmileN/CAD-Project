# -*- coding: utf-8 -*-
"""Complementarity + current-generation-level analysis for the encoding screen.

Definition of complementarity gain (same convention as pilot_v2 analysis):
    per-sample gain = q(combination) - max(q(best single component))
    mean gain, % samples strictly better, paired bootstrap 95% CI.

Also outputs overall generation-level stats (all tasks / successful tasks /
per-difficulty), and a couple of report figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

BASE = Path(__file__).resolve().parents[1] / "outputs" / "encoding_screen_n20" / "analysis"
FIG = BASE / "figures" / "report_v3"
FIG.mkdir(parents=True, exist_ok=True)

tasks = pd.read_csv(BASE / "encoding_task_rows.csv")
cond = pd.read_csv(BASE / "encoding_condition_summary.csv")

piv = tasks.pivot_table(index="sample_id", columns="condition", values="joint_quality")
assert piv.shape == (20, 63), piv.shape
SAMPLES = list(piv.index)

rng = np.random.default_rng(20260813)
NBOOT = 5000
boot_idx = [rng.choice(20, size=20, replace=True) for _ in range(NBOOT)]


def paired_stats(q):
    """q: array of per-sample values -> mean, win rate, bootstrap 95% CI."""
    q = np.asarray(q, dtype=float)
    mean = float(q.mean())
    win = float((q > 1e-9).mean())  # strictly better (ties -> not a win)
    ci = np.percentile([q[idx].mean() for idx in boot_idx], [2.5, 97.5])
    return mean, win, float(ci[0]), float(ci[1])


def components(cid: str):
    """Split a condition id into its single-modality parts."""
    parts, buf = [], ""
    for ch in cid:
        if ch in "TIP":
            if buf:
                parts.append(buf)
            buf = ch
        else:
            buf += ch
    if buf:
        parts.append(buf)
    return parts


def best_single(cid: str, s: str):
    """max over single components of the combination."""
    return max(piv.loc[s, c] for c in components(cid))


def best_bimodal(cid: str, s: str):
    comps = components(cid)
    best = -1.0
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            key = comps[i] + comps[j]
            best = max(best, piv.loc[s, key])
    return best


results = {"bimodal": [], "trimodal": []}
for cid in piv.columns:
    comps = components(cid)
    if len(comps) == 2:
        gain = np.array([piv.loc[s, cid] - best_single(cid, s) for s in SAMPLES])
        mean, win, lo, hi = paired_stats(gain)
        results["bimodal"].append({
            "condition": cid, "components": comps, "mean": mean, "win_rate": win,
            "ci_low": lo, "ci_high": hi,
            "quality": float(cond.set_index("condition").loc[cid, "joint_quality_mean"]),
            "success": float(cond.set_index("condition").loc[cid, "execution_success_rate"]),
            "significant": lo > 0,
        })
    elif len(comps) == 3:
        gain_bim = np.array([piv.loc[s, cid] - best_bimodal(cid, s) for s in SAMPLES])
        gain_sin = np.array([piv.loc[s, cid] - best_single(cid, s) for s in SAMPLES])
        mb, wb, lb, hb = paired_stats(gain_bim)
        ms, ws, ls, hs = paired_stats(gain_sin)
        results["trimodal"].append({
            "condition": cid, "components": comps,
            "gain_vs_best_bimodal": mb, "win_rate_vs_bimodal": wb, "ci_low_vs_bimodal": lb, "ci_high_vs_bimodal": hb,
            "gain_vs_best_single": ms, "win_rate_vs_single": ws, "ci_low_vs_single": ls, "ci_high_vs_single": hs,
            "quality": float(cond.set_index("condition").loc[cid, "joint_quality_mean"]),
            "success": float(cond.set_index("condition").loc[cid, "execution_success_rate"]),
            "significant_vs_bimodal": lb > 0,
            "significant_vs_single": ls > 0,
        })

bi = pd.DataFrame(results["bimodal"]).sort_values("mean", ascending=False)
tri = pd.DataFrame(results["trimodal"]).sort_values("gain_vs_best_bimodal", ascending=False)

# ---- shared conditions vs previous round (pilot) ----
shared = {
    "T2I1": ("TI", -0.0812), "T2P3": ("TP", -0.0853),
    "I1P3": ("IP", -0.0732), "T2I1P3": ("TIP", -0.1152),
}
shared_rows = []
for cid, (old_name, old_gain) in shared.items():
    if cid in bi["condition"].values:
        row = bi[bi["condition"] == cid].iloc[0]
        shared_rows.append({"condition": cid, "pilot_name": old_name, "pilot_gain": old_gain,
                            "screen_gain": row["mean"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                            "win_rate": row["win_rate"]})
    else:
        row = tri[tri["condition"] == cid].iloc[0]
        shared_rows.append({"condition": cid, "pilot_name": old_name, "pilot_gain": old_gain,
                            "screen_gain": row["gain_vs_best_single"], "ci_low": row["ci_low_vs_single"],
                            "ci_high": row["ci_high_vs_single"], "win_rate": row["win_rate_vs_single"]})

# ---- generation-level stats ----
allq = tasks["joint_quality"].to_numpy()
ok = tasks[tasks["status"] == "completed"]["joint_quality"].to_numpy()
level = {
    "n_tasks": int(len(tasks)),
    "success_rate": float((tasks["status"] == "completed").mean()),
    "mean_all": float(allq.mean()),
    "median_all": float(np.median(allq)),
    "mean_completed": float(ok.mean()),
    "median_completed": float(np.median(ok)),
    "q25_completed": float(np.percentile(ok, 25)),
    "q75_completed": float(np.percentile(ok, 75)),
    "frac_below_02": float((allq < 0.2).mean()),
    "frac_below_01": float((allq < 0.1).mean()),
    "best_condition": float(cond["joint_quality_mean"].max()),
    "best_single": float(cond[cond["condition"].str.len() == 2]["joint_quality_mean"].max()),
    "best_bimodal": float(cond[cond["condition"].str.len() == 4]["joint_quality_mean"].max()),
    "best_trimodal": float(cond[cond["condition"].str.len() == 6]["joint_quality_mean"].max()),
    "difficulty": tasks.groupby("difficulty").agg(
        success=("execution_success", "mean"), quality=("joint_quality", "mean")).round(3).to_dict("index"),
}

out = {"bimodal_summary": {
    "n": len(bi), "mean_gain": float(bi["mean"].mean()), "median_gain": float(bi["mean"].median()),
    "n_positive": int((bi["mean"] > 0).sum()), "n_significant": int(bi["significant"].sum()),
    "mean_win_rate": float(bi["win_rate"].mean()),
    "top": bi.head(8)[["condition", "mean", "win_rate", "ci_low", "ci_high", "quality", "success"]].round(3).to_dict("records"),
    "bottom": bi.tail(5)[["condition", "mean", "win_rate", "ci_low", "ci_high", "quality", "success"]].round(3).to_dict("records"),
}, "trimodal_summary": {
    "n": len(tri),
    "mean_gain_vs_bimodal": float(tri["gain_vs_best_bimodal"].mean()),
    "median_gain_vs_bimodal": float(tri["gain_vs_best_bimodal"].median()),
    "n_positive_vs_bimodal": int((tri["gain_vs_best_bimodal"] > 0).sum()),
    "n_significant_vs_bimodal": int(tri["significant_vs_bimodal"].sum()),
    "mean_gain_vs_single": float(tri["gain_vs_best_single"].mean()),
    "n_positive_vs_single": int((tri["gain_vs_best_single"] > 0).sum()),
    "n_significant_vs_single": int(tri["significant_vs_single"].sum()),
    "top": tri.head(8)[["condition", "gain_vs_best_bimodal", "win_rate_vs_bimodal", "ci_low_vs_bimodal",
                        "ci_high_vs_bimodal", "gain_vs_best_single", "quality", "success"]].round(3).to_dict("records"),
}, "shared_with_pilot": shared_rows, "generation_level": level,
    "best_per_text_level": (
        tasks.assign(tl=tasks["condition"].str[:2]).groupby("tl")["joint_quality"].mean().round(3).to_dict()),
}

with open(BASE / "complementarity_summary.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2, default=str)

# ----------------------------------------------------------------------
# figure A: complementarity gains (bimodal vs best single; trimodal vs best bimodal)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, df, ycol, lo, hi, title, ylabel in (
    (axes[0], bi, "mean", "ci_low", "ci_high", "双模态组合 − 最佳单模态\n（27 个双模态条件，按增益排序）", "逐样本配对增益"),
    (axes[1], tri, "gain_vs_best_bimodal", "ci_low_vs_bimodal", "ci_high_vs_bimodal",
     "三模态组合 − 最佳双模态\n（27 个三模态条件，按增益排序）", "逐样本配对增益"),
):
    colors = []
    for cid in df["condition"]:
        comps = components(cid)
        colors.append({"TI": "#f4b400", "TP": "#4285f4", "IP": "#34a853"}[comps[0][0] + comps[1][0]])
    sig = df[lo] > 0
    ax.bar(df["condition"], df[ycol], color=colors, alpha=0.88,
           yerr=[df[ycol] - df[lo], df[hi] - df[ycol]], capsize=3, error_kw={"elinewidth": 1})
    for i, (_, r) in enumerate(df.iterrows()):
        if r[lo] > 0:
            ax.text(i, r[ycol] + 0.012, "★", ha="center", fontsize=10, color="#c62828")
    ax.axhline(0, color="#333", lw=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11.5)
    ax.tick_params(axis="x", labelrotation=90, labelsize=7.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(-0.30, 0.34)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#f4b400", "#4285f4", "#34a853")]
axes[0].legend(handles, ["文本+图像 (TI)", "文本+点云 (TP)", "图像+点云 (IP)"], loc="upper left", frameon=False, fontsize=9)
fig.suptitle("互补性：加一个模态，比'只用最好的那一个模态'更好吗？\n（★ = 增益的 95% 置信区间整体在 0 以上；绿色条 = 图像+点云）", fontsize=12)
fig.savefig(FIG / "v3_fig9_complementarity.png")
plt.close(fig)

# ----------------------------------------------------------------------
# figure B: current generation level
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
axes[0].hist(allq, bins=30, color="#9aa0a6", alpha=0.7, edgecolor="white", label="全部 1260 次（失败记 0）")
axes[0].hist(ok, bins=30, color="#1a73e8", alpha=0.75, edgecolor="white", label="成功 883 次")
axes[0].axvline(allq.mean(), color="#333", ls="--", lw=1.2)
axes[0].text(allq.mean() + 0.005, 205, f"全员均值 {allq.mean():.3f}", fontsize=8.5)
axes[0].axvline(ok.mean(), color="#1a73e8", ls="--", lw=1.2)
axes[0].text(ok.mean() + 0.005, 175, f"成功者均值 {ok.mean():.3f}", fontsize=8.5)
axes[0].set_xlabel("joint quality（0~1）")
axes[0].set_ylabel("任务数")
axes[0].set_title("质量分布：0 附近的柱子 = 失败任务")
axes[0].legend(fontsize=8.5, frameon=False)

cats = ["最佳单模态\n(T2)", "最佳双模态\n(T2I1)", "最佳三模态\n(T2I2P1)"]
qs = [0.3699, 0.4293, 0.4463]
ss = [0.75, 0.70, 0.75]
x = np.arange(3)
b = axes[1].bar(x, qs, 0.5, color=["#4285f4", "#f4b400", "#34a853"], alpha=0.9)
axes[1].set_xticks(x); axes[1].set_xticklabels(cats, fontsize=9)
for i in range(3):
    axes[1].text(i, qs[i] + 0.012, f"{qs[i]:.3f}", ha="center", fontweight="bold")
axes[1].set_ylim(0, 0.62)
axes[1].set_ylabel("joint quality 均值")
axes[1].set_title("目前最好水平（满分 1.0）\n成功率：T2 75% / T2I1 70% / T2I2P1 75%")
axes[1].grid(axis="y", alpha=0.25)
fig.suptitle("目前生成水平如何？", fontsize=12)
fig.savefig(FIG / "v3_fig10_generation_level.png")
plt.close(fig)

print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
