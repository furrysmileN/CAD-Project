"""Generate figures for EXPERIMENT_REPORT_V2 (encoding screen n20 + pilot comparison)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENC_DIR = ROOT / "outputs" / "encoding_screen_n20"
ANA = ENC_DIR / "analysis"
FIG = ANA / "figures" / "report_v2"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
})

C_MODALITY = {
    "T1": "#5B8FF9", "T2": "#2C6BE0", "T3": "#9EC5FE",
    "I1": "#F6BD16", "I2": "#E8910A", "I3": "#FBE3A4",
    "P1": "#5AD8A6", "P2": "#2FAE7E", "P3": "#A8EFD0",
}
C_OLD = "#B8B8B8"

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
enc = pd.read_csv(ANA / "encoding_condition_summary.csv")
cost = pd.read_csv(ANA / "encoding_cost_summary.csv")
pilot = pd.read_csv(ROOT / "outputs" / "pilot_v2" / "analysis" / "condition_summary.csv")

# marginal means (from ENCODING_SCREEN_REPORT_ZH.md)
marginal = {
    "Text":   [("T1", 0.1807, 0.1139, 0.2534), ("T2", 0.3864, 0.2422, 0.5332), ("T3", 0.3103, 0.1762, 0.4499)],
    "Render": [("I1", 0.3132, 0.2349, 0.3915), ("I2", 0.3025, 0.2176, 0.3883), ("I3", 0.2818, 0.1920, 0.3786)],
    "Point":  [("P1", 0.2998, 0.2181, 0.3842), ("P2", 0.2898, 0.2167, 0.3656), ("P3", 0.2791, 0.2043, 0.3537)],
}

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
for ax, (group, items) in zip(axes, marginal.items()):
    names = [it[0] for it in items]
    means = [it[1] for it in items]
    lows = [it[2] for it in items]
    highs = [it[3] for it in items]
    err = [[m - l for m, l in zip(means, lows)], [h - m for m, h in zip(means, highs)]]
    colors = [C_MODALITY[n] for n in names]
    bars = ax.bar(names, means, yerr=err, capsize=4, color=colors, edgecolor="#333333", linewidth=0.6, error_kw={"elinewidth": 1})
    ax.set_title(group, fontsize=11)
    ax.set_ylabel("Failure-aware joint quality (marginal mean)")
    ax.set_ylim(0, 0.60)
    for b, m in zip(bars, means):
        ax.annotate(f"{m:.3f}", (b.get_x() + b.get_width() / 2, m + 0.02), ha="center", fontsize=9)
    ax.tick_params(axis="x", labelsize=10)
axes[0].set_ylabel("Failure-aware joint quality\n(marginal mean, with 95% bootstrap CI)", fontsize=9)
fig.suptitle("Encoding screening (n=20 x 63): which representation of each modality works best?",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(FIG / "v2_fig1_marginal_means.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# fig 2: top15 / bottom10 joint quality of 63 conditions
# ---------------------------------------------------------------------------
enc_sorted = enc.sort_values("joint_quality_mean", ascending=False)
top = enc_sorted.head(15).iloc[::-1]
bot = enc_sorted.tail(10).iloc[::-1]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=False)
ax = axes[0]
names = top["condition"].tolist()
vals = top["joint_quality_mean"].tolist()
colors = []
for n in names:
    if "T2" in n:
        colors.append(C_MODALITY["T2"])
    elif "T3" in n:
        colors.append(C_MODALITY["T3"])
    else:
        colors.append(C_MODALITY["T1"])
ax.barh(names, vals, color=colors, edgecolor="#333", linewidth=0.5)
ax.set_xlim(0, 0.55)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.3f}", (v + 0.008, i), va="center", fontsize=8.5)
ax.set_title("Top 15 conditions (failure-aware joint quality)", fontsize=11)
ax.set_xlabel("Joint quality (higher = better)")

ax = axes[1]
names = bot["condition"].tolist()
vals = bot["joint_quality_mean"].tolist()
colors = [C_MODALITY["T1"] if n.startswith("T1") or "T1" in n else "#999999" for n in names]
colors = []
for n in names:
    if "T1" in n and "T2" not in n and "T3" not in n:
        colors.append(C_MODALITY["T1"])
    elif "I3" in n and "I1" not in n and "I2" not in n and "T2" not in n:
        colors.append("#777777")
    else:
        colors.append("#BBBBBB")
ax.barh(names, vals, color=colors, edgecolor="#333", linewidth=0.5)
ax.set_xlim(0, 0.35)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.3f}", (v + 0.008, i), va="center", fontsize=8.5)
ax.set_title("Bottom 10 conditions", fontsize=11)
ax.set_xlabel("Joint quality (higher = better)")
fig.suptitle("63-condition ranking: T2 (detailed L3 text) families dominate the top; T1 (one-line) dominates the bottom",
             fontsize=11.5, y=1.0)
fig.tight_layout()
fig.savefig(FIG / "v2_fig2_ranking.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# fig 3: execution success vs joint quality scatter (63 conditions)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 6.4))
x = enc["execution_success_rate"].values
y = enc["joint_quality_mean"].values
labels = enc["condition"].values
has_t2 = enc["condition"].str.contains("T2")
has_t1_only = ~enc["condition"].str.contains("T") | enc["condition"].str.contains("T1")
size = [90 if h else 55 for h in has_t2]
color = []
for n in labels:
    if "T2" in n:
        color.append("#2C6BE0")
    elif "T1" in n:
        color.append("#5B8FF9")
    elif "T3" in n:
        color.append("#9EC5FE")
    else:
        color.append("#5AD8A6")
ax.scatter(x, y, s=size, c=color, alpha=0.75, edgecolors="#333", linewidths=0.5, zorder=3)
for xi, yi, lab in zip(x, y, labels):
    if yi > 0.42 or yi < 0.12 or (xi >= 0.9 and yi > 0.2):
        ax.annotate(lab, (xi, yi), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Execution success rate (higher = better)")
ax.set_ylabel("Failure-aware joint quality (higher = better)")
ax.set_xlim(0.4, 1.0)
ax.set_ylim(0.0, 0.5)
ax.axvline(0.7, color="#999", ls="--", lw=1)
ax.axhline(0.3, color="#999", ls="--", lw=1)
ax.annotate("P-only / I-only:\nreliable but plain", (0.93, 0.10), fontsize=9, ha="right", color="#2FAE7E")
ax.annotate("T2 combos:\nhigh quality,\nmedium reliability", (0.60, 0.44), fontsize=9, color="#2C6BE0")
ax.annotate("T1 combos:\nlow quality", (0.50, 0.06), fontsize=9, color="#5B8FF9")
ax.set_title("63 conditions: success rate vs quality\n(blue = text conditions, green = visual-only; large dots = T2)")
fig.tight_layout()
fig.savefig(FIG / "v2_fig3_success_vs_quality.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# fig 4: old pilot (7 cond, qwen3.7-plus) vs new encoding screen (same codes)
# ---------------------------------------------------------------------------
mapping = {"T": "T2", "I": "I1", "P": "P3", "TI": "T2I1", "TP": "T2P3", "IP": "I1P3", "TIP": "T2I1P3"}
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
old_conds = ["T", "I", "P", "TI", "TP", "IP", "TIP"]
new_conds = [mapping[c] for c in old_conds]
old_valid = pilot.set_index("condition").loc[old_conds, "valid_rate"].values
new_valid = enc.set_index("condition").loc[new_conds, "execution_success_rate"].values
old_jq = pilot.set_index("condition").loc[old_conds, "joint_quality_mean"].values
new_jq = enc.set_index("condition").loc[new_conds, "joint_quality_mean"].values

ax = axes[0]
w = 0.36
xx = np.arange(len(old_conds))
ax.bar(xx - w / 2, old_valid, w, label="Pilot v2 (100 samples, qwen3.7-plus)", color="#B8B8B8", edgecolor="#333", linewidth=0.5)
ax.bar(xx + w / 2, new_valid, w, label="Encoding screen (20 samples, qwen3.8-max)", color="#2FAE7E", edgecolor="#333", linewidth=0.5)
ax.set_xticks(xx)
ax.set_xticklabels([f"{o}\n={n}" for o, n in zip(old_conds, new_conds)], fontsize=9)
ax.set_ylabel("Geometry-valid rate")
ax.set_ylim(0, 1.0)
ax.set_title("Valid-geometry rate", fontsize=11)
ax.legend(fontsize=8, loc="lower left")

ax = axes[1]
ax.bar(xx - w / 2, old_jq, w, label="Pilot v2 (100 samples, qwen3.7-plus)", color="#B8B8B8", edgecolor="#333", linewidth=0.5)
ax.bar(xx + w / 2, new_jq, w, label="Encoding screen (20 samples, qwen3.8-max)", color="#2C6BE0", edgecolor="#333", linewidth=0.5)
ax.set_xticks(xx)
ax.set_xticklabels([f"{o}\n={n}" for o, n in zip(old_conds, new_conds)], fontsize=9)
ax.set_ylabel("Failure-aware joint quality")
ax.set_ylim(0, 0.55)
ax.set_title("Joint quality", fontsize=11)
ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Same 7 conditions, two rounds: new model + repair v2.1 improve execution across the board\n(different samples & n, descriptive only)", fontsize=11.5, y=1.0)
fig.tight_layout()
fig.savefig(FIG / "v2_fig4_pilot_vs_screen.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# fig 5: pipeline failure composition, old vs new (7 shared conditions)
# ---------------------------------------------------------------------------
old_fail = pd.read_csv(ROOT / "outputs" / "pilot_v2" / "analysis" / "pipeline_summary.csv")
enc_fail_rows = []
for n in new_conds:
    parse = int((enc.set_index("condition").loc[n, "parse_rate"]) * 20)
    n_parse_fail = 20 - parse
    schema_ok = int(20 * enc.set_index("condition").loc[n, "schema_valid_rate"])
    n_schema_fail = 20 - n_parse_fail - schema_ok
    n_exec = int(20 * enc.set_index("condition").loc[n, "execution_success_rate"])
    n_runtime_fail = 20 - n_parse_fail - n_schema_fail - n_exec
    enc_fail_rows.append({"condition": n, "parse_failed": n_parse_fail, "schema_failed": n_schema_fail,
                          "runtime_failed": n_runtime_fail, "success": n_exec})
enc_fail = pd.DataFrame(enc_fail_rows)

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
cats = ["parse_failed", "schema_failed", "runtime_failed", "success"]
colors = ["#D9544D", "#E8910A", "#F6BD16", "#2FAE7E"]
for ax, df, conds, title in [
    (axes[0], None, old_conds, "Pilot v2 (100 samples, strict Plan v2)"),
    (axes[1], enc_fail, new_conds, "Encoding screen (20 samples, repair v2.1 R4)"),
]:
    if df is None:
        # old: recompute from pipeline_summary
        rows = []
        old_pipe = pd.read_csv(ROOT / "outputs" / "pilot_v2" / "analysis" / "pipeline_summary.csv")
        for c in old_conds:
            s = old_pipe.set_index("condition").loc[c]
            rows.append({
                "condition": c,
                "parse_failed": s["parse_failed"],
                "schema_failed": s["validation_failed"],
                "runtime_failed": s["runtime_failed"],
                "success": int(round(s["execution_success_rate"] * s["n"])),
            })
        df = pd.DataFrame(rows)
    df = df.set_index("condition").loc[conds]
    bottom = np.zeros(len(df))
    for cat, col in zip(cats, colors):
        vals = df[cat].values.astype(float)
        vals = vals / df.sum(axis=1).values * 100
        ax.bar(range(len(df)), vals, bottom=bottom, color=col, edgecolor="#333", linewidth=0.4,
               label=cat.replace("_", " "), width=0.62)
        bottom += vals
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(conds, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of tasks (%)")
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper right", ncol=1)
fig.suptitle("Where do tasks fail? Repair v2.1 cuts schema failures for T2/P-conditions\n(different models and n; descriptive only)", fontsize=11.5, y=1.0)
fig.tight_layout()
fig.savefig(FIG / "v2_fig5_failure_composition.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# fig 6: cost vs quality (token cost from encoding_cost_summary)
# ---------------------------------------------------------------------------
merged = enc.merge(cost[["condition", "total_tokens_mean", "latency_sec_mean"]], on="condition")
fig, ax = plt.subplots(figsize=(8.6, 5.6))
x = merged["total_tokens_mean"].values
y = merged["joint_quality_mean"].values
labels = merged["condition"].values
color = ["#2C6BE0" if "T2" in n else ("#9EC5FE" if "T3" in n else ("#5B8FF9" if "T1" in n else "#5AD8A6")) for n in labels]
size = [95 if "T2" in n else 55 for n in labels]
ax.scatter(x, y, s=size, c=color, alpha=0.75, edgecolors="#333", linewidths=0.5, zorder=3)
for xi, yi, lab in zip(x, y, labels):
    if yi > 0.40 or (xi < 1600 and yi > 0.2) or (yi < 0.12 and xi > 1500):
        ax.annotate(lab, (xi, yi), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Mean total tokens per request (proxy of cost)")
ax.set_ylabel("Failure-aware joint quality")
ax.set_title("Cost vs quality: T2 alone is nearly free yet beats most image-heavy combos")
fig.tight_layout()
fig.savefig(FIG / "v2_fig6_cost_quality.png", bbox_inches="tight")
plt.close(fig)

print("figures written to", FIG)
print([p.name for p in sorted(FIG.glob("*.png"))])
