# -*- coding: utf-8 -*-
"""Generate report-v5 figures for the complementarity confirmation.

Reads Phase C analysis CSVs / V4 confirm summary and writes PNGs. Offline.
"""
from __future__ import annotations

import json
from collections import Counter
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

ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT / "outputs" / "v5_complementarity" / "repeats" / "analysis"
V4 = ROOT / "outputs" / "native_pointcloud_v1" / "confirm_n100" / "analysis"
FIG = C_DIR / "figures" / "report_v5"
FIG.mkdir(parents=True, exist_ok=True)

LABEL = {
    "I1": "照片",
    "T1I1": "一句话+照片",
    "T2I1": "详细步骤+照片",
    "P_proj": "旧剪影",
    "I1P_proj": "照片+旧剪影",
    "I1P_shuffle": "照片+错配几何",
    "P_geom": "几何说明",
    "I1P_geom": "照片+几何说明",
    "T1I1P_geom": "一句话+照片+几何",
    "T2I1P_geom": "详细步骤+照片+几何",
}
COLOR = {
    "I1": "#9aa0a6",
    "T1I1": "#9aa0a6",
    "T2I1": "#5f6368",
    "P_proj": "#e8716a",
    "I1P_proj": "#f4b400",
    "I1P_shuffle": "#f6bf26",
    "P_geom": "#34a853",
    "I1P_geom": "#188038",
    "T1I1P_geom": "#137333",
    "T2I1P_geom": "#0d652d",
}
ORDER = [
    "I1",
    "T1I1",
    "T2I1",
    "P_proj",
    "I1P_proj",
    "I1P_shuffle",
    "P_geom",
    "I1P_geom",
    "T1I1P_geom",
    "T2I1P_geom",
]
V4_MAP = {"P_geom": "P_geom_tool"}

agg = pd.read_csv(C_DIR / "primary_metrics.csv")
holm = pd.read_csv(C_DIR / "contrasts_holm.csv")
v4 = pd.read_csv(V4 / "pc_condition_summary.csv")
report = json.loads((C_DIR / "v5_phase_c_live_report.json").read_text(encoding="utf-8"))
cond_rows = {row["condition"]: row for row in report["official"]["condition_summary"]}


def _ordered_summary() -> pd.DataFrame:
    rows = []
    for cid in ORDER:
        item = cond_rows[cid]
        rows.append(
            {
                "condition": cid,
                "label": LABEL[cid],
                **item,
            }
        )
    return pd.DataFrame(rows)


summary = _ordered_summary()

# ----------------------------------------------------------------------
# fig1: 10-condition ranking
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.2, 5.4))
colors = [COLOR[c] for c in summary["condition"]]
ax.bar(summary["label"], summary["mean_joint_quality"], color=colors, alpha=0.9, edgecolor="white")
for i, r in summary.iterrows():
    ax.text(
        i,
        r["mean_joint_quality"] + 0.012,
        f"{r['mean_joint_quality']:.3f}",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
ax.set_ylim(0, 0.72)
ax.set_ylabel("总分 joint quality（失败记 0）")
ax.set_title("确认实验：10 种给法的平均总分（新 100 零件，三次重复先取均值）", fontsize=13)
ax.tick_params(axis="x", rotation=22)
ax.grid(axis="y", alpha=0.25)
fig.savefig(FIG / "v5_fig1_condition_means.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig2: Holm forest C1–C8
# ----------------------------------------------------------------------
FOREST = [
    ("C1", "几何说明 − 旧剪影"),
    ("C2", "照片+几何 − 纯照片"),
    ("C3", "照片+几何 − 纯几何说明"),
    ("C4", "弱文本下：加几何 − 不加"),
    ("C5", "强文本下：加几何 − 不加"),
    ("C6", "弱文本增益 − 强文本增益"),
    ("C7", "照片+几何 − 照片+旧剪影"),
    ("C8", "照片+几何 − 照片+错配几何"),
]
hc = holm.set_index("id").loc[[k for k, _ in FOREST]].reset_index()
hc["name"] = [n for _, n in FOREST]
hc = hc.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10.0, 5.6))
for i, r in hc.iterrows():
    lo, hi, mu = r["ci95_low"], r["ci95_high"], r["mean_delta"]
    color = "#188038" if lo > 0 else ("#e8716a" if hi < 0 else "#9aa0a6")
    ax.plot([lo, hi], [i, i], color=color, lw=2.4)
    ax.plot(mu, i, "o", color=color, ms=8)
    p = float(r["p_holm"])
    p_txt = "<0.001" if p < 0.001 else f"{p:.3f}"
    ax.text(
        hi + 0.008,
        i,
        f"{mu:+.3f}  {int(r['n_left_better'])}/{int(r['n_right_better'])}/{int(r['n_tie'])}  Holm p={p_txt}",
        va="center",
        fontsize=8.5,
    )
ax.axvline(0, color="#333", lw=1)
ax.set_yticks(range(len(hc)))
ax.set_yticklabels(hc["name"])
ax.set_xlabel("同一零件上的总分差（左更好为正；误差线 = 95% 区间；右侧为 赢/输/平 与 Holm p）")
ax.set_title("确认实验：预注册对比 C1–C8（n=100，失败记 0，Holm 校正）", fontsize=12)
ax.set_xlim(-0.05, 0.55)
ax.grid(axis="x", alpha=0.25)
fig.savefig(FIG / "v5_fig2_paired_forest.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig3: V4 old 100 vs V5 new 100
# ----------------------------------------------------------------------
shared = ["I1", "T1I1", "P_proj", "I1P_proj", "P_geom", "I1P_geom", "T1I1P_geom"]
v4_idx = v4.set_index("condition")
v4_vals = []
v5_vals = []
for cid in shared:
    v4_key = V4_MAP.get(cid, cid)
    v4_vals.append(float(v4_idx.loc[v4_key, "mean_joint_quality"]))
    v5_vals.append(float(cond_rows[cid]["mean_joint_quality"]))
x = np.arange(len(shared))
w = 0.38
fig, ax = plt.subplots(figsize=(10.4, 5.2))
ax.bar(x - w / 2, v4_vals, w, label="V4 旧 100 零件", color="#8ab4f8", edgecolor="white")
ax.bar(x + w / 2, v5_vals, w, label="V5 新 100 零件", color="#188038", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([LABEL[k] for k in shared], rotation=18)
ax.set_ylabel("总分 joint quality")
ax.set_title("旧 100 vs 新 100：方向一致，绝对水平接近", fontsize=13)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)
ax.set_ylim(0, 0.75)
fig.savefig(FIG / "v5_fig3_v4_vs_v5.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig4: difficulty
# ----------------------------------------------------------------------
want = ["I1", "P_proj", "P_geom", "I1P_geom", "T2I1P_geom"]
levels = ["easy", "medium", "hard"]
level_zh = {"easy": "简单", "medium": "中等", "hard": "困难"}
fig, ax = plt.subplots(figsize=(9.6, 5.0))
x = np.arange(len(levels))
w = 0.16
for i, cond_id in enumerate(want):
    vals = []
    for lv in levels:
        sub = agg[(agg["condition"] == cond_id) & (agg["difficulty"] == lv)]
        vals.append(float(sub["joint_quality"].mean()) if len(sub) else np.nan)
    ax.bar(
        x + (i - 2) * w,
        vals,
        w,
        label=LABEL[cond_id],
        color=COLOR[cond_id],
        alpha=0.92,
        edgecolor="white",
    )
ax.set_xticks(x)
ax.set_xticklabels([level_zh[lv] for lv in levels])
ax.set_ylabel("总分 joint quality")
ax.set_title("按零件难度分层：三种难度上，几何说明都高于旧剪影和纯照片", fontsize=12)
ax.legend(frameon=False, ncol=3, fontsize=9)
ax.grid(axis="y", alpha=0.25)
ax.set_ylim(0, 0.78)
fig.savefig(FIG / "v5_fig4_difficulty.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig5: success vs quality
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.8, 5.4))
for _, r in summary.iterrows():
    ax.scatter(r["success_rate"] * 100, r["mean_joint_quality"], s=90, color=COLOR[r["condition"]], zorder=3)
    ax.annotate(
        r["label"],
        (r["success_rate"] * 100, r["mean_joint_quality"]),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=8.5,
    )
ax.set_xlabel("成功率（%）")
ax.set_ylabel("总分 joint quality")
ax.set_title("成功率 vs 总分：几乎都能画出零件，拉开差距的是像不像", fontsize=12)
ax.set_xlim(97.6, 100.6)
ax.set_ylim(0.22, 0.62)
ax.grid(alpha=0.25)
fig.savefig(FIG / "v5_fig5_success_quality.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig6: JQ distribution
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.4, 5.2))
bins = np.linspace(0, 1, 21)
for cond_id, alpha in (("I1", 0.4), ("P_geom", 0.45), ("I1P_geom", 0.55)):
    vals = agg.loc[agg["condition"] == cond_id, "joint_quality"].astype(float).fillna(0)
    ax.hist(vals, bins=bins, alpha=alpha, label=LABEL[cond_id], color=COLOR[cond_id], edgecolor="white")
ax.set_xlabel("单个零件的总分（三次重复先取均值）")
ax.set_ylabel("零件个数（各 100 个）")
ax.set_title("分数分布：几何说明把一批零件从低分区推进到高分区", fontsize=12)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)
fig.savefig(FIG / "v5_fig6_jq_hist.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig7: cost vs quality
# ----------------------------------------------------------------------
summary["tokens"] = summary["mean_prompt_tokens"] + summary["mean_completion_tokens"]
fig, ax = plt.subplots(figsize=(9.0, 5.4))
for _, r in summary.iterrows():
    ax.scatter(r["tokens"], r["mean_joint_quality"], s=90, color=COLOR[r["condition"]], zorder=3)
    ax.annotate(
        r["label"],
        (r["tokens"], r["mean_joint_quality"]),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=8.5,
    )
ax.set_xlabel("平均每次请求的 token 数（提示词 + 回复）")
ax.set_ylabel("总分 joint quality")
ax.set_title("成本 vs 质量：几何说明本身不贵；叠照片才明显涨 token", fontsize=12)
ax.grid(alpha=0.25)
ax.set_ylim(0.22, 0.62)
fig.savefig(FIG / "v5_fig7_cost_quality.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig8: geometry endpoints on completed rows
# ----------------------------------------------------------------------
geom_rows = []
for cond_id in ["I1", "P_proj", "P_geom", "I1P_geom", "T2I1P_geom"]:
    sub = agg[agg["condition"] == cond_id]
    ok = sub[sub["completed"].astype(str).str.lower().isin(["true", "1"])]
    geom_rows.append(
        {
            "label": LABEL[cond_id],
            "cd": pd.to_numeric(ok["common_frame_cd"], errors="coerce").mean(),
            "iou": pd.to_numeric(ok["voxel_iou"], errors="coerce").mean(),
            "f1": pd.to_numeric(ok["f1_shape"], errors="coerce").mean(),
        }
    )
g = pd.DataFrame(geom_rows)
fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.3))
axes[0].bar(g["label"], g["cd"], color="#e8716a", alpha=0.9)
axes[0].set_title("common-frame CD（越低越好）")
axes[1].bar(g["label"], g["iou"], color="#188038", alpha=0.9)
axes[1].set_title("体积重合 IoU（越高越好）")
axes[2].bar(g["label"], g["f1"], color="#1a73e8", alpha=0.9)
axes[2].set_title("F1@1%（越高越好，门槛很严）")
for ax in axes:
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("几何口径（样本×条件，三次重复已取均值；仅 completed）", fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "v5_fig8_geometry.png")
plt.close(fig)

# extras for the markdown report
diff_counts = Counter(agg[agg["condition"] == "I1"]["difficulty"].tolist())
hist_stats = {}
for cid in ["I1", "P_proj", "P_geom", "I1P_geom", "T2I1P_geom"]:
    vals = agg.loc[agg["condition"] == cid, "joint_quality"].astype(float)
    hist_stats[cid] = {
        "n": int(len(vals)),
        "lt02": int((vals < 0.2).sum()),
        "gt08": int((vals > 0.8).sum()),
        "median": float(vals.median()),
    }
diff_table = {}
for cid in want:
    diff_table[cid] = {}
    for lv in levels:
        sub = agg[(agg["condition"] == cid) & (agg["difficulty"] == lv)]
        diff_table[cid][lv] = {
            "n": int(len(sub)),
            "mean_jq": float(sub["joint_quality"].mean()) if len(sub) else None,
        }

extras = {
    "difficulty_counts": dict(diff_counts),
    "hist": hist_stats,
    "difficulty_means": diff_table,
    "geometry_completed": geom_rows,
    "complementarity": report["official"]["complementarity"],
    "run_counts": report["run_counts"],
}
(C_DIR / "report_v5_extras.json").write_text(json.dumps(extras, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote figures to {FIG}")
print(json.dumps(extras, ensure_ascii=False, indent=2))
