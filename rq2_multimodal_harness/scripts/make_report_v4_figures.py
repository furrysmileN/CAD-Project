# -*- coding: utf-8 -*-
"""Generate report-v4 figures for the native point-cloud (P_geom) experiment.

Reads analysis CSVs and writes PNGs. Offline and deterministic.
"""
from __future__ import annotations

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
CONFIRM = ROOT / "outputs" / "native_pointcloud_v1" / "confirm_n100" / "analysis"
SCREEN = ROOT / "outputs" / "native_pointcloud_v1" / "analysis"
FIG = CONFIRM / "figures" / "report_v4"
FIG.mkdir(parents=True, exist_ok=True)

LABEL = {
    "I1": "照片",
    "T1I1": "一句话+照片",
    "P_proj": "旧剪影",
    "P_geom_static": "几何说明(静态)",
    "P_geom_tool": "几何说明",
    "I1P_proj": "照片+旧剪影",
    "I1P_geom": "照片+几何说明",
    "T1I1P_proj": "一句话+照片+旧剪影",
    "T1I1P_geom": "一句话+照片+几何说明",
}
COLOR = {
    "I1": "#9aa0a6",
    "T1I1": "#9aa0a6",
    "P_proj": "#e8716a",
    "P_geom_static": "#34a853",
    "P_geom_tool": "#34a853",
    "I1P_proj": "#f4b400",
    "I1P_geom": "#188038",
    "T1I1P_proj": "#f4b400",
    "T1I1P_geom": "#188038",
}

cond = pd.read_csv(CONFIRM / "pc_condition_summary.csv")
paired = pd.read_csv(CONFIRM / "pc_paired_all.csv")
diff = pd.read_csv(CONFIRM / "pc_by_difficulty.csv")
tasks = pd.read_csv(CONFIRM / "pc_task_rows.csv")
screen = pd.read_csv(SCREEN / "pc_condition_summary.csv")
screen_paired = pd.read_csv(SCREEN / "pc_paired_all.csv")

cond["label"] = cond["condition"].map(LABEL)
ORDER = [
    "I1",
    "T1I1",
    "P_proj",
    "I1P_proj",
    "T1I1P_proj",
    "P_geom_tool",
    "I1P_geom",
    "T1I1P_geom",
]


def _ordered(frame: pd.DataFrame, key: str = "condition") -> pd.DataFrame:
    return frame.set_index(key).loc[ORDER].reset_index()


# ----------------------------------------------------------------------
# fig1: 8-condition ranking
# ----------------------------------------------------------------------
d = _ordered(cond)
fig, ax = plt.subplots(figsize=(10, 5.2))
colors = [COLOR[c] for c in d["condition"]]
ax.bar(d["label"], d["mean_joint_quality"], color=colors, alpha=0.9, edgecolor="white")
for i, r in d.iterrows():
    ax.text(i, r["mean_joint_quality"] + 0.012, f"{r['mean_joint_quality']:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, 0.72)
ax.set_ylabel("总分 joint quality（失败记 0）")
ax.set_title("确认实验：8 种给法的平均总分（100 个零件）", fontsize=13)
ax.tick_params(axis="x", rotation=18)
ax.grid(axis="y", alpha=0.25)
fig.savefig(FIG / "v4_fig1_condition_means.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig2: forest of preregistered paired deltas
# ----------------------------------------------------------------------
FOREST = [
    ("P_geom_tool-P_proj", "几何说明 − 旧剪影"),
    ("I1P_geom-I1P_proj", "照片+几何 − 照片+旧剪影"),
    ("I1P_geom-I1", "照片+几何 − 纯照片"),
    ("T1I1P_geom-T1I1", "一句话+照片+几何 − 一句话+照片"),
    ("T1I1P_geom-T1I1P_proj", "一句话+照片+几何 − 一句话+照片+旧剪影"),
    ("I1P_proj-I1", "照片+旧剪影 − 纯照片（负向对照）"),
]
pc = paired.set_index("contrast").loc[[k for k, _ in FOREST]].reset_index()
pc["name"] = [n for _, n in FOREST]
pc = pc.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.2, 4.8))
for i, r in pc.iterrows():
    lo, hi, mu = r["ci95_low"], r["ci95_high"], r["mean_delta_joint_quality"]
    color = "#188038" if lo > 0 else ("#e8716a" if hi < 0 else "#9aa0a6")
    ax.plot([lo, hi], [i, i], color=color, lw=2.4)
    ax.plot(mu, i, "o", color=color, ms=8)
    ax.text(hi + 0.01, i, f"{mu:+.3f}  {int(r['n_left_better'])}/{int(r['n_right_better'])}/{int(r['n_tie'])}", va="center", fontsize=9)
ax.axvline(0, color="#333", lw=1)
ax.set_yticks(range(len(pc)))
ax.set_yticklabels(pc["name"])
ax.set_xlabel("同一零件上的总分差（左更好为正；误差线 = 95% 区间；右侧为 赢/输/平）")
ax.set_title("确认实验：预注册配对对比（n=100，失败记 0，探索性）", fontsize=12)
ax.set_xlim(-0.08, 0.48)
ax.grid(axis="x", alpha=0.25)
fig.savefig(FIG / "v4_fig2_paired_forest.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig3: screen n=20 vs confirm n=100
# ----------------------------------------------------------------------
shared = [c for c in ORDER if c in set(screen["condition"])]
s = screen.set_index("condition").loc[shared]
c = cond.set_index("condition").loc[shared]
x = np.arange(len(shared))
w = 0.38
fig, ax = plt.subplots(figsize=(10.2, 5.2))
ax.bar(x - w / 2, s["mean_joint_quality"], w, label="筛选 20 零件", color="#8ab4f8", edgecolor="white")
ax.bar(x + w / 2, c["mean_joint_quality"], w, label="确认 100 零件", color="#188038", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([LABEL[k] for k in shared], rotation=18)
ax.set_ylabel("总分 joint quality")
ax.set_title("筛选（20）vs 确认（100）：方向一致，确认阶段幅度略收", fontsize=13)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)
ax.set_ylim(0, 0.75)
fig.savefig(FIG / "v4_fig3_screen_vs_confirm.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig4: difficulty
# ----------------------------------------------------------------------
want = ["I1", "P_proj", "P_geom_tool", "I1P_geom", "T1I1P_geom"]
levels = ["easy", "medium", "hard"]
level_zh = {"easy": "简单", "medium": "中等", "hard": "困难"}
fig, ax = plt.subplots(figsize=(9.4, 5.0))
x = np.arange(len(levels))
w = 0.16
for i, cond_id in enumerate(want):
    vals = [
        float(diff[(diff["condition"] == cond_id) & (diff["level"] == lv)]["mean_joint_quality"].iloc[0])
        for lv in levels
    ]
    ax.bar(x + (i - 2) * w, vals, w, label=LABEL[cond_id], color=COLOR[cond_id], alpha=0.92, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([level_zh[lv] for lv in levels])
ax.set_ylabel("总分 joint quality")
ax.set_title("按零件难度分层：三种难度上，几何说明都高于旧剪影和纯照片", fontsize=12)
ax.legend(frameon=False, ncol=3, fontsize=9)
ax.grid(axis="y", alpha=0.25)
ax.set_ylim(0, 0.78)
fig.savefig(FIG / "v4_fig4_difficulty.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig5: success vs quality
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 5.4))
for _, r in cond.iterrows():
    ax.scatter(r["success_rate"] * 100, r["mean_joint_quality"], s=90, color=COLOR[r["condition"]], zorder=3)
    ax.annotate(r["label"], (r["success_rate"] * 100, r["mean_joint_quality"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xlabel("成功率（%）")
ax.set_ylabel("总分 joint quality")
ax.set_title("成功率 vs 总分：本轮几乎都能画出零件，拉开差距的是像不像", fontsize=12)
ax.set_xlim(95.5, 100.8)
ax.set_ylim(0.18, 0.62)
ax.grid(alpha=0.25)
fig.savefig(FIG / "v4_fig5_success_quality.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig6: JQ distribution of key conditions
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 5.2))
bins = np.linspace(0, 1, 21)
for cond_id, alpha in (("P_proj", 0.45), ("P_geom_tool", 0.45), ("T1I1P_geom", 0.55)):
    vals = tasks.loc[tasks["condition"] == cond_id, "joint_quality"].astype(float).fillna(0)
    ax.hist(vals, bins=bins, alpha=alpha, label=LABEL[cond_id], color=COLOR[cond_id], edgecolor="white")
ax.set_xlabel("单个零件的总分")
ax.set_ylabel("零件个数（各 100 个）")
ax.set_title("分数分布：几何说明把一批零件从低分区推进到高分区", fontsize=12)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)
fig.savefig(FIG / "v4_fig6_jq_hist.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig7: cost vs quality
# ----------------------------------------------------------------------
cond["tokens"] = cond["mean_prompt_tokens"] + cond["mean_completion_tokens"]
fig, ax = plt.subplots(figsize=(8.8, 5.4))
for _, r in cond.iterrows():
    ax.scatter(r["tokens"], r["mean_joint_quality"], s=90, color=COLOR[r["condition"]], zorder=3)
    ax.annotate(r["label"], (r["tokens"], r["mean_joint_quality"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xlabel("平均每次请求的 token 数（提示词 + 回复）")
ax.set_ylabel("总分 joint quality")
ax.set_title("成本 vs 质量：几何说明本身不贵；叠照片才明显涨 token", fontsize=12)
ax.grid(alpha=0.25)
ax.set_ylim(0.18, 0.62)
fig.savefig(FIG / "v4_fig7_cost_quality.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig8: literature-aligned geometry for best vs baselines
# ----------------------------------------------------------------------
geom_rows = []
for cond_id in ["I1", "P_proj", "P_geom_tool", "I1P_geom", "T1I1P_geom"]:
    sub = tasks[tasks["condition"] == cond_id]
    ok = sub[sub["completed"].astype(str).str.lower() == "true"]
    geom_rows.append(
        {
            "label": LABEL[cond_id],
            "cd": ok["shape_only_cd"].astype(float).mean(),
            "iou": ok["voxel_iou"].astype(float).mean(),
            "f1": ok["f1_shape"].astype(float).mean(),
        }
    )
g = pd.DataFrame(geom_rows)
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
axes[0].bar(g["label"], g["cd"], color="#e8716a", alpha=0.9)
axes[0].set_title("表面误差 CD（越低越好）")
axes[1].bar(g["label"], g["iou"], color="#188038", alpha=0.9)
axes[1].set_title("体积重合 IoU（越高越好）")
axes[2].bar(g["label"], g["f1"], color="#1a73e8", alpha=0.9)
axes[2].set_title("F1@1%（越高越好，门槛很严）")
for ax in axes:
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("文献更常见的几何口径（仅画成功的零件）", fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "v4_fig8_geometry.png")
plt.close(fig)

print(f"wrote figures to {FIG}")
