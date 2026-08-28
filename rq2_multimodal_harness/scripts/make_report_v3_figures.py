# -*- coding: utf-8 -*-
"""Generate report-v3 figures for the encoding screen experiment (20 samples x 63 conditions).

Reads the analysis CSVs produced by analyze.py and writes PNG charts plus a
summary JSON used by the report. Pure offline, deterministic.
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

cond = pd.read_csv(BASE / "encoding_condition_summary.csv")
inter = pd.read_csv(BASE / "encoding_interactions.csv")
fail = pd.read_csv(BASE / "encoding_failure_summary.csv")
cost = pd.read_csv(BASE / "encoding_cost_summary.csv")
tasks = pd.read_csv(BASE / "encoding_task_rows.csv")

PILOT = {  # previous round (pilot_v2, 100 samples x 7 conditions), from pilot_v2/analysis/report_zh.md
    "T2":  ("T", 0.5800, 0.2496),
    "I1":  ("I", 0.6300, 0.1763),
    "P3":  ("P", 0.8000, 0.2132),
    "T2I1": ("TI", 0.5600, 0.2382),
    "T2P3": ("TP", 0.5500, 0.2644),
    "I1P3": ("IP", 0.6700, 0.2005),
    "T2I1P3": ("TIP", 0.6000, 0.2649),
}

COLOR_TEXT = {"none": "#9aa0a6", "T1": "#e8716a", "T2": "#34a853", "T3": "#f4b400"}
COLOR_MOD = {"T": "#4285f4", "I": "#ea4335", "P": "#34a853"}
COND_ORDER = cond.sort_values("joint_quality_mean", ascending=False)["condition"].tolist()


def text_level(c: str) -> str:
    if c.startswith("T3"):
        return "T3"
    if c.startswith("T2"):
        return "T2"
    if c.startswith("T1"):
        return "T1"
    return "none"


def n_visual(c: str) -> int:
    return sum(1 for ch in ("I", "P") if ch in c)


summary: dict = {}

# ----------------------------------------------------------------------
# fig1: marginal means per encoding (T1..P3)
# ----------------------------------------------------------------------
mm = inter[inter["row_type"] == "marginal_mean"].copy()
mm["encoding"] = mm["contrast"]
order = ["T1", "T2", "T3", "I1", "I2", "I3", "P1", "P2", "P3"]
mm = mm.set_index("encoding").loc[order].reset_index()
mm["mod"] = mm["encoding"].str[0]

fig, ax = plt.subplots(figsize=(9, 5))
colors = [COLOR_MOD[m] for m in mm["mod"]]
ax.bar(mm["encoding"], mm["mean"], yerr=[mm["mean"] - mm["ci_low"], mm["ci_high"] - mm["mean"]],
       color=colors, capsize=4, alpha=0.88, edgecolor="white")
for i, r in mm.iterrows():
    ax.text(i, r["mean"] + 0.016, f"{r['mean']:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, 0.62)
ax.set_ylabel("joint quality 边际均值（失败记 0 的总分）")
ax.set_title("问题一：同一模态用哪种“表示形式”最好？\n（固定该编码、对其他模态取平均后的总分，误差线 = 95% bootstrap 区间）", fontsize=12)
ax.grid(axis="y", alpha=0.25)
for x in (2.5, 5.5):
    ax.axvline(x, color="#dddddd", lw=1)
handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_MOD[m]) for m in ("T", "I", "P")]
ax.legend(handles, ["文本 Text", "图像 Image", "点云 Point"], loc="upper left", frameon=False, fontsize=10)
fig.savefig(FIG / "v3_fig1_marginal_means.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig2: full ranking of all 63 conditions
# ----------------------------------------------------------------------
d = cond.sort_values("joint_quality_mean")
fig, ax = plt.subplots(figsize=(10, 12.5))
colors = [COLOR_TEXT[text_level(c)] for c in d["condition"]]
bars = ax.barh(d["condition"], d["joint_quality_mean"], color=colors, alpha=0.9)
for i, r in d.iterrows():
    ax.text(r["joint_quality_mean"] + 0.004, i, f"{r['joint_quality_mean']:.3f}", va="center", fontsize=8)
ax.set_xlabel("joint quality 均值（0~1，失败任务记 0）")
ax.set_title("63 种信息组合大排名（20 个零件的平均总分）", fontsize=13)
ax.grid(axis="x", alpha=0.25)
ax.set_xlim(0, 0.55)
ax.axvline(cond["joint_quality_mean"].mean(), color="gray", ls="--", lw=1)
ax.text(cond["joint_quality_mean"].mean() + 0.004, 61.5, "63 条件均值", fontsize=8, color="gray")
handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_TEXT[k]) for k in ("T2", "T3", "T1", "none")]
ax.legend(handles, ["含 T2 详细文本", "含 T3 结构化文本", "含 T1 一句话文本", "无文本（纯视觉）"],
          loc="lower right", frameon=False, fontsize=10)
fig.savefig(FIG / "v3_fig2_ranking.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig3: forest plot of the 9 preregistered single-modality paired comparisons
# ----------------------------------------------------------------------
pc = inter[inter["comparison_family"] == "single_modality_encoding"].copy()
pc = pc.sort_values("mean", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for i, r in pc.iterrows():
    sig = r["ci_low"] > 0 or r["ci_high"] < 0
    color = "#1a73e8" if sig else "#9aa0a6"
    ax.errorbar(r["mean"], i, xerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]],
                fmt="o", color=color, capsize=4, ms=7)
    ax.text(0.30, i, f"赢/平/输 = {int(r['wins'])}/{int(r['ties'])}/{int(r['losses'])}",
            va="center", fontsize=8.5, color="#555555")
ax.axvline(0, color="#333333", lw=1)
ax.set_yticks(range(len(pc)))
ax.set_yticklabels(pc["contrast"], fontsize=10)
ax.invert_yaxis()
ax.set_xlim(-0.28, 0.42)
ax.set_xlabel("同一零件上的配对总分差（>0 = 前者更好）")
ax.set_title("问题一·补充：同一种模态内部两两对决（带 95% CI）\n蓝点 = 置信区间不含 0（差异方向可信）；灰点 = 未分开", fontsize=12)
ax.grid(axis="x", alpha=0.25)
fig.savefig(FIG / "v3_fig3_paired_forest.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig4: success rate vs quality scatter
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6.2))
xmed, ymed = cond["execution_success_rate"].median(), cond["joint_quality_mean"].median()
ax.axvline(xmed, color="#bbbbbb", ls="--", lw=1)
ax.axhline(ymed, color="#bbbbbb", ls="--", lw=1)
for level in ("none", "T1", "T2", "T3"):
    sub = cond[[text_level(c) == level for c in cond["condition"]]]
    ax.scatter(sub["execution_success_rate"], sub["joint_quality_mean"], s=55,
               color=COLOR_TEXT[level], alpha=0.85, edgecolor="white", label={"none": "无文本", "T1": "T1 一句话", "T2": "T2 详细", "T3": "T3 结构化"}[level])
annot = {
    "T2I2P1": (0.75, 0.4463, "最优组合"),
    "T2": (0.75, 0.3699, "纯文本"),
    "I1": (0.85, 0.3120, "纯图像"),
    "P2": (0.95, 0.2095, "纯点云"),
    "T1": (0.60, 0.1564, "弱文本"),
    "T1P1": (0.65, 0.1007, "垫底"),
    "T3I1": (0.45, 0.2651, "成功率最低"),
}
for c, (x, y, label) in annot.items():
    ax.annotate(f"{c} {label}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7))
ax.text(0.99, 0.97, "左上：质量高但常跑不通", transform=ax.transAxes, ha="right", fontsize=9, color="#555555")
ax.text(0.99, 0.035, "右下：跑得通但模型简单", transform=ax.transAxes, ha="right", fontsize=9, color="#555555")
ax.text(0.012, 0.035, "左下：又差又常失败", transform=ax.transAxes, ha="left", fontsize=9, color="#555555")
ax.text(0.012, 0.97, "右上：既好又稳（理想）", transform=ax.transAxes, ha="left", fontsize=9, color="#555555")
ax.set_xlabel("执行成功率（20 个零件里真正建模成功的比例）")
ax.set_ylabel("joint quality 总分均值")
ax.set_title("问题三：质量和成功率如何权衡？\n（每个点 = 一种信息组合；虚线 = 63 条件的中位数）", fontsize=12)
ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.72))
fig.savefig(FIG / "v3_fig4_success_quality.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig5: previous round (pilot_v2) vs this round (screen) on 7 shared conditions
# ----------------------------------------------------------------------
conds7 = ["T2", "I1", "P3", "T2I1", "T2P3", "I1P3", "T2I1P3"]
labels7 = ["T", "I", "P", "T+I", "T+P", "I+P", "T+I+P"]
pilot_q = [PILOT[c][2] for c in conds7]
pilot_s = [PILOT[c][1] for c in conds7]
screen = cond.set_index("condition")
screen_q = [screen.loc[c, "joint_quality_mean"] for c in conds7]
screen_s = [screen.loc[c, "execution_success_rate"] for c in conds7]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
x = np.arange(len(conds7))
w = 0.38
axes[0].bar(x - w / 2, pilot_q, w, label="上一轮 100 样本\n(qwen3.7-plus)", color="#b0bec5")
axes[0].bar(x + w / 2, screen_q, w, label="本轮 20 样本筛选\n(qwen3.8-max)", color="#1a73e8")
axes[0].set_xticks(x); axes[0].set_xticklabels(labels7)
axes[0].set_ylabel("joint quality 均值"); axes[0].set_title("质量：几乎全面上升")
axes[0].grid(axis="y", alpha=0.25); axes[0].legend(fontsize=8.5, frameon=False)
axes[1].bar(x - w / 2, [v * 100 for v in pilot_s], w, label="上一轮", color="#b0bec5")
axes[1].bar(x + w / 2, [v * 100 for v in screen_s], w, label="本轮", color="#1a73e8")
axes[1].set_xticks(x); axes[1].set_xticklabels(labels7)
axes[1].set_ylabel("执行成功率 (%)"); axes[1].set_title("成功率：7 个条件全部上升")
axes[1].grid(axis="y", alpha=0.25); axes[1].legend(fontsize=8.5, frameon=False)
fig.suptitle("问题四：和上一轮相比，同样的 7 种信息组合进步了吗？\n（注意：两轮模型和样本都不同，只能看趋势）", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(FIG / "v3_fig5_pilot_vs_screen.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig6: failure composition
# ----------------------------------------------------------------------
status = tasks["status"].value_counts()
summary["status_counts"] = status.to_dict()

stage_of = {
    "invalid_json": "解析失败（JSON 语法）",
    "parse_failed": "解析失败（JSON 语法）",
}
def fail_label(row):
    code = row["failure_code"]
    if pd.isna(code):
        return "成功"
    if code in ("invalid_json", "parse_failed"):
        return "解析失败（JSON 语法）"
    if code == "invalid_step_shape":
        return "几何无效（STEP 形状）"
    if code == "plan_validation_failed":
        return "Schema 校验失败（格式不合规）"
    return "运行时异常（操作执行崩溃）"

tasks["flabel"] = tasks.apply(fail_label, axis=1)
stage_order = ["成功", "Schema 校验失败（格式不合规）", "运行时异常（操作执行崩溃）",
               "解析失败（JSON 语法）", "几何无效（STEP 形状）"]
stage_counts = tasks["flabel"].value_counts().reindex(stage_order).fillna(0).astype(int)
summary["stage_counts"] = stage_counts.to_dict()

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1, 1.35]})
colors_stage = {"成功": "#34a853", "Schema 校验失败（格式不合规）": "#e8716a",
                "运行时异常（操作执行崩溃）": "#f4b400", "解析失败（JSON 语法）": "#9aa0a6",
                "几何无效（STEP 形状）": "#7c4dff"}
axes[0].pie(stage_counts.values, labels=[f"{k}\n{v} ({v/1260:.0%})" for k, v in stage_counts.items()],
            colors=[colors_stage[k] for k in stage_counts.index], startangle=90, counterclock=False,
            textprops={"fontsize": 8.5}, wedgeprops={"edgecolor": "white", "linewidth": 1})
axes[0].set_title("全部 1260 次任务的最终去向")

by_t = tasks.copy()
by_t["tlevel"] = by_t["condition"].map(text_level)
stack = by_t.groupby(["tlevel", "flabel"]).size().unstack(fill_value=0).reindex(columns=stage_order, fill_value=0)
torder = ["none", "T1", "T2", "T3"]
stack = stack.reindex(torder)
labels_t = {"none": "无文本", "T1": "T1 一句话", "T2": "T2 详细", "T3": "T3 结构化"}
bottom = np.zeros(len(stack))
for k in stage_order:
    if k == "成功":
        continue
    axes[1].bar([labels_t[t] for t in stack.index], stack[k], bottom=bottom,
                color=colors_stage[k], label=k.replace("（", "\n（"), edgecolor="white")
    bottom += stack[k].values
axes[1].set_ylabel("任务数")
axes[1].set_title("按文本强弱分组的失败原因")
axes[1].legend(fontsize=7.5, frameon=False)
axes[1].grid(axis="y", alpha=0.25)
fig.suptitle("问题五：失败的 377 次任务都死在哪一步？", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(FIG / "v3_fig6_failure_composition.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig7: cost (tokens) vs quality
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.8))
for nv, color in ((0, "#4285f4"), (1, "#f4b400"), (2, "#ea4335")):
    sub = cost[[n_visual(c) == nv for c in cost["condition"]]].merge(
        cond[["condition", "joint_quality_mean"]], on="condition")
    ax.scatter(sub["total_tokens_mean"], sub["joint_quality_mean"], s=52, color=color,
               alpha=0.85, edgecolor="white", label=f"{nv} 个视觉模态")
for c, dx, dy in (("T2", 6, 0.012), ("T2I2P1", -66, -0.030), ("I1", 4, 0.010), ("P1", 6, 0.008), ("T1", 6, 0.008)):
    row = cost.set_index("condition").loc[c]
    q = cond.set_index("condition").loc[c, "joint_quality_mean"]
    ax.annotate(c, (row["total_tokens_mean"], q), textcoords="offset points", xytext=(dx, dy), fontsize=9,
                fontweight="bold" if c == "T2I2P1" else "normal")
ax.set_xlabel("平均每次请求的总 token 数（越大越贵）")
ax.set_ylabel("joint quality 总分均值")
ax.set_title("问题六：多花的钱（token）换来了质量吗？\n（每张图 ≈ 500+ token；三模态约是纯文本的 2.5 倍）", fontsize=12)
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.grid(alpha=0.25)
fig.savefig(FIG / "v3_fig7_cost_quality.png")
plt.close(fig)

# ----------------------------------------------------------------------
# fig8: heatmap of the full factorial (rows = text, cols = image x point)
# ----------------------------------------------------------------------
rows_t = [("none", "无文本"), ("T1", "T1 一句话"), ("T2", "T2 详细"), ("T3", "T3 结构化")]
cols_v = ["无视觉", "I1", "I2", "I3"]
cols_v += [f"I{i}P{j}" for i in "123" for j in "123"]
conds_by_cell = {}
for _, r in cond.iterrows():
    c = r["condition"]
    t = text_level(c)
    if c.startswith(t):
        rest = c[len(t):]
    else:
        rest = c
    if t == "none":
        rest = c
    key = (t, rest if rest else "无视觉")
    conds_by_cell[key] = r["joint_quality_mean"]

mat = np.full((4, 13), np.nan)
for i, (tk, _) in enumerate(rows_t):
    for j, v in enumerate(cols_v):
        if v == "无视觉":
            cid = tk
        else:
            cid = (tk + v) if tk != "none" else v
        if cid in cond.set_index("condition").index:
            mat[i, j] = cond.set_index("condition").loc[cid, "joint_quality_mean"]

fig, ax = plt.subplots(figsize=(11.5, 4.6))
im = ax.imshow(mat, cmap="RdYlGn", vmin=0.05, vmax=0.50, aspect="auto")
ax.set_xticks(range(13)); ax.set_xticklabels(cols_v, fontsize=8.5)
ax.set_yticks(range(4)); ax.set_yticklabels([r[1] for r in rows_t], fontsize=10)
for i in range(4):
    for j in range(13):
        v = mat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="black" if 0.15 < v < 0.40 else "white")
ax.set_title("63 个格子的总分热力图：行 = 文本强弱，列 = 图像 × 点云组合\n（绿 = 好，红 = 差；最后一列是纯文本基线）", fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.9, label="joint quality")
fig.savefig(FIG / "v3_fig8_heatmap.png")
plt.close(fig)

# ----------------------------------------------------------------------
# summary numbers for the report
# ----------------------------------------------------------------------
summary["n_tasks"] = int(len(tasks))
summary["n_completed"] = int((tasks["status"] == "completed").sum())
summary["overall_success_rate"] = float((tasks["status"] == "completed").mean())
summary["mean_joint_quality_all_tasks"] = float(tasks["joint_quality"].mean())
summary["total_tokens"] = int(tasks["total_tokens"].sum())
summary["mean_total_tokens"] = float(tasks["total_tokens"].mean())
summary["mean_latency_sec"] = float(tasks["latency_sec"].mean())
summary["n_samples"] = 20
summary["n_conditions"] = 63
summary["failure_code_counts"] = fail.assign(key=lambda d: d["stage"] + "/" + d["failure_code"]).groupby("key")["count"].sum().astype(int).to_dict()
summary["validation_fail_total"] = int(fail[fail["failure_code"] == "plan_validation_failed"]["count"].sum())
summary["operation_exception_total"] = int(fail[fail["failure_code"] == "operation_exception"]["count"].sum())
summary["invalid_shape_total"] = int(fail[fail["failure_code"] == "invalid_shape_after_operation"]["count"].sum())
summary["empty_total"] = int(fail[fail["failure_code"] == "empty_after_operation"]["count"].sum())
summary["invalid_step_total"] = int(fail[fail["failure_code"] == "invalid_step_shape"]["count"].sum())
summary["parse_total"] = int(fail[fail["stage"] == "parse"]["count"].sum()) + int(fail[fail["stage"] == "schema"]["count"].sum())
summary["top10"] = cond.sort_values("joint_quality_mean", ascending=False).head(10)[["condition", "joint_quality_mean", "execution_success_rate"]].to_dict("records")
summary["bottom10"] = cond.sort_values("joint_quality_mean").head(10)[["condition", "joint_quality_mean", "execution_success_rate"]].to_dict("records")
summary["margin"] = mm[["encoding", "mod", "mean", "ci_low", "ci_high"]].to_dict("records")
summary["paired9"] = pc[["contrast", "mean", "ci_low", "ci_high", "wins", "ties", "losses"]].to_dict("records")
# geometric metrics of the shared 7 conditions
summary["geom7"] = screen.loc[conds7, ["shape_only_cd_mean", "common_frame_cd_mean", "voxel_iou_mean"]].reset_index().to_dict("records")
# difficulty stratification of success
strat = tasks.groupby("difficulty").agg(success=("execution_success", "mean"), quality=("joint_quality", "mean"), n=("sample_id", "size"))
summary["difficulty_strat"] = strat.reset_index().to_dict("records")

with open(BASE / "report_v3_summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)

print("figures written to", FIG)
print(json.dumps(summary, ensure_ascii=False, indent=2, default=str)[:3000])
