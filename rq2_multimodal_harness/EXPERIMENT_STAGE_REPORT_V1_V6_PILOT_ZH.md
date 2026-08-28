# 多模态 CAD 生成阶段性实验报告（V1–V6 pilot）

> **日期**：2026-08-26  
> **性质**：阶段综述。把已经完成的实验链写成一条可核对的叙事，而不是各轮分数汇编。  
> **范围**：V1–V5 已冻结结论 + V6 工程落地与 20×6 live pilot + 第一层对照尸检。V6 全貌（含 V6b）以 `EXPERIMENT_REPORT_V6_ZH.md` 为准。  
> **读者**：组会 / 未跟过每一轮细节的人。  
> **口径**：V6 仅为工程 pilot（n=20），只做描述；不得当作 confirm 的显著性结论。

---

## 0. 现阶段可以怎样讲

整条工作要回答的不是「再多给一种输入会不会更好」，而是：

> 在固定模型、固定 Plan schema、固定 Harness 执行栈的前提下，文本 / 图像 / 点云派生几何证据是否提供了**彼此缺失的 CAD 信息**。

到 V5 为止，在自然零件上已经可以支持：**执行栈稳定之后，图像与 point-cloud-derived structured geometric evidence 双向互补，但不对称；弱文本时点云一侧的边际更大。** 点云始终是本地结构化证据，不是模型直接消费的 embedding。

V6 把问题收窄为因果机制：**增益来自独有 CAD 事实，还是来自更长输入、重复表达或结构化模板？** 20×6 live 之后，前半句有描述性支持，后半句**尚未被有效检验**——不是因为 H2/H3 已被证伪，而是因为对照字段在多数样本上并未构成可用的信息操纵。

**允许的表述**

- V1–V5：双向但不对称的模态互补；轮廓投影不是点云；C 臂之后比较的是几何质量。
- V6 pilot：弱文本下，图像与点云证据仍表现为双向边际增益；C 臂交付率已饱和。
- V6 尸检：K3/K4（独有事实 vs 重复 / 正确 vs 局部错误）在本轮大多不是有效对照。

**不允许的表述**

- 通义千问直接理解了原生点云 token 或 embedding；
- V6 已经证明「互补来自各自独有的 CAD 事实」；
- 已经用 Holm 校正对外宣称 V6 显著性；
- 交互工具、字段消融、`P_enc`、换模型已经完成。

---

## 1. 固定流水线

后续所有轮次操纵的是输入表征与事实集合，而不是更换评分口径。

```text
零件观察（文本 / RGB 图像 / 点云派生结构化几何证据）
    → 通义千问输出受约束的 HarnessCAD Plan JSON
    → C 臂：schema 校验 + 确定性修复 R4 + 至多 2 轮 schema/execution 回馈
    → 与真值 STEP 对齐评分；失败记 0
```

- 模型：V1 为 `qwen3.7-plus`；其后为 `qwen3.8-max`（DashScope compatible-mode）。
- 主几何指标：failure-aware `joint_quality`（越大越好）。几何 endpoint 另报 `common_frame_cd`（越小越好）。
- 点云对外表述：仅允许 **point-cloud-derived structured geometric evidence**。

---

## 2. V1–V5：已经收窄到哪一步

```text
V1  模态组合是否优于单模态？
    → 组合增益为负。瓶颈是计划不合规，不是「看不见形状」。

V2/V3  是否编码方式不当？
    → 主导因素是文本细化。复杂视觉编码无稳定增益。失败仍大量来自 schema。

C 臂  先使执行栈饱和
    → 100×7 成功率约 97.6%。此后只比较几何质量。

V4  点云表征是否得当？
    → 本地几何描述远优于轮廓投影。增益来自几何，而非成功率。

V5  独立 100 + 预注册双向互补
    → 双向成立、不对称。弱文本时描述边际更大。
       轮廓投影对照、错配描述对照均更差。Phase B/D 仅 dry-run。
```

V5 确认矩阵的可引用结果（独立 100，核心条件 3 次重复，Holm）：图像叠加完整几何描述相对纯图像约 **+0.26**，相对纯描述约 **+0.06**；错配描述明显更差。细节以 `EXPERIMENT_REPORT_V5_ZH.md` 为准。

V5 留下的机制问题正是 V6 的入口：**整包描述有效，是否等于「新增模态带入了图像缺失的那条 CAD 事实」？**

各轮正式报告与落盘索引见文末附录。V1–V5 叙事以 `EXPERIMENT_ARC_V1_V5_ZH.md` 为冻结稿，本文件不改写其判定。

---

## 3. V6 研究设计（相对 V5 改了什么）

V5 的自变量是「模态有无」。V6 的自变量是「每个模态携带哪些 CAD 事实」。

样本改为 Harness DSL 可完整表达的程序化零件，五个 family：`plate_holes`、`flange_array`、`stepped_shaft`、`bracket_back_feature`、`block_pocket_slot`。`flange_array` 限矩形 2×2 孔阵，不扩展 DSL。

六个条件全部带相同弱文本 **T0**（不含尺寸、孔数、深度、family 名）：

| 对模型 ID | 分析侧语义 | 输入 |
|---|---|---|
| C0 | C0_BASE | T0 |
| C1 | C1_IMAGE | T0 + 四视图 |
| C2 | C2_POINT | T0 + `P_comp` |
| C3 | C3_COMPLEMENT | T0 + 四视图 + `P_comp` |
| C4 | C4_REPEAT | T0 + 四视图 + `P_repeat` |
| C5 | C5_WRONG | T0 + 四视图 + `P_wrong` |

- `P_comp`：仅从点云测出的完整结构化证据（不是 V5 的投影图 `P_proj`）。
- `P_repeat`：同 schema，去掉 primary critical fact，用可见表面冗余测量补 token（不是 shuffle）。
- `P_wrong`：按 sample hash 只改一个预注册字段（不是 `I1P_shuffle`）。

预注册假设：H1a C3>C1，H1b C3>C2，H2 C3>C4，H3 C3>C5，H4 关键 feature 与缺口同向。主比较 K1–K4；K5 为 K3 的解释性重述；K6 为协同项 `C3−C1−C2+C0`。

双 endpoint，且 `keep_best=false`：

- **`first_attempt`**：第一次 API → 确定性 R4 → 首次执行评分（主结论）。
- **`final_delivery`**：C 臂最多 2 轮 schema/execution 回馈后的最终交付。

---

## 4. V6 已经完成的工程，而不是「又跑了一次 V5」

输出根目录：`outputs/v6_information_complementarity/`。未改 V5 的 `pc_conditions.py` / `feedback.py` 语义，也未覆盖 V5 state。

| 步骤 | 状态 | 要点 |
|---|---|---|
| 计划与预注册落仓 | 完成 | `V6_INFORMATION_COMPLEMENTARITY_EXPERIMENT_PLAN_ZH.md`，`preregistration.md` |
| Phase 0 迁移审计 | Go | 本机重放 50 个 V5 冻结 Plan，status / 拓扑 / 几何 50/50 |
| 程序化数据集 | 完成 | pilot 20 与 confirm 100，参数组合无重叠；每样本含 latent、GT STEP/STL、四视图、点云、三种 evidence |
| payload / 泄漏审计 | 完成 | 条件 ID 仅 C0–C5；禁止 latent、GT Plan、路径、语义条件名进入 prompt |
| Dry-run | 完成 | 20×6=120，全部 `dry_run_completed` |
| Live pilot | 完成 | 20×6×1=120，全部 `completed`；模型 `qwen3.8-max` |
| 第一层对照核对 | 完成 | 见第 7 节 |

Live 启动前发现两处会把整轮烧成 schema 失败的实现问题，已在正式 120 之前修正，不进入主结论解释：

1. C 臂必须在 schema 未过时仍把已解析的 Plan 交给 R4，再校验（与 V5 一致）；不得因 `parse_plan_response.ok=false` 丢弃对象。
2. 用户提示须包含与 V5 相同的 Plan JSON 模板。缺模板时模型系统性漏写 `coordinate_system`。

V6 状态目录与 dry-run 分离：`pilot/live/state/`。中途曾有旧进程写回旧 fingerprint，已终止后按 fingerprint 续跑覆盖；最终 120/120 fingerprint 一致。

---

## 5. V6 live 20×6：描述结果

主 endpoint：`first_attempt` 的 `joint_quality`。n=20，1 次重复。**不做显著性宣称。**

### 5.1 条件均值

| 条件 | 输入 | 首次 jq | 最终 jq | 首次几何有效 | 首次 schema |
|---|---|---:|---:|---:|---:|
| C0 | T0 | 0.247 | 0.345 | 15/20 | 20/20 |
| C1 | T0+I | 0.552 | 0.614 | 18/20 | 20/20 |
| C2 | T0+P_comp | 0.738 | 0.738 | 20/20 | 20/20 |
| C3 | T0+I+P_comp | 0.834 | 0.834 | 20/20 | 20/20 |
| C4 | T0+I+P_repeat | 0.837 | 0.837 | 20/20 | 20/20 |
| C5 | T0+I+P_wrong | 0.834 | 0.834 | 20/20 | 20/20 |

C 臂几乎已经无事可做：113/120 第一次即完成，仅 7 次用了 1 轮回馈，且主要抬的是 C0/C1。C2–C5 的首次与最终完全相同。此后差异是几何质量，不是「能否出合法 Plan」。

### 5.2 预注册对比（只报方向与赢次）

| 对比 | 含义 | 均值差 | 赢 / 输 / 平 |
|---|---|---:|---|
| K1 C3−C1 | 点云证据叠在图像上 | +0.282 | 19 / 1 / 0 |
| K2 C3−C2 | 图像叠在点云证据上 | +0.096 | 17 / 0 / 3 |
| K3 C3−C4 | 独有事实 vs 重复测量 | −0.003 | 4 / 6 / 10 |
| K4 C3−C5 | 正确 vs 局部错误事实 | +0.000 | 3 / 7 / 10 |
| K6 C3−C1−C2+C0 | 协同项 | −0.209 | 5 / 15 / 0 |

几何 CD 与 K1 同向：C3 相对 C1 的 CD 更小（描述性改善约 0.047）。特征 exact 在 C1–C5 均为 5/20，H4 未见条件间分化。

按 family 的首次 jq 均值：

| family | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| 孔板 `plate_holes` | 0.22 | 0.70 | 0.91 | 0.92 |
| 法兰孔阵 `flange_array` | 0.35 | 0.36 | 0.68 | 0.79 |
| 阶梯轴 `stepped_shaft` | 0.18 | 0.56 | 0.42 | 0.75 |
| 背面支架 `bracket_back_feature` | 0.29 | 0.62 | 0.85 | 0.86 |
| 口袋块 `block_pocket_slot` | 0.20 | 0.52 | 0.84 | 0.85 |

法兰几乎全靠点云证据（C1≈C0）；阶梯轴是图像与点云真正分开的一族（C2 单独甚至低于 C1，合在一起最好）。口袋与支架在 C2 已接近饱和，C3 再叠加图像的几何增益很小。

### 5.3 对 H1–H4 的阶段判断

| 假设 | pilot 描述判断 | 说明 |
|---|---|---|
| H1a C3>C1 | 方向清楚 | 19/20 支持；与 V5「点云补图像」同向 |
| H1b C3>C2 | 方向清楚、幅度较小 | 17 更好 / 3 平；与 V5「图像补点云较弱」同向 |
| H2 C3>C4 | **未能检验** | 均值打平，半数样本几何分数完全相同 |
| H3 C3>C5 | **未能检验** | 关键 feature 预测 19/20 与 C3 相同 |
| H4 feature 同向 | **未见** | exact 不随 C3/C4/C5 变化 |
| K6 协同 >0 | 描述为负 | 次加性：合在一起好，但小于两路增益之和 |

一句话：**pilot 支持「图像和点云证据互补」；还不支持「互补来自各自独有的 CAD 事实」。**

---

## 6. 第一层核对：为什么 K3/K4 是空结果

原工程审计（有没有 `primary_critical`、P_repeat 是否删掉该 `fact_id`、P_wrong 是否只改这一条）**20/20 通过**。那只保证「改了 JSON 里的一个标记」，不保证「改到了模型可能使用、且等于 GT 缺口的测量」。

科学审计结果：

1. **关键事实测准率 5/20。** 法兰间距 4/4 为 `null`（`point_cloud_unresolved`）；口袋深度 4/4 被标成 bbox 最长边 `1.0`；背面特征 4/4 测成 `false`（GT 为 `true`）。对这些样本做 C4/C5，改的不是 GT 缺口。
2. **轴线负对照有碰撞。** 0017 的 `P_wrong` 随机替换撞回 `[0,0,1]`，C5 作废。
3. **真正合格的 K3/K4 对照只有 4 例**（0000、0010 通/盲孔；0002、0007 轴线）。这 4 例上 C5 的关键预测**全部仍等于 C3**：盲孔没有跟随 `through`，轴线没有跟随 X/Y。
4. 几何上 C3 与 C4 的 `joint_quality` 有 10/20 完全相同；C5 与 C3 的 feature 预测 19/20 相同。

因此 C3≈C4≈C5 **首先应解释为对照失效**，不能直接写成「模型不使用点云证据」。K1/K2 已经表明模型在使用**整包**点云证据和图像；它没有使用的是被标记为 `primary_critical` 的那一个字段。

明细：`outputs/v6_information_complementarity/pilot/live/analysis/layer1_audit.json`。

---

## 7. 现阶段总表

| 轮次 | 研究问题 | 规模 | 主结论 | 遗留 |
|---|---|---|---|---|
| V1 | 组合是否优于单模态 | 100×7 | 组合为负；瓶颈在 schema | 须稳定执行栈 |
| V2/V3 | 是否编码不当 | 20×63 | 文本细化主导 | 执行栈仍须固定 |
| C 臂 | 能否稳定导出 | 100×7 | 成功率约 97.6% | 此后比几何 |
| V4 | 点云表征 | 100×8 等 | 描述 ≫ 投影 | 缺独立样本与正式推断 |
| V5 | 是否真互补 | 新 100×(8×3+2) | 双向、不对称；弱文本时描述边际大 | 字段因果未测；工具未 live |
| V6 pilot | 增益是否来自独有事实 | 20×6 | H1 描述成立；H2/H3 对照未生效 | 须先修测量与操纵，再 confirm |

---

## 8. 已冻结的下一步（V6b）

阶段报告第 8 节的旧建议**仅供审阅，未执行**。2026-08-26 的决定如下。

**暂停 100×6×3 confirm。** 现有 20×6 live 冻结为仪器失效诊断轮，不进正式推断。H1 已有 V5 与本 diagnostic 支持，不再用 1800 任务重复确认。K6<0 不修复。

真正未回答的问题改为：

> 当图像无法区分两个 CAD 反事实，而点云证据只在一个关键事实处发生自洽变化时，模型生成的 CAD 是否随该事实变化？

落地顺序（API 越往后越少）：

1. 完全离线四层审计 + Oracle：**评分 / Plan 已通过**；测量层（evidence v2）**92/100**。
2. 最小反事实零件对已生成：`pilot_v2_minimal_pairs/`，12 对中 **11 对离线合格**，覆盖四类事实。C5 为配对点云的自洽 `P_B`，不是手工改字段。
3. Evidence readback：**P_full 11/11，P_counterfactual 跟随 11/11**（门槛 ≥80%，通过）。非正式主实验；未覆盖 `pilot/live/`。
4. V6b 探针 11×6×2=132 已完成。C3/C5 Plan 从不相同；**严格跟随 9/22**（口袋 6/6，其余弱）。不开 confirm。
5. 仪器修复（2026-08-28）：评分 `rq2.v6.feature.v3` + 配对 `rq2.v6b.pair.v2`（B 为非默认值）。新对在 `pilot_v2_instrument_fix/`，不覆盖旧探针。confirm100 Oracle 仍 100/100。修复后探针严格跟随 **8/22**（口袋 6/6；通/盲与背面 0）。不开 confirm。
6. 门槛通过后再开 `confirm_v2_minimal_pairs/`。

修订文件：`V6B_MINIMAL_COUNTERFACTUAL_PAIRS_PLAN_ZH.md`，`outputs/v6_information_complementarity/preregistration_amendment_v6b.md`。

---

## 附录. 报告与产物索引

| 版本 | 文件 | 落盘 |
|---|---|---|
| V1 | `EXPERIMENT_REPORT_ZH.md` | `outputs/pilot_v2/` |
| V2 / V3 | `EXPERIMENT_REPORT_V2_ZH.md` / `_V3_` | `outputs/encoding_screen_n20/` |
| C 臂 | 写入 V3/V4 叙述 | `outputs/confirm_n100/arms/C/` |
| V4 | `EXPERIMENT_REPORT_V4_ZH.md` | `outputs/native_pointcloud_v1/` |
| V5 | `EXPERIMENT_REPORT_V5_ZH.md` | `outputs/v5_complementarity/` |
| V1–V5 综述 | `EXPERIMENT_ARC_V1_V5_ZH.md` | 以上全部 |
| V6 正式报告 | `EXPERIMENT_REPORT_V6_ZH.md` | `outputs/v6_information_complementarity/`（pilot + V6b） |
| V6 计划 / 预注册 | `V6_INFORMATION_COMPLEMENTARITY_EXPERIMENT_PLAN_ZH.md`，`outputs/v6_information_complementarity/preregistration.md` | 同目录 |
| V6 live | 本文件第 5–6 节 | `outputs/v6_information_complementarity/pilot/live/` |
| V6 描述统计 | | `pilot/live/analysis/descriptive.json` |
| V6 对照尸检 | | `pilot/live/analysis/layer1_audit.json` |
| V6 诊断轮冻结 | `outputs/v6_information_complementarity/PILOT_FROZEN_AS_INSTRUMENT_DIAGNOSTIC_ZH.md` | `pilot/live/` |
| V6b 计划 / 修订 | `V6B_MINIMAL_COUNTERFACTUAL_PAIRS_PLAN_ZH.md`，`preregistration_amendment_v6b.md` | 不覆盖原 `preregistration.md` |
| V6b 最小对 | | `outputs/v6_information_complementarity/pilot_v2_minimal_pairs/` |
| V6b readback | | `pilot_v2_minimal_pairs/readback/live/` |
| V6b 探针 | | `pilot_v2_minimal_pairs/probe/live/` |
| V6b 仪器修复探针 | `configs/v6b_probe_fix.yaml` | `pilot_v2_instrument_fix/` |

V5 的 `repeats/run_summary.json` 仍可能只是最后一次续跑计数，引用 V5 全量时以 state 与 `analysis/v5_phase_c_live_report.json` 为准。
