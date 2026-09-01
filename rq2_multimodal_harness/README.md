# RQ2：100 样本多模态 Harness 实验

> **给后续 agent**：先读同目录 [`AGENT_HANDOFF_REPORT.md`](AGENT_HANDOFF_REPORT.md)
> （流水线、四轮实验、指标公式、命令、权威数字路径、已知陷阱）。不要使用
> `cad_multimodal_harness_core/`。
>
> **给机外模型做方法评估**：复制 [`EXTERNAL_EVALUATION_BRIEF.md`](EXTERNAL_EVALUATION_BRIEF.md)
> （实验思路、代码分层、实际做了什么、可反驳的结论与局限；不依赖本机路径）。

本目录实现独立、可断点续跑的 `T/I/P/TI/TP/IP/TIP` 七条件实验。它复用了 RQ1 的
确定性抽样、固定视图、单位 bbox 和 failure-aware 评分思想，但 `rq2_harness` 不依赖
`experiments/rq1_context_scaling` 的导入路径。

## 实验边界

- 母池是 BenchCAD 前 1000 个已对齐样本，以点云元数据的 `sample_idx/stem` 为对齐锚点。
- 按 `family × difficulty × complexity_bin` 分层，seed=42 确定性抽取 100 个完整样本。
- 主文本只使用 hdv3 `en/L3`；准备阶段同时核验并记录 `en/L1`，便于后续敏感性分析。
- `I` 是 `view_0/2/4/6` 四张最终 HD render；`P` 是 2048 点 `.npy` 经固定相机生成的
  front/side/top/isometric 深度+轮廓 PNG。点云使用 mmap 读取并按输入 hash/参数缓存。
- Prompt 中使用 stem 的不可逆短 hash 作为 `sample_id`，不发送 condition 标签、真实 stem、
  GT code、family、difficulty 或 complexity。组合条件只拼接被允许的模态。
- 主配置生成 `harnesscad.plan.v2`，并直调 HarnessCAD Episode v2，支持严格 schema、
  preflight、逐操作追踪及扩展操作。`configs/v1_smoke.yaml` 保留受限 Plan v1 第一阶段，
  用于先验证 box/cylinder/sphere+布尔运算的端到端闭环。
- 冲突场景只做单独审计，明确不进入 7 条件主矩阵。

## 命令链

从项目根目录运行：

```powershell
# 1. 准备 100 样本、hash、点云视图和 manifest；不调用模型 API
python experiments/rq2_multimodal_harness/scripts/run_experiment.py --prepare-only

# 2. 第一阶段：Plan v1 小样本闭环 dry-run
python experiments/rq2_multimodal_harness/scripts/run_experiment.py `
  --config experiments/rq2_multimodal_harness/configs/v1_smoke.yaml `
  --prepare-only
python experiments/rq2_multimodal_harness/scripts/run_experiment.py `
  --config experiments/rq2_multimodal_harness/configs/v1_smoke.yaml `
  --dry-run

# 3. 第二阶段：Plan v2 主配置检查 prompt 隔离、任务顺序和原子状态
python experiments/rq2_multimodal_harness/scripts/run_experiment.py --dry-run --limit 2 --conditions T I P

# 4. 正式运行（OpenAI-compatible；不要把 key 写入 YAML）
$env:VLM_API_KEY = "..."
$env:VLM_BASE_URL = "https://provider.example/v1"
$env:VLM_MODEL = "provider-model"
python experiments/rq2_multimodal_harness/scripts/run_experiment.py

# 失败后直接重跑会跳过已有终态；指定任务重算需加 --force
python experiments/rq2_multimodal_harness/scripts/run_experiment.py --limit 10 --conditions TI,IP

# 5. 主分析：JSON/CSV/中文 Markdown
python experiments/rq2_multimodal_harness/scripts/analyze.py

# 6. Plan 表达性及可选 oracle plan 审计
python experiments/rq2_multimodal_harness/scripts/audit_expressivity.py
python experiments/rq2_multimodal_harness/scripts/audit_expressivity.py --oracle-plan-dir path/to/oracle_plans

# 7. 冲突场景 URI、空点云、模态声明审计
python experiments/rq2_multimodal_harness/scripts/audit_conflicts.py

# 8. 单元测试
python -m unittest discover -s experiments/rq2_multimodal_harness/tests -v
```

`--force` 会重建准备产物并覆盖所选任务状态，应谨慎使用。`--limit` 限制 manifest 前 N 个
样本，不改变任务的固定 hash 排序。`--conditions` 接受空格或逗号分隔。

## RQ2b 反馈修正实验

在 20 样本 × 6 条件（`conditions.subset`）上比较 5 臂：

- A0：v2 prompt 无反馈，复用 `outputs/encoding_screen_n20/state` 既有结果，不重跑；
- A1：v3 prompt（revolve/rotate 硬规则与 few-shot）无反馈；
- B1：v2 prompt + 1 轮反馈（仅 schema/格式错误）；
- B2：v2 prompt + 最多 2 轮反馈（schema + 执行崩溃，反馈轮（第 1 轮起）温度 0.3）；
- C：v3 prompt + B2 反馈。

```powershell
# 离线修复潜力评估（零 API 成本）
python experiments/rq2_multimodal_harness/scripts/assess_feedback_potential.py

# 各臂运行：--arm 自动选择 arms.<arm>.output_dir 与 feedback 预设，条件取 conditions.subset
$env:VLM_API_KEY = "..."; $env:VLM_BASE_URL = "..."; $env:VLM_MODEL = "..."
python experiments/rq2_multimodal_harness/scripts/run_encoding_screen.py `
  --config experiments/rq2_multimodal_harness/configs/feedback_n20.yaml --arm A1
# 其余臂同理：--arm B1 / --arm B2 / --arm C（A0 不重跑）

# 各臂对比分析：修复率（按失败类型）、成功率、joint quality、token 增量
python experiments/rq2_multimodal_harness/scripts/analyze_feedback.py
```

输出位于 `outputs/feedback_n20/arms/<arm>/` 与 `outputs/feedback_n20/analysis/`（含
`FEEDBACK_REPORT_ZH.md`）。反馈轮次细节记录在各 state 的 `feedback.rounds` 中。

**注意事项**：

- B1 已用修复后的 runner 全量重跑（2026-08-15，120 任务，旧 state 归档至
  `history/`），其反馈轮数据现为修复后代码产物；分析报告已按重跑数据更新。
- 反馈轮上下文为非累积式：每轮 = 原始输入 + 上一轮输出 + 本轮错误反馈，不携带更早
  轮次历史（见 `feedback.feedback_turn` 文档），以控制 token 增量；RQ2b 各臂均按此
  行为运行，分析口径以此为准。
- 反馈轮温度 `round2_temperature`（0.3）从第 1 轮反馈（round>=1）起生效，并非仅第 2 轮。

## RQ2b 确认实验（confirm_n100）

反馈筛选实验结论（C 相对 A0 成功率 +19.2%）进入确认阶段：**冻结双臂 × 全量 100 样本
× 7 条件（T/I/P/TI/TP/IP/TIP），共 700 任务/臂**，按 sample×condition 配对比较。

- A0：qwen3.8-max + v2 prompt + 无 repair + 无反馈（与 pilot_v2 管线一致，仅换模型，
  作为模型配对基线；pilot_v2 的 qwen3.7-plus 结果保留为历史参照）；
- C：qwen3.8-max + v3 prompt + repair R4 + 最多 2 轮反馈（schema + execution）。

输入复用 pilot_v2 冻结 manifest（同一样本、同一文本/渲染/点云编码），确认实验本身
不重新 prepare；state 格式与 encoding_runner 对齐（含 `feedback.rounds`）。

```powershell
# 冒烟：dry-run 不调用模型 API
python experiments/rq2_multimodal_harness/scripts/run_confirmation.py `
  --dry-run --limit 2 --conditions T I --arm A0

# 正式运行（OpenAI-compatible；不要把 key 写入 YAML）
$env:VLM_API_KEY = "..."; $env:VLM_BASE_URL = "..."; $env:VLM_MODEL = "..."
python experiments/rq2_multimodal_harness/scripts/run_confirmation.py --arm A0
python experiments/rq2_multimodal_harness/scripts/run_confirmation.py --arm C

# 失败后直接重跑会跳过已有终态；指定任务重算需加 --force（旧 state 自动归档 history/）
python experiments/rq2_multimodal_harness/scripts/run_confirmation.py `
  --arm C --conditions TI,IP

# 确认分析：A0 vs C 配对 McNemar / Wilcoxon / 样本级 bootstrap + 分层 + 报告
python experiments/rq2_multimodal_harness/scripts/analyze_confirmation.py
```

输出位于 `outputs/confirm_n100/arms/<arm>/`（state/runs/history/run_summary.json）与
`outputs/confirm_n100/analysis/`（`CONFIRM_REPORT_ZH.md`、`confirm_analysis.json`、
`confirm_arm_summary.csv`、`confirm_task_rows.csv`）。pilot_v2 作为历史参照自动纳入
分析（同 pipeline、仅模型不同，不作处理效应归因）。

**结果摘要（2026-08-17，qwen3.8-max，100 样本 × 7 条件 = 700 任务/臂）**：

- 成功率：A0 68.7%（481/700）vs C 97.6%（683/700），配对 McNemar p<0.0001
  （仅 A0 完成 4 / 仅 C 完成 206）；pilot_v2（qwen3.7-plus）63.6% 作方向参照。
- joint quality：A0 均值 0.2814 vs C 0.3659；配对 Wilcoxon 平均差 +0.0845
  （p<0.0001），样本级 bootstrap 95% CI [+0.0605, +0.1082]（不含 0）。
- 修复：C 总体修复率 87.1%（格式类 77.0%、执行类 100.0%），多轮任务引入新错误率
  3.0%；A0 无反馈故修复率 0。
- 逐条件：C 在全部 7 条件成功率 ≥96% 且 joint quality 全面高于 A0；hard 难度档
  提升最大（60.7% → 95.6%）。

## 输出与恢复

Plan v2 主配置输出位于 `outputs/pilot_v2/`，v1 smoke 位于 `outputs/v1_smoke/`：

- `manifest.jsonl`、`prepare_meta.json`、`prepare_failures.jsonl`：输入路径、流式 SHA256、
  对齐/核验结果；
- `pointcloud_views/`：固定编码 PNG 及参数/hash 缓存；
- `gt_code/`：仅离线审计使用的 100 份 GT 源码，runner 不把内容传给模型；
- `task_order.json`：seed 固定的 700 个 sample×condition 任务顺序；
- `state/{sample}/{condition}.json`：原子写入的独立状态，含输入 hash、prompt hash、
  raw response、唯一一次解析/格式修复记录、Episode、预测 STEP 路径和评分；
- `analysis/`：条件汇总、互补增益、sample bootstrap 95% CI、配对 Wilcoxon + Holm、
  family/difficulty/complexity 分层结果；
- `audits/`：表达性和冲突场景报告。

JSON 修复最多执行一次。修复只允许清理尾逗号/智能引号/裸 key，或删除多余字段、把数字
operation id 转为字符串；不会补造 primitive、尺寸、中心、轴或 operation。

## 几何指标

- `shape_only_cd`：预测和 GT 分别中心化并按最长 bbox 边归一化后的双向 Chamfer；
- `common_frame_cd`：GT 映射到中心为零、最长边为 1 的 canonical frame，预测保持 Plan
  canonical 坐标，因而保留尺度/中心误差；
- bbox：轴向比例 L1、尺度比/对数误差、中心偏移；
- voxel IoU：用 `trimesh` 在共同 canonical frame 体素化；依赖或体素化失败时写入
  `status=degraded` 和原因，不伪造数值；
- `shape_voxel_iou`：预测与 GT 各自归一化后的体素 IoU（shape-only 口径，对齐文献）；
- `fscore_shape` / `fscore_common`：F1@0.01 的 precision/recall/f1，分别在 shape-only
  与 common frame 计算；
- `invalid_ratio`（IR）：分析阶段输出，等于 1 − 几何有效率，即未形成有效可评分几何的
  任务比例，对齐 text-to-CAD 文献的 Invalid Ratio 口径；
- `joint_quality`：有效样本的 failure-aware 距离质量与体素项联合；任何无效几何固定为 0。
  其计算方式不因新增指标而改变。

已有实验结果的指标回填（纯离线、确定性，不调用模型 API）：

```powershell
# 只核对旧指标能否复现，不写文件
python experiments/rq2_multimodal_harness/scripts/backfill_metrics.py --check-only

# 正式回填：重算新增指标并写回 state，旧指标不一致的任务只记录不覆盖
python experiments/rq2_multimodal_harness/scripts/backfill_metrics.py
```

## 成本提示

正式矩阵是 **100 × 7 = 700 次付费 VLM 请求**。包含 I 和/或 P 的条件会上传 4 或 8 张
图像，图像 token 通常是主要成本；具体价格、图像计费粒度、重试计费和 JSON mode 支持均由
供应商决定。先用 `--dry-run`，再用 `--limit 1 --conditions T I P` 做真实闭环并核对账单，
确认模型、超时和 `api.json_mode` 后再扩到 700 次。本仓库测试不会调用真实 API。
