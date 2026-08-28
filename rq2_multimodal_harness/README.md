# RQ2 多模态 CAD Harness 实验（代码与报告）

本目录是 CAD 多模态生成实验 V1–V6 的**核心代码 + 实验结果报告**快照。  
**不包含**点云、STEP/STL、图像、live state、API key 或 HarnessCAD 执行后端。

对应本地工程：`experiments/rq2_multimodal_harness/`。

## 实验在做什么

固定通义千问（DashScope）与 HarnessCAD C 臂，只改变「给模型看什么」：

```text
文本 T0 / RGB 四视图 / 点云派生结构化几何证据
  → 模型输出受约束的 Plan JSON
  → C 臂校验、修复、建模
  → 与真值 STEP 评分
```

点云对外表述仅为 **point-cloud-derived structured geometric evidence**，不是模型直接消费的 embedding。

## 报告（请从这里读结论）

| 文件 | 内容 |
|---|---|
| [`EXPERIMENT_REPORT_ZH.md`](EXPERIMENT_REPORT_ZH.md) | V1 |
| [`EXPERIMENT_REPORT_V2_ZH.md`](EXPERIMENT_REPORT_V2_ZH.md) | V2 |
| [`EXPERIMENT_REPORT_V3_ZH.md`](EXPERIMENT_REPORT_V3_ZH.md) | V3 / C 臂 |
| [`EXPERIMENT_REPORT_V4_ZH.md`](EXPERIMENT_REPORT_V4_ZH.md) | V4 点云表征 |
| [`EXPERIMENT_REPORT_V5_ZH.md`](EXPERIMENT_REPORT_V5_ZH.md) | V5 双向互补（正式推断） |
| [`EXPERIMENT_REPORT_V6_ZH.md`](EXPERIMENT_REPORT_V6_ZH.md) | V6 / V6b（描述性；confirm 未开） |
| [`EXPERIMENT_ARC_V1_V5_ZH.md`](EXPERIMENT_ARC_V1_V5_ZH.md) | V1–V5 叙事 |
| [`EXPERIMENT_STAGE_REPORT_V1_V6_PILOT_ZH.md`](EXPERIMENT_STAGE_REPORT_V1_V6_PILOT_ZH.md) | V1–V6 阶段综述 |

## 代码布局

| 路径 | 作用 |
|---|---|
| `rq2_harness/` | 条件组装、Prompt、评分、V5/V6 runner |
| `scripts/` | 准备 / dry-run / live / 分析入口 |
| `configs/` | YAML；API key 只引用环境变量名 `VLM_API_KEY` |
| `tests/` | 单元测试 |

配置里的路径按原仓库根目录书写（例如 `HarnessCAD/HarnessCAD`）。本快照不含该后端与数据集，不能直接复现 live。

密钥不要写入 YAML。需要跑 API 时在环境中设置 `VLM_API_KEY` / `VLM_BASE_URL` / `VLM_MODEL`。
