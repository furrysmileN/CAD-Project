# CAD-Project

本仓库包含 CAD 多模态训练、数据流水线与生成实验的核心代码，仅提交源码与实验报告，不包含数据集、模型权重与训练/评测产物。

## 目录结构

- `cad_data_gen/`：数据生成与处理流水线，包含 STEP 资产构建、Blender 渲染、点云与批处理脚本。
- `modified_cadrille/`：基于 cadrille 的训练、推理与评测代码（含 Qwen3-VL 相关改造）。
- `HarnessCAD/`：FeaturePlan 编译执行器（Plan v2 / v3 / v3.1）与前端预览；不含运行产物与 `node_modules`。
- `rq2_multimodal_harness/`：多模态 CAD 生成实验代码（含 HVC 三臂对照、Plan v5 prompt）与 V1–V6 报告（不含点云 / STEP / live 数据）。

## 不入库内容

以下内容默认在 `.gitignore` 中排除：

- 模型与权重：`models/`
- 生成数据与中间产物：`benchcad_codegen_qwen/`
- 训练输出与 checkpoint：`modified_cadrille/work_dirs*/`
- 第三方或数据集目录：`PointTransformerV3/`、`BenchCAD/`
- 实验 live 产物：`rq2_multimodal_harness/outputs/`
- API key：`rq2_multimodal_harness/qwenapikey.txt`
- HarnessCAD 运行产物与前端依赖：`backend/harness_runs*`、`.venv/`、`frontend/node_modules/`、`frontend/dist/`

## 运行依赖

- 训练与数据流水线：参考 `modified_cadrille/Dockerfile`。
- HarnessCAD：见 `HarnessCAD/README.md`。
- RQ2 生成实验：见 `rq2_multimodal_harness/README.md`。完整复现还需要 BenchCAD 数据集与本地 API key（均不在本仓库分发）。

## 说明

`PDF_RESTORE_REPORT.md`、`PDF_RESTORE_MANIFEST.json` 等文件用于记录代码恢复与校验过程，便于追溯。
