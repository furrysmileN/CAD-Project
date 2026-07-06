# CAD-Project

本仓库包含 CAD 多模态训练与数据流水线的核心代码，仅提交源码，不包含数据集、模型权重与训练产物。

## 目录结构

- `cad_data_gen/`：数据生成与处理流水线，包含 STEP 资产构建、Blender 渲染、点云与批处理脚本。
- `modified_cadrille/`：基于 cadrille 的训练、推理与评测代码（含 Qwen3-VL 相关改造）。

## 不入库内容

以下内容默认在 `.gitignore` 中排除：

- 模型与权重：`models/`
- 生成数据与中间产物：`benchcad_codegen_qwen/`
- 训练输出与 checkpoint：`modified_cadrille/work_dirs*/`
- 第三方或数据集目录：`PointTransformerV3/`、`BenchCAD/`

## 运行依赖

- 参考 `modified_cadrille/Dockerfile` 配置环境。
- 如需运行完整流程，请在本地准备模型权重与数据集路径（不在本仓库分发）。

## 说明

`PDF_RESTORE_REPORT.md`、`PDF_RESTORE_MANIFEST.json` 等文件用于记录代码恢复与校验过程，便于追溯。
