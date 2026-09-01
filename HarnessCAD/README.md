# HarnessCAD

HarnessCAD 是一个独立的、可审计的 CAD Plan 实验执行器。它把受约束的 JSON Plan 编译为确定性 CadQuery 代码，执行建模与导出，并记录逐阶段、逐操作的几何状态。

## 当前用途

- 检查 AI 生成的 CAD Plan 是否满足 schema 与规范化约束。
- 在执行前发现完全切除、无效布尔操作、断开实体等风险。
- 定位 CadQuery 失败发生在哪个操作和阶段。
- 保存 STEP、STL、生成代码、运行环境、SHA256 与操作轨迹。
- 为 image-only、point-cloud-only、image+point-cloud 等实验条件提供统一执行层。

Plan v1 继续支持 `box`、`cylinder`、`sphere`，以及 `new`、`add`、`cut`、`intersect`。Plan v2 使用
`schema_version: "harnesscad.plan.v2"`，并增加 `polygon_extrude`、`revolve_profile`、全局平面的
`hole`/`slot`、`transform`、`fillet`、`chamfer` 和 `linear_pattern`。两个版本都由 Episode v2
端点按 `schema_version` 自动分派，坐标采用最长包围盒边为 `1.0`、中心位于原点的 normalized frame。

## Plan v2 摘要

Plan v2 的每个操作使用严格的 `op` 判别字段。实体构造操作带 `combine`；`transform` 和
`linear_pattern` 的 `source` 只能引用更早的实体构造操作；`fillet`/`chamfer` 直接修改当前累计结果。
未声明字段、非有限数、越界坐标、前向/未知引用、非单位轴、开放/退化/自交多边形都会在执行前拒绝。
编译器只从已校验 JSON 生成固定 CadQuery 调用，不执行 Plan 中的任意代码。

```json
{
  "schema_version": "harnesscad.plan.v2",
  "sample_id": "plate",
  "coordinate_system": {
    "units": "normalized",
    "origin": [0, 0, 0],
    "longest_bbox_edge": 1
  },
  "operations": [
    {
      "id": "plate",
      "op": "polygon_extrude",
      "combine": "new",
      "workplane": "XY",
      "points": [[-0.5, -0.3], [0.5, -0.3], [0.5, 0.3], [-0.5, 0.3], [-0.5, -0.3]],
      "depth": 0.2,
      "centered": true,
      "offset": [0, 0, 0]
    },
    {
      "id": "mount_hole",
      "op": "hole",
      "combine": "cut",
      "workplane": "XY",
      "center": [0, 0, 0],
      "diameter": 0.12,
      "depth": 0.4
    }
  ]
}
```

`XY` 的局部 `(u,v,n)` 为 `(X,Y,+Z)`，`XZ` 为 `(X,Z,-Y)`，`YZ` 为 `(Y,Z,+X)`。
`polygon_extrude.points` 和 `revolve_profile.profile` 必须用重复首点显式闭合。当前安全子集只提供直线
slot、绕明确二维轴的 revolve、全边或按全局轴平行边的 fillet/chamfer，以及等距直线阵列。

## 目录

```text
HarnessCAD/
├─ backend/
│  ├─ main.py                 # 独立 FastAPI 入口
│  ├─ harness_api.py          # v1 校验/编译/执行
│  ├─ harness_api_v2.py       # preflight、逐操作追踪、Episode v2
│  ├─ plan_v2_schema.py       # Plan v2 严格 schema/语义校验
│  ├─ plan_v2_compiler.py     # Plan v2 确定性安全编译器
│  ├─ test_harness_v2.py      # Episode v2 与 Plan v1 回归
│  ├─ test_plan_v2.py         # Plan v2 schema/操作/空几何测试
│  ├─ plan_v3_schema.py       # Plan v3：sweep / loft / 圆弧
│  ├─ plan_v31_schema.py      # Plan v3.1 字段与姿态约束
│  ├─ test_plan_v3.py
│  ├─ test_plan_v31.py
│  └─ requirements.txt
├─ frontend/
│  ├─ src/                    # React 页面与 Three.js 预览组件
│  ├─ harness.html            # v1 页面
│  ├─ harness-v2.html         # 推荐的 Episode v2 页面
│  └─ vite.config.ts
├─ examples/                  # 成功、空几何、断开实体示例
├─ start_harness.cmd
└─ run_tests.cmd
```

`backend/harness_runs/` 和 `backend/harness_runs_v2/` 是运行时自动创建的实验产物目录，不属于源代码。

## 使用现有 Demo 环境启动

该目录位于原 `demo示范` 内时，`start_harness.cmd` 会优先使用自己的 `.venv`，否则自动复用上一级的 `.venv`。

双击：

```text
start_harness.cmd
```

打开：

- 首页：http://localhost:5173/
- Episode v2：http://localhost:5173/harness-v2.html
- API 文档：http://127.0.0.1:8000/docs

## 在其他电脑上建立环境

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
cd .\frontend
npm install
```

然后回到工程根目录运行 `start_harness.cmd`。

## 测试

```text
run_tests.cmd
```

测试覆盖：v1 回归、全部 Plan v2 操作、非法字段/引用/坐标、多边形约束、完全切除为空、非规范尺度和断开实体。前端随后执行 TypeScript 检查与生产构建。

## 最小实验流程

1. 从 `examples/plan_success.json` 复制一个 Plan。
2. 在 `metadata` 中写入 `condition`、模型名、模态和 prompt 版本。
3. 在 v2 页面先执行 `校验 + Preflight`。
4. 执行并记录 Episode。
5. 从 `backend/harness_runs_v2/<run_id>/episode_v2.json` 汇总成功率、警告、首次失败操作和几何复杂度。

默认推荐使用 v2。v1 仅用于和早期实验记录保持兼容。
