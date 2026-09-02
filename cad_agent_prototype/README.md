# CadQuery CAD Agent 原型

该原型提供第一条可验证链路：

```text
生成的 CadQuery Python
  -> 独立 Python 子进程
  -> result.step
  -> CadQuery/OCP 回读
  -> manifest.json（结构化验证或错误信号）
```

STEP/B-Rep 是权威输出；后续 STL、点云和渲染都应由 STEP 派生。子进程用于隔离崩溃并限制执行时间，但**不是安全沙箱**，不能直接运行不可信代码。

## 生成代码契约

输入 Python 文件必须定义：

```python
def build_model(params):
    ...
    return cadquery_workplane_or_assembly
```

- `params` 是 JSON 对象；
- 长度单位统一为 mm；
- 返回 CadQuery `Workplane`、`Shape` 或 `Assembly`；
- 不依赖全局变量 `r`；
- 原型只验证实体 B-Rep，不接受只有 wire/surface 的结果。

## 命令行

在安装了 CadQuery/OCP 的 Python 环境中，从仓库根目录运行：

```powershell
python -m cad_agent_prototype cad_agent_prototype/examples/parametric_plate.py `
  -o outputs/cad_agent/plate `
  --params '{"width": 80, "hole_spacing": 30}'
```

也可以指定包含 CadQuery 的 Python：

```powershell
python -m cad_agent_prototype model.py -o outputs/cad_agent/run --python C:\path\to\python.exe
```

成功产物：

- `result.step`：可回读的权威 CAD 几何；
- `manifest.json`：供 agent/实验框架消费的最终结果；
- `compile_request.json`、`compile_response.json`：子进程边界记录。

## Python API

```python
from cad_agent_prototype import compile_cadquery

result = compile_cadquery(
    "model.py",
    "outputs/cad_agent/run",
    parameters={"width": 80.0},
    timeout_s=60,
    python_executable=r"C:\path\to\python.exe",
)

if not result.ok:
    print(result.signals[0].code, result.signals[0].message)
```

当前机器可读错误包括：

- `worker_start_failed`、`worker_timeout`、`worker_crashed`；
- `dependency_unavailable`、`source_not_found`、`source_load_failed`；
- `contract_missing`、`build_failed`、`export_failed`；
- `step_import_failed`、`empty_step`、`invalid_brep`、`empty_solid`；
- `nonpositive_volume`、`nonfinite_geometry`、`zero_size_geometry`。

## 下一阶段接口

`manifest.json` 可直接成为后续两类适配的共同边界：

1. 在 `rq2_multimodal_harness` 增加 CadQuery agent backend，并复用 STEP 几何评分；
2. 在编译结果之后调用 `cad_data_gen/build_step_assets.py`，生成点云和多视图，构建 Procedura 式视觉纠错闭环。
