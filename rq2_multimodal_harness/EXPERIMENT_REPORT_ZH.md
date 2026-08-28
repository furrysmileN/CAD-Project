# HarnessCAD 多模态互补 CAD 生成实验报告

## 1. 报告摘要

本实验研究一个实际问题：当大模型根据零件描述生成可执行 CAD 模型时，文本、普通渲染图和点云信息能否互相补充，从而提高生成结果的几何质量和可靠性。

实验没有训练本地模型，而是调用阿里云 DashScope 上的 `qwen3.7-plus` 进行推理。模型不直接输出任意 Python 或 CadQuery 代码，而是输出受严格约束的 HarnessCAD Plan JSON。HarnessCAD 随后负责校验 Plan、编译成 CadQuery、执行建模操作、导出 STEP，并记录逐操作状态。预测 STEP 最终与 BenchCAD 的真实 STEP 进行几何比较。

正式实验从 1000 个已对齐 BenchCAD 样本中，按零件类别、难度和代码复杂度确定性抽取 100 个样本。每个样本分别测试七种输入条件：

- T：高细节程序化文本；
- I：四视角普通渲染图；
- P：由 2048 点点云生成的四视角深度与轮廓图；
- TI：文本和普通图像；
- TP：文本和点云视图；
- IP：普通图像和点云视图；
- TIP：三种模态全部使用。

主实验共完成 700 次付费模型请求。所有请求都获得了 API 返回，其中 445 个 Plan 在 HarnessCAD 中执行成功，439 个结果最终形成了有效且可评分的几何。整体结果显示：

- 点云条件 P 的几何有效率最高，为 80%；
- 文本 T 是质量最好的单模态条件；
- TP 和 TIP 在成功生成时通常有更好的几何相似度；
- 但组合模态产生了更多 Schema 和执行失败，抵消了几何质量收益；
- 在当前实验设置下，没有足够统计证据证明多模态相对强单模态带来了稳定、显著的整体提升；
- 当前主要瓶颈是模型如何把理解到的几何稳定编码为严格 Plan v2，而不是输入中缺少几何信息。

因此，本轮实验支持的结论不是“多模态没有用”，而是“当前 HarnessCAD Plan 接口尚未稳定地把多模态信息转化为可执行 CAD”。下一阶段应优先降低 Plan 校验和执行失败，再重新检验模态互补性。

## 2. 研究背景

### 2.1 为什么要做这个实验

从文本、图像或三维观测生成 CAD，与生成普通图片不同。一个看起来合理的 CAD 结果还必须满足：

- 几何实体有效；
- 建模步骤可以执行；
- 布尔操作不会产生空形状；
- 尺寸、位置和方向合理；
- 输出可以保存为标准 STEP；
- 结果能够复现和审计。

大模型可以生成 CadQuery 或其他 CAD 代码，但自由代码生成容易出现语法错误、不可控执行、无效几何和难以定位的中间失败。HarnessCAD 的作用是在模型与 CAD 内核之间增加一个结构化、可验证、可追踪的执行层。

另一方面，不同输入模态包含的信息并不相同：

- 文本容易描述零件语义、构成和操作顺序，但可能缺少准确外形；
- 普通渲染图可以表现外观和遮挡关系，但尺寸和内部结构不明确；
- 点云可以表现三维轮廓和比例，但缺少语义、特征名称和建模顺序。

理论上，这些模态可能互补。例如文本说明“这是一个带通孔的阶梯轴”，点云补充各段比例，图像补充孔和表面外观。但额外模态也可能增加上下文长度、引入冲突，或者诱导模型生成更复杂、更容易失败的 Plan。因此必须通过同一样本上的配对消融实验，而不是只观察少数成功案例。

### 2.2 研究问题

本轮实验主要回答四个问题：

1. 单独使用文本、图像或点云时，哪种输入最容易生成有效 CAD？
2. 两种或三种模态组合后，是否提高整体几何质量？
3. 多模态的收益是否能够抵消解析、Schema 校验和执行失败？
4. 当前 HarnessCAD 的主要瓶颈是输入信息不足、Plan 表达能力不足，还是模型不遵守结构化接口？

## 3. 项目与资源概况

### 3.1 项目位置与总体资产

项目位于：

`D:\AI代码\多模态数据对比实验计划`

前期存储审计显示，整个项目约占 219.6 GB，主要资源包括：

- `processed/renders/benchcad/`：约 145.9 GB，是最大的资产，包含约 16.1 万张 BenchCAD PNG 渲染图；
- `data/zero_to_cad/`：约 60.4 GB，主要是 HuggingFace Arrow 数据分片；
- `processed/models/benchcad/`：约 11.0 GB，包含处理后的 BenchCAD 三维模型；
- `processed/point_clouds/`：约 1.2 GB；
- `data/benchcad/`：约 1.1 GB。

本轮 100 样本主实验只使用 BenchCAD 相关资产。Zero-to-CAD 虽然存在于项目中，但没有进入本轮主矩阵。

### 3.2 BenchCAD 主数据集

BenchCAD 的本地数据集位于：

`data/benchcad/code_gen/dataset`

它提供样本索引、零件 stem、family、difficulty 和 GT CadQuery 代码等信息。GT 代码在本实验中仅用于：

- 样本对齐；
- 计算代码复杂度；
- 审计 HarnessCAD Plan 的表达能力；
- 离线保存和复核。

GT CadQuery 代码没有发送给 Qwen，也没有出现在正式推理 prompt 中。

### 3.3 GT STEP 模型

用于几何评分的真实 STEP 位于：

`processed/models/benchcad`

每个预测 STEP 都与同一 stem 对应的 GT STEP 比较。GT STEP 只在模型推理完成之后进入离线评分，不会发送给模型。

### 3.4 普通渲染图

图像模态来自：

`processed/renders/benchcad`

每个样本固定使用四张最终高细节渲染：

- `view_0.png`
- `view_2.png`
- `view_4.png`
- `view_6.png`

输入 API 前，图像最长边限制为 1024 像素。这些图像表现零件最终外观，但不带尺寸标注、类别标签或 GT 代码。

### 3.5 点云

原始点云来自：

`processed/point_clouds/benchcad/2048`

每个 `.npy` 文件包含 2048 个 XYZ 点。由于本轮使用的视觉语言模型 API 不能直接接收 `.npy`，实验将点云归一化后确定性编码为四张 512×512 PNG：

- front；
- side；
- top；
- isometric。

每张图同时表达深度和轮廓。编码版本为 `rq2.pc_depth_contour.v1`，padding 为 0.06。

因此，本报告中的 P 不是原生点云网络输入，而是“点云的多视角视觉编码”。实验结论不能直接外推到 PointNet、Point Transformer 或支持原生三维 token 的模型。

### 3.6 文本描述

文本来自：

`processed/text_descriptions/benchcad/code_gen_hdv3.jsonl`

数据源字段标记为 `cursor_hd_v3`，每个样本包含英文和中文的 L1、L2、L3 描述。本轮准备阶段核验并保存英文 L1 和 L3，但正式主实验只发送英文 L3。

L1 通常是一句话概述，例如：

`A threaded adapter with a hex base and a stepped threaded cylinder.`

实际使用的 L3 则可能包含：

- 建模操作顺序；
- primitive 或特征类型；
- 拉伸、旋转、钻孔等动作；
- 近似直径、半径、长度和深度；
- 最终需要验证的结构。

例如 threaded adapter 的 L3 会说明先拉伸六边形底座，再添加两段圆柱，最后钻通孔，并给出各段近似尺寸。

这意味着本轮 T 不是普通用户的一句话提示，而是高信息量、接近 CAD 操作配方的“程序化文本”。仓库中没有保留英文 L3 的完整生成溯源，因此无法严格证明其与 GT 代码完全独立。虽然实验运行器没有直接泄漏 GT 代码，但 L3 在语义上属于特权信息。这是解释文本条件较强时必须考虑的限制。

### 3.7 HarnessCAD

HarnessCAD 位于：

`HarnessCAD/HarnessCAD`

它负责：

- Plan JSON Schema 校验；
- 语义规则检查；
- 静态 preflight；
- 确定性编译为 CadQuery；
- 受控执行；
- 逐操作几何状态记录；
- STEP 输出；
- 警告和失败分类。

原始 Plan v1 只支持 box、cylinder、sphere 和有限布尔操作，无法覆盖多数 BenchCAD 零件。为本实验新增并接入了 Plan v2，支持：

- box、cylinder、sphere；
- polygon extrusion；
- revolve profile；
- hole、slot；
- transform；
- fillet、chamfer；
- linear pattern；
- new、add、cut、intersect。

静态表达能力审计结果为：

- Plan v1 直接操作覆盖率约 15.9%；
- Plan v1 在 100 个样本中没有估计完全可表达的样本；
- Plan v2 直接操作覆盖率约 66.5%；
- Plan v2 估计可完全表达 73/100 个样本；
- 允许简单近似时，Plan v2 操作覆盖率约 90.0%。

这里的覆盖率是对 GT CadQuery 调用的静态代理，不等同于模型一定可以生成，也不代表几何可以无损复现。

### 3.8 大模型与 API

正式实验使用：

- 服务：阿里云 DashScope OpenAI-compatible API；
- 地址：`https://dashscope.aliyuncs.com/compatible-mode/v1`；
- 模型：`qwen3.7-plus`；
- temperature：0；
- 最大输出：4096 tokens；
- JSON mode：开启；
- thinking：关闭；
- 单次超时：180 秒；
- 最大重试：3 次。

API 凭据由用户此前提供，并通过环境变量注入，没有写入配置文件或报告。

主实验共消耗约 3,156,566 tokens。各请求记录的 API 延迟累计约 2.82 小时；完整后台任务包括 API 等待、HarnessCAD 执行、STEP 导出和几何评分，墙钟时间约 3.52 小时。

### 3.9 本地计算环境

本地环境审计时的配置为：

- 系统：Windows 11 专业版 64 位；
- CPU：Intel Core i5-13600KF，14 核、20 线程；
- 内存：约 32 GB，审计时可用约 17.6 GB；
- GPU：AMD Radeon RX 6800 XT，约 16 GB 显存；
- NVIDIA CUDA：不可用；
- C 盘：约 688.9 GB 可用；
- D 盘：约 120.0 GB 可用。

本实验不进行本地大模型训练。Qwen 推理由云端完成，本地主要承担数据读取、图像编码、CadQuery 执行、STEP 处理和几何评分。因此 AMD GPU 和无 CUDA 并没有阻止主实验，但 D 盘空间对后续扩大渲染和缓存规模构成约束。

## 4. 样本准备

### 4.1 对齐母池

实验以点云元数据中的 `sample_idx` 和 `stem` 为对齐锚点，限制在 BenchCAD 前 1000 个样本。

进入候选池的样本必须同时具备：

- 可解析的 BenchCAD 元数据；
- 非空且语法有效的 GT CadQuery 代码；
- GT STEP；
- 2048 点点云；
- 四张指定渲染图；
- 英文 L1；
- 英文 L3。

准备结果为：

- 对齐候选：1000；
- 有效候选：1000；
- 准备失败：0；
- 最终抽样：100。

所有源文件都计算 SHA256，保证后续恢复运行时能够检测输入变化。

### 4.2 分层抽样

使用固定 seed 42，按以下维度确定性分层：

- family；
- difficulty；
- complexity_bin。

代码复杂度通过解析 GT Python AST，统计函数调用、循环和条件分支数量，再按照母池分成三个复杂度区间。

最终 100 个样本包括：

- 73 个 family；
- easy：26 个；
- medium：38 个；
- hard：36 个；
- complexity bin 0：25 个；
- complexity bin 1：36 个；
- complexity bin 2：39 个；
- 复杂度最小值：5；
- 中位数：18；
- 最大值：126。

这种方法强调类别和难度覆盖，而不是严格复现 BenchCAD 的自然分布。由于多数 family 只有 1–2 个样本，family 级结论只能作为探索性观察。

### 4.3 信息隔离

模型收到的 sample ID 是真实 stem 的不可逆短 SHA256，不会收到：

- 真实 stem；
- family；
- difficulty；
- complexity；
- condition 名称；
- GT CadQuery 代码；
- GT STEP；
- 数据路径。

组合条件只拼接该条件允许的模态。所有 prompt 和输入文件都有 hash，用于检查模态隔离和断点恢复。

## 5. 实验设计

### 5.1 七条件配对消融

每个样本都运行以下七个条件：

- T：只发送英文 L3；
- I：只发送四张普通渲染图；
- P：只发送四张点云深度与轮廓图；
- TI：L3 加四张普通渲染图；
- TP：L3 加四张点云视图；
- IP：四张普通渲染图加四张点云视图；
- TIP：L3、四张普通渲染图和四张点云视图。

这形成 100 × 7 = 700 个任务。同一个样本在七个条件下使用相同 GT、相同评分参数和相同 HarnessCAD 版本，因此可以进行配对比较。

### 5.2 模型任务

模型被要求输出一个 `harnesscad.plan.v2` JSON，而不是解释文字或 Python 代码。

统一约束包括：

- 坐标单位为 normalized；
- bbox 中心为 `[0,0,0]`；
- 最长 bbox 边为 1；
- 仅使用 Plan v2 允许的操作；
- 第一个实体操作使用 `new`；
- 后续可使用 `add`、`cut` 或 `intersect`；
- 操作数量为 1–64；
- 轴向量必须是有限单位向量；
- polygon/profile 必须显式闭合；
- operation 必须包含对应 Schema 要求的全部字段。

L3 中的尺寸以毫米描述，而输出 Plan 必须位于归一化坐标系。模型需要从描述的相对尺寸关系中自行换算。这一设计可以考察尺度推理，但也可能带来额外误差。

### 5.3 输出解析与有限修复

模型输出后最多执行一次格式修复。允许的修复包括：

- 提取 Markdown 围栏中的 JSON；
- 删除尾逗号；
- 替换智能引号；
- 给裸 JSON key 加引号；
- 删除 Schema 外的多余字段；
- 把数字 operation ID 转为字符串。

修复不会补造 primitive、尺寸、中心、轴或缺失操作。

700 个任务中：

- 693 个没有触发格式修复；
- 4 个触发 JSON 语法修复；
- 3 个触发字段格式修复。

这说明普通 JSON 语法并不是本轮最主要的问题。

### 5.4 HarnessCAD 执行

解析通过后，Plan 进入 HarnessCAD Episode v2：

1. 严格 Schema 校验；
2. 几何与引用语义检查；
3. preflight 风险检查；
4. 编译成确定性 CadQuery；
5. 在 30 秒限制内逐操作执行；
6. 检查每一步形状是否有效；
7. 导出最终 STEP；
8. 保存警告、失败码和操作 trace。

每个 sample × condition 的完整状态独立写入 JSON，包括输入 hash、prompt hash、模型原始回复、解析结果、Episode 响应、预测 STEP 路径和几何评分。任务可断点恢复，prompt 或输入 hash 改变时不会错误复用旧结果。

## 6. 评价指标

### 6.1 流水线成功率

实验区分多个阶段：

- API 返回；
- JSON/Plan 解析；
- HarnessCAD Schema 通过；
- CAD 执行成功；
- STEP 几何有效；
- 几何评分完成。

这样可以区分“模型没返回”“结构化输出错误”“Plan 合法但 CAD 执行失败”和“CAD 执行完成但几何无法评分”。

### 6.2 Shape-only Chamfer Distance

从预测 STEP 和 GT STEP 的表面各采样 2048 个点。两者分别中心化并按自身最长 bbox 边归一化，再计算双向 Chamfer Distance。

该指标主要衡量形状相似度，弱化整体位置和尺度误差。数值越低越好。

### 6.3 Common-frame Chamfer Distance

GT 被映射到中心为零、最长边为一的 canonical frame，预测则保持 Plan 输出坐标。

该指标会同时惩罚：

- 形状错误；
- 尺度错误；
- 中心偏移。

数值越低越好。

### 6.4 Bbox 指标

额外记录：

- 三轴比例误差；
- 最长边尺度比；
- 对数尺度误差；
- 中心距离；
- 三轴中心偏移。

### 6.5 Voxel IoU

使用 `trimesh` 将预测和 GT 放入共同 canonical frame，以 48 分辨率体素化并计算交并比。数值越高越好。

如果体素化依赖或网格处理失败，会明确记录 degraded 状态，不会伪造数值。

### 6.6 Failure-aware joint quality

有效样本的 joint quality 综合：

- shape-only Chamfer；
- common-frame Chamfer；
- voxel IoU。

任何解析失败、Schema 失败、执行失败或无效几何的 joint quality 固定为 0。

因此 joint quality 不只是“成功结果有多好”，而是同时衡量：

- 模型是否能生成；
- Plan 是否合规；
- CAD 是否能执行；
- 最终几何是否接近 GT。

## 7. 实际执行过程

### 7.1 数据与冲突资产审计

首先完成了项目存储、本地环境和多模态资产盘点。随后审计了已有的 8400 条 complementary、conflict、degraded、missing 场景。

这些场景存在大量：

- 旧版 URI；
- 空点云引用；
- 模态声明不一致。

因此它们被明确排除在主矩阵之外，避免把数据完整性问题误判为模型的多模态冲突能力。

### 7.2 Plan v1 dry-run

使用 20 个样本和七个条件构造了 140 个 Plan v1 dry-run 任务，用于检查：

- 输入准备；
- 模态隔离；
- prompt；
- 固定任务顺序；
- 状态保存；
- 恢复逻辑。

这一步没有调用真实模型 API。表达能力审计显示 Plan v1 不足以支撑完整 BenchCAD 主实验，因此没有把 v1 结果作为研究结论。

### 7.3 Plan v2 扩展与测试

随后扩展 HarnessCAD Plan v2，并完成：

- Schema；
- 编译器；
- Episode v2 分发；
- 新操作执行；
- v1 回归兼容；
- 非法输入与空几何测试；
- prompt 隔离与解析测试。

实验框架最终有 13 项核心单元测试通过，未发现新增 lint 错误。

### 7.4 真实 API 小规模闭环

正式扩大规模之前，先选择 1 个样本运行七个条件。七次调用都获得 API 返回，Plan 可以进入 HarnessCAD 并完成几何评分，从而确认真实模型、JSON mode、HarnessCAD 和 STEP 评价链路连通。

早期输出暴露了 cylinder 缺少 center/axis 等问题，因此系统 prompt 增加了每种 operation 的明确字段要求，并关闭 thinking，以减少非结构化输出。

### 7.5 正式 100 样本主矩阵

随后执行 100 样本 × 7 条件的完整矩阵。

运行支持断点恢复，最初 1 个样本的 7 个结果被直接复用，其余 693 个任务继续完成。最终状态目录中共有 700 个终态记录。

## 8. 主实验结果

### 8.1 总体完成情况

700 个任务全部获得 API 返回，没有遗留 running、API fatal error 或 task crash。

最终状态为：

- HarnessCAD 执行成功：445；
- Episode 失败：227；
- Plan 解析失败：28。

445 个执行成功结果中，有 439 个形成有效且可评分的几何，另外 6 个在 STEP 导入或几何评价阶段被判为无效。

从完整 700 个任务看：

- API 返回率：100%；
- Plan 解析率：96%；
- HarnessCAD 执行成功率：63.6%；
- 最终几何有效率：62.7%。

### 8.2 各条件结果

T，文本：

- 最终几何有效率：58%；
- joint quality：0.2496；
- shape-only CD：0.1761；
- common-frame CD：0.1902；
- voxel IoU：0.3851。

I，普通渲染图：

- 最终几何有效率：63%；
- joint quality：0.1763；
- shape-only CD：0.2409；
- common-frame CD：0.2529；
- voxel IoU：0.2167。

P，点云视图：

- 最终几何有效率：80%；
- joint quality：0.2132；
- shape-only CD：0.2296；
- common-frame CD：0.2438；
- voxel IoU：0.1728。

TI，文本加普通图像：

- 最终几何有效率：56%；
- joint quality：0.2382；
- shape-only CD：0.1508；
- common-frame CD：0.1743；
- voxel IoU：0.3585。

TP，文本加点云视图：

- 最终几何有效率：55%；
- joint quality：0.2644；
- shape-only CD：0.1419；
- common-frame CD：0.1604；
- voxel IoU：0.4373。

IP，普通图像加点云视图：

- 最终几何有效率：67%；
- joint quality：0.2005；
- shape-only CD：0.2384；
- common-frame CD：0.2417；
- voxel IoU：0.2355。

TIP，三模态：

- 最终几何有效率：60%；
- joint quality：0.2649；
- shape-only CD：0.1508；
- common-frame CD：0.1753；
- voxel IoU：0.3898。

### 8.3 结果的第一层含义

从单模态看：

- T 的平均 joint quality 最高，说明高细节程序化文本能够直接提供操作和尺寸信息；
- P 的有效率最高，说明点云投影视图更容易诱导模型输出简单、可执行的整体几何；
- I 的有效率不低，但最终几何质量最低，说明普通渲染图在缺少语义和尺寸时难以反推出准确 CAD。

从组合模态看：

- TP 和 TIP 的 joint quality 最高；
- TP 的 shape-only CD、common-frame CD 和 voxel IoU 都是七个条件中最好；
- 这说明组合信息在成功执行时确实可能改善几何；
- 但 TP 有效率只有 55%，低于 T 的 58% 和 P 的 80%；
- 组合模态提高了部分成功结果的质量，却让模型更容易生成复杂或不合规的 Plan。

### 8.4 配对直接比较

在同一样本上直接计算组合条件减去组成单模态：

- TI − T：平均 −0.0115，95% bootstrap CI 为 [−0.0551, 0.0322]；
- TI − I：平均 +0.0619，CI 为 [0.0035, 0.1198]；
- TP − T：平均 +0.0148，CI 为 [−0.0309, 0.0629]；
- TP − P：平均 +0.0512，CI 为 [−0.0121, 0.1148]；
- IP − I：平均 +0.0242，CI 为 [0.0010, 0.0496]；
- IP − P：平均 −0.0128，CI 为 [−0.0549, 0.0286]；
- TIP − T：平均 +0.0152，CI 为 [−0.0291, 0.0600]；
- TIP − I：平均 +0.0886，CI 为 [0.0272, 0.1500]；
- TIP − P：平均 +0.0516，CI 为 [−0.0104, 0.1153]。

这些结果说明：

- 组合条件相对弱图像基线 I 通常有明显提升；
- 图像加入强文本或点云后没有显示稳定增益；
- TP/TIP 相对 T 或 P 有正向趋势，但置信区间跨过 0；
- 目前不足以宣称稳定的文本—点云互补效应。

### 8.5 保守互补增益

主分析还定义了更严格的“oracle 互补增益”：

- TI 与同一样本 T、I 中较好的结果比较；
- TP 与 T、P 中较好的结果比较；
- IP 与 I、P 中较好的结果比较；
- TIP 与 T、I、P 中最好的结果比较。

结果为：

- TI：−0.0812，95% CI [−0.1314, −0.0337]；
- TP：−0.0853，95% CI [−0.1386, −0.0341]；
- IP：−0.0732，95% CI [−0.1035, −0.0442]；
- TIP：−0.1152，95% CI [−0.1696, −0.0616]。

这个指标使用“每个样本事后选择最佳单模态”的强 oracle 作为基线，本身偏保守。它不能简单解释成“多模态一定有害”，但说明当前组合条件还不能稳定超越单模态中针对每个样本选出的最佳结果。

### 8.6 显著性检验

分析以样本为配对单位：

- bootstrap 重复 5000 次；
- seed 为 42；
- 使用配对 Wilcoxon signed-rank 检验；
- 对全部条件对执行 Holm 多重比较校正。

Holm 校正后没有任何条件对达到 0.05 显著性水平，最小校正后 p 值约为 0.167。

因此目前只能报告趋势，不能报告“多模态显著优于单模态”。

### 8.7 分层趋势

在 hard 子集上：

- TP − T 平均约 +0.0624；
- TIP − T 平均约 +0.0787。

这提示困难样本可能更需要视觉或三维轮廓信息。但这些是事后分层观察，没有单独进行多重校正，而且每个 family 的样本很少，因此只能用于设计下一轮实验。

## 9. 失败分析

### 9.1 失败层级

未形成 HarnessCAD 成功结果的 255 个任务包括：

- 解析失败：28；
- Plan 校验失败：182；
- operation exception：43；
- operation 后形状无效：2。

在 445 个执行成功任务中，仍有 6 个结果在后续 STEP/几何评分阶段无效。

### 9.2 Schema 校验是最大瓶颈

Plan 校验失败占 255 个首轮失败中的约 71%。

高频校验问题包括：

- invalid revolve axis：136 次；
- invalid vector：81 次；
- invalid rotate：75 次；
- invalid number：35 次；
- missing field：33 次；
- self-intersecting polygon：23 次；
- duplicate polygon vertex：19 次；
- invalid reference：17 次；
- degenerate polygon：12 次；
- invalid slot dimensions：11 次；
- invalid pattern count：9 次。

同一个 Plan 可以同时出现多个问题，因此这些次数不能直接相加为失败任务数。

这说明模型常常能够生成“看起来像 CAD 操作”的内容，但无法稳定满足 Plan v2 对轴、旋转、向量、字段集合和草图拓扑的严格定义。

### 9.3 成功结果仍有大量警告

成功 Episode 中出现的主要警告包括：

- noncanonical final geometry：322 次；
- ineffective cut likely：129 次；
- disconnected add likely：88 次；
- multiple final solids：62 次；
- declared scale mismatch likely：11 次。

多个警告可以同时出现在一个任务中。

这些警告表明即使 Plan 能执行，也可能存在：

- 最终几何没有严格位于 canonical frame；
- 切除操作没有真正切到实体；
- add 产生不连接实体；
- 最终留下多个 solid；
- 声明尺度与实际尺度不一致。

因此“执行成功”不能等同于“CAD 正确”，必须继续保留几何评分和 warning 分析。

## 10. 成本与效率

七个条件的平均 token 大致为：

- T：1724；
- I：5520；
- P：2329；
- TI：5826；
- TP：2790；
- IP：6560；
- TIP：6817。

图像 token 是主要成本来源。TIP 的平均 token 约为 TP 的 2.44 倍，但两者 joint quality 几乎相同：

- TP：0.2644；
- TIP：0.2649。

在当前设置下，TIP 相对 TP 没有表现出足以覆盖成本差异的收益。下一轮优先保留 T、P、TP，比继续运行完整七条件更经济。

## 11. 实验说明了什么

### 11.1 可以支持的结论

本实验支持以下结论：

1. 高细节程序化文本是很强的 CAD 生成输入，因为它已经包含操作顺序和尺寸。
2. 点云投影视图提供了较稳定的整体三维轮廓，使 P 成为有效率最高的条件。
3. 普通渲染图单独使用时无法稳定恢复准确尺寸和建模结构。
4. 文本加点云在成功样本上产生了最好的几何指标，说明模态互补具有潜力。
5. 组合模态会增加上下文和推理复杂度，也会提高 Plan 不合规和 CAD 执行失败风险。
6. 当前失败的主要来源是 Plan v2 接口遵循，而不是 JSON 语法，也不是 API 可用性。
7. 当前数据不足以证明多模态相对强单模态有统计显著的整体提升。

### 11.2 不能支持的结论

本实验不能证明：

- 多模态对所有 CAD 生成任务都无效；
- 图像模态天然没有价值；
- 原生点云输入不会带来提升；
- 其他模型会得到相同结果；
- 普通用户短文本下也不会出现明显互补；
- 自由 CadQuery 生成一定不如 HarnessCAD；
- 当前 Plan v2 已覆盖全部 BenchCAD 几何。

## 12. 主要限制

### 12.1 文本过强

正式 T 使用 L3，而不是 L1。L3 包含操作顺序和近似尺寸，接近结构化 CAD 配方，可能显著压缩图像和点云的边际收益。

### 12.2 点云不是原生输入

P 是点云的四视角 PNG 编码，不是原生三维 token。它会丢失部分点级拓扑和遮挡信息。

### 12.3 token 和图像数量不完全受控

不同条件发送的图像数量不同：

- I/P：4 张；
- TI/TP：4 张加可选文本；
- IP/TIP：8 张加可选文本。

因此模态类型与上下文长度、图像 token 成本存在混杂。

### 12.4 每个条件只运行一次

temperature 虽为 0，但云模型仍可能存在服务端非确定性。本轮没有对同一个 sample × condition 进行多次重复，无法估计模型自身运行波动。

### 12.5 只测试了一个模型

所有正式结果来自 `qwen3.7-plus`，不能直接推广到 GPT、Claude、Gemini 或其他 Qwen 模型。

### 12.6 Plan v2 仍不是完整 CAD 语言

表达能力审计估计有 27/100 样本无法被 Plan v2 完整表达，sweep、loft、shell、复杂 spline、torus、极坐标阵列等能力仍有限。

### 12.7 冲突与缺失模态实验尚未执行

已有 conflict scenes 因 URI、点云引用和声明问题被排除。因此本轮只研究完整、对齐模态下的组合效果，没有研究：

- 模态互相矛盾；
- 图像模糊或遮挡；
- 文本遗漏；
- 点云缺失；
- 单个模态被恶意扰动。

### 12.8 尚未执行 L1 压力测试

L1 已准备并核验，但当前 700 次主实验全部使用 L3。短文本 L1、减少图像视角、点云遮挡和重复 seed 属于下一阶段，不能算作本轮已完成结果。

## 13. 当前进度

截至本报告生成时，已经完成：

- 项目存储和本地计算环境审计；
- BenchCAD、渲染图、点云、文本和 STEP 资产对齐；
- 100 样本确定性分层抽样；
- 点云四视角编码；
- 七条件 prompt 隔离；
- HarnessCAD Plan v2 Schema 和编译能力扩展；
- Plan v1/v2 表达性审计；
- 冲突场景数据质量审计；
- dry-run 和真实 API 小规模闭环；
- 100 × 7 正式推理；
- HarnessCAD 执行和 STEP 导出；
- 几何评分；
- failure-aware 统计分析；
- bootstrap、Wilcoxon 和 Holm 校正；
- 失败分类、token 和延迟汇总；
- 13 项核心自动化测试。

当前主实验矩阵完成度为 700/700，分析完成度为 100%。

尚未完成、应视为下一阶段的工作包括：

- Plan v2.1 安全修复层；
- 对现有失败输出进行离线重放；
- 把 HarnessCAD 校验错误反馈给模型进行一次定向修正；
- L1 短文本主实验；
- 高难度样本多次重复；
- 视图减少、模糊、遮挡和点云缺失压力测试；
- 修复后的 conflict scenes 实验；
- 第二个模型的交叉验证；
- 更完整 CAD 操作和 oracle Plan 几何上限测试。

## 14. 下一阶段建议

### 14.1 第一优先级：降低 Plan 失败

建议发布 Plan v2.1，重点改进：

- 把 revolve 和 rotate 改成更容易生成的显式结构；
- 对合法非单位轴向量做确定性归一化；
- 安全转换明确可转换的数字字符串；
- 自动闭合首尾只差闭合点的 polygon；
- 删除连续重复点；
- 保留所有修复日志；
- 禁止自动猜测缺失尺寸、中心或操作。

先使用现有 700 个 raw response 进行离线重放，不调用 API，即可量化“接口修复本身”能挽救多少任务。

### 14.2 第二优先级：错误反馈修正

对本地无法安全修复的解析和 Schema 失败任务，把具体校验错误以及原始 Plan 发回模型，只允许一次定向修正。

需要分别报告：

- first-pass success；
- deterministic repair success；
- model repair success；
- 最终几何质量。

这样可以区分模型第一次生成能力和 Harness 修复能力。

### 14.3 第三优先级：聚焦 T、P、TP

下一轮不必立即重复七个条件。建议保留：

- T：强文本基线；
- P：高有效率三维基线；
- TP：最有潜力的互补组合。

在修复后的 Plan 接口上重新运行 100 × 3，可以用约本轮三分之一的请求量回答最关键问题。

主要预注册比较应为：

- TP 对 T；
- TP 对 P。

“相对每个样本最佳单模态”的 oracle 指标继续作为保守次要指标，不应取代直接配对比较。

### 14.4 第四优先级：使用 L1 检验真正互补

L3 已包含大量建模信息。为了测试视觉和点云是否能补充自然语言，应在同一批样本上改用 L1。

建议先选择 30 个 hard 样本，运行：

- L1-T；
- L1-P；
- L1-TP；
- 每个条件重复 3 次。

如果 L1-TP 明显高于 L1-T，而 L3-TP 与 L3-T 接近，就可以证明视觉/点云的价值主要出现在文本信息不足时。

### 14.5 第五优先级：控制成本和上下文

需要控制以下混杂因素：

- 固定图像数量；
- 固定总像素；
- 尽量匹配 token 预算；
- 比较单视图、双视图和四视图；
- 比较普通渲染与点云视图的等量输入；
- 单独记录图像 token 和文本 token。

## 15. 最终结论

本轮工作已经建立了一条完整、可恢复、可审计的多模态 CAD 推理实验链路：

`BenchCAD 多模态输入 → Qwen Plan v2 → HarnessCAD 校验与执行 → STEP → 几何评分 → 配对统计`

从工程角度，700 个正式任务已全部完成，数据、prompt、模型响应、HarnessCAD trace、预测 STEP 和分析结果均已保存。

从研究角度，实验观察到文本与点云组合在成功样本上具有更好的几何质量，特别是在高难度样本中出现正向趋势。但由于组合条件的 Schema 和执行失败率较高，失败感知的整体收益没有形成稳定统计证据。

因此当前最重要的结论是：

> 多模态并非没有提供额外几何信息；真正限制结果的是模型把这些信息稳定转换成严格、可执行 CAD Plan 的能力。

后续工作的优先顺序应是：

1. 修复模型与 Plan v2 的接口契约；
2. 离线重放现有失败结果；
3. 聚焦 T、P、TP 做低成本复验；
4. 使用 L1 短文本和 hard 样本检验真正的信息互补；
5. 最后再扩大到冲突场景、重复推理和多模型验证。

## 16. 结果与复核文件

完整实验目录：

`experiments/rq2_multimodal_harness`

主要复核文件：

- 实验配置：`configs/pilot.yaml`
- 100 样本输入清单：`outputs/pilot_v2/manifest.jsonl`
- 准备元数据：`outputs/pilot_v2/prepare_meta.json`
- 固定任务顺序：`outputs/pilot_v2/task_order.json`
- 每任务完整状态：`outputs/pilot_v2/state`
- 预测 STEP：由各任务 state 中的 `result_step_path` 定位
- 自动生成指标摘要：`outputs/pilot_v2/analysis/report_zh.md`
- 总体分析 JSON：`outputs/pilot_v2/analysis/analysis.json`
- 条件汇总：`outputs/pilot_v2/analysis/condition_summary.csv`
- 流水线汇总：`outputs/pilot_v2/analysis/pipeline_summary.csv`
- 失败汇总：`outputs/pilot_v2/analysis/failure_summary.csv`
- 逐任务指标：`outputs/pilot_v2/analysis/task_rows.csv`
- 分层结果：`outputs/pilot_v2/analysis/stratified_summary.csv`
- API 用量：`outputs/pilot_v2/analysis/usage_summary.csv`
- Plan 表达能力审计：`outputs/pilot_v2/audits/expressivity.json`
- 冲突场景审计：`outputs/pilot_v2/audits/conflict_scenes.json`

