"""abc_batch 工作目录布局常量与辅助函数。

约定的 `<work-root>/` 目录结构：

    <work-root>/
      extracted/
        step/<archive_stem>/...        # 每个 STEP 压缩包独立子目录
        ofs/<archive_stem>/...         # 每个 OFS 压缩包独立子目录
        step/<archive_stem>.done       # JSON 标记，记录解压元数据
        ofs/<archive_stem>.done
      assets/                          # build_step_assets.py 输出
        manifest.jsonl
        ...
      occlusion/                       # build_occlusion_assets.py 输出
        manifest.jsonl
        audit.jsonl
        summary.json
      contexts/
        compact_contexts.jsonl         # clean_cad_contexts.py 输出
      descriptions/                    # batch_describe_cad_with_qwen.py 输出
        descriptions.jsonl
        failures.jsonl
        batches.jsonl
      pairing_manifest.jsonl           # 主样本清单（paired + step_only）
      orphan_ofs.jsonl                 # 仅有 OFS、无 STEP 的样本
      pairing_summary.json
      extract_failures.jsonl
      extract_summary.json
      global_samples.jsonl             # 带输入大小估计的全局样本清单
      batches/                         # 分批 manifest、状态与摘要
        batches_index.jsonl
        <batch_id>/manifest.jsonl
        <batch_id>/state.json
        <batch_id>/summary.json
      stage_assets.log
      make_batches.log
      stage_occlusion.log
      stage_describe.log
      run_state.json                   # 顶层每阶段状态
      final_summary.json               # 顶层汇总

所有阶段统一从这里读取路径常量，避免散落的硬编码字符串。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkRootLayout:
    """工作目录的所有路径常量。"""

    work_root: Path

    # 解压目录
    @property
    def extracted_root(self) -> Path:
        return self.work_root / "extracted"

    @property
    def extracted_step_root(self) -> Path:
        return self.extracted_root / "step"

    @property
    def extracted_ofs_root(self) -> Path:
        return self.extracted_root / "ofs"

    # 阶段输出目录
    @property
    def assets_dir(self) -> Path:
        return self.work_root / "assets"

    @property
    def assets_manifest(self) -> Path:
        return self.assets_dir / "manifest.jsonl"

    @property
    def occlusion_dir(self) -> Path:
        return self.work_root / "occlusion"

    @property
    def contexts_dir(self) -> Path:
        return self.work_root / "contexts"

    @property
    def compact_contexts(self) -> Path:
        return self.contexts_dir / "compact_contexts.jsonl"

    @property
    def descriptions_dir(self) -> Path:
        return self.work_root / "descriptions"

    @property
    def batches_dir(self) -> Path:
        return self.work_root / "batches"

    @property
    def batches_index(self) -> Path:
        return self.batches_dir / "batches_index.jsonl"

    @property
    def global_samples_manifest(self) -> Path:
        return self.work_root / "global_samples.jsonl"

    # 配对产物
    @property
    def pairing_manifest(self) -> Path:
        return self.work_root / "pairing_manifest.jsonl"

    @property
    def step_only_manifest(self) -> Path:
        return self.work_root / "step_only_manifest.jsonl"

    @property
    def orphan_ofs(self) -> Path:
        return self.work_root / "orphan_ofs.jsonl"

    @property
    def pairing_summary(self) -> Path:
        return self.work_root / "pairing_summary.json"

    # 解压副产物
    @property
    def extract_failures(self) -> Path:
        return self.work_root / "extract_failures.jsonl"

    @property
    def extract_summary(self) -> Path:
        return self.work_root / "extract_summary.json"

    # 阶段日志
    @property
    def stage_assets_log(self) -> Path:
        return self.work_root / "stage_assets.log"

    @property
    def make_batches_log(self) -> Path:
        return self.work_root / "make_batches.log"

    @property
    def stage_occlusion_log(self) -> Path:
        return self.work_root / "stage_occlusion.log"

    @property
    def stage_describe_log(self) -> Path:
        return self.work_root / "stage_describe.log"

    # 顶层
    @property
    def run_state(self) -> Path:
        return self.work_root / "run_state.json"

    @property
    def final_summary(self) -> Path:
        return self.work_root / "final_summary.json"

    def ensure_dirs(self) -> None:
        """创建所有顶层目录（不创建 archive_stem 子目录，留给解压阶段）。"""
        for d in (
            self.work_root,
            self.extracted_root,
            self.extracted_step_root,
            self.extracted_ofs_root,
            self.assets_dir,
            self.occlusion_dir,
            self.contexts_dir,
            self.descriptions_dir,
            self.batches_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def work_root_layout(work_root: str | Path) -> WorkRootLayout:
    """构造一个 `WorkRootLayout`，path 会被规范化为绝对路径。"""
    return WorkRootLayout(work_root=Path(work_root).expanduser().resolve())


# 解压侧支持的扩展名（按优先级匹配）
SUPPORTED_ARCHIVE_SUFFIXES: tuple[str, ...] = (
    ".tar.gz",
    ".tar.bz2",
    ".tgz",
    ".tbz2",
    ".tar",
    ".zip",
    ".7z",
)


def archive_stem(archive_path: Path) -> str:
    """剥掉所有支持的复合后缀，得到压缩包的 stem。

    例如 `abc_0001.tar.gz` -> `abc_0001`，`step_part1.zip` -> `step_part1`。
    对未识别后缀回退到 `Path.stem`。
    """
    name = archive_path.name
    lower = name.lower()
    for suf in SUPPORTED_ARCHIVE_SUFFIXES:
        if lower.endswith(suf):
            return name[: -len(suf)]
    return archive_path.stem
