from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class AssetPaths:
    """单个样本在基础资产阶段使用的标准输出路径。"""

    output_dir: Path
    sample_id: str

    @property
    def point_path(self) -> Path:
        return self.output_dir / "points" / f"{self.sample_id}.npz"

    @property
    def image_dir(self) -> Path:
        return self.output_dir / "images" / self.sample_id

    @property
    def mesh_dir(self) -> Path:
        return self.output_dir / "meshes" / self.sample_id


@dataclass(frozen=True)
class SampleContext:
    """跨 STEP 转换、点云、渲染、遮挡和描述阶段共享的样本上下文。"""

    step_path: Path
    rel_step: Path
    sample_id: str
    paths: AssetPaths


@dataclass
class StageError:
    """统一记录阶段失败，便于 manifest/failures.jsonl 和外层日志排查。"""

    stage: str
    error_type: str
    error: str
    step_path: str
    sample_id: Optional[str] = None
    backend: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in (None, {}, [])}


@dataclass
class StageStatus:
    """单样本阶段状态，用于后续断点续跑和阶段级进度统计。"""

    sample_id: str
    status: str
    completed_stages: list[str] = field(default_factory=list)
    failed_stage: Optional[str] = None
    error: Optional[StageError] = None
    outputs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_done(self, stage: str, **outputs: str) -> None:
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        self.outputs.update({key: str(value) for key, value in outputs.items()})
        self.status = "running"

    def mark_failed(self, error: StageError) -> None:
        self.status = "failed"
        self.failed_stage = error.stage
        self.error = error

    def mark_success(self) -> None:
        self.status = "done"
        self.failed_stage = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.error is not None:
            data["error"] = self.error.to_dict()
        return data
