from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class EncoderConfig:
    """点云 encoder 的统一配置。"""

    target_num_points: int = 2048
    input_dims: int = 3
    feature_dim: int = 256
    backbone: str = "statistical"
    normalize: str = "unit_sphere"
    sampling: str = "deterministic"
    padding: str = "repeat"
    include_normals: bool = False
    seed: int = 42
    dtype: str = "float32"
    batch_size: int = 16
    max_batch_points: int = 262144
    debug: bool = False
    projection_scale: float = 0.05
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_num_points <= 0:
            raise ValueError("target_num_points must be positive")
        if self.input_dims < 3:
            raise ValueError("input_dims must be at least 3")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_batch_points <= 0:
            raise ValueError("max_batch_points must be positive")
        if self.normalize not in {"none", "unit_sphere", "unit_cube"}:
            raise ValueError("normalize must be one of: none, unit_sphere, unit_cube")
        if self.sampling not in {"deterministic", "random"}:
            raise ValueError("sampling must be one of: deterministic, random")
        if self.padding not in {"repeat", "zeros"}:
            raise ValueError("padding must be one of: repeat, zeros")
        if self.backbone not in {"statistical", "random_projection"}:
            raise ValueError("backbone must be one of: statistical, random_projection")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be one of: float32, float64")

    @property
    def effective_input_dims(self) -> int:
        return 6 if self.include_normals else self.input_dims

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(config: EncoderConfig | Mapping[str, Any] | str | Path | None = None) -> EncoderConfig:
    """从 dataclass、字典、YAML/JSON 文件或空值加载 encoder 配置。"""
    if config is None:
        return EncoderConfig()
    if isinstance(config, EncoderConfig):
        return config
    if isinstance(config, (str, Path)):
        path = Path(config)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"encoder config must be a mapping: {path}")
        return load_config(data)
    data = dict(config)
    known_fields = {field_name for field_name in EncoderConfig.__dataclass_fields__ if field_name != "extra"}
    known_values = {key: value for key, value in data.items() if key in known_fields}
    extra = {key: value for key, value in data.items() if key not in known_fields}
    if "extra" in data and isinstance(data["extra"], Mapping):
        extra.update(dict(data["extra"]))
    return EncoderConfig(**known_values, extra=extra)


def save_config(config: EncoderConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, allow_unicode=True, sort_keys=False)


def check_dependencies() -> dict[str, str]:
    """返回 encoder 必需依赖的可用版本信息，缺失时抛出清晰错误。"""
    required = ["numpy", "yaml"]
    versions: dict[str, str] = {}
    missing: list[str] = []
    for module_name in required:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
            continue
        versions[module_name] = str(getattr(module, "__version__", "unknown"))
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"缺少 encoder 依赖：{joined}。请在 cad_data_gen 环境中安装 requirements.txt。")
    return versions
