from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import EncoderConfig


@dataclass(frozen=True)
class PreprocessMetadata:
    sample_id: str | None
    original_shape: tuple[int, ...]
    processed_shape: tuple[int, ...]
    sampling_action: str
    normalize: str
    center: list[float]
    scale: float
    dtype: str
    source_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "original_shape": list(self.original_shape),
            "processed_shape": list(self.processed_shape),
            "sampling_action": self.sampling_action,
            "normalize": self.normalize,
            "center": self.center,
            "scale": self.scale,
            "dtype": self.dtype,
            "source_path": self.source_path,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SampleError:
    sample_id: str | None
    source_path: str | None
    stage: str
    error_type: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_path": self.source_path,
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
            "context": self.context,
        }


class PointCloudValidationError(ValueError):
    """点云输入不符合 encoder 协议。"""


@dataclass(frozen=True)
class PreprocessResult:
    points: np.ndarray
    metadata: PreprocessMetadata


def make_sample_error(
    exc: BaseException,
    *,
    sample_id: str | None,
    source_path: str | None,
    stage: str,
    context: dict[str, Any] | None = None,
) -> SampleError:
    return SampleError(
        sample_id=sample_id,
        source_path=source_path,
        stage=stage,
        error_type=type(exc).__name__,
        message=str(exc),
        context=context or {},
    )


def load_point_cloud(path: str | Path, *, include_normals: bool = False, normal_policy: str = "error") -> np.ndarray:
    """读取现有数据集中的 .npz/.npy 点云文件。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"point cloud file not found: {source}")
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as data:
            if "points" not in data:
                raise PointCloudValidationError(f"npz file has no 'points' array: {source}")
            points = np.asarray(data["points"])
            if include_normals:
                if "normals" not in data:
                    if normal_policy == "fallback":
                        normals = np.zeros((points.shape[0], 3), dtype=points.dtype)
                    else:
                        raise PointCloudValidationError(f"npz file has no 'normals' array: {source}")
                else:
                    normals = np.asarray(data["normals"])
                    if normals.shape[0] != points.shape[0]:
                        if normal_policy == "fallback":
                            normals = np.zeros((points.shape[0], 3), dtype=points.dtype)
                        else:
                            raise PointCloudValidationError(
                                f"normals row count mismatch in {source}: {normals.shape[0]} != {points.shape[0]}"
                            )
                points = np.concatenate([points, normals[:, :3]], axis=1)
            return points
    if source.suffix.lower() == ".npy":
        points = np.asarray(np.load(source, allow_pickle=False))
        if include_normals and points.shape[1] < 6:
            if normal_policy == "fallback":
                normals = np.zeros((points.shape[0], 3), dtype=points.dtype)
                return np.concatenate([points[:, :3], normals], axis=1)
            raise PointCloudValidationError(f"npy file has no normals columns: {source}")
        return points
    raise PointCloudValidationError(f"unsupported point cloud file extension: {source.suffix}")


def validate_point_cloud(points: np.ndarray, config: EncoderConfig) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim != 2:
        raise PointCloudValidationError(f"point cloud must be a 2D array, got shape={arr.shape}")
    if arr.shape[0] == 0:
        raise PointCloudValidationError("point cloud is empty")
    min_dims = config.effective_input_dims if config.include_normals else 3
    if arr.shape[1] < min_dims:
        raise PointCloudValidationError(f"point cloud must have at least {min_dims} columns, got {arr.shape[1]}")
    if not np.issubdtype(arr.dtype, np.number):
        raise PointCloudValidationError(f"point cloud dtype must be numeric, got {arr.dtype}")
    if not np.isfinite(arr).all():
        raise PointCloudValidationError("point cloud contains NaN or Inf")
    return arr.astype(config.dtype, copy=False)


def sample_or_pad_points(points: np.ndarray, config: EncoderConfig) -> tuple[np.ndarray, str]:
    target = int(config.target_num_points)
    count = int(points.shape[0])
    if count == target:
        return points.copy(), "unchanged"
    if count > target:
        if config.sampling == "random":
            rng = np.random.default_rng(int(config.seed))
            indices = np.sort(rng.choice(count, size=target, replace=False))
        else:
            indices = np.linspace(0, count - 1, num=target, dtype=np.int64)
        return points[indices].copy(), f"sampled:{count}->{target}"
    pad_count = target - count
    if config.padding == "zeros":
        padding = np.zeros((pad_count, points.shape[1]), dtype=points.dtype)
    else:
        repeats = np.resize(np.arange(count, dtype=np.int64), pad_count)
        padding = points[repeats]
    return np.concatenate([points, padding], axis=0), f"padded:{count}->{target}"


def normalize_points(points: np.ndarray, config: EncoderConfig) -> tuple[np.ndarray, np.ndarray, float, str]:
    if config.normalize == "none":
        return points.astype(config.dtype, copy=True), np.zeros(3, dtype=np.float64), 1.0, "none"

    result = points.astype(np.float64, copy=True)
    xyz = result[:, :3]
    center = np.mean(xyz, axis=0)
    xyz -= center
    if config.normalize == "unit_cube":
        scale = float(np.max(np.ptp(xyz, axis=0)))
    else:
        scale = float(np.max(np.linalg.norm(xyz, axis=1)))
    if scale <= 1e-12:
        scale = 1.0
    result[:, :3] = xyz / scale
    return result.astype(config.dtype, copy=False), center.astype(np.float64), float(scale), config.normalize


def preprocess_point_cloud(
    points: np.ndarray,
    config: EncoderConfig,
    *,
    sample_id: str | None = None,
    source_path: str | Path | None = None,
    warnings: list[str] | None = None,
) -> PreprocessResult:
    source = str(source_path) if source_path is not None else None
    validated = validate_point_cloud(points, config)
    if config.include_normals:
        selected = validated[:, :6]
    else:
        selected = validated[:, : config.input_dims]
    resized, action = sample_or_pad_points(selected, config)
    normalized, center, scale, normalize_name = normalize_points(resized, config)
    metadata = PreprocessMetadata(
        sample_id=sample_id,
        original_shape=tuple(int(v) for v in validated.shape),
        processed_shape=tuple(int(v) for v in normalized.shape),
        sampling_action=action,
        normalize=normalize_name,
        center=[float(v) for v in center.tolist()],
        scale=float(scale),
        dtype=str(normalized.dtype),
        source_path=source,
        warnings=list(warnings or []),
    )
    return PreprocessResult(points=normalized, metadata=metadata)


def load_and_preprocess_point_cloud(
    path: str | Path,
    config: EncoderConfig,
    *,
    sample_id: str | None = None,
) -> PreprocessResult:
    normal_policy = str((config.extra or {}).get("normal_policy", "error"))
    warnings: list[str] = []
    if config.include_normals:
        source = Path(path)
        if source.suffix.lower() == ".npz":
            with np.load(source, allow_pickle=False) as data:
                if "normals" not in data:
                    warnings.append("normals missing; fallback zero normals were used" if normal_policy == "fallback" else "normals missing")
                elif "points" in data and np.asarray(data["normals"]).shape[0] != np.asarray(data["points"]).shape[0]:
                    warnings.append("normals row count mismatch; fallback zero normals were used" if normal_policy == "fallback" else "normals row count mismatch")
        elif source.suffix.lower() == ".npy":
            raw = np.load(source, allow_pickle=False, mmap_mode="r")
            if raw.ndim == 2 and raw.shape[1] < 6:
                warnings.append("normals columns missing; fallback zero normals were used" if normal_policy == "fallback" else "normals columns missing")
    points = load_point_cloud(path, include_normals=config.include_normals, normal_policy=normal_policy)
    return preprocess_point_cloud(points, config, sample_id=sample_id, source_path=path, warnings=warnings)
