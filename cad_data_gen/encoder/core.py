from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import EncoderConfig, load_config
from .preprocess import PreprocessMetadata, PreprocessResult, load_and_preprocess_point_cloud, preprocess_point_cloud


@dataclass(frozen=True)
class EncodeResult:
    feature: np.ndarray
    metadata: PreprocessMetadata
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature.tolist(),
            "metadata": self.metadata.to_dict(),
            "stats": self.stats,
        }


@dataclass(frozen=True)
class BatchEncodeResult:
    features: np.ndarray
    results: list[EncodeResult]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": self.features.tolist(),
            "results": [result.to_dict() for result in self.results],
            "stats": self.stats,
        }


class PointCloudEncoder:
    """纯 numpy 点云 encoder baseline，提供稳定固定维度特征。"""

    def __init__(self, config: EncoderConfig | dict[str, Any] | str | Path | None = None) -> None:
        self.config = load_config(config)
        self._projection = self._build_projection()

    @property
    def feature_dim(self) -> int:
        return int(self.config.feature_dim)

    def encode(
        self,
        points: np.ndarray,
        *,
        sample_id: str | None = None,
        source_path: str | Path | None = None,
    ) -> EncodeResult:
        started = time.perf_counter()
        preprocessed = preprocess_point_cloud(points, self.config, sample_id=sample_id, source_path=source_path)
        feature = self._encode_preprocessed(preprocessed.points)
        elapsed = time.perf_counter() - started
        stats = {
            "elapsed_seconds": elapsed,
            "feature_dim": int(feature.shape[0]),
            "dtype": str(feature.dtype),
            "backbone": self.config.backbone,
        }
        return EncodeResult(feature=feature, metadata=preprocessed.metadata, stats=stats)

    def encode_file(self, path: str | Path, *, sample_id: str | None = None) -> EncodeResult:
        started = time.perf_counter()
        preprocessed = load_and_preprocess_point_cloud(path, self.config, sample_id=sample_id)
        feature = self._encode_preprocessed(preprocessed.points)
        elapsed = time.perf_counter() - started
        stats = {
            "elapsed_seconds": elapsed,
            "feature_dim": int(feature.shape[0]),
            "dtype": str(feature.dtype),
            "backbone": self.config.backbone,
        }
        return EncodeResult(feature=feature, metadata=preprocessed.metadata, stats=stats)

    def encode_many(
        self,
        samples: Sequence[np.ndarray] | Iterable[np.ndarray],
        *,
        sample_ids: Sequence[str | None] | None = None,
    ) -> BatchEncodeResult:
        started = time.perf_counter()
        results: list[EncodeResult] = []
        ids = list(sample_ids) if sample_ids is not None else None
        for index, points in enumerate(samples):
            sample_id = ids[index] if ids is not None and index < len(ids) else None
            results.append(self.encode(points, sample_id=sample_id))
        features = self._stack_features(results)
        elapsed = time.perf_counter() - started
        stats = self._make_batch_stats(results, elapsed)
        return BatchEncodeResult(features=features, results=results, stats=stats)

    def encode_files(
        self,
        paths: Sequence[str | Path] | Iterable[str | Path],
        *,
        sample_ids: Sequence[str | None] | None = None,
    ) -> BatchEncodeResult:
        started = time.perf_counter()
        results: list[EncodeResult] = []
        ids = list(sample_ids) if sample_ids is not None else None
        for index, path in enumerate(paths):
            sample_id = ids[index] if ids is not None and index < len(ids) else None
            results.append(self.encode_file(path, sample_id=sample_id))
        features = self._stack_features(results)
        elapsed = time.perf_counter() - started
        stats = self._make_batch_stats(results, elapsed)
        return BatchEncodeResult(features=features, results=results, stats=stats)

    def _encode_preprocessed(self, points: np.ndarray) -> np.ndarray:
        if self.config.backbone == "random_projection":
            return self._random_projection_features(points)
        return self._statistical_features(points)

    def _statistical_features(self, points: np.ndarray) -> np.ndarray:
        return self._fit_feature_dim(self._base_descriptors(points))

    def _random_projection_features(self, points: np.ndarray) -> np.ndarray:

nts))
        projected = descriptors @ self._projectionprojected = np.tanh(projected * float(self.config.projection_scal
e))
        norm = float(np.linalg.norm(projected))
        if norm > 1e-12:
            projected = projected / norm
        return projected.astype(self.config.dtype, copy=False)

    def _base_descriptors(self, points: np.ndarray) -> np.ndarray:
        arr = np.asarray(points, dtype=np.float64)
        xyz = arr[:, :3]
        centroid = xyz.mean(axis=0)
        centered = xyz - centroid
        radius = np.linalg.norm(centered, axis=1)
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)percentiles = np.percentile(xyz, [5, 25, 50, 75, 95], axis=0).res
hape(-1)
        base = np.concatenate(
            [
                centroid,
                xyz.std(axis=0),
                mins,
                maxs,
                maxs - mins,
                percentiles,
                np.asarray(
                    [
                        radius.mean(),
                        radius.std(),
                        radius.min(),
                        radius.max(),
                        float(np.mean(radius <= 0.25)),
                        float(np.mean(radius <= 0.50)),
                        float(np.mean(radius <= 0.75)),
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        if arr.shape[1] > 3:
            attrs = arr[:, 3:]base = np.concatenate([base, attrs.mean(axis=0), attrs.std(ax
is=0), attrs.min(axis=0), attrs.max(axis=0)])
        return base

    @staticmethoddef _fit_descriptor_dim(values: np.ndarray, target: int = 64) -> np.n
darray:
        feature = np.asarray(values, dtype=np.float64).reshape(-1)
        if feature.size < target:
            repeats = int(np.ceil(target / max(feature.size, 1)))
            feature = np.tile(feature, repeats)
        feature = feature[:target]
        norm = float(np.linalg.norm(feature))
        if norm > 1e-12:
            feature = feature / norm
        return feature

    def _fit_feature_dim(self, values: np.ndarray) -> np.ndarray:
        feature = np.asarray(values, dtype=np.float64).reshape(-1)
        target = self.feature_dim
        if feature.size < target:
            repeats = int(np.ceil(target / max(feature.size, 1)))
            feature = np.tile(feature, repeats)
        feature = feature[:target]
        norm = float(np.linalg.norm(feature))
        if norm > 1e-12:
            feature = feature / norm
        return feature.astype(self.config.dtype, copy=False)

    def _build_projection(self) -> np.ndarray:
        if self.config.backbone != "random_projection":
            return np.empty((0, 0), dtype=np.float64)
        rows = 64seed_material = f"{self.config.seed}:{self.config.feature_dim}:{s
elf.config.backbone}"
        digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**32)
        rng = np.random.default_rng(seed)return rng.standard_normal((rows, int(self.config.feature_dim))).
astype(np.float64)

    @staticmethod
    def _stack_features(results: Sequence[EncodeResult]) -> np.ndarray:
        if not results:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([result.feature for result in results], axis=0)

    @staticmethoddef _make_batch_stats(results: Sequence[EncodeResult], elapsed: floa
t) -> dict[str, Any]:
        count = len(results)
        return {
            "elapsed_seconds": elapsed,
            "num_samples": count,"feature_shape": list(PointCloudEncoder._stack_features(resul
ts).shape),
            "average_latency_seconds": elapsed / count if count else 0.0,
        }

def load_encoder(config: EncoderConfig | dict[str, Any] | str | Path | No
ne = None) -> PointCloudEncoder:
    return PointCloudEncoder(config)

def encode(points: np.ndarray, config: EncoderConfig | dict[str, Any] | s
tr | Path | None = None) -> EncodeResult:return load_encoder(config).encode(points)
