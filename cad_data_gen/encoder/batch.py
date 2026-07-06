from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .config import EncoderConfig
from .core import EncodeResult, PointCloudEncoder, load_encoder
from .preprocess import SampleError, make_sample_error


@dataclass(frozen=True)
class BatchSample:
    index: int
    sample_id: str | None
    point_path: Path
    raw_record: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackedEncodeRecord:
    index: int
    sample_id: str | None
    point_path: str
    status: str
    feature: list[float] | None
    metadata: dict[str, Any] | None
    stats: dict[str, Any] | None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "sample_id": self.sample_id,
            "point_path": self.point_path,
            "status": self.status,
            "feature": self.feature,
            "metadata": self.metadata,
            "stats": self.stats,
            "error": self.error,
        }


@dataclass(frozen=True)
class BatchTrackingSummary:
    total: int
    success: int
    failed: int
    elapsed_seconds: float
    average_latency_seconds: float
    feature_dim: int
    output_path: str | None = None
    failure_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "elapsed_seconds": self.elapsed_seconds,
            "average_latency_seconds": self.average_latency_seconds,
            "feature_dim": self.feature_dim,
            "output_path": self.output_path,
            "failure_path": self.failure_path,
        }


def resolve_point_path(record: Mapping[str, Any], dataset_root: str | Path | None = None) -> Path:
    value = record.get("point_path") or record.get("point_cloud_path") or record.get("points_path")
    if not isinstance(value, str) or not value:
        raise ValueError("manifest record has no point_path/point_cloud_path/points_path")
    path = Path(value)
    if not path.is_absolute() and dataset_root is not None:
        path = Path(dataset_root) / path
    return path


def read_manifest_samples(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    limit: int | None = None,
) -> list[BatchSample]:
    manifest = Path(manifest_path)
    root = Path(dataset_root) if dataset_root is not None else manifest.parent
    samples: list[BatchSample] = []
    with manifest.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if limit is not None and len(samples) >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            point_path = resolve_point_path(record, root)
            sample_id = record.get("sample_id")
            samples.append(
                BatchSample(
                    index=index,
                    sample_id=str(sample_id) if sample_id is not None else None,
                    point_path=point_path,
                    raw_record=dict(record),
                )
            )
    return samples


def iter_chunks(samples: Sequence[BatchSample], config: EncoderConfig) -> Iterator[list[BatchSample]]:
    chunk: list[BatchSample] = []
    max_samples = max(1, int(config.batch_size))
    max_points = max(1, int(config.max_batch_points))
    for sample in samples:
        chunk.append(sample)
        reached_samples = len(chunk) >= max_samples
        reached_points = len(chunk) * int(config.target_num_points) >= max_points
        if reached_samples or reached_points:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def encode_samples_tracked(
    samples: Sequence[BatchSample],
    encoder: PointCloudEncoder,
    *,
    output_path: str | Path | None = None,
    failure_path: str | Path | None = None,
    debug: bool | None = None,
) -> tuple[list[TrackedEncodeRecord], BatchTrackingSummary]:
    started = time.perf_counter()
    records: list[TrackedEncodeRecord] = []
    failures: list[dict[str, Any]] = []
    debug_enabled = encoder.config.debug if debug is None else bool(debug)

    for chunk in iter_chunks(samples, encoder.config):
        for sample in chunk:
            try:
                result = encoder.encode_file(sample.point_path, sample_id=sample.sample_id)
                stats = dict(result.stats)
                if debug_enabled:
                    stats["debug"] = {
                        "input_index": sample.index,
                        "batch_shape": list(result.metadata.processed_shape),
                    }
                records.append(_success_record(sample, result, stats))
            except Exception as exc:  # noqa: BLE001 - 样本级失败需要继续处理后续数据
                error = make_sample_error(
                    exc,
                    sample_id=sample.sample_id,
                    source_path=str(sample.point_path),
                    stage="encode",
                    context={"index": sample.index},
                )
                error_dict = error.to_dict()
                failures.append(error_dict)
                records.append(_failure_record(sample, error))

    elapsed = time.perf_counter() - started
    success_count = sum(1 for record in records if record.status == "success")
    failed_count = len(records) - success_count

    if output_path is not None:
        write_jsonl(output_path, (record.to_dict() for record in records))
    if failure_path is not None:
        write_jsonl(failure_path, failures)

    summary = BatchTrackingSummary(
        total=len(records),
        success=success_count,
        failed=failed_count,
        elapsed_seconds=elapsed,
        average_latency_seconds=elapsed / len(records) if records else 0.0,
        feature_dim=int(encoder.feature_dim),
        output_path=str(output_path) if output_path is not None else None,
        failure_path=str(failure_path) if failure_path is not None else None,
    )
    return records, summary


def encode_manifest_tracked(
    manifest_path: str | Path,
    *,
    config: EncoderConfig | dict[str, Any] | str | Path | None = None,
    dataset_root: str | Path | None = None,
    limit: int | None = None,
    output_path: str | Path | None = None,
    failure_path: str | Path | None = None,
    debug: bool | None = None,
) -> tuple[list[TrackedEncodeRecord], BatchTrackingSummary]:
    encoder = load_encoder(config)
    samples = read_manifest_samples(manifest_path, dataset_root=dataset_root, limit=limit)
    return encode_samples_tracked(
        samples,
        encoder,
        output_path=output_path,
        failure_path=failure_path,
        debug=debug,
    )


def tracked_records_to_feature_array(records: Sequence[TrackedEncodeRecord]) -> np.ndarray:
    features = [record.feature for record in records if record.status == "success" and record.feature is not None]
    if not features:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(features, dtype=np.float32)


def _success_record(sample: BatchSample, result: EncodeResult, stats: dict[str, Any]) -> TrackedEncodeRecord:
    return TrackedEncodeRecord(
        index=sample.index,
        sample_id=sample.sample_id,
        point_path=str(sample.point_path),
        status="success",
        feature=result.feature.astype(np.float32, copy=False).tolist(),
        metadata=result.metadata.to_dict(),
        stats=stats,
        error=None,
    )


def _failure_record(sample: BatchSample, error: SampleError) -> TrackedEncodeRecord:
    return TrackedEncodeRecord(
        index=sample.index,
        sample_id=sample.sample_id,
        point_path=str(sample.point_path),
        status="failed",
        feature=None,
        metadata=None,
        stats=None,
        error=error.to_dict(),
    )
