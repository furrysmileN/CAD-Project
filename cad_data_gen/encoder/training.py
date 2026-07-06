from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .batch import BatchSample, read_manifest_samples
from .config import EncoderConfig, load_config, save_config
from .core import PointCloudEncoder, load_encoder
from .preprocess import SampleError, make_sample_error


@dataclass(frozen=True)
class TrainingConfig:
    manifest_path: str
    output_dir: str
    dataset_root: str | None = None
    encoder: dict[str, Any] = field(default_factory=dict)
    label_field: str | None = None
    epochs: int = 5
    learning_rate: float = 0.05
    validation_split: float = 0.2
    seed: int = 42
    checkpoint_every: int = 1
    resume_checkpoint: str | None = None
    limit: int | None = None
    debug: bool = False

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1)")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetFeatures:
    features: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    samples: list[BatchSample]
    errors: list[SampleError]


@dataclass(frozen=True)
class TrainingState:
    weights: np.ndarray
    bias: np.ndarray
    epoch: int
    best_metric: float
    class_names: list[str]


@dataclass(frozen=True)
class TrainingResult:
    output_dir: str
    best_checkpoint: str
    last_checkpoint: str
    metrics_path: str
    final_metrics: dict[str, Any]
    failures_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_training_config(config: TrainingConfig | Mapping[str, Any] | str | Path) -> TrainingConfig:
    if isinstance(config, TrainingConfig):
        return config
    if isinstance(config, (str, Path)):
        path = Path(config)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"training config must be a mapping: {path}")
        return load_training_config(data)
    return TrainingConfig(**dict(config))


def extract_dataset_features(samples: Sequence[BatchSample], encoder: PointCloudEncoder, label_field: str | None = None) -> DatasetFeatures:
    features: list[np.ndarray] = []
    label_names: list[str] = []
    kept_samples: list[BatchSample] = []
    errors: list[SampleError] = []
    for sample in samples:
        try:
            result = encoder.encode_file(sample.point_path, sample_id=sample.sample_id)
            features.append(result.feature.astype(np.float64, copy=False))
            label_names.append(_label_from_record(sample, label_field))
            kept_samples.append(sample)
        except Exception as exc:  # noqa: BLE001 - 训练数据抽取需要记录坏样本并继续
            errors.append(
                make_sample_error(
                    exc,
                    sample_id=sample.sample_id,
                    source_path=str(sample.point_path),
                    stage="train_feature_extract",
                    context={"index": sample.index},
                )
            )
    if not features:
        raise ValueError("no valid training samples after feature extraction")
    class_names = sorted(set(label_names))
    label_to_index = {name: index for index, name in enumerate(class_names)}
    labels = np.asarray([label_to_index[name] for name in label_names], dtype=np.int64)
    return DatasetFeatures(
        features=np.stack(features, axis=0),
        labels=labels,
        class_names=class_names,
        samples=kept_samples,
        errors=errors,
    )


def train_from_config(config: TrainingConfig | Mapping[str, Any] | str | Path) -> TrainingResult:
    train_config = load_training_config(config)
    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_config = load_config(train_config.encoder)
    encoder = load_encoder(encoder_config)
    samples = read_manifest_samples(
        train_config.manifest_path,
        dataset_root=train_config.dataset_root,
        limit=train_config.limit,
    )
    dataset = extract_dataset_features(samples, encoder, train_config.label_field)
    train_indices, val_indices = _split_indices(len(dataset.labels), train_config.validation_split, train_config.seed)
    state = _initial_state(dataset.features.shape[1], len(dataset.class_names), train_config.seed, dataset.class_names)
    if train_config.resume_checkpoint:
        state = load_checkpoint(train_config.resume_checkpoint)

    save_config(encoder_config, output_dir / "encoder_config.yaml")
    _write_json(output_dir / "training_config.json", train_config.to_dict())
    failures_path = output_dir / "feature_failures.jsonl"
    if dataset.errors:
        _write_jsonl(failures_path, [error.to_dict() for error in dataset.errors])
    else:
        failures_path = None

    history: list[dict[str, Any]] = []
    best_checkpoint = output_dir / "best_checkpoint.npz"
    last_checkpoint = output_dir / "last_checkpoint.npz"
    best_metric = float(state.best_metric)
    started = time.perf_counter()
    try:
        for epoch in range(int(state.epoch) + 1, train_config.epochs + 1):
            state, train_metrics = _train_one_epoch(state, dataset.features, dataset.labels, train_indices, train_config.learning_rate)
            val_metrics = evaluate_state(state, dataset.features, dataset.labels, val_indices)
            current_metric = float(val_metrics.get("accuracy", 0.0))
            if current_metric >= best_metric:
                best_metric = current_metric
                state = TrainingState(state.weights, state.bias, epoch, best_metric, state.class_names)
                save_checkpoint(best_checkpoint, state, {"epoch": epoch, "split": "best", **val_metrics})
            if epoch % train_config.checkpoint_every == 0 or epoch == train_config.epochs:
                state = TrainingState(state.weights, state.bias, epoch, best_metric, state.class_names)
                save_checkpoint(last_checkpoint, state, {"epoch": epoch, "split": "last", **val_metrics})
            history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
    except Exception as exc:
        interrupted = TrainingState(state.weights, state.bias, int(state.epoch), best_metric, state.class_names)
        save_checkpoint(output_dir / "interrupted_checkpoint.npz", interrupted, {"error_type": type(exc).__name__, "error": str(exc)})
        raise

    elapsed = time.perf_counter() - started
    final_metrics = {
        "epochs": train_config.epochs,
        "elapsed_seconds": elapsed,
        "num_samples": int(len(dataset.labels)),
        "num_classes": int(len(dataset.class_names)),
        "best_accuracy": best_metric,
        "history": history,
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, final_metrics)
    if not best_checkpoint.is_file():
        save_checkpoint(best_checkpoint, state, {"epoch": int(state.epoch), "split": "best_fallback"})
    if not last_checkpoint.is_file():
        save_checkpoint(last_checkpoint, state, {"epoch": int(state.epoch), "split": "last_fallback"})
    return TrainingResult(
        output_dir=str(output_dir),
        best_checkpoint=str(best_checkpoint),
        last_checkpoint=str(last_checkpoint),
        metrics_path=str(metrics_path),
        final_metrics=final_metrics,
        failures_path=str(failures_path) if failures_path is not None else None,
    )


def evaluate_checkpoint(checkpoint_path: str | Path, features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    state = load_checkpoint(checkpoint_path)
    indices = np.arange(labels.shape[0], dtype=np.int64)
    return evaluate_state(state, features, labels, indices)


def evaluate_state(state: TrainingState, features: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    if indices.size == 0:
        indices = np.arange(labels.shape[0], dtype=np.int64)
    x = features[indices]
    y = labels[indices]
    logits = x @ state.weights + state.bias
    probs = _softmax(logits)
    loss = _cross_entropy(probs, y)
    preds = np.argmax(probs, axis=1)
    accuracy = float(np.mean(preds == y)) if y.size else 0.0
    return {
        "loss": float(loss),
        "accuracy": accuracy,
        "num_samples": int(y.size),
    }


def save_checkpoint(path: str | Path, state: TrainingState, metrics: Mapping[str, Any] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        weights=state.weights.astype(np.float32),
        bias=state.bias.astype(np.float32),
        epoch=np.asarray(state.epoch, dtype=np.int64),
        best_metric=np.asarray(state.best_metric, dtype=np.float32),
        class_names=np.asarray(state.class_names, dtype=object),
        metrics_json=json.dumps(dict(metrics or {}), ensure_ascii=False),
    )


def load_checkpoint(path: str | Path) -> TrainingState:
    source = Path(path)
    with np.load(source, allow_pickle=True) as data:
        return TrainingState(
            weights=np.asarray(data["weights"], dtype=np.float64),
            bias=np.asarray(data["bias"], dtype=np.float64),
            epoch=int(np.asarray(data["epoch"]).item()),
            best_metric=float(np.asarray(data["best_metric"]).item()),
            class_names=[str(v) for v in data["class_names"].tolist()],
        )


def _train_one_epoch(
    state: TrainingState,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    learning_rate: float,
) -> tuple[TrainingState, dict[str, Any]]:
    if indices.size == 0:
        indices = np.arange(labels.shape[0], dtype=np.int64)
    x = features[indices]
    y = labels[indices]
    logits = x @ state.weights + state.bias
    probs = _softmax(logits)
    loss = _cross_entropy(probs, y)
    grad_logits = probs
    grad_logits[np.arange(y.size), y] -= 1.0
    grad_logits /= max(1, y.size)
    grad_w = x.T @ grad_logits
    grad_b = grad_logits.sum(axis=0)
    weights = state.weights - learning_rate * grad_w
    bias = state.bias - learning_rate * grad_b
    preds = np.argmax(probs, axis=1)
    metrics = {
        "loss": float(loss),
        "accuracy": float(np.mean(preds == y)) if y.size else 0.0,
        "num_samples": int(y.size),
    }
    return TrainingState(weights, bias, state.epoch + 1, state.best_metric, state.class_names), metrics


def _initial_state(feature_dim: int, num_classes: int, seed: int, class_names: Sequence[str]) -> TrainingState:
    rng = np.random.default_rng(int(seed))
    scale = 1.0 / max(1.0, np.sqrt(float(feature_dim)))
    weights = rng.normal(0.0, scale, size=(feature_dim, max(1, num_classes))).astype(np.float64)
    bias = np.zeros(max(1, num_classes), dtype=np.float64)
    return TrainingState(weights=weights, bias=bias, epoch=0, best_metric=-1.0, class_names=list(class_names))


def _split_indices(size: int, validation_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    val_size = int(round(size * validation_split))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if train_indices.size == 0:
        train_indices = indices
    return train_indices, val_indices


def _label_from_record(sample: BatchSample, label_field: str | None) -> str:
    candidates = [label_field] if label_field else []
    candidates.extend(["label", "category", "class", "class_id", "shape_type"])
    for key in candidates:
        if key and key in sample.raw_record and sample.raw_record[key] is not None:
            return str(sample.raw_record[key])
    step_path = sample.raw_record.get("step_path")
    if isinstance(step_path, str) and step_path:
        parent = Path(step_path).parent.name
        if parent and parent != ".":
            return parent
    return sample.sample_id or f"sample_{sample.index}"


def _softmax(logits: np.ndarray) -> np.ndarray:
    stable = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(stable)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    chosen = probs[np.arange(labels.size), labels]
    return float(-np.mean(np.log(np.maximum(chosen, 1e-12))))


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
