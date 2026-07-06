import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None


def _resolve_record_path(path_value: Optional[str], base_dir: str) -> Optional[str]:
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _coerce_points_array(obj: Any) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        arr = obj
    elif isinstance(obj, dict):
        for key in ("points", "point_cloud", "vertices", "xyz", "pc"):
            if key in obj:
                arr = np.asarray(obj[key])
                break
        else:
            first_key = next(iter(obj))
            arr = np.asarray(obj[first_key])
    else:
        arr = np.asarray(obj)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"点云数组必须是 [N, >=3]，实际形状: {arr.shape}")
    return arr[:, :3]


def _load_npz_points(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path)
        return _coerce_points_array({key: data[key] for key in data.files})
    if ext == ".npy":
        return _coerce_points_array(np.load(path))
    if ext in {".txt", ".xyz", ".csv"}:
        return _coerce_points_array(np.loadtxt(path, delimiter="," if ext == ".csv" else None))
    raise ValueError(f"不支持的点云文件格式: {path}")


def _sample_mesh_points(mesh_path: str, num_points: int) -> np.ndarray:
    try:
        import trimesh
    except Exception as exc:
        raise ImportError("从 mesh 采样点云需要安装 trimesh") from exc

    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"mesh 为空: {mesh_path}")
    sampled, _ = trimesh.sample.sample_surface(mesh, max(num_points, 1))
    return np.asarray(sampled, dtype=np.float32)


def _sample_or_pad_points(points: np.ndarray, num_points: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if len(points) >= num_points:
        ids = np.linspace(0, len(points) - 1, num_points).round().astype(np.int64)
        return points[ids].astype(np.float32)

    pad_count = num_points - len(points)
    pad = np.repeat(points[-1:, :], pad_count, axis=0)
    return np.concatenate([points, pad], axis=0).astype(np.float32)


@dataclass
class SerializePointCloudAdapter:
    num_points: int = 256
    precision: int = 4
    output_mode: str = "text"

    def __call__(self, points: Optional[np.ndarray]) -> str:
        if points is None:
            return "Point cloud: unavailable."

        points = _sample_or_pad_points(_coerce_points_array(points), self.num_points)
        fmt = f"{{:.{self.precision}f}}"
        rows = [
            ",".join(fmt.format(float(value)) for value in point[:3])
            for point in points
        ]
        return "<point_cloud>\n" + "\n".join(rows) + "\n</point_cloud>"


@dataclass
class TensorPointCloudAdapter:
    num_points: int = 256
    precision: int = 4
    output_mode: str = "tensor"
    normalize: bool = True

    def __call__(self, points: Optional[np.ndarray]) -> np.ndarray:
        if points is None:
            return np.zeros((self.num_points, 3), dtype=np.float32)

        points = _sample_or_pad_points(_coerce_points_array(points), self.num_points)
        if not self.normalize:
            return points.astype(np.float32)

        center = (points.max(axis=0) + points.min(axis=0)) / 2.0
        points = points - center
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 1e-6:
            points = points / scale
        return points.astype(np.float32)


@dataclass
class NoPointCloudAdapter:
    num_points: int = 0
    precision: int = 0
    output_mode: str = "none"

    def __call__(self, points: Optional[np.ndarray]) -> None:
        return None


def build_pointcloud_adapter(name: str = "serialize", num_points: int = 256, precision: int = 4):
    key = (name or "serialize").lower()
    if key in {"none", "no", "image_text", "image-only", "image_text_only"}:
        return NoPointCloudAdapter()
    if key in {"serialize", "text"}:
        return SerializePointCloudAdapter(num_points=num_points, precision=precision)
    if key in {"tensor", "ptv3", "point_transformer_v3"}:
        return TensorPointCloudAdapter(num_points=num_points, precision=precision)
    raise ValueError(f"不支持的 pointcloud_adapter={name}，可选: serialize/tensor/none")


class BenchCADIndexedDataset(Dataset):
    """JSONL-backed CAD dataset for Qwen3-VL LoRA training.

    Supported record fields:
    - split, sample_id
    - cadquery_path/code_path/answer_path or answer
    - pointcloud_path, mesh_path
    - image_paths/images/image_path
    """

    def __init__(
        self,
        index_path: str,
        split: str,
        prompt: str,
        pointcloud_adapter,
        max_images: int = 4,
    ):
        super().__init__()
        self.index_path = os.path.abspath(index_path)
        self.base_dir = os.path.dirname(self.index_path)
        self.split = split
        self.prompt = prompt
        self.pointcloud_adapter = pointcloud_adapter
        self.max_images = max_images

        records = []
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec.setdefault("_line_no", line_no)
                if rec.get("split") == split:
                    records.append(rec)
        self.records = records

    def __len__(self):
        return len(self.records)

    def _resolve(self, path_value: Optional[str]) -> Optional[str]:
        return _resolve_record_path(path_value, self.base_dir)

    def _answer_from_record(self, rec: Dict[str, Any]) -> str:
        if "answer" in rec and rec["answer"] is not None:
            return str(rec["answer"])
        for key in ("cadquery_path", "code_path", "answer_path"):
            path = self._resolve(rec.get(key))
            if path:
                return _read_text(path)
        raise KeyError(f"样本缺少 answer/cadquery_path: {rec.get('sample_id', rec.get('_line_no'))}")

    def _points_from_record(self, rec: Dict[str, Any]) -> Optional[np.ndarray]:
        pc_path = self._resolve(rec.get("pointcloud_path"))
        if pc_path and os.path.exists(pc_path):
            return _load_npz_points(pc_path)

        mesh_path = self._resolve(rec.get("mesh_path"))
        if mesh_path and os.path.exists(mesh_path):
            num_points = getattr(self.pointcloud_adapter, "num_points", 256)
            return _sample_mesh_points(mesh_path, int(num_points))

        if "points" in rec:
            return _coerce_points_array(rec["points"])
        return None

    def _image_paths_from_record(self, rec: Dict[str, Any]) -> List[str]:
        raw: List[str] = []
        if isinstance(rec.get("image_paths"), list):
            raw.extend(rec["image_paths"])
        if isinstance(rec.get("images"), list):
            raw.extend(rec["images"])
        if rec.get("image_path"):
            raw.append(rec["image_path"])

        paths = []
        for item in raw[: self.max_images]:
            path = self._resolve(str(item))
            if path and os.path.exists(path):
                paths.append(path)
        return paths

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.records[index]
        answer = self._answer_from_record(rec)
        prompt = rec.get("prompt", self.prompt)
        pointcloud_mode = getattr(self.pointcloud_adapter, "output_mode", "text")
        is_tensor_pc = pointcloud_mode == "tensor"

        if pointcloud_mode == "none":
            pointcloud_value = None
            user_text = prompt
        else:
            points = self._points_from_record(rec)
            pointcloud_value = self.pointcloud_adapter(points)
            user_text = prompt if is_tensor_pc else f"{prompt}\n\n{pointcloud_value}"
        image_paths = self._image_paths_from_record(rec)
        item = {
            "sample_id": rec.get("sample_id", str(index)),
            "prompt": prompt,
            "user_text": user_text,
            "answer": answer,
            "image_paths": image_paths,
            "raw_record": rec,
        }
        if is_tensor_pc:
            item["point_cloud"] = pointcloud_value
        elif pointcloud_mode == "text":
            item["pointcloud_text"] = pointcloud_value
        return item


def _message_for_sample(sample: Dict[str, Any], include_answer: bool, add_generation_prompt: bool = False):
    content = []
    for image_path in sample.get("image_paths", []):
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": sample["user_text"]})

    messages = [{"role": "user", "content": content}]
    if include_answer:
        messages.append({"role": "assistant", "content": sample["answer"]})
    return messages, add_generation_prompt


def _fallback_process_vision_info(messages: Sequence[List[Dict[str, Any]]]):
    image_inputs = []
    video_inputs = []
    for conversation in messages:
        for message in conversation:
            for content in message.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "image" and content.get("image"):
                    image_inputs.append(Image.open(content["image"]).convert("RGB"))
                elif content.get("type") == "video" and content.get("video"):
                    video_inputs.append(content["video"])
    return image_inputs or None, video_inputs or None


def _processor_call(processor, messages: Sequence[List[Dict[str, Any]]]):
    texts = [
        processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        for conversation in messages
    ]

    if process_vision_info is not None:
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        image_inputs, video_inputs = _fallback_process_vision_info(messages)

    kwargs = {
        "text": texts,
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs is not None:
        kwargs["images"] = image_inputs
    if video_inputs is not None:
        kwargs["videos"] = video_inputs
    return processor(**kwargs)


def _get_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def _point_placeholder_text(processor, num_tokens: int) -> str:
    if num_tokens <= 0:
        return ""
    tokenizer = _get_tokenizer(processor)
    unit = " x"
    tokenized = tokenizer(unit * num_tokens, add_special_tokens=False)["input_ids"]
    if len(tokenized) != num_tokens:
        raise ValueError(
            f"点云占位符必须严格对应 {num_tokens} 个 token，实际得到 {len(tokenized)} 个。"
        )
    return unit * num_tokens


def _sample_with_point_placeholders(sample: Dict[str, Any], processor) -> Dict[str, Any]:
    if "point_cloud" not in sample:
        return sample
    point_cloud = sample["point_cloud"]
    num_points = int(point_cloud.shape[0] if hasattr(point_cloud, "shape") else len(point_cloud))
    cloned = dict(sample)
    cloned["user_text"] = _point_placeholder_text(processor, num_points) + sample["user_text"]
    return cloned


def _prompt_lengths(processor, samples: Sequence[Dict[str, Any]]) -> List[int]:
    lengths = []
    for sample in samples:
        prompt_messages, _ = _message_for_sample(sample, include_answer=False, add_generation_prompt=True)
        text = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        messages = [prompt_messages]
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs, video_inputs = _fallback_process_vision_info(messages)

        kwargs = {"text": [text], "padding": False, "return_tensors": "pt"}
        if image_inputs is not None:
            kwargs["images"] = image_inputs
        if video_inputs is not None:
            kwargs["videos"] = video_inputs
        encoded = processor(**kwargs)
        lengths.append(int(encoded["input_ids"].shape[1]))
    return lengths


def collate_qwen3vl(
    batch: Sequence[Dict[str, Any]],
    processor,
    train_mode: bool = True,
) -> Dict[str, torch.Tensor]:
    if len(batch) == 0:
        raise ValueError("batch 不能为空")

    prepared_batch = [_sample_with_point_placeholders(sample, processor) for sample in batch]
    full_messages = [_message_for_sample(sample, include_answer=train_mode)[0] for sample in prepared_batch]
    encoded = _processor_call(processor, full_messages)
    result = dict(encoded)

    has_point_clouds = any("point_cloud" in sample for sample in batch)
    if has_point_clouds:
        tokenizer = _get_tokenizer(processor)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("point prefix 需要 tokenizer.pad_token_id")
        point_clouds = []
        point_token_mask = result["input_ids"].new_zeros(result["input_ids"].shape, dtype=torch.bool)
        attention_mask = result.get("attention_mask")
        seq_len = int(result["input_ids"].shape[1])
        padding_side = getattr(tokenizer, "padding_side", "right")

        for row, sample in enumerate(batch):
            if "point_cloud" not in sample:
                point_cloud = np.zeros((0, 3), dtype=np.float32)
            else:
                point_cloud = np.asarray(sample["point_cloud"], dtype=np.float32)
            point_clouds.append(torch.tensor(point_cloud, dtype=torch.float32))
            num_points = int(point_cloud.shape[0])

            if attention_mask is not None:
                real_len = int(attention_mask[row].sum().item())
            else:
                real_len = seq_len
            start = seq_len - real_len if padding_side == "left" else 0
            point_token_mask[row, start : start + num_points] = True

        result["point_clouds"] = torch.stack(point_clouds)
        result["point_token_mask"] = point_token_mask

    if not train_mode:
        result["sample_ids"] = [sample.get("sample_id", str(i)) for i, sample in enumerate(batch)]
        return result

    tokenizer = _get_tokenizer(processor)
    labels = result["input_ids"].clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    prompt_lens = _prompt_lengths(processor, prepared_batch)
    attention_mask = result.get("attention_mask")
    seq_len = int(labels.shape[1])
    padding_side = getattr(tokenizer, "padding_side", "right")

    for row, prompt_len in enumerate(prompt_lens):
        if attention_mask is not None:
            real_len = int(attention_mask[row].sum().item())
        else:
            real_len = seq_len

        if padding_side == "left":
            start = seq_len - real_len
            labels[row, start : start + prompt_len] = -100
        else:
            labels[row, :prompt_len] = -100

    result["labels"] = labels
    return result
