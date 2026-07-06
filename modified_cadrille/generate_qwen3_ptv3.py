import argparse
import json
import os
from functools import partial
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from cadrille_qwen3_ptv3 import (
    Qwen3VLWithPointTransformerV3,
    initialize_point_encoder_weights,
    reset_point_transformer_v3_luts,
)
from qwen3vl_data import BenchCADIndexedDataset, build_pointcloud_adapter, collate_qwen3vl
from train_qwen import _load_vision_text_model, _torch_dtype_from_name

try:
    import yaml
except Exception:
    yaml = None


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            if yaml is None:
                raise ImportError("读取 yaml 配置需要安装 PyYAML")
            return yaml.safe_load(f) or {}
        return json.load(f)


def _resolve(path_value: Optional[str], base_dir: str) -> Optional[str]:
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def _compat_code(py_string: str) -> str:
    prefix = """import cadquery as cq

_shown_objects = []
def show_object(obj, *args, **kwargs):
    _shown_objects.append(obj)
    return obj

"""
    suffix = """

# Compatibility for modified_cadrille/evaluate.py
try:
    r
except NameError:
    if 'result' in globals():
        r = result
    elif _shown_objects:
        r = _shown_objects[-1]
"""
    return prefix + py_string.strip() + suffix


def _load_model(cfg: Dict[str, Any], checkpoint_path: Optional[str]):
    model_cfg = cfg["model"]
    model_path = model_cfg["model_path"]
    model_kwargs = dict(
        torch_dtype=_torch_dtype_from_name(model_cfg.get("dtype", "bfloat16")),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        device_map="auto",
    )
    attn_impl = model_cfg.get("attn_implementation")
    if attn_impl and attn_impl != "auto":
        model_kwargs["attn_implementation"] = attn_impl

    point_cfg = dict(cfg.get("point_encoder") or {})
    if point_cfg.get("type") == "point_transformer_v3":
        model = Qwen3VLWithPointTransformerV3.from_pretrained(
            model_path,
            point_encoder_config=point_cfg,
            **model_kwargs,
        )
        model.point_encoder.float()
        initialize_point_encoder_weights(model.point_encoder)
        reset_point_transformer_v3_luts()
    else:
        model = _load_vision_text_model(model_path, model_kwargs)

    if checkpoint_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    return model


def run(args: argparse.Namespace):
    cfg_path = os.path.abspath(args.config)
    cfg = _load_config(cfg_path)
    base_dir = os.path.dirname(cfg_path)
    cfg["data"]["index_path"] = _resolve(cfg["data"]["index_path"], base_dir)
    cfg["model"]["model_path"] = _resolve(cfg["model"]["model_path"], base_dir)
    cfg.setdefault("point_encoder", {})
    if cfg["point_encoder"].get("type") == "point_transformer_v3":
        cfg["point_encoder"].setdefault("num_points", int(cfg["data"].get("pointcloud_num_points", 256)))
        cfg["point_encoder"].setdefault("repo_path", "/root/autodl-tmp")

    processor = AutoProcessor.from_pretrained(
        cfg["model"]["model_path"],
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
        min_pixels=cfg["model"].get("min_pixels", None),
        max_pixels=cfg["model"].get("max_pixels", None),
        padding_side=cfg["model"].get("padding_side", "left"),
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = cfg["model"].get("padding_side", "left")

    adapter = build_pointcloud_adapter(
        name=cfg["data"].get("pointcloud_adapter", "tensor"),
        num_points=int(cfg["data"].get("pointcloud_num_points", 256)),
        precision=int(cfg["data"].get("pointcloud_precision", 4)),
    )
    dataset = BenchCADIndexedDataset(
        index_path=cfg["data"]["index_path"],
        split=args.split,
        prompt=cfg["data"].get("prompt", "Generate cadquery code"),
        pointcloud_adapter=adapter,
        max_images=int(cfg["data"].get("max_images", 4)),
    )
    if args.max_samples is not None:
        dataset.records = dataset.records[: args.max_samples]

    os.makedirs(args.output_dir, exist_ok=True)
    if os.listdir(args.output_dir) and not args.overwrite:
        raise RuntimeError(f"输出目录非空: {args.output_dir}。如需覆盖请加 --overwrite")
    if args.overwrite:
        for name in os.listdir(args.output_dir):
            path = os.path.join(args.output_dir, name)
            if os.path.isfile(path):
                os.remove(path)

    model = _load_model(cfg, args.checkpoint_path)
    collate_fn = partial(collate_qwen3vl, processor=processor, train_mode=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = next(model.parameters()).device
    for batch in tqdm(loader, total=len(loader)):
        sample_ids = batch.pop("sample_ids")
        model_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids)
        ]
        py_strings = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for sample_id, py_string in zip(sample_ids, py_strings):
            out_path = os.path.join(args.output_dir, f"{sample_id}+0.py")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(_compat_code(py_string))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CadQuery code with Qwen3-VL + PTv3")
    parser.add_argument("--config", default="configs-qwen3_ptv3_train.yaml")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
