"""把 BenchCAD 40 件打成 CADrille 可消费的 split（吃我们的图/点云，不喂 GT mesh）。

官方 CadRecodeDataset 会从 split/*.stl 渲染，若放入 GT STL 会泄漏几何。
因此导出 view_0/2/4/6 与 2048.npy，并写出 drop-in dataset / infer 脚本。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .common import atomic_write_json, project_path, read_jsonl

SPLIT_NAME = "benchcad_hvc40"
VIEW_ORDER = ("view_0", "view_2", "view_4", "view_6")
IMG_SIZE = 128

DROPIN_DATASET = r'''
"""Drop-in: BenchCAD HVC40 for cadrille. Place next to dataset.py. Does NOT load GT mesh."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

SPLIT_NAME = "benchcad_hvc40"
VIEW_ORDER = ("view_0", "view_2", "view_4", "view_6")


class BenchCADHvcDataset(Dataset):
    def __init__(self, split_root, mode="img", img_size=128, n_points=256):
        self.split_root = Path(split_root)
        self.mode = mode
        self.img_size = img_size
        self.n_points = n_points
        records = []
        for line in (self.split_root / f"{SPLIT_NAME}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        if self.mode == "pc":
            item = self._pc(row)
        elif self.mode == "img":
            item = self._img(row)
        else:
            raise ValueError(f"unsupported mode {self.mode}")
        item["file_name"] = row["sample_id"]
        return item

    def _img(self, row):
        by_view = {item["view"]: item["path"] for item in row["images"]}
        tiles = []
        for view in VIEW_ORDER:
            image = Image.open(by_view[view]).convert("RGB")
            image = image.resize((self.img_size, self.img_size), Image.Resampling.BICUBIC)
            tiles.append(ImageOps.expand(image, border=3, fill="black"))
        mosaic = Image.fromarray(
            np.vstack(
                (
                    np.hstack((np.array(tiles[0]), np.array(tiles[1]))),
                    np.hstack((np.array(tiles[2]), np.array(tiles[3]))),
                )
            )
        )
        return {"video": [mosaic], "description": "Generate cadquery code"}

    def _pc(self, row):
        path = row.get("point_cloud_256") or row["point_cloud"]
        points = np.load(path).astype(np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"expected Nx3, got {points.shape}")
        if len(points) > self.n_points:
            points = points[: self.n_points]
        # 256 文件已是 bbox->[0,1]；与官方 test 的 (x-0.5)*2 对齐到 [-1,1]
        points = (points - 0.5) * 2.0
        return {"point_cloud": points, "description": "Generate cadquery code"}
'''

DROPIN_TEST = r'''
"""Linux GPU infer for BenchCAD HVC40. Copy into the cadrille repo root."""
import os
from argparse import ArgumentParser
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from tqdm import tqdm

from cadrille import Cadrille, collate
from benchcad_hvc_dataset import BenchCADHvcDataset


def run(split_root, mode, checkpoint_path, py_path):
    os.makedirs(py_path, exist_ok=True)
    model = Cadrille.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side="left",
    )
    dataset = BenchCADHvcDataset(split_root, mode=mode)
    loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=4,
        collate_fn=partial(collate, processor=processor, n_points=256, eval=True),
    )
    for batch in tqdm(loader):
        generated_ids = model.generate(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            point_clouds=batch["point_clouds"].to(model.device),
            is_pc=batch["is_pc"].to(model.device),
            is_img=batch["is_img"].to(model.device),
            pixel_values_videos=batch["pixel_values_videos"].to(model.device)
            if batch.get("pixel_values_videos", None) is not None
            else None,
            video_grid_thw=batch["video_grid_thw"].to(model.device)
            if batch.get("video_grid_thw", None) is not None
            else None,
            max_new_tokens=768,
        )
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(batch.input_ids, generated_ids)]
        texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for stem, text in zip(batch["file_name"], texts):
            with open(os.path.join(py_path, f"{stem}.py"), "w", encoding="utf-8") as handle:
                handle.write(text)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--mode", default="img", choices=("img", "pc"))
    parser.add_argument("--checkpoint-path", default="maksimko123/cadrille-rl")
    parser.add_argument("--py-path", required=True)
    args = parser.parse_args()
    run(args.split_root, args.mode, args.checkpoint_path, args.py_path)
'''


def _fps(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return points.astype(np.float32)
    chosen = [0]
    dist = np.full(len(points), np.inf, dtype=np.float64)
    for _ in range(1, count):
        last = points[chosen[-1]]
        dist = np.minimum(dist, np.sum((points - last) ** 2, axis=1))
        chosen.append(int(np.argmax(dist)))
    return points[np.asarray(chosen)].astype(np.float32)


def _canonicalize_pc(points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = float(np.max(maxs - mins)) or 1.0
    canonical = (points - center) / scale
    return (canonical + 0.5).astype(np.float32)


def export_cadrille_split(
    manifest_path: Path | None = None,
    dest: Path | None = None,
) -> dict[str, Any]:
    manifest = Path(manifest_path) if manifest_path else project_path(
        "experiments/rq2_multimodal_harness/outputs/harness_vs_cadrille/manifest_n40.jsonl"
    )
    root = Path(dest) if dest else project_path(
        "experiments/rq2_multimodal_harness/outputs/harness_vs_cadrille/cadrille_split"
    )
    img_dir = root / "images" / SPLIT_NAME
    pc_dir = root / "pointclouds" / SPLIT_NAME
    pc256_dir = root / "pointclouds256" / SPLIT_NAME
    mosaic_dir = root / "mosaic128" / SPLIT_NAME
    img_dir.mkdir(parents=True, exist_ok=True)
    pc_dir.mkdir(parents=True, exist_ok=True)
    pc256_dir.mkdir(parents=True, exist_ok=True)
    mosaic_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in read_jsonl(manifest):
        sample_id = row["sample_id"]
        views = []
        tiles = []
        by_view = {item["view"]: item for item in row["images"]}
        for view in VIEW_ORDER:
            item = by_view[view]
            suffix = f"{sample_id}__{view}.png"
            target = img_dir / suffix
            shutil.copyfile(item["path"], target)
            views.append({"view": view, "path": str(target)})
            image = Image.open(item["path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
            tiles.append(ImageOps.expand(image, border=3, fill="black"))
        mosaic = Image.fromarray(
            np.vstack(
                (
                    np.hstack((np.array(tiles[0]), np.array(tiles[1]))),
                    np.hstack((np.array(tiles[2]), np.array(tiles[3]))),
                )
            )
        )
        mosaic_path = mosaic_dir / f"{sample_id}.png"
        mosaic.save(mosaic_path)
        pc_target = pc_dir / f"{sample_id}.npy"
        shutil.copyfile(row["point_cloud"]["path"], pc_target)
        raw = np.load(row["point_cloud"]["path"])
        pc256 = _fps(_canonicalize_pc(raw), 256)
        pc256_path = pc256_dir / f"{sample_id}.npy"
        np.save(pc256_path, pc256)
        records.append(
            {
                "uid": sample_id,
                "sample_id": sample_id,
                "family": row.get("family"),
                "stratum": row.get("stratum"),
                "images": views,
                "mosaic128": str(mosaic_path),
                "point_cloud": str(pc_target),
                "point_cloud_256": str(pc256_path),
                "text": row.get("text", {}).get("L1", ""),
                "gt_step": row["step"]["path"],
            }
        )
    (root / f"{SPLIT_NAME}.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    (root / "benchcad_hvc_dataset.py").write_text(DROPIN_DATASET.lstrip("\n"), encoding="utf-8")
    (root / "test_benchcad_hvc.py").write_text(DROPIN_TEST.lstrip("\n"), encoding="utf-8")
    atomic_write_json(
        root / "README.json",
        {
            "split": SPLIT_NAME,
            "n": len(records),
            "weights": {"rl": "maksimko123/cadrille-rl", "sft": "maksimko123/cadrille"},
            "repo": "https://github.com/col14m/cadrille",
            "fairness": "Do not feed GT STL into official CadRecodeDataset; that renders GT mesh.",
            "infer_img": "python test_benchcad_hvc.py --split-root <this_dir> --mode img --checkpoint-path maksimko123/cadrille-rl --py-path predictions_img",
            "infer_pc": "python test_benchcad_hvc.py --split-root <this_dir> --mode pc --checkpoint-path maksimko123/cadrille-rl --py-path predictions_pc",
            "deepcad_smoke": "python test.py --split deepcad_test_mesh --mode img --checkpoint-path maksimko123/cadrille-rl  # 仅抽 5 条确认能出码，不进主表",
            "img_protocol": "view_0/2/4/6 resized to 128 and tiled 2x2, matching cadrille num_imgs=4",
            "pc_protocol": "2048 npy -> FPS 256, bbox canonicalized to [0,1] so their (x-0.5)*2 is not applied twice",
        },
    )
    return {"n": len(records), "root": str(root), "split": SPLIT_NAME}
