import argparse
import json
import os
import pickle
from typing import Dict, Iterable, List, Optional


def _rel(path: str, base_dir: str) -> str:
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return os.path.abspath(path)


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _candidate_pointcloud_path(pointcloud_dir: Optional[str], sample_id: str) -> Optional[str]:
    if not pointcloud_dir:
        return None
    for ext in (".npz", ".npy", ".xyz", ".txt", ".csv"):
        path = os.path.join(pointcloud_dir, sample_id + ext)
        if os.path.exists(path):
            return path
    return None


def _candidate_image_paths(image_dir: Optional[str], sample_id: str, max_images: int) -> List[str]:
    if not image_dir:
        return []

    candidates = []
    direct_dir = os.path.join(image_dir, sample_id)
    if os.path.isdir(direct_dir):
        for name in sorted(os.listdir(direct_dir)):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                candidates.append(os.path.join(direct_dir, name))
    else:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            path = os.path.join(image_dir, sample_id + ext)
            if os.path.exists(path):
                candidates.append(path)

    return candidates[:max_images]


def _record_from_annotation(
    root: str,
    split: str,
    ann: Dict,
    output_dir: str,
    pointcloud_dir: Optional[str],
    image_dir: Optional[str],
    max_images: int,
) -> Optional[Dict]:
    mesh_rel = ann.get("mesh_path")
    code_rel = ann.get("py_path") or ann.get("cadquery_path") or ann.get("code_path")
    if not mesh_rel or not code_rel:
        return None

    mesh_path = os.path.abspath(os.path.join(root, mesh_rel))
    code_path = os.path.abspath(os.path.join(root, code_rel))
    if not os.path.exists(mesh_path) or not os.path.exists(code_path):
        return None

    sample_id = os.path.splitext(mesh_rel.replace(os.sep, "__"))[0]
    pc_path = _candidate_pointcloud_path(pointcloud_dir, sample_id)
    image_paths = _candidate_image_paths(image_dir, sample_id, max_images)
    record = {
        "sample_id": sample_id,
        "split": split,
        "mesh_path": _rel(mesh_path, output_dir),
        "cadquery_path": _rel(code_path, output_dir),
    }
    if pc_path:
        record["pointcloud_path"] = _rel(os.path.abspath(pc_path), output_dir)
    if image_paths:
        record["image_paths"] = [_rel(os.path.abspath(path), output_dir) for path in image_paths]
    return record


def _records_from_pickle(
    root: str,
    split: str,
    output_dir: str,
    pointcloud_dir: Optional[str],
    image_dir: Optional[str],
    max_images: int,
) -> List[Dict]:
    pkl_path = os.path.join(root, f"{split}.pkl")
    if not os.path.exists(pkl_path):
        return []

    annotations = _load_pickle(pkl_path)
    records = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        rec = _record_from_annotation(
            root=root,
            split=split,
            ann=ann,
            output_dir=output_dir,
            pointcloud_dir=pointcloud_dir,
            image_dir=image_dir,
            max_images=max_images,
        )
        if rec is not None:
            records.append(rec)
    return records


def _records_from_directory(
    root: str,
    split: str,
    output_dir: str,
    pointcloud_dir: Optional[str],
    image_dir: Optional[str],
    max_images: int,
) -> List[Dict]:
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        return []

    code_by_stem = {}
    mesh_by_stem = {}
    for dirpath, _, filenames in os.walk(split_dir):
        for name in filenames:
            path = os.path.join(dirpath, name)
            stem, ext = os.path.splitext(os.path.relpath(path, split_dir))
            ext = ext.lower()
            if ext == ".py":
                code_by_stem[stem] = path
            elif ext in {".stl", ".obj", ".ply"}:
                mesh_by_stem[stem] = path

    records = []
    for stem in sorted(set(code_by_stem) & set(mesh_by_stem)):
        sample_id = f"{split}__{stem.replace(os.sep, '__')}"
        pc_path = _candidate_pointcloud_path(pointcloud_dir, sample_id)
        image_paths = _candidate_image_paths(image_dir, sample_id, max_images)
        rec = {
            "sample_id": sample_id,
            "split": split,
            "mesh_path": _rel(os.path.abspath(mesh_by_stem[stem]), output_dir),
            "cadquery_path": _rel(os.path.abspath(code_by_stem[stem]), output_dir),
        }
        if pc_path:
            rec["pointcloud_path"] = _rel(os.path.abspath(pc_path), output_dir)
        if image_paths:
            rec["image_paths"] = [_rel(os.path.abspath(path), output_dir) for path in image_paths]
        records.append(rec)
    return records


def build_index(
    root: str,
    output_path: str,
    splits: Iterable[str],
    pointcloud_dir: Optional[str],
    image_dir: Optional[str],
    max_images: int,
) -> List[Dict]:
    root = os.path.abspath(root)
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    pointcloud_dir = os.path.abspath(pointcloud_dir) if pointcloud_dir else None
    image_dir = os.path.abspath(image_dir) if image_dir else None

    records = []
    for split in splits:
        split_records = _records_from_pickle(
            root=root,
            split=split,
            output_dir=output_dir,
            pointcloud_dir=pointcloud_dir,
            image_dir=image_dir,
            max_images=max_images,
        )
        if not split_records:
            split_records = _records_from_directory(
                root=root,
                split=split,
                output_dir=output_dir,
                pointcloud_dir=pointcloud_dir,
                image_dir=image_dir,
                max_images=max_images,
            )
        records.extend(split_records)

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary_path = os.path.splitext(output_path)[0] + ".summary.json"
    summary = {
        "root": root,
        "output_path": output_path,
        "splits": list(splits),
        "num_records": len(records),
        "num_by_split": {split: sum(1 for rec in records if rec["split"] == split) for split in splits},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[build_index] 写入 {len(records)} 条记录: {output_path}", flush=True)
    print(f"[build_index] 摘要: {summary_path}", flush=True)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 Qwen3-VL CAD JSONL 索引")
    parser.add_argument("--root", default="./data/cad-recode-v1.5", help="CAD-Recode 或兼容数据根目录")
    parser.add_argument("--output-path", default="./data/benchcad_index.jsonl", help="输出 JSONL 路径")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="需要写入的 split")
    parser.add_argument("--pointcloud-dir", default=None, help="可选：预生成点云目录")
    parser.add_argument("--image-dir", default=None, help="可选：预生成图片目录")
    parser.add_argument("--max-images", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    build_index(
        root=args.root,
        output_path=args.output_path,
        splits=args.splits,
        pointcloud_dir=args.pointcloud_dir,
        image_dir=args.image_dir,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
