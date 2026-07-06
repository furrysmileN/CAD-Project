import argparse
import glob
import json
import os
import shutil
import sys
from typing import Dict, List

from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen3vl_data import build_pointcloud_adapter, _load_npz_points  # noqa: E402


def _normalize_local_model_files(model_path: str):
    if not os.path.isdir(model_path):
        return
    aliases = {
        "config.json": ["config*.json"],
        "tokenizer.json": ["tokenizer*.json"],
        "tokenizer_config.json": ["tokenizer_config*.json"],
        "generation_config.json": ["generation_config*.json"],
    }
    for canonical, patterns in aliases.items():
        dst = os.path.join(model_path, canonical)
        if os.path.exists(dst):
            continue
        src = None
        for pattern in patterns:
            matches = sorted(glob.glob(os.path.join(model_path, pattern)))
            matches = [m for m in matches if os.path.isfile(m)]
            if matches:
                src = matches[0]
                break
        if src is None or os.path.basename(src) == canonical:
            continue
        try:
            os.symlink(os.path.basename(src), dst)
        except OSError:
            shutil.copy2(src, dst)


def _estimate_seq_len(tokenizer, answer: str, fixed_overhead: int) -> int:
    """估算样本进入 collate 后的真实 seq_len。

    经实测（qwen3vl_data 序列化点云 256 点 + 单张图 max_pixels=262144 + chat 模板），
    seq_len ≈ answer_token + FIXED，其中 FIXED 部分（点云文本+图像占位+模板）方差极小
    （mean≈5731, std≈29）。因此只需 tokenize 答案代码即可精确估算，无需加载点云 npz，
    从而避免在慢速分布式存储上逐样本读取 8192 点的开销。点云精度完全不变。
    """
    ans_tok = len(tokenizer(answer, add_special_tokens=False)["input_ids"])
    return ans_tok + fixed_overhead


def filter_index(
    index_path: str,
    output_path: str,
    model_path: str,
    max_seq_len: int,
    fixed_overhead: int,
    calibrate: bool,
    calibrate_n: int,
) -> Dict:
    index_path = os.path.abspath(index_path)
    output_path = os.path.abspath(output_path)

    print(f"[filter] 读取索引: {index_path}", flush=True)
    records: List[Dict] = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[filter] 总样本: {len(records)}", flush=True)

    print(f"[filter] 加载 tokenizer: {model_path}", flush=True)
    _normalize_local_model_files(model_path)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        min_pixels=200704,
        max_pixels=262144,
        padding_side="left",
    )
    tokenizer = processor.tokenizer

    # 可选：用少量样本真实标定 fixed_overhead（跑一次真实 collate），
    # 默认关闭以避免加载点云 npz；标定仅在明确开启时执行。
    if calibrate:
        import random

        from qwen3vl_data import BenchCADIndexedDataset, collate_qwen3vl

        adapter = build_pointcloud_adapter("serialize", num_points=256, precision=4)
        ds = BenchCADIndexedDataset(
            index_path, "train", "Generate cadquery code", adapter, max_images=1
        )
        random.seed(0)
        sample_idx = random.sample(range(len(ds)), min(calibrate_n, len(ds)))
        diffs = []
        for j in sample_idx:
            s = ds[j]
            real = collate_qwen3vl([s], processor=processor, train_mode=True)[
                "input_ids"
            ].shape[1]
            ans_tok = len(tokenizer(s["answer"], add_special_tokens=False)["input_ids"])
            diffs.append(real - ans_tok)
        fixed_overhead = int(max(diffs)) + 8  # 取上界并留少量余量
        print(
            f"[filter] 标定完成 fixed_overhead={fixed_overhead} "
            f"(min={min(diffs)} max={max(diffs)})",
            flush=True,
        )

    print(
        f"[filter] 使用 fixed_overhead={fixed_overhead}, max_seq_len={max_seq_len}",
        flush=True,
    )

    kept: List[Dict] = []
    dropped: List[Dict] = []
    for i, rec in enumerate(records):
        if (i + 1) % 3000 == 0:
            print(f"[filter] 进度 {i + 1}/{len(records)} | 已保留 {len(kept)}", flush=True)
        # 只对训练集做复杂度筛查；val/test 原样保留，保证评测集不被改动。
        if rec.get("split") != "train":
            kept.append(rec)
            continue
        try:
            answer = open(rec["cadquery_path"], "r", encoding="utf-8").read()
        except Exception as exc:
            print(f"[filter] 读取答案失败,剔除 {rec.get('sample_id')}: {exc}", flush=True)
            dropped.append({"sample_id": rec.get("sample_id"), "reason": "read_error"})
            continue
        seq_len = _estimate_seq_len(tokenizer, answer, fixed_overhead)
        if seq_len <= max_seq_len:
            rec = dict(rec)
            rec["est_seq_len"] = int(seq_len)
            kept.append(rec)
        else:
            dropped.append(
                {"sample_id": rec.get("sample_id"), "est_seq_len": int(seq_len)}
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_train_kept = sum(1 for r in kept if r.get("split") == "train")
    n_train_total = sum(1 for r in records if r.get("split") == "train")
    summary = {
        "index_path": index_path,
        "output_path": output_path,
        "max_seq_len": max_seq_len,
        "fixed_overhead": fixed_overhead,
        "total_records": len(records),
        "kept_records": len(kept),
        "train_total": n_train_total,
        "train_kept": n_train_kept,
        "train_dropped": n_train_total - n_train_kept,
        "dropped_examples": dropped[:20],
    }
    summary_path = os.path.splitext(output_path)[0] + ".filter_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[filter] 完成 | 训练集 {n_train_total} -> {n_train_kept} "
        f"(剔除 {n_train_total - n_train_kept}) | 输出: {output_path}",
        flush=True,
    )
    print(f"[filter] 摘要: {summary_path}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按序列长度筛查过复杂样本(点云精度不变)")
    p.add_argument("--index-path", default="./data/benchcad_index.jsonl")
    p.add_argument("--output-path", default="./data/benchcad_index_filtered.jsonl")
    p.add_argument(
        "--model-path",
        default="/root/autodl-tmp/models/Qwen3-VL-2B-Instruct",
    )
    p.add_argument("--max-seq-len", type=int, default=8192, help="保留样本的 seq_len 上限")
    p.add_argument(
        "--fixed-overhead",
        type=int,
        default=5760,
        help="点云文本+图像+模板的固定 token 开销(实测≈5731)",
    )
    p.add_argument("--calibrate", action="store_true", help="用真实 collate 标定 overhead")
    p.add_argument("--calibrate-n", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    filter_index(
        index_path=args.index_path,
        output_path=args.output_path,
        model_path=args.model_path,
        max_seq_len=args.max_seq_len,
        fixed_overhead=args.fixed_overhead,
        calibrate=args.calibrate,
        calibrate_n=args.calibrate_n,
    )


if __name__ == "__main__":
    main()
