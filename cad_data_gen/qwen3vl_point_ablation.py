from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cad_data_gen.qwen3vl_point_finetune import CORE_POINT_ENCODER_TYPE, FinetuneConfig, save_finetune_config


# 消融矩阵默认围绕核心 conditional_encoders/MICHE-style encoder 展开；
# image_text 是无点云基线，其余 point_* / image_point_* 用于验证核心点云输入是否被模型利用。
ABLATION_MATRIX = [
    ("baseline_image_text", {"fusion_mode": "image_text", "point_ablation": "none"}),
    ("point_text", {"fusion_mode": "point_text", "point_ablation": "none"}),
    ("point_text_zero", {"fusion_mode": "point_text", "point_ablation": "zero"}),
    ("point_text_shuffle", {"fusion_mode": "point_text", "point_ablation": "shuffle"}),
    ("point_text_replace", {"fusion_mode": "point_text", "point_ablation": "replace"}),
    ("image_point_text", {"fusion_mode": "image_point_text", "point_ablation": "none"}),
]


def build_ablation_plan(base_config: FinetuneConfig, output_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    plan: list[dict[str, Any]] = []
    for name, overrides in ABLATION_MATRIX:
        data = base_config.to_dict()
        data.update(overrides)
        data["output_dir"] = str(root / name)
        config = FinetuneConfig(**data)
        config_path = root / f"{name}.yaml"
        save_finetune_config(config, config_path)
        command = [
            "python",
            "-m",
            "cad_data_gen.train_qwen3vl_point_finetune",
            "--config",
            str(config_path),
        ]
        plan.append(
            {
                "name": name,
                "config_path": str(config_path),
                "output_dir": config.output_dir,
                "fusion_mode": config.fusion_mode,
                "point_ablation": config.point_ablation,
                "encoder_type": config.encoder_type,
                "point_encoder_role": config.point_encoder_role,
                "point_encoder_source": config.point_encoder_source,
                "command": command,
            }
        )
    with (root / "ablation_plan.json").open("w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    with (root / "commands.sh").open("w", encoding="utf-8") as f:
        f.write("set -e\n")
        for item in plan:
            f.write(" ".join(item["command"]) + "\n")
    return plan


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create Qwen3-VL point-cloud ablation configs.")
    parser.add_argument("--output-dir", default="cad_data_gen/runs/qwen3vl_point_ablation")
    parser.add_argument("--data-root", default="cad_data_gen/data/cad_shape_1000_english_blender512_20260527")
    parser.add_argument("--model-path", default="models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--stage", default="overfit8", choices=["single_batch", "overfit8", "overfit32", "overfit128", "full"])
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--overfit-count", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    base = FinetuneConfig(
        data_root=args.data_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        stage=args.stage,
        split_mode="overfit",
        limit=args.limit,
        overfit_count=args.overfit_count,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        device=args.device,
        dtype=args.dtype,
        encoder_type=CORE_POINT_ENCODER_TYPE,
        pointcloud_mode="miche",
        point_token_count=2048,
        point_feature_dim=6,
        include_normals=True,
        point_feature_channels=("xyz", "normals"),
    )
    plan = build_ablation_plan(base, args.output_dir)
    print(json.dumps({"output_dir": args.output_dir, "experiments": plan}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
