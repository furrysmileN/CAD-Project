from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from cad_data_gen.encoder.core import PointCloudEncoder
from cad_data_gen.qwen3vl_sync import (
    SyncConfig,
    build_early_concat_fusion,
    check_qwen3vl_environment,
    load_manifest_samples,
    tokenize_sync_sample,
)


def run_baseline(sample: Any, config: SyncConfig) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "baseline_mode": config.baseline_mode,
        "image_count": len(sample.image_paths),
        "has_point_cloud": bool(sample.point_path),
        "occlusion_image_count": len(sample.occlusion_image_paths),
        "mask_count": len(sample.mask_paths),
        "rendered_view_count": len(sample.rendered_view_paths),
        "source_tags": {
            "real_images": sample.image_paths,
            "occlusion_images": sample.occlusion_image_paths,
            "occlusion_masks": sample.mask_paths,
            "rendered_views": sample.rendered_view_paths,
        },
    }
    if config.baseline_mode == "image_only":
        result["messages"] = [{"role": "user", "content": [{"type": "image", "image": p} for p in sample.image_paths] + [{"type": "text", "text": sample.prompt}]}]
    elif config.baseline_mode == "point_global":
        if not sample.point_path:
            raise ValueError(f"sample {sample.sample_id} has no point cloud")
        encoder = PointCloudEncoder({"feature_dim": config.point_feature_dim, "backbone": config.point_encoder})
        encoded = encoder.encode_file(sample.point_path, sample_id=sample.sample_id)
        result["point_global_shape"] = list(encoded.feature.shape)
        result["point_global_norm"] = float(np.linalg.norm(encoded.feature))
        result["point_metadata"] = encoded.metadata.to_dict()
    elif config.baseline_mode == "point_tokens":
        tokenized = tokenize_sync_sample(sample, sync_config=config)
        result["point_token_shape"] = list(tokenized.tokens.shape)
        result["point_token_metadata"] = tokenized.metadata.to_dict()
    elif config.baseline_mode == "image_point_tokens":
        fused = build_early_concat_fusion(sample, config=config)
        result["fusion"] = {
            "point_embedding_shape": list(fused.point_embeddings.shape),
            "attention_mask_shape": list(fused.attention_mask.shape),
            "position_ids_shape": list(fused.position_ids.shape),
            "strategy": fused.fusion_plan.strategy,
            "spans": [span.to_dict() for span in fused.fusion_plan.spans],
            "alignment": fused.fusion_plan.metadata.get("alignment", {}),
        }
    elif config.baseline_mode == "rendered_views":
        result["status"] = "placeholder"
        result["note"] = "点云转多视图/深度图通道需要上游渲染器生成 rendered_view_paths 后复用图像通道。"
    else:
        raise ValueError(f"unsupported baseline_mode: {config.baseline_mode}")
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL image/point-cloud sync baselines.")
    parser.add_argument("--manifest", required=True, help="输入 manifest.jsonl")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--model-path", default="models/Qwen3-VL-2B-Instruct", help="本地 Qwen3-VL 模型目录")
    parser.add_argument(
        "--baseline-mode",
        default="image_point_tokens",
        choices=["image_only", "point_global", "point_tokens", "image_point_tokens", "rendered_views"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--missing-modality-policy", default="error", choices=["error", "allow_single", "skip"])
    parser.add_argument("--point-token-count", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=2048)
    args = parser.parse_args()

    config = SyncConfig(
        model_path=args.model_path,
        baseline_mode=args.baseline_mode,
        missing_modality_policy=args.missing_modality_policy,
        point_token_count=args.point_token_count,
        hidden_size=args.hidden_size,
    )
    check = check_qwen3vl_environment(config).to_dict()
    samples = load_manifest_samples(args.manifest, config=config, limit=args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "environment", "check": check}, ensure_ascii=False) + "\n")
        for sample in samples:
            try:
                record = run_baseline(sample, config)
                record["status"] = record.get("status", "ok")
            except Exception as exc:  # noqa: BLE001 - CLI 需要逐样本隔离失败
                record = {
                    "sample_id": sample.sample_id,
                    "baseline_mode": config.baseline_mode,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
