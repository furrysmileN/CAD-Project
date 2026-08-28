# -*- coding: utf-8 -*-
"""P2：PointEvidence 离线质量审计（零 API）。

在冻结 20 样本上生成 PointEvidence，仅评分阶段使用 GT STEP 检查 bbox/主轴/
截面/对称的准确率，输出门禁报告。门禁未通过时应先修点云模块，不进入 API 实验。
"""
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.evidence_audit import audit_evidence


def main(argv: list[str] | None = None) -> int:
    import argparse

    experiment_dir = Path(__file__).resolve().parents[1]
    prepared = experiment_dir / "outputs" / "encoding_screen_n20" / "prepared_samples"
    sample_ids = sorted(path.stem for path in prepared.glob("*.json"))
    parser = argparse.ArgumentParser(description="P2：PointEvidence 离线质量审计")
    parser.add_argument("--sample-ids", nargs="*", default=sample_ids)
    parser.add_argument("--manifest", default=str(experiment_dir / "outputs" / "pilot_v2" / "manifest.jsonl"))
    parser.add_argument("--pointcloud-root", default=str(experiment_dir.parents[1] / "processed" / "point_clouds" / "benchcad"))
    parser.add_argument("--evidence-dir", default=str(experiment_dir / "outputs" / "native_pointcloud_v1" / "evidence"))
    parser.add_argument("--output", default=str(experiment_dir / "outputs" / "native_pointcloud_v1" / "analysis" / "phase2"))
    parser.add_argument("--density", type=int, default=2048)
    parser.add_argument("--gt-n-points", type=int, default=8192)
    parser.add_argument("--gt-seed", type=int, default=42)
    args = parser.parse_args(argv)

    report = audit_evidence(
        args.sample_ids,
        Path(args.manifest),
        Path(args.pointcloud_root),
        Path(args.evidence_dir),
        density=args.density,
        gt_n_points=args.gt_n_points,
        gt_seed=args.gt_seed,
        output_dir=Path(args.output),
    )
    print(f"P2 审计完成：{args.output}")
    print(f"  门禁: {'通过' if report['gate']['passed'] else '未通过'} "
          f"(bbox {report['summary']['bbox_pass_rate']:.2f} / 主轴 {report['summary']['axis_pass_rate']:.2f} / "
          f"截面 {report['summary']['section_pass_rate']:.2f} / 对称 {report['summary']['symmetry_pass_rate']:.2f})")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
