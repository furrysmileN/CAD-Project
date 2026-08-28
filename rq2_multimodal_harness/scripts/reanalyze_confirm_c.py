# -*- coding: utf-8 -*-
"""P0：confirm C 臂零成本重分析（不调用 API）。

读取 outputs/confirm_n100/arms/C/state，重算旧 P_proj 在七条件矩阵中的互补性、
paired-valid-only 几何与贡献分解，输出独立小报告。该报告是 P_geom 实验的基线对照。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.p0_reanalysis import analyze_c_arm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0：confirm C 臂重分析（零 API）")
    parser.add_argument("--arm", default="C", choices=["A0", "C"], help="默认 C（执行失败基本被控制的臂）")
    parser.add_argument("--state-dir", help="覆盖 state 目录（默认 outputs/confirm_n100/arms/<arm>/state）")
    parser.add_argument("--manifest", help="覆盖 pilot_v2 manifest 路径（用于 family/difficulty）")
    parser.add_argument("--output", help="输出目录（默认 outputs/native_pointcloud_v1/analysis/phase0）")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = parser.parse_args(argv)

    output = Path(args.output) if args.output else (
        Path(__file__).resolve().parents[1] / "outputs" / "native_pointcloud_v1" / "analysis" / "phase0"
    )
    report = analyze_c_arm(
        arm=args.arm,
        seed=args.seed,
        bootstrap_repeats=args.bootstrap_repeats,
        state_dir=Path(args.state_dir) if args.state_dir else None,
        manifest_path=Path(args.manifest) if args.manifest else None,
        output_dir=output,
    )
    print(f"P0 重分析完成：{output}")
    print(f"  arm={args.arm} 样本数={report['n_samples']} 条件={len(report['condition_summary'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
