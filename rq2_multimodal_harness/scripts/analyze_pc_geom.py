from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import load_config, project_path
from rq2_harness.pc_analysis import analyze_pc_geom


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="分析 P_geom 20×9 筛选结果")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "pc_geom_screen.yaml"),
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output_dir = project_path(config["paths"]["output_root"])
    manifest = project_path(config["paths"]["manifest"])
    payload = analyze_pc_geom(
        output_dir,
        manifest,
        condition_ids=tuple(config.get("conditions") or []),
        kind="confirm" if int(config.get("n") or 0) >= 100 else "screen",
    )
    print(f"分析完成：{output_dir / 'analysis'}")
    print(f"有效任务行：{payload['n_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
