from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import load_config, project_path
from rq2_harness.pc_conditions import V5_CONFIRM_IDS, V5_CONTROL_IDS
from rq2_harness.v5_analysis import analyze_v5_confirm


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="分析 V5 Phase C 确认结果")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "v5_phase_c_confirm.yaml"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    payload = analyze_v5_confirm(
        project_path(config["paths"]["output_root"]),
        project_path(config["paths"]["manifest"]),
        condition_ids=V5_CONFIRM_IDS + V5_CONTROL_IDS,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
