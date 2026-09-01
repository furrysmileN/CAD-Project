"""Cut 3 点云附表：cadrille --mode pc 预测进同一套沙箱与评分。"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import load_config
from rq2_harness.hvc_analysis import analyze_hvc
from rq2_harness.hvc_runner import run_hvc


def main() -> int:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "harness_vs_cadrille.yaml")
    analysis = analyze_hvc(config)
    if not analysis["gates"].get("cadrille_pc_appendix"):
        print(json.dumps({"status": "blocked", "gates": analysis["gates"]}, ensure_ascii=False, indent=2))
        return 0
    pc_config = deepcopy(config)
    pc_config["paths"] = {
        **pc_config["paths"],
        "output_root": "experiments/rq2_multimodal_harness/outputs/harness_vs_cadrille/cut3_pc",
        "cadrille_predictions": config["paths"]["cadrille_pc_predictions"],
    }
    summary = run_hvc(pc_config, dry_run="--dry-run" in sys.argv, arms=["cadrille_rl"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
