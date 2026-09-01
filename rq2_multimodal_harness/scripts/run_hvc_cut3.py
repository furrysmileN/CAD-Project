"""Cut 3：仅当 Cut 2 过门才准备扩 100 / 第二模型 / pc 附表。"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import load_config
from rq2_harness.hvc_analysis import analyze_hvc, prepare_cut3_expand100
from rq2_harness.hvc_runner import run_hvc


def main() -> int:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "harness_vs_cadrille.yaml")
    prepared = prepare_cut3_expand100(config)
    if prepared.get("status") != "ready":
        print(json.dumps(prepared, ensure_ascii=False, indent=2))
        return 0
    second = deepcopy(config)
    second["api"] = {**second["api"], "default_model": (config.get("cut3") or {}).get("second_model") or "qwen3-vl-plus"}
    second["paths"] = {
        **second["paths"],
        "output_root": "experiments/rq2_multimodal_harness/outputs/harness_vs_cadrille/cut3_second_qwen",
        "cadrille_predictions": config["paths"]["cadrille_pc_predictions"],
    }
    # 第二模型只跑赢的对照：raw vs harness；pc 附表只评 cadrille
    live = "--live" in sys.argv
    if live:
        summary = run_hvc(second, dry_run=False, arms=["qwen_raw", "qwen_harness"])
        print(json.dumps({"expand": prepared, "second_qwen": summary}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"expand": prepared, "second_qwen": "prepared_not_live"}, ensure_ascii=False, indent=2))
    analysis = analyze_hvc(config)
    print(json.dumps(analysis.get("gates"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
