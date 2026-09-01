"""Cut 2 live：只跑两个 Qwen 臂。不跑 CADrille / GPU。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import load_config, project_path
from rq2_harness.hvc_analysis import analyze_hvc
from rq2_harness.hvc_runner import run_hvc
from rq2_harness.qwen_keyfile import apply_qwen_keyfile


def main() -> int:
    apply_qwen_keyfile()
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "harness_vs_cadrille.yaml")
    dest = project_path(config["paths"]["output_root"])
    dest.mkdir(parents=True, exist_ok=True)
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    force = "--force" in sys.argv
    try:
        summary = run_hvc(
            config,
            dry_run=False,
            limit=limit,
            force=force,
            arms=["qwen_raw", "qwen_harness"],
        )
    except Exception as exc:
        payload = {"status": "live_failed", "error_type": type(exc).__name__, "error": str(exc)[:400]}
        (dest / "CUT2_STATUS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1
    cadrille = run_hvc(config, dry_run=False, limit=limit, force=False, arms=["cadrille_rl"])
    analysis = analyze_hvc(config)
    payload = {
        "status": "qwen_live_done",
        "skipped": ["cadrille_gpu", "deepcad_smoke"],
        "qwen": summary,
        "cadrille": cadrille,
        "gates": analysis.get("gates"),
        "means": analysis.get("means"),
        "success_rate": analysis.get("success_rate"),
        "delta_harness_raw": analysis.get("delta_harness_raw"),
        "cadrille_note": analysis.get("cadrille_note"),
    }
    (dest / "CUT2_STATUS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
