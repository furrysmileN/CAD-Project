"""Harness vs CADrille Cut 2 / Cut 3 入口。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import load_config
from rq2_harness.hvc_analysis import analyze_hvc
from rq2_harness.hvc_runner import run_hvc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "harness_vs_cadrille.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=("cadrille_rl", "qwen_raw", "qwen_harness"))
    args = parser.parse_args()
    config = load_config(args.config)
    if not args.analyze_only:
        summary = run_hvc(
            config,
            dry_run=args.dry_run,
            limit=args.limit,
            force=args.force,
            arms=args.arms,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    analysis = analyze_hvc(config)
    print(json.dumps(analysis.get("gates") or analysis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
