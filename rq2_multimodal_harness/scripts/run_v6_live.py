# -*- coding: utf-8 -*-
"""V6 live runner. Default: 20 x 6 pilot. Does not touch V5 state."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.v6_runner import run_v6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "v6_pilot_live.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = run_v6(args.config, dry_run=False, limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("counts", {}).get("fatal_api_error"):
        return 3
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
