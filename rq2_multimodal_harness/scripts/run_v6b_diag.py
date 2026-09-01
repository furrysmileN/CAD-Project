# -*- coding: utf-8 -*-
"""C2B / TB diagnostic live/dry-run. Does not write probe/live or V6 pilot/live."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.v6b_diag_runner import run_v6b_diag


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = run_v6b_diag(args.config, dry_run=args.dry_run, limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("counts", {}).get("fatal_api_error"):
        return 3
    if args.dry_run and summary.get("counts", {}).get("payload_audit_failed"):
        return 2
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
