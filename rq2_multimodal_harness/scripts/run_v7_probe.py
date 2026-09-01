# -*- coding: utf-8 -*-
"""V7 probe live/dry-run. Refuses to write outside v7_shape_transfer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.v7_runner import run_v7


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "v7_shape_transfer_probe.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = run_v7(args.config, dry_run=args.dry_run, limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("counts", {}).get("fatal_api_error"):
        return 3
    if args.dry_run and summary.get("counts", {}).get("payload_audit_failed"):
        return 2
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
