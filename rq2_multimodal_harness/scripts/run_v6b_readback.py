# -*- coding: utf-8 -*-
"""V6b evidence readback. Informal diagnostic; does not run CAD probe or confirm."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.v6b_readback import run_v6b_readback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "v6b_readback.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = run_v6b_readback(args.config, dry_run=args.dry_run, limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("counts", {}).get("fatal_api_error"):
        return 3
    if summary.get("counts", {}).get("payload_audit_failed"):
        return 2
    if not args.dry_run and not (summary.get("gates") or {}).get("pass"):
        return 4
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
