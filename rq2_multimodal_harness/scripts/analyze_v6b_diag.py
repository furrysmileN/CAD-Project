# -*- coding: utf-8 -*-
"""Analyze C2B or TB follow-B vs frozen C5. Descriptive only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.v6b_diag_analysis import write_diag_analysis

FIX = ROOT / "outputs" / "v6_information_complementarity" / "pilot_v2_instrument_fix"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["C2B", "TB"], required=True)
    parser.add_argument("--diag-state-dir", default=None)
    parser.add_argument("--probe-state-dir", default=str(FIX / "probe" / "live" / "state"))
    parser.add_argument("--manifest", default=str(FIX / "manifest_probe.jsonl"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    split = "diag_c2b" if args.condition == "C2B" else "diag_tb"
    diag_state = Path(args.diag_state_dir) if args.diag_state_dir else FIX / split / "live" / "state"
    output_dir = Path(args.output_dir) if args.output_dir else FIX / split / "live" / "analysis"
    c2b_state = FIX / "diag_c2b" / "live" / "state" if args.condition == "TB" else None
    summary = write_diag_analysis(
        diag_state,
        Path(args.probe_state_dir),
        Path(args.manifest),
        output_dir,
        diag_condition=args.condition,
        c2b_state_dir=c2b_state,
    )
    printable = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
