# -*- coding: utf-8 -*-
"""Analyze V7 transfer probe. Descriptive only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.v7_analysis import write_v7_analysis

FIX = ROOT / "outputs" / "v6_information_complementarity" / "v7_shape_transfer"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default=str(FIX / "probe" / "live" / "state"))
    parser.add_argument("--manifest", default=str(FIX / "manifest_probe.jsonl"))
    parser.add_argument("--output-dir", default=str(FIX / "probe" / "live" / "analysis"))
    args = parser.parse_args()
    summary = write_v7_analysis(Path(args.state_dir), Path(args.manifest), Path(args.output_dir))
    printable = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
