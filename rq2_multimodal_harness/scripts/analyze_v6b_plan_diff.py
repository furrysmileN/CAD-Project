# -*- coding: utf-8 -*-
"""Offline C3→C5 Plan-diff. No API. Does not overwrite v6b_probe_descriptive.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.v6b_plan_diff import write_plan_diff

FIX = ROOT / "outputs" / "v6_information_complementarity" / "pilot_v2_instrument_fix"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default=str(FIX / "probe" / "live" / "state"))
    parser.add_argument("--manifest", default=str(FIX / "manifest_probe.jsonl"))
    parser.add_argument("--latent-dir", default=str(FIX / "latent_specs"))
    parser.add_argument("--output-dir", default=str(FIX / "probe" / "live" / "analysis"))
    args = parser.parse_args()
    summary = write_plan_diff(Path(args.state_dir), Path(args.manifest), Path(args.latent_dir), Path(args.output_dir))
    printable = {k: v for k, v in summary.items() if k != "rows"}
    printable["n_rows"] = len(summary.get("rows") or [])
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
