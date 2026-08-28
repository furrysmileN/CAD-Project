# -*- coding: utf-8 -*-
"""Descriptive analysis for V6b probe. No significance claims."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.v6b_probe_analysis import write_probe_analysis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-dir",
        default=str(
            ROOT
            / "outputs"
            / "v6_information_complementarity"
            / "pilot_v2_minimal_pairs"
            / "probe"
            / "live"
            / "state"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(
            ROOT
            / "outputs"
            / "v6_information_complementarity"
            / "pilot_v2_minimal_pairs"
            / "manifest_probe.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT
            / "outputs"
            / "v6_information_complementarity"
            / "pilot_v2_minimal_pairs"
            / "probe"
            / "live"
            / "analysis"
        ),
    )
    args = parser.parse_args()
    summary = write_probe_analysis(Path(args.state_dir), Path(args.manifest), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
