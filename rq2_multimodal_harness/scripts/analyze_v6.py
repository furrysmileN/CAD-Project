# -*- coding: utf-8 -*-
"""V6 K1–K6 analysis. Default uses mock rows to verify contrast signs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.common import atomic_write_json, project_path
from rq2_harness.v6_analysis import analyze_v6, mock_rows


def _load_live_rows(state_dir: Path) -> list[dict]:
    rows = []
    for path in state_dir.glob("*/*/r*.json"):
        state = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "sample_id": state.get("sample_id"),
                "condition": state.get("condition"),
                "repeat_id": state.get("repeat_id"),
                "first_attempt": (state.get("first_attempt") or {}).get("geometry")
                or state.get("first_attempt"),
                "final_delivery": (state.get("final_delivery") or {}).get("geometry")
                or state.get("final_delivery"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--live-state", default="")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "v6_information_complementarity" / "analysis"),
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.live_state:
        rows = _load_live_rows(Path(args.live_state))
        source = "live"
    else:
        rows = mock_rows(20, repeats=1)
        source = "mock"
    jq = analyze_v6(rows, endpoint="first_attempt", metric="joint_quality")
    cd = analyze_v6(rows, endpoint="first_attempt", metric="common_frame_cd", invert=True)
    atomic_write_json(output / "contrasts_holm.json", jq)
    atomic_write_json(output / "contrasts_cd.json", cd)
    with (output / "contrasts_holm.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "title", "mean_delta", "ci95_low", "ci95_high", "wilcoxon_p", "p_holm", "n_left_better"],
        )
        writer.writeheader()
        for row in jq["contrasts_holm"]:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    signs = {
        "source": source,
        "k1_positive": (jq["contrasts_holm"][0].get("mean_delta") or 0) > 0,
        "k2_positive": (jq["contrasts_holm"][1].get("mean_delta") or 0) > 0,
        "k3_positive": (jq["contrasts_holm"][2].get("mean_delta") or 0) > 0,
        "k4_positive": (jq["contrasts_holm"][3].get("mean_delta") or 0) > 0,
        "cd_k1_positive": (cd["contrasts_holm"][0].get("mean_delta") or 0) > 0,
        "jq": jq,
    }
    atomic_write_json(output / "sign_check.json", {key: value for key, value in signs.items() if key != "jq"})
    print(json.dumps({k: v for k, v in signs.items() if k != "jq"}, ensure_ascii=False, indent=2))
    ok = all(signs[k] for k in ("k1_positive", "k2_positive", "k3_positive", "k4_positive", "cd_k1_positive"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
