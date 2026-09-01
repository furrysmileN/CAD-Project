"""Cut 0：冻结 Harness vs CADrille 的 40 件评测集。零 API。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.hvc_sample import run_cut0


def main() -> int:
    result = run_cut0()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    coverage = result.get("coverage") or {}
    if not result.get("overlap_empty"):
        return 2
    if not coverage.get("pass_gate"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
