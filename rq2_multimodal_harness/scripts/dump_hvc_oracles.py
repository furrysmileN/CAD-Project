"""写出 10 条 oracle JSON 并可选执行。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.backend import run_episode
from rq2_harness.common import project_path
from rq2_harness.hvc_oracles import ORACLES


def main() -> int:
    dest = project_path("experiments/rq2_multimodal_harness/outputs/harness_vs_cadrille/oracles")
    dest.mkdir(parents=True, exist_ok=True)
    config = {"episode_version": "v2", "root": "HarnessCAD/HarnessCAD", "timeout_sec": 30}
    failures = []
    for plan in ORACLES:
        (dest / f"{plan['sample_id']}.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        result = run_episode(plan, config)
        status = (result.get("response") or {}).get("status")
        if status not in {"success", "success_with_warnings"}:
            failures.append({"id": plan["sample_id"], "result": result})
    report = {"n": len(ORACLES), "failures": failures}
    (dest / "oracle_run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
