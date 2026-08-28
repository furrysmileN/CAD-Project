# -*- coding: utf-8 -*-
"""Phase 0: freeze environment hashes and replay frozen V5 plans (no API)."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2_harness.backend import run_episode
from rq2_harness.common import PROJECT_ROOT, atomic_write_json, sha256_file, sha256_json

BACKEND = {
    "episode_version": "v2",
    "root": str(PROJECT_ROOT / "HarnessCAD" / "HarnessCAD"),
    "timeout_sec": 30,
}
OUT = ROOT / "outputs" / "v6_information_complementarity"
CODE_FILES = [
    ROOT / "rq2_harness" / name
    for name in (
        "backend.py",
        "feedback.py",
        "repair_v21.py",
        "geometry.py",
        "prompting.py",
        "pc_runner.py",
        "v6_runner.py",
        "v6_conditions.py",
    )
]
TOL_REL = 1e-4
TOL_ABS = 1e-5


def _env() -> dict:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend_root": BACKEND["root"],
    }
    try:
        import cadquery as cq

        info["cadquery"] = getattr(cq, "__version__", "unknown")
    except Exception as exc:
        info["cadquery"] = f"unavailable: {exc}"
    try:
        import OCP

        info["ocp"] = getattr(OCP, "__version__", "present")
    except Exception:
        info["ocp"] = "unavailable"
    files = {str(path.relative_to(ROOT)): sha256_file(path) for path in CODE_FILES if path.is_file()}
    info["code_hashes"] = files
    info["code_sha256"] = sha256_json(files)
    return info


def _metrics(response: dict) -> dict:
    metrics = response.get("metrics") or {}
    failure = response.get("failure") or {}
    return {
        "status": response.get("status"),
        "first_failure_op": failure.get("operationId") or failure.get("operation_id") or failure.get("code"),
        "solidCount": metrics.get("solidCount"),
        "faceCount": metrics.get("faceCount"),
        "edgeCount": metrics.get("edgeCount"),
        "vertexCount": metrics.get("vertexCount"),
        "volume": metrics.get("volume"),
        "area": metrics.get("area"),
        "bboxSize": metrics.get("bboxSize"),
        "bboxCenter": metrics.get("bboxCenter"),
    }


def _close(a, b) -> bool:
    if a is None or b is None:
        return a == b
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(_close(x, y) for x, y in zip(a, b))
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= max(TOL_ABS, TOL_REL * max(abs(fa), abs(fb), 1.0))
    except (TypeError, ValueError):
        return a == b


def _collect_plans(limit: int = 50) -> list[dict]:
    state_root = ROOT / "outputs" / "v5_complementarity" / "repeats" / "state"
    items = []
    for path in sorted(state_root.glob("*/*/r01.json")):
        if len(items) >= limit:
            break
        state = json.loads(path.read_text(encoding="utf-8"))
        plan = state.get("repaired_plan")
        if not isinstance(plan, dict):
            continue
        if state.get("status") != "completed":
            continue
        episode = ((state.get("episode") or {}).get("response")) or {}
        items.append({"path": str(path), "plan": plan, "baseline": _metrics(episode)})
    return items


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audits").mkdir(parents=True, exist_ok=True)
    env = _env()
    atomic_write_json(OUT / "audits" / "environment.json", env)
    plans = _collect_plans(50)
    replay_root = OUT / "audits" / "golden_replay"
    rows = []
    n_status = n_topo = n_geom = 0
    for item in plans:
        key = hashlib.sha256(item["path"].encode("utf-8")).hexdigest()[:12]
        try:
            result = run_episode(item["plan"], BACKEND, run_root=replay_root)
            current = _metrics(result.get("response") or {})
            error = None
        except Exception as exc:
            current = {}
            error = f"{type(exc).__name__}: {exc}"
        base = item["baseline"]
        status_ok = current.get("status") == base.get("status")
        topo_ok = all(
            current.get(k) == base.get(k) for k in ("solidCount", "faceCount", "edgeCount", "vertexCount", "first_failure_op")
        )
        geom_ok = all(_close(current.get(k), base.get(k)) for k in ("volume", "area", "bboxSize", "bboxCenter"))
        n_status += int(status_ok)
        n_topo += int(topo_ok)
        n_geom += int(geom_ok)
        rows.append(
            {
                "source": item["path"],
                "key": key,
                "status_ok": status_ok,
                "topology_ok": topo_ok,
                "geometry_ok": geom_ok,
                "baseline_status": base.get("status"),
                "replay_status": current.get("status"),
                "error": error,
            }
        )
    csv_path = OUT / "audits" / "golden_episode_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["source"])
        writer.writeheader()
        writer.writerows(rows)
    n = max(len(rows), 1)
    go = len(rows) >= 50 and n_status == len(rows) and n_topo == len(rows) and n_geom == len(rows)
    report = {
        "n_replayed": len(rows),
        "status_match": n_status,
        "topology_match": n_topo,
        "geometry_match": n_geom,
        "go": go,
        "note": "同机冻结而非跨机迁移。不要求 STEP 二进制 SHA 一致。",
    }
    atomic_write_json(OUT / "audits" / "golden_episode_summary.json", report)
    md = OUT / "MIGRATION_AUDIT_ZH.md"
    md.write_text(
        "\n".join(
            [
                "# V6 Phase 0 环境冻结与 Golden Episode 审计",
                "",
                f"- 时间：{env['created_at']}",
                f"- Python：{env['python'].splitlines()[0]}",
                f"- CadQuery：{env.get('cadquery')}",
                f"- 重放样本：{len(rows)}（目标 50）",
                f"- status 一致：{n_status}/{len(rows)}",
                f"- 拓扑一致：{n_topo}/{len(rows)}",
                f"- 几何量在容差内：{n_geom}/{len(rows)}",
                f"- Go/No-Go：{'GO' if go else 'NO-GO（详见 audits/golden_episode_comparison.csv）'}",
                "",
                "本机即 V5 实验机，本步骤是同机冻结，不是跨机迁移。未调用模型 API。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if len(rows) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
