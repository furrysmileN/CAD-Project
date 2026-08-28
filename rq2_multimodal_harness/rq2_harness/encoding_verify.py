from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import atomic_write_json, project_path, read_jsonl, sha256_file


def _check_image_entry(
    entry: dict[str, Any],
    *,
    sample_id: str,
    encoding: str,
    issues: list[dict[str, Any]],
) -> None:
    images = entry.get("images")
    if not isinstance(images, list) or len(images) != 4:
        issues.append(
            {
                "code": "invalid_image_count",
                "sample_id": sample_id,
                "encoding": encoding,
                "actual": len(images) if isinstance(images, list) else None,
            }
        )
        return
    expected_views = ["front", "side", "top", "isometric"]
    if [item.get("view") for item in images] != expected_views:
        issues.append(
            {
                "code": "invalid_view_order",
                "sample_id": sample_id,
                "encoding": encoding,
            }
        )
    for item in images:
        path = Path(str(item.get("path", "")))
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(
                {
                    "code": "missing_image",
                    "sample_id": sample_id,
                    "encoding": encoding,
                    "path": str(path),
                }
            )
        elif item.get("sha256") != sha256_file(path):
            issues.append(
                {
                    "code": "image_hash_mismatch",
                    "sample_id": sample_id,
                    "encoding": encoding,
                    "path": str(path),
                }
            )


def verify_encoding_screen(
    output_dir: Path,
    *,
    expected_samples: int = 20,
    expected_conditions: int = 63,
    require_complete: bool = False,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    manifest_path = output_dir / "sample_manifest.jsonl"
    rows = list(read_jsonl(manifest_path)) if manifest_path.is_file() else []
    if len(rows) != expected_samples:
        issues.append(
            {
                "code": "sample_count_mismatch",
                "expected": expected_samples,
                "actual": len(rows),
            }
        )
    sample_ids = [str(row.get("sample_id")) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        issues.append({"code": "duplicate_sample_id"})

    for row in rows:
        sample_id = str(row.get("sample_id"))
        texts = row.get("text_encodings")
        if not isinstance(texts, dict) or set(texts) != {"T1", "T2", "T3"}:
            issues.append(
                {
                    "code": "invalid_text_encodings",
                    "sample_id": sample_id,
                }
            )
        else:
            for encoding, entry in texts.items():
                if not isinstance(entry, dict) or not str(entry.get("text", "")).strip():
                    issues.append(
                        {
                            "code": "empty_text_encoding",
                            "sample_id": sample_id,
                            "encoding": encoding,
                        }
                    )
        for group, names in (
            ("render_encodings", ("I1", "I2", "I3")),
            ("point_encodings", ("P1", "P2", "P3")),
        ):
            entries = row.get(group)
            if not isinstance(entries, dict) or set(entries) != set(names):
                issues.append(
                    {
                        "code": f"invalid_{group}",
                        "sample_id": sample_id,
                    }
                )
                continue
            for name in names:
                _check_image_entry(
                    entries[name],
                    sample_id=sample_id,
                    encoding=name,
                    issues=issues,
                )

    order_path = output_dir / "task_order.json"
    task_order = (
        json.loads(order_path.read_text(encoding="utf-8"))
        if order_path.is_file()
        else {}
    )
    tasks = task_order.get("tasks") or []
    expected_tasks = expected_samples * expected_conditions
    if tasks and len(tasks) != expected_tasks:
        issues.append(
            {
                "code": "task_order_count_mismatch",
                "expected": expected_tasks,
                "actual": len(tasks),
            }
        )
    task_keys = [
        (str(item.get("sample_id")), str(item.get("condition_id")))
        for item in tasks
    ]
    if len(task_keys) != len(set(task_keys)):
        issues.append({"code": "duplicate_task_key"})

    states = []
    state_root = output_dir / "state"
    if state_root.is_dir():
        for path in sorted(state_root.glob("*/*.json")):
            state = json.loads(path.read_text(encoding="utf-8"))
            states.append(state)
    if tasks and len(states) != len(tasks):
        issues.append(
            {
                "code": "state_count_mismatch",
                "expected": len(tasks),
                "actual": len(states),
            }
        )

    status_counts = Counter(str(state.get("status")) for state in states)
    if require_complete:
        bad = [
            state
            for state in states
            if state.get("status")
            not in {"completed", "parse_failed", "episode_failed"}
        ]
        if bad:
            issues.append(
                {
                    "code": "nonterminal_states",
                    "count": len(bad),
                }
            )
        missing_raw = [
            state
            for state in states
            if "raw_response" not in state
        ]
        if missing_raw:
            issues.append(
                {
                    "code": "missing_raw_response",
                    "count": len(missing_raw),
                }
            )
        missing_step = [
            state
            for state in states
            if state.get("status") == "completed"
            and (
                not state.get("result_step_path")
                or not Path(str(state["result_step_path"])).is_file()
            )
        ]
        if missing_step:
            issues.append(
                {
                    "code": "missing_completed_step",
                    "count": len(missing_step),
                }
            )

    report = {
        "schema_version": "rq2.encoding_screen.verification.v1",
        "output_dir": str(output_dir.resolve()),
        "expected_samples": expected_samples,
        "expected_conditions": expected_conditions,
        "expected_tasks": expected_tasks,
        "manifest_rows": len(rows),
        "task_order_rows": len(tasks),
        "state_rows": len(states),
        "status_counts": dict(status_counts),
        "require_complete": require_complete,
        "passed": not issues,
        "issues": issues,
    }
    atomic_write_json(output_dir / "verification.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="核验编码筛选资产与任务完整性")
    parser.add_argument(
        "--output-dir",
        default="experiments/rq2_multimodal_harness/outputs/encoding_screen_n20",
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    report = verify_encoding_screen(
        project_path(args.output_dir),
        require_complete=args.require_complete,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
