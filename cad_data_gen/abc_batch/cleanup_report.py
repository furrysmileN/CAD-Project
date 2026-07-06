"""批次安全清理与全局完成报告。"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .archive_batch import ARCHIVED_MARKER
from .logging_utils import iter_jsonl, now_iso, read_json, safe_disk_usage, stage_logger, write_json
from .paths import work_root_layout

LOGGER_NAME = "cleanup_report"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _is_batch_archived(archive_root: Path, batch_id: str) -> bool:
    return (archive_root / "batches" / batch_id / ARCHIVED_MARKER).is_file()


def run_cleanup_batch(
    *,
    work_root: Path,
    batch_id: str,
    staging_root: Path,
    archive_root: Path,
    remove_assets_input: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    layout = work_root_layout(work_root)
    staging_dir = staging_root.expanduser().resolve() / batch_id
    archive = archive_root.expanduser().resolve()
    if not _is_batch_archived(archive, batch_id):
        raise RuntimeError(f"batch {batch_id} is not archived under {archive}; refuse cleanup")

    candidates = [staging_dir]
    if remove_assets_input:
        candidates.append(layout.work_root / "assets_input")

    removed: list[dict[str, Any]] = []
    reclaimed = 0
    for path in candidates:
        size = _dir_size(path)
        removed.append({"path": str(path), "exists": path.exists(), "size_bytes": size, "removed": False})
        if not path.exists():
            continue
        reclaimed += size
        if not dry_run:
            shutil.rmtree(path)
            removed[-1]["removed"] = True

    report = {
        "generated_at": now_iso(),
        "work_root": str(layout.work_root),
        "batch_id": batch_id,
        "archive_root": str(archive),
        "dry_run": dry_run,
        "reclaimed_bytes": reclaimed,
        "removed": removed,
    }
    write_json(layout.work_root / f"cleanup_{batch_id}.json", report)
    return report


def _count_jsonl(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path)) if path.exists() else 0


def _status_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in iter_jsonl(path):
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def run_completion_report(
    *,
    archive_root: Path,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    root = archive_root.expanduser().resolve()
    batches_root = root / "batches"
    started = time.time()
    batches: list[dict[str, Any]] = []
    totals = {
        "n_batches": 0,
        "n_archived_batches": 0,
        "n_success_samples": 0,
        "n_failed_samples": 0,
        "n_pending_samples": 0,
        "n_missing_ofs_samples": 0,
        "n_asset_records": 0,
        "n_occlusion_records": 0,
        "n_descriptions": 0,
        "total_output_bytes": 0,
    }

    for batch_dir in sorted(p for p in batches_root.iterdir() if p.is_dir()) if batches_root.exists() else []:
        totals["n_batches"] += 1
        archived = (batch_dir / ARCHIVED_MARKER).is_file()
        if archived:
            totals["n_archived_batches"] += 1
        asset_manifest = batch_dir / "assets" / "manifest.jsonl"
        occlusion_manifest = batch_dir / "occlusion" / "manifest.jsonl"
        descriptions = batch_dir / "descriptions" / "descriptions.jsonl"
        sample_status = batch_dir / "assets" / "sample_status.jsonl"
        state = read_json(batch_dir / "archive_state.json") or {}

        status_counts = _status_counts(sample_status)
        n_success = status_counts.get("done", 0)
        n_failed = status_counts.get("failed", 0)
        n_pending = status_counts.get("pending", 0)
        n_missing_ofs = 0
        for row in iter_jsonl(asset_manifest):
            if not row.get("has_ofs"):
                n_missing_ofs += 1

        n_assets = _count_jsonl(asset_manifest)
        n_occlusion = _count_jsonl(occlusion_manifest)
        n_descriptions = _count_jsonl(descriptions)
        output_bytes = int(state.get("total_bytes") or _dir_size(batch_dir))

        totals["n_success_samples"] += n_success
        totals["n_failed_samples"] += n_failed
        totals["n_pending_samples"] += n_pending
        totals["n_missing_ofs_samples"] += n_missing_ofs
        totals["n_asset_records"] += n_assets
        totals["n_occlusion_records"] += n_occlusion
        totals["n_descriptions"] += n_descriptions
        totals["total_output_bytes"] += output_bytes
        batches.append(
            {
                "batch_id": batch_dir.name,
                "archived": archived,
                "n_success_samples": n_success,
                "n_failed_samples": n_failed,
                "n_pending_samples": n_pending,
                "n_missing_ofs_samples": n_missing_ofs,
                "n_asset_records": n_assets,
                "n_occlusion_records": n_occlusion,
                "n_descriptions": n_descriptions,
                "output_bytes": output_bytes,
            }
        )

    disk = safe_disk_usage(root)
    report = {
        "generated_at": now_iso(),
        "archive_root": str(root),
        "elapsed_s": round(time.time() - started, 2),
        "totals": totals,
        "disk": {
            "total_bytes": disk[0],
            "used_bytes": disk[1],
            "free_bytes": disk[2],
        }
        if disk
        else None,
        "batches": batches,
        "status": "ok" if totals["n_batches"] == totals["n_archived_batches"] else "incomplete",
    }
    out = output_path.expanduser().resolve() if output_path else root / "completion_report.json"
    write_json(out, report)
    return report


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup staged ABC batch data or write completion report.")
    sub = parser.add_subparsers(dest="command", required=True)

    cleanup = sub.add_parser("cleanup", help="成功归档后清理 staging 与 assets_input")
    cleanup.add_argument("--work-root", type=Path, required=True)
    cleanup.add_argument("--batch-id", required=True)
    cleanup.add_argument("--staging-root", type=Path, required=True)
    cleanup.add_argument("--archive-root", type=Path, required=True)
    cleanup.add_argument("--keep-assets-input", action="store_true")
    cleanup.add_argument("--dry-run", action="store_true")

    report = sub.add_parser("report", help="生成全局完成报告")
    report.add_argument("--archive-root", type=Path, required=True)
    report.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    log_root = args.work_root if hasattr(args, "work_root") else args.archive_root
    with stage_logger(LOGGER_NAME, Path(log_root).expanduser().resolve() / "cleanup_report.log"):
        if args.command == "cleanup":
            run_cleanup_batch(
                work_root=args.work_root,
                batch_id=args.batch_id,
                staging_root=args.staging_root,
                archive_root=args.archive_root,
                remove_assets_input=not args.keep_assets_input,
                dry_run=args.dry_run,
            )
            return 0
        if args.command == "report":
            run_completion_report(archive_root=args.archive_root, output_path=args.output)
            return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
