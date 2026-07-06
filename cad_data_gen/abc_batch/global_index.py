"""全局索引合并与补跑批次生成工具。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from .archive_batch import ARCHIVED_MARKER
from .logging_utils import append_jsonl, iter_jsonl, now_iso, read_json, stage_logger, write_json

LOGGER_NAME = "global_index"


def _reset_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.touch()


def _load_archive_state(batch_dir: Path) -> dict[str, Any]:
    state = read_json(batch_dir / "archive_state.json")
    return state if isinstance(state, dict) else {}


def _is_archived_batch(batch_dir: Path) -> tuple[bool, str]:
    marker = batch_dir / ARCHIVED_MARKER
    if not marker.is_file():
        return False, "missing_ARCHIVED_marker"
    manifest = batch_dir / "archive_manifest.jsonl"
    if not manifest.is_file():
        return False, "missing_archive_manifest"
    state = _load_archive_state(batch_dir)
    verification = state.get("extra", {}).get("verification") if isinstance(state.get("extra"), dict) else None
    if isinstance(verification, dict) and not verification.get("ok", False):
        return False, "archive_verification_failed"
    return True, "ok"


def _asset_manifest_for_batch(batch_dir: Path) -> Optional[Path]:
    candidates = [
        batch_dir / "assets" / "manifest.jsonl",
        batch_dir / "manifest.jsonl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _sample_status_for_batch(batch_dir: Path) -> Optional[Path]:
    path = batch_dir / "assets" / "sample_status.jsonl"
    return path if path.is_file() else None


def run_merge_global_index(
    *,
    archive_root: Path,
    output_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
    logger: logging.Logger,
) -> dict[str, Any]:
    root = archive_root.expanduser().resolve()
    batches_root = root / "batches"
    out_path = output_path.expanduser().resolve() if output_path else root / "global_index.jsonl"
    rep_path = report_path.expanduser().resolve() if report_path else root / "global_index_report.json"
    _reset_jsonl(out_path)

    n_batches = 0
    n_archived = 0
    n_skipped = 0
    n_records = 0
    n_duplicate_samples = 0
    seen: set[str] = set()
    skipped_batches: list[dict[str, Any]] = []

    for batch_dir in sorted(p for p in batches_root.iterdir() if p.is_dir()) if batches_root.exists() else []:
        n_batches += 1
        ok, reason = _is_archived_batch(batch_dir)
        if not ok:
            n_skipped += 1
            skipped_batches.append({"batch_id": batch_dir.name, "reason": reason})
            continue
        asset_manifest = _asset_manifest_for_batch(batch_dir)
        if asset_manifest is None:
            n_skipped += 1
            skipped_batches.append({"batch_id": batch_dir.name, "reason": "missing_assets_manifest"})
            continue
        n_archived += 1
        status_path = _sample_status_for_batch(batch_dir)
        status_by_sample = {
            str(row.get("sample_id")): row for row in iter_jsonl(status_path)
        } if status_path else {}
        for row in iter_jsonl(asset_manifest):
            sample_id = str(row.get("sample_id") or "").strip()
            if not sample_id:
                continue
            if sample_id in seen:
                n_duplicate_samples += 1
            seen.add(sample_id)
            out = dict(row)
            out["batch_id"] = batch_dir.name
            out["archive_batch_dir"] = str(batch_dir)
            out["asset_manifest_path"] = str(asset_manifest)
            if sample_id in status_by_sample:
                out["sample_status"] = status_by_sample[sample_id].get("status")
            append_jsonl(out_path, out)
            n_records += 1

    report = {
        "generated_at": now_iso(),
        "archive_root": str(root),
        "global_index_path": str(out_path),
        "n_batches": n_batches,
        "n_archived_batches": n_archived,
        "n_skipped_batches": n_skipped,
        "n_index_records": n_records,
        "n_duplicate_samples": n_duplicate_samples,
        "skipped_batches": skipped_batches,
        "status": "ok" if n_records > 0 or n_batches == 0 else "empty",
    }
    write_json(rep_path, report)
    logger.info("global index written: records=%d archived_batches=%d skipped=%d", n_records, n_archived, n_skipped)
    return report


def _failure_rows_from_archives(archive_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batches_root = archive_root / "batches"
    if not batches_root.exists():
        return rows
    for batch_dir in sorted(p for p in batches_root.iterdir() if p.is_dir()):
        status_path = _sample_status_for_batch(batch_dir)
        if status_path:
            for row in iter_jsonl(status_path):
                if row.get("status") == "failed":
                    failed = dict(row)
                    failed.setdefault("batch_id", batch_dir.name)
                    rows.append(failed)
        failures_path = batch_dir / "assets" / "failures.jsonl"
        if failures_path.is_file():
            for row in iter_jsonl(failures_path):
                failed = dict(row)
                failed.setdefault("batch_id", batch_dir.name)
                rows.append(failed)
    return rows


def _source_record_lookup(source_manifest: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(source_manifest):
        sample_id = str(row.get("sample_id") or "").strip()
        if sample_id:
            lookup[sample_id] = dict(row)
    return lookup


def run_make_retry_batch(
    *,
    archive_root: Path,
    source_manifest: Path,
    output_dir: Path,
    batch_id: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    root = archive_root.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    manifest_path = output / "manifest.jsonl"
    failure_list_path = output / "retry_failures.jsonl"
    state_path = output / "state.json"
    output.mkdir(parents=True, exist_ok=True)
    _reset_jsonl(manifest_path)
    _reset_jsonl(failure_list_path)

    lookup = _source_record_lookup(source_manifest.expanduser().resolve())
    failed_rows = _failure_rows_from_archives(root)
    seen: set[str] = set()
    n_written = 0
    n_missing_source = 0
    for failure in failed_rows:
        sample_id = str(failure.get("sample_id") or "").strip()
        if not sample_id or sample_id in seen:
            continue
        seen.add(sample_id)
        source = lookup.get(sample_id)
        if source is None:
            n_missing_source += 1
            append_jsonl(failure_list_path, {**failure, "retry_status": "missing_source_record"})
            continue
        record = dict(source)
        record["batch_id"] = batch_id
        record["batch_status"] = "pending"
        record["retry_of_batch_id"] = failure.get("batch_id")
        record["retry_reason"] = failure.get("latest_failure") or failure.get("error") or failure.get("error_type")
        append_jsonl(manifest_path, record)
        append_jsonl(failure_list_path, {**failure, "retry_status": "included"})
        n_written += 1

    state = {
        "batch_id": batch_id,
        "status": "pending",
        "created_at": now_iso(),
        "manifest_path": str(manifest_path),
        "failure_list_path": str(failure_list_path),
        "source_manifest": str(source_manifest),
        "n_failed_rows_seen": len(failed_rows),
        "n_retry_samples": n_written,
        "n_missing_source_records": n_missing_source,
    }
    write_json(state_path, state)
    logger.info("retry batch %s written: samples=%d missing_source=%d", batch_id, n_written, n_missing_source)
    return state


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge archived ABC batches or create retry batch manifests.")
    sub = parser.add_subparsers(dest="command", required=True)

    merge = sub.add_parser("merge", help="合并已归档批次的全局索引")
    merge.add_argument("--archive-root", type=Path, required=True)
    merge.add_argument("--output", type=Path, default=None)
    merge.add_argument("--report", type=Path, default=None)

    retry = sub.add_parser("retry", help="根据失败列表生成补跑批次")
    retry.add_argument("--archive-root", type=Path, required=True)
    retry.add_argument("--source-manifest", type=Path, required=True)
    retry.add_argument("--output-dir", type=Path, required=True)
    retry.add_argument("--batch-id", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    log_root = args.archive_root if hasattr(args, "archive_root") else Path.cwd()
    with stage_logger(LOGGER_NAME, Path(log_root).expanduser().resolve() / "global_index.log") as logger:
        if args.command == "merge":
            run_merge_global_index(
                archive_root=args.archive_root,
                output_path=args.output,
                report_path=args.report,
                logger=logger,
            )
            return 0
        if args.command == "retry":
            run_make_retry_batch(
                archive_root=args.archive_root,
                source_manifest=args.source_manifest,
                output_dir=args.output_dir,
                batch_id=args.batch_id,
                logger=logger,
            )
            return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
