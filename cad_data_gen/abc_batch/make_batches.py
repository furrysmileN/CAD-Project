"""从 STEP/OFS 配对清单生成可搬运的批次 manifest。

该阶段读取 `pair_samples.py` 产出的 `pairing_manifest.jsonl`，补充输入文件大小估计，
并按样本数量或预计输入字节数切分为 `<work-root>/batches/<batch_id>/manifest.jsonl`。

输出用于真实环境侧 staging、虚拟环境侧资产生成和最终归档索引。
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import traceback
from pathlib import Path
from typing import Iterable, Optional

from .logging_utils import append_jsonl, iter_jsonl, now_iso, stage_logger, write_json
from .paths import WorkRootLayout, work_root_layout

LOGGER_NAME = "make_batches"
DEFAULT_STATUS = "pending"


def _safe_file_size(path_value: object) -> int:
    if not path_value:
        return 0
    try:
        path = Path(str(path_value))
        if not path.is_file():
            return 0
        return path.stat().st_size
    except OSError:
        return 0


def _enrich_record(record: dict) -> dict:
    step_size = _safe_file_size(record.get("step_path"))
    ofs_size = _safe_file_size(record.get("ofs_path"))
    input_size = step_size + ofs_size
    enriched = dict(record)
    enriched["step_size_bytes"] = step_size
    enriched["ofs_size_bytes"] = ofs_size
    enriched["estimated_input_size_bytes"] = input_size
    enriched.setdefault("estimated_output_size_bytes", None)
    enriched["batch_status"] = DEFAULT_STATUS
    enriched.setdefault("has_ofs", bool(enriched.get("ofs_path")))
    enriched.setdefault("ofs_status", "matched" if enriched.get("has_ofs") else "unmatched")
    enriched.setdefault("status", "paired" if enriched.get("has_ofs") else "step_only")
    return enriched


def _reset_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.touch()


def _reset_batch_dir(batch_dir: Path) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.jsonl", "state.json", "summary.json"):
        path = batch_dir / name
        if path.exists():
            path.unlink()


def _batch_id(prefix: str, index: int, width: int) -> str:
    return f"{prefix}{index:0{width}d}"


def _split_batches(
    records: Iterable[dict],
    *,
    max_samples_per_batch: Optional[int],
    max_input_bytes_per_batch: Optional[int],
) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = 0

    for record in records:
        record_bytes = int(record.get("estimated_input_size_bytes") or 0)
        count_limit_hit = (
            max_samples_per_batch is not None
            and len(current) >= max_samples_per_batch
        )
        byte_limit_hit = (
            max_input_bytes_per_batch is not None
            and current
            and current_bytes + record_bytes > max_input_bytes_per_batch
        )
        if count_limit_hit or byte_limit_hit:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(record)
        current_bytes += record_bytes

    if current:
        batches.append(current)
    return batches


def _write_batch(
    layout: WorkRootLayout,
    *,
    batch_id: str,
    records: list[dict],
    source_manifest: Path,
) -> dict:
    batch_dir = layout.batches_dir / batch_id
    _reset_batch_dir(batch_dir)
    manifest_path = batch_dir / "manifest.jsonl"
    state_path = batch_dir / "state.json"
    summary_path = batch_dir / "summary.json"
    manifest_path.touch()

    n_has_ofs = 0
    estimated_input_size = 0
    sample_ids: list[str] = []
    for record in records:
        out = dict(record)
        out["batch_id"] = batch_id
        out["batch_status"] = DEFAULT_STATUS
        append_jsonl(manifest_path, out)
        sample_ids.append(str(out.get("sample_id")))
        estimated_input_size += int(out.get("estimated_input_size_bytes") or 0)
        if out.get("has_ofs"):
            n_has_ofs += 1

    state = {
        "batch_id": batch_id,
        "status": DEFAULT_STATUS,
        "created_at": now_iso(),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "source_manifest": str(source_manifest),
        "n_samples": len(records),
        "n_has_ofs": n_has_ofs,
        "n_missing_ofs": len(records) - n_has_ofs,
        "estimated_input_size_bytes": estimated_input_size,
        "estimated_output_size_bytes": None,
    }
    summary = {
        **state,
        "sample_id_first": sample_ids[0] if sample_ids else None,
        "sample_id_last": sample_ids[-1] if sample_ids else None,
    }
    write_json(state_path, state)
    write_json(summary_path, summary)
    return {
        "batch_id": batch_id,
        "status": DEFAULT_STATUS,
        "batch_dir": str(batch_dir),
        "manifest_path": str(manifest_path),
        "state_path": str(state_path),
        "summary_path": str(summary_path),
        "n_samples": len(records),
        "n_has_ofs": n_has_ofs,
        "n_missing_ofs": len(records) - n_has_ofs,
        "estimated_input_size_bytes": estimated_input_size,
        "estimated_output_size_bytes": None,
    }


def run_make_batches(
    layout: WorkRootLayout,
    *,
    source_manifest: Optional[Path],
    batch_prefix: str,
    max_samples_per_batch: Optional[int],
    max_input_bytes_per_batch: Optional[int],
    logger: logging.Logger,
) -> dict:
    manifest = (source_manifest or layout.pairing_manifest).expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"source manifest not found: {manifest}")
    if max_samples_per_batch is None and max_input_bytes_per_batch is None:

-max-input-bytes")
    if max_samples_per_batch is not None and max_samples_per_batch <= 0:
        raise ValueError("--max-samples-per-batch must be > 0")if max_input_bytes_per_batch is not None and max_input_bytes_per_batc
h <= 0:
        raise ValueError("--max-input-bytes/--max-input-gb must be > 0")

    layout.ensure_dirs()
    _reset_file(layout.global_samples_manifest)
    _reset_file(layout.batches_index)

    logger.info("loading source manifest: %s", manifest)
    enriched_records: list[dict] = []
    n_missing_ofs = 0
    total_input_size = 0
    for record in iter_jsonl(manifest):
        enriched = _enrich_record(record)
        append_jsonl(layout.global_samples_manifest, enriched)
        enriched_records.append(enriched)total_input_size += int(enriched.get("estimated_input_size_byte
s") or 0)
        if not enriched.get("has_ofs"):
            n_missing_ofs += 1

    batches = _split_batches(
        enriched_records,
        max_samples_per_batch=max_samples_per_batch,
        max_input_bytes_per_batch=max_input_bytes_per_batch,
    )
    width = max(4, int(math.log10(max(len(batches), 1))) + 1)
    batch_records: list[dict] = []
    for idx, records in enumerate(batches):
        bid = _batch_id(batch_prefix, idx, width)batch_record = _write_batch(layout, batch_id=bid, records=record
s, source_manifest=manifest)
        append_jsonl(layout.batches_index, batch_record)
        batch_records.append(batch_record)

    summary = {
        "ts": now_iso(),
        "work_root": str(layout.work_root),
        "source_manifest": str(manifest),
        "global_samples_manifest": str(layout.global_samples_manifest),
        "batches_dir": str(layout.batches_dir),
        "batches_index": str(layout.batches_index),
        "n_samples": len(enriched_records),
        "n_batches": len(batch_records),
        "n_missing_ofs": n_missing_ofs,
        "n_has_ofs": len(enriched_records) - n_missing_ofs,
        "estimated_input_size_bytes": total_input_size,
        "batch_prefix": batch_prefix,
        "max_samples_per_batch": max_samples_per_batch,
        "max_input_bytes_per_batch": max_input_bytes_per_batch,"status": "ok" if batch_records or not enriched_records else "emp
ty",
    }
    write_json(layout.batches_dir / "make_batches_summary.json", summary)
    logger.info(
        "created %d batches for %d samples under %s",
        len(batch_records),
        len(enriched_records),
        layout.batches_dir,
    )
    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create batch manifests from abc_batch pairing_manife
st.jsonl."
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="默认读取 <work-root>/pairing_manifest.jsonl",
    )parser.add_argument("--batch-prefix", default="batch_", help="批次 I
D 前缀")parser.add_argument("--max-samples-per-batch", type=int, default=Non
e)
    parser.add_argument("--max-input-bytes", type=int, default=None)
    parser.add_argument(
        "--max-input-gb",
        type=float,
        default=None,help="按 GiB 设置单批预计输入大小上限，可与 --max-samples-per-batch 同
时使用",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    layout = work_root_layout(args.work_root)
    max_input_bytes = args.max_input_bytes
    if args.max_input_gb is not None:
        max_input_bytes = int(args.max_input_gb * 1024**3)

    with stage_logger(LOGGER_NAME, layout.make_batches_log) as logger:
        try:
            run_make_batches(
                layout,
                source_manifest=args.source_manifest,
                batch_prefix=args.batch_prefix,
                max_samples_per_batch=args.max_samples_per_batch,
                max_input_bytes_per_batch=max_input_bytes,
                logger=logger,
            )
        except KeyboardInterrupt:
            logger.warning("interrupted by user")
            return 130
        except BaseException as exc:  # noqa: BLE001logger.error("make_batches failed: %s\n%s", exc, traceback.fo
rmat_exc())
            return 1
    return 0


if __name__ == "__main__":sys exit(main())
