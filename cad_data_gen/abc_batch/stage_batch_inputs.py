"""真实环境侧批次输入 staging 工具。

读取 `make_batches.py` 生成的 `<work-root>/batches/<batch_id>/manifest.jsonl`，
只复制该批次列出的 STEP/OFS 文件到虚拟环境可访问的 staging 区，并写出：

- `manifest.jsonl`：输入 manifest 快照，包含 `staged_step_path` / `staged_ofs_path`
- `checksums.jsonl`：每个已复制文件的大小与 sha256
- `state.json`：批次 staging 状态
- `STAGED`：可校验的完成标记

该模块只负责“数据盘 -> staging 区”的输入搬运，不运行资产生成。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .logging_utils import append_jsonl, iter_jsonl, now_iso, read_json, safe_disk_usage, stage_logger, write_json
from .paths import WorkRootLayout, work_root_layout

LOGGER_NAME = "stage_batch_inputs"
DONE_MARKER = "STAGED"


@dataclass
class StageBatchInputsResult:
    stage: str = "stage_batch_inputs"
    status: str = "pending"
    batch_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    elapsed_s: float = 0.0
    n_samples: int = 0
    n_step_files: int = 0
    n_ofs_files: int = 0
    n_missing_files: int = 0
    copied_bytes: int = 0
    staging_dir: str = ""
    manifest_path: str = ""
    checksums_path: str = ""
    state_path: str = ""
    marker_path: str = ""
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_sample_component(value: object, fallback: str) -> str:
    text = str(value or fallback).strip() or fallback
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)[:180] or fallback


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _copy_or_link_file(
    src: Path,
    dst: Path,
    *,
    mode: str,
    overwrite: bool,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _load_batch_records(manifest_path: Path) -> list[dict[str, Any]]:
    records = [dict(r) for r in iter_jsonl(manifest_path)]
    if not records:
        raise ValueError(f"empty batch manifest: {manifest_path}")
    return records


def _estimate_required_bytes(records: list[dict[str, Any]]) -> int:
    total = 0
    for record in records:
        if record.get("estimated_input_size_bytes") is not None:
            try:
                total += int(record.get("estimated_input_size_bytes") or 0)
                continue
            except (TypeError, ValueError):
                pass
        for key in ("step_path", "ofs_path"):
            path_value = record.get(key)
            if not path_value:
                continue
            try:
                path = Path(str(path_value))
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    return total


def _check_free_space(
    staging_root: Path,
    *,
    required_bytes: int,
    min_free_bytes: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    staging_root.mkdir(parents=True, exist_ok=True)
    usage = safe_disk_usage(staging_root)
    info: dict[str, Any] = {
        "required_bytes": required_bytes,
        "min_free_bytes": min_free_bytes,
        "disk_total_bytes": None,
        "disk_used_bytes": None,
        "disk_free_bytes": None,
        "ok": True,
    }
    if usage is None:
        logger.warning("cannot inspect disk usage for %s; continue without free-space guard", staging_root)
        info["ok"] = True
        return info
    total, used, free = usage
    info.update({"disk_total_bytes": total, "disk_used_bytes": used, "disk_free_bytes": free})
    need = required_bytes + min_free_bytes
    info["ok"] = free >= need
    if not info["ok"]:
        raise RuntimeError(
            f"insufficient free space under {staging_root}: free={free}, "
            f"required_input={required_bytes}, min_free={min_free_bytes}"
        )
    return info


def _write_empty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _reset_output_files(staging_dir: Path) -> None:
    for name in ("manifest.jsonl", "checksums.jsonl", "state.json", DONE_MARKER):
        path = staging_dir / name
        if path.exists() or path.is_symlink():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()


def _copy_record_inputs(
    record: dict[str, Any],
    *,
    index: int,
    staging_dir: Path,
    copy_mode: str,
    overwrite: bool,
    checksums_path: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    sample_id = _safe_sample_component(record.get("sample_id"), f"sample_{index:06d}")
    staged_record = dict(record)
    stats = {"step": 0, "ofs": 0, "missing": 0, "bytes": 0}

    for kind, source_key, staged_key in (
        ("step", "step_path", "staged_step_path"),
        ("ofs", "ofs_path", "staged_ofs_path"),
    ):
        source_value = record.get(source_key)
        if not source_value:
            staged_record[staged_key] = None
            continue
        src = Path(str(source_value)).expanduser()
        if not src.is_file():
            stats["missing"] += 1
            staged_record[staged_key] = None
            append_jsonl(
                checksums_path,
                {
                    "sample_id": record.get("sample_id"),
                    "kind": kind,
                    "source_path": str(src),
                    "staged_path": None,
                    "status": "missing",
                    "ts": now_iso(),
                },
            )
            continue
        dst = staging_dir / "inputs" / kind / sample_id / src.name
        _copy_or_link_file(src, dst, mode=copy_mode, overwrite=overwrite)
        size = dst.stat().st_size
        checksum = _sha256_file(dst)
        staged_record[staged_key] = str(dst)
        stats[kind] += 1
        stats["bytes"] += size
        append_jsonl(
            checksums_path,
            {
                "sample_id": record.get("sample_id"),
                "kind": kind,
                "source_path": str(src),
                "staged_path": str(dst),
                "size_bytes": size,
                "sha256": checksum,
                "copy_mode": copy_mode,
                "status": "ok",
                "ts": now_iso(),
            },
        )

    staged_record["staging_status"] = "staged" if staged_record.get("staged_step_path") else "missing_step"
    staged_record["staged_at"] = now_iso()
    return staged_record, stats


def _update_source_batch_state(batch_dir: Path, result_state: dict[str, Any]) -> None:
    state_path = batch_dir / "state.json"
    current = read_json(state_path) or {}
    if not isinstance(current, dict):
        current = {}
    current.update(
        {
            "status": "staged",
            "staged_at": result_state.get("ended_at"),
            "staging_dir": result_state.get("staging_dir"),
            "staged_manifest_path": result_state.get("manifest_path"),
            "staged_checksums_path": result_state.get("checksums_path"),
            "staged_marker_path": result_state.get("marker_path"),
            "n_staged_samples": result_state.get("n_samples"),
            "n_staged_step_files": result_state.get("n_step_files"),
            "n_staged_ofs_files": result_state.get("n_ofs_files"),
            "n_staging_missing_files": result_state.get("n_missing_files"),
        }
    )
    write_json(state_path, current)


def run_stage_batch_inputs(
    layout: WorkRootLayout,
    *,
    batch_id: str,
    staging_root: Path,
    min_free_bytes: int = 0,
    copy_mode: str = "copy",
    overwrite: bool = False,
    dry_run: bool = False,
    logger: logging.Logger,
) -> StageBatchInputsResult:
    if copy_mode not in {"copy", "symlink"}:
        raise ValueError("copy_mode must be 'copy' or 'symlink'")

    import time

    started_t = time.time()
    res = StageBatchInputsResult(batch_id=batch_id, started_at=now_iso())
    batch_dir = layout.batches_dir / batch_id
    batch_manifest = batch_dir / "manifest.jsonl"
    if not batch_manifest.exists():
        raise FileNotFoundError(f"batch manifest not found: {batch_manifest}")

    staging_dir = staging_root.expanduser().resolve() / batch_id
    manifest_path = staging_dir / "manifest.jsonl"
    checksums_path = staging_dir / "checksums.jsonl"
    state_path = staging_dir / "state.json"
    marker_path = staging_dir / DONE_MARKER
    res.staging_dir = str(staging_dir)
    res.manifest_path = str(manifest_path)
    res.checksums_path = str(checksums_path)
    res.state_path = str(state_path)
    res.marker_path = str(marker_path)

    try:
        records = _load_batch_records(batch_manifest)
        required_bytes = _estimate_required_bytes(records)
        space_info = _check_free_space(
            staging_root,
            required_bytes=required_bytes,
            min_free_bytes=min_free_bytes,
            logger=logger,
        )
        res.extra["space_check"] = space_info
        res.extra["source_batch_manifest"] = str(batch_manifest)
        res.n_samples = len(records)

        if dry_run:
            res.status = "dry_run"
            logger.info("dry-run staging batch %s: samples=%d required_bytes=%d", batch_id, len(records), required_bytes)
            return res

        staging_dir.mkdir(parents=True, exist_ok=True)
        _reset_output_files(staging_dir)
        _write_empty(manifest_path)
        _write_empty(checksums_path)

        for idx, record in enumerate(records):
            staged_record, stats = _copy_record_inputs(
                record,
                index=idx,
                staging_dir=staging_dir,
                copy_mode=copy_mode,
                overwrite=overwrite,
                checksums_path=checksums_path,
            )
            append_jsonl(manifest_path, staged_record)
            res.n_step_files += stats["step"]
            res.n_ofs_files += stats["ofs"]
            res.n_missing_files += stats["missing"]
            res.copied_bytes += stats["bytes"]

        if res.n_step_files == 0:
            raise RuntimeError(f"no STEP files staged for batch {batch_id}")

        res.status = "staged"
        res.ended_at = now_iso()
        res.elapsed_s = round(time.time() - started_t, 2)
        state = res.to_dict()
        write_json(state_path, state)
        marker_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        _update_source_batch_state(batch_dir, state)
        logger.info(
            "staged batch %s: samples=%d step=%d ofs=%d missing=%d bytes=%d dir=%s",
            batch_id,
            res.n_samples,
            res.n_step_files,
            res.n_ofs_files,
            res.n_missing_files,
            res.copied_bytes,
            staging_dir,
        )
        return res
    except BaseException as exc:  # noqa: BLE001
        res.status = "failed"
        res.error = f"{type(exc).__name__}: {exc}"
        logger.error("stage batch inputs failed: %s\n%s", exc, traceback.format_exc())
        raise
    finally:
        if not res.ended_at:
            res.ended_at = now_iso()
            res.elapsed_s = round(time.time() - started_t, 2)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage one ABC batch manifest into a virtual-environment accessible input directory."
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--min-free-bytes", type=int, default=0)
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="额外保留的空闲空间，按 GiB 计算；会覆盖 --min-free-bytes",
    )
    parser.add_argument("--copy-mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    layout = work_root_layout(args.work_root)
    min_free_bytes = args.min_free_bytes
    if args.min_free_gb is not None:
        min_free_bytes = int(args.min_free_gb * 1024**3)

    log_path = layout.work_root / "stage_batch_inputs.log"
    with stage_logger(LOGGER_NAME, log_path) as logger:
        try:
            result = run_stage_batch_inputs(
                layout,
                batch_id=args.batch_id,
                staging_root=args.staging_root,
                min_free_bytes=min_free_bytes,
                copy_mode=args.copy_mode,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                logger=logger,
            )
            return 0 if result.status in {"staged", "dry_run"} else 1
        except KeyboardInterrupt:
            logger.warning("interrupted by user")
            return 130
        except BaseException:  # noqa: BLE001
            return 1


if __name__ == "__main__":
    sys.exit(main())
