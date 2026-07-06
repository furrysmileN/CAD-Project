"""ABCdataset 压缩包发现与解压。

用法（在 GPU/数据机器上执行；本机仅做静态检查）::

    python -m cad_data_gen.abc_batch.extract_archives \
        --abc-root /apdcephfs_zwfy/share_303204533/jackjliu/cad/data/ABCdataset \
        --work-root /path/to/work_root \
        --max-extract-workers 8

约定：

* `<abc-root>/step/` 与 `<abc-root>/ofs/` 是固定的两个子目录，每个内含若干压缩包。
* 每个压缩包独立解压到 `<work-root>/extracted/{step,ofs}/<archive_stem>/`。
* 单个压缩包成功后写 `<archive_stem>.done`（JSON 标记），失败则清理半成品并写 `extract_failures.jsonl`。
* `.zip` / `.tar` / `.tar.gz` / `.tar.bz2` / `.tgz` / `.tbz2` 走 stdlib，
  `.7z` 走 `py7zr`（按需 try-import）或 `7z`/`7zz` 二进制（`shutil.which` 探测）。
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .logging_utils import (
    append_jsonl,
    now_iso,
    stage_logger,
    truncate_str,
    write_json,
)
from .paths import (
    SUPPORTED_ARCHIVE_SUFFIXES,
    WorkRootLayout,
    archive_stem,
    work_root_layout,
)

LOGGER_NAME = "extract_archives"


# ---------- 数据结构 ----------


@dataclass
class ArchiveTask:
    """单个压缩包的解压任务。"""

    side: str  # "step" or "ofs"
    archive_path: Path
    target_dir: Path  # <work-root>/extracted/{side}/<archive_stem>
    done_marker: Path  # <work-root>/extracted/{side}/<archive_stem>.done


@dataclass
class ArchiveResult:
    side: str
    archive: str
    archive_stem: str
    status: str  # "done" / "skipped" / "failed"
    n_files: int = 0
    bytes: int = 0
    elapsed_s: float = 0.0
    error_type: Optional[str] = None
    error_msg: Optional[str] = None


# ---------- 单个压缩包解压 ----------


def _is_safe_member(name: str) -> bool:
    """阻止 zip-slip / tar-slip：禁止绝对路径与 `..` 上跳。"""
    if not name:
        return False
    if name.startswith("/") or name.startswith("\\"):
        return False
    parts = name.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        return False
    return True


def _extract_zip(archive: Path, target: Path) -> tuple[int, int]:
    n_files = 0
    total_bytes = 0
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if not _is_safe_member(info.filename):
                continue
            zf.extract(info, path=target)
            n_files += 1
            total_bytes += info.file_size
    return n_files, total_bytes


def _extract_tar(archive: Path, target: Path) -> tuple[int, int]:
    n_files = 0
    total_bytes = 0
    # 自动检测压缩模式：tarfile.open 默认 mode="r:*"
    with tarfile.open(archive, "r:*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            if not _is_safe_member(member.name):
                continue
            tf.extract(member, path=target)
            n_files += 1
            total_bytes += member.size
    return n_files, total_bytes


def _extract_7z(archive: Path, target: Path) -> tuple[int, int]:
    """优先使用系统 `7z`/`7zz` 命令行（兼容性更好），失败/缺失时回退到 py7zr。

    历史教训：py7zr 1.x 对部分 .7z 头格式（含较新的 zstd/bcj2 编码或较旧的
    headers）解析不完整，会抛 `Bad7zFile: invalid header data`。系统二进制
    7z/7zz 由 7-Zip 官方维护，对所有编码全支持，应作为首选。
    """
    target.mkdir(parents=True, exist_ok=True)
    binary = shutil.which("7z") or shutil.which("7zz") or shutil.which("7za")
    last_err: Optional[str] = None

    if binary is not None:
        cmd = [binary, "x", "-y", f"-o{str(target)}", str(archive)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return _scan_dir(target)
        last_err = (
            f"`{binary} x` exited with {proc.returncode}: "
            f"{truncate_str(proc.stderr or proc.stdout, 400)}"
        )

    # 回退到 py7zr（纯 Python，慢且兼容性较弱，但作为兜底）
    try:
        import py7zr  # type: ignore
    except ModuleNotFoundError:
        py7zr = None  # type: ignore

    if py7zr is not None:
        # 清理可能由失败的 7z 留下的半成品
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(archive, mode="r") as sz:  # type: ignore[attr-defined]
            sz.extractall(path=target)
        return _scan_dir(target)

    if last_err is not None:
        raise RuntimeError(f"system 7z failed and py7zr unavailable: {last_err}")
    raise RuntimeError(
        "Neither `py7zr` nor `7z`/`7zz` is available; install one to extract .7z archives"
    )


def _scan_dir(target: Path) -> tuple[int, int]:
    n = 0
    total = 0
    for root, _dirs, files in os.walk(target):
        for fname in files:
            try:
                st = os.stat(os.path.join(root, fname))
            except OSError:
                continue
            n += 1
            total += st.st_size
    return n, total


def _detect_archive_kind(archive: Path) -> str:
    lower = archive.name.lower()
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith(".7z"):
        return "7z"
    for suf in (".tar.gz", ".tar.bz2", ".tgz", ".tbz2", ".tar"):
        if lower.endswith(suf):
            return "tar"
    return "unknown"


def _extract_one_worker(
    side: str,
    archive_path_str: str,
    target_dir_str: str,
    done_marker_str: str,
) -> dict:
    """子进程入口：解压单个压缩包，返回结果字典（dataclass 不能跨进程序列化时更稳）。"""
    archive_path = Path(archive_path_str)
    target_dir = Path(target_dir_str)
    done_marker = Path(done_marker_str)
    stem = archive_stem(archive_path)
    started = time.time()

    kind = _detect_archive_kind(archive_path)
    if kind == "unknown":
        return {
            "side": side,
            "archive": archive_path.name,
            "archive_stem": stem,
            "status": "failed",
            "error_type": "UnsupportedArchive",
            "error_msg": f"unrecognized archive suffix: {archive_path.name}",
            "elapsed_s": 0.0,
            "n_files": 0,
            "bytes": 0,
        }

    # 清理可能的半成品目录（同名 .done 不存在但目录存在 → 上次失败）
    if target_dir.exists():
        try:
            shutil.rmtree(target_dir)
        except OSError as exc:
            return {
                "side": side,
                "archive": archive_path.name,
                "archive_stem": stem,
                "status": "failed",
                "error_type": "CleanupError",
                "error_msg": f"failed to remove stale target {target_dir}: {exc}",
                "elapsed_s": 0.0,
                "n_files": 0,
                "bytes": 0,
            }

    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        if kind == "zip":
            n_files, total_bytes = _extract_zip(archive_path, target_dir)
        elif kind == "tar":
            n_files, total_bytes = _extract_tar(archive_path, target_dir)
        elif kind == "7z":
            n_files, total_bytes = _extract_7z(archive_path, target_dir)
        else:
            raise RuntimeError(f"unhandled kind: {kind}")
    except KeyboardInterrupt:
        # 半成品清理后再抛
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        raise
    except BaseException as exc:  # noqa: BLE001
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        return {
            "side": side,
            "archive": archive_path.name,
            "archive_stem": stem,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_msg": truncate_str(f"{exc}\n{traceback.format_exc()}", 1200),
            "elapsed_s": time.time() - started,
            "n_files": 0,
            "bytes": 0,
        }

    elapsed = time.time() - started
    marker = {
        "archive": archive_path.name,
        "archive_path": str(archive_path),
        "archive_stem": stem,
        "side": side,
        "n_files": n_files,
        "bytes": total_bytes,
        "elapsed_s": round(elapsed, 3),
        "mtime": now_iso(),
    }
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    with open(done_marker, "w", encoding="utf-8") as f:
        import json

        json.dump(marker, f, ensure_ascii=False, indent=2)

    return {
        "side": side,
        "archive": archive_path.name,
        "archive_stem": stem,
        "status": "done",
        "n_files": n_files,
        "bytes": total_bytes,
        "elapsed_s": elapsed,
    }


# ---------- 任务发现 ----------


def discover_archives(side_root: Path) -> list[Path]:
    """列出 `<abc-root>/{step,ofs}/` 下所有受支持后缀的压缩包，按文件名排序。"""
    if not side_root.exists():
        return []
    archives: list[Path] = []
    for entry in sorted(side_root.iterdir()):
        if not entry.is_file():
            continue
        lower = entry.name.lower()
        if any(lower.endswith(suf) for suf in SUPPORTED_ARCHIVE_SUFFIXES):
            archives.append(entry)
    return archives


def build_tasks(layout: WorkRootLayout, abc_root: Path, sides: Iterable[str]) -> list[ArchiveTask]:
    tasks: list[ArchiveTask] = []
    for side in sides:
        side_src = abc_root / side
        side_dst = layout.extracted_step_root if side == "step" else layout.extracted_ofs_root
        for arch in discover_archives(side_src):
            stem = archive_stem(arch)
            tasks.append(
                ArchiveTask(
                    side=side,
                    archive_path=arch,
                    target_dir=side_dst / stem,
                    done_marker=side_dst / f"{stem}.done",
                )
            )
    return tasks


# ---------- 主流程 ----------


def run_extract(
    abc_root: Path,
    work_root: Path,
    sides: Iterable[str],
    max_workers: int,
    logger: logging.Logger,
) -> dict:
    layout = work_root_layout(work_root)
    layout.ensure_dirs()

    sides_list = list(sides)
    tasks = build_tasks(layout, abc_root, sides_list)
    logger.info(
        "discovered %d archives across sides=%s under %s",
        len(tasks),
        sides_list,
        abc_root,
    )

    # 跳过已完成
    pending: list[ArchiveTask] = []
    skipped = 0
    for t in tasks:
        if t.done_marker.exists():
            skipped += 1
            continue
        pending.append(t)

    logger.info("pending=%d skipped=%d (already .done)", len(pending), skipped)

    summary_per_side: dict[str, dict] = {
        "step": {"n_archives_total": 0, "n_archives_done": 0, "n_files_total": 0, "total_bytes": 0},
        "ofs": {"n_archives_total": 0, "n_archives_done": 0, "n_files_total": 0, "total_bytes": 0},
    }
    for t in tasks:
        summary_per_side[t.side]["n_archives_total"] += 1
        if t.done_marker.exists():
            summary_per_side[t.side]["n_archives_done"] += 1
            try:
                import json

                with open(t.done_marker, "r", encoding="utf-8") as f:
                    marker = json.load(f)
                summary_per_side[t.side]["n_files_total"] += int(marker.get("n_files", 0))
                summary_per_side[t.side]["total_bytes"] += int(marker.get("bytes", 0))
            except (OSError, ValueError):
                pass

    failures = 0
    if pending:
        max_workers = max(1, min(max_workers, len(pending)))
        logger.info("starting ProcessPool with %d workers", max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _extract_one_worker,
                    t.side,
                    str(t.archive_path),
                    str(t.target_dir),
                    str(t.done_marker),
                ): t
                for t in pending
            }
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    res = fut.result()
                except BaseException as exc:  # noqa: BLE001
                    logger.error("worker crashed for %s: %s", t.archive_path, exc)
                    res = {
                        "side": t.side,
                        "archive": t.archive_path.name,
                        "archive_stem": archive_stem(t.archive_path),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_msg": truncate_str(str(exc), 400),
                        "elapsed_s": 0.0,
                        "n_files": 0,
                        "bytes": 0,
                    }

                if res["status"] == "done":
                    logger.info(
                        "[%s] OK %s: %d files, %.1f MB, %.1fs",
                        res["side"],
                        res["archive"],
                        res["n_files"],
                        res["bytes"] / 1024 / 1024,
                        res["elapsed_s"],
                    )
                    summary_per_side[res["side"]]["n_archives_done"] += 1
                    summary_per_side[res["side"]]["n_files_total"] += res["n_files"]
                    summary_per_side[res["side"]]["total_bytes"] += res["bytes"]
                else:
                    failures += 1
                    logger.error(
                        "[%s] FAIL %s: %s (%s)",
                        res["side"],
                        res["archive"],
                        res.get("error_type"),
                        truncate_str(res.get("error_msg") or "", 200),
                    )
                    append_jsonl(
                        layout.extract_failures,
                        {
                            "ts": now_iso(),
                            "side": res["side"],
                            "archive": res["archive"],
                            "archive_path": str(t.archive_path),
                            "error_type": res.get("error_type"),
                            "error_msg": res.get("error_msg"),
                        },
                    )

    summary = {
        "ts": now_iso(),
        "abc_root": str(abc_root),
        "work_root": str(layout.work_root),
        "sides_processed": sides_list,
        "n_failures_this_run": failures,
        "step": summary_per_side["step"],
        "ofs": summary_per_side["ofs"],
    }
    write_json(layout.extract_summary, summary)
    logger.info("extract_summary written to %s", layout.extract_summary)
    return summary


# ---------- CLI ----------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and extract ABCdataset archives (step/ + ofs/) to a work-root layout."
    )
    parser.add_argument("--abc-root", type=Path, required=True, help="ABCdataset root containing step/ and ofs/")
    parser.add_argument("--work-root", type=Path, required=True, help="Output work-root for extraction")
    parser.add_argument(
        "--max-extract-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel extraction processes (one archive per worker)",
    )
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction; verify existing layout only")
    parser.add_argument("--extract-step-only", action="store_true", help="Only process <abc-root>/step/")
    parser.add_argument("--extract-ofs-only", action="store_true", help="Only process <abc-root>/ofs/")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    layout = work_root_layout(args.work_root)
    layout.ensure_dirs()

    log_path = layout.work_root / "stage_extract.log"
    with stage_logger(LOGGER_NAME, log_path) as logger:
        if args.skip_extract:
            logger.info("--skip-extract set; nothing to do")
            return 0

        if args.extract_step_only and args.extract_ofs_only:
            logger.error("--extract-step-only and --extract-ofs-only are mutually exclusive")
            return 2

        if args.extract_step_only:
            sides = ["step"]
        elif args.extract_ofs_only:
            sides = ["ofs"]
        else:
            sides = ["step", "ofs"]

        if not args.abc_root.exists():
            logger.error("abc-root does not exist: %s", args.abc_root)
            return 2

        try:
            summary = run_extract(
                abc_root=args.abc_root,
                work_root=args.work_root,
                sides=sides,
                max_workers=args.max_extract_workers,
                logger=logger,
            )
        except KeyboardInterrupt:
            logger.warning("interrupted by user")
            return 130
        except BaseException as exc:  # noqa: BLE001
            logger.error("extract pipeline crashed: %s\n%s", exc, traceback.format_exc())
            return 1

        if summary["n_failures_this_run"] > 0:
            logger.warning(
                "extraction finished with %d failures (see %s)",
                summary["n_failures_this_run"],
                layout.extract_failures,
            )
        return 0


if __name__ == "__main__":
    sys.exit(main())
