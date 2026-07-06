"""真实环境侧批次归档与路径重写工具。

将虚拟环境中单批次生成结果归档到最终数据盘的批次目录，并写出：

- `archive_manifest.jsonl`：归档文件大小与 sha256
- `ARCHIVED`：归档完成标记
- `archive_state.json`：归档状态摘要

该阶段只负责“虚拟环境输出 -> 最终数据盘”的搬运与校验，不重新生成资产。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .logging_utils import append_jsonl, iter_jsonl, now_iso, read_json, stage_logger, write_json
from .paths import WorkRootLayout, work_root_layout

LOGGER_NAME = "archive_batch"
ARCHIVED_MARKER = "ARCHIVED"


@dataclass
class ArchiveBatchResult:
    stage: str = "archive_batch"
    status: str = "pending"
    batch_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    elapsed_s: float = 0.0
    archive_dir: str = ""
    archive_manifest_path: str = ""
    state_path: str = ""
    marker_path: str = ""
    n_files: int = 0
    total_bytes: int = 0
    n_rewritten_files: int = 0
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.copy2(src, dst)


def _copy_tree_contents(src_root: Path, dst_root: Path, *, overwrite: bool) -> int:
    if not src_root.exists():
        return 0
    copied = 0
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        _copy_file(src, dst_root / rel, overwrite=overwrite)
        copied += 1
    return copied


def _rewrite_string(value: str, rewrite_map: dict[str, str]) -> str:
    out = value
    for src, dst in sorted(rewrite_map.items(), key=lambda item: len(item[0]), reverse=True):
        if src and out.startswith(src):
            return dst + out[len(src) :]
    return out


def _rewrite_obj(obj: Any, rewrite_map: dict[str, str]) -> tuple[Any, bool]:
    if isinstance(obj, str):
        rewritten = _rewrite_string(obj, rewrite_map)
        return rewritten, rewritten != obj
    if isinstance(obj, list):
        changed = False
        values = []
        for item in obj:
            new_item, item_changed = _rewrite_obj(item, rewrite_map)
            changed = changed or item_changed
            values.append(new_item)
        return values, changed
    if isinstance(obj, dict):
        changed = False
        values: dict[str, Any] = {}
        for key, item in obj.items():
            new_item, item_changed = _rewrite_obj(item, rewrite_map)
            changed = changed or item_changed
            values[key] = new_item
        return values, changed
    return obj, False


def _rewrite_json_file(path: Path, rewrite_map: dict[str, str]) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    rewritten, changed = _rewrite_obj(payload, rewrite_map)
    if changed:
        path.write_text(json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def _rewrite_jsonl_file(path: Path, rewrite_map: dict[str, str]) -> bool:
    changed_any = False
    rows: list[Any] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rows.append(line)
            continue
        rewritten, changed = _rewrite_obj(row, rewrite_map)
        changed_any = changed_any or changed
        rows.append(rewritten)
    if changed_any:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                if isinstance(row, str):
                    f.write(row + "\n")
                else:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return changed_any


def _rewrite_paths_under(root: Path, rewrite_map: dict[str, str]) -> int:
    if not rewrite_map:
        return 0
    n = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            n += int(_rewrite_json_file(path, rewrite_map))
        elif path.suffix == ".jsonl":
            n += int(_rewrite_jsonl_file(path, rewrite_map))
    return n


def _write_archive_manifest(archive_dir: Path, manifest_path: Path) -> tuple[int, int]:
    if manifest_path.exists():
        manifest_path.unlink()
    manifest_path.touch()
    n_files = 0
    total_bytes = 0
    for path in sorted(archive_dir.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rel = path.relative_to(archive_dir)
        size = path.stat().st_size
        append_jsonl(
            manifest_path,
            {
                "relative_path": str(rel),
                "path": str(path),
                "size_bytes": size,
                "sha256": _sha256_file(path),
                "status": "ok",
            },
        )
        n_files += 1
        total_bytes += size
    return n_files, total_bytes


def _verify_archive_manifest(archive_dir: Path, manifest_path: Path) -> dict[str, Any]:
    n_checked = 0
    n_missing = 0
    n_mismatch = 0
    for row in iter_jsonl(manifest_path):
        path = archive_dir / str(row.get("relative_path"))
        if not path.is_file():
            n_missing += 1
            continue
        n_checked += 1
        if path.stat().st_size != int(row.get("size_bytes") or -1):
            n_mismatch += 1
            continue
        if _sha256_file(path) != row.get("sha256"):
            n_mismatch += 1
    return {
        "n_checked": n_checked,
        "n_missing": n_missing,
        "n_mismatch": n_mismatch,
        "ok": n_missing == 0 and n_mismatch == 0,
    }


def _update_source_batch_state(layout: WorkRootLayout, batch_id: str, result: dict[str, Any]) -> None:
    state_path = layout.batches_dir / batch_id / "state.json"
    current = read_json(state_path) or {}
    if not isinstance(current, dict):
        current = {}
    current.update(
        {
            "status": "archived",
            "archived_at": result.get("ended_at"),
            "archive_dir": result.get("archive_dir"),
            "archive_manifest_path": result.get("archive_manifest_path"),
            "archive_marker_path": result.get("marker_path"),
        }
    )
    write_json(state_path, current)


def run_archive_batch(
    layout: WorkRootLayout,
    *,
    batch_id: str,
    archive_root: Path,
    rewrite_map: Optional[dict[str, str]] = None,
    overwrite: bool = False,
    logger: logging.Logger,
) -> ArchiveBatchResult:
    started_t = time.time()
    res = ArchiveBatchResult(batch_id=batch_id, started_at=now_iso())
    archive_dir = archive_root.expanduser().resolve() / "batches" / batch_id
    archive_manifest = archive_dir / "archive_manifest.jsonl"
    state_path = archive_dir / "archive_state.json"
    marker_path = archive_dir / ARCHIVED_MARKER
    res.archive_dir = str(archive_dir)
    res.archive_manifest_path = str(archive_manifest)
    res.state_path = str(state_path)
    res.marker_path = str(marker_path)

    try:
        if archive_dir.exists() and overwrite:
            shutil.rmtree(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)

        copied_sections: dict[str, int] = {}
        section_sources = {
            "batch": layout.batches_dir / batch_id,
            "assets": layout.assets_dir,
            "occlusion": layout.occlusion_dir,
            "contexts": layout.contexts_dir,
            "descriptions": layout.descriptions_dir,
        }
        for name, src in section_sources.items():
            copied_sections[name] = _copy_tree_contents(src, archive_dir / name, overwrite=overwrite)

        for log_path in (
            layout.stage_assets_log,
            layout.stage_occlusion_log,
            layout.stage_describe_log,
            layout.work_root / "stage_batch_inputs.log",
            layout.work_root / "run_abc_batch.log",
        ):
            if log_path.is_file():
                _copy_file(log_path, archive_dir / "logs" / log_path.name, overwrite=overwrite)

        final_rewrite_map = dict(rewrite_map or {})
        final_rewrite_map.setdefault(str(layout.work_root), str(archive_dir))
        res.n_rewritten_files = _rewrite_paths_under(archive_dir, final_rewrite_map)
        res.n_files, res.total_bytes = _write_archive_manifest(archive_dir, archive_manifest)
        verify = _verify_archive_manifest(archive_dir, archive_manifest)
        res.extra["copied_sections"] = copied_sections
        res.extra["verification"] = verify
        if not verify["ok"]:
            raise RuntimeError(f"archive verification failed: {verify}")

        res.status = "archived"
        res.ended_at = now_iso()
        res.elapsed_s = round(time.time() - started_t, 2)
        state = res.to_dict()
        write_json(state_path, state)
        marker_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _update_source_batch_state(layout, batch_id, state)
        logger.info("archived batch %s to %s files=%d bytes=%d", batch_id, archive_dir, res.n_files, res.total_bytes)
        return res
    except BaseException as exc:  # noqa: BLE001
        res.status = "failed"
        res.error = f"{type(exc).__name__}: {exc}"
        logger.error("archive batch failed: %s\n%s", exc, traceback.format_exc())
        raise
    finally:
        if not res.ended_at:
            res.ended_at = now_iso()
            res.elapsed_s = round(time.time() - started_t, 2)


def _parse_rewrite(values: Optional[list[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--rewrite must be SOURCE=DEST")
        src, dst = value.split("=", 1)
        mapping[str(Path(src).expanduser().resolve())] = str(Path(dst).expanduser().resolve())
    return mapping


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive one generated ABC batch into the final data root.")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--rewrite", action="append", default=None, help="路径重写规则 SOURCE=DEST，可重复")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    layout = work_root_layout(args.work_root)
    log_path = layout.work_root / "archive_batch.log"
    with stage_logger(LOGGER_NAME, log_path) as logger:
        try:
            result = run_archive_batch(
                layout,
                batch_id=args.batch_id,
                archive_root=args.archive_root,
                rewrite_map=_parse_rewrite(args.rewrite),
                overwrite=args.overwrite,
                logger=logger,
            )
            return 0 if result.status == "archived" else 1
        except KeyboardInterrupt:
            logger.warning("interrupted by user")
            return 130
        except BaseException:  # noqa: BLE001
            return 1


if __name__ == "__main__":
    sys.exit(main())
