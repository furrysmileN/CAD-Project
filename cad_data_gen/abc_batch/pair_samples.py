"""STEP ↔ OFS 一一配对。

输入：`<work-root>/extracted/step/<archive_stem>/...` 与
`<work-root>/extracted/ofs/<archive_stem>/...`（由 `extract_archives.py` 产出）。

输出：

* `<work-root>/pairing_manifest.jsonl`：主样本清单，每行一条记录
* `<work-root>/orphan_ofs.jsonl`：仅有 OFS 而 STEP 缺失的样本
* `<work-root>/pairing_summary.json`：全量统计

`sample_id` 计算方式（与既有 `cad_data_gen.step_assets.sample_id_from_relative_path`
保持一致）：取相对于 `<work-root>/extracted/{step,ofs}/` 的相对路径
（不去 `<archive_stem>` 那一层），再用现有规则做后缀剥离 + 安全字符替换。
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional

from ..step_assets import iter_step_files, sample_id_from_relative_path
from .logging_utils import (
    append_jsonl,
    now_iso,
    stage_logger,
    write_json,
)
from .paths import WorkRootLayout, work_root_layout

LOGGER_NAME = "pair_samples"
OFS_SUFFIXES: tuple[str, ...] = (".ofs", ".json", ".pkl", ".yaml", ".yml")


# ---------- sample_id ----------


def _sample_id_for(root: Path, abs_path: Path) -> str:
    """统一 sample_id 计算：相对路径 → safe-string。

    `step_assets.sample_id_from_relative_path` 接收的是相对路径字符串，
    这里把绝对路径换算成相对路径后透传给它。
    """
    rel = abs_path.resolve().relative_to(root.resolve())
    return sample_id_from_relative_path(str(rel))


def _pair_key_for_sample_id(sample_id: str) -> str:
    """生成 STEP/OFS 共享的配对键。

    ABC 数据中同一条样本的 STEP 与 OFS 文件名通常分别形如
    `..._step_003.step` 与 `..._featurescript_003.yml`。资产阶段仍然需要
    使用 STEP 的原始 sample_id；这里只在配对阶段把这两个后缀规范化，避免
    `--require-ofs` 时误判为完全不同的样本。
    """
    for marker in ("_step_", "_featurescript_"):
        if marker in sample_id:
            prefix, suffix = sample_id.rsplit(marker, 1)
            if suffix:
                return f"{prefix}_{suffix}"
    return sample_id


# ---------- 索引收集 ----------


def _iter_ofs_files(ofs_root: Path) -> Iterator[Path]:
    if not ofs_root.exists():
        return
    for path in ofs_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in OFS_SUFFIXES:
            continue
        if path.stat().st_size == 0:
            continue
        yield path


def _archive_stem_of(root: Path, abs_path: Path) -> Optional[str]:
    """提取相对路径的第一段，即压缩包 stem 子目录名。"""
    try:
        rel = abs_path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    return parts[0]


def collect_step_index(step_root: Path) -> dict[str, dict]:
    """{pair_key: {sample_id, pair_key, step_path, source_step_archive}}"""
    index: dict[str, dict] = {}
    if not step_root.exists():
        return index
    for path in iter_step_files(step_root, recursive=True):
        sid = _sample_id_for(step_root, path)
        pair_key = _pair_key_for_sample_id(sid)
        index[pair_key] = {
            "sample_id": sid,
            "pair_key": pair_key,
            "step_path": str(path.resolve()),
            "source_step_archive": _archive_stem_of(step_root, path),
        }
    return index


def collect_ofs_index(ofs_root: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for path in _iter_ofs_files(ofs_root):
        sid = _sample_id_for(ofs_root, path)
        pair_key = _pair_key_for_sample_id(sid)
        index[pair_key] = {
            "sample_id": sid,
            "pair_key": pair_key,
            "ofs_path": str(path.resolve()),
            "source_ofs_archive": _archive_stem_of(ofs_root, path),
        }
    return index


# ---------- 切分 ----------


def _stable_hash_mod(sample_id: str, modulo: int) -> int:
    h = hashlib.md5(sample_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % modulo


def _select(
    sample_ids: list[str],
    *,
    offset: int,
    limit: Optional[int],
    shard_index: Optional[int],
    shard_total: Optional[int],
) -> list[str]:
    """先按 offset/limit 做窗口切分；再按 shard_index/shard_total 做哈希分片。"""
    selected = sample_ids
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[: limit]
    if shard_total is not None and shard_total > 1:
        if shard_index is None or not (0 <= shard_index < shard_total):
            raise ValueError(
                f"--shard-index must be in [0, {shard_total}) when --shard-total={shard_total}"
            )
        selected = [sid for sid in selected if _stable_hash_mod(sid, shard_total) == shard_index]
    return selected


# ---------- 核心 ----------


def run_pair(
    layout: WorkRootLayout,
    *,
    offset: int,
    limit: Optional[int],
    shard_index: Optional[int],
    shard_total: Optional[int],
    require_ofs: bool,
    logger: logging.Logger,
    scan_ofs: bool = True,
) -> dict:
    step_root = layout.extracted_step_root
    ofs_root = layout.extracted_ofs_root

    logger.info("scanning STEP under %s ...", step_root)
    step_index = collect_step_index(step_root)
    logger.info("found %d unique STEP sample_ids", len(step_index))

    if scan_ofs:
        logger.info("scanning OFS under %s ...", ofs_root)
        ofs_index = collect_ofs_index(ofs_root)
        logger.info("found %d unique OFS sample_ids", len(ofs_index))
    else:
        ofs_index = {}
        logger.info("skipping OFS scan for STEP-first asset generation; OFS can be loaded later in describe stage")

    # 重置输出文件（覆盖式：每次 pair 重新生成 manifest）
    for p in (layout.pairing_manifest, layout.step_only_manifest, layout.orphan_ofs):
        if p.exists():
            p.unlink()
        p.parent.mkdir(parents=True, exist_ok=True)
        # 即使本 shard 没有候选样本也保留空文件，便于下游区分“0条记录”和“写文件失败”。
        p.touch()

    # 全量统计
    paired_ids = sorted(set(step_index) & set(ofs_index))
    step_only_ids = sorted(set(step_index) - set(ofs_index))
    ofs_only_ids = sorted(set(ofs_index) - set(step_index))

    logger.info(
        "totals: paired=%d step_only=%d ofs_only=%d",
        len(paired_ids),
        len(step_only_ids),
        len(ofs_only_ids),
    )

    # 决定要写入主 manifest 的候选集（受 require_ofs 影响）
    candidate_ids = list(paired_ids)
    if not require_ofs:
        candidate_ids.extend(step_only_ids)
        candidate_ids.sort()

    # 应用切分
    candidate_ids = _select(
        candidate_ids,
        offset=offset,
        limit=limit,
        shard_index=shard_index,
        shard_total=shard_total,
    )
    logger.info(
        "after offset=%d limit=%s shard=%s/%s: writing %d records to manifest",
        offset,
        limit,
        shard_index,
        shard_total,
        len(candidate_ids),
    )

    selected_step_only_ids = _select(
        step_only_ids,
        offset=offset,
        limit=limit,
        shard_index=shard_index,
        shard_total=shard_total,
    )
    logger.info(
        "after offset=%d limit=%s shard=%s/%s: writing %d STEP-only diagnostic records",
        offset,
        limit,
        shard_index,
        shard_total,
        len(selected_step_only_ids),
    )

    n_paired_written = 0
    n_step_only_written = 0
    for sid in candidate_ids:
        step_meta = step_index.get(sid)
        ofs_meta = ofs_index.get(sid)
        if step_meta is None:
            # 不应发生：候选 id 来自 step_index 或 paired
            continue
        record = {
            "sample_id": step_meta["sample_id"],
            "pair_key": sid,
            "step_path": step_meta["step_path"],
            "source_step_archive": step_meta["source_step_archive"],
            "ofs_sample_id": ofs_meta["sample_id"] if ofs_meta else None,
            "ofs_path": ofs_meta["ofs_path"] if ofs_meta else None,
            "source_ofs_archive": ofs_meta["source_ofs_archive"] if ofs_meta else None,
            "has_ofs": ofs_meta is not None,
            "ofs_status": "matched" if ofs_meta else "unmatched",
            "status": "paired" if ofs_meta else "step_only",
        }
        append_jsonl(layout.pairing_manifest, record)
        if ofs_meta:
            n_paired_written += 1
        else:
            n_step_only_written += 1

    for sid in selected_step_only_ids:
        step_meta = step_index.get(sid)
        if step_meta is None:
            continue
        append_jsonl(
            layout.step_only_manifest,
            {
                "sample_id": step_meta["sample_id"],
                "pair_key": sid,
                "step_path": step_meta["step_path"],
                "source_step_archive": step_meta["source_step_archive"],
                "ofs_sample_id": None,
                "ofs_path": None,
                "source_ofs_archive": None,
                "has_ofs": False,
                "ofs_status": "unmatched",
                "status": "step_only",
            },
        )

    if require_ofs and len(step_index) > 0 and len(paired_ids) == 0:
        logger.warning(
            "require_ofs=true and n_paired=0 while n_step=%d; STEP assets will be blocked. "
            "Disable --require-ofs / require_ofs_for_assets or fix OFS pairing rules to continue asset generation.",
            len(step_index),
        )
    elif not require_ofs and len(step_index) > 0 and len(paired_ids) == 0:
        logger.warning(
            "n_paired=0 while n_step=%d; continuing with STEP-only asset generation because require_ofs=false.",
            len(step_index),
        )

    # orphan_ofs：始终写全量（不参与切分）
    for sid in ofs_only_ids:
        ofs_meta = ofs_index[sid]
        append_jsonl(
            layout.orphan_ofs,
            {
                "sample_id": ofs_meta["sample_id"],
                "pair_key": sid,
                "ofs_path": ofs_meta["ofs_path"],
                "source_ofs_archive": ofs_meta["source_ofs_archive"],
            },
        )

    # 重复检测（按 archive_stem + relpath 之外可能的全局碰撞）
    sid_counter: Counter[str] = Counter()
    for sid in step_index:
        sid_counter[sid] += 1
    for sid in ofs_index:
        sid_counter[sid] += 1
    # sid_counter[sid] 期望最多为 2（step + ofs 各 1）；>2 则路径碰撞
    dup_ids = sorted(
        ((sid, c) for sid, c in sid_counter.items() if c > 2),
        key=lambda x: (-x[1], x[0]),
    )
    top_dup = [{"sample_id": sid, "count": c} for sid, c in dup_ids[:50]]
    if top_dup:
        logger.warning(
            "detected %d sample_ids with collision count > 2 (path collision across archives); top: %s",
            len(dup_ids),
            top_dup[:3],
        )

    summary = {
        "ts": now_iso(),
        "work_root": str(layout.work_root),
        "n_step": len(step_index),
        "n_ofs": len(ofs_index),
        "n_paired": len(paired_ids),
        "n_step_only": len(step_only_ids),
        "n_ofs_only": len(ofs_only_ids),
        "n_paired_written": n_paired_written,
        "n_step_only_written": n_step_only_written,
        "n_step_only_diagnostic_written": len(selected_step_only_ids),
        "manifest_path": str(layout.pairing_manifest),
        "step_only_manifest_path": str(layout.step_only_manifest),
        "orphan_ofs_path": str(layout.orphan_ofs),
        "manifest_exists": layout.pairing_manifest.exists(),
        "step_only_manifest_exists": layout.step_only_manifest.exists(),
        "diagnostic_status": (
            "no_step_samples"
            if len(step_index) == 0
            else "ofs_requirement_blocked"
            if require_ofs and len(candidate_ids) == 0 and len(step_index) > 0
            else "empty_shard_or_selection"
            if len(candidate_ids) == 0
            else "ok"
        ),
        "failure_classification": None,
        "selection": {
            "offset": offset,
            "limit": limit,
            "shard_index": shard_index,
            "shard_total": shard_total,
            "require_ofs": require_ofs,
            "scan_ofs": scan_ofs,
        },
        "top_dup_sample_ids": top_dup,
    }
    write_json(layout.pairing_summary, summary)
    logger.info("pairing_summary written to %s", layout.pairing_summary)
    return summary


# ---------- CLI ----------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair STEP and OFS files by sample_id under <work-root>/extracted/."
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0, help="Skip first N sample_ids (sorted)")
    parser.add_argument("--limit", type=int, default=None, help="Keep at most N sample_ids after offset")
    parser.add_argument("--shard-index", type=int, default=None, help="0-based shard index")
    parser.add_argument("--shard-total", type=int, default=None, help="Total number of shards")
    parser.add_argument(
        "--require-ofs",
        action="store_true",
        help="Drop step_only samples from the main manifest",
    )
    parser.add_argument(
        "--skip-ofs-scan",
        action="store_true",
        help="Build STEP-only manifests without scanning extracted/ofs",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    layout = work_root_layout(args.work_root)
    layout.ensure_dirs()

    log_path = layout.work_root / "stage_pair.log"
    with stage_logger(LOGGER_NAME, log_path) as logger:
        if not layout.extracted_step_root.exists() or not layout.extracted_ofs_root.exists():
            logger.error(
                "extracted/{step,ofs} not found under %s; run extract_archives first",
                layout.work_root,
            )
            return 2
        try:
            run_pair(
                layout,
                offset=args.offset,
                limit=args.limit,
                shard_index=args.shard_index,
                shard_total=args.shard_total,
                require_ofs=args.require_ofs,
                scan_ofs=not args.skip_ofs_scan,
                logger=logger,
            )
        except KeyboardInterrupt:
            logger.warning("interrupted by user")
            return 130
        except BaseException as exc:  # noqa: BLE001
            logger.error("pairing failed: %s\n%s", exc, traceback.format_exc())
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
