"""并行跟随式 Qwen/API 文本描述入口。

该模块面向正在增长的 `<work-root>/assets/manifest.jsonl` 与
`<work-root>/occlusion/manifest.jsonl`：

- 只读扫描已有资产 manifest，不修改主资产生成状态；
- 将基础资产与遮挡增强资产规范化为现有 `describe_step_with_qwen.StepRecord`；
- 复用现有 Qwen prompt/API 调用逻辑；
- 追加写入独立描述目录，支持断点续跑、幂等跳过与轮询等待新资产。
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from cad_data_gen.describe_step_with_qwen import (
    DEFAULT_API_BASE,
    DEFAULT_API_KEY_FILE,
    DEFAULT_MODEL,
    StepRecord,
    build_ofs_index,
    load_api_key,
    process_record_with_qwen,
)

from .logging_utils import append_jsonl, iter_jsonl, now_iso, stage_logger, write_json
from .paths import work_root_layout

LOGGER_NAME = "follow_describe_assets"
DEFAULT_OUTPUT_DIRNAME = "descriptions_parallel"


@dataclass(frozen=True)
class DescriptionTask:
    """一条可描述资产任务。"""

    asset_key: str
    asset_type: str
    augmentation_type: str
    source_manifest: str
    source_manifest_index: int
    record: StepRecord
    source_sample_id: Optional[str] = None
    label_path: Optional[str] = None
    mask_paths: tuple[str, ...] = ()
    source_record: dict[str, Any] = field(default_factory=dict)


@dataclass
class FollowDescribeStats:
    """运行统计。"""

    discovered: int = 0
    already_done: int = 0
    processed_ok: int = 0
    failed: int = 0
    skipped: int = 0
    dry_run: int = 0
    loops: int = 0
    last_new_task_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(value: Any, root: Path) -> Optional[Path]:
    if value is None or value == "":
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _first_existing(paths: Iterable[Optional[Path]]) -> Optional[Path]:
    for path in paths:
        if path is not None and path.is_file():
            return path
    return None


def _build_step_filename_index(extracted_step_root: Path) -> dict[str, Path]:
    """按文件名索引原始 STEP；用于 `assets_input` 镜像被后续批次替换后的兜底定位。"""
    if not extracted_step_root.is_dir():
        return {}
    index: dict[str, Path] = {}
    for path in extracted_step_root.rglob("*.step"):
        if path.is_file():
            index.setdefault(path.name, path.resolve())
    return index


def _build_pairing_step_index(pairing_manifest: Path) -> dict[str, Path]:
    """从配对 manifest 读取 `sample_id -> 原始 STEP`，这是存量资产最稳定的路径来源。"""
    index: dict[str, Path] = {}
    for row in iter_jsonl(pairing_manifest):
        sample_id = row.get("sample_id")
        step_path_value = row.get("step_path")
        if not sample_id or not step_path_value:
            continue
        step_path = Path(str(step_path_value)).expanduser().resolve()
        if step_path.is_file():
            index[str(sample_id)] = step_path
    return index


def _resolve_step_path(value: Any, input_root: Path, step_index: dict[str, Path]) -> Optional[Path]:
    candidate = _resolve_path(value, input_root)
    if candidate is not None and candidate.is_file():
        return candidate
    if value is None or value == "":
        return candidate
    return step_index.get(Path(str(value)).name) or candidate


def _asset_key(asset_type: str, sample_id: str) -> str:
    return f"{asset_type}:{sample_id}"


def _status_ok(row: dict[str, Any]) -> bool:
    status = row.get("status") or row.get("stage_status")
    if isinstance(status, dict):
        return not status.get("error")
    return status in (None, "ok", "done")


def _load_completed_asset_keys(descriptions_path: Path) -> dict[str, str]:
    """读取已有输出，最后一条成功记录视为有效完成。"""
    states: dict[str, str] = {}
    for row in iter_jsonl(descriptions_path):
        key = row.get("asset_key")
        if not key and row.get("sample_id"):
            key = _asset_key(str(row.get("asset_type") or "baseline"), str(row["sample_id"]))
        if not key:
            continue
        states[str(key)] = str(row.get("status") or "unknown")
    return {key: status for key, status in states.items() if status == "ok"}


def _build_asset_index(
    assets_manifest: Path,
    input_root: Path,
    assets_root: Path,
    step_index: dict[str, Path],
    pairing_step_index: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(assets_manifest):
        sample_id = row.get("sample_id")
        if not sample_id:
            continue
        step_path = pairing_step_index.get(str(sample_id)) or _resolve_step_path(row.get("step_path") or row.get("original_step_path"), input_root, step_index)
        if step_path is None:
            continue
        image_paths = [
            p
            for p in (_resolve_path(value, assets_root) for value in (row.get("image_paths") or []))
            if p is not None and p.is_file()
        ]
        point_path = _resolve_path(row.get("point_path"), assets_root)
        mesh_path = _resolve_path(row.get("mesh_path"), assets_root)
        index[str(sample_id)] = {
            "row": row,
            "step_path": step_path,
            "image_paths": tuple(image_paths),

path.is_file() else None,"mesh_path": mesh_path if mesh_path is not None and mesh_pat
h.is_file() else None,
        }
    return index


def _normalize_baseline_tasks(
    *,
    manifest_path: Path,
    input_root: Path,
    assets_root: Path,
    step_index: dict[str, Path],
    pairing_step_index: dict[str, Path],
    max_tasks: Optional[int] = None,
) -> list[DescriptionTask]:
    tasks: list[DescriptionTask] = []
    for idx, row in enumerate(iter_jsonl(manifest_path)):
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id or not _status_ok(row):
            continuestep_path = pairing_step_index.get(sample_id) or _resolve_step_path(row.get("step_path") or row.get("original_step_path"), input_root, ste
p_index)
        if step_path is None:
            continue
        image_paths = tuple(
            pfor p in (_resolve_path(value, assets_root) for value in (ro
w.get("image_paths") or []))
            if p is not None and p.is_file()
        )
        point_path = _resolve_path(row.get("point_path"), assets_root)
        mesh_path = _resolve_path(row.get("mesh_path"), assets_root)
        record = StepRecord(
            sample_id=sample_id,
            step_path=step_path,relative_step_path=step_path.name if not step_path.is_relativ
e_to(input_root) else str(step_path.relative_to(input_root)),
            source_index=row.get("source_index", idx),
            dataset_key=row.get("dataset_key"),
            render_dir=image_paths[0].parent if image_paths else None,
            image_paths=image_paths,point_path=point_path if point_path is not None and point_pat
h.is_file() else None,mesh_path=mesh_path if mesh_path is not None and mesh_path.is
_file() else None,
("loader"),conversion_metadata=row.get("conversion_metadata") if isinsta
nce(row.get("conversion_metadata"), dict) else None,mesh_metrics=row.get("mesh_metrics") if isinstance(row.get("m
esh_metrics"), dict) else None,mesh_metrics_error=row.get("mesh_metrics_error") if isinstanc
e(row.get("mesh_metrics_error"), str) else None,
        )
        tasks.append(
            DescriptionTask(
                asset_key=_asset_key("baseline", sample_id),
                asset_type="baseline",
                augmentation_type="baseline",
                source_manifest=str(manifest_path),
                source_manifest_index=idx,
                record=record,
                source_sample_id=sample_id,
                source_record=row,
            )
        )
        if max_tasks is not None and len(tasks) >= max_tasks:
            break
    return tasks


def _normalize_occlusion_tasks(
    *,
    manifest_path: Path,
    input_root: Path,
    assets_manifest: Path,
    assets_root: Path,
    occlusion_root: Path,
    step_index: dict[str, Path],
    pairing_step_index: dict[str, Path],
    max_tasks: Optional[int] = None,
) -> list[DescriptionTask]:source_assets = _build_asset_index(assets_manifest, input_root, assets_root, step_index, pairing_step_index) if assets_manifest.is_file() els
e {}
    tasks: list[DescriptionTask] = []
    for idx, row in enumerate(iter_jsonl(manifest_path)):
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id or not _status_ok(row):
            continuesource_sample_id = str(row.get("source_sample_id") or "").strip
() or None
        source = source_assets.get(source_sample_id or "")
        step_path = _first_existing(
            (
                pairing_step_index.get(source_sample_id or ""),_resolve_step_path(row.get("step_path"), input_root, step
_index),_resolve_step_path(row.get("source_step_path"), input_roo
t, step_index),
                source.get("step_path") if source else None,
            )
        )
        if step_path is None:
            continue
        image_paths = tuple(
            pfor p in (_resolve_path(value, occlusion_root) for value in
(row.get("image_paths") or []))
            if p is not None and p.is_file()
        )
        point_path = _resolve_path(row.get("point_path"), occlusion_root)
        source_row = source.get("row", {}) if source else {}
        mesh_path = source.get("mesh_path") if source else None
        record = StepRecord(
            sample_id=sample_id,
            step_path=step_path,relative_step_path=step_path.name if not step_path.is_relativ
e_to(input_root) else str(step_path.relative_to(input_root)),
            source_index=row.get("result_index", idx),
            dataset_key=source_row.get("dataset_key"),
            render_dir=image_paths[0].parent if image_paths else None,
            image_paths=image_paths,point_path=point_path if point_path is not None and point_pat
h.is_file() else None,mesh_path=mesh_path if isinstance(mesh_path, Path) and mesh_p
ath.is_file() else None,conversion_backend=row.get("loader") or source_row.get("conve
rsion_backend") or source_row.get("loader"),conversion_metadata=source_row.get("conversion_metadata") if
isinstance(source_row.get("conversion_metadata"), dict) else None,mesh_metrics=source_row.get("mesh_metrics") if isinstance(sou
rce_row.get("mesh_metrics"), dict) else None,mesh_metrics_error=source_row.get("mesh_metrics_error") if is
instance(source_row.get("mesh_metrics_error"), str) else None,
        )
        tasks.append(
            DescriptionTask(
                asset_key=_asset_key("occlusion", sample_id),
                asset_type="occlusion",
                augmentation_type=str(row.get("variant_type") or row.get
("occlusion_mode") or "occlusion"),
                source_manifest=str(manifest_path),
                source_manifest_index=idx,
                record=record,source_sample_id=source_sample_id,
                label_path=str(_resolve_path(row.get("label_path"), occlusion_root)) if row.get("label_path") else None,mask_paths=tuple(str(p) for p in (_resolve_path(value, oc
clusion_root) for value in (row.get("mask_paths") or [])) if p is not Non
e),
                source_record=row,
            )
        )
        if max_tasks is not None and len(tasks) >= max_tasks:
            break
    return tasks


def scan_description_tasks(
    *,
    work_root: Path,
    include_baseline: bool = True,
    include_occlusion: bool = True,
    assets_manifest: Optional[Path] = None,
    occlusion_manifest: Optional[Path] = None,
    max_tasks: Optional[int] = None,
) -> list[DescriptionTask]:
    """扫描当前已有基础资产/遮挡资产，返回规范化任务。"""
    layout = work_root_layout(work_root)input_root = layout.work_root / "assets_input"
    pairing_step_index = _build_pairing_step_index(layout.pairing_manifest)
    step_index = {} if pairing_step_index else _build_step_filename_index
(layout.extracted_step_root)assets_manifest_path = assets_manifest or layout.assets_manifest
    occlusion_manifest_path = occlusion_manifest or (layout.occlusion_di
r / "manifest.jsonl")tasks: list[DescriptionTask] = []
    if include_baseline and assets_manifest_path.is_file() and (max_task
s is None or len(tasks) < max_tasks):
        tasks.extend(
            _normalize_baseline_tasks(
                manifest_path=assets_manifest_path,
                input_root=input_root,
                assets_root=layout.assets_dir,
                step_index=step_index,
                pairing_step_index=pairing_step_index,
                max_tasks=max_tasks,
            ))
    if include_occlusion and occlusion_manifest_path.is_file() and (max_t
asks is None or len(tasks) < max_tasks):
        tasks.extend(
            _normalize_occlusion_tasks(
                manifest_path=occlusion_manifest_path,
                input_root=input_root,
                assets_manifest=assets_manifest_path,
                assets_root=layout.assets_dir,
                occlusion_root=layout.occlusion_dir,
                step_index=step_index,pairing_step_index=pairing_step_index,
                max_tasks=None if max_tasks is None else max_tasks - len
(tasks),
            )
        )
    return tasks

def _write_failure(output_dir: Path, task: DescriptionTask, stage: str, e
rror: str) -> None:
    append_jsonl(
        output_dir / "failures.jsonl",
        {
            "asset_key": task.asset_key,
            "asset_type": task.asset_type,
            "augmentation_type": task.augmentation_type,
            "sample_id": task.record.sample_id,
            "source_sample_id": task.source_sample_id,
            "source_manifest": task.source_manifest,
            "source_manifest_index": task.source_manifest_index,
            "status": "failed",
            "stage": stage,
            "error": error,
            "created_at": _utc_now(),
        },
    )

def _write_skip(output_dir: Path, task: DescriptionTask, reason: str) ->
None:
    append_jsonl(
        output_dir / "skipped.jsonl",
        {
            "asset_key": task.asset_key,
            "asset_type": task.asset_type,
            "augmentation_type": task.augmentation_type,
            "sample_id": task.record.sample_id,
            "source_sample_id": task.source_sample_id,
            "source_manifest": task.source_manifest,
            "source_manifest_index": task.source_manifest_index,
            "status": "skipped",
            "reason": reason,
            "created_at": _utc_now(),
        },
    )

def _result_with_task_metadata(result: dict[str, Any], task: DescriptionT
ask) -> dict[str, Any]:
    enriched = dict(result)
    enriched.update(
        {
            "asset_key": task.asset_key,
            "asset_type": task.asset_type,
            "augmentation_type": task.augmentation_type,
            "source_sample_id": task.source_sample_id,
            "source_manifest": task.source_manifest,
            "source_manifest_index": task.source_manifest_index,
            "label_path": task.label_path,
            "mask_paths": list(task.mask_paths),
        }
    )
    return enriched


def _process_task(
    *,
    task: DescriptionTask,
    args: argparse.Namespace,
    api_key: Optional[str],
    ofs_index: dict[str, Path],
    output_dir: Path,
) -> str:if not task.record.step_path.is_file():
        _write_skip(output_dir, task, f"STEP file not found: {task.recor
d.step_path}")
        return "skipped"if not task.record.image_paths:
        _write_skip(output_dir, task, "no render image paths in manifes
t")
        return "skipped"
    if args.dry_run:
        append_jsonl(
            output_dir / "dry_run_tasks.jsonl",
            {
                "asset_key": task.asset_key,
                "asset_type": task.asset_type,
                "augmentation_type": task.augmentation_type,
                "sample_id": task.record.sample_id,
                "source_sample_id": task.source_sample_id,"step_path": str(task.record.step_path),
                "image_paths": [str(path) for path in task.record.image_paths],
                "point_path": str(task.record.point_path) if task.record.
point_path else None,
                "source_manifest": task.source_manifest,
                "source_manifest_index": task.source_manifest_index,
                "status": "dry_run",
                "created_at": _utc_now(),
            },
        )
        return "dry_run"
    result, failure = process_record_with_qwen(task.record, args, api_ke
y, ofs_index)if result is not None:
        append_jsonl(output_dir / "descriptions.jsonl", _result_with_task
_metadata(result, task))
        return "ok"
    if failure is not None:
        failure = dict(failure)
        failure.update(
            {
                "asset_key": task.asset_key,
                "asset_type": task.asset_type,
                "augmentation_type": task.augmentation_type,
                "source_sample_id": task.source_sample_id,
                "source_manifest": task.source_manifest,
                "source_manifest_index": task.source_manifest_index,
            }
        )
        append_jsonl(output_dir / "failures.jsonl", failure)return "failed"
    _write_failure(output_dir, task, "unknown", "process_record_with_qwe
n returned no result and no failure")
    return "failed"


def run_follow_describe(args: argparse.Namespace) -> FollowDescribeStats:
    work_root = args.work_root.expanduser().resolve()layout = work_root_layout(work_root)
    output_dir = (args.output_dir or (layout.work_root / DEFAULT_OUTPUT_D
IRNAME)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "follow_describe_assets.log"
    stats = FollowDescribeStats()
    stop_requested = False

    def _handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    with stage_logger(LOGGER_NAME, log_path) as logger:
        logger.info("follow describe start: work_root=%s output_dir=%s",
work_root, output_dir)if args.concurrency != 1:
            logger.warning("--concurrency=%d accepted but current implementation processes sequentially", args.concurrency)
        api_key = None if args.dry_run else load_api_key(args.api_key_en
v, str(args.api_key_file) if args.api_key_file else None)if not args.dry_run and not api_key:
            raise RuntimeError(f"missing API key: set ${args.api_key_env} or provide --api-key-file")
        ofs_index = build_ofs_index(layout.extracted_ofs_root) if args.use_ofs_context else {}
        completed = _load_completed_asset_keys(output_dir / "description
s.jsonl") if args.resume else {}
        idle_started = time.time()
        last_log = 0.0

        while not stop_requested:
            stats.loops += 1
            tasks = scan_description_tasks(
                work_root=work_root,
                include_baseline=args.include_baseline,
                include_occlusion=args.include_occlusion,
                assets_manifest=args.assets_manifest,
                occlusion_manifest=args.occlusion_manifest,
                max_tasks=args.limit,
            )stats.discovered = len(tasks)
            pending = [task for task in tasks if task.asset_key not in co
mpleted]
            if args.limit is not None:
                pending = pending[: args.limit]
            if pending:
                idle_started = time.time()stats.last_new_task_at = now_iso()
                logger.info("found pending description tasks: %d/%d", len
(pending), len(tasks))
            else:
                now = time.time()
                if now - last_log >= args.log_interval:logger.info(
                        "waiting for new assets: discovered=%d completed
=%d ok=%d failed=%d skipped=%d dry_run=%d",
                        stats.discovered,
                        len(completed),
                        stats.processed_ok,
                        stats.failed,
                        stats.skipped,
                        stats.dry_run,
                    )
                    last_log = now
                if not args.follow:break
                if args.idle_timeout_s is not None and time.time() - idle_started >= args.idle_timeout_s:
                    logger.info("idle timeout reached: %.1fs", args.idle_
timeout_s)
                    break
                time.sleep(args.poll_interval)
                continue

            for task in pending:
                if stop_requested:break
                status = _process_task(task=task, args=args, api_key=api_
key, ofs_index=ofs_index, output_dir=output_dir)
                if status == "ok":
                    stats.processed_ok += 1
                    completed[task.asset_key] = "ok"
                elif status == "failed":
                    stats.failed += 1
                elif status == "skipped":
                    stats.skipped += 1
                elif status == "dry_run":
                    stats.dry_run += 1
                    completed[task.asset_key] = "ok"
                if args.qps and args.qps > 0:
                    time.sleep(1.0 / args.qps)
            stats.already_done = len(completed)

            write_json(output_dir / "follow_describe_state.json", stats.to_dict())
            logger.info(
                "progress: discovered=%d completed=%d ok=%d failed=%d ski
pped=%d dry_run=%d",
                stats.discovered,
                stats.already_done,
                stats.processed_ok,
                stats.failed,
                stats.skipped,
                stats.dry_run,
            )
            if not args.follow:
                break
            if args.limit is not None:
                logger.info("limit reached for this run: %d", args.limit)break
        write_json(output_dir / "follow_describe_state.json", stats.to_di
ct())
        logger.info("follow describe exit: %s", json.dumps(stats.to_dict
(), ensure_ascii=False))
    return stats

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Follow growing ABC asse
ts/occlusion manifests and generate Qwen text descriptions.")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)parser.add_argument("--assets-manifest", type=Path, default=None)
    parser.add_argument("--occlusion-manifest", type=Path, default=None)parser.add_argument("--include-baseline", action=argparse.BooleanOpti
onalAction, default=True)parser.add_argument("--include-occlusion", action=argparse.BooleanOpt
ionalAction, default=True)
    parser.add_argument("--follow", action=argparse.BooleanOptionalActio
n, default=True)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--idle-timeout-s", type=float, default=None)parser.add_argument("--log-interval", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalActio
n, default=True)parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_
KEY_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--qps", type=float, default=None)
    parser.add_argument("--write-text-files", action="store_true")
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--max-ofs-features", type=int, default=80)
    parser.add_argument("--max-step-chars", type=int, default=0)parser.add_argument("--triangle-face-tol", type=float, default=0.01)parser.add_argument("--angle-tol-rads", type=float, default=0.1)parser.add_argument("--use-ofs-context", action=argparse.BooleanOptionalActiondefault=True)
