"""基础资产阶段封装：调用 `cad_data_gen.build_step_assets`。

输入：`<work-root>/pairing_manifest.jsonl`（由 `pair_samples` 产出）
输出：`<work-root>/assets/`（由 `build_step_assets.py` 产出 manifest.jsonl + 点云 + 渲染图）

实现策略：

`build_step_assets.py` 是按 `--input-dir` 扫目录的，不直接吃 manifest。
为使分片切分生效（manifest 内只含选中的 sample_id），本阶段把 manifest 中的
`step_path` 在 `<work-root>/assets_input/` 下 **以 symlink 方式镜像**（保留
相对路径），让底层 `--recursive` 扫到的 sample 集恰好与 manifest 一致。

只在 manifest 与 mirror 之间做 symlink，不复制，O(1) 磁盘开销。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .logging_utils import (
    iter_jsonl,
    now_iso,
    stage_logger,
)
from .paths import WorkRootLayout, work_root_layout

LOGGER_NAME = "stage_assets"
STAGED_MARKER = "STAGED"
ASSETS_DONE_MARKER = "DONE"
ASSETS_SAMPLE_STATUS = "sample_status.jsonl"
ASSETS_FAILURE_SUMMARY = "failure_summary.json"


@dataclass
class StageResult:
    stage: str
    status: str  # "done" / "skipped" / "failed"
    started_at: str = ""
    ended_at: str = ""
    elapsed_s: float = 0.0
    cmd: list[str] = field(default_factory=list)
    return_code: Optional[int] = None
    error: Optional[str] = None
    n_inputs: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _materialize_input_mirror(
    manifest_path: Path,
    extracted_step_root: Path,
    mirror_root: Path,
    logger: logging.Logger,
) -> int:
    """把 manifest 中的 STEP 文件以 symlink 的方式镜像到 mirror_root。

    普通 pair manifest 使用 `step_path`，并尽量保留其相对 `<work-root>/extracted/step`
    的路径；staged batch manifest 优先使用 `staged_step_path`，并用稳定的
    `sample_id + 原始后缀` 作为镜像相对路径，确保 `build_step_assets` 只扫描当前
    批次且生成的 sample_id 与批次 manifest 保持一致。

    返回成功 link 的数量。
    """
    if mirror_root.exists():
        # 重置 mirror（保证只有 manifest 中的样本被链接进来）
        shutil.rmtree(mirror_root)
    mirror_root.mkdir(parents=True, exist_ok=True)

    n = 0
    n_missing = 0
    for record in iter_jsonl(manifest_path):
        step_path = record.get("staged_step_path") or record.get("step_path")
        if not step_path:
            continue
        step_p = Path(str(step_path))
        if not step_p.exists():
            n_missing += 1
            continue
        rel: Path
        if record.get("staged_step_path"):
            sample_id = str(record.get("sample_id") or step_p.stem)
            suffix = step_p.suffix if step_p.suffix.lower() in (".step", ".stp") else ".step"
            rel = Path(f"{sample_id}{suffix}")
        else:
            try:
                rel = step_p.resolve().relative_to(extracted_step_root.resolve())
            except ValueError:
                sample_id = str(record.get("sample_id") or step_p.stem)
                suffix = step_p.suffix if step_p.suffix.lower() in (".step", ".stp") else ".step"
                rel = Path(f"{sample_id}{suffix}")
        link_path = mirror_root / rel
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.exists() or link_path.is_symlink():
            try:
                link_path.unlink()
            except OSError:
                pass
        try:
            os.symlink(step_p.resolve(), link_path)
        except OSError as exc:
            logger.warning("symlink failed for %s: %s; falling back to hard copy stub", step_p, exc)
            try:
                shutil.copy2(step_p, link_path)
            except OSError as exc2:
                logger.error("copy fallback failed for %s: %s", step_p, exc2)
                continue
        n += 1
    logger.info(
        "mirrored %d step files into %s (skipped missing=%d)",
        n,
        mirror_root,
        n_missing,
    )
    return n


def _count_jsonl_records(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def _tail_jsonl_records(path: Path, limit: int = 3) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for record in iter_jsonl(path):
        records.append(record)
        if len(records) > limit:
            records.pop(0)
    return records


def _resolve_batch_manifest_from_cfg(layout: WorkRootLayout, cfg: dict) -> Optional[Path]:
    manifest_value = cfg.get("staged_manifest") or cfg.get("batch_manifest") or cfg.get("input_manifest")
    if manifest_value:
        return Path(str(manifest_value)).expanduser().resolve()

    batch_id = cfg.get("batch_id")
    if not batch_id:
        return None
    staging_root = cfg.get("staging_root")
    if staging_root:
        return Path(str(staging_root)).expanduser().resolve() / str(batch_id) / "manifest.jsonl"
    return layout.batches_dir / str(batch_id) / "manifest.jsonl"


def _validate_staged_manifest(manifest_path: Path, logger: logging.Logger) -> dict[str, Any]:
    marker_path = manifest_path.parent / STAGED_MARKER
    n_records = 0
    n_staged_step = 0
    n_missing_step = 0
    for record in iter_jsonl(manifest_path):
        n_records += 1
        staged_step = record.get("staged_step_path")
        if staged_step and Path(str(staged_step)).is_file():
            n_staged_step += 1
        elif record.get("staged_step_path"):
            n_missing_step += 1
    info = {
        "staged_marker": str(marker_path),
        "staged_marker_exists": marker_path.exists(),
        "n_staged_manifest_records": n_records,
        "n_staged_step_files": n_staged_step,
        "n_missing_staged_step_files": n_missing_step,
    }
    if not marker_path.exists():
        logger.warning("staged marker missing beside manifest: %s", marker_path)
    return info


def _select_assets_manifest(
    layout: WorkRootLayout,
    cfg: dict,
    logger: logging.Logger,
) -> tuple[Optional[Path], int, str]:
    """选择资产阶段输入清单。

    优先使用显式传入的 staged/batch/input manifest；否则沿用历史逻辑：默认优先
    使用 `pairing_manifest.jsonl`，当它为空且资产阶段不要求 OFS 时，自动退到
    `step_only_manifest.jsonl`。
    """
    explicit_manifest = _resolve_batch_manifest_from_cfg(layout, cfg)
    if explicit_manifest is not None:
        if not explicit_manifest.exists():
            logger.error("explicit assets input manifest does not exist: %s", explicit_manifest)
            return None, 0, "explicit_missing"
        n_records = _count_jsonl_records(explicit_manifest)
        if n_records <= 0:
            logger.error("explicit assets input manifest is empty: %s", explicit_manifest)
            return None, 0, "explicit_empty"
        return explicit_manifest, n_records, "explicit_manifest"

    require_ofs_for_assets = bool(cfg.get("require_ofs_for_assets", cfg.get("require_ofs", False)))
    candidates = [(layout.pairing_manifest, "pairing_manifest")]
    if not require_ofs_for_assets:
        candidates.append((layout.step_only_manifest, "step_only_manifest"))

    missing: list[str] = []
    empty: list[str] = []
    for path, source in candidates:
        if not path.exists():
            missing.append(str(path))
            continue
        n_records = _count_jsonl_records(path)
        if n_records > 0:
            if source == "step_only_manifest":
                logger.warning(
                    "pairing manifest has no usable records; using STEP-only manifest %s for asset generation",
                    path,
                )
            return path, n_records, source
        empty.append(str(path))

    if require_ofs_for_assets:
        logger.error(
            "no paired STEP+OFS records available for assets while require_ofs_for_assets=true; "
            "disable --require-ofs / require_ofs_for_assets or fix OFS pairing rules"
        )
    else:
        logger.error(
            "no STEP records available for assets; missing=%s empty=%s",
            missing,
            empty,
        )
    return None, 0, "missing_or_empty"


def _load_pairing_metadata(manifest_path: Path) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(manifest_path):
        sample_id = record.get("sample_id")
        if not sample_id:
            continue
        meta[str(sample_id)] = {
            "has_ofs": bool(record.get("has_ofs")),
            "ofs_status": record.get("ofs_status") or ("matched" if record.get("has_ofs") else "unmatched"),
            "ofs_path": record.get("ofs_path"),
            "source_ofs_archive": record.get("source_ofs_archive"),
            "source_step_archive": record.get("source_step_archive"),
            "pairing_status": record.get("status"),
        }
    return meta


def _sample_ids_from_manifest(manifest_path: Path) -> list[str]:
    sample_ids: list[str] = []
    seen: set[str] = set()
    for record in iter_jsonl(manifest_path):
        sample_id = str(record.get("sample_id") or "").strip()
        if not sample_id or sample_id in seen:
            continue
        sample_ids.append(sample_id)
        seen.add(sample_id)
    return sample_ids


def _asset_records_by_sample_id(assets_manifest: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(assets_manifest):
        sample_id = str(record.get("sample_id") or "").strip()
        if sample_id:
            records[sample_id] = record
    return records


def _failures_by_sample_id(failures_path: Path) -> dict[str, list[dict[str, Any]]]:
    failures: dict[str, list[dict[str, Any]]] = {}
    for record in iter_jsonl(failures_path):
        sample_id = str(record.get("sample_id") or "").strip()
        if not sample_id:
            sample_id = str(record.get("step_path") or "unknown")
        failures.setdefault(sample_id, []).append(record)
    return failures


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def _write_assets_completion_snapshot(
    assets_dir: Path,
    input_manifest: Path,
    assets_manifest: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    """写入资产阶段的样本级状态、失败摘要和批次完成标记。"""
    failures_path = assets_dir / "failures.jsonl"
    progress_path = assets_dir / "progress.json"
    summary_path = assets_dir / "summary.json"
    sample_status_path = assets_dir / ASSETS_SAMPLE_STATUS
    failure_summary_path = assets_dir / ASSETS_FAILURE_SUMMARY
    done_marker_path = assets_dir / ASSETS_DONE_MARKER

    expected_sample_ids = _sample_ids_from_manifest(input_manifest)
    expected_sample_id_set = set(expected_sample_ids)
    asset_records = _asset_records_by_sample_id(assets_manifest)
    failures = _failures_by_sample_id(failures_path)

    sample_status_records: list[dict[str, Any]] = []
    n_done = 0
    n_failed = 0
    n_pending = 0
    for sample_id in expected_sample_ids:
        asset_record = asset_records.get(sample_id)
        sample_failures = failures.get(sample_id, [])
        if asset_record is not None:
            status = "done"
            n_done += 1
        elif sample_failures:
            status = "failed"
            n_failed += 1
        else:
            status = "pending"
            n_pending += 1
        sample_status_records.append(
            {
                "sample_id": sample_id,
                "status": status,
                "point_path": asset_record.get("point_path") if asset_record else None,
                "image_paths": asset_record.get("image_paths") if asset_record else [],
                "failure_count": len(sample_failures),
                "latest_failure": sample_failures[-1] if sample_failures else None,
            }
        )

    orphan_failure_count = sum(
        len(items) for sample_id, items in failures.items() if sample_id not in expected_sample_id_set
    )
    failure_stages: dict[str, int] = {}
    failure_types: dict[str, int] = {}
    for items in failures.values():
        for failure in items:
            stage = str(failure.get("stage") or "unknown")
            error_type = str(failure.get("error_type") or "unknown")
            failure_stages[stage] = failure_stages.get(stage, 0) + 1
            failure_types[error_type] = failure_types.get(error_type, 0) + 1

    _write_jsonl(sample_status_path, sample_status_records)
    failure_summary = {
        "generated_at": now_iso(),
        "failures_path": str(failures_path),
        "n_failed_samples": n_failed,
        "n_failure_records": sum(len(items) for items in failures.values()),
        "n_orphan_failure_records": orphan_failure_count,
        "failure_stages": failure_stages,
        "failure_types": failure_types,
        "recent_failures": _tail_jsonl_records(failures_path, limit=5),
    }
    _write_json(failure_summary_path, failure_summary)

    completion_status = "done" if n_pending == 0 else "incomplete"
    done_payload = {
        "stage": "assets",
        "status": completion_status,
        "generated_at": now_iso(),
        "input_manifest": str(input_manifest),
        "assets_manifest": str(assets_manifest),
        "sample_status": str(sample_status_path),
        "failure_summary": str(failure_summary_path),
        "progress": str(progress_path) if progress_path.exists() else None,
        "summary": str(summary_path) if summary_path.exists() else None,
        "n_expected_samples": len(expected_sample_ids),
        "n_done_samples": n_done,
        "n_failed_samples": n_failed,
        "n_pending_samples": n_pending,
    }
    _write_json(done_marker_path, done_payload)
    logger.info(
        "assets completion snapshot written: done=%d failed=%d pending=%d marker=%s",
        n_done,
        n_failed,
        n_pending,
        done_marker_path,
    )
    return {
        "assets_done_marker": str(done_marker_path),
        "assets_sample_status": str(sample_status_path),
        "assets_failure_summary": str(failure_summary_path),
        "n_assets_done_samples": n_done,
        "n_assets_failed_samples": n_failed,
        "n_assets_pending_samples": n_pending,
    }


def _annotate_assets_manifest(
    assets_manifest: Path,
    pairing_manifest: Path,
    logger: logging.Logger,
) -> dict[str, int]:
    pairing_meta = _load_pairing_metadata(pairing_manifest)
    tmp_path = assets_manifest.with_suffix(assets_manifest.suffix + ".tmp")
    n_total = 0
    n_with_ofs = 0
    n_without_ofs = 0
    n_missing_pair_meta = 0
    with assets_manifest.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dst.write(line)
                continue
            sample_id = str(row.get("sample_id") or "")
            meta = pairing_meta.get(sample_id)
            if meta is None:
                n_missing_pair_meta += 1
                row.setdefault("has_ofs", False)
                row.setdefault("ofs_status", "unknown")
            else:
                row.update(meta)
            if row.get("has_ofs"):
                n_with_ofs += 1
            else:
                n_without_ofs += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_total += 1
    tmp_path.replace(assets_manifest)
    logger.info(
        "annotated assets manifest with OFS metadata: total=%d with_ofs=%d without_ofs=%d missing_pair_meta=%d",
        n_total,
        n_with_ofs,
        n_without_ofs,
        n_missing_pair_meta,
    )
    return {
        "n_assets_records": n_total,
        "n_assets_with_ofs": n_with_ofs,
        "n_assets_without_ofs": n_without_ofs,
        "n_assets_missing_pair_meta": n_missing_pair_meta,
    }


def _build_cmd(
    input_dir: Path,
    output_dir: Path,
    cfg: dict,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "cad_data_gen.build_step_assets",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--recursive",
    ]
    # 默认值（与 cad_shape_1000_english_blender512_20260527 一致）
    num_points = cfg.get("num_points", 8192)
    num_views = cfg.get("num_views", 8)
    img_size = cfg.get("img_size", 512)
    render_backend = cfg.get("render_backend", "blender-step")
    blender_style = cfg.get("blender_style", "visualization")
    cmd += [
        "--num-points",
        str(num_points),
        "--num-views",
        str(num_views),
        "--img-size",
        str(img_size),
        "--render-backend",
        render_backend,
        "--blender-style",
        blender_style,
    ]

    # 可选透传
    if "num_processes" in cfg:
        cmd += ["--num-processes", str(cfg["num_processes"])]
    if cfg.get("blender_bin"):
        cmd += ["--blender-bin", str(cfg["blender_bin"])]
    if cfg.get("blender_script"):
        cmd += ["--blender-script", str(cfg["blender_script"])]
    if cfg.get("visualization_root"):
        cmd += ["--visualization-root", str(cfg["visualization_root"])]
    if cfg.get("incremental_manifest", True):
        cmd += ["--incremental-manifest"]
    if cfg.get("resume_manifest", True):
        cmd += ["--resume-manifest"]
    if cfg.get("skip_existing", True):
        cmd += ["--skip-existing"]

    # 任意额外覆盖（按 yaml 暴露的高级参数）
    for key in (
        "camera_distance",
        "camera_jitter_degrees",
        "camera_jitter_seed",
        "triangle_face_tol",
        "angle_tol_rads",
        "step_mesh_backend",
        "step_mesh_fallback_backends",
        "step_mesh_work_dir",
        "step_mesh_format",
        "freecad_cmd",
        "step_mesh_timeout_s",
        "blender_engine",
        "blender_samples",
    ):
        if key in cfg and cfg[key] is not None:
            cmd += [f"--{key.replace('_', '-')}", str(cfg[key])]
    if cfg.get("discard_intermediate_mesh"):
        cmd += ["--discard-intermediate-mesh"]

    return cmd


def run_assets_stage(
    cfg: dict,
    work_root: str | Path,
    *,
    skip: bool = False,
    logger: Optional[logging.Logger] = None,
) -> StageResult:
    layout = work_root_layout(work_root)
    layout.ensure_dirs()

    log_path = layout.stage_assets_log
    owns_logger = logger is None
    cm = stage_logger(LOGGER_NAME, log_path) if owns_logger else None
    log = cm.__enter__() if cm is not None else logger
    assert log is not None

    res = StageResult(stage="assets", status="pending", started_at=now_iso())
    started_t = __import__("time").time()
    try:
        if skip:
            if not layout.assets_manifest.exists():
                res.status = "failed"
                res.error = (
                    f"--skip-assets set but {layout.assets_manifest} does not exist"
                )
                log.error(res.error)
                return res
            log.info("--skip-assets set; assets/manifest.jsonl exists, skipping")
            res.status = "skipped"
            return res

        manifest_path, n_manifest_records, manifest_source = _select_assets_manifest(layout, cfg, log)
        if manifest_path is None:
            res.status = "failed"
            res.error = "no usable STEP manifest for assets stage"
            log.error(res.error)
            return res
        res.extra["input_manifest"] = str(manifest_path)
        res.extra["input_manifest_source"] = manifest_source
        res.extra["n_input_manifest_records"] = n_manifest_records
        if manifest_source == "explicit_manifest":
            res.extra.update(_validate_staged_manifest(manifest_path, log))

        mirror_root = layout.work_root / "assets_input"
        n_inputs = _materialize_input_mirror(
            manifest_path,
            layout.extracted_step_root,
            mirror_root,
            log,
        )
        res.n_inputs = n_inputs
        if n_inputs == 0:
            res.status = "failed"
            res.error = "no step files materialized into assets_input mirror"
            log.error(res.error)
            return res

        cmd = _build_cmd(mirror_root, layout.assets_dir, cfg)
        res.cmd = cmd
        log.info("running: %s", " ".join(cmd))

        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(f"\n=== {now_iso()} === assets stage cmd: {' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(cmd, stdout=logf, stderr=logf)
        res.return_code = proc.returncode
        if proc.returncode != 0:
            res.status = "failed"
            res.error = f"build_step_assets exited with {proc.returncode}; see {log_path}"
            log.error(res.error)
            return res

        if not layout.assets_manifest.exists():
            res.status = "failed"
            res.error = f"assets manifest not produced at {layout.assets_manifest}"
            log.error(res.error)
            return res

        annotation_stats = _annotate_assets_manifest(layout.assets_manifest, manifest_path, log)
        completion_stats = _write_assets_completion_snapshot(
            layout.assets_dir,
            manifest_path,
            layout.assets_manifest,
            log,
        )
        res.status = "done"
        res.extra.update(annotation_stats)
        res.extra.update(completion_stats)
        log.info("assets stage done: %s", res.extra)
        return res
    except BaseException as exc:  # noqa: BLE001
        res.status = "failed"
        res.error = f"assets stage crashed: {exc}\n{traceback.format_exc()}"
        log.error(res.error)
        return res
    finally:
        res.ended_at = now_iso()
        res.elapsed_s = round(__import__("time").time() - started_t, 2)
        if cm is not None:
            cm.__exit__(None, None, None)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ABC asset generation stage.")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file")
    parser.add_argument("--skip", action="store_true")
    args = parser.parse_args(argv)

    cfg: dict[str, Any] = {}
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise SystemExit("asset stage config must be a JSON object")

    res = run_assets_stage(cfg, args.work_root, skip=args.skip)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    return 0 if res.status in ("done", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
