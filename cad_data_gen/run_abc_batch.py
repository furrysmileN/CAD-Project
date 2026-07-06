"""Top-level ABC dataset batch generation pipeline.

The minimal supported chain is:

    extract_archives -> pair_samples -> stage_assets

Occlusion and Qwen description stages are optional and imported lazily so that
the core asset pipeline can run even when those extension modules are absent or
under repair.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .abc_batch.extract_archives import run_extract
from .abc_batch.logging_utils import now_iso, safe_disk_usage, stage_logger, write_json
from .abc_batch.pair_samples import run_pair
from .abc_batch.paths import work_root_layout
from .abc_batch.stage_assets import StageResult, run_assets_stage

LOGGER_NAME = "run_abc_batch"
DISK_FREE_WARN_BYTES = 200 * 1024**3


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required to load --config; install via `pip install pyyaml`") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"yaml config root must be a mapping, got {type(data).__name__}")
    return data


@dataclass
class RunState:
    started_at: str = ""
    ended_at: str = ""
    config_path: Optional[str] = None
    work_root: str = ""
    abc_root: str = ""
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    final_status: str = "running"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _save_run_state(layout, state: RunState) -> None:
    write_json(layout.run_state, state.to_dict())


def _stage_record(
    name: str,
    status: str,
    *,
    started_at: str = "",
    ended_at: str = "",
    elapsed_s: float = 0.0,
    cmd: Optional[list[str]] = None,
    error: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "stage": name,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_s": elapsed_s,
    }
    if cmd is not None:
        rec["cmd"] = cmd
    if error is not None:
        rec["error"] = error
    if extra:
        rec["extra"] = extra
    return rec


def _stage_dict_from_result(res: StageResult) -> dict[str, Any]:
    return res.to_dict()


def _disk_check(work_root: Path, logger) -> None:
    work_root.mkdir(parents=True, exist_ok=True)
    usage = safe_disk_usage(work_root)
    if usage is None:
        logger.warning("could not stat disk usage for %s", work_root)
        return
    total, used, free = usage
    logger.info(
        "disk: total=%.1f GB used=%.1f GB free=%.1f GB at %s",
        total / 1024**3,
        used / 1024**3,
        free / 1024**3,
        work_root,
    )
    if free < DISK_FREE_WARN_BYTES:
        logger.warning(
            "free disk %.1f GB < %.1f GB threshold; large extraction may fail",
            free / 1024**3,
            DISK_FREE_WARN_BYTES / 1024**3,
        )


def _copy_top_level_asset_keys(cfg: dict[str, Any], stage_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(stage_cfg)
    for key in (
        "num_points",
        "num_views",
        "img_size",
        "render_backend",
        "blender_style",
        "blender_bin",
        "blender_device",
        "step_mesh_backend",
        "step_mesh_fallback_backends",
        "step_mesh_work_dir",
        "step_mesh_format",
        "freecad_cmd",
        "step_mesh_timeout_s",
    ):
        if key in cfg and key not in out:
            out[key] = cfg[key]
    return out


def _fail_stage(layout, state: RunState, logger, stage: str, exc: BaseException) -> int:
    failure_msg = f"{stage} crashed: {exc}\n{traceback.format_exc()}"
    state.stages[stage] = _stage_record(stage, "failed", ended_at=now_iso(), error=str(exc))
    logger.error(failure_msg)
    state.ended_at = now_iso()
    state.final_status = "failed"
    _save_run_state(layout, state)
    return 1


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ABC dataset batch generation pipeline.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("--abc-root", type=Path, default=None, help="Override yaml abc_root")
    parser.add_argument("--work-root", type=Path, default=None, help="Override yaml work_root")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-pair", action="store_true")
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--skip-occlusion", action="store_true")
    parser.add_argument("--skip-description", action="store_true")
    parser.add_argument("--only-assets", action="store_true", help="Run pair + assets only; skip extract/occlusion/description")
    parser.add_argument("--only-occlusion", action="store_true", help="Run occlusion only from existing assets")
    parser.add_argument("--only-description", action="store_true", help="Run description only from existing assets")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-total", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--staging-root", type=Path, default=None)
    parser.add_argument("--staged-manifest", type=Path, default=None)
    parser.add_argument("--require-ofs", action="store_true")
    parser.add_argument("--require-ofs-for-assets", action="store_true")
    parser.add_argument("--require-ofs-for-describe", action="store_true")
    parser.add_argument("--extract-step-only", action="store_true")
    parser.add_argument("--extract-ofs-only", action="store_true")
    return parser.parse_args(argv)


def _apply_only_modes(args: argparse.Namespace) -> Optional[str]:
    only_modes = [args.only_assets, args.only_occlusion, args.only_description]
    if sum(bool(v) for v in only_modes) > 1:
        return "only one of --only-assets/--only-occlusion/--only-description can be set"
    if args.only_assets:
        args.skip_extract = True
        args.skip_occlusion = True
        args.skip_description = True
    elif args.only_occlusion:
        args.skip_extract = True
        args.skip_pair = True
        args.skip_assets = True
        args.skip_description = True
    elif args.only_description:
        args.skip_extract = True
        args.skip_pair = True
        args.skip_assets = True
        args.skip_occlusion = True
    return None


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    mode_error = _apply_only_modes(args)
    if mode_error:
        print(mode_error, file=sys.stderr)
        return 2
    if not args.config.exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2

    cfg = _load_yaml(args.config)
    abc_root = Path(args.abc_root or cfg.get("abc_root") or "").expanduser()
    work_root = Path(args.work_root or cfg.get("work_root") or "").expanduser()
    if not abc_root or str(abc_root) == ".":
        print("abc_root must be set in yaml or via --abc-root", file=sys.stderr)
        return 2
    if not work_root or str(work_root) == ".":
        print("work_root must be set in yaml or via --work-root", file=sys.stderr)
        return 2
    abc_root = abc_root.resolve()
    work_root = work_root.resolve()

    layout = work_root_layout(work_root)
    layout.ensure_dirs()
    state = RunState(
        started_at=now_iso(),
        config_path=str(args.config.resolve()),
        work_root=str(work_root),
        abc_root=str(abc_root),
        stages={
            "extract": _stage_record("extract", "pending"),
            "pair": _stage_record("pair", "pending"),
            "assets": _stage_record("assets", "pending"),
            "occlusion": _stage_record("occlusion", "pending"),
            "describe": _stage_record("describe", "pending"),
        },
    )
    _save_run_state(layout, state)

    log_path = work_root / "run_abc_batch.log"
    with stage_logger(LOGGER_NAME, log_path) as logger:
        logger.info("config=%s abc_root=%s work_root=%s", args.config, abc_root, work_root)
        _disk_check(work_root, logger)

        assets_cfg_raw = cfg.get("assets", {}) or {}
        qwen_cfg_raw = cfg.get("qwen", {}) or {}
        require_ofs_for_assets = bool(
            args.require_ofs
            or args.require_ofs_for_assets
            or cfg.get("require_ofs_for_assets", False)
            or assets_cfg_raw.get("require_ofs_for_assets", False)
        )
        require_ofs_for_describe = bool(
            args.require_ofs_for_describe
            or cfg.get("require_ofs_for_describe", False)
            or qwen_cfg_raw.get("require_ofs_for_describe", False)
        )
        logger.info(
            "ofs policy: require_for_assets=%s require_for_describe=%s legacy_require_ofs_cli=%s",
            require_ofs_for_assets,
            require_ofs_for_describe,
            args.require_ofs,
        )

        try:
            if args.skip_extract or cfg.get("skip_extract", False):
                state.stages["extract"] = _stage_record("extract", "skipped", started_at=now_iso(), ended_at=now_iso())
            else:
                t0 = time.time()
                sides = ["step"] if args.extract_step_only else ["ofs"] if args.extract_ofs_only else ["step", "ofs"]
                summary = run_extract(
                    abc_root=abc_root,
                    work_root=work_root,
                    sides=sides,
                    max_workers=int(cfg.get("max_extract_workers", min(8, os.cpu_count() or 1))),
                    logger=logger,
                )
                state.stages["extract"] = _stage_record(
                    "extract",
                    "done" if summary["n_failures_this_run"] == 0 else "done_with_failures",
                    started_at=now_iso(),
                    ended_at=now_iso(),
                    elapsed_s=round(time.time() - t0, 2),
                    extra={
                        "n_failures": summary["n_failures_this_run"],
                        "step": summary["step"],
                        "ofs": summary["ofs"],
                    },
                )
            _save_run_state(layout, state)
        except BaseException as exc:  # noqa: BLE001
            return _fail_stage(layout, state, logger, "extract", exc)

        try:
            if args.skip_pair or cfg.get("skip_pair", False):
                state.stages["pair"] = _stage_record("pair", "skipped", started_at=now_iso(), ended_at=now_iso())
                if not (args.skip_assets or cfg.get("skip_assets", False)):
                    if not (layout.pairing_manifest.exists() or layout.step_only_manifest.exists()):
                        raise RuntimeError(
                            "pair skipped but no pairing or step-only manifest exists: "
                            f"{layout.pairing_manifest}, {layout.step_only_manifest}"
                        )
            else:
                t0 = time.time()
                pair_cfg = cfg.get("pair", {}) or {}
                offset = args.offset if args.offset is not None else int(pair_cfg.get("offset", 0))
                limit = args.limit if args.limit is not None else pair_cfg.get("limit")
                shard_index = args.shard_index if args.shard_index is not None else pair_cfg.get("shard_index")
                shard_total = args.shard_total if args.shard_total is not None else pair_cfg.get("shard_total")
                require_ofs = require_ofs_for_assets or bool(pair_cfg.get("require_ofs", False))
                scan_ofs = bool(require_ofs or require_ofs_for_describe or pair_cfg.get("scan_ofs", False))
                summary = run_pair(
                    layout,
                    offset=offset,
                    limit=limit,
                    shard_index=shard_index,
                    shard_total=shard_total,
                    require_ofs=require_ofs,
                    logger=logger,
                    scan_ofs=scan_ofs,
                )
                state.stages["pair"] = _stage_record(
                    "pair",
                    "done",
                    started_at=now_iso(),
                    ended_at=now_iso(),
                    elapsed_s=round(time.time() - t0, 2),
                    extra={
                        "n_step": summary["n_step"],
                        "n_ofs": summary["n_ofs"],
                        "n_paired": summary["n_paired"],
                        "n_paired_written": summary["n_paired_written"],
                        "n_step_only_written": summary["n_step_only_written"],
                        "n_step_only_diagnostic_written": summary.get("n_step_only_diagnostic_written", 0),
                        "diagnostic_status": summary.get("diagnostic_status"),
                        "manifest_path": summary.get("manifest_path"),
                        "step_only_manifest_path": summary.get("step_only_manifest_path"),
                        "orphan_ofs_path": summary.get("orphan_ofs_path"),
                        "selection": summary.get("selection", {}),
                        "ofs_policy": {
                            "require_ofs_for_assets": require_ofs_for_assets,
                            "require_ofs_for_describe": require_ofs_for_describe,
                            "scan_ofs": scan_ofs,
                        },
                    },
                )
            _save_run_state(layout, state)
        except BaseException as exc:  # noqa: BLE001
            return _fail_stage(layout, state, logger, "pair", exc)

        assets_cfg = _copy_top_level_asset_keys(cfg, assets_cfg_raw)
        assets_cfg["require_ofs_for_assets"] = require_ofs_for_assets
        if args.batch_id:
            assets_cfg["batch_id"] = args.batch_id
        if args.staging_root:
            assets_cfg["staging_root"] = str(args.staging_root)
        if args.staged_manifest:
            assets_cfg["staged_manifest"] = str(args.staged_manifest)
        res_assets = run_assets_stage(
            assets_cfg,
            work_root,
            skip=args.skip_assets or cfg.get("skip_assets", False),
            logger=logger,
        )
        state.stages["assets"] = _stage_dict_from_result(res_assets)
        _save_run_state(layout, state)
        if res_assets.status not in ("done", "skipped"):
            state.ended_at = now_iso()
            state.final_status = "failed"
            _save_run_state(layout, state)
            return 1

        skip_occlusion = args.skip_occlusion or cfg.get("skip_occlusion", False)
        if skip_occlusion:
            state.stages["occlusion"] = _stage_record("occlusion", "skipped", started_at=now_iso(), ended_at=now_iso())
        else:
            from .abc_batch.stage_occlusion import run_occlusion_stage

            occlusion_cfg = _copy_top_level_asset_keys(cfg, cfg.get("occlusion", {}) or {})
            res_occ = run_occlusion_stage(occlusion_cfg, work_root, skip=False, logger=logger)
            state.stages["occlusion"] = _stage_dict_from_result(res_occ)
            if res_occ.status not in ("done", "skipped"):
                state.ended_at = now_iso()
                state.final_status = "failed"
                _save_run_state(layout, state)
                return 1
        _save_run_state(layout, state)

        skip_description = args.skip_description or cfg.get("skip_description", False)
        if skip_description:
            state.stages["describe"] = _stage_record("describe", "skipped", started_at=now_iso(), ended_at=now_iso())
        else:
            from .abc_batch.stage_describe import run_describe_stage

            describe_cfg = dict(qwen_cfg_raw)
            if "model" in cfg and "model" not in describe_cfg:
                describe_cfg["model"] = cfg["model"]
            if "api_base" in cfg and "api_base" not in describe_cfg:
                describe_cfg["api_base"] = cfg["api_base"]
            describe_cfg["require_ofs_for_describe"] = require_ofs_for_describe
            res_desc = run_describe_stage(describe_cfg, work_root, skip=False, logger=logger)
            state.stages["describe"] = _stage_dict_from_result(res_desc)
            if res_desc.status not in ("done", "skipped"):
                state.ended_at = now_iso()
                state.final_status = "failed"
                _save_run_state(layout, state)
                return 1
        _save_run_state(layout, state)

        n_extracted = sum(
            state.stages["extract"].get("extra", {}).get(side, {}).get("n_files_total", 0)
            for side in ("step", "ofs")
        )
        final_summary = {
            "ts": now_iso(),
            "abc_root": str(abc_root),
            "work_root": str(work_root),
            "n_extracted_files": n_extracted,
            "n_paired": state.stages["pair"].get("extra", {}).get("n_paired_written", 0),
            "n_assets_ok": state.stages["assets"].get("extra", {}).get("n_assets_records", 0),
            "n_assets_with_ofs": state.stages["assets"].get("extra", {}).get("n_assets_with_ofs", 0),
            "n_assets_without_ofs": state.stages["assets"].get("extra", {}).get("n_assets_without_ofs", 0),
            "n_occlusion_ok": state.stages["occlusion"].get("extra", {}).get("n_occlusion_records", 0),
            "n_descriptions_ok": state.stages["describe"].get("extra", {}).get("n_descriptions", 0),
            "ofs_policy": {
                "require_ofs_for_assets": require_ofs_for_assets,
                "require_ofs_for_describe": require_ofs_for_describe,
            },
            "stages": state.stages,
        }
        write_json(layout.final_summary, final_summary)
        logger.info("final_summary written to %s", layout.final_summary)

    state.ended_at = now_iso()
    state.final_status = "done"
    _save_run_state(layout, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
