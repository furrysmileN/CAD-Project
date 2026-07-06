"""遮挡阶段封装：调用 `cad_data_gen.build_occlusion_assets`（occluder 模式）。

输入：

* `<work-root>/assets/manifest.jsonl`（来自基础资产阶段）
* `<work-root>/assets_input/`（基础资产阶段使用的 mirror dir，即 manifest 中相对
  `step_path` 的解析根 = `--step-root`）

输出：`<work-root>/occlusion/`（manifest.jsonl / audit.jsonl / summary.json 等
全部由底层脚本落地，外层不二次包装）。

注意：本批跑只生成「遮挡板」增强（occluder 模式），与「光照/相机抖动」类增强
相互独立——不在外层叠加光照或相机抖动开关 [[memory:1q7u8y82]]。
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from .logging_utils import iter_jsonl, now_iso, stage_logger
from .paths import work_root_layout
from .stage_assets import StageResult

LOGGER_NAME = "stage_occlusion"


def _build_cmd(
    manifest_path: Path,
    source_assets_dir: Path,
    step_root: Path,
    output_dir: Path,
    cfg: dict,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "cad_data_gen.build_occlusion_assets",
        "--manifest",
        str(manifest_path),
        "--source-assets-dir",
        str(source_assets_dir),
        "--step-root",
        str(step_root),
        "--output-dir",
        str(output_dir),
    ]
    # 默认值（与 cad_shape_1000_occlusion_* 一致）
    mode = cfg.get("mode", "occluder")
    target_dims = cfg.get("target_dims", "point_cloud,image")
    num_views = cfg.get("num_views", 8)
    img_size = cfg.get("img_size", 512)
    render_backend = cfg.get("render_backend", "blender-step")
    blender_style = cfg.get("blender_style", "visualization")
    variants = cfg.get("variants_per_sample", 1)
    cmd += [
        "--mode",
        str(mode),
        "--target-dims",
        str(target_dims),
        "--num-views",
        str(num_views),
        "--img-size",
        str(img_size),
        "--render-backend",
        str(render_backend),
        "--blender-style",
        str(blender_style),
        "--variants-per-sample",
        str(variants),
    ]
    if cfg.get("blender_bin"):
        cmd += ["--blender-bin", str(cfg["blender_bin"])]
    if cfg.get("visualization_root"):
        cmd += ["--visualization-root", str(cfg["visualization_root"])]
    if cfg.get("blender_script"):
        cmd += ["--blender-script", str(cfg["blender_script"])]
    if "num_processes" in cfg:
        cmd += ["--num-processes", str(cfg["num_processes"])]
    if "seed" in cfg:
        cmd += ["--seed", str(cfg["seed"])]
    if cfg.get("skip_existing", True):
        cmd += ["--skip-existing"]
    if cfg.get("append_manifest", True):
        cmd += ["--append-manifest"]

    # 透传若干 occluder 大小参数（与底层 CLI 一致）
    for key in (
        "foreground_occluder_size_min",
        "foreground_occluder_size_max",
        "foreground_occluder_depth",
        "size_min",
        "size_max",
        "min_removed_ratio",
        "max_removed_ratio",
        "occluder_color",
        "blender_engine",
        "blender_samples",
        "blender_device",
        "step_mesh_backend",
        "step_mesh_fallback_backends",
        "step_mesh_work_dir",
        "step_mesh_format",
        "freecad_cmd",
        "step_mesh_timeout_s",
        "limit",
        "num_points",
    ):
        if key in cfg and cfg[key] is not None:
            cmd += [f"--{key.replace('_', '-')}", str(cfg[key])]

    return cmd


def run_occlusion_stage(
    cfg: dict,
    work_root: str | Path,
    *,
    skip: bool = False,
    logger: Optional[logging.Logger] = None,
) -> StageResult:
    layout = work_root_layout(work_root)
    layout.ensure_dirs()

    log_path = layout.stage_occlusion_log
    owns_logger = logger is None
    cm = stage_logger(LOGGER_NAME, log_path) if owns_logger else None
    log = cm.__enter__() if cm is not None else logger
    assert log is not None

    res = StageResult(stage="occlusion", status="pending", started_at=now_iso())
    started_t = time.time()
    try:
        if skip:
            log.info("--skip-occlusion set; skipping")
            res.status = "skipped"
            return res

        if not layout.assets_manifest.exists():
            res.status = "failed"
            res.error = f"assets manifest missing: {layout.assets_manifest}"
            log.error(res.error)
            return res

        # mirror dir 是 build_step_assets 的 --input-dir，等于 manifest 中相对 step_path 的根
        mirror_root = layout.work_root / "assets_input"
        if not mirror_root.exists():
            res.status = "failed"
            res.error = (
                f"assets_input mirror missing: {mirror_root}; "
                "occlusion stage relies on build_step_assets's input mirror"
            )
            log.error(res.error)
            return res

        cmd = _build_cmd(
            manifest_path=layout.assets_manifest,
            source_assets_dir=layout.assets_dir,
            step_root=mirror_root,
            output_dir=layout.occlusion_dir,
            cfg=cfg,
        )
        res.cmd = cmd
        res.n_inputs = sum(1 for _ in iter_jsonl(layout.assets_manifest))
        existing_manifest = layout.occlusion_dir / "manifest.jsonl"
        n_existing = sum(1 for _ in iter_jsonl(existing_manifest)) if existing_manifest.exists() else 0
        res.extra["n_source_assets"] = res.n_inputs
        res.extra["n_existing_occlusion_records"] = n_existing
        res.extra["augmentation_mode"] = cfg.get("mode", "occluder")
        res.extra["occlusion_separate_from_lighting_and_camera"] = True
        log.info(
            "occlusion input: source_assets=%d existing_occlusion_records=%d step_root=%s",
            res.n_inputs,
            n_existing,
            mirror_root,
        )
        log.info("running: %s", " ".join(cmd))

        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(f"\n=== {now_iso()} === occlusion stage cmd: {' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(cmd, stdout=logf, stderr=logf)
        res.return_code = proc.returncode
        if proc.returncode != 0:
            res.status = "failed"
            res.error = f"build_occlusion_assets exited with {proc.returncode}; see {log_path}"
            log.error(res.error)
            return res

        out_manifest = layout.occlusion_dir / "manifest.jsonl"
        if not out_manifest.exists():
            res.status = "failed"
            res.error = f"occlusion/manifest.jsonl not produced at {out_manifest}"
            log.error(res.error)
            return res

        n_out = sum(1 for _ in iter_jsonl(out_manifest))
        res.extra["n_occlusion_records"] = n_out
        res.status = "done"
        log.info("occlusion stage done: %d records in %s", n_out, out_manifest)
        return res
    except KeyboardInterrupt:
        res.status = "failed"
        res.error = "interrupted"
        log.warning("occlusion stage interrupted")
        return res
    except BaseException as exc:  # noqa: BLE001
        res.status = "failed"
        res.error = f"{type(exc).__name__}: {exc}"
        log.error("occlusion stage crashed: %s\n%s", exc, traceback.format_exc())
        return res
    finally:
        res.ended_at = now_iso()
        res.elapsed_s = round(time.time() - started_t, 2)
        if cm is not None:
            cm.__exit__(None, None, None)


# ---------- CLI ----------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run occlusion stage: build_occlusion_assets in occluder mode."
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--mode", default="occluder")
    parser.add_argument("--target-dims", default="point_cloud,image")
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--render-backend", default="blender-step")
    parser.add_argument("--blender-style", default="visualization")
    parser.add_argument("--variants-per-sample", type=int, default=1)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blender-bin", default=None)
    parser.add_argument("--visualization-root", default=None)
    parser.add_argument("--blender-script", default=None)
    parser.add_argument("--skip-occlusion", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg: dict[str, Any] = {
        "mode": args.mode,
        "target_dims": args.target_dims,
        "num_views": args.num_views,
        "img_size": args.img_size,
        "render_backend": args.render_backend,
        "blender_style": args.blender_style,
        "variants_per_sample": args.variants_per_sample,
        "num_processes": args.num_processes,
        "seed": args.seed,
        "num_points": args.num_points,
    }
    if args.blender_bin:
        cfg["blender_bin"] = args.blender_bin
    if args.visualization_root:
        cfg["visualization_root"] = args.visualization_root
    if args.blender_script:
        cfg["blender_script"] = args.blender_script

    res = run_occlusion_stage(cfg, args.work_root, skip=args.skip_occlusion)
    if res.status in ("done", "skipped"):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
