"""Qwen 文本描述阶段封装。

两步子进程：

1. `cad_data_gen.clean_cad_contexts`：从 `assets/manifest.jsonl` + `extracted/ofs/`
   生成 `contexts/compact_contexts.jsonl`。
2. `cad_data_gen.batch_describe_cad_with_qwen`：调用 Qwen API，输出
   `descriptions/{descriptions.jsonl,failures.jsonl,batches.jsonl}`。

外层强制 CLI 透传：

* `--api-base https://dashscope.aliyuncs.com/compatible-mode/v1`（国内地址）
* `--model qwen3.7-plus`
* `--api-key-file <repo>/.secrets/qwen_api_key`（默认）

启动时立即校验 API Key 是否存在；缺失则直接报错退出，不等到第一次 HTTP 失败。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from .logging_utils import iter_jsonl, now_iso, stage_logger
from .paths import work_root_layout
from .stage_assets import StageResult

LOGGER_NAME = "stage_describe"

DEFAULT_API_BASE_CN = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL_CN = "qwen3.7-plus"
DEFAULT_API_KEY_ENV = "QWEN_API_KEY"


def _resolve_repo_root() -> Path:
    """`<repo>/cad_data_gen/` 根目录（用于定位 `.secrets/qwen_api_key`）。"""
    # __file__ = .../cad_data_gen/src/cad_data_gen/abc_batch/stage_describe.py
    return Path(__file__).resolve().parents[3]


def _resolve_api_key_file(cfg: dict) -> Path:
    if cfg.get("api_key_file"):
        return Path(cfg["api_key_file"]).expanduser().resolve()
    return _resolve_repo_root() / ".secrets" / "qwen_api_key"


def _check_api_key(cfg: dict, logger: logging.Logger) -> Optional[str]:
    """校验 API Key 是否可获得；返回错误描述或 None。"""
    env_name = cfg.get("api_key_env", DEFAULT_API_KEY_ENV)
    if os.environ.get(env_name):
        logger.info("Qwen API key picked up from env $%s", env_name)
        return None
    key_file = _resolve_api_key_file(cfg)
    if key_file.exists() and key_file.stat().st_size > 0:
        logger.info("Qwen API key file: %s", key_file)
        return None
    return (
        f"Qwen API key not found: env ${env_name} is empty and key file {key_file} "
        "does not exist or is empty"
    )


def _build_clean_cmd(
    work_root_layout_obj,
    cfg: dict,
) -> list[str]:
    layout = work_root_layout_obj
    cmd: list[str] = [
        sys.executable,
        "-m",
        "cad_data_gen.clean_cad_contexts",
        "--input-dir",
        str(layout.work_root / "assets_input"),
        "--output",
        str(layout.compact_contexts),
        "--manifest",
        str(layout.assets_manifest),
        "--render-root",
        str(layout.assets_dir),
        "--recursive",
    ]
    if cfg.get("use_ofs_context", True):
        cmd += ["--ofs-dir", str(layout.extracted_ofs_root)]
    if cfg.get("context_mode"):
        cmd += ["--context-mode", str(cfg["context_mode"])]
    if cfg.get("compact_max_ofs_features") is not None:
        cmd += ["--compact-max-ofs-features", str(cfg["compact_max_ofs_features"])]
    if cfg.get("max_images") is not None:
        cmd += ["--max-images", str(cfg["max_images"])]
    if cfg.get("skip_mesh_metrics"):
        cmd += ["--skip-mesh-metrics"]
    if cfg.get("resume_clean", True):
        cmd += ["--resume"]
    return cmd


def _build_describe_cmd(
    work_root_layout_obj,
    cfg: dict,
) -> list[str]:
    layout = work_root_layout_obj
    cmd: list[str] = [
        sys.executable,
        "-m",
        "cad_data_gen.batch_describe_cad_with_qwen",
        "--contexts",
        str(layout.compact_contexts),
        "--output-dir",
        str(layout.descriptions_dir),
        "--api-base",
        str(cfg.get("api_base", DEFAULT_API_BASE_CN)),
        "--model",
        str(cfg.get("model", DEFAULT_MODEL_CN)),
        "--api-key-env",
        str(cfg.get("api_key_env", DEFAULT_API_KEY_ENV)),
        "--api-key-file",
        str(_resolve_api_key_file(cfg)),
    ]
    # 透传可选参数
    for key in (
        "temperature",
        "max_tokens",
        "timeout",
        "retries",
        "batch_size",
        "max_images_per_sample",
        "limit",
        "offset",
    ):
        if key in cfg and cfg[key] is not None:
            cmd += [f"--{key.replace('_', '-')}", str(cfg[key])]
    if cfg.get("resume_describe", True):
        cmd += ["--resume"]
    if cfg.get("write_text_files"):
        cmd += ["--write-text-files"]
    if cfg.get("save_raw_batch_responses"):
        cmd += ["--save-raw-batch-responses"]
    return cmd


def _filter_contexts_for_ofs(context_path: Path, logger: logging.Logger) -> dict[str, int]:
    tmp_path = context_path.with_suffix(context_path.suffix + ".tmp")
    n_total = 0
    n_with_ofs = 0
    n_without_ofs = 0
    with context_path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            n_total += 1
            has_ofs = bool(row.get("has_ofs") or row.get("ofs_path"))
            if has_ofs:
                n_with_ofs += 1
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                n_without_ofs += 1
    tmp_path.replace(context_path)
    logger.info(
        "filtered contexts for require_ofs_for_describe=true: total=%d kept_with_ofs=%d dropped_without_ofs=%d",
        n_total,
        n_with_ofs,
        n_without_ofs,
    )
    return {
        "n_contexts_before_filter": n_total,
        "n_contexts_with_ofs": n_with_ofs,
        "n_contexts_without_ofs": n_without_ofs,
    }


def _context_level_stats(context_path: Path) -> dict[str, int]:
    stats = {
        "n_contexts_with_ofs": 0,
        "n_contexts_without_ofs": 0,
        "n_contexts_visual_geometry_with_ofs": 0,
        "n_contexts_visual_geometry_only": 0,
    }
    for row in iter_jsonl(context_path):
        has_ofs = bool(row.get("has_ofs") or row.get("ofs_path"))
        if has_ofs:
            stats["n_contexts_with_ofs"] += 1
        else:
            stats["n_contexts_without_ofs"] += 1
        level = row.get("description_context_level")
        if level == "visual_geometry_with_ofs":
            stats["n_contexts_visual_geometry_with_ofs"] += 1
        elif level == "visual_geometry_only":
            stats["n_contexts_visual_geometry_only"] += 1
    return stats


def run_describe_stage(
    cfg: dict,
    work_root: str | Path,
    *,
    skip: bool = False,
    logger: Optional[logging.Logger] = None,
) -> StageResult:
    layout = work_root_layout(work_root)
    layout.ensure_dirs()

    log_path = layout.stage_describe_log
    owns_logger = logger is None
    cm = stage_logger(LOGGER_NAME, log_path) if owns_logger else None
    log = cm.__enter__() if cm is not None else logger
    assert log is not None

    res = StageResult(stage="describe", status="pending", started_at=now_iso())
    started_t = time.time()
    try:
        if skip:
            log.info("--skip-description set; skipping")
            res.status = "skipped"
            return res

        if not layout.assets_manifest.exists():
            res.status = "failed"

t}"
            log.error(res.error)
            return res
        require_ofs_for_describe = bool(cfg.get("require_ofs_for_describ
e", False))
        if require_ofs_for_describe:
            cfg["use_ofs_context"] = True
        log.info("describe OFS policy: require_ofs_for_describe=%s use_ofs_con
text=%s",
            require_ofs_for_describe,
            cfg.get("use_ofs_context", True),
        )

        # 提前校验 API Key
        err = _check_api_key(cfg, log)
        if err:
            res.status = "failed"
            res.error = err
            log.error(res.error)
            return res

        # Step A: clean_cad_contexts
        clean_cmd = _build_clean_cmd(layout, cfg)
        res.cmd = list(clean_cmd)
        log.info("running clean: %s", " ".join(clean_cmd))
        with open(log_path, "a", encoding="utf-8") as logf:logf.write(f"\n=== {now_iso()} === clean_cad_contexts cmd:
{' '.join(clean_cmd)}\n")
            logf.flush()proc_clean = subprocess.run(clean_cmd, stdout=logf, stderr=lo
gf)
        if proc_clean.returncode != 0:
            res.return_code = proc_clean.returncode
            res.status = "failed"res.error = f"clean_cad_contexts exited with {proc_clean.retu
rncode}; see {log_path}"
            log.error(res.error)
            return res
        if not layout.compact_contexts.exists():
            res.status = "failed"res.error = f"compact_contexts not produced at {layout.compac
t_contexts}"
            log.error(res.error)
            return res
        n_contexts = sum(1 for _ in iter_jsonl(layout.compact_contexts))
        res.extra["n_contexts"] = n_contexts
        if require_ofs_for_describe:res.extra.update(_filter_contexts_for_ofs(layout.compact_cont
exts, log))n_contexts = sum(1 for _ in iter_jsonl(layout.compact_context
s))
            res.extra["n_contexts"] = n_contexts
        res.extra.update(_context_level_stats(layout.compact_contexts))
        log.info("clean produced %d contexts", n_contexts)

        # Step B: batch_describe_cad_with_qwen
        describe_cmd = _build_describe_cmd(layout, cfg)
        res.cmd = list(describe_cmd)  # 覆盖为 describe 命令（更具诊断价值）
        log.info("running describe: %s", " ".join(describe_cmd))
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(f"\n=== {now_iso()} === batch_describe_cad_with_qwen cm
d: {' '.join(describe_cmd)}\n"
            )
            logf.flush()proc_desc = subprocess.run(describe_cmd, stdout=logf, stderr=
logf)
        res.return_code = proc_desc.returncode
        if proc_desc.returncode != 0:
            res.status = "failed"
            res.error = (f"batch_describe_cad_with_qwen exited with {proc_desc.ret
urncode}; see {log_path}"
            )
            log.error(res.error)
            return res

        out_descriptions = layout.descriptions_dir / "descriptions.jsonl"
        if not out_descriptions.exists():
            # 底层有可能落地为不同名称；做 best-effort 检查
            log.warning(
                "expected %s not found; check %s for actual outputs",
                out_descriptions,
                layout.descriptions_dir,
            )
            res.extra["n_descriptions"] = 0
        else:
            n_desc = sum(1 for _ in iter_jsonl(out_descriptions))
            res.extra["n_descriptions"] = n_desc
            log.info("describe produced %d descriptions", n_desc)

        res.n_inputs = n_contexts
        res.status = "done"
        return res
    except KeyboardInterrupt:
        res.status = "failed"
        res.error = "interrupted"
        log.warning("describe stage interrupted")
        return res
    except BaseException as exc:  # noqa: BLE001
        res.status = "failed"
        res.error = f"{type(exc).__name__}: {exc}"log.error("describe stage crashed: %s\n%s", exc, traceback.format
_exc())
        return res
    finally:
        res.ended_at = now_iso()
        res.elapsed_s = round(time.time() - started_t, 2)
        if cm is not None:
            cm.__exit__(None, None, None)


# ---------- CLI ----------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text description stage: clean_cad_contexts + bat
ch_describe_cad_with_qwen."
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE_CN)
    parser.add_argument("--model", default=DEFAULT_MODEL_CN)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)parser.add_argument("--max-images-per-sample", type=int, default=Non
e)
    parser.add_argument("--context-mode", default=None)parser.add_argument("--require-ofs-for-describe", action="store_tru
e")
    parser.add_argument("--no-ofs-context", action="store_true")
    parser.add_argument("--skip-description", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg: dict[str, Any] = {
        "api_base": args.api_base,
        "model": args.model,
        "api_key_env": args.api_key_env,
        "retries": args.retries,
        "require_ofs_for_describe": args.require_ofs_for_describe,
        "use_ofs_context": not args.no_ofs_context,
    }
    if args.api_key_file:
        cfg["api_key_file"] = args.api_key_file
    for key in (
        "temperature",
        "max_tokens",
        "timeout",
        "batch_size",
        "max_images_per_sample",
        "context_mode",
    ):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    res = run_describe_stage(cfg, args.work_root, skip=args.skip_descript
ion)
    if res.status in ("done", "skipped"):
        return 0
    return 1


if __name__ == "__main__":sys exit(main())
