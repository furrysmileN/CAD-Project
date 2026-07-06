"""归档后 Qwen/API 文本生成入口。

该入口面向真实环境：基础资产和上下文已经归档到最终数据盘，真实环境只负责
读取归档批次目录中的 `contexts/compact_contexts.jsonl` 并调用 Qwen API，输出到
`descriptions/`，不再运行点云、渲染或遮挡生成。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .logging_utils import iter_jsonl, now_iso, stage_logger, write_json
from .stage_describe import DEFAULT_API_BASE_CN, DEFAULT_API_KEY_ENV, DEFAULT_MODEL_CN

LOGGER_NAME = "post_archive_describe"


@dataclass
class PostArchiveDescribeResult:
    stage: str = "post_archive_describe"
    status: str = "pending"
    started_at: str = ""
    ended_at: str = ""
    elapsed_s: float = 0.0
    batch_dir: str = ""
    contexts_path: str = ""
    assets_manifest_path: str = ""
    descriptions_dir: str = ""
    cmd: list[str] = field(default_factory=list)
    return_code: Optional[int] = None
    n_contexts: int = 0
    n_existing_descriptions: int = 0
    n_descriptions: int = 0
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_api_key_file(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _check_api_key(*, env_name: str, api_key_file: Optional[Path]) -> Optional[str]:
    if os.environ.get(env_name):
        return None
    if api_key_file and api_key_file.exists() and api_key_file.stat().st_size > 0:
        return None
    return f"Qwen API key not found: env ${env_name} is empty and api_key_file={api_key_file} is unavailable"


def _count_jsonl(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path)) if path.exists() else 0


def _validate_context_paths(contexts_path: Path) -> dict[str, int]:
    stats = {
        "n_contexts": 0,
        "n_missing_image_paths": 0,
        "n_missing_point_paths": 0,
        "n_with_ofs": 0,
        "n_without_ofs": 0,
    }
    for row in iter_jsonl(contexts_path):
        stats["n_contexts"] += 1
        has_ofs = bool(row.get("has_ofs") or row.get("ofs_path"))
        if has_ofs:
            stats["n_with_ofs"] += 1
        else:
            stats["n_without_ofs"] += 1
        point_path = row.get("point_path")
        if point_path and not Path(str(point_path)).exists():
            stats["n_missing_point_paths"] += 1
        for image_path in row.get("image_paths") or []:
            if image_path and not Path(str(image_path)).exists():
                stats["n_missing_image_paths"] += 1
    return stats


def _build_describe_cmd(
    *,
    contexts_path: Path,
    output_dir: Path,
    api_base: str,
    model: str,
    api_key_env: str,
    api_key_file: Optional[Path],
    cfg: dict[str, Any],
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "cad_data_gen.batch_describe_cad_with_qwen",
        "--contexts",
        str(contexts_path),
        "--output-dir",
        str(output_dir),
        "--api-base",
        api_base,
        "--model",
        model,
        "--api-key-env",
        api_key_env,
    ]
    if api_key_file:
        cmd += ["--api-key-file", str(api_key_file)]
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
    if cfg.get("resume", True):
        cmd += ["--resume"]
    if cfg.get("write_text_files"):
        cmd += ["--write-text-files"]
    if cfg.get("save_raw_batch_responses"):
        cmd += ["--save-raw-batch-responses"]
    return cmd


def run_post_archive_describe(
    *,
    batch_dir: Path,
    api_base: str = DEFAULT_API_BASE_CN,
    model: str = DEFAULT_MODEL_CN,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    api_key_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    cfg: Optional[dict[str, Any]] = None,
) -> PostArchiveDescribeResult:
    started_t = time.time()
    batch = batch_dir.expanduser().resolve()
    contexts_path = batch / "contexts" / "compact_contexts.jsonl"
    assets_manifest = batch / "assets" / "manifest.jsonl"
    descriptions_dir = output_dir.expanduser().resolve() if output_dir else batch / "descriptions"
    log_path = batch / "post_archive_describe.log"
    config = dict(cfg or {})
    key_file = _resolve_api_key_file(str(api_key_file) if api_key_file else None)

    res = PostArchiveDescribeResult(
        started_at=now_iso(),
        batch_dir=str(batch),
        contexts_path=str(contexts_path),
        assets_manifest_path=str(assets_manifest),
        descriptions_dir=str(descriptions_dir),
    )

    with stage_logger(LOGGER_NAME, log_path) as logger:
        try:
            if not batch.is_dir():
                raise FileNotFoundError(f"archived batch dir not found: {batch}")
            if not assets_manifest.is_file():
                raise FileNotFoundError(f"archived assets manifest not found: {assets_manifest}")
            if not contexts_path.is_file():
                raise FileNotFoundError(f"archived compact contexts not found: {contexts_path}")
            err = _check_api_key(env_name=api_key_env, api_key_file=key_file)
            if err:
                raise RuntimeError(err)

            descriptions_dir.mkdir(parents=True, exist_ok=True)
            res.n_contexts = _count_jsonl(contexts_path)
            existing = descriptions_dir / "descriptions.jsonl"
            res.n_existing_descriptions = _count_jsonl(existing)
            res.extra.update(_validate_context_paths(contexts_path))
            res.cmd = _build_describe_cmd(
                contexts_path=contexts_path,
                output_dir=descriptions_dir,
                api_base=api_base,
                model=model,
                api_key_env=api_key_env,
                api_key_file=key_file,
                cfg=config,
            )
            logger.info("running post-archive describe: %s", " ".join(res.cmd))
            with log_path.open("a", encoding="utf-8") as logf:
                logf.write(f"\n=== {now_iso()} === post archive describe cmd: {' '.join(res.cmd)}\n")
                logf.flush()
                proc = subprocess.run(res.cmd, stdout=logf, stderr=logf)
            res.return_code = proc.returncode
            if proc.returncode != 0:
                raise RuntimeError(f"batch_describe_cad_with_qwen exited with {proc.returncode}; see {log_path}")
            res.n_descriptions = _count_jsonl(existing)
            res.status = "done"
            write_json(descriptions_dir / "post_archive_describe_state.json", res.to_dict())
            return res
        except BaseException as exc:  # noqa: BLE001
            res.status = "failed"
            res.error = f"{type(exc).__name__}: {exc}"
            logger.error(res.error)
            write_json(descriptions_dir / "post_archive_describe_state.json", res.to_dict())
            return res
        finally:
            res.ended_at = now_iso()
            res.elapsed_s = round(time.time() - started_t, 2)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen/API description on an archived batch directory.")
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE_CN)
    parser.add_argument("--model", default=DEFAULT_MODEL_CN)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-images-per-sample", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg: dict[str, Any] = {"retries": args.retries, "resume": not args.no_resume}
    for key in ("temperature", "max_tokens", "timeout", "batch_size", "max_images_per_sample", "limit", "offset"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    res = run_post_archive_describe(
        batch_dir=args.batch_dir,
        api_base=args.api_base,
        model=args.model,
        api_key_env=args.api_key_env,
        api_key_file=args.api_key_file,
        output_dir=args.output_dir,
        cfg=cfg,
    )
    return 0 if res.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
