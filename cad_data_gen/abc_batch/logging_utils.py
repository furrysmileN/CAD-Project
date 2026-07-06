"""abc_batch 共享日志/落盘工具。

集中实现 jsonl/json 的原子追加与写入，所有阶段通过这里统一格式。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


def append_jsonl(path: str | Path, obj: Mapping[str, Any]) -> None:
    """原子追加一条 jsonl 记录。

    使用 `open(..., 'a')` 单 line write，配合各底层脚本一致的 append-mode 续跑语义。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, sort_keys=False)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    """将对象覆盖式写入 JSON 文件（带父目录创建）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, sort_keys=False)


def read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # 容错：跳过损坏行
                continue


@contextmanager
def stage_logger(stage_name: str, log_path: str | Path) -> Iterator[logging.Logger]:
    """构造一个临时 logger，同时写到 stderr 与指定 log 文件。

    退出时自动 flush 并移除 handler，避免不同阶段共用同一个 logger 时句柄叠加。
    """
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"abc_batch.{stage_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_h = logging.FileHandler(p, mode="a", encoding="utf-8")
    file_h.setFormatter(fmt)
    stream_h = logging.StreamHandler(sys.stderr)
    stream_h.setFormatter(fmt)

    logger.addHandler(file_h)
    logger.addHandler(stream_h)

    try:
        yield logger
    finally:
        for h in (file_h, stream_h):
            h.flush()
            logger.removeHandler(h)
            h.close()


def now_iso() -> str:
    """ISO8601 本地时间戳（秒级）。"""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def safe_disk_usage(path: str | Path) -> tuple[int, int, int] | None:
    """返回 (total, used, free)；目录不存在或异常时返回 None。"""
    try:
        usage = os.statvfs(str(path))
    except (FileNotFoundError, OSError):
        return None
    total = usage.f_blocks * usage.f_frsize
    free = usage.f_bavail * usage.f_frsize
    used = total - free
    return total, used, free


def truncate_str(s: str, n: int = 800) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"... <truncated {len(s) - n} chars>"
