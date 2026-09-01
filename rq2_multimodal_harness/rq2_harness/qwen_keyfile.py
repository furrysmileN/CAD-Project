"""从本地 qwenapikey.txt 注入环境变量；不把密钥写进配置或日志。"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from .common import EXPERIMENT_DIR


def apply_qwen_keyfile(path: Path | None = None) -> dict[str, object]:
    keyfile = Path(path) if path else EXPERIMENT_DIR / "qwenapikey.txt"
    if not keyfile.is_file():
        return {"loaded": False, "has_key": False, "has_base": False}
    data: dict[str, str] = {}
    label: str | None = None
    for raw in keyfile.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.endswith(":") and not line.lower().startswith("http") and not line.lower().startswith("sk-"):
            label = line[:-1].strip().lower()
            continue
        if label:
            data[label] = line
            label = None
    api_key = data.get("qwenapikey") or data.get("apikey") or ""
    base = data.get("openai兼容地址") or data.get("base_url") or ""
    if api_key and not os.environ.get("VLM_API_KEY", "").strip():
        os.environ["VLM_API_KEY"] = api_key
    if base and not os.environ.get("VLM_BASE_URL", "").strip():
        os.environ["VLM_BASE_URL"] = base
    host = urlparse(base).netloc if base else ""
    return {"loaded": True, "has_key": bool(api_key), "has_base": bool(base), "base_host": host}
