from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from .common import project_path


def run_episode(
    plan: dict[str, Any],
    config: dict[str, Any],
    *,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    """Call the backend's Python Episode API through an explicit package-root adapter."""
    backend_root = project_path(config["root"]).resolve()
    if not (backend_root / "backend" / "harness_api.py").is_file():
        raise FileNotFoundError(f"HarnessCAD backend 不存在: {backend_root}")
    root_string = str(backend_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    version = str(config.get("episode_version", "v1")).lower()
    timeout = float(config["timeout_sec"])
    if version == "v1":
        if run_root is not None:
            raise ValueError("自定义 run_root 仅支持 Episode v2")
        module = importlib.import_module("backend.harness_api")
        response = module.run_harness_plan(module.HarnessRunRequest(plan=plan, timeout_sec=timeout))
        run_dir = Path(module.HARNESS_RUNS_DIR) / response["runId"]
        episode_path = run_dir / "episode.json"
    elif version == "v2":
        module = importlib.import_module("backend.harness_api_v2")
        target_root = Path(run_root).resolve() if run_root is not None else Path(module.HARNESS_RUNS_V2_DIR)
        target_root.mkdir(parents=True, exist_ok=True)
        previous_root = module.HARNESS_RUNS_V2_DIR
        module.HARNESS_RUNS_V2_DIR = target_root
        try:
            response = module.run_endpoint(module.RunRequest(plan=plan, timeout_sec=timeout))
        finally:
            module.HARNESS_RUNS_V2_DIR = previous_root
        run_dir = target_root / response["runId"]
        episode_path = run_dir / "episode_v2.json"
    else:
        raise ValueError("backend.episode_version 仅支持 v1 或 v2")
    step_path = run_dir / "result.step"
    return {
        "backend_version": version,
        "run_dir": str(run_dir.resolve()),
        "episode_path": str(episode_path.resolve()) if episode_path.is_file() else None,
        "result_step_path": str(step_path.resolve()) if step_path.is_file() and step_path.stat().st_size else None,
        "response": response,
    }
