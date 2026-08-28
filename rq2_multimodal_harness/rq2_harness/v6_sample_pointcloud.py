"""Fixed-seed surface sampling from STEP to .npy."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import sha256_file
from .pointcloud.compare import sample_step


def sample_pointcloud(
    step_path: str | Path,
    npy_path: str | Path,
    *,
    n_points: int = 2048,
    seed: int = 42,
) -> dict[str, Any]:
    points = sample_step(step_path, n_points=n_points, seed=seed)
    npy_path = Path(npy_path)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, np.asarray(points, dtype=np.float64))
    return {
        "path": str(npy_path.resolve()),
        "sha256": sha256_file(npy_path),
        "n_points": int(len(points)),
        "seed": int(seed),
    }
