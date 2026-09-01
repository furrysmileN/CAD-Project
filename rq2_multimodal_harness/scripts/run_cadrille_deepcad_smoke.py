"""DeepCAD 5 条冒烟：只确认 cadrille-rl 能出码，不进主表。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import project_path


def main() -> int:
    dest = project_path("experiments/rq2_multimodal_harness/outputs/harness_vs_cadrille/cadrille_runs")
    dest.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ.get("CADRILLE_ROOT") or project_path("third_party/cadrille"))
    note = dest / "DEEPCAD_SMOKE_STATUS.json"
    if not (root / "test.py").is_file():
        note.write_text(
            json.dumps({"status": "waiting_repo", "detail": "clone cadrille first"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(note.read_text(encoding="utf-8"))
        return 0
    try:
        import torch
    except Exception:
        torch = None
    if torch is None or not torch.cuda.is_available():
        note.write_text(
            json.dumps(
                {
                    "status": "waiting_gpu",
                    "detail": "Linux GPU: python test.py --split deepcad_test_mesh --mode img --checkpoint-path maksimko123/cadrille-rl --py-path work_dirs/deepcad_smoke",
                    "keep": 5,
                    "in_main_table": False,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(note.read_text(encoding="utf-8"))
        return 0
    py_path = dest / "deepcad_smoke_py"
    if py_path.exists():
        shutil.rmtree(py_path)
    py_path.mkdir(parents=True)
    completed = subprocess.run(
        [
            sys.executable,
            "test.py",
            "--split",
            "deepcad_test_mesh",
            "--mode",
            "img",
            "--checkpoint-path",
            "maksimko123/cadrille-rl",
            "--py-path",
            str(py_path),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=600,
    )
    files = sorted(py_path.glob("*.py"))[:5]
    note.write_text(
        json.dumps(
            {
                "status": "ok" if completed.returncode == 0 and files else "failed",
                "returncode": completed.returncode,
                "n_written": len(list(py_path.glob("*.py"))),
                "kept": [path.name for path in files],
                "stderr": (completed.stderr or "")[-1500:],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(note.read_text(encoding="utf-8"))
    return 0 if completed.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
