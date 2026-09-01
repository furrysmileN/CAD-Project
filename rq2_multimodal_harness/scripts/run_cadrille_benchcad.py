"""CADrille BenchCAD adapter + 可选 Linux GPU 推理。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.cadrille_adapter import SPLIT_NAME, export_cadrille_split
from rq2_harness.common import project_path
from rq2_harness.cq_sandbox import run_cadquery_sandbox


def _write_smoke_note(dest: Path, status: str, detail: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SMOKE_STATUS.json").write_text(
        json.dumps(
            {
                "status": status,
                "detail": detail,
                "repo": "https://github.com/col14m/cadrille",
                "weights": "maksimko123/cadrille-rl",
                "deepcad_smoke": "python test.py --split deepcad_test_mesh --mode img --checkpoint-path maksimko123/cadrille-rl  # 仅冒烟 5 条，不进主表",
                "benchcad": "python test_benchcad_hvc.py --split-root <cadrille_split> --mode img --checkpoint-path maksimko123/cadrille-rl --py-path predictions_img",
                "fairness": "Do not feed GT STL into official CadRecodeDataset.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def try_clone(root: Path) -> Path | None:
    if (root / "test.py").is_file():
        return root
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/col14m/cadrille.git", str(root)],
            check=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return root if (root / "test.py").is_file() else None


def main() -> int:
    export = export_cadrille_split()
    out_root = project_path("experiments/rq2_multimodal_harness/outputs/harness_vs_cadrille/cadrille_runs")
    env_root = os.environ.get("CADRILLE_ROOT")
    repo = Path(env_root) if env_root else project_path("third_party/cadrille")
    cloned = try_clone(repo)
    split_root = project_path("experiments/rq2_multimodal_harness/outputs/harness_vs_cadrille/cadrille_split")
    if cloned is not None:
        for name in ("benchcad_hvc_dataset.py", "test_benchcad_hvc.py"):
            src = split_root / name
            if src.is_file():
                shutil.copyfile(src, cloned / name)
    can_infer = False
    if cloned is not None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            can_infer = bool(torch.cuda.is_available())
        except Exception:
            can_infer = False
    if cloned is None or not can_infer:
        _write_smoke_note(
            out_root,
            "adapter_ready_waiting_gpu",
            "split 已导出，drop-in 已写入 cadrille_split。本机无 CUDA/transformers，不跑官方 test.py（且不可把 GT STL 喂给 CadRecodeDataset）。Linux GPU 上用 test_benchcad_hvc.py。",
        )
        print(json.dumps({"export": export, "infer": "pending_gpu", "repo": str(cloned) if cloned else None}, ensure_ascii=False, indent=2))
        return 0

    predictions = out_root / "predictions_img"
    predictions.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "test_benchcad_hvc.py",
        "--split-root",
        str(split_root),
        "--mode",
        "img",
        "--checkpoint-path",
        "maksimko123/cadrille-rl",
        "--py-path",
        str(predictions),
    ]
    try:
        completed = subprocess.run(cmd, cwd=str(cloned), timeout=600, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as exc:
        _write_smoke_note(out_root, "infer_failed", str(exc))
        print(json.dumps({"export": export, "infer": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 0

    (out_root / "cadrille_test_stdout.txt").write_text((completed.stdout or "")[-8000:], encoding="utf-8")
    if completed.returncode != 0:
        _write_smoke_note(out_root, "infer_nonzero", completed.stderr[-2000:] if completed.stderr else "nonzero")
        return 0

    # 若官方脚本把 py 写到 cwd，拷到 predictions
    for path in cloned.glob("*.py"):
        if path.name in {"test.py", "train.py", "evaluate.py"}:
            continue
        shutil.copyfile(path, predictions / path.name)
    _write_smoke_note(out_root, "infer_ok", "cadrille test.py finished")
    print(json.dumps({"export": export, "infer": "ok", "predictions": str(predictions)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
