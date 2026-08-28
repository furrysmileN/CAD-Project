from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .common import PROJECT_ROOT, atomic_write_json, load_config, project_path, read_jsonl


OLD_URI = re.compile(r"^(?:benchcad|ztc|file)://", re.IGNORECASE)


def _declared_modalities(row: dict[str, Any]) -> set[str] | None:
    value = row.get("modality_combination")
    if not isinstance(value, str):
        return None
    return {letter for letter in value.upper() if letter in {"T", "I", "P"}}


def _present_modalities(row: dict[str, Any]) -> set[str]:
    present = set()
    if isinstance(row.get("text"), str) and row["text"].strip():
        present.add("T")
    if isinstance(row.get("image_path"), str) and row["image_path"].strip():
        present.add("I")
    if isinstance(row.get("pointcloud_path"), str) and row["pointcloud_path"].strip():
        present.add("P")
    return present


def audit(config: dict[str, Any], output: Path | None = None) -> dict[str, Any]:
    root = project_path(config["paths"]["conflict_scenes"])
    issues: list[dict[str, Any]] = []
    counts = Counter()
    files = sorted(root.rglob("*.jsonl"))
    for source in files:
        for line_number, row in enumerate(read_jsonl(source), 1):
            counts["rows"] += 1
            scene_id = row.get("scene_id")
            for field in ("image_path", "pointcloud_path"):
                value = row.get(field)
                if isinstance(value, str) and OLD_URI.match(value):
                    issues.append(
                        {"code": "legacy_uri", "scene_id": scene_id, "source": str(source), "line": line_number, "field": field, "value": value}
                    )
            pc_value = row.get("pointcloud_path")
            if "pointcloud_path" in row and (pc_value is None or not str(pc_value).strip()):
                issues.append(
                    {
                        "code": "empty_pointcloud_reference",
                        "scene_id": scene_id,
                        "source": str(source),
                        "line": line_number,
                    }
                )
            if isinstance(pc_value, str) and pc_value.strip() and not OLD_URI.match(pc_value):
                path = Path(pc_value)
                path = path if path.is_absolute() else PROJECT_ROOT / path
                if not path.is_file() or path.stat().st_size == 0:
                    issues.append({"code": "pointcloud_missing_or_empty_file", "scene_id": scene_id, "source": str(source), "line": line_number, "value": pc_value})
                else:
                    try:
                        array = np.load(path, mmap_mode="r", allow_pickle=False)
                        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 3:
                            issues.append({"code": "pointcloud_empty_or_bad_shape", "scene_id": scene_id, "source": str(source), "line": line_number, "shape": list(array.shape)})
                        elif not np.isfinite(array).all():
                            issues.append({"code": "pointcloud_nonfinite", "scene_id": scene_id, "source": str(source), "line": line_number})
                    except Exception as exc:
                        issues.append({"code": "pointcloud_unreadable", "scene_id": scene_id, "source": str(source), "line": line_number, "error": str(exc)[:300]})
            declared = _declared_modalities(row)
            if declared is not None:
                present = _present_modalities(row)
                if declared != present:
                    issues.append(
                        {
                            "code": "modality_declaration_mismatch",
                            "scene_id": scene_id,
                            "source": str(source),
                            "line": line_number,
                            "declared": sorted(declared),
                            "present": sorted(present),
                        }
                    )
    counts.update(issue["code"] for issue in issues)
    report = {
        "schema_version": "rq2.conflict_audit.v1",
        "excluded_from_main_matrix": True,
        "root": str(root.resolve()),
        "files": [str(path.resolve()) for path in files],
        "counts": dict(counts),
        "issues": issues,
        "note": "本报告仅审计冲突/缺失/退化场景，不将任何记录并入七条件主实验矩阵。",
    }
    target = output or project_path(config["paths"]["output_dir"]) / "audits" / "conflict_scenes.json"
    atomic_write_json(target, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="审计旧 URI、空点云与模态声明")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"))
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = audit(load_config(args.config), Path(args.output).resolve() if args.output else None)
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
