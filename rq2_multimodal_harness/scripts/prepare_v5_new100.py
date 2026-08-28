from pathlib import Path
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import load_config, project_path, read_jsonl, write_jsonl
from rq2_harness.prepare import prepare


def overlap_audit(new_rows: list[dict], old_manifest: Path, extra_manifests: list[Path] | None = None) -> dict:
    old = {row["sample_id"] for row in read_jsonl(old_manifest)}
    new = {row["sample_id"] for row in new_rows}
    extras = {}
    extra_overlap: set[str] = set()
    for path in extra_manifests or []:
        if path.is_file():
            ids = {row["sample_id"] for row in read_jsonl(path)}
            hit = sorted(new & ids)
            extras[path.name] = {"n": len(ids), "overlap": hit}
            extra_overlap.update(hit)
    return {
        "n_new": len(new),
        "n_old": len(old),
        "overlap": sorted(new & old),
        "overlap_empty": not (new & old) and not extra_overlap,
        "extra": extras,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="抽取 V5 全新 100 样本并审计无重叠")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "v5_prepare_new100.yaml"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-ids", help="先保留这些 sample_id（一行一个），再补齐到 n")
    args = parser.parse_args()
    config = load_config(args.config)
    keep_ids = None
    if args.keep_ids:
        keep_path = Path(args.keep_ids)
        keep_ids = [
            line.strip()
            for line in keep_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    result = prepare(config, force=args.force, keep_ids=keep_ids)
    output_dir = project_path(config["paths"]["output_dir"])
    src = output_dir / "manifest.jsonl"
    dest = output_dir / "manifest_new100.jsonl"
    if src.is_file():
        shutil.copyfile(src, dest)
    rows = list(read_jsonl(dest)) if dest.is_file() else []
    audit = overlap_audit(
        rows,
        project_path("experiments/rq2_multimodal_harness/outputs/pilot_v2/manifest.jsonl"),
        extra_manifests=[
            project_path("experiments/rq2_multimodal_harness/outputs/encoding_screen_n20/sample_manifest.jsonl")
        ],
    )
    (output_dir / "overlap_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**result, "overlap": audit, "manifest": str(dest)}, ensure_ascii=False, indent=2))
    return 0 if audit["overlap_empty"] and len(rows) == int(config["n"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
