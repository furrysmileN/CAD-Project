from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import project_path, read_jsonl
from rq2_harness.evidence_audit import audit_evidence


def main() -> int:
    import argparse

    experiment_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="V5 新 100 样本 PointEvidence 离线审计")
    parser.add_argument(
        "--manifest",
        default=str(experiment_dir / "outputs" / "v5_complementarity" / "manifest_new100.jsonl"),
    )
    parser.add_argument(
        "--pointcloud-root",
        default=str(experiment_dir.parents[1] / "processed" / "point_clouds" / "benchcad"),
    )
    parser.add_argument(
        "--evidence-dir",
        default=str(experiment_dir / "outputs" / "v5_complementarity" / "evidence"),
    )
    parser.add_argument(
        "--output",
        default=str(experiment_dir / "outputs" / "v5_complementarity" / "evidence_audit"),
    )
    parser.add_argument("--density", type=int, default=2048)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    sample_ids = [row["sample_id"] for row in read_jsonl(manifest)]
    report = audit_evidence(
        sample_ids,
        manifest,
        Path(args.pointcloud_root),
        Path(args.evidence_dir),
        density=args.density,
        output_dir=Path(args.output),
    )
    payload = {
        "n": len(sample_ids),
        "gate": report.get("gate"),
        "summary": report.get("summary"),
        "output": args.output,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.get("gate", {}).get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
