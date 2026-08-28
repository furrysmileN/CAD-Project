from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import read_jsonl
from rq2_harness.pointcloud.evidence import build_evidence
from rq2_harness.v5_shuffle import build_shuffle_mapping, write_shuffle_mapping


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="构建 V5 size-matched shuffle 映射")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    rows = []
    for row in read_jsonl(Path(args.manifest)):
        npy = Path(str((row.get("point_cloud") or {}).get("path") or ""))
        frame = {}
        if npy.is_file():
            evidence = build_evidence(npy)
            frame = evidence.get("frame") or {}
        rows.append({**row, "evidence_frame": frame})
    payload = build_shuffle_mapping(rows, seed=args.seed)
    write_shuffle_mapping(Path(args.output), payload)
    print(json.dumps({"n": payload["n"], "sha256": payload["sha256"], "path": args.output}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
