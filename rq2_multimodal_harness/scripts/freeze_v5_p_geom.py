from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import atomic_write_json, project_path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="冻结 V5 P_geom 证据组成")
    parser.add_argument("--composition", default="P_full")
    parser.add_argument(
        "--output",
        default="experiments/rq2_multimodal_harness/outputs/v5_complementarity/ablation_n20/p_geom_freeze.json",
    )
    parser.add_argument("--status", default="pending_live_ablation")
    parser.add_argument(
        "--note",
        default="Phase B live 20×6 未开 API。按指南默认冻结 P_full（bbox+axes+symmetry+sections+primitives+hypotheses）。",
    )
    args = parser.parse_args()
    payload = {
        "schema_version": "rq2.v5.p_geom_freeze.v1",
        "p_geom_composition": args.composition,
        "status": args.status,
        "note": args.note,
        "allowed_profiles": ["bbox", "axes", "sym", "full"],
    }
    path = project_path(args.output)
    atomic_write_json(path, payload)
    print(json.dumps({"path": str(path), **payload}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
