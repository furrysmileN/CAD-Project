from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import PROJECT_ROOT, atomic_write_json, project_path, read_jsonl
from rq2_harness.v8_autopsy import (
    SAMPLE_N,
    SAMPLE_SEED,
    patch_preregistration,
    run_autopsy,
    sample_n20,
    write_n20_manifest,
)


def _overlap(ids: set[str], path: Path) -> list[str]:
    if not path.is_file():
        return []
    return sorted(ids & {row["sample_id"] for row in read_jsonl(path)})


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="V8 Cut 1 尸检 + 分层抽 20（零 API）")
    parser.add_argument("--skip-sample", action="store_true")
    args = parser.parse_args()

    output_dir = project_path("experiments/rq2_multimodal_harness/outputs/v8_residual_complementarity")
    autopsy_dir = output_dir / "cut1_autopsy"
    result = run_autopsy(
        metrics_path=project_path(
            "experiments/rq2_multimodal_harness/outputs/v5_complementarity/repeats/analysis/primary_metrics.csv"
        ),
        manifest_path=project_path(
            "experiments/rq2_multimodal_harness/outputs/v5_complementarity/manifest_new100.jsonl"
        ),
        state_root=project_path("experiments/rq2_multimodal_harness/outputs/v5_complementarity/repeats"),
        evidence_dir=project_path("experiments/rq2_multimodal_harness/outputs/v5_complementarity/evidence"),
        evidence_audit_path=project_path(
            "experiments/rq2_multimodal_harness/outputs/v5_complementarity/evidence_audit/evidence_audit.json"
        ),
        output_dir=autopsy_dir,
    )
    payload: dict = {"autopsy": result}
    if not args.skip_sample:
        if not result.get("go", {}).get("proceed_cut2"):
            print(json.dumps({**payload, "sample": "skipped_no_go"}, ensure_ascii=False, indent=2))
            return 2
        parent = project_path("experiments/rq2_multimodal_harness/outputs/v5_complementarity/manifest_new100.jsonl")
        parent_rows = list(read_jsonl(parent))
        drawn = sample_n20(parent_rows, seed=SAMPLE_SEED, n=SAMPLE_N)
        ids = set(drawn["sample_ids"])
        overlap = {
            "encoding_screen_n20": _overlap(
                ids,
                project_path("experiments/rq2_multimodal_harness/outputs/encoding_screen_n20/sample_manifest.jsonl"),
            ),
            "pilot_v2": _overlap(
                ids,
                project_path("experiments/rq2_multimodal_harness/outputs/pilot_v2/manifest.jsonl"),
            ),
        }
        if any(overlap.values()):
            raise RuntimeError(f"20 件与旧清单重叠: {overlap}")
        dest = output_dir / "manifest_n20.jsonl"
        write_n20_manifest(parent, drawn["sample_ids"], dest)
        atomic_write_json(output_dir / "sample_n20.json", {**drawn, "overlap": overlap})
        prereg = output_dir / "preregistration.md"
        if prereg.is_file():
            patch_preregistration(prereg, drawn["sample_ids"])
        payload["sample"] = {
            "n": len(drawn["sample_ids"]),
            "seed": SAMPLE_SEED,
            "manifest": str(dest.relative_to(PROJECT_ROOT)),
            "ids": drawn["sample_ids"],
            "overlap": overlap,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.get("go", {}).get("proceed_cut2") else 2


if __name__ == "__main__":
    raise SystemExit(main())
