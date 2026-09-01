from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import project_path
from rq2_harness.v8_analysis import analyze_cut2


def main() -> int:
    payload = analyze_cut2(
        v8_output=project_path(
            "experiments/rq2_multimodal_harness/outputs/v8_residual_complementarity/cut2_i1_ablation"
        ),
        manifest_path=project_path(
            "experiments/rq2_multimodal_harness/outputs/v8_residual_complementarity/manifest_n20.jsonl"
        ),
        v5_metrics=project_path(
            "experiments/rq2_multimodal_harness/outputs/v5_complementarity/repeats/analysis/primary_metrics.csv"
        ),
        dest=project_path(
            "experiments/rq2_multimodal_harness/outputs/v8_residual_complementarity/cut2_i1_ablation/analysis"
        ),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
