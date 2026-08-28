from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.v5_reaudit import run_heldout80


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    payload = run_heldout80(
        state_dir=root / "outputs" / "native_pointcloud_v1" / "confirm_n100" / "state",
        manifest_path=root / "outputs" / "pilot_v2" / "manifest.jsonl",
        selection_summary=root / "outputs" / "encoding_screen_n20" / "selection_summary.json",
        output_dir=root / "outputs" / "native_pointcloud_v1" / "analysis" / "v5_reaudit",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("gate", {}).get("proceed") else 3


if __name__ == "__main__":
    raise SystemExit(main())
