from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.pc_runner import main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv[1:1] = [
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs" / "v5_phase_c_confirm.yaml"),
        ]
    raise SystemExit(main())
