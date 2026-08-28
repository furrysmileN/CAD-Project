from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.pc_runner import main


if __name__ == "__main__":
    config = Path(__file__).resolve().parents[1] / "configs" / "pc_geom_confirm.yaml"
    raise SystemExit(main(["--config", str(config), *sys.argv[1:]]))
