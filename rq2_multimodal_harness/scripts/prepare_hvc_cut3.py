"""Cut 3 门控：第二模型 / pc 附表 / 扩 100 只在 Cut 2 过门后准备。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import load_config
from rq2_harness.hvc_analysis import prepare_cut3_expand100


def main() -> int:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "harness_vs_cadrille.yaml")
    result = prepare_cut3_expand100(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"ready", "blocked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
