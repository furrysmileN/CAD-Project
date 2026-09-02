from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compile import compile_cadquery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile a trusted CadQuery build_model(params) program into validated STEP."
    )
    parser.add_argument("source", type=Path, help="Python file defining build_model(params)")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--params", default="{}", help="JSON object passed to build_model")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--python", dest="python_executable")
    args = parser.parse_args(argv)

    try:
        parameters = json.loads(args.params)
    except json.JSONDecodeError as exc:
        parser.error(f"--params is not valid JSON: {exc}")
    if not isinstance(parameters, dict):
        parser.error("--params must decode to a JSON object")

    result = compile_cadquery(
        args.source,
        args.output,
        parameters=parameters,
        timeout_s=args.timeout,
        python_executable=args.python_executable,
    )
    json.dump(result.to_dict(), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
