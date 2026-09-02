from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .contract import CompileRequest, CompileResult, CompileSignal, read_json, write_json


def _failed_result(
    request: CompileRequest,
    code: str,
    message: str,
    *,
    detail: dict[str, Any] | None = None,
    elapsed_s: float | None = None,
    stdout: str = "",
    stderr: str = "",
) -> CompileResult:
    timings = {"worker_process": elapsed_s} if elapsed_s is not None else {}
    return CompileResult(
        status="failed",
        source_path=request.source_path,
        output_dir=request.output_dir,
        parameters=request.parameters,
        signals=[CompileSignal(code=code, message=message, detail=detail or {})],
        timings_s=timings,
        stdout=stdout,
        stderr=stderr,
    )


def compile_cadquery(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    parameters: dict[str, Any] | None = None,
    timeout_s: float = 60.0,
    python_executable: str | Path | None = None,
) -> CompileResult:
    """Run generated CadQuery code in a child process and validate its STEP output.

    The process boundary contains crashes and enforces a wall-clock timeout. It is
    not a security sandbox; only execute code from a trusted model or environment.
    """

    source = Path(source_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    request = CompileRequest(
        source_path=str(source),
        output_dir=str(output),
        parameters=dict(parameters or {}),
    )
    request_path = output / "compile_request.json"
    response_path = output / "compile_response.json"
    manifest_path = output / "manifest.json"
    write_json(request_path, request.to_dict())
    response_path.unlink(missing_ok=True)

    executable = str(python_executable or sys.executable)
    command = [
        executable,
        "-m",
        "cad_agent_prototype.worker",
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]
    env = os.environ.copy()
    repository_root = str(Path(__file__).resolve().parents[1])
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        repository_root if not old_pythonpath else os.pathsep.join((repository_root, old_pythonpath))
    )

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(source.parent if source.parent.is_dir() else Path.cwd()),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        result = _failed_result(
            request,
            "worker_timeout",
            f"CadQuery worker exceeded the {timeout_s:g}s timeout",
            detail={"timeout_s": timeout_s},
            elapsed_s=elapsed,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
        )
        write_json(manifest_path, result.to_dict())
        return result
    except OSError as exc:
        elapsed = time.perf_counter() - started
        result = _failed_result(
            request,
            "worker_start_failed",
            f"failed to start CadQuery worker: {type(exc).__name__}: {exc}",
            detail={"python_executable": executable},
            elapsed_s=elapsed,
        )
        write_json(manifest_path, result.to_dict())
        return result

    elapsed = time.perf_counter() - started
    if response_path.is_file():
        try:
            result = CompileResult.from_dict(read_json(response_path))
        except Exception as exc:
            result = _failed_result(
                request,
                "invalid_worker_response",
                f"failed to parse worker response: {type(exc).__name__}: {exc}",
                elapsed_s=elapsed,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
    else:
        result = _failed_result(
            request,
            "worker_crashed",
            f"CadQuery worker exited with code {completed.returncode} without a response",
            detail={"return_code": completed.returncode},
            elapsed_s=elapsed,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    result.stdout = completed.stdout
    result.stderr = completed.stderr
    result.timings_s["worker_process"] = elapsed
    write_json(manifest_path, result.to_dict())
    return result
