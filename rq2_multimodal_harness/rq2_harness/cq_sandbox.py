"""超时 CadQuery 沙箱：qwen_raw 与 CADrille 共用，不走 C 臂。"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

DENIED_MODULES = frozenset(
    {
        "os",
        "subprocess",
        "socket",
        "pathlib",
        "shutil",
        "sys",
        "ctypes",
        "importlib",
        "multiprocessing",
        "http",
        "urllib",
        "requests",
        "builtins",
    }
)
DENIED_NAMES = frozenset({"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"})
WRAPPER = r'''
import sys
from pathlib import Path
import cadquery as cq

code = Path(sys.argv[1]).read_text(encoding="utf-8")
ns = {"cq": cq, "cadquery": cq, "__name__": "__sandbox__"}
exec(compile(code, sys.argv[1], "exec"), ns, ns)
result = ns.get("result")
if result is None:
    result = ns.get("solid")
if result is None:
    result = ns.get("shape")
if result is None:
    for value in reversed(list(ns.values())):
        if hasattr(value, "val") and hasattr(value, "objects"):
            result = value
            break
        if hasattr(value, "ShapeType"):
            result = value
            break
if result is None:
    raise RuntimeError("sandbox_no_result: assign result / solid, or leave a Workplane")
if hasattr(result, "val"):
    result = result.val()
cq.exporters.export(result, sys.argv[2])
'''


def audit_cadquery_source(source: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [{"code": "syntax_error", "message": str(exc)}]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in {"cadquery", "cq", "math"}:
                    issues.append({"code": "denied_import", "message": alias.name})
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in {"cadquery", "cq", "math"}:
                issues.append({"code": "denied_import", "message": node.module or ""})
        elif isinstance(node, ast.Name) and node.id in DENIED_NAMES:
            issues.append({"code": "denied_name", "message": node.id})
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            issues.append({"code": "denied_dunder", "message": node.attr})
    return issues


def extract_cadquery_source(text: str) -> str:
    cleaned = text.strip()
    fences = []
    start = 0
    while True:
        begin = cleaned.find("```", start)
        if begin < 0:
            break
        newline = cleaned.find("\n", begin)
        end = cleaned.find("```", begin + 3)
        if newline < 0 or end < 0:
            break
        fences.append(cleaned[newline + 1 : end].strip())
        start = end + 3
    if fences:
        pythonish = [block for block in fences if "cadquery" in block or "cq." in block or "import cq" in block]
        return (pythonish[-1] if pythonish else fences[-1]).strip()
    return cleaned


def run_cadquery_sandbox(
    source: str,
    output_step: str | Path,
    *,
    timeout_sec: float = 30.0,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    issues = audit_cadquery_source(source)
    if issues:
        return {
            "ok": False,
            "step_path": None,
            "issues": issues,
            "returncode": None,
            "stderr": "",
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }
    dest = Path(output_step)
    dest.parent.mkdir(parents=True, exist_ok=True)
    root = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="hvc_sandbox_"))
    root.mkdir(parents=True, exist_ok=True)
    user_path = root / "user_model.py"
    wrapper_path = root / "wrapper.py"
    user_path.write_text(source.rstrip() + "\n", encoding="utf-8")
    wrapper_path.write_text(WRAPPER, encoding="utf-8")
    try:
        completed = subprocess.run(
            [sys.executable, str(wrapper_path), str(user_path), str(dest)],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "step_path": None,
            "issues": [{"code": "timeout", "message": f">{timeout_sec}s"}],
            "returncode": None,
            "stderr": (exc.stderr or "") if isinstance(exc.stderr, str) else "",
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }
    ok = completed.returncode == 0 and dest.is_file() and dest.stat().st_size > 0
    return {
        "ok": ok,
        "step_path": str(dest) if ok else None,
        "issues": [] if ok else [{"code": "execution_failed", "message": (completed.stderr or completed.stdout)[-2000:]}],
        "returncode": completed.returncode,
        "stderr": completed.stderr[-2000:],
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }
