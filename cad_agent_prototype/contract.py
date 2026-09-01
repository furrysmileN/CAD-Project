from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "cad-agent.compile.v1"


@dataclass(frozen=True)
class CompileRequest:
    source_path: str
    output_dir: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_path": self.source_path,
            "output_dir": self.output_dir,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompileRequest":
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {version!r}")
        return cls(
            source_path=str(data["source_path"]),
            output_dir=str(data["output_dir"]),
            parameters=dict(data.get("parameters") or {}),
        )


@dataclass(frozen=True)
class CompileSignal:
    code: str
    message: str
    severity: str = "error"
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompileResult:
    status: str
    source_path: str
    output_dir: str
    parameters: dict[str, Any]
    step_path: str | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    signals: list[CompileSignal] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    schema_version: str = SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "source_path": self.source_path,
            "output_dir": self.output_dir,
            "parameters": self.parameters,
            "step_path": self.step_path,
            "validation": self.validation,
            "signals": [signal.to_dict() for signal in self.signals],
            "timings_s": self.timings_s,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompileResult":
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            status=str(data["status"]),
            source_path=str(data["source_path"]),
            output_dir=str(data["output_dir"]),
            parameters=dict(data.get("parameters") or {}),
            step_path=data.get("step_path"),
            validation=dict(data.get("validation") or {}),
            signals=[CompileSignal(**signal) for signal in data.get("signals", [])],
            timings_s={str(key): float(value) for key, value in (data.get("timings_s") or {}).items()},
            stdout=str(data.get("stdout") or ""),
            stderr=str(data.get("stderr") or ""),
        )


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
