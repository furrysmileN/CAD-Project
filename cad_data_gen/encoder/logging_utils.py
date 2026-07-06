from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class StructuredLogRecord:
    event: str
    timestamp: float = field(default_factory=time.time)
    level: str = "info"
    sample_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonlLogger:
    """简单 JSONL 结构化日志写入器。"""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, *, level: str = "info", sample_id: str | None = None, **payload: Any) -> None:
        if self.path is None:
            return
        record = StructuredLogRecord(event=event, level=level, sample_id=sample_id, payload=payload)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def log_mapping(self, event: str, data: Mapping[str, Any], *, level: str = "info", sample_id: str | None = None) -> None:
        self.log(event, level=level, sample_id=sample_id, **dict(data))


def write_summary(path: str | Path, summary: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(dict(summary), f, ensure_ascii=False, indent=2)
