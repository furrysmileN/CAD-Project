from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterable


TEXT_LEVELS = ("T1", "T2", "T3")
RENDER_LEVELS = ("I1", "I2", "I3")
POINT_LEVELS = ("P1", "P2", "P3")
_LEVELS = {"text": TEXT_LEVELS, "render": RENDER_LEVELS, "point": POINT_LEVELS}
_TOKEN_RE = re.compile(r"T[1-3]|I[1-3]|P[1-3]")


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    """A condition contains encoding choices only; prompt construction owns leakage control."""

    text: str | None = None
    render: str | None = None
    point: str | None = None

    def __post_init__(self) -> None:
        for field, value in (("text", self.text), ("render", self.render), ("point", self.point)):
            if value is not None and value not in _LEVELS[field]:
                raise ValueError(f"{field} 编码无效: {value!r}；允许值为 None 或 {_LEVELS[field]}")
        if self.text is None and self.render is None and self.point is None:
            raise ValueError("条件不能同时缺少 text、render 和 point")

    @property
    def condition_id(self) -> str:
        return "".join(value for value in (self.text, self.render, self.point) if value is not None)

    @property
    def modalities(self) -> frozenset[str]:
        return frozenset(
            name for name, value in (("text", self.text), ("render", self.render), ("point", self.point)) if value
        )

    @classmethod
    def parse(cls, condition_id: str) -> "ConditionSpec":
        if not isinstance(condition_id, str) or not condition_id:
            raise ValueError("condition_id 必须是非空字符串")
        tokens = _TOKEN_RE.findall(condition_id)
        if "".join(tokens) != condition_id:
            raise ValueError(f"condition_id 格式无效: {condition_id!r}")
        values: dict[str, str | None] = {"text": None, "render": None, "point": None}
        positions = {"T": 0, "I": 1, "P": 2}
        previous = -1
        for token in tokens:
            position = positions[token[0]]
            if position <= previous:
                raise ValueError(f"condition_id 必须按 T、I、P 排列且每类至多一次: {condition_id!r}")
            previous = position
            values[("text", "render", "point")[position]] = token
        return cls(**values)


def enumerate_conditions() -> tuple[ConditionSpec, ...]:
    """Enumerate levels in the stable legacy modality order T, I, P, TI, TP, IP, TIP."""
    result: list[ConditionSpec] = []
    modality_groups = (
        ("text",),
        ("render",),
        ("point",),
        ("text", "render"),
        ("text", "point"),
        ("render", "point"),
        ("text", "render", "point"),
    )
    for group in modality_groups:
        for levels in itertools.product(*(_LEVELS[name] for name in group)):
            result.append(ConditionSpec(**dict(zip(group, levels))))
    return tuple(result)


CONDITIONS = enumerate_conditions()
CONDITION_IDS = tuple(condition.condition_id for condition in CONDITIONS)


def parse_condition(condition_id: str) -> ConditionSpec:
    condition = ConditionSpec.parse(condition_id)
    if condition.condition_id not in CONDITION_IDS:
        raise ValueError(f"未知条件: {condition_id!r}")
    return condition


def validate_conditions(values: Iterable[str | ConditionSpec]) -> tuple[ConditionSpec, ...]:
    parsed = tuple(value if isinstance(value, ConditionSpec) else parse_condition(value) for value in values)
    ids = [condition.condition_id for condition in parsed]
    if len(ids) != len(set(ids)):
        raise ValueError("条件清单包含重复 condition_id")
    return parsed


def modalities_for(condition: str | ConditionSpec) -> frozenset[str]:
    return (condition if isinstance(condition, ConditionSpec) else parse_condition(condition)).modalities
