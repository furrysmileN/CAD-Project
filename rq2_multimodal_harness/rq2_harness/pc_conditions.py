"""P_geom / V5 条件 ID 空间。

不污染 rq2_harness.conditions 的 7 条件，也不复用 encoding 的 T1–P3 网格。
V4 的 SCREEN/CONFIRM ID 保持不变；V5 在同一 registry 中追加消融、确认与工具臂。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


SCREEN_CONDITION_IDS = (
    "I1",
    "P_proj",
    "P_geom_static",
    "P_geom_tool",
    "I1P_proj",
    "I1P_geom",
    "T1I1",
    "T1I1P_proj",
    "T1I1P_geom",
)

CONFIRM_CONDITION_IDS = (
    "I1",
    "P_proj",
    "P_geom_tool",
    "I1P_proj",
    "I1P_geom",
    "T1I1",
    "T1I1P_proj",
    "T1I1P_geom",
)

V5_ABLATION_IDS = (
    "P_proj",
    "P_bbox",
    "P_axes",
    "P_sym",
    "P_full",
    "P_shuffle",
)

V5_CONFIRM_IDS = (
    "P_proj",
    "P_geom",
    "I1",
    "I1P_geom",
    "T1I1",
    "T1I1P_geom",
    "T2I1",
    "T2I1P_geom",
)

V5_CONTROL_IDS = (
    "I1P_proj",
    "I1P_shuffle",
)

V5_TOOL_IDS = (
    "STATIC_PARTIAL",
    "OPTIONAL_TOOL",
    "FORCED_QUERY",
)

V8_I1_ABLATION_IDS = (
    "I1P_bbox",
    "I1P_axes",
    "I1P_sym",
)

EVIDENCE_PROFILES = ("bbox", "axes", "sym", "full", "partial")
TEXT_LEVELS = ("T1", "T2")
TOOL_PROTOCOLS = ("none", "legacy", "query_or_submit", "forced_query")


@dataclass(frozen=True, slots=True)
class PCConditionSpec:
    """模态开关 + 点云证据档 + 文本强度 + 工具协议。"""

    condition_id: str
    text: bool = False
    text_level: str | None = None
    images: bool = False
    point_proj: bool = False
    point_geom: bool = False
    tools: bool = False
    evidence_profile: str | None = None
    shuffle: bool = False
    tool_protocol: str = "none"

    def __post_init__(self) -> None:
        if not (self.text or self.images or self.point_proj or self.point_geom):
            raise ValueError(f"条件 {self.condition_id} 不能没有任何模态")
        if self.tools and not self.point_geom:
            raise ValueError(f"条件 {self.condition_id} 启用工具必须同时启用 point_geom")
        if self.point_proj and self.point_geom:
            raise ValueError(f"条件 {self.condition_id} 不能同时含 P_proj 与 P_geom")
        if self.text and self.text_level not in TEXT_LEVELS:
            raise ValueError(f"条件 {self.condition_id} 的 text_level 必须是 T1 或 T2")
        if self.point_geom and (self.evidence_profile or "full") not in EVIDENCE_PROFILES:
            raise ValueError(f"条件 {self.condition_id} 的 evidence_profile 非法")
        if self.tool_protocol not in TOOL_PROTOCOLS:
            raise ValueError(f"条件 {self.condition_id} 的 tool_protocol 非法")
        if self.tools and self.tool_protocol == "none":
            object.__setattr__(self, "tool_protocol", "legacy")

    @property
    def modalities(self) -> frozenset[str]:
        names = []
        if self.text:
            names.append("text")
        if self.images:
            names.append("images")
        if self.point_proj:
            names.append("point_proj")
        if self.point_geom:
            names.append("point_geom")
        if self.tools:
            names.append("tools")
        return frozenset(names)

    @property
    def resolved_profile(self) -> str | None:
        if not self.point_geom:
            return None
        return self.evidence_profile or "full"


def _spec(**kwargs: object) -> PCConditionSpec:
    return PCConditionSpec(**kwargs)  # type: ignore[arg-type]


_SPECS: dict[str, PCConditionSpec] = {
    "I1": _spec(condition_id="I1", images=True),
    "P_proj": _spec(condition_id="P_proj", point_proj=True),
    "P_geom_static": _spec(condition_id="P_geom_static", point_geom=True, evidence_profile="full"),
    "P_geom_tool": _spec(
        condition_id="P_geom_tool", point_geom=True, tools=True, evidence_profile="full", tool_protocol="legacy"
    ),
    "I1P_proj": _spec(condition_id="I1P_proj", images=True, point_proj=True),
    "I1P_geom": _spec(
        condition_id="I1P_geom", images=True, point_geom=True, tools=True, evidence_profile="full", tool_protocol="legacy"
    ),
    "T1I1": _spec(condition_id="T1I1", text=True, text_level="T1", images=True),
    "T1I1P_proj": _spec(condition_id="T1I1P_proj", text=True, text_level="T1", images=True, point_proj=True),
    "T1I1P_geom": _spec(
        condition_id="T1I1P_geom",
        text=True,
        text_level="T1",
        images=True,
        point_geom=True,
        tools=True,
        evidence_profile="full",
        tool_protocol="legacy",
    ),
    "P_bbox": _spec(condition_id="P_bbox", point_geom=True, evidence_profile="bbox"),
    "P_axes": _spec(condition_id="P_axes", point_geom=True, evidence_profile="axes"),
    "P_sym": _spec(condition_id="P_sym", point_geom=True, evidence_profile="sym"),
    "P_full": _spec(condition_id="P_full", point_geom=True, evidence_profile="full"),
    "P_shuffle": _spec(condition_id="P_shuffle", point_geom=True, evidence_profile="full", shuffle=True),
    "P_geom": _spec(condition_id="P_geom", point_geom=True, evidence_profile="full"),
    "T2I1": _spec(condition_id="T2I1", text=True, text_level="T2", images=True),
    "T2I1P_geom": _spec(
        condition_id="T2I1P_geom", text=True, text_level="T2", images=True, point_geom=True, evidence_profile="full"
    ),
    "I1P_shuffle": _spec(
        condition_id="I1P_shuffle", images=True, point_geom=True, evidence_profile="full", shuffle=True
    ),
    "I1P_bbox": _spec(condition_id="I1P_bbox", images=True, point_geom=True, evidence_profile="bbox"),
    "I1P_axes": _spec(condition_id="I1P_axes", images=True, point_geom=True, evidence_profile="axes"),
    "I1P_sym": _spec(condition_id="I1P_sym", images=True, point_geom=True, evidence_profile="sym"),
    "STATIC_PARTIAL": _spec(condition_id="STATIC_PARTIAL", point_geom=True, evidence_profile="partial"),
    "OPTIONAL_TOOL": _spec(
        condition_id="OPTIONAL_TOOL",
        point_geom=True,
        tools=True,
        evidence_profile="partial",
        tool_protocol="query_or_submit",
    ),
    "FORCED_QUERY": _spec(
        condition_id="FORCED_QUERY",
        point_geom=True,
        tools=True,
        evidence_profile="partial",
        tool_protocol="forced_query",
    ),
}

ALL_CONDITION_IDS = tuple(_SPECS)


def parse_condition(condition_id: str) -> PCConditionSpec:
    spec = _SPECS.get(condition_id)
    if spec is None:
        raise ValueError(f"未知 P_geom 条件 {condition_id!r}，允许值: {', '.join(ALL_CONDITION_IDS)}")
    return spec


def validate_conditions(values: Iterable[str | PCConditionSpec]) -> tuple[PCConditionSpec, ...]:
    parsed = tuple(value if isinstance(value, PCConditionSpec) else parse_condition(value) for value in values)
    ids = [spec.condition_id for spec in parsed]
    if len(ids) != len(set(ids)):
        raise ValueError("条件清单包含重复 condition_id")
    return parsed


def modalities_for(condition: str | PCConditionSpec) -> frozenset[str]:
    return (condition if isinstance(condition, PCConditionSpec) else parse_condition(condition)).modalities
