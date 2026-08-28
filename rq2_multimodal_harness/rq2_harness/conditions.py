from __future__ import annotations

CONDITIONS = ("T", "I", "P", "TI", "TP", "IP", "TIP")
_MODALITIES = {
    "T": frozenset({"text"}),
    "I": frozenset({"images"}),
    "P": frozenset({"point_cloud"}),
    "TI": frozenset({"text", "images"}),
    "TP": frozenset({"text", "point_cloud"}),
    "IP": frozenset({"images", "point_cloud"}),
    "TIP": frozenset({"text", "images", "point_cloud"}),
}


def modalities_for(condition: str) -> frozenset[str]:
    try:
        return _MODALITIES[condition]
    except KeyError as exc:
        raise ValueError(f"未知条件 {condition!r}，允许值: {', '.join(CONDITIONS)}") from exc


def validate_conditions(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    unique = tuple(dict.fromkeys(values))
    for value in unique:
        modalities_for(value)
    return unique
