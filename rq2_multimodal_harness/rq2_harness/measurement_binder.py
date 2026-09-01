"""Bind semantic evidence references to measured Plan v3.1 operations."""
from __future__ import annotations

import copy
from typing import Any

from .pointcloud.centerline import bound_sweep_operations


class BindingError(ValueError):
    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(message)
        self.issue = {"code": code, "path": path, "message": message}


def bind_evidence_references(
    plan: dict[str, Any],
    path_graph: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Expand sweep_path_ref without letting the model edit measured parameters."""
    bound = copy.deepcopy(plan)
    operations = bound.get("operations")
    if not isinstance(operations, list):
        return bound, []
    components = {
        str(item.get("id")): item
        for item in ((path_graph or {}).get("components") or [])
        if isinstance(item, dict) and item.get("id")
    }
    semantic_ids = {
        str(item.get("id"))
        for item in operations
        if isinstance(item, dict) and item.get("op") == "sweep_path_ref"
    }
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict) or operation.get("op") == "sweep_path_ref":
            continue
        if operation.get("source") in semantic_ids:
            raise BindingError(
                "semantic_source_unsupported",
                f"$.operations[{index}].source",
                "A later transform/pattern cannot source a semantic path reference in binder v1.",
            )

    expanded: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict) or operation.get("op") != "sweep_path_ref":
            expanded.append(operation)
            continue
        reference = operation.get("evidence_ref")
        component = components.get(str(reference))
        if component is None:
            raise BindingError(
                "unknown_evidence_ref",
                f"$.operations[{index}].evidence_ref",
                f"Evidence reference {reference!r} is not resolved for this input.",
            )
        operation_id = operation.get("id")
        combine = operation.get("combine")
        if not isinstance(operation_id, str) or combine not in {
            "new",
            "add",
            "cut",
            "intersect",
        }:
            raise BindingError(
                "invalid_semantic_operation",
                f"$.operations[{index}]",
                "sweep_path_ref requires string id and a valid combine.",
            )
        replacements = bound_sweep_operations(
            component,
            operation_id=operation_id,
            combine=combine,
        )
        expanded.extend(replacements)
        audit.append(
            {
                "operation_index": index,
                "operation_id": operation_id,
                "evidence_ref": reference,
                "confidence": component.get("confidence"),
                "expanded_ids": [item["id"] for item in replacements],
            }
        )
    bound["operations"] = expanded
    return bound, audit
