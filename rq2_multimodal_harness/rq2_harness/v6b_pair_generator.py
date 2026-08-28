"""V6b minimal counterfactual part pairs. Latent specs never enter API payloads."""
from __future__ import annotations

import copy
import hashlib
from typing import Any

GENERATOR_VERSION = "rq2.v6b.pair.v2"
PAIR_SEED_BASE = 30_000
PAIR_KINDS = (
    "blind_depth",
    "through_vs_blind",
    "hidden_presence",
    "pocket_depth",
)


def _rng_values(seed: int) -> dict[str, float]:
    digest = hashlib.sha256(f"v6b-pair:{seed}".encode("utf-8")).digest()
    floats = [b / 255.0 for b in digest[:8]]
    return {
        "a": floats[0],
        "b": floats[1],
        "c": floats[2],
        "d": floats[3],
        "e": floats[4],
    }


def _plan(sample_id: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "harnesscad.plan.v2",
        "sample_id": sample_id,
        "coordinate_system": {
            "units": "normalized",
            "origin": [0.0, 0.0, 0.0],
            "longest_bbox_edge": 1.0,
        },
        "metadata": {"suite": "v6b", "generator": GENERATOR_VERSION},
        "operations": operations,
    }


def _spec(
    sample_id: str,
    family: str,
    seed: int,
    operations: list[dict[str, Any]],
    critical: dict[str, Any],
    *,
    pair_id: str,
    variant: str,
    kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": GENERATOR_VERSION,
        "sample_id": sample_id,
        "pair_id": pair_id,
        "variant": variant,
        "kind": kind,
        "family": family,
        "split": "probe",
        "generator_seed": seed,
        "operations": operations,
        "gt_plan": _plan(sample_id, operations),
        "critical_fact": critical,
        "image_cameras": ["front", "side", "top", "isometric"],
        "notes": {
            "visibility_policy": "critical fact is designed to be absent or unresolvable in the four frozen RGB views",
            "pair": "A and B differ only in the pre-registered latent f",
        },
    }


def _blind_depth_pair(pair_id: str, seed: int) -> dict[str, Any]:
    r = _rng_values(seed)
    thickness = 0.42
    hole_d = round(0.12 + 0.02 * r["a"], 4)
    x1 = round(-0.16 + 0.04 * r["b"], 4)
    y1 = round(0.08 + 0.04 * r["c"], 4)
    through_d = round(0.10 + 0.02 * r["d"], 4)
    x2 = round(0.22 - 0.03 * r["e"], 4)
    depth_a, depth_b = 0.16, 0.30

    def ops(depth: float) -> list[dict[str, Any]]:
        half_z = round(thickness / 2.0, 4)
        center_z = round(half_z - depth / 2.0, 4)
        return [
            {"id": "base", "op": "box", "combine": "new", "center": [0.0, 0.0, 0.0], "size": [1.0, 0.72, thickness]},
            {
                "id": "hole_blind",
                "op": "hole",
                "combine": "cut",
                "workplane": "XY",
                "center": [x1, y1, center_z],
                "diameter": hole_d,
                "depth": depth,
            },
            {
                "id": "hole_through",
                "op": "hole",
                "combine": "cut",
                "workplane": "XY",
                "center": [x2, -0.10, 0.0],
                "diameter": through_d,
                "depth": round(thickness + 0.08, 4),
            },
        ]

    def critical(depth: float) -> dict[str, Any]:
        return {
            "fact_id": "hole_blind.depth",
            "category": "depth",
            "value": depth,
            "visibility_in_images": False,
            "recoverable_from_pointcloud": True,
            "operation_id": "hole_blind",
            "field": "depth",
        }

    spec_a = _spec(f"{pair_id}a", "plate_holes", seed, ops(depth_a), critical(depth_a), pair_id=pair_id, variant="A", kind="blind_depth")
    spec_b = _spec(f"{pair_id}b", "plate_holes", seed, ops(depth_b), critical(depth_b), pair_id=pair_id, variant="B", kind="blind_depth")
    return {"pair_id": pair_id, "kind": "blind_depth", "family": "plate_holes", "spec_a": spec_a, "spec_b": spec_b}


def _through_vs_blind_pair(pair_id: str, seed: int) -> dict[str, Any]:
    r = _rng_values(seed)
    thickness = 0.28
    hole_d = round(0.12 + 0.02 * r["a"], 4)
    x1 = round(-0.16 + 0.04 * r["b"], 4)
    y1 = round(0.06 + 0.04 * r["c"], 4)
    depth_a = 0.12
    depth_b = round(thickness + 0.10, 4)

    def ops(depth: float, center_z: float) -> list[dict[str, Any]]:
        return [
            {"id": "base", "op": "box", "combine": "new", "center": [0.0, 0.0, 0.0], "size": [1.0, 0.72, thickness]},
            {
                "id": "hole_blind",
                "op": "hole",
                "combine": "cut",
                "workplane": "XY",
                "center": [x1, y1, center_z],
                "diameter": hole_d,
                "depth": depth,
            },
        ]

    half_z = round(thickness / 2.0, 4)
    ops_a = ops(depth_b, 0.0)
    ops_b = ops(depth_a, round(half_z - depth_a / 2.0, 4))
    crit_a = {
        "fact_id": "hole_blind.through_vs_blind",
        "category": "through_vs_blind",
        "value": "through",
        "visibility_in_images": False,
        "recoverable_from_pointcloud": True,
        "operation_id": "hole_blind",
        "field": "through_vs_blind",
    }
    crit_b = dict(crit_a)
    crit_b["value"] = "blind"
    spec_a = _spec(f"{pair_id}a", "plate_holes", seed, ops_a, crit_a, pair_id=pair_id, variant="A", kind="through_vs_blind")
    spec_b = _spec(f"{pair_id}b", "plate_holes", seed, ops_b, crit_b, pair_id=pair_id, variant="B", kind="through_vs_blind")
    return {"pair_id": pair_id, "kind": "through_vs_blind", "family": "plate_holes", "spec_a": spec_a, "spec_b": spec_b}


def _hidden_presence_pair(pair_id: str, seed: int) -> dict[str, Any]:
    r = _rng_values(seed)
    size_y = round(0.50 + 0.04 * r["a"], 4)
    hole_d = round(0.12 + 0.02 * r["b"], 4)
    hole_depth = round(0.15 + 0.02 * r["c"], 4)
    half_y = round(size_y / 2.0, 4)
    center_y = round(half_y - hole_depth / 2.0, 4)
    body = [
        {"id": "body", "op": "box", "combine": "new", "center": [0.0, 0.0, 0.0], "size": [0.80, size_y, 0.42]},
        {"id": "front_cut", "op": "box", "combine": "cut", "center": [0.18, -half_y + 0.06, 0.10], "size": [0.22, 0.12, 0.14]},
    ]
    back_hole = {
        "id": "back_hole",
        "op": "hole",
        "combine": "cut",
        "workplane": "XZ",
        "center": [-0.12, center_y, 0.0],
        "diameter": hole_d,
        "depth": hole_depth,
    }
    crit_present = {
        "fact_id": "back_hole.hidden_presence",
        "category": "hidden_presence",
        "value": True,
        "visibility_in_images": False,
        "recoverable_from_pointcloud": True,
        "operation_id": "back_hole",
        "field": "presence",
        "diameter": hole_d,
        "depth": hole_depth,
    }
    crit_absent = dict(crit_present)
    crit_absent["value"] = False
    spec_a = _spec(
        f"{pair_id}a",
        "bracket_back_feature",
        seed,
        copy.deepcopy(body),
        crit_absent,
        pair_id=pair_id,
        variant="A",
        kind="hidden_presence",
    )
    spec_b = _spec(
        f"{pair_id}b",
        "bracket_back_feature",
        seed,
        body + [back_hole],
        crit_present,
        pair_id=pair_id,
        variant="B",
        kind="hidden_presence",
    )
    return {"pair_id": pair_id, "kind": "hidden_presence", "family": "bracket_back_feature", "spec_a": spec_a, "spec_b": spec_b}


def _pocket_depth_pair(pair_id: str, seed: int) -> dict[str, Any]:
    r = _rng_values(seed)
    height = 0.40
    pocket_w = round(0.34 + 0.04 * r["a"], 4)
    pocket_l = round(0.26 + 0.04 * r["b"], 4)
    depth_a, depth_b = 0.12, 0.24
    half_z = round(height / 2.0, 4)

    def ops(depth: float) -> list[dict[str, Any]]:
        pocket_cz = round(half_z - depth / 2.0, 4)
        return [
            {"id": "block", "op": "box", "combine": "new", "center": [0.0, 0.0, 0.0], "size": [0.92, 0.70, height]},
            {
                "id": "pocket",
                "op": "box",
                "combine": "cut",
                "center": [-0.10, 0.04, pocket_cz],
                "size": [pocket_w, pocket_l, depth],
            },
            {
                "id": "end_slot",
                "op": "slot",
                "combine": "cut",
                "workplane": "XY",
                "center": [0.28, -0.12, half_z - 0.04],
                "length": 0.22,
                "width": 0.08,
                "depth": 0.10,
                "angle": 0.0,
            },
        ]

    def critical(depth: float) -> dict[str, Any]:
        return {
            "fact_id": "pocket.depth",
            "category": "depth",
            "value": depth,
            "visibility_in_images": False,
            "recoverable_from_pointcloud": True,
            "operation_id": "pocket",
            "field": "depth",
        }

    spec_a = _spec(f"{pair_id}a", "block_pocket_slot", seed, ops(depth_a), critical(depth_a), pair_id=pair_id, variant="A", kind="pocket_depth")
    spec_b = _spec(f"{pair_id}b", "block_pocket_slot", seed, ops(depth_b), critical(depth_b), pair_id=pair_id, variant="B", kind="pocket_depth")
    return {"pair_id": pair_id, "kind": "pocket_depth", "family": "block_pocket_slot", "spec_a": spec_a, "spec_b": spec_b}


_BUILDERS = {
    "blind_depth": _blind_depth_pair,
    "through_vs_blind": _through_vs_blind_pair,
    "hidden_presence": _hidden_presence_pair,
    "pocket_depth": _pocket_depth_pair,
}


def pair_id_for(index: int) -> str:
    return f"v6b_probe_{index:04d}"


def generate_pair(index: int) -> dict[str, Any]:
    kind = PAIR_KINDS[index % len(PAIR_KINDS)]
    seed = PAIR_SEED_BASE + index
    return _BUILDERS[kind](pair_id_for(index), seed)


def generate_pairs(n: int) -> list[dict[str, Any]]:
    return [generate_pair(index) for index in range(n)]


def operations_differ_only_in_critical(pair: dict[str, Any]) -> bool:
    """Shared ops equal except the pre-registered critical operation(s)."""
    ops_a = pair["spec_a"]["operations"]
    ops_b = pair["spec_b"]["operations"]
    kind = pair["kind"]
    if kind == "hidden_presence":
        ids_a = {op["id"] for op in ops_a}
        ids_b = {op["id"] for op in ops_b}
        return ids_b - ids_a == {"back_hole"} and ids_a <= ids_b and all(
            op in ops_a or op["id"] == "back_hole" for op in ops_b
        )
    if len(ops_a) != len(ops_b):
        return False
    critical_ids = {
        "blind_depth": {"hole_blind"},
        "through_vs_blind": {"hole_blind"},
        "pocket_depth": {"pocket"},
    }[kind]
    for left, right in zip(ops_a, ops_b):
        if left.get("id") in critical_ids:
            continue
        if left != right:
            return False
    return True
