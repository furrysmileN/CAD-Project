"""Programmatic V6 latent specs and GT Plans. Latent specs never enter API payloads."""
from __future__ import annotations

import hashlib
from typing import Any

FAMILIES = (
    "plate_holes",
    "flange_array",
    "stepped_shaft",
    "bracket_back_feature",
    "block_pocket_slot",
)

FACT_CATEGORIES = (
    "depth",
    "through_vs_blind",
    "hidden_presence",
    "radius_or_width",
    "offset_or_spacing",
    "axis_or_symmetry",
)

PILOT_SEED_BASE = 10_000
CONFIRM_SEED_BASE = 20_000
GENERATOR_VERSION = "rq2.v6.latent.v1"


def _rng_values(seed: int) -> dict[str, float]:
    digest = hashlib.sha256(f"v6-latent:{seed}".encode("utf-8")).digest()
    floats = [b / 255.0 for b in digest[:12]]
    return {
        "a": floats[0],
        "b": floats[1],
        "c": floats[2],
        "d": floats[3],
        "e": floats[4],
        "sign": 1.0 if floats[5] >= 0.5 else -1.0,
        "flip": floats[6] >= 0.5,
        "tier": floats[7],
    }


def _difficulty(tier: float) -> str:
    if tier < 0.34:
        return "easy"
    if tier < 0.67:
        return "medium"
    return "hard"


def _plan(sample_id: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "harnesscad.plan.v2",
        "sample_id": sample_id,
        "coordinate_system": {
            "units": "normalized",
            "origin": [0.0, 0.0, 0.0],
            "longest_bbox_edge": 1.0,
        },
        "metadata": {"suite": "v6", "generator": GENERATOR_VERSION},
        "operations": operations,
    }


def _spec(
    sample_id: str,
    family: str,
    seed: int,
    difficulty: str,
    operations: list[dict[str, Any]],
    critical: dict[str, Any],
    secondary: list[dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    return {
        "schema_version": GENERATOR_VERSION,
        "sample_id": sample_id,
        "family": family,
        "split": split,
        "difficulty": difficulty,
        "generator_seed": seed,
        "operations": operations,
        "gt_plan": _plan(sample_id, operations),
        "critical_fact": critical,
        "secondary_facts": secondary,
        "image_cameras": ["front", "side", "top", "isometric"],
        "notes": {
            "visibility_policy": "critical fact is designed to be absent from the four frozen RGB views",
            "dsl": "plan_v2; flange_array uses a rectangular hole grid, not circular_pattern",
        },
    }


def _plate_holes(sample_id: str, seed: int, split: str) -> dict[str, Any]:
    r = _rng_values(seed)
    thickness = round(0.16 + 0.04 * r["a"], 4)
    blind_depth = round(0.07 + 0.04 * r["b"], 4)
    hole_d = round(0.12 + 0.04 * r["c"], 4)
    through_d = round(0.10 + 0.03 * r["d"], 4)
    x1 = round(-0.22 + 0.04 * r["e"], 4)
    x2 = round(0.22 - 0.04 * r["a"], 4)
    half_z = round(thickness / 2.0, 4)
    blind_center_z = round(half_z - blind_depth / 2.0, 4)
    through_depth = round(thickness + 0.08, 4)
    ops = [
        {
            "id": "base",
            "op": "box",
            "combine": "new",
            "center": [0.0, 0.0, 0.0],
            "size": [1.0, 0.72, thickness],
        },
        {
            "id": "hole_blind",
            "op": "hole",
            "combine": "cut",
            "workplane": "XY",
            "center": [x1, 0.12, blind_center_z],
            "diameter": hole_d,
            "depth": blind_depth,
        },
        {
            "id": "hole_through",
            "op": "hole",
            "combine": "cut",
            "workplane": "XY",
            "center": [x2, -0.10, 0.0],
            "diameter": through_d,
            "depth": through_depth,
        },
    ]
    use_depth = r["flip"]
    if use_depth:
        critical = {
            "fact_id": "hole_blind.depth",
            "category": "depth",
            "value": blind_depth,
            "visibility_in_images": False,
            "recoverable_from_pointcloud": True,
            "operation_id": "hole_blind",
            "field": "depth",
        }
    else:
        critical = {
            "fact_id": "hole_blind.through_vs_blind",
            "category": "through_vs_blind",
            "value": "blind",
            "visibility_in_images": False,
            "recoverable_from_pointcloud": True,
            "operation_id": "hole_blind",
            "field": "through_vs_blind",
        }
    secondary = [
        {
            "fact_id": "hole_through.diameter",
            "category": "radius_or_width",
            "value": through_d,
            "visibility_in_images": True,
        }
    ]
    return _spec(sample_id, "plate_holes", seed, _difficulty(r["tier"]), ops, critical, secondary, split=split)


def _flange_array(sample_id: str, seed: int, split: str) -> dict[str, Any]:
    r = _rng_values(seed)
    height = round(0.12 + 0.03 * r["a"], 4)
    center_d = round(0.20 + 0.04 * r["b"], 4)
    bolt_d = round(0.08 + 0.02 * r["c"], 4)
    spacing = round(0.26 + 0.06 * r["d"], 4)
    ops = [
        {
            "id": "disk",
            "op": "cylinder",
            "combine": "new",
            "center": [0.0, 0.0, 0.0],
            "radius": 0.50,
            "height": height,
            "axis": [0.0, 0.0, 1.0],
        },
        {
            "id": "center_hole",
            "op": "hole",
            "combine": "cut",
            "workplane": "XY",
            "center": [0.0, 0.0, 0.0],
            "diameter": center_d,
            "depth": round(height + 0.06, 4),
        },
    ]
    half = round(spacing / 2.0, 4)
    bolt_depth = round(height + 0.06, 4)
    for index, (x, y) in enumerate(((-half, -half), (half, -half), (-half, half), (half, half))):
        ops.append(
            {
                "id": f"bolt_{index}",
                "op": "hole",
                "combine": "cut",
                "workplane": "XY",
                "center": [x, y, 0.0],
                "diameter": bolt_d,
                "depth": bolt_depth,
            }
        )
    critical = {
        "fact_id": "bolt_grid.spacing",
        "category": "offset_or_spacing",
        "value": spacing,
        "visibility_in_images": False,
        "recoverable_from_pointcloud": True,
        "operation_id": "bolt_0",
        "field": "spacing",
        "note": "true spacing is recoverable from point-cloud hole centers; top RGB does not include numeric spacing",
    }
    secondary = [
        {"fact_id": "center_hole.diameter", "category": "radius_or_width", "value": center_d, "visibility_in_images": True}
    ]
    return _spec(sample_id, "flange_array", seed, _difficulty(r["tier"]), ops, critical, secondary, split=split)


def _stepped_shaft(sample_id: str, seed: int, split: str) -> dict[str, Any]:
    r = _rng_values(seed)
    r_small = round(0.12 + 0.03 * r["a"], 4)
    r_large = round(0.22 + 0.04 * r["b"], 4)
    bore = round(0.06 + 0.02 * r["c"], 4)
    step_z = 0.0
    profile = [
        [0.0, -0.50],
        [r_small, -0.50],
        [r_small, step_z],
        [r_large, step_z],
        [r_large, 0.50],
        [0.0, 0.50],
        [0.0, -0.50],
    ]
    ops = [
        {
            "id": "shaft",
            "op": "revolve_profile",
            "combine": "new",
            "workplane": "XZ",
            "profile": [[p[0], p[1]] for p in profile],
            "axis": [[0.0, -1.0], [0.0, 1.0]],
            "angle": 360.0,
            "offset": [0.0, 0.0, 0.0],
        },
        {
            "id": "bore",
            "op": "hole",
            "combine": "cut",
            "workplane": "XY",
            "center": [0.0, 0.0, 0.0],
            "diameter": bore * 2.0,
            "depth": 1.10,
        },
    ]
    if r["flip"]:
        critical = {
            "fact_id": "shaft.step_radius",
            "category": "radius_or_width",
            "value": r_large,
            "visibility_in_images": False,
            "recoverable_from_pointcloud": True,
            "field": "radius",
        }
    else:
        critical = {
            "fact_id": "shaft.axis",
            "category": "axis_or_symmetry",
            "value": [0.0, 0.0, 1.0],
            "visibility_in_images": False,
            "recoverable_from_pointcloud": True,
            "field": "axis",
        }
    secondary = [
        {"fact_id": "shaft.small_radius", "category": "radius_or_width", "value": r_small, "visibility_in_images": True}
    ]
    return _spec(sample_id, "stepped_shaft", seed, _difficulty(r["tier"]), ops, critical, secondary, split=split)


def _bracket_back_feature(sample_id: str, seed: int, split: str) -> dict[str, Any]:
    r = _rng_values(seed)
    size_y = round(0.46 + 0.06 * r["a"], 4)
    hole_d = round(0.12 + 0.03 * r["b"], 4)
    hole_depth = round(0.14 + 0.04 * r["c"], 4)
    half_y = round(size_y / 2.0, 4)
    # XZ hole axis is (0, -1, 0). Back face at +Y; hole cuts inward toward -Y.
    center_y = round(half_y - hole_depth / 2.0, 4)
    ops = [
        {
            "id": "body",
            "op": "box",
            "combine": "new",
            "center": [0.0, 0.0, 0.0],
            "size": [0.80, size_y, 0.42],
        },
        {
            "id": "front_cut",
            "op": "box",
            "combine": "cut",
            "center": [0.18, -half_y + 0.06, 0.10],
            "size": [0.22, 0.12, 0.14],
        },
        {
            "id": "back_hole",
            "op": "hole",
            "combine": "cut",
            "workplane": "XZ",
            "center": [-0.12, center_y, 0.0],
            "diameter": hole_d,
            "depth": hole_depth,
        },
    ]
    critical = {
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
    secondary = [
        {"fact_id": "front_cut.size", "category": "radius_or_width", "value": 0.22, "visibility_in_images": True}
    ]
    return _spec(
        sample_id, "bracket_back_feature", seed, _difficulty(r["tier"]), ops, critical, secondary, split=split
    )


def _block_pocket_slot(sample_id: str, seed: int, split: str) -> dict[str, Any]:
    r = _rng_values(seed)
    height = round(0.36 + 0.06 * r["a"], 4)
    pocket_depth = round(0.12 + 0.05 * r["b"], 4)
    pocket_w = round(0.30 + 0.06 * r["c"], 4)
    pocket_l = round(0.22 + 0.05 * r["d"], 4)
    half_z = round(height / 2.0, 4)
    pocket_cz = round(half_z - pocket_depth / 2.0, 4)
    ops = [
        {
            "id": "block",
            "op": "box",
            "combine": "new",
            "center": [0.0, 0.0, 0.0],
            "size": [0.92, 0.70, height],
        },
        {
            "id": "pocket",
            "op": "box",
            "combine": "cut",
            "center": [-0.10, 0.04, pocket_cz],
            "size": [pocket_w, pocket_l, pocket_depth],
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
    critical = {
        "fact_id": "pocket.depth",
        "category": "depth",
        "value": pocket_depth,
        "visibility_in_images": False,
        "recoverable_from_pointcloud": True,
        "operation_id": "pocket",
        "field": "depth",
    }
    secondary = [
        {"fact_id": "end_slot.width", "category": "radius_or_width", "value": 0.08, "visibility_in_images": True}
    ]
    return _spec(sample_id, "block_pocket_slot", seed, _difficulty(r["tier"]), ops, critical, secondary, split=split)


_BUILDERS = {
    "plate_holes": _plate_holes,
    "flange_array": _flange_array,
    "stepped_shaft": _stepped_shaft,
    "bracket_back_feature": _bracket_back_feature,
    "block_pocket_slot": _block_pocket_slot,
}


def sample_id_for(split: str, index: int) -> str:
    return f"v6_{split}_{index:04d}"


def generate_one(split: str, index: int) -> dict[str, Any]:
    if split not in {"pilot", "confirm"}:
        raise ValueError("split 必须是 pilot 或 confirm")
    family = FAMILIES[index % len(FAMILIES)]
    seed = (PILOT_SEED_BASE if split == "pilot" else CONFIRM_SEED_BASE) + index
    return _BUILDERS[family](sample_id_for(split, index), seed, split)


def generate_split(split: str, n: int) -> list[dict[str, Any]]:
    return [generate_one(split, index) for index in range(n)]


def parameter_signature(spec: dict[str, Any]) -> str:
    critical = spec.get("critical_fact") or {}
    payload = {
        "family": spec.get("family"),
        "operations": spec.get("operations"),
        "critical": {key: critical.get(key) for key in ("category", "value", "field")},
    }
    raw = repr(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
