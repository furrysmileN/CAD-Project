"""V7 shape-transfer pairs: new hosts x old edit types. Existing DSL only.

Must not clone V6b plate_holes / block_pocket_slot / bracket_back_feature shells.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable

GENERATOR_VERSION = "rq2.v7.pair.v1"
PAIR_SEED_BASE = 40_000
HOSTS = (
    "flange_neck",
    "stepped_shaft_collar",
    "l_frame",
    "u_channel",
)
KINDS = (
    "blind_depth",
    "through_vs_blind",
    "hidden_presence",
    "pocket_depth",
)
KIND_LAYER = {
    "pocket_depth": "L1",
    "blind_depth": "L1",
    "through_vs_blind": "L2",
    "hidden_presence": "L3",
}
FORBIDDEN_FAMILIES = frozenset({"plate_holes", "block_pocket_slot", "bracket_back_feature"})
MIN_OPS = 5


def _rng_values(seed: int) -> dict[str, float]:
    digest = hashlib.sha256(f"v7-pair:{seed}".encode("utf-8")).digest()
    floats = [b / 255.0 for b in digest[:8]]
    return {"a": floats[0], "b": floats[1], "c": floats[2], "d": floats[3], "e": floats[4]}


def _plan(sample_id: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "harnesscad.plan.v2",
        "sample_id": sample_id,
        "coordinate_system": {
            "units": "normalized",
            "origin": [0.0, 0.0, 0.0],
            "longest_bbox_edge": 1.0,
        },
        "metadata": {"suite": "v7", "generator": GENERATOR_VERSION},
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
    if family in FORBIDDEN_FAMILIES:
        raise ValueError(f"V7 禁止旧模板 family: {family}")
    return {
        "schema_version": GENERATOR_VERSION,
        "sample_id": sample_id,
        "pair_id": pair_id,
        "variant": variant,
        "kind": kind,
        "family": family,
        "host": family,
        "edit_layer": KIND_LAYER[kind],
        "split": "v7_probe",
        "generator_seed": seed,
        "operations": operations,
        "gt_plan": _plan(sample_id, operations),
        "critical_fact": critical,
        "image_cameras": ["front", "side", "top", "isometric"],
        "notes": {
            "visibility_policy": "critical fact is designed to be unresolvable in the four frozen RGB views",
            "pair": "A and B differ only in the pre-registered latent f",
            "transfer": "host is not a V6b template shell",
        },
    }


def _hole(op_id: str, workplane: str, center: list[float], diameter: float, depth: float) -> dict[str, Any]:
    return {
        "id": op_id,
        "op": "hole",
        "combine": "cut",
        "workplane": workplane,
        "center": [round(c, 4) for c in center],
        "diameter": round(diameter, 4),
        "depth": round(depth, 4),
    }


def _box(op_id: str, combine: str, center: list[float], size: list[float]) -> dict[str, Any]:
    return {
        "id": op_id,
        "op": "box",
        "combine": combine,
        "center": [round(c, 4) for c in center],
        "size": [round(s, 4) for s in size],
    }


def _cylinder(op_id: str, combine: str, center: list[float], radius: float, height: float) -> dict[str, Any]:
    return {
        "id": op_id,
        "op": "cylinder",
        "combine": combine,
        "center": [round(c, 4) for c in center],
        "radius": round(radius, 4),
        "height": round(height, 4),
        "axis": [0.0, 0.0, 1.0],
    }


def _slot(op_id: str, center: list[float], length: float, width: float, depth: float) -> dict[str, Any]:
    return {
        "id": op_id,
        "op": "slot",
        "combine": "cut",
        "workplane": "XY",
        "center": [round(c, 4) for c in center],
        "length": round(length, 4),
        "width": round(width, 4),
        "depth": round(depth, 4),
        "angle": 0.0,
    }


class HostLayout:
    def __init__(self, family: str, distractors: list[dict[str, Any]], meta: dict[str, Any]):
        self.family = family
        self.distractors = distractors
        self.meta = meta


def _flange_neck(r: dict[str, float], kind: str) -> HostLayout:
    disk_h = 0.36
    hub_h = 0.18
    disk_r = 0.48
    hub_r = round(0.15 + 0.02 * r["a"], 4)
    pad_y = round(disk_r + 0.06, 4)
    half_z = disk_h / 2.0
    solids = [
        _cylinder("disk", "new", [0.0, 0.0, 0.0], disk_r, disk_h),
        _cylinder("hub", "add", [0.0, 0.0, round(-(half_z + hub_h / 2.0), 4)], hub_r, hub_h),
        _box("rim_pad", "add", [0.0, pad_y, 0.0], [0.44, 0.12, 0.32]),
        _box("square_boss", "add", [-0.28, -0.28, round(-half_z - 0.04, 4)], [0.16, 0.16, 0.08]),
        _box("clip", "add", [0.30, -0.30, round(-half_z - 0.03, 4)], [0.12, 0.12, 0.06]),
    ]
    if kind == "through_vs_blind":
        solids = [
            _cylinder("disk", "new", [0.0, 0.0, 0.0], disk_r, disk_h),
            _box("rim_pad", "add", [0.0, pad_y, 0.0], [0.44, 0.12, disk_h]),
            _box("square_boss", "add", [-0.28, -0.28, 0.0], [0.16, 0.16, disk_h]),
            _box("lobe", "add", [0.32, 0.22, 0.0], [0.14, 0.18, disk_h]),
        ]
        y_max = pad_y + 0.06
        return HostLayout(
            "flange_neck",
            solids,
            {
                "xy_hole": [round(0.18 + 0.03 * r["d"], 4), round(-0.12 + 0.03 * r["e"], 4)],
                "top_z": half_z,
                "body_h": disk_h,
                "through_center_z": 0.0,
                "pocket_xy": [0.20, -0.14],
                "pocket_wh": [0.22, 0.16],
                "back_y_max": y_max,
                "back_hole_xz": [-0.08, 0.0],
            },
        )
    y_max = pad_y + 0.06
    return HostLayout(
        "flange_neck",
        solids,
        {
            "xy_hole": [round(0.22 + 0.03 * r["d"], 4), round(-0.16 + 0.03 * r["e"], 4)],
            "top_z": half_z,
            "body_h": disk_h,
            "through_center_z": 0.0,
            "pocket_xy": [0.20, -0.14],
            "pocket_wh": [0.22, 0.16],
            "back_y_max": y_max,
            "back_hole_xz": [-0.08, 0.0],
        },
    )


def _stepped_shaft_collar(r: dict[str, float], kind: str) -> HostLayout:
    r_small = round(0.13 + 0.02 * r["a"], 4)
    r_large = round(0.22 + 0.02 * r["b"], 4)
    profile = [
        [0.0, -0.48],
        [r_small, -0.48],
        [r_small, 0.0],
        [r_large, 0.0],
        [r_large, 0.48],
        [0.0, 0.48],
        [0.0, -0.48],
    ]
    solids = [
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
        _cylinder("collar", "add", [0.0, 0.0, -0.40], round(r_small + 0.06, 4), 0.16),
        _box("end_pad", "add", [0.0, round(r_large + 0.08, 4), 0.28], [0.42, 0.12, 0.30]),
        _box("foot", "add", [0.0, 0.0, -0.50], [0.18, 0.18, 0.08]),
        _box("clip", "add", [round(-r_large - 0.05, 4), 0.0, -0.48], [0.10, 0.10, 0.06]),
    ]
    if kind == "through_vs_blind":
        solids = [
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
            _box("end_pad", "add", [0.0, round(r_large + 0.08, 4), 0.0], [0.42, 0.12, 0.96]),
            _box("key_block", "add", [round(r_large + 0.05, 4), 0.0, 0.0], [0.10, 0.14, 0.96]),
            _box("lobe", "add", [0.0, round(-r_large - 0.06, 4), 0.0], [0.16, 0.10, 0.96]),
        ]
        y_max = r_large + 0.08 + 0.06
        return HostLayout(
            "stepped_shaft_collar",
            solids,
            {
                "xy_hole": [0.0, 0.0],
                "top_z": 0.48,
                "body_h": 0.96,
                "through_center_z": 0.0,
                "pocket_xy": [0.06, 0.06],
                "pocket_wh": [0.12, 0.12],
                "back_y_max": y_max,
                "back_hole_xz": [-0.06, 0.26],
            },
        )
    y_max = r_large + 0.08 + 0.06
    xy_hole = [0.0, 0.0] if kind == "through_vs_blind" else [0.10, -0.06]
    return HostLayout(
        "stepped_shaft_collar",
        solids,
        {
            "xy_hole": xy_hole,
            "top_z": 0.48,
            "body_h": 0.96,
            "through_center_z": 0.0,
            "pocket_xy": [0.06, 0.06],
            "pocket_wh": [0.12, 0.12],
            "back_y_max": y_max,
            "back_hole_xz": [-0.06, 0.26],
        },
    )


def _l_frame(r: dict[str, float], kind: str) -> HostLayout:
    shelf_h = 0.32
    upright_t = 0.14
    upright = _box("upright", "new", [0.0, 0.22, 0.0], [0.88, upright_t, 0.48])
    shelf = _box("shelf", "add", [0.0, -0.04, 0.31], [0.88, 0.52, shelf_h])
    rib = _box("rib", "add", [0.28, 0.08, 0.12], [0.10, 0.24, 0.20])
    foot = _box("foot", "add", [0.0, 0.22, -0.28], [0.30, 0.20, 0.08])
    solids = [upright, shelf, rib, foot]
    solids.append(_box("clip", "add", [-0.36, 0.22, -0.28], [0.12, 0.12, 0.06]))
    if kind == "through_vs_blind":
        h = 0.32
        solids = [
            _box("shelf", "new", [0.0, -0.06, 0.0], [0.88, 0.40, h]),
            _box("wing", "add", [0.0, 0.28, 0.0], [0.36, 0.28, h]),
            _box("rib", "add", [0.30, 0.08, 0.0], [0.10, 0.18, h]),
            _box("foot", "add", [-0.32, 0.28, 0.0], [0.16, 0.16, h]),
        ]
        y_max = 0.28 + 0.14
        return HostLayout(
            "l_frame",
            solids,
            {
                "xy_hole": [round(0.10 + 0.03 * r["c"], 4), round(-0.04 + 0.03 * r["d"], 4)],
                "top_z": round(h / 2.0, 4),
                "body_h": h,
                "through_center_z": 0.0,
                "pocket_xy": [0.14, 0.02],
                "pocket_wh": [0.24, 0.16],
                "back_y_max": y_max,
                "back_hole_xz": [round(-0.10 + 0.04 * r["e"], 4), 0.0],
            },
        )
    y_max = 0.22 + upright_t / 2.0
    return HostLayout(
        "l_frame",
        solids,
        {
            "xy_hole": [round(0.16 + 0.03 * r["c"], 4), round(-0.06 + 0.03 * r["d"], 4)],
            "top_z": round(0.31 + shelf_h / 2.0, 4),
            "body_h": shelf_h,
            "through_center_z": 0.31,
            "pocket_xy": [0.14, 0.02],
            "pocket_wh": [0.24, 0.16],
            "back_y_max": y_max,
            "back_hole_xz": [round(-0.10 + 0.04 * r["e"], 4), 0.04],
        },
    )


def _u_channel(r: dict[str, float], kind: str) -> HostLayout:
    top_h = 0.32
    top = _box("top_plate", "new", [0.0, -0.04, 0.22], [0.82, 0.40, top_h])
    left = _box("left_leg", "add", [-0.36, -0.04, -0.04], [0.10, 0.40, 0.38])
    right = _box("right_leg", "add", [0.36, -0.04, -0.04], [0.10, 0.40, 0.38])
    rear = _box("rear_wall", "add", [0.0, 0.22, 0.04], [0.82, 0.12, 0.50])
    solids = [top, left, right, rear]
    solids.append(_box("clip", "add", [0.0, -0.04, -0.20], [0.12, 0.12, 0.06]))
    if kind == "through_vs_blind":
        h = 0.32
        solids = [
            _box("top_plate", "new", [0.0, 0.0, 0.0], [0.50, 0.82, h]),
            _box("left_leg", "add", [-0.36, 0.0, 0.0], [0.22, 0.82, h]),
            _box("right_leg", "add", [0.36, 0.0, 0.0], [0.22, 0.82, h]),
            _box("rear_wall", "add", [0.0, 0.40, 0.0], [0.82, 0.18, h]),
        ]
        y_max = 0.40 + 0.09
        return HostLayout(
            "u_channel",
            solids,
            {
                "xy_hole": [0.0, round(-0.10 + 0.03 * r["b"], 4)],
                "top_z": round(h / 2.0, 4),
                "body_h": h,
                "through_center_z": 0.0,
                "pocket_xy": [0.08, 0.0],
                "pocket_wh": [0.24, 0.14],
                "back_y_max": y_max,
                "back_hole_xz": [round(-0.10 + 0.04 * r["c"], 4), 0.0],
            },
        )
    y_max = 0.22 + 0.06
    return HostLayout(
        "u_channel",
        solids,
        {
            "xy_hole": [round(0.10 + 0.03 * r["a"], 4), round(-0.06 + 0.03 * r["b"], 4)],
            "top_z": round(0.22 + top_h / 2.0, 4),
            "body_h": top_h,
            "through_center_z": 0.22,
            "pocket_xy": [0.08, 0.0],
            "pocket_wh": [0.24, 0.14],
            "back_y_max": y_max,
            "back_hole_xz": [round(-0.10 + 0.04 * r["c"], 4), 0.06],
        },
    )


_HOST_BUILDERS: dict[str, Callable[..., HostLayout]] = {
    "flange_neck": _flange_neck,
    "stepped_shaft_collar": _stepped_shaft_collar,
    "l_frame": _l_frame,
    "u_channel": _u_channel,
}

def _blind_ops(layout: HostLayout, depth: float, diameter: float) -> list[dict[str, Any]]:
    x, y = layout.meta["xy_hole"]
    top = float(layout.meta["top_z"])
    center_z = round(top - depth / 2.0, 4)
    return layout.distractors + [_hole("hole_blind", "XY", [x, y, center_z], diameter, depth)]


def _through_ops(layout: HostLayout, depth: float, diameter: float, *, through: bool) -> list[dict[str, Any]]:
    x, y = layout.meta["xy_hole"]
    if through:
        center_z = float(layout.meta.get("through_center_z") or 0.0)
    else:
        top = float(layout.meta["top_z"])
        center_z = round(top - depth / 2.0, 4)
    return layout.distractors + [_hole("hole_blind", "XY", [x, y, center_z], diameter, depth)]


def _pocket_ops(layout: HostLayout, depth: float) -> list[dict[str, Any]]:
    px, py = layout.meta["pocket_xy"]
    w, l = layout.meta["pocket_wh"]
    top = float(layout.meta["top_z"])
    cz = round(top - depth / 2.0, 4)
    return layout.distractors + [_box("pocket", "cut", [px, py, cz], [w, l, depth])]


def _hidden_ops(layout: HostLayout, *, present: bool, diameter: float, depth: float) -> list[dict[str, Any]]:
    ops = list(layout.distractors)
    if not present:
        return ops
    y_max = float(layout.meta["back_y_max"])
    xz = layout.meta["back_hole_xz"]
    center_y = round(y_max - depth / 2.0, 4)
    ops.append(_hole("back_hole", "XZ", [xz[0], center_y, xz[1]], diameter, depth))
    return ops


def _make_pair(pair_id: str, seed: int, host: str, kind: str) -> dict[str, Any]:
    r = _rng_values(seed)
    layout = _HOST_BUILDERS[host](r, kind)
    diameter = round(0.11 + 0.02 * r["a"], 4)
    if kind == "blind_depth":
        depth_a, depth_b = 0.12, 0.28
        ops_a = _blind_ops(layout, depth_a, diameter)
        ops_b = _blind_ops(layout, depth_b, diameter)

        def crit(depth: float) -> dict[str, Any]:
            return {
                "fact_id": "hole_blind.depth",
                "category": "depth",
                "value": depth,
                "visibility_in_images": False,
                "recoverable_from_pointcloud": True,
                "operation_id": "hole_blind",
                "field": "depth",
            }

        spec_a = _spec(f"{pair_id}a", host, seed, ops_a, crit(depth_a), pair_id=pair_id, variant="A", kind=kind)
        spec_b = _spec(f"{pair_id}b", host, seed, ops_b, crit(depth_b), pair_id=pair_id, variant="B", kind=kind)
    elif kind == "through_vs_blind":
        blind_d = 0.12
        through_d = round(float(layout.meta["body_h"]) + 0.12, 4)
        ops_a = _through_ops(layout, through_d, diameter, through=True)
        ops_b = _through_ops(layout, blind_d, diameter, through=False)
        crit_a = {
            "fact_id": "hole_blind.through_vs_blind",
            "category": "through_vs_blind",
            "value": "through",
            "visibility_in_images": False,
            "recoverable_from_pointcloud": True,
            "operation_id": "hole_blind",
            "field": "through_vs_blind",
            "depth": through_d,
        }
        crit_b = {**crit_a, "value": "blind", "depth": blind_d}
        spec_a = _spec(f"{pair_id}a", host, seed, ops_a, crit_a, pair_id=pair_id, variant="A", kind=kind)
        spec_b = _spec(f"{pair_id}b", host, seed, ops_b, crit_b, pair_id=pair_id, variant="B", kind=kind)
    elif kind == "hidden_presence":
        hole_d = round(0.10 + 0.015 * r["b"], 4)
        hole_depth = 0.14
        ops_a = _hidden_ops(layout, present=False, diameter=hole_d, depth=hole_depth)
        ops_b = _hidden_ops(layout, present=True, diameter=hole_d, depth=hole_depth)

        def crit(present: bool) -> dict[str, Any]:
            return {
                "fact_id": "back_hole.hidden_presence",
                "category": "hidden_presence",
                "value": present,
                "visibility_in_images": False,
                "recoverable_from_pointcloud": True,
                "operation_id": "back_hole",
                "field": "presence",
                "diameter": hole_d,
                "depth": hole_depth,
            }

        spec_a = _spec(f"{pair_id}a", host, seed, ops_a, crit(False), pair_id=pair_id, variant="A", kind=kind)
        spec_b = _spec(f"{pair_id}b", host, seed, ops_b, crit(True), pair_id=pair_id, variant="B", kind=kind)
    elif kind == "pocket_depth":
        depth_a, depth_b = 0.10, 0.22
        ops_a = _pocket_ops(layout, depth_a)
        ops_b = _pocket_ops(layout, depth_b)

        def crit(depth: float) -> dict[str, Any]:
            return {
                "fact_id": "pocket.depth",
                "category": "depth",
                "value": depth,
                "visibility_in_images": False,
                "recoverable_from_pointcloud": True,
                "operation_id": "pocket",
                "field": "depth",
            }

        spec_a = _spec(f"{pair_id}a", host, seed, ops_a, crit(depth_a), pair_id=pair_id, variant="A", kind=kind)
        spec_b = _spec(f"{pair_id}b", host, seed, ops_b, crit(depth_b), pair_id=pair_id, variant="B", kind=kind)
    else:
        raise ValueError(f"未知 kind: {kind}")
    if len(spec_b["operations"]) < MIN_OPS:
        raise RuntimeError(f"{pair_id} 操作数不足")
    return {
        "pair_id": pair_id,
        "kind": kind,
        "family": host,
        "host": host,
        "edit_layer": KIND_LAYER[kind],
        "spec_a": spec_a,
        "spec_b": spec_b,
    }


def pair_id_for(index: int) -> str:
    return f"v7_probe_{index:04d}"


def generate_pair(index: int) -> dict[str, Any]:
    host = HOSTS[index // len(KINDS) % len(HOSTS)]
    kind = KINDS[index % len(KINDS)]
    seed = PAIR_SEED_BASE + index
    return _make_pair(pair_id_for(index), seed, host, kind)


def generate_pairs(n: int) -> list[dict[str, Any]]:
    return [generate_pair(index) for index in range(n)]


def operations_differ_only_in_critical(pair: dict[str, Any]) -> bool:
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


def assert_not_v6b_shell(pair: dict[str, Any]) -> None:
    if pair["family"] in FORBIDDEN_FAMILIES:
        raise AssertionError(pair["family"])
    ids = {op["id"] for op in pair["spec_a"]["operations"]}
    if ids <= {"base", "hole_blind", "hole_through"}:
        raise AssertionError("looks like plate_holes")
    if ids <= {"block", "pocket", "end_slot"}:
        raise AssertionError("looks like block_pocket_slot")
    if ids <= {"body", "front_cut", "back_hole"}:
        raise AssertionError("looks like bracket_back_feature")
