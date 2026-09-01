"""10 条手写 FeaturePlan v3 oracle，供编译器门控。"""
from __future__ import annotations

from typing import Any

SQUARE = [[-0.12, -0.12], [0.12, -0.12], [0.12, 0.12], [-0.12, 0.12], [-0.12, -0.12]]
RECT = [[-0.35, -0.2], [0.35, -0.2], [0.35, 0.2], [-0.35, 0.2], [-0.35, -0.2]]
ARC_WIRE = [
    {"kind": "move", "to": [-0.18, 0.0]},
    {"kind": "three_point_arc", "through": [0.0, 0.18], "to": [0.18, 0.0]},
    {"kind": "three_point_arc", "through": [0.0, -0.18], "to": [-0.18, 0.0]},
]


def _base(sample_id: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "harnesscad.plan.v3",
        "sample_id": sample_id,
        "coordinate_system": {"units": "normalized", "origin": [0.0, 0.0, 0.0], "longest_bbox_edge": 1.0},
        "operations": operations,
    }


ORACLES: list[dict[str, Any]] = [
    _base("oracle_box_hole", [
        {"id": "base", "op": "box", "combine": "new", "center": [0.0, 0.0, 0.0], "size": [0.8, 0.5, 0.16]},
        {"id": "bore", "op": "hole", "combine": "cut", "workplane": "XY", "center": [0.0, 0.0, 0.0], "diameter": 0.16, "depth": 0.4},
    ]),
    _base("oracle_cylinder", [
        {"id": "shaft", "op": "cylinder", "combine": "new", "center": [0.0, 0.0, 0.0], "radius": 0.18, "height": 0.7, "axis": [0.0, 0.0, 1.0]},
    ]),
    _base("oracle_revolve", [
        {
            "id": "knob",
            "op": "revolve_profile",
            "combine": "new",
            "workplane": "XZ",
            "profile": [[0.08, -0.2], [0.22, -0.2], [0.22, 0.2], [0.08, 0.2], [0.08, -0.2]],
            "axis": [[0.0, -1.0], [0.0, 1.0]],
            "angle": 360.0,
            "offset": [0.0, 0.0, 0.0],
        }
    ]),
    _base("oracle_polygon", [
        {
            "id": "plate",
            "op": "polygon_extrude",
            "combine": "new",
            "workplane": "XY",
            "points": RECT,
            "depth": 0.1,
            "centered": True,
            "offset": [0.0, 0.0, 0.0],
        }
    ]),
    _base("oracle_arc_disk", [
        {
            "id": "disk",
            "op": "polygon_extrude",
            "combine": "new",
            "workplane": "XY",
            "wire": ARC_WIRE,
            "depth": 0.1,
            "centered": True,
            "offset": [0.0, 0.0, 0.0],
        }
    ]),
    _base("oracle_sweep_path", [
        {
            "id": "elbow",
            "op": "sweep_profile",
            "combine": "new",
            "workplane": "XY",
            "profile": SQUARE,
            "path": [[0.0, 0.0, -0.3], [0.15, 0.0, 0.0], [0.0, 0.0, 0.3]],
            "offset": [0.0, 0.0, 0.0],
        }
    ]),
    _base("oracle_sweep_helix", [
        {
            "id": "coil",
            "op": "sweep_profile",
            "combine": "new",
            "workplane": "XY",
            "profile": [[0.16, -0.025], [0.22, -0.025], [0.22, 0.025], [0.16, 0.025], [0.16, -0.025]],
            "helix": {"radius": 0.19, "pitch": 0.07, "turns": 3.0, "axis": [0.0, 0.0, 1.0], "center": [0.0, 0.0, 0.0]},
            "offset": [0.0, 0.0, 0.0],
        }
    ]),
    _base("oracle_loft", [
        {
            "id": "taper",
            "op": "loft_profiles",
            "combine": "new",
            "workplane": "XY",
            "profiles": [
                {"points": [[-0.28, -0.28], [0.28, -0.28], [0.28, 0.28], [-0.28, 0.28], [-0.28, -0.28]], "offset": [0.0, 0.0, -0.18]},
                {"points": [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1]], "offset": [0.0, 0.0, 0.18]},
            ],
        }
    ]),
    _base("oracle_sweep_cut", [
        {"id": "block", "op": "box", "combine": "new", "center": [0.0, 0.0, 0.0], "size": [0.7, 0.4, 0.24]},
        {
            "id": "tunnel",
            "op": "sweep_profile",
            "combine": "cut",
            "workplane": "YZ",
            "profile": [[-0.06, -0.06], [0.06, -0.06], [0.06, 0.06], [-0.06, 0.06], [-0.06, -0.06]],
            "path": [[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]],
            "offset": [0.0, 0.0, 0.0],
        },
    ]),
    _base("oracle_loft_fillet", [
        {
            "id": "body",
            "op": "loft_profiles",
            "combine": "new",
            "workplane": "XY",
            "profiles": [
                {"points": [[-0.25, -0.18], [0.25, -0.18], [0.25, 0.18], [-0.25, 0.18], [-0.25, -0.18]], "offset": [0.0, 0.0, -0.15]},
                {"points": [[-0.18, -0.12], [0.18, -0.12], [0.18, 0.12], [-0.18, 0.12], [-0.18, -0.12]], "offset": [0.0, 0.0, 0.15]},
            ],
        },
        {"id": "round", "op": "fillet", "radius": 0.02},
    ]),
]
