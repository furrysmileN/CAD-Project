from __future__ import annotations

import numpy as np
import unittest

from rq2_harness.measurement_binder import BindingError, bind_evidence_references
from rq2_harness.pointcloud.centerline import infer_path_graph
from rq2_harness.prompting import parse_plan_response


def _ring(
    center: np.ndarray,
    tangent: list[float],
    radii: tuple[float, float] = (0.08, 0.12),
    n: int = 24,
) -> list[np.ndarray]:
    direction = np.asarray(tangent, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    binormal = np.asarray([0.0, 1.0, 0.0])
    radial = np.cross(direction, binormal)
    radial /= np.linalg.norm(radial)
    return [
        center + radius * (np.cos(angle) * binormal + np.sin(angle) * radial)
        for radius in radii
        for angle in np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ]


def _single_bend_cloud() -> np.ndarray:
    points: list[np.ndarray] = []
    for x_value in np.linspace(0.2, 0.5, 16):
        points.extend(_ring(np.asarray([x_value, 0.0, 0.3]), [-1.0, 0.0, 0.0]))
    for angle in np.linspace(np.pi / 2.0, np.pi, 20):
        center = np.asarray(
            [0.2 + 0.3 * np.cos(angle), 0.0, 0.3 * np.sin(angle)]
        )
        points.extend(
            _ring(center, [-np.sin(angle), 0.0, np.cos(angle)])
        )
    for z_value in np.linspace(-0.5, 0.0, 20):
        points.extend(_ring(np.asarray([-0.1, 0.0, z_value]), [0.0, 0.0, -1.0]))
    return np.asarray(points)


def test_centerline_recovers_line_arc_line_without_family_name() -> None:
    graph = infer_path_graph(_single_bend_cloud())
    assert graph["status"] == "resolved"
    component = graph["components"][0]
    assert component["kind"] == "line_arc_line"
    assert component["confidence"] >= 0.75
    arc = component["segments"][1]
    assert arc["kind"] == "arc"
    assert abs(arc["radius"] - 0.3) < 0.015
    assert arc["fit_p90_error"] < 0.01
    assert abs(component["section_profile"]["inner_radius"] - 0.08) < 0.01
    assert abs(component["section_profile"]["outer_radius"] - 0.12) < 0.01


def test_centerline_abstains_on_solid_surface() -> None:
    rng = np.random.default_rng(7)
    points = rng.uniform(-0.5, 0.5, size=(2000, 3))
    face_axis = rng.integers(0, 3, size=len(points))
    points[np.arange(len(points)), face_axis] = rng.choice(
        [-0.5, 0.5], size=len(points)
    )
    graph = infer_path_graph(points)
    assert graph["status"] == "unresolved"
    assert graph["components"] == []


def test_semantic_path_reference_is_bound_and_unknown_ref_rejected() -> None:
    graph = infer_path_graph(_single_bend_cloud())
    semantic = {
        "schema_version": "harnesscad.plan.v3.1",
        "sample_id": "synthetic",
        "coordinate_system": {
            "units": "normalized",
            "origin": [0, 0, 0],
            "longest_bbox_edge": 1.0,
        },
        "operations": [
            {
                "id": "measured",
                "op": "sweep_path_ref",
                "combine": "new",
                "evidence_ref": "path_01",
            }
        ],
    }
    parsed = parse_plan_response(
        __import__("json").dumps(semantic),
        plan_version="v6",
    )
    assert parsed["ok"]
    bound, audit = bind_evidence_references(semantic, graph)
    assert [item["op"] for item in bound["operations"]] == [
        "sweep_profile",
        "sweep_profile",
    ]
    assert bound["operations"][0]["path_wire"][2]["kind"] == "three_point_arc"
    assert audit[0]["evidence_ref"] == "path_01"

    semantic["operations"][0]["evidence_ref"] = "missing"
    try:
        bind_evidence_references(semantic, graph)
    except BindingError as exc:
        assert exc.issue["code"] == "unknown_evidence_ref"
    else:
        raise AssertionError("unknown evidence reference must fail closed")


class CenterlineV2Tests(unittest.TestCase):
    def test_recovers_line_arc_line(self) -> None:
        test_centerline_recovers_line_arc_line_without_family_name()

    def test_abstains_on_solid_surface(self) -> None:
        test_centerline_abstains_on_solid_surface()

    def test_binds_semantic_reference(self) -> None:
        test_semantic_path_reference_is_bound_and_unknown_ref_rejected()
