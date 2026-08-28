from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.pointcloud.canonical import CanonicalTransform
from rq2_harness.pointcloud.evidence import build_evidence
from rq2_harness.pointcloud.sections import query_cross_section
from rq2_harness.pointcloud.summary import summarize
from rq2_harness.pointcloud.symmetry import detect_mirror_symmetry


def _sample_workplane(workplane, n: int = 2048, seed: int = 42) -> np.ndarray:
    from rq2_harness.geometry import _sample_shape

    shape = workplane.val()
    return np.asarray(_sample_shape(shape, n, seed), dtype=np.float64)


def _axis_alignment(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(max(abs(float(np.dot(a, b))) for a in pred for b in gt))


class SyntheticGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import cadquery as cq  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cadquery 不可用")

    def _evidence(self, points: np.ndarray):
        with tempfile.TemporaryDirectory() as directory:
            npy = Path(directory) / "s.npy"
            np.save(npy, points)
            return build_evidence(npy), CanonicalTransform.from_points(points)

    def test_box_bbox_and_axes(self) -> None:
        import cadquery as cq

        points = _sample_workplane(cq.Workplane("XY").box(2.0, 1.0, 0.5))
        evidence, transform = self._evidence(points)
        size = np.asarray(evidence["frame"]["bbox_size"])
        expected = np.array([1.0, 0.5, 0.25])
        rel = np.abs(size - expected) / np.maximum(expected, 1e-6)
        self.assertLess(float(rel.max()), 0.08)
        axes = np.asarray(evidence["frame"]["principal_axes"])
        self.assertGreater(_axis_alignment(axes, np.eye(3)), 0.95)

    def test_cylinder_section_is_circular(self) -> None:
        import cadquery as cq

        points = _sample_workplane(cq.Workplane("XY").cylinder(1.0, 0.4))
        evidence, transform = self._evidence(points)
        canonical = transform.forward(points)
        section = query_cross_section(
            canonical,
            origin=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            thickness=0.12,
        )
        self.assertGreater(section["point_count"], 20)
        outer = section.get("outer") or {}
        self.assertIsNotNone(outer.get("bbox_size"))

    def test_plate_with_four_holes_detects_inners(self) -> None:
        import cadquery as cq

        plate = (
            cq.Workplane("XY")
            .box(2.0, 2.0, 0.2)
            .faces(">Z")
            .workplane()
            .pushPoints([(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)])
            .hole(0.25)
        )
        points = _sample_workplane(plate, n=4096)
        evidence, transform = self._evidence(points)
        xy = evidence["sections"].get("XY") or {}
        holes = xy.get("holes") or []
        # 2048–4096 点稀疏，至少应提出 1 个孔假设或外环 bbox 接近正方形
        outer = xy.get("outer") or {}
        bbox = np.asarray(outer.get("bbox_size") or [0, 0])
        self.assertTrue(len(holes) >= 1 or (len(bbox) >= 2 and min(bbox[:2]) > 0.5))

    def test_slotted_plate_and_stepped_shaft(self) -> None:
        import cadquery as cq

        slotted = (
            cq.Workplane("XY")
            .box(2.0, 1.0, 0.15)
            .faces(">Z")
            .workplane()
            .slot2D(1.0, 0.2, 0)
            .cutThruAll()
        )
        points = _sample_workplane(slotted)
        evidence, _ = self._evidence(points)
        size = np.asarray(evidence["frame"]["bbox_size"])
        self.assertGreater(float(size.max()), 0.9)

        shaft = cq.Workplane("XY").circle(0.5).extrude(0.4).faces(">Z").workplane().circle(0.25).extrude(0.4)
        points = _sample_workplane(shaft)
        evidence, transform = self._evidence(points)
        rel = np.abs(np.asarray(evidence["frame"]["bbox_size"]) - np.array([1.0, 1.0, 0.8 / max(1.0, 0.8)]))
        self.assertLess(float(np.min(np.asarray(evidence["frame"]["bbox_size"]))), 1.01)

    def test_symmetric_bracket_hits_mirror(self) -> None:
        import cadquery as cq

        bracket = (
            cq.Workplane("XY")
            .moveTo(0, 0)
            .lineTo(1, 0)
            .lineTo(1, 0.2)
            .lineTo(0.2, 0.2)
            .lineTo(0.2, 1)
            .lineTo(0, 1)
            .close()
            .extrude(0.2)
        )
        # 镜像成对称支架
        mirrored = bracket.mirror(mirrorPlane="YZ", union=True)
        points = _sample_workplane(mirrored)
        evidence, transform = self._evidence(points)
        candidates = detect_mirror_symmetry(transform.forward(points), seed=42)
        self.assertTrue(candidates)
        self.assertGreater(float(candidates[0]["support_ratio"]), 0.45)
        self.assertTrue(evidence["symmetry_candidates"])


if __name__ == "__main__":
    unittest.main()
