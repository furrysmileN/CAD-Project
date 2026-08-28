from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.encoding_visual import (
    CAMERAS,
    P3_VERSION,
    VisualEncodingConfig,
    bbox_iou,
    canonicalize_points,
    image_qc,
    prepare_visual_encodings,
    render_p1,
    render_p2,
    render_p3,
)


class VisualEncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = np.asarray(
            [
                [-2.0, -1.0, -0.5],
                [2.0, -1.0, -0.5],
                [-2.0, 1.0, 0.5],
                [2.0, 1.0, 0.5],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.canonical, self.normalization = canonicalize_points(self.points)

    def test_shared_cameras_and_canonical_bbox(self) -> None:
        self.assertEqual(CAMERAS["front"], (0.0, -1.0, 0.0))
        self.assertEqual(CAMERAS["side"], (1.0, 0.0, 0.0))
        self.assertEqual(CAMERAS["top"], (0.0, 0.0, 1.0))
        self.assertEqual(CAMERAS["isometric"], (1.0, -1.0, 1.0))
        low, high = self.canonical.min(axis=0), self.canonical.max(axis=0)
        np.testing.assert_allclose((low + high) / 2, np.zeros(3))
        self.assertAlmostEqual(float((high - low).max()), 1.0)

    def test_p1_is_white_with_exact_two_pixel_glyph(self) -> None:
        image = render_p1(np.asarray([[0.0, 0.0, 0.0]]), CAMERAS["front"], resolution=32, padding=0.1)
        array = np.asarray(image)
        black = np.all(array == 0, axis=2)
        self.assertEqual(int(black.sum()), 4)
        self.assertTrue(np.all(array[~black] == 255))

    def test_p2_is_discrete_and_uses_fixed_depth(self) -> None:
        image = render_p2(
            np.asarray([[0.0, 0.0, -0.5], [0.25, 0.0, 0.5]]),
            CAMERAS["top"],
            resolution=64,
            padding=0.06,
        )
        occupied = np.any(np.asarray(image) != 255, axis=2)
        self.assertEqual(int(occupied.sum()), 2)

    def test_p3_fixed_range_not_per_sample_range(self) -> None:
        low = render_p3(np.asarray([[0.0, 0.0, -0.25]]), CAMERAS["top"], resolution=32, padding=0.1)
        high = render_p3(np.asarray([[0.0, 0.0, 0.25]]), CAMERAS["top"], resolution=32, padding=0.1)
        # The occupied point is cyan contour in both; compare dilated gray neighbours.
        low_colors = {tuple(color) for color in np.asarray(low).reshape(-1, 3)}
        high_colors = {tuple(color) for color in np.asarray(high).reshape(-1, 3)}
        self.assertIn((36, 36, 36), low_colors)
        self.assertIn((36, 36, 36), high_colors)
        self.assertEqual(P3_VERSION, "v2-fixed-range")

    def test_point_cache_is_idempotent_and_parameter_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "part.npy"
            np.save(source, self.points)
            kwargs = {
                "cache_dir": root / "cache",
                "pointcloud_path": source,
                "encodings": ("P1", "P2", "P3"),
                "views": ("front", "top"),
                "resolution": 64,
                "padding": 0.08,
            }
            first = prepare_visual_encodings(**kwargs)
            second = prepare_visual_encodings(**kwargs)
            for encoding in ("P1", "P2", "P3"):
                self.assertEqual(
                    first["encodings"][encoding]["cache_key"],
                    second["encodings"][encoding]["cache_key"],
                )
                self.assertEqual(
                    [item["sha256"] for item in first["encodings"][encoding]["images"]],
                    [item["sha256"] for item in second["encodings"][encoding]["images"]],
                )
                self.assertTrue(all(item["qc"]["size_ok"] for item in first["encodings"][encoding]["images"]))
                self.assertTrue(all(item["qc"]["nonempty"] for item in first["encodings"][encoding]["images"]))
            changed = prepare_visual_encodings(**{**kwargs, "padding": 0.09})
            self.assertNotEqual(
                first["encodings"]["P1"]["cache_key"],
                changed["encodings"]["P1"]["cache_key"],
            )

    def test_qc_and_bbox_iou(self) -> None:
        image = render_p1(self.canonical, CAMERAS["front"], resolution=64, padding=0.06)
        qc = image_qc(image, expected_size=64)
        self.assertTrue(qc["size_ok"])
        self.assertTrue(qc["nonempty"])
        self.assertEqual(bbox_iou(qc["bbox"], qc["bbox"]), 1.0)
        self.assertEqual(bbox_iou(None, qc["bbox"]), 0.0)

    def test_config_validation(self) -> None:
        VisualEncodingConfig().validate()
        with self.assertRaises(ValueError):
            VisualEncodingConfig(views=("unknown",)).validate()


if __name__ == "__main__":
    unittest.main()
