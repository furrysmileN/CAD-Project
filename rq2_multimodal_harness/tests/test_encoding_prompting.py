from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.common import sha256_file, sha256_json
from rq2_harness.encoding_prompting import build_encoding_messages


class EncodingPromptingTests(unittest.TestCase):
    def _images(self, root: Path, prefix: str) -> list[dict[str, str]]:
        result = []
        for index, view in enumerate(("front", "side", "top", "isometric")):
            path = root / f"{prefix}_{view}.png"
            Image.new("RGB", (8, 8), (index * 40, 20, 10)).save(path)
            result.append(
                {
                    "view": view,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
        return result

    def _row(self, root: Path) -> dict:
        render = {
            name: {"images": self._images(root, name), "params": {"version": name}}
            for name in ("I1", "I2", "I3")
        }
        point = {
            name: {
                "images": self._images(root, name),
                "params": {"version": name},
                "source_sha256": "point-source",
            }
            for name in ("P1", "P2", "P3")
        }
        return {
            "sample_id": "secret_family_0001",
            "family": "secret_family",
            "difficulty": "hard",
            "gt_code": {"sha256": "gt-secret"},
            "text_encodings": {
                "T1": {"text": "short part", "sha256": sha256_json("short part")},
                "T2": {"text": "detailed part", "sha256": sha256_json("detailed part")},
                "T3": {"text": "Part type: adapter", "sha256": sha256_json("Part type: adapter")},
            },
            "render_encodings": render,
            "point_encodings": point,
        }

    def test_slot_order_and_no_condition_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = self._row(Path(directory))
            messages, audit = build_encoding_messages(
                row,
                SimpleNamespace(text="T3", render="I2", point="P2"),
                image_max_edge=8,
            )
            content = messages[1]["content"]
            text_positions = [
                (index, item["text"])
                for index, item in enumerate(content)
                if item["type"] == "text"
            ]
            self.assertIn("Structured CAD brief", text_positions[1][1])
            self.assertIn("object-space normal", text_positions[2][1])
            self.assertIn("Point-sampled inputs", text_positions[3][1])
            self.assertIn("Required JSON shape", text_positions[4][1])
            self.assertEqual(
                sum(item["type"] == "image_url" for item in content),
                8,
            )
            serialized = json.dumps(messages)
            self.assertNotIn("T3I2P2", serialized)
            self.assertNotIn("secret_family", serialized)
            self.assertNotIn("gt-secret", serialized)
            self.assertEqual(
                audit["slot_order"],
                ["task", "text", "render", "point_cloud", "plan_constraints"],
            )

    def test_single_modalities_have_expected_image_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = self._row(Path(directory))
            for spec, expected in (
                (SimpleNamespace(text="T1", render=None, point=None), 0),
                (SimpleNamespace(text=None, render="I3", point=None), 4),
                (SimpleNamespace(text=None, render=None, point="P1"), 4),
            ):
                messages, _ = build_encoding_messages(row, spec, image_max_edge=8)
                count = sum(
                    item["type"] == "image_url"
                    for item in messages[1]["content"]
                )
                self.assertEqual(count, expected)

    def test_invalid_view_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = self._row(Path(directory))
            row["render_encodings"]["I1"]["images"].reverse()
            with self.assertRaises(ValueError):
                build_encoding_messages(
                    row,
                    SimpleNamespace(text=None, render="I1", point=None),
                    image_max_edge=8,
                )


if __name__ == "__main__":
    unittest.main()
