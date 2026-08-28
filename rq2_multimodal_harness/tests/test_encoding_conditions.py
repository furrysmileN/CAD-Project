from __future__ import annotations

import copy
import sys
import unittest
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.encoding_conditions import (
    CONDITION_IDS,
    ConditionSpec,
    enumerate_conditions,
    modalities_for,
    parse_condition,
    validate_conditions,
)
from rq2_harness.encoding_selection import freeze_selection, select_encoding_samples


class EncodingConditionTests(unittest.TestCase):
    def test_enumeration_has_63_unique_stable_conditions(self) -> None:
        first = enumerate_conditions()
        second = enumerate_conditions()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 63)
        self.assertEqual(len(set(CONDITION_IDS)), 63)
        self.assertEqual(CONDITION_IDS[:9], ("T1", "T2", "T3", "I1", "I2", "I3", "P1", "P2", "P3"))
        self.assertEqual(CONDITION_IDS[-1], "T3I3P3")

    def test_parse_validate_and_modalities(self) -> None:
        parsed = parse_condition("T1I2P3")
        self.assertEqual(parsed, ConditionSpec(text="T1", render="I2", point="P3"))
        self.assertEqual(parsed.condition_id, "T1I2P3")
        self.assertEqual(modalities_for(parsed), frozenset({"text", "render", "point"}))
        self.assertEqual(parse_condition("T1").modalities, frozenset({"text"}))
        self.assertEqual(validate_conditions(["T1", "I2P3"]), (parse_condition("T1"), parse_condition("I2P3")))
        with self.assertRaises(ValueError):
            parse_condition("I2T1")
        with self.assertRaises(ValueError):
            parse_condition("T4")
        with self.assertRaises(ValueError):
            ConditionSpec()
        with self.assertRaises(ValueError):
            validate_conditions(["T1", "T1"])
        with self.assertRaises(FrozenInstanceError):
            parsed.text = "T2"  # type: ignore[misc]


class EncodingSelectionTests(unittest.TestCase):
    @staticmethod
    def _fixtures() -> tuple[list[dict], dict]:
        rows = []
        difficulties = ("easy", "medium", "hard")
        for index in range(45):
            rows.append(
                {
                    "schema_version": "rq2.manifest.v1",
                    "sample_id": f"sample_{index:02d}",
                    "difficulty": difficulties[index % 3],
                    "complexity_bin": index % 3,
                    "family": f"family_{index:02d}",
                    "payload": {"keep": index},
                }
            )
        audit = {
            "samples": [
                {"sample_id": row["sample_id"], "v2_fully_representable_estimate": index < 42}
                for index, row in enumerate(rows)
            ]
        }
        return rows, audit

    def test_quota_selection_is_deterministic_and_input_order_independent(self) -> None:
        rows, audit = self._fixtures()
        first = select_encoding_samples(rows, audit, seed=42)
        second = select_encoding_samples(reversed(rows), audit, seed=42)
        self.assertEqual([row["sample_id"] for row in first], [row["sample_id"] for row in second])
        self.assertEqual(Counter(row["difficulty"] for row in first), {"easy": 7, "medium": 7, "hard": 6})
        self.assertEqual({row["complexity_bin"] for row in first}, {0, 1, 2})
        self.assertEqual(len({row["family"] for row in first}), 20)

    def test_freeze_functions_are_pure(self) -> None:
        rows, audit = self._fixtures()
        original = copy.deepcopy(rows)
        frozen, summary = freeze_selection(rows, audit, seed=42)
        self.assertEqual(rows, original)
        self.assertEqual(len(frozen), 20)
        self.assertEqual(summary["selected_count"], 20)
        self.assertEqual(summary["eligible_v2_fully_representable_count"], 42)
        frozen[0]["payload"]["keep"] = -1
        source = next(row for row in rows if row["sample_id"] == frozen[0]["sample_id"])
        self.assertNotEqual(source["payload"]["keep"], -1)


if __name__ == "__main__":
    unittest.main()
