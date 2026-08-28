from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.api_client import APISettings
from rq2_harness.encoding_t3 import (
    T3_FIELDS,
    build_t3_messages,
    check_no_added_numbers_or_units,
    encode_t3,
    generate_t3_audit_checklist,
)


def settings() -> APISettings:
    return APISettings(
        api_key_env="TEST_KEY",
        base_url_env="TEST_URL",
        model_env="TEST_MODEL",
        default_base_url="https://unused.invalid/v1",
        default_model="mock-model",
        timeout_sec=1,
        max_retries=1,
        retry_base_sec=0,
        temperature=0.7,
        max_tokens=512,
        json_mode=True,
        extra_body={"other": "kept"},
    )


class T3EncodingTests(unittest.TestCase):
    def test_messages_contain_only_l3_source(self) -> None:
        messages = build_t3_messages("Plate with 2 holes of diameter 5 mm.")
        serialized = json.dumps(messages)
        self.assertIn("Plate with 2 holes", serialized)
        self.assertNotIn("L1", serialized)
        self.assertEqual(len(messages), 2)

    def test_mock_api_fixed_settings_cache_and_audit_fields(self) -> None:
        calls = []
        response = {field: "" for field in T3_FIELDS}
        response.update(
            {
                "object_type": "plate",
                "primary_features": "2 holes",
                "dimensions_and_units": "diameter 5 mm",
            }
        )

        def mock_api(messages, api_settings):
            calls.append((messages, api_settings))
            return {
                "text": json.dumps(response),
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }

        with tempfile.TemporaryDirectory() as directory:
            first = encode_t3(
                "Plate with 2 holes of diameter 5 mm.",
                directory,
                settings(),
                source_id="sample/1",
                api_call=mock_api,
            )
            second = encode_t3(
                "Plate with 2 holes of diameter 5 mm.",
                directory,
                settings(),
                source_id="sample/1",
                api_call=mock_api,
            )
            self.assertEqual(len(calls), 1)
            fixed = calls[0][1]
            self.assertEqual(fixed.temperature, 0.0)
            self.assertIs(fixed.extra_body["enable_thinking"], False)
            self.assertEqual(fixed.extra_body["other"], "kept")
            self.assertEqual(set(first["after"]), set(T3_FIELDS))
            self.assertEqual(first["cache_key"], second["cache_key"])
            self.assertTrue(first["number_unit_preservation"]["ok"])
            self.assertEqual(first["usage"]["total_tokens"], 30)
            self.assertIn("before_sha256", first)
            self.assertIn("after_sha256", first)
            self.assertIn("record_sha256", first)

    def test_rejects_non_exact_schema(self) -> None:
        def bad_api(messages, api_settings):
            return {"text": '{"object_type":"plate"}', "usage": {}}

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                encode_t3("A plate.", directory, settings(), api_call=bad_api)

    def test_detects_added_number_and_unit(self) -> None:
        encoded = {field: "" for field in T3_FIELDS}
        encoded["dimensions_and_units"] = "10 cm"
        check = check_no_added_numbers_or_units("A plain plate of width 5 mm.", encoded)
        self.assertFalse(check["ok"])
        self.assertEqual(check["added_numbers"], ["10"])
        self.assertEqual(check["added_units"], ["cm"])

    def test_generates_deterministic_twelve_sample_checklist(self) -> None:
        records = [
            {
                "source_id": f"s{i:02d}",
                "cache_key": f"k{i:02d}",
                "source_sha256": f"h{i:02d}",
                "before": "{}",
                "after": {field: "" for field in T3_FIELDS},
                "number_unit_preservation": {"ok": True},
            }
            for i in range(20)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            first = generate_t3_audit_checklist(records, path)
            second = generate_t3_audit_checklist(reversed(records))
            self.assertEqual(len(first), 12)
            self.assertEqual(
                [item["source_id"] for item in first],
                [item["source_id"] for item in second],
            )
            self.assertTrue(path.is_file())
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["checklist"]), 12)
            self.assertIn("audit_sha256", saved)


if __name__ == "__main__":
    unittest.main()
