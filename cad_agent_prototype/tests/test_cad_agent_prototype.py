from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from cad_agent_prototype import CompileRequest, CompileResult, compile_cadquery


HAS_CADQUERY = importlib.util.find_spec("cadquery") is not None


class ContractTests(unittest.TestCase):
    def test_request_and_result_round_trip(self) -> None:
        request = CompileRequest(
            source_path="model.py",
            output_dir="run",
            parameters={"width": 12.5},
        )
        restored_request = CompileRequest.from_dict(request.to_dict())
        self.assertEqual(restored_request, request)

        result = CompileResult(
            status="success",
            source_path="model.py",
            output_dir="run",
            parameters=request.parameters,
            step_path="run/result.step",
            validation={"brep_valid": True},
        )
        restored_result = CompileResult.from_dict(result.to_dict())
        self.assertTrue(restored_result.ok)
        self.assertEqual(restored_result.validation, {"brep_valid": True})

    def test_missing_worker_executable_returns_structured_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.py"
            source.write_text("def build_model(params):\n    return None\n", encoding="utf-8")
            result = compile_cadquery(
                source,
                root / "run",
                python_executable=root / "missing-python",
            )
            self.assertFalse(result.ok)
            self.assertEqual(result.signals[0].code, "worker_start_failed")
            manifest = json.loads((root / "run" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["signals"][0]["code"], "worker_start_failed")


@unittest.skipUnless(HAS_CADQUERY, "CadQuery is not installed")
class CadQueryWorkerTests(unittest.TestCase):
    def test_compiles_parameterized_model_and_validates_step(self) -> None:
        source_text = """
import cadquery as cq

def build_model(params):
    width = float(params.get("width", 10.0))
    return cq.Workplane("XY").box(width, 8.0, 3.0)
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.py"
            source.write_text(source_text, encoding="utf-8")
            result = compile_cadquery(
                source,
                root / "run",
                parameters={"width": 12.0},
                python_executable=sys.executable,
            )
            self.assertTrue(result.ok, result.to_dict())
            self.assertTrue(Path(result.step_path or "").is_file())
            self.assertTrue(result.validation["brep_valid"])
            self.assertGreater(result.validation["volume"], 0.0)
            self.assertEqual(result.validation["units"], "mm")

    def test_missing_build_model_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.py"
            source.write_text("value = 1\n", encoding="utf-8")
            result = compile_cadquery(source, root / "run", python_executable=sys.executable)
            self.assertFalse(result.ok)
            self.assertEqual(result.signals[0].code, "contract_missing")


if __name__ == "__main__":
    unittest.main()
