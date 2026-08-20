#!/usr/bin/env python3
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Phase59ContractTests(unittest.TestCase):
    def test_runtime_and_candidate_budget(self):
        contract = json.loads((HERE / "experiment.json").read_text(encoding="utf-8")); search = contract["search_contract"]
        self.assertEqual(search["max_rounds"] * (search["train_candidates_per_round"] + search["blend_candidates_per_round"]), search["max_total_candidates"])
        self.assertLessEqual(search["hard_total_runtime_seconds"], 10.5 * 3600)

    def test_thresholds_unchanged(self):
        gate = json.loads((HERE / "experiment.json").read_text(encoding="utf-8"))["acceptance_gate"]
        self.assertIn("0.10", gate["overall_histogram"]); self.assertIn("0.15", gate["model_histogram"]); self.assertIn("0.15", gate["segment_histogram"]); self.assertIn("0.05", gate["total_wape"])

    def test_outputs(self):
        required = json.loads((HERE / "expected_outputs.json").read_text(encoding="utf-8"))["required"]
        self.assertIn("analysis/continuation_spec.json", required)

    def test_recovery_contract(self):
        contract = json.loads((HERE / "experiment.json").read_text(encoding="utf-8"))
        self.assertIn("checkpoint after every completed candidate", contract["search_contract"]["recovery_policy"])
        source = (HERE / "run.py").read_text(encoding="utf-8")
        self.assertIn("atomic_save_runtime_state", source)
        self.assertIn("runtime_state_restored", source)

    def test_runtime_state_atomic_roundtrip(self):
        spec = importlib.util.spec_from_file_location("phase59_recovery_test_run", HERE / "run.py")
        self.assertIsNotNone(spec); self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.pkl.gz"; output = Path(directory) / "formal-result"
            state = {"schema_version": module.RUNTIME_STATE_SCHEMA, "workflow_commit": "abc123", "output_dir": str(output), "sentinel": [1, 2, 3]}
            module.atomic_save_runtime_state(checkpoint, state)
            self.assertEqual(module.load_runtime_state(checkpoint, "abc123", output)["sentinel"], [1, 2, 3])
            self.assertFalse(any(path.name.startswith(f".{checkpoint.name}.tmp-") for path in checkpoint.parent.iterdir()))


if __name__ == "__main__":
    unittest.main()
