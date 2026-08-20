#!/usr/bin/env python3
import json
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


if __name__ == "__main__":
    unittest.main()
