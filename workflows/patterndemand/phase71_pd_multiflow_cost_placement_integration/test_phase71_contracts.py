#!/usr/bin/env python3
"""CPU-only Phase71 contract tests."""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from analysis import CONFIGURATIONS, POLICIES, _integration_intervals, _mapped_quantile, build_analysis, load_json, read_csv, supported_configurations, validate_inputs  # noqa:E402


class Phase71ContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_json(HERE / "experiment.json")
        cls.paths = {row["name"]: ROOT / row["path"] for row in cls.spec["pinned_inputs"]}
        cls.predictions = read_csv(cls.paths["phase49_frozen_predictions"])
        cls.targets = read_csv(cls.paths["phase50_hfull_targets"])
        cls.curves = load_json(cls.paths["phase51_curves"])["curves"]
        cls.layouts = load_json(cls.paths["phase51_layouts"])["layouts"]

    def test_input_and_coverage_contract(self):
        audit = validate_inputs(self.predictions, self.targets, self.curves, self.layouts, self.spec)
        self.assertEqual((audit["prediction_rows"], audit["target_rows"], audit["supported_units"]), (3600, 1800, 15600))
        self.assertEqual(supported_configurations("qwen3-8b"), tuple(CONFIGURATIONS))
        self.assertEqual(supported_configurations("llama-3.2-3b-instruct"), ("P1D1", "P1D2", "P2D1"))

    def test_wave_policies_are_piecewise_complete(self):
        entries = [(1.0, 0.2), (4.0, 0.3), (32.0, 0.5)]
        for flows in (1, 2, 4):
            for policy in POLICIES:
                intervals = _integration_intervals(entries, flows, policy)
                self.assertTrue(math.isclose(sum(right - left for left, right in intervals), 1.0, abs_tol=1e-12))
                for edge in range(flows):
                    self.assertTrue(all(0.0 <= _mapped_quantile(policy, edge, flows, (left + right) / 2) <= 1.0 for left, right in intervals))

    def test_full_analysis_cardinality_without_outcome_assertion(self):
        result = build_analysis(self.predictions, self.targets, self.curves, self.layouts, load_json(self.paths["r61_model"]), load_json(self.paths["r67_model"]), load_json(self.paths["r69_model"]), self.spec)
        expected = self.spec["expected_counts"]
        self.assertEqual((len(result["costs"]), len(result["decisions"]), len(result["cost_metrics"]), len(result["placement_metrics"]), len(result["wave_sensitivity"])), (expected["unit_configuration_topology_cost_rows"], expected["placement_decision_rows"], expected["cost_metric_rows"], expected["placement_metric_rows"], expected["wave_sensitivity_rows"]))
        self.assertEqual({row["official_policy"] for row in result["wave_sensitivity"]}, {"bin_aligned"})
        self.assertTrue(all(row["model"] in ("qwen3-8b", "deepseek-v2-lite") for row in result["costs"] if row["physical_model"] == "frozen_r69_multiflow"))


if __name__ == "__main__":
    unittest.main()
