#!/usr/bin/env python3
"""CPU-only Phase61 contract tests."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from model import CANDIDATES, evaluate_candidates, fit_model, predict, read_points  # noqa: E402


class TestPhase61(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = json.loads((HERE / "experiment.json").read_text())
        self.rows = read_points(ROOT / self.contract["dataset_contract"]["source"])

    def test_source_matrix_and_blind_separation(self) -> None:
        self.assertEqual(len(self.rows), 120)
        self.assertEqual(len({row["pair_id"] for row in self.rows}), 20)
        grid = json.loads((ROOT / "experiment-results/phase60_pd_multi_endpoint_composability/contracts/payload_pair_grid.json").read_text())
        reserved = {row["pair_id"] for values in grid["reserved_future_blind"].values() for row in values}
        self.assertFalse({row["pair_id"] for row in self.rows} & reserved)

    def test_simple_candidate_is_selected(self) -> None:
        result = evaluate_candidates(self.rows, self.contract)
        self.assertAlmostEqual(result["baseline_wape"], 0.27908318299410906, places=12)
        self.assertEqual(result["pair_folds"], 20)
        self.assertFalse(result["candidates"][0]["gate"]["target_guard"])
        self.assertEqual(result["selected"]["candidate_id"], "global_affine_max_min")
        self.assertLess(result["selected"]["gate"]["overall_wape"], 0.03)
        self.assertLess(result["selected"]["gate"]["max_configuration_topology_wape"], 0.04)

    def test_refit_formula(self) -> None:
        specification = next(row for row in CANDIDATES if row["candidate_id"] == "global_affine_max_min")
        model = fit_model(self.rows, specification)
        coefficients = model["groups"]["__global__"]
        self.assertAlmostEqual(coefficients["intercept_us"], -109.83183764072663, places=6)
        self.assertAlmostEqual(coefficients["beta_max"], 0.9764784060893608, places=9)
        self.assertAlmostEqual(coefficients["beta_min"], 0.8501015546505347, places=9)
        self.assertTrue(all(predict(model, row) > 0 for row in self.rows))


if __name__ == "__main__":
    unittest.main()
