#!/usr/bin/env python3
"""Small pure contract tests; no dataset, GPU or network access."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("phase57_run_for_tests", HERE / "run.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import Phase57 run")
RUN = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(RUN)


class Phase57ContractTests(unittest.TestCase):
    def test_causal_matrix_is_finite_and_deterministic(self) -> None:
        rows = [{"model": RUN.MODEL_IDS[0], "segment": RUN.SEGMENTS[0], "feature_pd_fixed_draining": "1", "feature_profile_request_count": "10", "h0_calls_bin_00": "2", "h0_logical_bytes_bin_00": "3", "feature_interarrival_cv": "99"}]
        first, names = RUN.feature_matrix(rows, "causal_structural_interactions")
        second, names2 = RUN.feature_matrix(rows, "causal_structural_interactions")
        self.assertEqual(names, names2); self.assertEqual(first.shape, second.shape); self.assertTrue((first == second).all())
        self.assertTrue((first == first).all())

    def test_scope_keys_are_stable(self) -> None:
        row = {"model": "m", "segment": "s"}
        self.assertEqual(RUN.scope_key(row, "model_segment"), "model_segment::m::s")
        self.assertEqual(RUN.scope_key(row, "model"), "model::m")

    def test_budget_shape(self) -> None:
        contract = RUN.load_json(HERE / "experiment.json")
        self.assertEqual(contract["search_contract"]["max_rounds"] * (contract["search_contract"]["seed_candidates_per_round"] + contract["search_contract"]["adaptive_candidates_per_round"]), 144)


if __name__ == "__main__": unittest.main()
