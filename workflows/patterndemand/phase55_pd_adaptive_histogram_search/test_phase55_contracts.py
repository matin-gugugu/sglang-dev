#!/usr/bin/env python3

from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import refine_candidates, seed_candidates
from preflight import run_checks
from common import load_json


class Phase55Contracts(unittest.TestCase):
    def test_budget_and_iteration_policy(self) -> None:
        contract = load_json(Path(__file__).resolve().parent / "experiment.json")
        seeds = seed_candidates(contract); variants = refine_candidates(seeds[:3], contract)
        self.assertEqual(len(seeds), 10)
        self.assertEqual(len(variants), 6)
        self.assertEqual(len(seeds) + len(variants), contract["search_contract"]["max_total_candidates"])
        self.assertTrue(all(row["stage"] == "B" and row["parent_candidate_id"] for row in variants))

    def test_blind_and_gpu_are_forbidden(self) -> None:
        contract = load_json(Path(__file__).resolve().parent / "experiment.json")
        self.assertFalse(contract["execution"]["gpu_permitted"])
        self.assertFalse(contract["execution"]["phase50_blind_access_permitted"])
        self.assertIn("validation profiles are used to generate or rank new candidates", contract["stop_and_block_if"])


if __name__ == "__main__": unittest.main()
