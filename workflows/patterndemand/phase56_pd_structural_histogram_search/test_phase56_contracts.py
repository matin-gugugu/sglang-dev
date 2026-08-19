#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run import refine_candidates, scope_key, seed_candidates  # noqa: E402


class Phase56Contracts(unittest.TestCase):
    def test_candidate_budget_and_adaptive_variants(self) -> None:
        contract = json.loads((HERE / "experiment.json").read_text(encoding="utf-8"))
        seeds = seed_candidates(contract)
        self.assertEqual(len(seeds), 20)
        fake = [{"config": row, "segments": {"burstgpt_1": {"composite_ratio": 0.8}, "burstgpt_2": {"composite_ratio": 0.9}, "burstgpt_3": {"composite_ratio": 0.82}}, "oof_bias": [{"bin": 10, "relative_signed_bias": 0.3}]} for row in seeds[:6]]
        variants = refine_candidates(fake, contract)
        self.assertEqual(len(variants), 12)
        self.assertEqual(len(seeds) + len(variants), 32)
        self.assertTrue(all(row["stage"] == "B" and row["parent_candidate_id"] for row in variants))
        self.assertTrue(any(row["head_scope"] == "model_segment" for row in variants))

    def test_scope_keys_are_explicit(self) -> None:
        row = {"model": "qwen3-8b", "segment": "burstgpt_2"}
        self.assertEqual(scope_key(row, "global"), "global")
        self.assertEqual(scope_key(row, "model"), "model::qwen3-8b")
        self.assertEqual(scope_key(row, "segment"), "segment::burstgpt_2")
        self.assertEqual(scope_key(row, "model_segment"), "model_segment::qwen3-8b::burstgpt_2")

    def test_forbidden_execution_scope(self) -> None:
        contract = json.loads((HERE / "experiment.json").read_text(encoding="utf-8"))
        self.assertFalse(contract["execution"]["gpu_permitted"])
        self.assertFalse(contract["execution"]["phase50_blind_access_permitted"])
        self.assertIn("validation profiles are used to generate or rank candidates", contract["stop_and_block_if"])


if __name__ == "__main__":
    unittest.main()
