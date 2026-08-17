#!/usr/bin/env python3

from __future__ import annotations

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Phase46Contracts(unittest.TestCase):
    def test_parent_and_freeze_boundaries(self) -> None:
        contract = json.loads((HERE / "experiment.json").read_text())
        self.assertEqual(contract["workflow_parent_result_commit"], "284f4b796b57bfee5002efb52937da26d0fe748f")
        execution = contract["execution"]; self.assertFalse(execution["training_permitted"]); self.assertFalse(execution["checkpoint_loading_permitted"]); self.assertFalse(execution["prediction_recompute_or_change_permitted"])

    def test_gate_is_predeclared_and_strict(self) -> None:
        gate = json.loads((HERE / "experiment.json").read_text())["acceptance_gate"]
        self.assertIn("strictly below 1.0", gate["overall"]); self.assertIn("each of three segments", gate["segments"]); self.assertIn("only if both", gate["scientific_outcome"])


if __name__ == "__main__": unittest.main()
