#!/usr/bin/env python3
import json, unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class Phase58ContractTests(unittest.TestCase):
    def test_budget(self):
        c=json.load(open(HERE/"experiment.json")); s=c["search_contract"]; self.assertEqual(s["max_rounds"]*(s["seed_candidates_per_round"]+s["adaptive_candidates_per_round"]),36); self.assertLessEqual(s["estimated_cpu_budget_hours"],8)
    def test_required_outputs(self):
        self.assertIn("analysis/diagnostic_summary.json",json.load(open(HERE/"expected_outputs.json"))["required"])
if __name__=="__main__": unittest.main()
