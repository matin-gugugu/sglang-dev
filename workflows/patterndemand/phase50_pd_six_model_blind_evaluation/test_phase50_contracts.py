#!/usr/bin/env python3
from __future__ import annotations
import sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from common import load_json,verify_pinned_inputs  # noqa:E402
class Tests(unittest.TestCase):
    def test_parent_and_pins(self):
        c=load_json(HERE/"experiment.json");self.assertEqual(c["workflow_parent_result_commit"],"1b9227753f941cf9c790af69bf0acb7cf8bc3796");self.assertTrue(all(v["ok"] for v in verify_pinned_inputs(c).values()))
    def test_blind_order_and_gates(self):
        c=load_json(HERE/"experiment.json");self.assertFalse(c["execution"]["target_access_before_1800_feature_reconstruction_passes"]);self.assertIn("each of six models",c["acceptance_gate"]["models"]);self.assertEqual(c["blind_contract"]["target_rows"],1800)
if __name__=="__main__":unittest.main()
