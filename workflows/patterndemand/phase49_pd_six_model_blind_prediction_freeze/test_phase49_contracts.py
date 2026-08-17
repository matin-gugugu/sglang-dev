#!/usr/bin/env python3
from __future__ import annotations
import sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from build_selection import select  # noqa:E402
from common import load_json,repo_root,verify_pinned_inputs  # noqa:E402
class Tests(unittest.TestCase):
    def test_parent_pins_selection(self):
        c=load_json(HERE/"experiment.json");self.assertEqual(c["workflow_parent_result_commit"],"f573507abd1b59fa08cdf09030d2f62048c7ee5c");self.assertTrue(all(v["ok"] for v in verify_pinned_inputs(c).values()));rows=select(repo_root());self.assertEqual(len(rows),300);self.assertEqual(sum(int(r["history_count"]) for r in rows),118985)
    def test_target_isolation(self):
        c=load_json(HERE/"experiment.json");self.assertFalse(c["execution"]["target_generation_or_access_permitted"]);self.assertEqual(c["predictor_contract"]["target_rows"],0);self.assertEqual(c["predictor_contract"]["prediction_rows"],3600)
if __name__=="__main__":unittest.main()
