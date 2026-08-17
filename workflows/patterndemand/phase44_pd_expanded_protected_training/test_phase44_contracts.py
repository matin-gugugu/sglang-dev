#!/usr/bin/env python3

from __future__ import annotations

import csv,sys,unittest
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent))
from build_selection import HISTORY_MS,select
from common import load_json,repo_root,verify_pinned_inputs


class Phase44Contracts(unittest.TestCase):
    def test_parent_and_pins(self)->None:
        contract=load_json(HERE/"experiment.json"); self.assertEqual(contract["workflow_parent_result_commit"],"4e627c8f72f2568111410d57f053eb82d61b1538"); self.assertTrue(all(value["ok"] for value in verify_pinned_inputs(contract).values()))
    def test_selection_reproduces_and_is_disjoint(self)->None:
        with (HERE/"selection/expanded_windows.csv").open(newline="") as source:frozen=list(csv.DictReader(source))
        reproduced=select(repo_root()); self.assertEqual([row["window_id"] for row in frozen],[row["window_id"] for row in reproduced]); self.assertEqual(len(frozen),1200); self.assertEqual(sum(int(row["history_count"]) for row in frozen),486242)
        roles=Counter((row["segment"],row["role"]) for row in frozen)
        for segment in ("burstgpt_1","burstgpt_2","burstgpt_3"): self.assertEqual(roles[(segment,"expanded_train")],320); self.assertEqual(roles[(segment,"expanded_validation")],80)
        by_segment={segment:sorted(int(row["cutoff_ms"]) for row in frozen if row["segment"]==segment) for segment in ("burstgpt_1","burstgpt_2","burstgpt_3")}
        self.assertTrue(all(right-left>=HISTORY_MS for values in by_segment.values() for left,right in zip(values,values[1:])))
    def test_hard_gate_is_strict(self)->None:
        gate=load_json(HERE/"experiment.json")["hard_acceptance_gate"]; self.assertIn("strictly beat H0",gate["oof"]); self.assertIn("strictly beat H0",gate["validation_overall"])


if __name__=="__main__": unittest.main()
