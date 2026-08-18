#!/usr/bin/env python3
"""CPU-only Phase52 deterministic contract tests."""
from __future__ import annotations
import json,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(HERE))
from analysis import aggregate_cost,aggregate_placement,compare_cost,compare_placement,cost_rows,curve_scenarios,placement,read_csv,validate_inputs  # noqa:E402

class TestPhase52(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads((HERE/"experiment.json").read_text());pins={r["name"]:ROOT/r["path"] for r in cls.spec["pinned_inputs"]};cls.pred=read_csv(pins["phase49_frozen_predictions"]);cls.target=read_csv(pins["phase50_hfull_targets"]);cls.curves=json.loads(pins["phase51_curves"].read_text())["curves"]
    def test_inputs(self):
        audit=validate_inputs(self.pred,self.target,self.curves,self.spec);self.assertEqual((audit["prediction_rows"],audit["target_rows"],len(audit["models"])),(3600,1800,6))
    def test_curve_policy(self):
        for curve in self.curves:
            scenario=curve_scenarios(curve);rows=scenario["rows"];self.assertTrue(all(r["lower"]<=r["official"]<=r["monotone_official"] for r in rows));self.assertTrue(all(b["monotone_official"]>=a["monotone_official"] for a,b in zip(rows,rows[1:])))
    def test_full_cardinality_without_outcome_assertion(self):
        costs,_=cost_rows(self.pred,self.target,self.curves);metrics=aggregate_cost(costs);comparison=compare_cost(metrics);rankings,decisions=placement(costs);pm=aggregate_placement(decisions);pc=compare_placement(pm);self.assertEqual((len(costs),len(metrics),len(comparison),len(rankings),len(decisions),len(pm),len(pc)),(10800,60,30,10800,3600,20,10));self.assertEqual({r["ranking_scope"] for r in decisions},{"communication_only_fixed_p1_d1_configuration"})
if __name__=="__main__":unittest.main()
