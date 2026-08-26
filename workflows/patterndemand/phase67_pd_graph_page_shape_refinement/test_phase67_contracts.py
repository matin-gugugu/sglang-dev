#!/usr/bin/env python3
from __future__ import annotations
import json,sys,unittest
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from common import repo_root  # noqa:E402
from model import evaluate,fit_model,read_development  # noqa:E402
from preflight import validate_phase68  # noqa:E402

class Phase67Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (HERE/"experiment.json").open(encoding="utf-8") as stream:cls.contract=json.load(stream)
        root=repo_root()
        with (root/"experiment-results/phase65_pd_graph_correction_development/model/multiflow_graph_correction.json").open(encoding="utf-8") as stream:r65=json.load(stream)
        cls.rows=read_development(root/cls.contract["dataset_contract"]["phase64_source"],root/cls.contract["dataset_contract"]["phase66_source"],r65);cls.evaluation=evaluate(cls.rows,cls.contract)
    def test_matrix(self):
        self.assertEqual(len(self.rows),480);self.assertEqual(Counter(r["source_phase"] for r in self.rows),Counter({"phase64":240,"phase66":240}));self.assertEqual(Counter((r["source_phase"],r["vector_index"]) for r in self.rows),Counter({(s,i):24 for s in ("phase64","phase66") for i in range(10)}))
    def test_blind_grid(self):
        value=validate_phase68(self.contract);self.assertEqual(value["reserved_pages"],[36,40,48,56,64]);self.assertTrue(set(value["reserved_pages"]).isdisjoint(value["development_pages"]))
    def test_fixed_selection(self):
        candidates=self.evaluation["candidates"];self.assertFalse(candidates[0]["target_guard"]);self.assertFalse(candidates[1]["target_guard"]);self.assertTrue(candidates[2]["target_guard"]);self.assertEqual(self.evaluation["selected"]["candidate_id"],"model_configuration_graph_page_sqrt")
    def test_all_four_guards(self):
        self.assertEqual(set(self.evaluation["selected"]["schemes"]),{"payload_cohort","topology","source_blocked","tail32"});self.assertTrue(all(v["pass"] for v in self.evaluation["selected"]["schemes"].values()))
    def test_full_refit(self):
        model=fit_model(self.rows,{"candidate_id":"model_configuration_graph_page_sqrt","rank":3,"feature_family":"page_sqrt"});self.assertEqual(len(model["groups"]),8);self.assertEqual({g["training_rows"] for g in model["groups"].values()},{60})

if __name__=="__main__":unittest.main()
