#!/usr/bin/env python3
from __future__ import annotations
import json,unittest
from pathlib import Path
from model import CANDIDATES,candidate_metric_rows,evaluate_candidates,fit_model,predict,read_points
from preflight import validate_reserved_grid

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

class Phase65ContractsTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls)->None:
  cls.contract=json.loads((HERE/"experiment.json").read_text(encoding="utf-8"))
  cls.rows=read_points(ROOT/cls.contract["dataset_contract"]["source"])
  cls.evaluation=evaluate_candidates(cls.rows,cls.contract)

 def test_development_matrix_and_folds(self)->None:
  self.assertEqual(len(self.rows),240)
  self.assertEqual({row["vector_index"] for row in self.rows},set(range(10)))
  self.assertEqual({row["topology_level"] for row in self.rows},{"L1","L2","L3"})
  self.assertEqual(self.evaluation["payload_folds"],10)
  self.assertEqual(self.evaluation["topology_folds"],3)
  self.assertEqual(len(self.evaluation["all_oof_predictions"]),3360)

 def test_first_passing_candidate_is_fixed(self)->None:
  selected=self.evaluation["selected"]
  self.assertIsNotNone(selected)
  self.assertEqual(selected["candidate_id"],"model_configuration_affine_graph")
  self.assertTrue(all(not row["target_guard"] for row in self.evaluation["candidates"][:-1]))
  self.assertTrue(self.evaluation["candidates"][-1]["target_guard"])

 def test_selected_candidate_passes_both_fixed_gates(self)->None:
  selected=self.evaluation["selected"]
  for scheme in ("payload","topology"):
   gate=selected["schemes"][scheme]["gate"]
   self.assertTrue(gate["pass"])
   self.assertTrue(all(gate["checks"].values()))
   self.assertLessEqual(gate["overall_wape"],0.10)
   self.assertLessEqual(gate["max_configuration_topology_wape"],0.15)
  self.assertAlmostEqual(selected["schemes"]["payload"]["gate"]["overall_wape"],0.039282228238020735,places=12)
  self.assertAlmostEqual(selected["schemes"]["topology"]["gate"]["overall_wape"],0.02543587951921807,places=12)

 def test_full_refit_is_positive_and_has_expected_groups(self)->None:
  spec=next(value for value in CANDIDATES if value["candidate_id"]=="model_configuration_affine_graph")
  model=fit_model(self.rows,spec)
  self.assertEqual(len(model["groups"]),8)
  self.assertEqual({group["training_rows"] for group in model["groups"].values()},{30})
  self.assertTrue(all(predict(model,row)>0 for row in self.rows))

 def test_baselines_and_reserved_blind_boundary(self)->None:
  metrics={row["candidate_id"]:row for row in candidate_metric_rows(self.evaluation)}
  self.assertAlmostEqual(metrics["max_edge"]["payload_overall_wape"],0.341216,places=5)
  self.assertAlmostEqual(metrics["r61_graph_extension"]["payload_overall_wape"],0.19693,places=5)
  audit=validate_reserved_grid(self.contract)
  self.assertEqual(audit["development_pages"],[1,2,4,8,16])
  self.assertEqual(audit["reserved_pages"],[3,6,12,24,32])
  self.assertTrue(all(audit["checks"].values()))

if __name__=="__main__":unittest.main()
