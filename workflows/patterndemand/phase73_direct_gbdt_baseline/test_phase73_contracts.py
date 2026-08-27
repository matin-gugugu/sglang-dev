#!/usr/bin/env python3
from __future__ import annotations
import json,sys,unittest
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from gbdt import decode_histograms,direct_feature_names,encode_histograms,feature_matrix,fit_model,predict_encoded  # noqa:E402


class Phase73Contracts(unittest.TestCase):
    def test_feature_scope(self):
        rows=[{"feature_a":"1","feature_b":"2","h0_calls_bin_00":"99","target_calls_bin_00":"88"},{"feature_a":"2","feature_b":"2","h0_calls_bin_00":"77","target_calls_bin_00":"66"}];names=direct_feature_names(rows);self.assertEqual(names,["feature_a"]);self.assertEqual(feature_matrix(rows,names).shape,(2,1))
    def test_encode_decode_nonnegative(self):
        calls=np.arange(1,25,dtype=float).reshape(2,12);logical=calls*50000;pc,pb=decode_histograms(encode_histograms(calls,logical));self.assertTrue((pc>=0).all());self.assertTrue((pb>=0).all());self.assertTrue(np.allclose(pc.sum(1),calls.sum(1)))
    def test_tiny_gbdt_learns_step(self):
        x=np.arange(80,dtype=float).reshape(-1,1);y=np.zeros((80,1));y[40:]=2;config={"candidate_id":"tiny","max_depth":1,"estimators":12,"learning_rate":.2,"min_leaf":8,"row_subsample":1.0,"feature_fraction":1.0,"threshold_quantiles":[.25,.5,.75]};model=fit_model(x,y,["feature_a"],config,73);prediction=predict_encoded(model,x)[:,0];self.assertLess(np.mean((prediction-y[:,0])**2),np.var(y[:,0])*.2);self.assertEqual(model["h0_inputs"],0)
    def test_contract_has_three_candidates(self):
        spec=json.loads((HERE/"experiment.json").read_text());self.assertEqual(len(spec["candidate_selection"]["candidates"]),3);self.assertTrue(spec["candidate_selection"]["phase50_labels_forbidden_during_selection"]);self.assertFalse(spec["h0_as_direct_input_permitted"])


if __name__=="__main__":unittest.main()
