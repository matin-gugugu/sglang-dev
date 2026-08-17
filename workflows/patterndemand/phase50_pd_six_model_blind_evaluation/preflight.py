#!/usr/bin/env python3
"""Verify immutable R49 freeze before Phase50 target access."""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;P41=HERE.parent/"phase41_pd_full_window_dataset";P42=HERE.parent/"phase42_pd_residual_training";P49=HERE.parent/"phase49_pd_six_model_blind_prediction_freeze"
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(P42));sys.path.insert(0,str(P41));sys.path.insert(0,str(P49));sys.path.insert(0,str(HERE))
from build_selection import select  # noqa:E402
from common import load_json,repo_root,require_clean_before_run,require_expected_head,verify_pinned_inputs  # noqa:E402
from model import read_csv_gz  # noqa:E402
from prepare_bundle import raw_source_audit  # noqa:E402
def run_checks(expected:str,raw_dir:Path)->dict:
    contract=load_json(HERE/"experiment.json");p41=load_json(P41/"experiment.json");head=require_expected_head(expected);require_clean_before_run(("data/",));pins=verify_pinned_inputs(contract)
    features=read_csv_gz(repo_root()/"experiment-results/phase49_pd_six_model_blind_prediction_freeze/dataset/pd_six_model_blind_target_free_features.csv.gz");pred=read_csv_gz(repo_root()/"experiment-results/phase49_pd_six_model_blind_prediction_freeze/predictions/pd_six_model_blind_frozen_predictions.csv.gz");selection=select(repo_root());forbidden=[name for row in features+pred for name in row if name.startswith(("target_","residual_","future_"))];methods=Counter(row["method"] for row in pred);selection_ids={row["profile_id"] for row in selection};feature_ids={row["profile_id"] for row in features};prediction_ids={row["profile_id"] for row in pred};keys={(row["profile_id"],row["model"]) for row in features}
    checks={"selection_300":len(selection)==300,"features_1800":len(features)==1800,"six_models":len({row["model"] for row in features})==6,"unit_keys_1800":len(keys)==1800,"predictions_3600":len(pred)==3600,"methods_exact":methods==Counter({"h0":1800,"h0_plus_dnn_residual":1800}),"profile_ids_exact":selection_ids==feature_ids==prediction_ids,"prediction_unit_keys_exact":{(row["profile_id"],row["model"]) for row in pred}==keys,"no_target_or_residual":not forbidden}
    if not all(checks.values()):raise RuntimeError({"freeze_checks":checks,"forbidden":sorted(set(forbidden))})
    return {"status":"PASS","workflow_commit":head,"pinned_inputs":pins,"freeze_checks":checks,"raw_source_audit":raw_source_audit(p41,raw_dir.expanduser().resolve()),"targets_accessed":False,"checkpoint_loaded":False,"prediction_recomputed":False,"gpu_used":False}
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--expected-workflow-commit",required=True);p.add_argument("--raw-dir",type=Path,required=True);a=p.parse_args();print(json.dumps(run_checks(a.expected_workflow_commit,a.raw_dir),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
