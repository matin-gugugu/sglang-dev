#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;P42=HERE.parent/"phase42_pd_residual_training";sys.path.insert(0,str(P42));sys.path.insert(0,str(HERE.parent))
from common import load_json,repo_root,verify_result_manifest  # noqa:E402
from model import read_csv_gz  # noqa:E402
def verify(output:Path)->dict:
    manifest=verify_result_manifest(output);summary=load_json(output/"summary.json");features=read_csv_gz(output/"dataset/pd_six_model_blind_target_free_features.csv.gz");profiles=read_csv_gz(output/"profiles/fresh_blind_lowdim_profiles.csv.gz");pred=read_csv_gz(output/"predictions/pd_six_model_blind_frozen_predictions.csv.gz");models={row["model"] for row in features};forbidden={name for row in features for name in row if name.startswith(("target_","residual_"))}
    checks={"manifest":manifest["ok"],"status":summary.get("status")=="PASS","done":(output/"DONE").read_text().strip()=="PASS","profiles_300":len(profiles)==300,"features_1800":len(features)==1800,"six_models":len(models)==6,"six_per_profile":all(sum(row["profile_id"]==pid for row in features)==6 for pid in {row["profile_id"] for row in features}),"predictions_3600":len(pred)==3600,"methods_exact":{row["method"] for row in pred}=={"h0","h0_plus_dnn_residual"},"no_target_columns":not forbidden,"target_rows_zero":summary["counts"]["target_rows"]==0,"no_complete_requests":summary["counts"]["complete_request_rows_in_git"]==0}
    if not all(checks.values()):raise RuntimeError(checks)
    return {"status":"PASS","checks":checks,"manifest_files":manifest["manifest"]["checked_files"]}
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase49_pd_six_model_blind_prediction_freeze");a=p.parse_args();print(json.dumps(verify(a.output_dir.resolve()),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
