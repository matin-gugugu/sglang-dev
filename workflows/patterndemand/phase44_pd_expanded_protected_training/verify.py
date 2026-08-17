#!/usr/bin/env python3
"""Verify Phase44 expanded data and H0 protection gates."""

from __future__ import annotations

import argparse,csv,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent; P42=HERE.parent/"phase42_pd_residual_training"
sys.path.insert(0,str(HERE.parent)); sys.path.insert(0,str(P42))
from common import load_json,repo_root,verify_result_manifest  # noqa: E402
from model import read_csv_gz,read_json_gz  # noqa: E402


def verify(output:Path)->dict:
    manifest=verify_result_manifest(output); summary=load_json(output/"summary.json"); examples=read_csv_gz(output/"dataset/pd_expanded_h0_residual_examples.csv.gz"); targets=read_csv_gz(output/"dataset/pd_expanded_hfull_targets.csv.gz"); predictions=read_csv_gz(output/"predictions/development_validation_predictions.csv.gz"); checkpoint=read_json_gz(output/"checkpoints/pd_qwen3_expanded_h0_protected_dnn.json.gz")
    forbidden={"requests","full_request_list","input_lens","output_lens","timestamp","arrival_time"}
    checks={"manifest":manifest["ok"],"status":summary.get("status")=="PASS","done":(output/"DONE").read_text().strip()=="PASS","examples_1200":len(examples)==1200,"targets_1200":len(targets)==1200,"train_960":sum(row["split_role"]=="expanded_train" for row in examples)==960,"validation_240":sum(row["split_role"]=="expanded_validation" for row in examples)==240,"prediction_rows_480":len(predictions)==480,"no_complete_requests":not forbidden.intersection(examples[0]) and int(summary["counts"]["complete_request_rows_in_git"])==0,"checkpoint_no_phase43":checkpoint["phase43_targets_accessed"] is False,"new_blind_matches_gate":summary["gates"]["new_blind_permitted"]==summary["gates"]["model_accepted"]}
    if not all(checks.values()): raise RuntimeError(checks)
    return {"status":"PASS","checks":checks,"manifest_files":manifest["manifest"]["checked_files"],"model_accepted":summary["gates"]["model_accepted"]}


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase44_pd_expanded_protected_training"); args=parser.parse_args(); print(json.dumps(verify(args.output_dir.resolve()),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
