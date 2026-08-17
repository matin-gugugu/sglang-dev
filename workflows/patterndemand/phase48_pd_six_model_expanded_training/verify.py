#!/usr/bin/env python3
"""Verify Phase48 result counts, isolation and scientific gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent; P42=HERE.parent/"phase42_pd_residual_training"
sys.path.insert(0,str(P42)); sys.path.insert(0,str(HERE.parent))
from common import load_json, repo_root, verify_result_manifest  # noqa: E402
from model import read_csv_gz, read_json_gz  # noqa: E402


def verify(output: Path) -> dict:
    manifest=verify_result_manifest(output); summary=load_json(output/"summary.json")
    examples=read_csv_gz(output/"dataset/pd_six_model_h0_residual_examples.csv.gz"); targets=read_csv_gz(output/"dataset/pd_six_model_hfull_targets.csv.gz"); predictions=read_csv_gz(output/"predictions/development_validation_predictions.csv.gz"); profiles=read_csv_gz(output/"profiles/expanded_lowdim_profiles.csv.gz"); checkpoint=read_json_gz(output/"checkpoints/pd_six_model_h0_protected_dnn.json.gz")
    forbidden={"requests","full_request_list","input_lens","output_lens","timestamp","arrival_time"}; models={row["model"] for row in examples}
    by_profile={}
    for row in examples: by_profile.setdefault(row["profile_id"],set()).add(row["model"])
    checks={
        "manifest":manifest["ok"], "status":summary.get("status")=="PASS", "done":(output/"DONE").read_text().strip()=="PASS",
        "profiles_1200":len(profiles)==1200 and len({row["profile_id"] for row in profiles})==1200,
        "examples_7200":len(examples)==7200, "targets_7200":len(targets)==7200,
        "train_5760":sum(row["split_role"]=="expanded_train" for row in examples)==5760,
        "validation_1440":sum(row["split_role"]=="expanded_validation" for row in examples)==1440,
        "six_models":len(models)==6, "six_models_per_profile":len(by_profile)==1200 and all(value==models for value in by_profile.values()),
        "prediction_rows_2880":len(predictions)==2880,
        "no_complete_requests":not forbidden.intersection(examples[0]) and int(summary["counts"]["complete_request_rows_in_git"])==0,
        "checkpoint_blind_isolation":checkpoint["phase45_or_phase46_targets_accessed"] is False,
        "six_alphas":set(checkpoint["alpha_by_model"])==models,
        "phase49_matches_gate":summary["gates"]["phase49_permitted"]==summary["gates"]["model_accepted"],
    }
    if not all(checks.values()): raise RuntimeError(checks)
    return {"status":"PASS","checks":checks,"manifest_files":manifest["manifest"]["checked_files"],"model_accepted":summary["gates"]["model_accepted"]}


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase48_pd_six_model_expanded_training"); args=parser.parse_args(); print(json.dumps(verify(args.output_dir.resolve()),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
