#!/usr/bin/env python3
"""Phase73 frozen-input, dependency and no-H0/no-teacher preflight."""
from __future__ import annotations
import argparse,json,os,sys
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from common import load_json,repo_root,require_clean_before_run,require_expected_head,run_git,utc_now,verify_pinned_inputs  # noqa:E402
from gbdt import direct_feature_names,read_csv_gz  # noqa:E402


def run_checks(expected:str)->dict:
    import numpy as np
    head=require_expected_head(expected);require_clean_before_run();spec=load_json(HERE/"experiment.json");run_git(["merge-base","--is-ancestor",spec["workflow_base_result_commit"],head]);pins=verify_pinned_inputs(spec);root=repo_root();source_checks={}
    for row in spec["source_results"]:
        directory=root/"experiment-results"/row["directory"];source_checks[row["phase"]]={"result_commit":run_git(["log","-1","--format=%H","--",str(directory.relative_to(root))])==row["result_commit"],"status":load_json(directory/"summary.json").get("status")==row["expected_status"],"done":(directory/"DONE").read_text().strip().startswith("PASS")}
    examples=read_csv_gz(root/next(x["path"] for x in spec["pinned_inputs"] if x["name"]=="phase48_examples"));features=read_csv_gz(root/next(x["path"] for x in spec["pinned_inputs"] if x["name"]=="phase49_features"));frozen=read_csv_gz(root/next(x["path"] for x in spec["pinned_inputs"] if x["name"]=="phase49_predictions"));phase50_summary=load_json(root/"experiment-results/phase50_pd_six_model_blind_evaluation/summary.json");names=direct_feature_names([r for r in examples if r["split_role"]=="expanded_train"]);counts=spec["expected_counts"]
    data_checks={"phase48_rows":len(examples)==counts["phase48_rows"],"train_validation":Counter(r["split_role"] for r in examples)==Counter({"expanded_train":counts["phase48_train_rows"],"expanded_validation":counts["phase48_validation_rows"]}),"phase49_features":len(features)==counts["phase49_feature_rows"],"phase49_predictions":len(frozen)==counts["phase49_frozen_prediction_rows"],"phase50_target_count_from_summary":int(phase50_summary["counts"]["target_rows"])==counts["phase50_target_rows"],"phase50_target_values_not_loaded":True,"six_models":len({r["model"] for r in features})==6,"feature_min":len(names)>=counts["features_min"],"feature_scope":all(name.startswith("feature_") and not name.startswith(("h0_","target_","residual_")) for name in names),"test_schema":all(name in features[0] for name in names),"target_open_acknowledged":phase50_summary.get("scientific_outcome")=="CONFIRMS_SIX_MODEL_H0_PROTECTED_IMPROVEMENT"}
    execution={"numpy_available":np.__version__,"cuda_hidden_or_unset":os.environ.get("CUDA_VISIBLE_DEVICES") in (None,"","-1"),"gpu_forbidden":spec["gpu_required"] is False,"network_forbidden":spec["network_required"] is False,"raw_forbidden":spec["raw_required"] is False,"teacher_forbidden":spec["teacher_execution_permitted"] is False,"pseudo_forbidden":spec["pseudo_request_construction_permitted"] is False,"h0_input_forbidden":spec["h0_as_direct_input_permitted"] is False}
    if not all(all(v.values()) for v in source_checks.values()) or not all(data_checks.values()) or not all(value if isinstance(value,bool) else bool(value) for value in execution.values()):raise RuntimeError({"sources":source_checks,"data":data_checks,"execution":execution})
    return {"schema_version":"phase73-input-freeze-v1","status":"PASS","workflow_commit":head,"captured_at_utc":utc_now(),"pinned_inputs":pins,"source_checks":source_checks,"data_checks":data_checks,"feature_names":names,"direct_feature_count":len(names),"execution":execution,"benchmark_classification":"target-open fixed benchmark; not fresh blind"}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--expected-workflow-commit",required=True);args=parser.parse_args();print(json.dumps(run_checks(args.expected_workflow_commit),ensure_ascii=False,indent=2))
