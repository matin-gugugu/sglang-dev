#!/usr/bin/env python3
"""Run the CPU-only Phase67 development refinement."""
from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
from typing import Any
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from common import environment_record,load_json,refresh_manifest,repo_root,utc_now,validate_result_tree,write_json  # noqa:E402
from model import BASELINES,CANDIDATES,baseline_value,evaluate,fit_model,predict,prediction_rows,read_development,slice_metrics  # noqa:E402
from preflight import run_checks  # noqa:E402

def write_csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:raise RuntimeError(f"empty CSV: {path}")
    path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)

def metric_rows(evaluation:dict[str,Any])->list[dict[str,Any]]:
    output=[];selected=None if evaluation["selected"] is None else evaluation["selected"]["candidate_id"]
    for candidate in evaluation["candidates"]:
        row={"candidate_id":candidate["candidate_id"],"complexity_rank":candidate["complexity_rank"],"feature_family":candidate["feature_family"],"target_guard":candidate["target_guard"],"selected":candidate["candidate_id"]==selected}
        for scheme,value in candidate["schemes"].items():
            for key,item in value.items():
                if key!="checks":row[f"{scheme}_{key}"]=item
            for key,item in value["checks"].items():row[f"{scheme}_check_{key}"]=item
        output.append(row)
    return output

def run(expected:str,output:Path)->dict[str,Any]:
    started=time.monotonic();preflight=run_checks(expected)
    if output.exists():raise RuntimeError(f"refuse overwrite: {output}")
    contract=load_json(HERE/"experiment.json");root=repo_root();r65=load_json(root/"experiment-results/phase65_pd_graph_correction_development/model/multiflow_graph_correction.json")
    rows=read_development(root/contract["dataset_contract"]["phase64_source"],root/contract["dataset_contract"]["phase66_source"],r65);evaluation=evaluate(rows,contract);selected=evaluation["selected"];status="PASS" if selected is not None else "PASS_TARGET_NOT_MET";metrics=metric_rows(evaluation)
    if selected is None:
        model_bundle={"schema_version":"phase67-multiflow-graph-page-correction-v1","status":"NOT_FROZEN","reason":"no fixed candidate passed all four guards","workflow_commit":preflight["workflow_commit"]}
    else:
        specification=next(c for c in CANDIDATES if c["candidate_id"]==selected["candidate_id"]);model_bundle=fit_model(rows,specification)
        model_bundle.update({"workflow_commit":preflight["workflow_commit"],"source_result_commit":contract["workflow_parent_result_commit"],"inference_formula":contract["candidate_contract"]["formula_of_selected_family"],"selection_evidence":{"schemes":selected["schemes"],"first_simplest_passing":True,"refit_metrics_used_for_selection":False},"blind_status":{"phase68_grid_sha256":preflight["phase68_reserved_grid"]["grid_sha256"],"phase68_targets_opened":False,"fresh_blind_validated":False}})
    refit_predictions=[]
    for baseline in BASELINES:refit_predictions+=prediction_rows(baseline,"refit","all_development",rows,[baseline_value(baseline,r) for r in rows])
    if selected is not None:refit_predictions+=prediction_rows(selected["candidate_id"],"refit","all_development",rows,[predict(model_bundle,r) for r in rows])
    refit_slices=[]
    for candidate_id in [*BASELINES,*([selected["candidate_id"]] if selected else [])]:refit_slices+=slice_metrics([r for r in refit_predictions if r["candidate_id"]==candidate_id])
    output.mkdir(parents=True);write_json(output/"contracts/experiment.json",contract);write_json(output/"contracts/phase68_reserved_blind_grid.json",load_json(HERE/"phase68_reserved_blind_grid.json"));write_json(output/"audit/preflight.json",preflight);write_json(output/"audit/input_freeze.json",{"schema_version":"phase67-input-freeze-v1","workflow_commit":preflight["workflow_commit"],"source_result_commit":contract["workflow_parent_result_commit"],"pinned_inputs":preflight["pinned_inputs"],"development_rows":480,"phase64_rows":240,"phase66_rows":240,"phase68_grid_sha256":preflight["phase68_reserved_grid"]["grid_sha256"],"selection_uses_held_out_predictions_only":True,"refit_metrics_used_for_selection":False,"phase68_measurements_or_targets_read":False})
    write_csv(output/"analysis/candidate_metrics.csv",metrics);write_csv(output/"analysis/oof_predictions.csv",evaluation["predictions"]);write_csv(output/"analysis/oof_slice_metrics.csv",evaluation["slices"]);write_csv(output/"analysis/refit_predictions.csv",refit_predictions);write_csv(output/"analysis/refit_slice_metrics.csv",refit_slices);write_json(output/"model/multiflow_graph_page_correction.json",model_bundle)
    elapsed=time.monotonic()-started;selected_id=None if selected is None else selected["candidate_id"]
    summary={"schema_version":"phase67-pd-graph-page-shape-refinement-result-v1","status":status,"workflow_commit":preflight["workflow_commit"],"source_result_commit":contract["workflow_parent_result_commit"],"completed_at_utc":utc_now(),"runtime_seconds":elapsed,"counts":{"development_points":480,"phase64_points":240,"phase66_points":240,"models":2,"configurations":4,"topologies":3,"candidates":3,"validation_schemes":4,"oof_predictions":len(evaluation["predictions"]),"gpu_measurements":0,"phase68_targets_used":0},"selection":{"selected_candidate_id":selected_id,"selected_complexity_rank":None if selected is None else selected["complexity_rank"],"first_simplest_passing":selected is not None,"schemes":None if selected is None else selected["schemes"]},"formula":None if selected is None else model_bundle["inference_formula"],"gpu_used":False,"network_used":False,"new_physical_measurement":False,"phase68_targets_opened":False,"fresh_blind_validated":False,"next_phase_permitted":status=="PASS","proved":"development cross-validation support for a lightweight graph+page correction on two models, four graph configurations and L1-L3","not_proved":"Phase68 fresh-blind accuracy, unseen models/configurations, graphs larger than four flows, or scheduler/end-to-end performance"};write_json(output/"summary.json",summary)
    scheme_text="；".join(f"{name} WAPE={100*value['overall_wape']:.3f}%" for name,value in ([] if selected is None else selected["schemes"].items()))
    (output/"README.md").write_text(f"# Phase67：PD多流graph+page修正开发\n\n状态：`{status}`。冻结候选：`{selected_id or 'none'}`。{scheme_text}。本阶段把R64/R66共480点作为development，仅使用CPU；Phase68新网格已在拟合前冻结，但未产生或读取任何Phase68 target。\n",encoding="utf-8")
    write_json(output/"audit/environment.json",environment_record());(output/"logs").mkdir();(output/"logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={preflight['workflow_commit']} status={status} selected={selected_id} runtime_seconds={elapsed:.6f}\ngpu=false network=false new_measurement=false phase68_target=false\n",encoding="utf-8");(output/"DONE").write_text(status+"\n",encoding="utf-8");refresh_manifest(output);tree=validate_result_tree(output)
    if not tree["ok"]:raise RuntimeError(tree)
    return summary

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--expected-workflow-commit",required=True);parser.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase67_pd_graph_page_shape_refinement");args=parser.parse_args();print(json.dumps(run(args.expected_workflow_commit,args.output_dir.resolve()),ensure_ascii=False,indent=2))
