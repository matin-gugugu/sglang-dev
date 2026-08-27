#!/usr/bin/env python3
"""Train, freeze and evaluate the independent Phase73 Direct-GBDT baseline."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from common import environment_record,load_json,refresh_manifest,repo_root,utc_now,validate_result_tree,write_json  # noqa:E402
from analysis import bootstrap,evaluate,refit,select_candidate,svg_models,svg_overall  # noqa:E402
from gbdt import feature_importance,read_csv_gz,write_csv,write_csv_gz,write_json_gz  # noqa:E402
from preflight import run_checks  # noqa:E402


def digest_rows(rows:list[dict])->str:return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def run(expected:str,output:Path)->dict:
    started=time.perf_counter();freeze=run_checks(expected);spec=load_json(HERE/"experiment.json");root=repo_root()
    if output.exists():raise RuntimeError(f"refuse overwrite: {output}")
    paths={row["name"]:root/row["path"] for row in spec["pinned_inputs"]};examples=read_csv_gz(paths["phase48_examples"])
    selected,candidate_rows,selection=select_candidate(examples,spec);model=refit(examples,selection["feature_names"],selected,int(spec["candidate_selection"]["seed"]));importance=feature_importance(model)
    # Freeze the selected model and selection audit before loading the already-open Phase50 labels.
    output.mkdir(parents=True);write_json_gz(output/"checkpoints/direct_gbdt.json.gz",model);write_json(output/"audit/training.json",{"schema_version":"phase73-training-audit-v1","workflow_commit":expected,"selection_data":"Phase48 only","phase50_labels_loaded_during_selection":False,"selection_digest_before_phase50_load":selection["selection_digest_before_phase50_load"],"selected_candidate":selected,"candidate_rows_without_elapsed":[{k:v for k,v in row.items() if k!="elapsed_seconds"} for row in candidate_rows],"feature_count":len(selection["feature_names"]),"feature_names":selection["feature_names"],"h0_inputs":0,"pseudo_requests":0,"teacher_calls":0,"refit_rows":len(examples),"outputs":26,"trees":26*int(selected["estimators"])})
    features=read_csv_gz(paths["phase49_features"]);frozen_predictions=read_csv_gz(paths["phase49_predictions"]);targets=read_csv_gz(paths["phase50_targets"]);result=evaluate(features,frozen_predictions,targets,model,spec);boot=bootstrap(result["per_unit"])
    counts={**spec["expected_counts"],"selected_features":len(selection["feature_names"]),"selected_trees":26*int(selected["estimators"])}
    if len(candidate_rows)!=counts["candidate_rows"] or len(result["per_unit"])!=counts["per_unit_rows"] or len(result["per_bin"])!=counts["per_bin_rows"]:raise RuntimeError("result cardinality mismatch")
    write_json(output/"contracts/experiment.json",spec);write_json(output/"audit/input_freeze.json",freeze);write_json(output/"audit/environment.json",{**environment_record(),"numpy":np.__version__,"gpu_used":False,"network_used":False,"raw_used":False,"teacher_executed":False,"pseudo_requests_constructed":False,"h0_used_as_direct_input":False})
    write_csv(output/"analysis/candidate_validation.csv",candidate_rows);write_csv(output/"analysis/aggregate_metrics.csv",result["aggregate"]);write_json(output/"analysis/model_metrics.json",result["models"]);write_json(output/"analysis/segment_metrics.json",result["segments"]);write_csv(output/"analysis/per_bin_metrics.csv",result["per_bin"]);write_csv_gz(output/"analysis/per_unit_metrics.csv.gz",result["per_unit"]);write_csv(output/"analysis/feature_importance.csv",importance);write_json(output/"analysis/profile_cluster_bootstrap.json",boot);write_csv_gz(output/"predictions/phase50_direct_gbdt_predictions.csv.gz",result["prediction_rows"])
    (output/"figures").mkdir();(output/"figures/overall_histogram_wape.svg").write_text(svg_overall(result["aggregate"]),encoding="utf-8");(output/"figures/model_histogram_wape.svg").write_text(svg_models(result["models"]),encoding="utf-8")
    metrics={row["method"]:row for row in result["aggregate"]};direct=metrics["direct_gbdt"]
    summary={"schema_version":"phase73-direct-gbdt-baseline-result-v1","status":"PASS","scientific_outcome":result["decision"]["scientific_outcome"],"workflow_commit":expected,"completed_at_utc":utc_now(),"benchmark_classification":"target-open fixed benchmark; not fresh blind","selected_candidate":selected,"counts":counts,"decision":result["decision"],"overall_metrics":metrics,"headline":{"direct_calls_histogram_wape":direct["calls_histogram_wape"],"direct_bytes_histogram_wape":direct["bytes_histogram_wape"],"h0_calls_histogram_wape":metrics["h0"]["calls_histogram_wape"],"h0_bytes_histogram_wape":metrics["h0"]["bytes_histogram_wape"],"dnn_calls_histogram_wape":metrics["h0_plus_dnn_residual"]["calls_histogram_wape"],"dnn_bytes_histogram_wape":metrics["h0_plus_dnn_residual"]["bytes_histogram_wape"]},"training_scope":{"phase48_only_for_selection_and_refit":True,"phase50_labels_loaded_after_model_freeze":True,"selection_digest_before_phase50_load":selection["selection_digest_before_phase50_load"]},"execution":{"gpu_used":False,"network_used":False,"raw_used":False,"teacher_executed":False,"pseudo_requests_constructed":False,"h0_used_as_direct_input":False,"elapsed_seconds":time.perf_counter()-started},"proved":"a no-H0 no-pseudo-request Direct-GBDT comparison on the frozen Phase50 target-open six-model benchmark","not_proved":"fresh-blind generalization, strict target attainment unless gates pass, unseen models, physical cost, placement or scheduler benefit"}
    write_json(output/"summary.json",summary);(output/"README.md").write_text(f"# Phase73：Direct-GBDT独立baseline\n\n状态：`PASS`；科学结果：`{summary['scientific_outcome']}`。本方法只读`feature_*`，不使用H0、伪请求或teacher。Phase50标签早已公开，因此这是target-open固定基准，不是fresh blind。Direct-GBDT calls/bytes histogram WAPE为{direct['calls_histogram_wape']*100:.2f}%/{direct['bytes_histogram_wape']*100:.2f}%。\n",encoding="utf-8");(output/"logs").mkdir();(output/"logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\nselected={selected['candidate_id']} features={len(selection['feature_names'])} trees={26*int(selected['estimators'])}\noutcome={summary['scientific_outcome']} direct_calls={direct['calls_histogram_wape']:.12f} direct_bytes={direct['bytes_histogram_wape']:.12f}\ngpu=false network=false raw=false teacher=false pseudo=false h0_input=false fresh_blind=false\n",encoding="utf-8");(output/"DONE").write_text("PASS\n",encoding="utf-8");refresh_manifest(output);tree=validate_result_tree(output)
    if not tree["ok"]:raise RuntimeError(tree)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--expected-workflow-commit",required=True);parser.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase73_direct_gbdt_baseline");args=parser.parse_args();print(json.dumps(run(args.expected_workflow_commit,args.output_dir.resolve()),ensure_ascii=False,indent=2))
