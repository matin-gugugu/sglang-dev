#!/usr/bin/env python3
"""Independent compact-result verifier for Phase60."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from common import load_json,repo_root,verify_result_manifest  # noqa:E402
from contracts import canonical_sha,payload_pairs,selected_layouts,validate_pair_contract,validate_plan  # noqa:E402

def read_csv(path:Path)->list[dict]:
    with path.open(encoding="utf-8",newline="") as source:return list(csv.DictReader(source))
def truth(value:str)->bool:return value.lower()=="true"
def expected_outcome(metrics:list[dict],spec:dict)->tuple[str,bool,bool]:
    overall=next(row for row in metrics if row["slice_type"]=="overall");slices=[row for row in metrics if row["slice_type"]=="config_topology"];dc=spec["development_decision_contract"]
    phase=float(overall["phase51_wape"])<=float(dc["overall_wape_threshold"]) and all(float(row["phase51_wape"])<=float(dc["config_topology_wape_threshold"]) for row in slices);matched=float(overall["matched_solo_wape"])<=float(dc["overall_wape_threshold"]) and all(float(row["matched_solo_wape"])<=float(dc["config_topology_wape_threshold"]) for row in slices)
    return ("P1D1_DIRECTLY_COMPOSABLE_DEVELOPMENT" if phase else "P1D1_CURVE_TRANSFER_DRIFT_REQUIRES_REVIEW" if matched else "CONTENTION_CORRECTION_CANDIDATE",phase,matched)
def verify(output:Path)->dict:
    expected=load_json(HERE/"expected_outputs.json");files={str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()};missing=[name for name in expected["required"] if name not in files]
    if missing:raise RuntimeError({"missing":missing})
    manifest=verify_result_manifest(output);spec=load_json(HERE/"experiment.json");result_spec=load_json(output/"contracts/experiment.json");plan=load_json(output/"contracts/topology_plan.json");plan_audit=validate_plan(plan);layouts=load_json(output/"contracts/selected_model_transfer_layouts.json")["layouts"];pair_grid=load_json(output/"contracts/payload_pair_grid.json");data=load_json(output/"data/development_composability_points.json");summary=load_json(output/"summary.json");raw=load_json(output/"audit/external_raw_manifest.json");quality=load_json(output/"audit/measurement_quality.json");environment=load_json(output/"audit/environment.json");points=read_csv(output/"analysis/composability_points.csv");metrics=read_csv(output/"analysis/composability_metrics.csv");replicas=read_csv(output/"analysis/replica_points.csv");spreads=read_csv(output/"analysis/replica_spread.csv");pairs=validate_pair_contract(spec);outcome,phase_pass,matched_pass=expected_outcome(metrics,spec)
    runtime_count=len(quality.get("final_runtime_variance",[]));placement_count=int(quality.get("placement_points_above_threshold",-1));expected_status="PASS_WITH_RUNTIME_AND_PLACEMENT_VARIANCE" if runtime_count and placement_count else "PASS_WITH_RUNTIME_VARIANCE" if runtime_count else "PASS_WITH_PLACEMENT_VARIANCE" if placement_count else "PASS";repeat_counts={row["measurement_id"]:int(row["repeat_count"]) for row in quality.get("measurements",[])};expected_raw_records=sum(repeat_counts.values())*10
    expected_pair_ids={row["pair_id"] for model in spec["selected_models"] for row in payload_pairs(model)};reserved_pair_ids={row["pair_id"] for model in spec["selected_models"] for row in payload_pairs(model,"reserved_future_blind_pairs")}
    checks={
      "manifest":manifest["ok"],"required_exact":set(expected["required"])==files,"status":summary.get("status")==expected_status and summary.get("status") in spec["accepted_result_statuses"] and (output/"DONE").read_text().strip()==expected_status,
      "contract_exact":result_spec==spec,"plan":plan_audit["measurements"]==24 and plan.get("workflow_commit")==summary.get("workflow_commit"),"layouts":layouts==selected_layouts(spec),
      "pair_grid":pair_grid.get("development_sha256")==pairs["development_sha256"] and pair_grid.get("reserved_sha256")==pairs["reserved_sha256"] and sum(len(v) for v in pair_grid.get("development",{}).values())==20 and sum(len(v) for v in pair_grid.get("reserved_future_blind",{}).values())==20,
      "point_counts":len(points)==120 and len(replicas)==240 and len(spreads)==120 and len(data.get("points",[]))==120 and len(data.get("replica_points",[]))==240,
      "pair_roles":{row["pair_id"] for row in points}==expected_pair_ids and not ({row["pair_id"] for row in points}&reserved_pair_ids),
      "metric_counts":len(metrics)==14 and sum(row["slice_type"]=="overall" for row in metrics)==1 and sum(row["slice_type"]=="config_topology" for row in metrics)==6,
      "positive_values":all(float(row[key])>0 for row in points for key in ("phase51_ideal_us","matched_solo_ideal_us","actual_concurrent_wave_us")),
      "official_replica_policy":all(abs(float(row["official_us"])-max(float(row["replica0_us"]),float(row["replica1_us"])))<=1e-7 for row in spreads),
      "decision":summary.get("scientific_outcome")==outcome and summary.get("decision",{}).get("scientific_outcome")==outcome and summary.get("decision",{}).get("phase51_baseline_pass")==phase_pass and summary.get("decision",{}).get("matched_solo_baseline_pass")==matched_pass,
      "raw_external":raw.get("raw_committed_to_git") is False and raw.get("counts",{}).get("measurements_with_data")==24 and raw.get("counts",{}).get("files")==sum(repeat_counts.values()) and raw.get("counts",{}).get("records")==expected_raw_records and len(raw.get("files",[]))==raw.get("counts",{}).get("files") and all(row.get("path","").endswith(".jsonl") and len(row.get("sha256",""))==64 and row.get("records")==10 for row in raw.get("files",[])),
      "quality":len(quality.get("measurements",[]))==24 and all(int(row.get("repeat_count",0)) in (5,7,9) for row in quality.get("measurements",[])),
      "runtime_endpoints":len(environment.get("gpu_measurement_runtime_endpoints",[]))==24 and all(len(row.get("endpoints",[]))==3 and all(ep.get("mooncake_protocol")=="rdma" and ep.get("with_nvidia_peermem")=="0" for ep in row.get("endpoints",[])) for row in environment.get("gpu_measurement_runtime_endpoints",[])),
      "no_raw":not list(output.rglob("*.jsonl")),"scope":summary.get("training_performed") is False and summary.get("contention_model_fitted") is False and summary.get("model_weights_loaded") is False and summary.get("inference_performed") is False and summary.get("histograms_recomputed") is False and summary.get("future_blind_opened") is False and summary.get("counts",{}).get("reserved_future_blind_points_measured")==0,
    }
    if not all(checks.values()):raise RuntimeError({"checks":checks,"manifest":manifest})
    return {"status":"PASS","checks":checks,"workflow_commit":summary["workflow_commit"],"result_status":summary["status"],"scientific_outcome":outcome,"points":120,"manifest_files":manifest["manifest"]["checked_files"]}
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase60_pd_multi_endpoint_composability");a=p.parse_args();print(json.dumps(verify(a.output_dir.resolve()),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
