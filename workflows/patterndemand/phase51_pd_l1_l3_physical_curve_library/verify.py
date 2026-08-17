#!/usr/bin/env python3
"""Independent compact-result verifier for Phase51."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from common import load_json,repo_root,verify_result_manifest  # noqa:E402
from contracts import canonical_sha,model_layouts,validate_plan  # noqa:E402

def read_csv(path:Path)->list[dict]:
    with path.open(encoding="utf-8",newline="") as source:return list(csv.DictReader(source))
def verify(output:Path)->dict:
    expected=load_json(HERE/"expected_outputs.json");missing=[name for name in expected["required"] if not (output/name).is_file()]
    if missing:raise RuntimeError({"missing":missing})
    manifest=verify_result_manifest(output);workflow=load_json(HERE/"experiment.json");result_contract=load_json(output/"contracts/experiment.json");plan=load_json(output/"contracts/topology_plan.json");plan_audit=validate_plan(plan);layouts=load_json(output/"contracts/model_transfer_layouts.json")["layouts"];curves=load_json(output/"curves/pd_mooncake_physical_curves.json");summary=load_json(output/"summary.json");raw=load_json(output/"audit/external_raw_manifest.json");quality=load_json(output/"audit/measurement_quality.json");environment=load_json(output/"audit/environment.json");knots=read_csv(output/"analysis/curve_knots.csv");spreads=read_csv(output/"analysis/replica_spread.csv");statuses=set(workflow["accepted_result_statuses"])
    curve_rows=curves.get("curves",[]);all_knots=[k for c in curve_rows for k in c.get("knots",[])];expected_ids={f"{m['model_id']}__{level.lower()}" for m in model_layouts() for level in ("L1","L2","L3")};layout_knots={row["model_id"]:len(row["knots"]) for row in model_layouts()};repeat_counts={row["measurement_id"]:int(row["repeat_count"]) for row in quality.get("measurements",[])};expected_raw_records=sum(repeat_counts.get(row["measurement_id"],0)*layout_knots[row["model_id"]]*2 for row in plan["measurements"]);runtime_count=len(quality.get("final_runtime_variance",[]));placement_count=int(quality.get("placement_knots_above_threshold",-1));expected_status="PASS" if not runtime_count and not placement_count else "PASS_WITH_RUNTIME_VARIANCE" if runtime_count and not placement_count else "PASS_WITH_PLACEMENT_VARIANCE" if placement_count and not runtime_count else "PASS_WITH_RUNTIME_AND_PLACEMENT_VARIANCE"
    checks={
      "manifest":manifest["ok"],"required_exact":set(expected["required"])=={str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()},
      "status":summary.get("status") in statuses and summary.get("status")==expected_status and (output/"DONE").read_text().strip()==summary.get("status"),"contract_exact":result_contract==workflow,
      "plan":plan_audit["measurements"]==36 and plan.get("workflow_commit")==summary.get("workflow_commit"),"layouts":layouts==model_layouts() and canonical_sha(layouts)==plan.get("model_layouts_sha256"),
      "curve_counts":len(curve_rows)==18 and len(all_knots)==396 and {c.get("curve_id") for c in curve_rows}==expected_ids,"csv_counts":len(knots)==396 and len(spreads)==396,
      "curve_policy":all(len(c.get("knots",[]))==(12 if c.get("model_id")=="deepseek-v2-lite" else 24) for c in curve_rows) and all(float(k.get("official_latency_us",0))>0 and float(k.get("official_effective_gib_per_s",0))>0 and len(k.get("replicas",[]))==2 and _official(k) for k in all_knots),
      "raw_external":raw.get("raw_committed_to_git") is False and raw.get("counts",{}).get("measurements_with_data")==36 and raw.get("counts",{}).get("files")==sum(repeat_counts.values()) and raw.get("counts",{}).get("records")==expected_raw_records and len(raw.get("files",[]))==raw.get("counts",{}).get("files") and all(row.get("path","").endswith(".jsonl") and len(row.get("sha256",""))==64 and row.get("records",0)>0 for row in raw.get("files",[])),
      "quality":len(quality.get("measurements",[]))==36 and all(int(row.get("repeat_count",0)) in (5,7,9) for row in quality.get("measurements",[])),
      "runtime_endpoints":len(environment.get("gpu_measurement_runtime_endpoints",[]))==36 and all(len(row.get("endpoints",[]))==2 and all(ep.get("mooncake_protocol")=="rdma" and ep.get("with_nvidia_peermem")=="0" for ep in row.get("endpoints",[])) for row in environment.get("gpu_measurement_runtime_endpoints",[])),
      "no_raw":not list(output.rglob("*.jsonl")),"scope":summary.get("training_performed") is False and summary.get("model_weights_loaded") is False and summary.get("inference_performed") is False and summary.get("histograms_recomputed") is False and summary.get("cost_convolution_performed") is False and summary.get("placement_decision_performed") is False,
    }
    if not all(checks.values()):raise RuntimeError({"checks":checks,"manifest":manifest})
    return {"status":"PASS","checks":checks,"workflow_commit":summary["workflow_commit"],"result_status":summary["status"],"curves":18,"knots":396,"manifest_files":manifest["manifest"]["checked_files"]}
def _official(knot:dict)->bool:
    replica_values=[]
    for replica in knot.get("replicas",[]):
        direction_values=[float(v["median_across_repeats_us"]) for v in replica.get("directions",{}).values()]
        if len(direction_values)!=2 or abs(float(replica.get("slower_direction_latency_us",-1))-max(direction_values))>1e-7:return False
        replica_values.append(max(direction_values))
    return len(replica_values)==2 and abs(float(knot.get("official_latency_us",-1))-max(replica_values))<=1e-7
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase51_pd_l1_l3_physical_curve_library");a=p.parse_args();print(json.dumps(verify(a.output_dir.resolve()),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
