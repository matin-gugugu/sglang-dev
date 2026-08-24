#!/usr/bin/env python3
"""Aggregate complete external Phase60 raw into compact development evidence."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from common import environment_record,load_json,refresh_manifest,repo_root,require_clean_before_run,require_expected_head,utc_now,validate_result_tree,verify_pinned_inputs,write_json  # noqa:E402
from contracts import file_sha,payload_pairs,selected_layouts,validate_pair_contract,validate_plan  # noqa:E402
from measurement import build_analysis,validate_raw  # noqa:E402

def write_csv(path:Path,rows:list[dict])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as target:w=csv.DictWriter(target,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def result_status(runtime:bool,placement:bool)->str:
    if runtime and placement:return "PASS_WITH_RUNTIME_AND_PLACEMENT_VARIANCE"
    if runtime:return "PASS_WITH_RUNTIME_VARIANCE"
    if placement:return "PASS_WITH_PLACEMENT_VARIANCE"
    return "PASS"
def run(expected:str,plan_path:Path,raw_dir:Path,preflight_path:Path,output:Path)->dict:
    head=require_expected_head(expected);require_clean_before_run();spec=load_json(HERE/"experiment.json");pins=verify_pinned_inputs(spec)
    if output.exists():raise RuntimeError(f"refuse overwrite: {output}")
    plan_path=plan_path.expanduser().resolve();raw_dir=raw_dir.expanduser().resolve();preflight_path=preflight_path.expanduser().resolve();plan=load_json(plan_path);plan_audit=validate_plan(plan)
    if plan["workflow_commit"]!=head:raise RuntimeError({"plan_workflow":plan["workflow_commit"],"HEAD":head})
    preflight=load_json(preflight_path);checks=[preflight.get(name,{}) for name in ("environment_checks","module_checks","runtime_checks")]
    if preflight.get("status")!="PASS" or preflight.get("workflow_commit")!=head or preflight.get("plan_audit",{}).get("plan_sha256")!=plan["plan_sha256"] or preflight.get("plan_file_sha256")!=file_sha(plan_path) or Path(preflight.get("raw_dir","")).resolve()!=raw_dir or preflight.get("environment",{}).get("declared_container_image")!=spec["container_contract"]["image"] or any(not row or not all(row.values()) for row in checks):raise RuntimeError("preflight/plan/raw binding failed")
    raw=validate_raw(plan,raw_dir,require_complete=True);analysis=build_analysis(plan,raw);runtime_variance=bool(raw["final_runtime_variance"]);placement_variance=any(row["above_threshold"] for row in analysis["spreads"]);status=result_status(runtime_variance,placement_variance);pairs=validate_pair_contract(spec)
    output.mkdir(parents=True);write_json(output/"contracts/experiment.json",spec);write_json(output/"contracts/topology_plan.json",plan);write_json(output/"contracts/selected_model_transfer_layouts.json",{"schema_version":"phase60-selected-model-layouts-v1","layouts":selected_layouts(spec)});write_json(output/"contracts/payload_pair_grid.json",{"schema_version":"phase60-payload-pair-grid-v1","development":{model:payload_pairs(model) for model in spec["selected_models"]},"reserved_future_blind":{model:payload_pairs(model,"reserved_future_blind_pairs") for model in spec["selected_models"]},"development_sha256":pairs["development_sha256"],"reserved_sha256":pairs["reserved_sha256"]})
    write_json(output/"data/development_composability_points.json",{"schema_version":"phase60-development-composability-points-v1","workflow_commit":head,"plan_sha256":plan["plan_sha256"],"points":analysis["points"],"replica_points":analysis["replica_points"]})
    write_csv(output/"analysis/composability_points.csv",analysis["points"]);write_csv(output/"analysis/composability_metrics.csv",analysis["metrics"]);write_csv(output/"analysis/replica_points.csv",analysis["replica_points"]);write_csv(output/"analysis/replica_spread.csv",analysis["spreads"])
    write_json(output/"audit/input_freeze.json",{"workflow_commit":head,"pinned_inputs":pins,"plan_sha256":plan["plan_sha256"],"plan_file_sha256":file_sha(plan_path),"preflight_file_sha256":file_sha(preflight_path),"development_pairs_sha256":pairs["development_sha256"],"reserved_future_blind_pairs_sha256":pairs["reserved_sha256"]})
    write_json(output/"audit/external_raw_manifest.json",{"schema_version":"phase60-external-raw-manifest-v1","raw_committed_to_git":False,"raw_root_recorded":str(raw_dir),"counts":raw["counts"],"files":raw["files"]})
    write_json(output/"audit/measurement_quality.json",{"schema_version":"phase60-measurement-quality-v1","repeat_policy":spec["measurement_contract"],"measurements":raw["measurements"],"final_runtime_variance":raw["final_runtime_variance"],"placement_spread_threshold":spec["measurement_contract"]["cross_replica_relative_spread_diagnostic_threshold"],"placement_points_above_threshold":sum(row["above_threshold"] for row in analysis["spreads"])})
    safe_keys=("rank","role","hostname","expected_host","physical_gpu","visible_gpu","gpu_name","gpu_uuid","ib_device","mooncake_protocol","with_nvidia_peermem","torch","cuda","python");runtime_endpoints=[]
    for measurement in plan["measurements"]:
        first=next(iter(next(iter(raw["records"][measurement["measurement_id"]].values()))));runtime_endpoints.append({"measurement_id":measurement["measurement_id"],"endpoints":[{key:endpoint.get(key) for key in safe_keys} for endpoint in first["runtime_endpoints"]]})
    write_json(output/"audit/environment.json",{"aggregation":environment_record(),"gpu_measurement_preflight":preflight,"gpu_measurement_runtime_endpoints":runtime_endpoints})
    summary={"schema_version":"phase60-pd-multi-endpoint-composability-result-v1","status":status,"scientific_outcome":analysis["decision"]["scientific_outcome"],"workflow_commit":head,"completed_at_utc":utc_now(),"counts":{"models":2,"configurations":2,"topology_levels":3,"placement_replicas":2,"measurement_shards":24,"development_points":len(analysis["points"]),"replica_points":len(analysis["replica_points"]),"raw_files":raw["counts"]["files"],"raw_records":raw["counts"]["records"],"reserved_future_blind_points_measured":0},"decision":analysis["decision"],"runtime_variance_measurements":len(raw["final_runtime_variance"]),"placement_variance_points":sum(row["above_threshold"] for row in analysis["spreads"]),"transport":"SGLang production MooncakeTransferEngine.batch_transfer_sync over RDMA/dma-buf","training_performed":False,"contention_model_fitted":False,"model_weights_loaded":False,"inference_performed":False,"histograms_recomputed":False,"future_blind_opened":False,"proved":"development physical comparison of frozen Phase51 P1D1 composition, matched solo anchors and P1D2/P2D1 two-flow wave completion","not_proved":"blind contention-model generalization, P2D2, end-to-end latency, dynamic routing/scheduling, compute, memory, queueing, unrelated-job congestion or overlap"}
    write_json(output/"summary.json",summary);(output/"README.md").write_text(f"# Phase60：P1D2/P2D1单链路可组合性\n\n状态：`{status}`；development科学判定：`{summary['scientific_outcome']}`。完成2模型×2配置×3拓扑×2 placement的24个三rank Mooncake/RDMA shard，形成120个official点和240个replica点。Phase60同时比较冻结Phase51曲线与同批次matched-solo锚点，不拟合contention模型，不测未来blind pair，不加载模型或重算直方图。raw逐次样本仅保存在Git外。\n",encoding="utf-8");(output/"logs").mkdir();(output/"logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={head}\nstatus={status} scientific_outcome={summary['scientific_outcome']}\nmeasurements=24 points=120 replica_points=240 raw_files={raw['counts']['files']} raw_records={raw['counts']['records']}\nruntime_variance={len(raw['final_runtime_variance'])} placement_variance_points={summary['placement_variance_points']}\ntraining=false inference=false weights=false future_blind=false raw_committed=false\n",encoding="utf-8");(output/"DONE").write_text(status+"\n",encoding="utf-8");refresh_manifest(output)
    tree=validate_result_tree(output)
    if not tree["ok"]:raise RuntimeError(tree)
    return summary
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--expected-workflow-commit",required=True);p.add_argument("--topology-plan",type=Path,required=True);p.add_argument("--raw-dir",type=Path,required=True);p.add_argument("--preflight-audit",type=Path,required=True);p.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase60_pd_multi_endpoint_composability");a=p.parse_args();print(json.dumps(run(a.expected_workflow_commit,a.topology_plan,a.raw_dir,a.preflight_audit,a.output_dir.resolve()),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
