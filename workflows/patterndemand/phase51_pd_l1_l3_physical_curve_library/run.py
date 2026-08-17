#!/usr/bin/env python3
"""Aggregate complete external Phase51 raw into a compact Git result."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
import sys;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from common import environment_record,load_json,refresh_manifest,repo_root,require_clean_before_run,require_expected_head,utc_now,validate_result_tree,verify_pinned_inputs,write_json  # noqa:E402
from contracts import file_sha,model_layouts,validate_plan  # noqa:E402
from measurement import build_curves,validate_raw  # noqa:E402

def write_csv(path:Path,rows:list[dict])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as target:w=csv.DictWriter(target,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run(expected:str,plan_path:Path,raw_dir:Path,preflight_path:Path,output:Path)->dict:
    head=require_expected_head(expected);require_clean_before_run();contract=load_json(HERE/"experiment.json");pins=verify_pinned_inputs(contract)
    if output.exists():raise RuntimeError(f"refuse overwrite: {output}")
    plan=load_json(plan_path.expanduser().resolve());plan_audit=validate_plan(plan)
    if plan["workflow_commit"]!=head:raise RuntimeError({"plan_workflow":plan["workflow_commit"],"HEAD":head})
    preflight=load_json(preflight_path.expanduser().resolve())
    preflight_checks=[preflight.get(name,{}) for name in ("environment_checks","module_checks","runtime_checks")]
    if preflight.get("status")!="PASS" or preflight.get("workflow_commit")!=head or preflight.get("plan_audit",{}).get("plan_sha256")!=plan["plan_sha256"] or preflight.get("plan_file_sha256")!=file_sha(plan_path.expanduser().resolve()) or Path(preflight.get("raw_dir","")).resolve()!=raw_dir.expanduser().resolve() or preflight.get("environment",{}).get("declared_container_image")!=contract["container_contract"]["image"] or any(not checks or not all(checks.values()) for checks in preflight_checks):raise RuntimeError("preflight/plan/raw binding failed")
    raw=validate_raw(plan,raw_dir,require_complete=True);library=build_curves(plan,raw);runtime_variance=bool(raw["final_runtime_variance"]);placement_variance=any(row["above_threshold"] for row in library["spreads"])
    status="PASS"+("_WITH_RUNTIME_VARIANCE" if runtime_variance else "")+("_AND_PLACEMENT_VARIANCE" if runtime_variance and placement_variance else "_WITH_PLACEMENT_VARIANCE" if placement_variance else "")
    if status=="PASS_WITH_RUNTIME_VARIANCE_AND_PLACEMENT_VARIANCE":status="PASS_WITH_RUNTIME_AND_PLACEMENT_VARIANCE"
    output.mkdir(parents=True);write_json(output/"curves/pd_mooncake_physical_curves.json",{k:v for k,v in library.items() if k!="spreads"});write_json(output/"contracts/experiment.json",contract);write_json(output/"contracts/topology_plan.json",plan);write_json(output/"contracts/model_transfer_layouts.json",{"schema_version":"phase51-model-transfer-layouts-v1","layouts":model_layouts()})
    knot_rows=[]
    for curve in library["curves"]:
        for knot in curve["knots"]:knot_rows.append({"curve_id":curve["curve_id"],"model_id":curve["model_id"],"topology_level":curve["topology_level"],"page_count":knot["page_count"],"payload_bytes":knot["payload_bytes"],"descriptor_count":knot["descriptor_count"],"descriptor_bytes":knot["descriptor_bytes"],"official_latency_us":knot["official_latency_us"],"official_effective_gib_per_s":knot["official_effective_gib_per_s"],"cross_replica_relative_spread":knot["cross_replica_relative_spread"]})
    write_csv(output/"analysis/curve_knots.csv",knot_rows);write_csv(output/"analysis/replica_spread.csv",library["spreads"])
    write_json(output/"audit/input_freeze.json",{"workflow_commit":head,"pinned_inputs":pins,"plan_sha256":plan["plan_sha256"],"plan_file_sha256":file_sha(plan_path.expanduser().resolve()),"preflight_file_sha256":file_sha(preflight_path.expanduser().resolve())})
    write_json(output/"audit/external_raw_manifest.json",{"schema_version":"phase51-external-raw-manifest-v1","raw_committed_to_git":False,"raw_root_recorded":str(raw_dir.expanduser().resolve()),"counts":raw["counts"],"files":raw["files"]})
    write_json(output/"audit/measurement_quality.json",{"schema_version":"phase51-measurement-quality-v1","repeat_policy":contract["measurement_contract"],"measurements":raw["measurements"],"final_runtime_variance":raw["final_runtime_variance"],"placement_spread_threshold":contract["measurement_contract"]["cross_replica_relative_spread_diagnostic_threshold"],"placement_knots_above_threshold":sum(row["above_threshold"] for row in library["spreads"])})
    safe_endpoint_keys=("rank","hostname","expected_host","physical_gpu","visible_gpu","gpu_name","gpu_uuid","ib_device","mooncake_protocol","with_nvidia_peermem","torch","cuda","python")
    runtime_endpoints=[]
    for measurement in plan["measurements"]:
        first_record=next(iter(next(iter(raw["records"][measurement["measurement_id"]].values()))))
        runtime_endpoints.append({"measurement_id":measurement["measurement_id"],"endpoints":[{key:endpoint.get(key) for key in safe_endpoint_keys} for endpoint in first_record["runtime_endpoints"]]})
    write_json(output/"audit/environment.json",{"aggregation":environment_record(),"gpu_measurement_preflight":preflight,"gpu_measurement_runtime_endpoints":runtime_endpoints})
    summary={"schema_version":"phase51-pd-l1-l3-physical-curve-result-v1","status":status,"workflow_commit":head,"completed_at_utc":utc_now(),"counts":{"models":6,"topology_levels":3,"placement_replicas_per_model_topology":2,"measurement_shards":36,"physical_curves":18,"curve_knots":396,"raw_files":raw["counts"]["files"],"raw_records":raw["counts"]["records"]},"transport":"SGLang production MooncakeTransferEngine.batch_transfer_sync over RDMA/dma-buf","curve_policy":"median independent repeats, slower direction, slower of two frozen placement replicas","runtime_variance_measurements":len(raw["final_runtime_variance"]),"placement_variance_knots":sum(row["above_threshold"] for row in library["spreads"]),"training_performed":False,"model_weights_loaded":False,"inference_performed":False,"histograms_recomputed":False,"cost_convolution_performed":False,"placement_decision_performed":False,"proved":"physical P-to-D KV batch-transfer curves on the 36 frozen endpoint/model-layout shards within measured payload support","not_proved":"end-to-end serving latency, scheduling quality, queueing/congestion, compute overlap, unseen topology generalization or placement optimality"}
    write_json(output/"summary.json",summary);(output/"README.md").write_text(f"# Phase51：纯PD L1–L3物理通信曲线库\n\n状态：`{status}`。使用SGLang生产Mooncake batch-transfer + RDMA/dma-buf，完成6模型×3拓扑×2冻结placement的36个测量shard，汇总18条模型相关曲线、396个物理knots。正式值保守取重复中位数、双向较慢值、两个placement较慢值。raw逐次样本仅保存在Git外。Phase51未训练、未加载模型、未重算直方图，也未做代价卷积或placement决策；后者属于Phase52。\n",encoding="utf-8");(output/"logs").mkdir();(output/"logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={head}\nstatus={status} measurements=36 curves=18 knots=396 raw_files={raw['counts']['files']} raw_records={raw['counts']['records']}\nruntime_variance={len(raw['final_runtime_variance'])} placement_variance_knots={sum(row['above_threshold'] for row in library['spreads'])}\ntraining=false inference=false weights=false raw_committed=false cost_convolution=false placement=false\n",encoding="utf-8");(output/"DONE").write_text(status+"\n",encoding="utf-8");refresh_manifest(output)
    tree=validate_result_tree(output)
    if not tree["ok"]:raise RuntimeError(tree)
    return summary
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--expected-workflow-commit",required=True);p.add_argument("--topology-plan",type=Path,required=True);p.add_argument("--raw-dir",type=Path,required=True);p.add_argument("--preflight-audit",type=Path,required=True);p.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase51_pd_l1_l3_physical_curve_library");a=p.parse_args();print(json.dumps(run(a.expected_workflow_commit,a.topology_plan,a.raw_dir,a.preflight_audit,a.output_dir.resolve()),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
