#!/usr/bin/env python3
"""Read-only Phase67 contract and blind-boundary audit."""
from __future__ import annotations
import hashlib,json,sys
from collections import Counter
from pathlib import Path
from typing import Any
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from common import load_json,repo_root,require_clean_before_run,require_expected_head,utc_now,verify_pinned_inputs  # noqa:E402
from model import read_development  # noqa:E402

def canonical_sha(value:Any)->str:return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def validate_phase68(contract:dict[str,Any])->dict[str,Any]:
    grid=load_json(repo_root()/contract["phase68_blind_boundary"]["reserved_grid"]); base={k:v for k,v in grid.items() if k!="grid_sha256"}
    reserved={int(p) for vectors in grid["configurations"].values() for vector in vectors for p in vector}; development=set(contract["dataset_contract"]["phase64_page_set"]+contract["dataset_contract"]["phase66_page_set"])
    checks={"schema":grid.get("schema_version")=="phase68-reserved-multiflow-second-blind-grid-v1","canonical_sha":grid.get("grid_sha256")==canonical_sha(base)==contract["phase68_blind_boundary"]["reserved_grid_canonical_sha256"],"frozen_closed":grid.get("frozen_before_phase67_fit") is True and grid.get("phase68_targets_opened") is False,"models":grid.get("models")==contract["dataset_contract"]["models"],"topologies":grid.get("topology_levels")==contract["dataset_contract"]["topology_levels"],"configurations":list(grid.get("configurations",{}))==contract["dataset_contract"]["configurations"],"ten_vectors":all(len(v)==10 for v in grid["configurations"].values()),"page_set":reserved==set(contract["phase68_blind_boundary"]["page_set"]),"zero_overlap":not(reserved&development),"curve_bracket":min(reserved)>=32 and max(reserved)<=64,"new_placement":grid.get("required_new_placement_policy","").startswith("Every Phase68 endpoint tuple")}
    if not all(checks.values()):raise RuntimeError({"phase68_grid":checks})
    return {"status":"PASS","checks":checks,"grid_sha256":grid["grid_sha256"],"development_pages":sorted(development),"reserved_pages":sorted(reserved),"phase68_targets_read":False}

def run_checks(expected:str)->dict[str,Any]:
    head=require_expected_head(expected);require_clean_before_run();contract=load_json(HERE/"experiment.json");pins=verify_pinned_inputs(contract);root=repo_root()
    s64=load_json(root/"experiment-results/phase64_pd_multiflow_graph_zero_shot/summary.json");s66=load_json(root/"experiment-results/phase66_pd_graph_correction_fresh_blind/summary.json")
    if s64.get("scientific_outcome")!="MULTIFLOW_GRAPH_ZERO_SHOT_FAIL_RETAIN_FOR_DEVELOPMENT" or s64.get("decision",{}).get("zero_shot_gate_pass") is not False:raise RuntimeError("R64 is not valid retained development evidence")
    if s66.get("scientific_outcome")!="MULTIFLOW_GRAPH_CORRECTION_FRESH_BLIND_FAIL_RETAIN_AS_BLIND_EVIDENCE" or s66.get("decision",{}).get("fresh_blind_gate_pass") is not False:raise RuntimeError("R66 is not valid retained blind evidence")
    r65=load_json(root/"experiment-results/phase65_pd_graph_correction_development/model/multiflow_graph_correction.json");rows=read_development(root/contract["dataset_contract"]["phase64_source"],root/contract["dataset_contract"]["phase66_source"],r65)
    counts={"source":Counter(r["source_phase"] for r in rows),"model":Counter(r["model_id"] for r in rows),"configuration":Counter(r["configuration"] for r in rows),"topology":Counter(r["topology_level"] for r in rows),"cohort":Counter((r["source_phase"],r["vector_index"]) for r in rows)}
    checks={"rows":len(rows)==480,"source":counts["source"]==Counter({"phase64":240,"phase66":240}),"models":counts["model"]==Counter({m:240 for m in contract["dataset_contract"]["models"]}),"configurations":counts["configuration"]==Counter({c:120 for c in contract["dataset_contract"]["configurations"]}),"topologies":counts["topology"]==Counter({t:160 for t in contract["dataset_contract"]["topology_levels"]}),"cohorts":len(counts["cohort"])==20 and set(counts["cohort"].values())=={24},"positive":all(r["actual_concurrent_wave_us"]>0 and r["max_edge_baseline_us"]>0 and r["r61_prediction_us"]>0 and r["r65_prediction_us"]>0 for r in rows)}
    if not all(checks.values()):raise RuntimeError({"development_matrix":checks,"counts":counts})
    blind=validate_phase68(contract)
    return {"schema_version":"phase67-preflight-v1","status":"PASS","workflow_commit":head,"captured_at_utc":utc_now(),"pinned_inputs":pins,"checks":checks,"phase64":{"status":s64["status"],"scientific_outcome":s64["scientific_outcome"],"labels_now_development":True},"phase66":{"status":s66["status"],"scientific_outcome":s66["scientific_outcome"],"labels_now_development":True},"phase68_reserved_grid":blind,"execution":{"gpu_used":False,"network_used":False,"new_physical_measurement":False,"phase68_targets_read":False}}

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--expected-workflow-commit",required=True);args=parser.parse_args();print(json.dumps(run_checks(args.expected_workflow_commit),ensure_ascii=False,indent=2,default=dict))
