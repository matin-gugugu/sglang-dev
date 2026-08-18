#!/usr/bin/env python3
"""CPU-only Phase52 ancestry, pin, roster and curve audit."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from common import load_json,repo_root,require_clean_before_run,require_expected_head,verify_pinned_inputs  # noqa:E402
from analysis import read_csv,validate_inputs  # noqa:E402

def run_checks(expected:str)->dict:
    spec=load_json(HERE/"experiment.json");head=require_expected_head(expected);require_clean_before_run(allowed_untracked_prefixes=())
    parents=subprocess.check_output(["git","show","-s","--format=%P",head],cwd=repo_root(),text=True).strip().split()
    if parents!=[spec["workflow_parent_result_commit"]]:raise RuntimeError({"W52_parent":parents,"expected_R51":spec["workflow_parent_result_commit"]})
    output=repo_root()/spec["result_dir"]
    if output.exists():raise RuntimeError(f"formal result already exists: {output}")
    pins=verify_pinned_inputs(spec);paths={row["name"]:repo_root()/row["path"] for row in spec["pinned_inputs"]};predictions=read_csv(paths["phase49_frozen_predictions"]);targets=read_csv(paths["phase50_hfull_targets"]);curve_payload=load_json(paths["phase51_curves"]);curves=curve_payload.get("curves",[]);inputs=validate_inputs(predictions,targets,curves,spec);p50=load_json(paths["phase50_summary"]);p51=load_json(paths["phase51_summary"]);quality=load_json(paths["phase51_quality"])
    source_checks={"phase50_confirmed":p50.get("gates",{}).get("confirmed") is True,"phase50_six_models":p50.get("counts",{}).get("models")==6,"phase51_accepted":p51.get("status") in load_json(repo_root()/"workflows/patterndemand/phase51_pd_l1_l3_physical_curve_library/experiment.json")["accepted_result_statuses"],"phase51_curves_18":p51.get("counts",{}).get("physical_curves")==18,"phase51_knots_396":p51.get("counts",{}).get("curve_knots")==396,"phase51_quality_36":len(quality.get("measurements",[]))==36,"phase51_plan_bound":curve_payload.get("plan_sha256")==load_json(paths["phase51_topology_plan"]).get("plan_sha256")}
    if not all(source_checks.values()):raise RuntimeError({"source_checks":source_checks})
    return {"schema_version":"phase52-preflight-v1","status":"PASS","workflow_commit":head,"workflow_parent_result_commit":parents[0],"pinned_inputs":pins,"input_audit":inputs,"source_checks":source_checks,"gpu_used":False,"network_used":False,"training_used":False,"checkpoint_loaded":False,"prediction_recomputed":False,"teacher_recomputed":False}
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--expected-workflow-commit",required=True);a=p.parse_args();print(json.dumps(run_checks(a.expected_workflow_commit),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
