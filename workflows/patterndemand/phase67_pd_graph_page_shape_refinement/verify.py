#!/usr/bin/env python3
"""Independent deterministic verifier for Phase67 compact result."""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
from typing import Any
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from common import load_json,repo_root,verify_result_manifest  # noqa:E402
from model import CANDIDATES,evaluate,fit_model,read_development  # noqa:E402
from preflight import validate_phase68  # noqa:E402
from run import metric_rows  # noqa:E402

def read_csv(path:Path)->list[dict[str,str]]:
    with path.open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))
def scalar_equal(left:str,right:Any)->bool:
    if isinstance(right,bool):return left.lower()==str(right).lower()
    if isinstance(right,(int,float)):return math.isclose(float(left),float(right),rel_tol=1e-12,abs_tol=1e-9)
    if right is None:return left==""
    return left==str(right)
def rows_equal(left:list[dict[str,str]],right:list[dict[str,Any]])->bool:return len(left)==len(right) and all(all(key in a and scalar_equal(a[key],value) for key,value in b.items()) for a,b in zip(left,right))

def verify(output:Path)->dict[str,Any]:
    expected=load_json(HERE/"expected_outputs.json");files={str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()};manifest=verify_result_manifest(output);contract=load_json(HERE/"experiment.json");summary=load_json(output/"summary.json");preflight=load_json(output/"audit/preflight.json");freeze=load_json(output/"audit/input_freeze.json");model=load_json(output/"model/multiflow_graph_page_correction.json");root=repo_root();r65=load_json(root/"experiment-results/phase65_pd_graph_correction_development/model/multiflow_graph_correction.json");rows=read_development(root/contract["dataset_contract"]["phase64_source"],root/contract["dataset_contract"]["phase66_source"],r65);evaluation=evaluate(rows,contract);selected=evaluation["selected"];status="PASS" if selected else "PASS_TARGET_NOT_MET";selected_id=None if selected is None else selected["candidate_id"];metrics=read_csv(output/"analysis/candidate_metrics.csv");predictions=read_csv(output/"analysis/oof_predictions.csv");slices=read_csv(output/"analysis/oof_slice_metrics.csv");expected_model=None if selected is None else fit_model(rows,next(c for c in CANDIDATES if c["candidate_id"]==selected_id));blind=validate_phase68(contract)
    checks={"manifest":manifest["ok"],"required_exact":files==set(expected["required"]),"contract_exact":load_json(output/"contracts/experiment.json")==contract,"grid_exact":load_json(output/"contracts/phase68_reserved_blind_grid.json")==load_json(HERE/"phase68_reserved_blind_grid.json") and blind["status"]=="PASS","status":summary.get("status")==status and (output/"DONE").read_text().strip()==status and status in contract["accepted_result_statuses"],"preflight":preflight.get("status")=="PASS" and all(preflight.get("checks",{}).values()) and preflight.get("execution",{}).get("phase68_targets_read") is False,"candidate_metrics":rows_equal(metrics,metric_rows(evaluation)),"oof_predictions":rows_equal(predictions,evaluation["predictions"]),"oof_slices":rows_equal(slices,evaluation["slices"]),"cardinality":len(metrics)==3 and len(predictions)==9180 and len(slices)==456,"selection":selected_id=="model_configuration_graph_page_sqrt" and summary.get("selection",{}).get("first_simplest_passing") is True,"model":status!="PASS" or (model.get("candidate_id")==selected_id and model.get("groups")==expected_model["groups"] and model.get("selection_evidence",{}).get("first_simplest_passing") is True),"blind_boundary":freeze.get("phase68_measurements_or_targets_read") is False and summary.get("phase68_targets_opened") is False and summary.get("fresh_blind_validated") is False,"scope":summary.get("gpu_used") is False and summary.get("network_used") is False and summary.get("new_physical_measurement") is False and summary.get("counts",{}).get("gpu_measurements")==0}
    if not all(checks.values()):raise RuntimeError({"phase67_checks":checks,"manifest":manifest})
    return {"status":"PASS","checks":checks,"workflow_commit":summary["workflow_commit"],"result_status":status,"selected_candidate_id":selected_id,"validation":summary["selection"]["schemes"],"next_phase_permitted":summary["next_phase_permitted"],"manifest_files":manifest["manifest"]["checked_files"]}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase67_pd_graph_page_shape_refinement");args=parser.parse_args();print(json.dumps(verify(args.output_dir.resolve()),ensure_ascii=False,indent=2))
