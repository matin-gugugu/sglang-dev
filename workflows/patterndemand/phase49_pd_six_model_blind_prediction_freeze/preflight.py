#!/usr/bin/env python3
"""Phase49 selection, R48 checkpoint and protected-raw preflight."""
from __future__ import annotations
import argparse,csv,importlib.util,json,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent; P41=HERE.parent/"phase41_pd_full_window_dataset"; P42=HERE.parent/"phase42_pd_residual_training"
sys.path.insert(0,str(HERE.parent)); sys.path.insert(0,str(P42)); sys.path.insert(0,str(P41)); sys.path.insert(0,str(HERE))
from build_selection import HISTORY_MS,PRIOR_SELECTIONS,select  # noqa:E402
from common import load_json,repo_root,require_clean_before_run,require_expected_head,verify_pinned_inputs  # noqa:E402
from model import read_json_gz  # noqa:E402
from prepare_bundle import raw_source_audit  # noqa:E402

def read_csv(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as source:return list(csv.DictReader(source))

def selection_audit(contract:dict)->dict:
    frozen=read_csv(repo_root()/contract["selection_contract"]["path"]); reproduced=select(repo_root()); prior={segment:[] for segment in contract["selection_contract"]["segments"]}
    for relative in PRIOR_SELECTIONS:
        for row in read_csv(repo_root()/relative):
            if row["segment"] in prior:prior[row["segment"]].append(int(row["cutoff_ms"]))
    overlaps=[(left["profile_id"],right["profile_id"]) for i,left in enumerate(frozen) for right in frozen[i+1:] if left["segment"]==right["segment"] and abs(int(left["cutoff_ms"])-int(right["cutoff_ms"]))<HISTORY_MS]
    embargo=[row["profile_id"] for row in frozen if any(abs(int(row["cutoff_ms"])-old)<HISTORY_MS for old in prior[row["segment"]])]; segments=Counter(row["segment"] for row in frozen); strata=Counter((row["segment"],row["request_count_stratum"]) for row in frozen)
    checks={"reproduced_exactly":[row["window_id"] for row in frozen]==[str(row["window_id"]) for row in reproduced],"profiles_300":len(frozen)==300,"requests_exact":sum(int(row["history_count"]) for row in frozen)==118985,"segments_exact":all(segments[s]==100 for s in prior),"strata_exact":all(strata[(s,str(i))]==10 for s in prior for i in range(10)),"pairwise_nonoverlap":not overlaps,"historical_embargo":not embargo,"target_free_columns":not any(name.startswith(("future_","target_","residual_")) for name in frozen[0])}
    if not all(checks.values()):raise RuntimeError({"selection_checks":checks,"overlaps":overlaps[:10],"embargo":embargo[:10]})
    return {"checks":checks,"segments":dict(segments)}

def checkpoint_audit(contract:dict)->dict:
    cp=read_json_gz(repo_root()/"experiment-results/phase48_pd_six_model_expanded_training/checkpoints/pd_six_model_h0_protected_dnn.json.gz"); summary=load_json(repo_root()/"experiment-results/phase48_pd_six_model_expanded_training/summary.json"); expected=contract["predictor_contract"]
    checks={"workflow_commit":cp["workflow_commit"]=="a5bce043c3b529193178e743198cd0ac469aba81","candidate":cp["selected_candidate"]["candidate_id"]==expected["candidate_id"],"feature_mode":cp["selected_candidate"]["feature_mode"]==expected["feature_mode"],"epochs":cp["selected_epochs"]==expected["epochs"],"alphas":cp["alpha_by_model"]==expected["alpha_by_model"],"models":len(cp["models"])==3,"blind_targets_not_accessed":cp["phase45_or_phase46_targets_accessed"] is False,"accepted":summary["gates"]["model_accepted"] is True and summary["gates"]["phase49_permitted"] is True}
    if not all(checks.values()):raise RuntimeError({"checkpoint_checks":checks})
    return checks

def run_checks(expected:str,raw_dir:Path)->dict:
    contract=load_json(HERE/"experiment.json"); p41=load_json(P41/"experiment.json"); head=require_expected_head(expected); require_clean_before_run(("data/",))
    return {"status":"PASS","workflow_commit":head,"pinned_inputs":verify_pinned_inputs(contract),"selection_audit":selection_audit(contract),"checkpoint_audit":checkpoint_audit(contract),"raw_source_audit":raw_source_audit(p41,raw_dir.expanduser().resolve()),"gpu_used":False,"training_used":False,"targets_accessed":False,"raw_read_only":True}

def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--expected-workflow-commit",required=True);p.add_argument("--raw-dir",type=Path,required=True);a=p.parse_args();print(json.dumps(run_checks(a.expected_workflow_commit,a.raw_dir),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
