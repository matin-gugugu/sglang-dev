#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;P42=HERE.parent/"phase42_pd_residual_training";sys.path.insert(0,str(P42));sys.path.insert(0,str(HERE.parent))
from common import load_json,repo_root,verify_result_manifest  # noqa:E402
from model import read_csv_gz  # noqa:E402
def verify(output:Path)->dict:
    manifest=verify_result_manifest(output);summary=load_json(output/"summary.json");targets=read_csv_gz(output/"labels/pd_six_model_blind_hfull_targets.csv.gz");units=read_csv_gz(output/"analysis/per_unit_metrics.csv.gz");keys={(r["profile_id"],r["model"]) for r in targets};checks={"manifest":manifest["ok"],"status":summary.get("status")=="PASS","done":(output/"DONE").read_text().strip()=="PASS","targets_1800":len(targets)==1800,"unit_keys_1800":len(keys)==1800,"profiles_300":len({r["profile_id"] for r in targets})==300,"models_6":len({r["model"] for r in targets})==6,"per_unit_metrics_3600":len(units)==3600,"methods_exact":{r["method"] for r in units}=={"h0","h0_plus_dnn_residual"},"no_complete_requests":summary["counts"]["complete_request_rows_in_git"]==0,"no_training":load_json(output/"audit/environment.json")["training_used"] is False,"confirmation_consistent":summary["gates"]["confirmed"]==all([summary["gates"]["overall_strict_four_metrics"],summary["gates"]["all_models_strict_four_metrics"],summary["gates"]["all_segments"]])}
    if not all(checks.values()):raise RuntimeError(checks)
    return {"status":"PASS","checks":checks,"manifest_files":manifest["manifest"]["checked_files"],"scientific_outcome":summary["scientific_outcome"]}
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase50_pd_six_model_blind_evaluation");a=p.parse_args();print(json.dumps(verify(a.output_dir.resolve()),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
