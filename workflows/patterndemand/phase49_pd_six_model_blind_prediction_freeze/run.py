#!/usr/bin/env python3
"""Freeze Phase49 target-free six-model features and R48 predictions."""
from __future__ import annotations
import argparse,importlib.util,json,sys
from pathlib import Path
from typing import Any
import numpy as np
HERE=Path(__file__).resolve().parent; P41=HERE.parent/"phase41_pd_full_window_dataset";P42=HERE.parent/"phase42_pd_residual_training";P48=HERE.parent/"phase48_pd_six_model_expanded_training"
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(P42));sys.path.insert(0,str(P41));sys.path.insert(0,str(HERE.parents[2]/"scripts"));sys.path.insert(0,str(HERE))
from common import environment_record,load_json,refresh_manifest,repo_root,utc_now,write_json  # noqa:E402
from model import decode_histograms,encode_histograms,histogram_arrays,model_from_json,predict_histograms,read_json_gz,write_csv_gz  # noqa:E402
from preflight import read_csv,run_checks  # noqa:E402
from prepare_bundle import reconstruct_profile  # noqa:E402
from prepare_phase15_trace_windows import BURST_FILES,MOONCAKE_FILES,load_segment  # noqa:E402
_S=importlib.util.spec_from_file_location("phase48_contracts",P48/"contracts.py")
if _S is None or _S.loader is None:raise RuntimeError("cannot load Phase48 contracts")
_P48=importlib.util.module_from_spec(_S);_S.loader.exec_module(_P48)
IDS=("profile_id","split_role","source","segment","source_split","window_id","cutoff_ms","model")

def shrink(rows:list[dict[str,Any]],calls:np.ndarray,bytes_:np.ndarray,alphas:dict[str,float])->tuple[np.ndarray,np.ndarray]:
    h0c,h0b=histogram_arrays(rows,"h0");h0=encode_histograms(h0c,h0b);raw=encode_histograms(calls,bytes_);a=np.asarray([alphas[row["model"]] for row in rows])[:,None];return decode_histograms(h0+a*(raw-h0))
def prediction_rows(rows,calls,bytes_,method):
    output=[]
    for row,cv,bv in zip(rows,calls,bytes_):
        value={name:row[name] for name in IDS};value["method"]=method;value["predicted_total_calls_per_1000"]=float(cv.sum());value["predicted_total_logical_bytes_per_1000"]=float(bv.sum())
        for i in range(12):value[f"predicted_calls_bin_{i:02d}"]=float(cv[i])
        for i in range(12):value[f"predicted_logical_bytes_bin_{i:02d}"]=float(bv[i])
        output.append(value)
    return output
def run(expected:str,raw_dir:Path,output:Path)->dict:
    pre=run_checks(expected,raw_dir);contract=load_json(HERE/"experiment.json");p41=load_json(P41/"experiment.json");fc=load_json(P41/"feature_contract.json");models=_P48.load_models()
    if output.exists():raise RuntimeError(f"refuse overwrite: {output}")
    selected=read_csv(repo_root()/contract["selection_contract"]["path"]);files={segment:raw_dir.expanduser().resolve()/name for name,(segment,_split) in {**BURST_FILES,**MOONCAKE_FILES}.items()};arrays={segment:load_segment(files[segment]) for segment in contract["selection_contract"]["segments"]}
    features=[];profiles=[];requests_total=0
    for row in selected:
        profile,requests=reconstruct_profile({**row,"split_role":row["role"]},arrays);feature_rows,target_rows=_P48.six_model_example_rows(profile=profile,requests=None,phase41_contract=p41,feature_contract=fc,models=models)
        if target_rows or any(name.startswith(("target_","residual_")) for feature in feature_rows for name in feature):raise RuntimeError("target leakage")
        features.extend(feature_rows);profiles.append(profile);requests_total+=len(requests)
    if (len(features),len(profiles),requests_total)!=(1800,300,118985):raise RuntimeError("blind counts mismatch")
    cp=read_json_gz(repo_root()/"experiment-results/phase48_pd_six_model_expanded_training/checkpoints/pd_six_model_h0_protected_dnn.json.gz");ensemble=[model_from_json(value) for value in cp["models"]];rawc,rawb=predict_histograms(features,cp["transform"],ensemble);dnnc,dnnb=shrink(features,rawc,rawb,cp["alpha_by_model"]);h0c,h0b=histogram_arrays(features,"h0");predictions=prediction_rows(features,h0c,h0b,"h0")+prediction_rows(features,dnnc,dnnb,"h0_plus_dnn_residual")
    output.mkdir(parents=True);write_csv_gz(output/"dataset/pd_six_model_blind_target_free_features.csv.gz",features);write_csv_gz(output/"profiles/fresh_blind_lowdim_profiles.csv.gz",profiles);write_csv_gz(output/"predictions/pd_six_model_blind_frozen_predictions.csv.gz",predictions)
    write_json(output/"audit/input_freeze.json",pre);write_json(output/"audit/prediction_freeze.json",{"profiles":300,"models":6,"feature_rows":1800,"prediction_rows":3600,"complete_requests_reconstructed_outside_git":requests_total,"complete_request_rows_committed":0,"target_rows":0,"training_used":False,"checkpoint_changed":False,"candidate_id":cp["selected_candidate"]["candidate_id"],"alpha_by_model":cp["alpha_by_model"],"epochs":cp["selected_epochs"]});write_json(output/"audit/environment.json",{**environment_record(),"numpy":np.__version__,"gpu_used":False,"network_used":False,"training_used":False,"targets_accessed":False,"raw_mutated":False})
    summary={"schema_version":"phase49-pd-six-model-blind-freeze-result-v1","status":"PASS","workflow_commit":expected,"completed_at_utc":utc_now(),"counts":{"blind_profiles":300,"models":6,"feature_rows":1800,"blind_complete_requests_reconstructed_outside_git":requests_total,"frozen_prediction_rows":3600,"target_rows":0,"complete_request_rows_in_git":0},"predictor":{"candidate_id":cp["selected_candidate"]["candidate_id"],"feature_mode":cp["selected_candidate"]["feature_mode"],"alpha_by_model":cp["alpha_by_model"],"epochs":cp["selected_epochs"],"ensemble_seeds":cp["ensemble_seeds"]},"blind_state":"target-free six-model features and H0/H0+DNN predictions frozen; no Hfull generated or accessed","next":"only after R49 formal integration may Phase50 reveal Hfull once","proved":"target-isolated fresh six-model blind cohort and exact R48 prediction freeze","not_proved":"blind accuracy/improvement, unseen models, physical time, placement, latency or online scheduling"}
    write_json(output/"summary.json",summary);(output/"README.md").write_text(f"# Phase49：六模型fresh blind预测冻结\n\n状态：PASS。冻结300个全新窗口、1800条六模型特征、3600条H0/H0+DNN预测；共重建{requests_total}个请求，但Hfull/target为0，完整请求未进入Git。只有R49合入后Phase50才可一次性揭示标签。\n",encoding="utf-8");(output/"logs").mkdir();(output/"logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\nprofiles=300 models=6 requests={requests_total} features=1800 predictions=3600 targets=0\ngpu=false training=false target_access=false raw_committed=false\n",encoding="utf-8");(output/"DONE").write_text("PASS\n",encoding="utf-8");refresh_manifest(output);return summary
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--expected-workflow-commit",required=True);p.add_argument("--raw-dir",type=Path,required=True);p.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase49_pd_six_model_blind_prediction_freeze");a=p.parse_args();print(json.dumps(run(a.expected_workflow_commit,a.raw_dir,a.output_dir.resolve()),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
