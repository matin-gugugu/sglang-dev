#!/usr/bin/env python3
"""One-time Phase50 reveal and six-model blind evaluation."""
from __future__ import annotations
import argparse,csv,importlib.util,json,sys
from pathlib import Path
from typing import Any
import numpy as np
HERE=Path(__file__).resolve().parent;P41=HERE.parent/"phase41_pd_full_window_dataset";P42=HERE.parent/"phase42_pd_residual_training";P48=HERE.parent/"phase48_pd_six_model_expanded_training";P49=HERE.parent/"phase49_pd_six_model_blind_prediction_freeze"
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(P42));sys.path.insert(0,str(P41));sys.path.insert(0,str(P49));sys.path.insert(0,str(HERE.parents[2]/"scripts"));sys.path.insert(0,str(HERE))
from common import environment_record,load_json,refresh_manifest,repo_root,utc_now,write_json  # noqa:E402
from metrics import SCORE_KEYS,metric_bundle  # noqa:E402
from model import read_csv_gz,write_csv_gz  # noqa:E402
from preflight import run_checks  # noqa:E402
from prepare_bundle import reconstruct_profile  # noqa:E402
from prepare_phase15_trace_windows import BURST_FILES,MOONCAKE_FILES,load_segment  # noqa:E402
_S=importlib.util.spec_from_file_location("phase48_contracts",P48/"contracts.py")
if _S is None or _S.loader is None:raise RuntimeError("cannot load Phase48 contracts")
_P48=importlib.util.module_from_spec(_S);_S.loader.exec_module(_P48)

def read_csv(path:Path):
    with path.open(newline="",encoding="utf-8") as source:return list(csv.DictReader(source))
def write_csv(path:Path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as output:w=csv.DictWriter(output,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def difference(saved:dict[str,str],generated:dict[str,Any])->dict:
    if set(saved)!=set(generated):return {"schema_exact":False,"identifiers_exact":False,"max_absolute_difference":float("inf")}
    ids={"profile_id","split_role","source","segment","source_split","window_id","cutoff_ms","model"};return {"schema_exact":True,"identifiers_exact":all(str(saved[n])==str(generated[n]) for n in ids),"max_absolute_difference":max([abs(float(saved[n])-float(generated[n])) for n in saved if n not in ids],default=0.0)}
def frozen_arrays(pred,key_order,method):
    by_key={(r["profile_id"],r["model"]):r for r in pred if r["method"]==method}
    if set(by_key)!=set(key_order):raise RuntimeError(f"frozen keys differ: {method}")
    c=np.asarray([[float(by_key[key][f"predicted_calls_bin_{i:02d}"]) for i in range(12)] for key in key_order]);b=np.asarray([[float(by_key[key][f"predicted_logical_bytes_bin_{i:02d}"]) for i in range(12)] for key in key_order]);return c,b
def target_arrays(targets):return (np.asarray([[float(r[f"target_calls_bin_{i:02d}"]) for i in range(12)] for r in targets]),np.asarray([[float(r[f"target_logical_bytes_bin_{i:02d}"]) for i in range(12)] for r in targets]))
def compare(dnn,h0):
    ratios={k:float(dnn[k])/max(float(h0[k]),1e-12) for k in SCORE_KEYS};return {"metric_ratios_to_h0":ratios,"composite_ratio":float(np.mean(list(ratios.values()))),"strict_four_metric_gate":all(v<1 for v in ratios.values())}
def group(indices,methods,targets):
    h=metric_bundle(methods["h0"][0][indices],methods["h0"][1][indices],targets[0][indices],targets[1][indices]);d=metric_bundle(methods["h0_plus_dnn_residual"][0][indices],methods["h0_plus_dnn_residual"][1][indices],targets[0][indices],targets[1][indices]);return {"h0":h,"h0_plus_dnn_residual":d,**compare(d,h)}
def unit_rows(keys,meta,methods,targets):
    out=[]
    for method,(c,b) in methods.items():
        for i,key in enumerate(keys):out.append({"profile_id":key[0],"model":key[1],"segment":meta[key]["segment"],"request_count_stratum":meta[key]["request_count_stratum"],"method":method,**metric_bundle(c[i:i+1],b[i:i+1],targets[0][i:i+1],targets[1][i:i+1])})
    return out
def bootstrap(rows):
    methods={m:{} for m in ("h0","h0_plus_dnn_residual")}
    for m in methods:
        for pid in {r["profile_id"] for r in rows}:methods[m][pid]=[r for r in rows if r["method"]==m and r["profile_id"]==pid]
    pids=sorted(methods["h0"]);rng=np.random.default_rng(500017);draws=20000;out={"schema_version":"phase50-profile-cluster-bootstrap-v1","seed":500017,"draws":draws,"cluster":"profile; six models stay together","difference":"H0 minus DNN; positive favors DNN"}
    for key in ("mean_profile_calls_l1","mean_profile_bytes_l1"):
        paired=np.asarray([np.mean([float(r[key]) for r in methods["h0"][pid]])-np.mean([float(r[key]) for r in methods["h0_plus_dnn_residual"][pid]]) for pid in pids]);sample=paired[rng.integers(0,len(paired),size=(draws,len(paired)))].mean(axis=1);out[key]={"observed_mean_difference":float(paired.mean()),"ci95_low":float(np.quantile(sample,.025)),"ci95_high":float(np.quantile(sample,.975)),"fraction_bootstrap_positive":float(np.mean(sample>0))}
    out["both_ci95_strictly_positive"]=all(out[k]["ci95_low"]>0 for k in ("mean_profile_calls_l1","mean_profile_bytes_l1"));return out
def run(expected:str,raw_dir:Path,output:Path)->dict:
    pre=run_checks(expected,raw_dir)
    if output.exists():raise RuntimeError(f"refuse overwrite: {output}")
    contract=load_json(HERE/"experiment.json");p41=load_json(P41/"experiment.json");fc=load_json(P41/"feature_contract.json");models=_P48.load_models();saved=read_csv_gz(repo_root()/"experiment-results/phase49_pd_six_model_blind_prediction_freeze/dataset/pd_six_model_blind_target_free_features.csv.gz");pred=read_csv_gz(repo_root()/"experiment-results/phase49_pd_six_model_blind_prediction_freeze/predictions/pd_six_model_blind_frozen_predictions.csv.gz");selection=read_csv(P49/"selection/fresh_blind_windows.csv");saved_by_key={(r["profile_id"],r["model"]):r for r in saved};files={segment:raw_dir.expanduser().resolve()/name for name,(segment,_split) in {**BURST_FILES,**MOONCAKE_FILES}.items()};arrays={s:load_segment(files[s]) for s in contract["blind_contract"]["segments"]}
    reconstructed=[];audit=[];total=0
    for row in selection:
        profile,requests=reconstruct_profile({**row,"split_role":row["role"]},arrays);features,targets=_P48.six_model_example_rows(profile=profile,requests=None,phase41_contract=p41,feature_contract=fc,models=models)
        if targets:raise RuntimeError("target opened before reconstruction gate")
        for feature in features:
            key=(feature["profile_id"],feature["model"]);d=difference(saved_by_key[key],feature);d.update({"profile_id":key[0],"model":key[1]});audit.append(d)
        reconstructed.append((profile,requests));total+=len(requests)
    tol=contract["blind_contract"]["feature_reconstruction_tolerance_lt"];gate=len(audit)==1800 and total==118985 and all(r["schema_exact"] and r["identifiers_exact"] and r["max_absolute_difference"]<tol for r in audit)
    if not gate:raise RuntimeError({"reconstruction_gate":gate,"rows":len(audit),"requests":total,"sample":audit[:5]})
    targets=[]
    for profile,requests in reconstructed:
        _,rows=_P48.six_model_example_rows(profile=profile,requests=[tuple(pair) for pair in requests],phase41_contract=p41,feature_contract=fc,models=models);targets.extend(rows)
    keys=[(r["profile_id"],r["model"]) for r in targets];target=target_arrays(targets);methods={m:frozen_arrays(pred,keys,m) for m in ("h0","h0_plus_dnn_residual")};overall=group(list(range(1800)),methods,target);selection_by_id={r["profile_id"]:r for r in selection};meta={key:selection_by_id[key[0]] for key in keys}
    model_metrics={};model_gate=True
    for model in [r["model_id"] for r in models]:
        indices=[i for i,key in enumerate(keys) if key[1]==model];v=group(indices,methods,target);model_metrics[model]=v;model_gate&=v["strict_four_metric_gate"]
    segments={};segment_gate=True
    for segment in contract["blind_contract"]["segments"]:
        indices=[i for i,key in enumerate(keys) if meta[key]["segment"]==segment];v=group(indices,methods,target);rat=v["metric_ratios_to_h0"];g=v["composite_ratio"]<1 and rat["calls_histogram_wape"]<=1.05 and rat["bytes_histogram_wape"]<=1.05;segments[segment]={**v,"gate":g};segment_gate&=g
    model_segment={}
    for model in [r["model_id"] for r in models]:
        model_segment[model]={segment:group([i for i,key in enumerate(keys) if key[1]==model and meta[key]["segment"]==segment],methods,target) for segment in contract["blind_contract"]["segments"]}
    strata={str(s):group([i for i,key in enumerate(keys) if int(meta[key]["request_count_stratum"])==s],methods,target) for s in range(10)};overall_gate=overall["strict_four_metric_gate"];confirmed=bool(overall_gate and model_gate and segment_gate);outcome="CONFIRMS_SIX_MODEL_H0_PROTECTED_IMPROVEMENT" if confirmed else "DOES_NOT_CONFIRM";per_unit=unit_rows(keys,meta,methods,target);boot=bootstrap(per_unit)
    output.mkdir(parents=True);write_csv_gz(output/"labels/pd_six_model_blind_hfull_targets.csv.gz",targets);write_csv(output/"analysis/aggregate_metrics.csv",[{"method":"h0",**overall["h0"],"composite_ratio_to_h0":1.0,"scientific_outcome":"BASELINE"},{"method":"h0_plus_dnn_residual",**overall["h0_plus_dnn_residual"],"composite_ratio_to_h0":overall["composite_ratio"],"scientific_outcome":outcome}]);write_csv_gz(output/"analysis/per_unit_metrics.csv.gz",per_unit);write_json(output/"analysis/model_metrics.json",model_metrics);write_json(output/"analysis/segment_metrics.json",segments);write_json(output/"analysis/model_segment_metrics.json",model_segment);write_json(output/"analysis/request_count_stratum_metrics.json",strata);write_json(output/"analysis/paired_profile_cluster_bootstrap.json",boot);write_json(output/"audit/input_freeze.json",pre);write_json(output/"audit/target_generation.json",{"workflow_commit":expected,"prediction_parent_result_commit":"1b9227753f941cf9c790af69bf0acb7cf8bc3796","profiles":300,"models":6,"target_rows":1800,"complete_requests_used_outside_git":total,"teacher_request_model_replays":total*6,"complete_request_rows_committed":0,"reconstruction_gate_passed_before_target_access":gate,"reconstruction":audit});write_json(output/"audit/environment.json",{**environment_record(),"numpy":np.__version__,"gpu_used":False,"network_used":False,"training_used":False,"checkpoint_loaded":False,"prediction_recomputed":False,"raw_mutated":False})
    summary={"schema_version":"phase50-pd-six-model-blind-evaluation-result-v1","status":"PASS","workflow_commit":expected,"completed_at_utc":utc_now(),"counts":{"blind_profiles":300,"models":6,"blind_units":1800,"blind_complete_requests":total,"teacher_request_model_replays":total*6,"target_rows":1800,"frozen_prediction_rows":3600,"per_unit_metric_rows":3600,"complete_request_rows_in_git":0},"gates":{"overall_strict_four_metrics":overall_gate,"all_models_strict_four_metrics":model_gate,"all_segments":segment_gate,"confirmed":confirmed},"scientific_outcome":outcome,"blind_metrics":overall,"models":model_metrics,"segments":segments,"paired_profile_cluster_bootstrap":boot,"proved":"one-time target-isolated 300-profile x six-known-model pure-PD fresh blind evaluation after R49 freeze","not_proved":"unseen-model or Mooncake generalization, physical time, placement, latency or online scheduling"};write_json(output/"summary.json",summary);(output/"README.md").write_text(f"# Phase50：六模型纯PD fresh blind评估\n\n状态：PASS；科学结论：`{outcome}`。R49冻结后才为300窗口×6模型生成1800行Hfull。overall gate={overall_gate}，六模型逐一={model_gate}，三segment={segment_gate}，composite ratio={overall['composite_ratio']:.6f}。无训练、checkpoint加载或预测重算。\n",encoding="utf-8");(output/"logs").mkdir();(output/"logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\nprofiles=300 models=6 requests={total} targets=1800 frozen_predictions=3600\noutcome={outcome} overall={overall_gate} models={model_gate} segments={segment_gate} composite={overall['composite_ratio']:.12f}\ngpu=false training=false checkpoint=false prediction_recompute=false raw_committed=false\n",encoding="utf-8");(output/"DONE").write_text("PASS\n",encoding="utf-8");refresh_manifest(output);return summary
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--expected-workflow-commit",required=True);p.add_argument("--raw-dir",type=Path,required=True);p.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase50_pd_six_model_blind_evaluation");a=p.parse_args();print(json.dumps(run(a.expected_workflow_commit,a.raw_dir,a.output_dir.resolve()),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
