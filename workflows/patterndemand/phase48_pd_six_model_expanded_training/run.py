#!/usr/bin/env python3
"""Generate six-model Phase48 labels and train a shared H0-protected residual DNN."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
P41 = HERE.parent / "phase41_pd_full_window_dataset"; P42 = HERE.parent / "phase42_pd_residual_training"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(P42)); sys.path.insert(0, str(P41)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from common import environment_record, load_json, refresh_manifest, repo_root, utc_now, write_json  # noqa: E402
from metrics import SCORE_KEYS, compare_to_h0, metric_bundle  # noqa: E402
from model import decode_histograms, encode_histograms, fit_model, fit_transform, histogram_arrays, model_to_json, predict_histograms, transform_inputs, transform_targets, write_csv_gz, write_json_gz  # noqa: E402
from prepare_bundle import reconstruct_profile  # noqa: E402
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("phase48_contracts", HERE / "contracts.py")
if _SPEC is None or _SPEC.loader is None: raise RuntimeError("cannot load Phase48 contracts")
_P48 = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_P48)
_PREFLIGHT_SPEC = importlib.util.spec_from_file_location("phase48_preflight", HERE / "preflight.py")
if _PREFLIGHT_SPEC is None or _PREFLIGHT_SPEC.loader is None: raise RuntimeError("cannot load Phase48 preflight")
_PREFLIGHT = importlib.util.module_from_spec(_PREFLIGHT_SPEC); _PREFLIGHT_SPEC.loader.exec_module(_PREFLIGHT)


ARRIVAL_TOKENS = ("_rps", "interarrival", "peak_to_mean", "fano")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def transform_for_mode(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    transform = fit_transform(rows)
    if mode == "full_target_free": return transform
    if mode != "fixed_draining_causal": raise ValueError(mode)
    keep = [i for i, name in enumerate(transform["input_names"]) if not any(token in name for token in ARRIVAL_TOKENS)]
    return {**transform, "input_names": [transform["input_names"][i] for i in keep], "input_mean": [transform["input_mean"][i] for i in keep], "input_scale": [transform["input_scale"][i] for i in keep]}


def fold_map(rows: list[dict[str, Any]], selection: dict[str, dict[str, str]], folds: int = 4) -> dict[str, int]:
    groups: dict[tuple[str, str], list[str]] = {}
    for profile_id in sorted({row["profile_id"] for row in rows}):
        row = next(value for value in rows if value["profile_id"] == profile_id); selected = selection[profile_id]
        groups.setdefault((row["segment"], selected["request_count_stratum"]), []).append(profile_id)
    result = {}
    for key, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda value: hashlib.sha256(f"phase48-fold:{key}:{value}".encode()).hexdigest())
        for index, profile_id in enumerate(ordered): result[profile_id] = index % folds
    return result


def strict(comparison: dict[str, Any]) -> bool:
    return all(float(comparison["metric_ratios_to_h0"][key]) < 1.0 for key in SCORE_KEYS)


def shrink(rows: list[dict[str, Any]], calls: np.ndarray, logical_bytes: np.ndarray, alpha_by_model: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    h0_calls, h0_bytes = histogram_arrays(rows, "h0"); h0 = encode_histograms(h0_calls, h0_bytes); raw = encode_histograms(calls, logical_bytes)
    alpha = np.asarray([float(alpha_by_model[row["model"]]) for row in rows], dtype=np.float64)[:, None]
    return decode_histograms(h0 + alpha * (raw - h0))


def subset_metrics(rows: list[dict[str, Any]], calls: np.ndarray, logical_bytes: np.ndarray, indices: list[int]) -> tuple[dict, dict, dict]:
    h0c, h0b = histogram_arrays([rows[i] for i in indices], "h0"); tc, tb = histogram_arrays([rows[i] for i in indices], "target")
    h0 = metric_bundle(h0c, h0b, tc, tb); dnn = metric_bundle(calls[indices], logical_bytes[indices], tc, tb)
    return h0, dnn, compare_to_h0(dnn, h0)


def prediction_rows(rows: list[dict[str, Any]], calls: np.ndarray, logical_bytes: np.ndarray, method: str) -> list[dict[str, Any]]:
    output=[]
    for row, cv, bv in zip(rows, calls, logical_bytes):
        value={name: row[name] for name in ("profile_id","split_role","source","segment","source_split","window_id","cutoff_ms","model")}; value["method"]=method
        for i in range(12): value[f"predicted_calls_bin_{i:02d}"]=float(cv[i])
        for i in range(12): value[f"predicted_logical_bytes_bin_{i:02d}"]=float(bv[i])
        output.append(value)
    return output


def run(expected: str, raw_dir: Path, output: Path) -> dict[str, Any]:
    preflight = _PREFLIGHT.run_checks(expected, raw_dir); contract = load_json(HERE/"experiment.json"); p41 = load_json(P41/"experiment.json"); feature_contract = load_json(P41/"feature_contract.json"); models = _P48.load_models()
    if output.exists(): raise RuntimeError(f"refuse overwrite: {output}")
    selected = _PREFLIGHT.read_selection(repo_root()/contract["dataset_contract"]["selection_path"]); selection = {row["profile_id"]: row for row in selected}
    file_by_segment = {segment: raw_dir.expanduser().resolve()/name for name,(segment,_split) in {**BURST_FILES,**MOONCAKE_FILES}.items()}
    arrays = {segment: load_segment(file_by_segment[segment]) for segment in ("burstgpt_1","burstgpt_2","burstgpt_3")}
    examples=[]; targets=[]; profiles=[]; unique_requests=0
    for selected_row in selected:
        profile, requests = reconstruct_profile({**selected_row, "split_role": selected_row["role"]}, arrays)
        rows, target_rows = _P48.six_model_example_rows(profile=profile, requests=[tuple(pair) for pair in requests], phase41_contract=p41, feature_contract=feature_contract, models=models)
        examples.extend(rows); targets.extend(target_rows); profiles.append(profile); unique_requests += len(requests)
    if (len(examples),len(targets),len(profiles),unique_requests)!=(7200,7200,1200,486242): raise RuntimeError("dataset counts mismatch")
    train=[row for row in examples if row["split_role"]=="expanded_train"]; validation=[row for row in examples if row["split_role"]=="expanded_validation"]
    if (len(train),len(validation))!=(5760,1440): raise RuntimeError("split mismatch")
    folds=fold_map(train,selection); alpha_grid=[float(v) for v in contract["predictor_contract"]["alpha_grid"]]; model_ids=[row["model_id"] for row in models]
    candidate_rows=[]; candidates=[]
    target_train_calls,target_train_bytes=histogram_arrays(train,"target")
    h0_train_calls,h0_train_bytes=histogram_arrays(train,"h0"); h0_train_metrics=metric_bundle(h0_train_calls,h0_train_bytes,target_train_calls,target_train_bytes)
    for candidate_index,config in enumerate(contract["predictor_contract"]["candidate_grid"]):
        raw_calls=np.zeros((len(train),12)); raw_bytes=np.zeros((len(train),12)); epochs=[]
        for fold in range(4):
            fit_idx=[i for i,row in enumerate(train) if folds[row["profile_id"]]!=fold]; val_idx=[i for i,row in enumerate(train) if folds[row["profile_id"]]==fold]
            fit_rows=[train[i] for i in fit_idx]; val_rows=[train[i] for i in val_idx]; transform=transform_for_mode(fit_rows,config["feature_mode"])
            model,audit=fit_model(transform_inputs(fit_rows,transform),transform_targets(fit_rows,transform),config,480000+candidate_index*1000+fold,validation=(transform_inputs(val_rows,transform),transform_targets(val_rows,transform)))
            calls,bytes_=predict_histograms(val_rows,transform,[model]); raw_calls[val_idx]=calls; raw_bytes[val_idx]=bytes_; epochs.append(int(audit["best_epoch"]))
        alpha_by_model={}; model_oof={}; all_models_gate=True
        for model_id in model_ids:
            indices=[i for i,row in enumerate(train) if row["model"]==model_id]; h0m=metric_bundle(h0_train_calls[indices],h0_train_bytes[indices],target_train_calls[indices],target_train_bytes[indices]); choices=[]
            for alpha in alpha_grid:
                calls,bytes_=shrink([train[i] for i in indices],raw_calls[indices],raw_bytes[indices],{model_id:alpha}); dnnm=metric_bundle(calls,bytes_,target_train_calls[indices],target_train_bytes[indices]); comp=compare_to_h0(dnnm,h0m); gate=strict(comp)
                candidate_rows.append({"candidate_id":config["candidate_id"],"model":model_id,"alpha":alpha,"strict_gate":gate,"composite_ratio":comp["composite_ratio"],**{f"ratio_{key}":comp["metric_ratios_to_h0"][key] for key in SCORE_KEYS}})
                choices.append((not gate,float(comp["composite_ratio"]),alpha,comp))
            choices.sort(key=lambda value:value[:3]); rejected,_,alpha,comp=choices[0]; alpha_by_model[model_id]=alpha; model_oof[model_id]={"alpha":alpha,**comp,"gate":not rejected}; all_models_gate &= not rejected
        oof_calls,oof_bytes=shrink(train,raw_calls,raw_bytes,alpha_by_model); overall=compare_to_h0(metric_bundle(oof_calls,oof_bytes,target_train_calls,target_train_bytes),h0_train_metrics); overall_gate=strict(overall); admissible=bool(overall_gate and all_models_gate)
        candidates.append({"sort":(not admissible,float(overall["composite_ratio"]),config["candidate_id"]),"config":config,"epochs":epochs,"alpha_by_model":alpha_by_model,"oof_overall":overall,"oof_models":model_oof,"admissible":admissible})
    candidates.sort(key=lambda row:row["sort"]); selected_candidate=candidates[0]; selected_config=selected_candidate["config"]; alpha_by_model=selected_candidate["alpha_by_model"]; oof_accepted=selected_candidate["admissible"]
    selected_epochs=int(np.clip(round(statistics.median(selected_candidate["epochs"])),100,int(selected_config["max_epochs"])))
    transform=transform_for_mode(train,selected_config["feature_mode"]); x=transform_inputs(train,transform); y=transform_targets(train,transform); final_models=[]; final_audits=[]
    for seed in contract["predictor_contract"]["ensemble_seeds"]:
        model,audit=fit_model(x,y,selected_config,int(seed),fixed_epochs=selected_epochs); final_models.append(model); final_audits.append({"seed":seed,**audit})
    raw_val_calls,raw_val_bytes=predict_histograms(validation,transform,final_models); dnn_calls,dnn_bytes=shrink(validation,raw_val_calls,raw_val_bytes,alpha_by_model)
    target_val_calls,target_val_bytes=histogram_arrays(validation,"target"); h0_val_calls,h0_val_bytes=histogram_arrays(validation,"h0")
    h0_val=metric_bundle(h0_val_calls,h0_val_bytes,target_val_calls,target_val_bytes); dnn_val=metric_bundle(dnn_calls,dnn_bytes,target_val_calls,target_val_bytes); overall=compare_to_h0(dnn_val,h0_val); overall_gate=strict(overall)
    model_audits={}; model_gate=True
    for model_id in model_ids:
        indices=[i for i,row in enumerate(validation) if row["model"]==model_id]; h0m,dnnm,comp=subset_metrics(validation,dnn_calls,dnn_bytes,indices); gate=strict(comp); model_audits[model_id]={"h0":h0m,"h0_plus_dnn":dnnm,**comp,"alpha":alpha_by_model[model_id],"gate":gate}; model_gate &= gate
    segment_audits={}; segment_gate=True
    for segment in ("burstgpt_1","burstgpt_2","burstgpt_3"):
        indices=[i for i,row in enumerate(validation) if row["segment"]==segment]; h0m,dnnm,comp=subset_metrics(validation,dnn_calls,dnn_bytes,indices); gate=float(comp["composite_ratio"])<1.0 and float(comp["metric_ratios_to_h0"]["calls_histogram_wape"])<=1.05 and float(comp["metric_ratios_to_h0"]["bytes_histogram_wape"])<=1.05
        segment_audits[segment]={"h0":h0m,"h0_plus_dnn":dnnm,**comp,"gate":gate}; segment_gate &= gate
    model_accepted=bool(oof_accepted and overall_gate and model_gate and segment_gate)
    output.mkdir(parents=True)
    write_csv_gz(output/"dataset/pd_six_model_h0_residual_examples.csv.gz",examples); write_csv_gz(output/"dataset/pd_six_model_hfull_targets.csv.gz",targets); write_csv_gz(output/"profiles/expanded_lowdim_profiles.csv.gz",profiles)
    write_csv(output/"analysis/candidate_model_alpha_metrics.csv",candidate_rows); write_json(output/"analysis/oof_selection.json",{key:value for key,value in selected_candidate.items() if key!="sort"}); write_json(output/"analysis/model_validation.json",model_audits); write_json(output/"analysis/segment_validation.json",segment_audits)
    write_csv(output/"analysis/development_validation_metrics.csv",[{"method":"h0",**h0_val,"composite_ratio_to_h0":1.0,"hard_gate":True},{"method":"h0_plus_dnn_residual",**dnn_val,"composite_ratio_to_h0":overall["composite_ratio"],"hard_gate":overall_gate}])
    write_csv_gz(output/"predictions/development_validation_predictions.csv.gz",prediction_rows(validation,h0_val_calls,h0_val_bytes,"h0")+prediction_rows(validation,dnn_calls,dnn_bytes,"h0_plus_dnn_residual"))
    checkpoint={"schema_version":"phase48-pd-six-model-checkpoint-v1","workflow_commit":expected,"selected_candidate":selected_config,"alpha_by_model":alpha_by_model,"selected_epochs":selected_epochs,"transform":transform,"ensemble_seeds":contract["predictor_contract"]["ensemble_seeds"],"models":[model_to_json(model) for model in final_models],"training_profile_ids":sorted({row["profile_id"] for row in train}),"phase45_or_phase46_targets_accessed":False}
    write_json_gz(output/"checkpoints/pd_six_model_h0_protected_dnn.json.gz",checkpoint)
    write_json(output/"audit/input_freeze.json",preflight); write_json(output/"audit/dataset_generation.json",{"profiles":1200,"models":6,"example_rows":7200,"unique_complete_requests":unique_requests,"teacher_request_model_replays":unique_requests*6,"complete_request_rows_in_git":0,"page_aware_teacher":True}); write_json(output/"audit/training.json",{"fold_assignment":folds,"selected_candidate_id":selected_config["candidate_id"],"alpha_by_model":alpha_by_model,"selected_epochs":selected_epochs,"oof_accepted":oof_accepted,"validation_overall_gate":overall_gate,"validation_all_models_gate":model_gate,"validation_all_segments_gate":segment_gate,"model_accepted":model_accepted,"final_models":final_audits}); write_json(output/"audit/environment.json",{**environment_record(),"numpy":np.__version__,"gpu_used":False,"network_used":False,"raw_mutated":False,"phase45_or_phase46_targets_accessed":False})
    summary={"schema_version":"phase48-pd-six-model-expanded-training-result-v1","status":"PASS","workflow_commit":expected,"completed_at_utc":utc_now(),"counts":{"profiles":1200,"models":6,"example_rows":7200,"train_rows":5760,"validation_rows":1440,"unique_complete_requests":unique_requests,"teacher_request_model_replays":unique_requests*6,"complete_request_rows_in_git":0},"selected":{"candidate_id":selected_config["candidate_id"],"feature_mode":selected_config["feature_mode"],"epochs":selected_epochs,"alpha_by_model":alpha_by_model},"gates":{"oof_accepted":oof_accepted,"validation_overall":overall_gate,"validation_all_models":model_gate,"validation_all_segments":segment_gate,"model_accepted":model_accepted,"phase49_permitted":model_accepted},"development_validation":{"h0":h0_val,"h0_plus_dnn_residual":dnn_val,**overall},"models":model_audits,"segments":segment_audits,"proved":"six-model BurstGPT development evidence for a model-structure-conditioned pure-PD H0-protected residual predictor","not_proved":"fresh blind generalization, unseen-model extrapolation, physical RDMA cost, placement, latency or online scheduling"}
    write_json(output/"summary.json",summary)
    (output/"README.md").write_text(f"# Phase48：六模型纯PD扩展训练\n\n状态：`PASS`。1200个冻结画像乘六模型生成7200条紧凑训练表；完整请求只在内存中读取，未进入Git。\n\n模型接受门：OOF={oof_accepted}，validation overall={overall_gate}，六模型逐一={model_gate}，三流量段={segment_gate}，最终model_accepted={model_accepted}。只有最终为true才允许Phase49冻结全新blind预测。\n",encoding="utf-8")
    (output/"logs").mkdir(); (output/"logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\nprofiles=1200 models=6 rows=7200 unique_requests={unique_requests}\nselected={selected_config['candidate_id']} epochs={selected_epochs} alpha_by_model={json.dumps(alpha_by_model,sort_keys=True)}\noof={oof_accepted} overall={overall_gate} models={model_gate} segments={segment_gate} accepted={model_accepted}\ngpu=false network=false raw_committed=false\n",encoding="utf-8")
    (output/"DONE").write_text("PASS\n",encoding="utf-8"); refresh_manifest(output); return summary


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit",required=True); parser.add_argument("--raw-dir",type=Path,required=True); parser.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase48_pd_six_model_expanded_training")
    args=parser.parse_args(); print(json.dumps(run(args.expected_workflow_commit,args.raw_dir,args.output_dir.resolve()),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
