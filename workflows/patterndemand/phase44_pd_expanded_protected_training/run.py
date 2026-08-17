#!/usr/bin/env python3
"""Generate Phase44 expanded labels and train an H0-protected residual DNN."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent; P41 = HERE.parent / "phase41_pd_full_window_dataset"; P42 = HERE.parent / "phase42_pd_residual_training"
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(P42)); sys.path.insert(0, str(P41)); sys.path.insert(0, str(HERE.parents[2] / "scripts")); sys.path.insert(0, str(HERE))
from common import environment_record, load_json, refresh_manifest, repo_root, utc_now, write_json  # noqa: E402
from contracts import profile_example_rows  # noqa: E402
from metrics import SCORE_KEYS, compare_to_h0, metric_bundle  # noqa: E402
from model import (decode_histograms, encode_histograms, fit_model, fit_transform, histogram_arrays, model_to_json, predict_histograms, transform_inputs, transform_targets, write_csv_gz, write_json_gz)  # noqa: E402
from preflight import read_csv, run_checks  # noqa: E402
from prepare_bundle import reconstruct_profile  # noqa: E402
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment  # noqa: E402


ARRIVAL_TOKENS = ("_rps", "interarrival", "peak_to_mean", "fano")


def transform_for_mode(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    transform = fit_transform(rows)
    if mode == "full_target_free": return transform
    if mode != "fixed_draining_causal": raise ValueError(mode)
    keep = [index for index, name in enumerate(transform["input_names"]) if not any(token in name for token in ARRIVAL_TOKENS)]
    return {**transform, "input_names": [transform["input_names"][index] for index in keep], "input_mean": [transform["input_mean"][index] for index in keep], "input_scale": [transform["input_scale"][index] for index in keep]}


def shrink_prediction(rows: list[dict[str, Any]], calls: np.ndarray, logical_bytes: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    h0_calls, h0_bytes = histogram_arrays(rows, "h0")
    h0_encoded = encode_histograms(h0_calls, h0_bytes); predicted = encode_histograms(calls, logical_bytes)
    return decode_histograms(h0_encoded + float(alpha) * (predicted - h0_encoded))


def fold_map(rows: list[dict[str, Any]], selection: dict[str, dict[str, str]], folds: int = 5) -> dict[str, int]:
    groups: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        selected = selection[row["profile_id"]]; groups.setdefault((row["segment"], selected["request_count_stratum"]), []).append(row["profile_id"])
    result = {}
    for key, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda value: hashlib.sha256(f"phase44-fold:{key}:{value}".encode()).hexdigest())
        for index, profile_id in enumerate(ordered): result[profile_id] = index % folds
    return result


def target_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]: return histogram_arrays(rows, "target")


def accepted(comparison: dict[str, Any]) -> bool: return all(float(comparison["metric_ratios_to_h0"][key]) < 1.0 for key in SCORE_KEYS)


def prediction_rows(rows: list[dict[str, Any]], calls: np.ndarray, logical_bytes: np.ndarray, method: str) -> list[dict[str, Any]]:
    output=[]
    for row, call_vector, byte_vector in zip(rows,calls,logical_bytes):
        value={name:row[name] for name in ("profile_id","split_role","source","segment","source_split","window_id","cutoff_ms","model")}; value["method"]=method
        for index in range(12): value[f"predicted_calls_bin_{index:02d}"]=float(call_vector[index])
        for index in range(12): value[f"predicted_logical_bytes_bin_{index:02d}"]=float(byte_vector[index])
        output.append(value)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def run(expected: str, raw_dir: Path, output: Path) -> dict[str, Any]:
    preflight=run_checks(expected,raw_dir); contract=load_json(HERE/"experiment.json"); phase41=load_json(P41/"experiment.json"); feature_contract=load_json(P41/"feature_contract.json")
    if output.exists(): raise RuntimeError(f"refuse overwrite: {output}")
    selected=read_csv(repo_root()/contract["selection_contract"]["path"]); selection_by_id={row["profile_id"]:row for row in selected}
    file_by_segment={segment:raw_dir.expanduser().resolve()/name for name,(segment,_split) in {**BURST_FILES,**MOONCAKE_FILES}.items()}
    arrays_by_segment={segment:load_segment(file_by_segment[segment]) for segment in contract["selection_contract"]["segments"]}
    model_contract=load_json(repo_root()/"experiment-results/phase41_pd_full_window_dataset/contracts/model_contract.json"); kv_bytes=int(model_contract["derived"]["kv_bytes_per_page"])
    examples=[]; targets=[]; profiles=[]; total_requests=0
    for row in selected:
        source_row={**row,"split_role":row["role"]}; profile,requests=reconstruct_profile(source_row,arrays_by_segment)
        example,target=profile_example_rows(profile=profile,requests=[tuple(pair) for pair in requests],contract=phase41,feature_contract=feature_contract,kv_bytes_per_page=kv_bytes)
        assert target is not None
        examples.append(example); targets.append(target); profiles.append(profile); total_requests+=len(requests)
    if len(examples)!=1200 or total_requests!=486242: raise RuntimeError({"profiles":len(examples),"requests":total_requests})
    train=[row for row in examples if row["split_role"]=="expanded_train"]; validation=[row for row in examples if row["split_role"]=="expanded_validation"]
    if (len(train),len(validation))!=(960,240): raise RuntimeError("split mismatch")
    folds=fold_map(train,selection_by_id); alpha_grid=[float(value) for value in contract["predictor_contract"]["alpha_grid"]]
    candidate_rows=[]; payloads=[]
    h0_train_calls,h0_train_bytes=histogram_arrays(train,"h0"); target_train_calls,target_train_bytes=target_arrays(train)
    h0_train_metrics=metric_bundle(h0_train_calls,h0_train_bytes,target_train_calls,target_train_bytes)
    for candidate_index,config in enumerate(contract["predictor_contract"]["candidate_grid"]):
        raw_calls=np.zeros((len(train),12)); raw_bytes=np.zeros((len(train),12)); epochs=[]
        for fold in range(5):
            fit_indices=[i for i,row in enumerate(train) if folds[row["profile_id"]]!=fold]; val_indices=[i for i,row in enumerate(train) if folds[row["profile_id"]]==fold]
            fit_rows=[train[i] for i in fit_indices]; val_rows=[train[i] for i in val_indices]; transform=transform_for_mode(fit_rows,config["feature_mode"])
            model,audit=fit_model(transform_inputs(fit_rows,transform),transform_targets(fit_rows,transform),config,440000+candidate_index*1000+fold,validation=(transform_inputs(val_rows,transform),transform_targets(val_rows,transform)))
            calls,logical_bytes=predict_histograms(val_rows,transform,[model]); raw_calls[val_indices]=calls; raw_bytes[val_indices]=logical_bytes; epochs.append(int(audit["best_epoch"]))
        for alpha in alpha_grid:
            calls,logical_bytes=shrink_prediction(train,raw_calls,raw_bytes,alpha); metrics=metric_bundle(calls,logical_bytes,target_train_calls,target_train_bytes); comparison=compare_to_h0(metrics,h0_train_metrics); gate=accepted(comparison)
            candidate_rows.append({"candidate_id":config["candidate_id"],"feature_mode":config["feature_mode"],"alpha":alpha,"median_best_epoch":int(statistics.median(epochs)),**{f"dnn_{key}":value for key,value in metrics.items()},**{f"h0_{key}":value for key,value in h0_train_metrics.items()},"composite_ratio":comparison["composite_ratio"],"strict_oof_gate":gate})
            payloads.append((not gate,float(comparison["composite_ratio"]),str(config["candidate_id"]),alpha,config,epochs))
    payloads.sort(key=lambda value:value[:4]); _not_gate,_score,_name,selected_alpha,selected_config,selected_fold_epochs=payloads[0]; oof_accepted=not _not_gate
    selected_epochs=int(np.clip(round(statistics.median(selected_fold_epochs)),100,int(selected_config["max_epochs"])))
    final_transform=transform_for_mode(train,selected_config["feature_mode"]); x_train=transform_inputs(train,final_transform); y_train=transform_targets(train,final_transform)
    final_models=[]; final_audits=[]
    for seed in contract["predictor_contract"]["ensemble_seeds"]:
        model,audit=fit_model(x_train,y_train,selected_config,int(seed),fixed_epochs=selected_epochs); final_models.append(model); final_audits.append({"seed":seed,**audit})
    raw_val_calls,raw_val_bytes=predict_histograms(validation,final_transform,final_models); dnn_val_calls,dnn_val_bytes=shrink_prediction(validation,raw_val_calls,raw_val_bytes,selected_alpha)
    h0_val_calls,h0_val_bytes=histogram_arrays(validation,"h0"); target_val_calls,target_val_bytes=target_arrays(validation)
    h0_val_metrics=metric_bundle(h0_val_calls,h0_val_bytes,target_val_calls,target_val_bytes); dnn_val_metrics=metric_bundle(dnn_val_calls,dnn_val_bytes,target_val_calls,target_val_bytes); overall=compare_to_h0(dnn_val_metrics,h0_val_metrics)
    segment_audits={}; segment_gate=True
    for segment in contract["selection_contract"]["segments"]:
        indices=[i for i,row in enumerate(validation) if row["segment"]==segment]
        h0m=metric_bundle(h0_val_calls[indices],h0_val_bytes[indices],target_val_calls[indices],target_val_bytes[indices]); dnnm=metric_bundle(dnn_val_calls[indices],dnn_val_bytes[indices],target_val_calls[indices],target_val_bytes[indices]); comp=compare_to_h0(dnnm,h0m)
        gate=float(comp["composite_ratio"])<1.0 and float(comp["metric_ratios_to_h0"]["calls_histogram_wape"])<=1.05 and float(comp["metric_ratios_to_h0"]["bytes_histogram_wape"])<=1.05
        segment_audits[segment]={"h0":h0m,"h0_plus_dnn":dnnm,**comp,"gate":gate}; segment_gate &= gate
    overall_gate=accepted(overall); model_accepted=bool(oof_accepted and overall_gate and segment_gate)
    output.mkdir(parents=True)
    write_csv_gz(output/"dataset/pd_expanded_h0_residual_examples.csv.gz",examples); write_csv_gz(output/"dataset/pd_expanded_hfull_targets.csv.gz",targets); write_csv_gz(output/"profiles/expanded_lowdim_profiles.csv.gz",profiles)
    write_csv(output/"analysis/candidate_alpha_metrics.csv",candidate_rows)
    aggregate=[]
    for method,metrics in (("h0",h0_val_metrics),("h0_plus_dnn_residual",dnn_val_metrics)):
        aggregate.append({"method":method,**metrics,"composite_ratio_to_h0":1.0 if method=="h0" else overall["composite_ratio"],"hard_gate":True if method=="h0" else overall_gate})
    write_csv(output/"analysis/development_validation_metrics.csv",aggregate); write_json(output/"analysis/segment_validation.json",segment_audits)
    write_csv_gz(output/"predictions/development_validation_predictions.csv.gz",prediction_rows(validation,h0_val_calls,h0_val_bytes,"h0")+prediction_rows(validation,dnn_val_calls,dnn_val_bytes,"h0_plus_dnn_residual"))
    checkpoint={"schema_version":"phase44-pd-expanded-protected-checkpoint-v1","workflow_commit":expected,"selected_candidate":selected_config,"selected_alpha":selected_alpha,"selected_epochs":selected_epochs,"transform":final_transform,"ensemble_seeds":contract["predictor_contract"]["ensemble_seeds"],"models":[model_to_json(model) for model in final_models],"training_profile_ids":[row["profile_id"] for row in train],"phase43_targets_accessed":False}
    write_json_gz(output/"checkpoints/pd_qwen3_expanded_h0_protected_dnn.json.gz",checkpoint)
    write_json(output/"audit/input_freeze.json",preflight); write_json(output/"audit/dataset_generation.json",{"schema_version":"phase44-dataset-generation-audit-v1","profiles":1200,"train_profiles":960,"validation_profiles":240,"complete_requests_used_outside_git":total_requests,"complete_request_rows_committed":0,"phase43_targets_accessed":False})
    write_json(output/"audit/training.json",{"schema_version":"phase44-training-audit-v1","fold_assignment":folds,"selected_candidate_id":selected_config["candidate_id"],"selected_alpha":selected_alpha,"selected_epochs":selected_epochs,"oof_accepted":oof_accepted,"validation_overall_gate":overall_gate,"validation_segment_gate":segment_gate,"model_accepted":model_accepted,"final_models":final_audits})
    write_json(output/"audit/environment.json",{**environment_record(),"numpy":np.__version__,"gpu_used":False,"network_used":False,"raw_mutated":False,"phase43_targets_accessed":False})
    summary={"schema_version":"phase44-pd-expanded-protected-training-result-v1","status":"PASS","workflow_commit":expected,"completed_at_utc":utc_now(),"counts":{"profiles":1200,"train_profiles":960,"validation_profiles":240,"complete_requests":total_requests,"candidate_alpha_rows":len(candidate_rows),"complete_request_rows_in_git":0},"selected":{"candidate_id":selected_config["candidate_id"],"feature_mode":selected_config["feature_mode"],"alpha":selected_alpha,"epochs":selected_epochs},"gates":{"oof_accepted":oof_accepted,"validation_overall":overall_gate,"validation_all_segments":segment_gate,"model_accepted":model_accepted,"new_blind_permitted":model_accepted},"development_validation":{"h0":h0_val_metrics,"h0_plus_dnn_residual":dnn_val_metrics,**overall},"segments":segment_audits,"proved":"expanded BurstGPT development evidence for a Qwen3 pure-PD H0-protected residual predictor","not_proved":"fresh blind generalization, Mooncake, other models, physical RDMA cost or placement"}
    write_json(output/"summary.json",summary)
    (output/"README.md").write_text(f"# Phase44：扩展PD开发集与H0保护残差训练\n\n状态：`PASS`。从1200个互不重叠且避开历次窗口的BurstGPT画像生成标签，共{total_requests}个完整请求；完整请求未进入Git。\n\n选中`{selected_config['candidate_id']}`、alpha=`{selected_alpha}`。OOF gate=`{oof_accepted}`，240画像validation overall gate=`{overall_gate}`，三segment gate=`{segment_gate}`，最终model_accepted=`{model_accepted}`。只有最终为true才允许新blind。\n",encoding="utf-8")
    (output/"logs").mkdir(); (output/"logs/runtime.log").write_text(f"completed={utc_now()} workflow_commit={expected}\nprofiles=1200 requests={total_requests} train=960 validation=240\nselected={selected_config['candidate_id']} alpha={selected_alpha} epochs={selected_epochs}\noof_accepted={oof_accepted} overall_gate={overall_gate} segment_gate={segment_gate} model_accepted={model_accepted}\ngpu=false raw_committed=false phase43_targets_accessed=false\n",encoding="utf-8")
    (output/"DONE").write_text("PASS\n",encoding="utf-8"); refresh_manifest(output); return summary


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit",required=True); parser.add_argument("--raw-dir",type=Path,required=True); parser.add_argument("--output-dir",type=Path,default=repo_root()/"experiment-results/phase44_pd_expanded_protected_training")
    args=parser.parse_args(); print(json.dumps(run(args.expected_workflow_commit,args.raw_dir,args.output_dir.resolve()),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
