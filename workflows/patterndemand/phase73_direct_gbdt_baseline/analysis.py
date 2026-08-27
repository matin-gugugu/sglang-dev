#!/usr/bin/env python3
"""Phase73 candidate selection and fixed target-open benchmark analysis."""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from gbdt import direct_feature_names,encode_histograms,feature_importance,feature_matrix,fit_model,histogram_arrays,metric_bundle,predict_histograms,read_csv_gz

METHODS=("h0","h0_plus_dnn_residual","direct_gbdt")
IDS=("profile_id","split_role","source","segment","source_split","window_id","cutoff_ms","model")


def _key(row:dict[str,str])->tuple[str,str]:return row["profile_id"],row["model"]


def select_candidate(examples:list[dict[str,str]],spec:dict[str,Any])->tuple[dict[str,Any],list[dict[str,Any]],dict[str,Any]]:
    train=[row for row in examples if row["split_role"]=="expanded_train"];validation=[row for row in examples if row["split_role"]=="expanded_validation"]
    names=direct_feature_names(train);X_train=feature_matrix(train,names);X_validation=feature_matrix(validation,names);tc_train,tb_train=histogram_arrays(train,"target");tc_val,tb_val=histogram_arrays(validation,"target");encoded=encode_histograms(tc_train,tb_train)
    candidate_rows=[];models={}
    for index,config in enumerate(spec["candidate_selection"]["candidates"]):
        started=time.perf_counter();model=fit_model(X_train,encoded,names,config,int(spec["candidate_selection"]["seed"])+index*10007);pc,pb=predict_histograms(model,validation);metrics=metric_bundle(pc,pb,tc_val,tb_val);objective=(metrics["calls_histogram_wape"]+metrics["bytes_histogram_wape"])/2
        candidate_rows.append({"candidate_id":config["candidate_id"],"max_depth":config["max_depth"],"estimators":config["estimators"],"learning_rate":config["learning_rate"],"min_leaf":config["min_leaf"],"features":len(names),"validation_calls_histogram_wape":metrics["calls_histogram_wape"],"validation_bytes_histogram_wape":metrics["bytes_histogram_wape"],"validation_mean_wape":objective,"elapsed_seconds":time.perf_counter()-started,"selected":False});models[config["candidate_id"]]=model
    ordered=sorted(candidate_rows,key=lambda row:(row["validation_mean_wape"],row["max_depth"],row["estimators"],row["candidate_id"]));selected_id=ordered[0]["candidate_id"]
    for row in candidate_rows:row["selected"]=row["candidate_id"]==selected_id
    selection_payload={"candidate_rows":[{key:value for key,value in row.items() if key!="elapsed_seconds"} for row in candidate_rows],"selected_candidate_id":selected_id,"phase50_labels_loaded":False}
    selection_digest=hashlib.sha256(json.dumps(selection_payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    selected_config=next(config for config in spec["candidate_selection"]["candidates"] if config["candidate_id"]==selected_id)
    return selected_config,candidate_rows,{"feature_names":names,"selection_digest_before_phase50_load":selection_digest,"selection_payload":selection_payload,"discarded_candidate_models":len(models)-1}


def refit(examples:list[dict[str,str]],names:list[str],config:dict[str,Any],seed:int)->dict[str,Any]:
    calls,logical=histogram_arrays(examples,"target");return fit_model(feature_matrix(examples,names),encode_histograms(calls,logical),names,config,seed+900001)


def frozen_methods(features:list[dict[str,str]],frozen_predictions:list[dict[str,str]],direct_model:dict[str,Any])->tuple[list[tuple[str,str]],dict[str,tuple[np.ndarray,np.ndarray]],list[dict[str,Any]]]:
    keys=[_key(row) for row in features];by_method={method:{} for method in ("h0","h0_plus_dnn_residual")}
    for row in frozen_predictions:by_method[row["method"]][_key(row)]=row
    methods={}
    for method in by_method:
        if set(by_method[method])!=set(keys):raise RuntimeError(f"frozen method key mismatch: {method}")
        ordered=[by_method[method][key] for key in keys];methods[method]=(np.asarray([[float(row[f"predicted_calls_bin_{i:02d}"]) for i in range(12)] for row in ordered]),np.asarray([[float(row[f"predicted_logical_bytes_bin_{i:02d}"]) for i in range(12)] for row in ordered]))
    dc,db=predict_histograms(direct_model,features);methods["direct_gbdt"]=(dc,db)
    prediction_rows=[]
    for row,calls,logical in zip(features,dc,db):prediction_rows.append({**{name:row[name] for name in IDS},"method":"direct_gbdt","predicted_total_calls_per_1000":float(calls.sum()),"predicted_total_logical_bytes_per_1000":float(logical.sum()),**{f"predicted_calls_bin_{i:02d}":float(calls[i]) for i in range(12)},**{f"predicted_logical_bytes_bin_{i:02d}":float(logical[i]) for i in range(12)}})
    return keys,methods,prediction_rows


def target_arrays(targets:list[dict[str,str]],keys:list[tuple[str,str]])->tuple[np.ndarray,np.ndarray,list[dict[str,str]]]:
    by_key={_key(row):row for row in targets}
    if set(by_key)!=set(keys):raise RuntimeError("Phase50 target key mismatch")
    ordered=[by_key[key] for key in keys];calls=np.asarray([[float(row[f"target_calls_bin_{i:02d}"]) for i in range(12)] for row in ordered]);logical=np.asarray([[float(row[f"target_logical_bytes_bin_{i:02d}"]) for i in range(12)] for row in ordered]);return calls,logical,ordered


def _group(indices:list[int],methods:dict[str,tuple[np.ndarray,np.ndarray]],target:tuple[np.ndarray,np.ndarray])->dict[str,Any]:
    idx=np.asarray(indices,dtype=int);return {method:metric_bundle(values[0][idx],values[1][idx],target[0][idx],target[1][idx]) for method,values in methods.items()}


def evaluate(features:list[dict[str,str]],frozen_predictions:list[dict[str,str]],targets:list[dict[str,str]],direct_model:dict[str,Any],spec:dict[str,Any])->dict[str,Any]:
    keys,methods,prediction_rows=frozen_methods(features,frozen_predictions,direct_model);tc,tb,ordered_targets=target_arrays(targets,keys);target=(tc,tb);all_indices=list(range(len(keys)));overall=_group(all_indices,methods,target)
    models={model:_group([i for i,key in enumerate(keys) if key[1]==model],methods,target) for model in sorted({key[1] for key in keys})}
    segments={segment:_group([i for i,row in enumerate(features) if row["segment"]==segment],methods,target) for segment in sorted({row["segment"] for row in features})}
    aggregate=[{"method":method,**overall[method]} for method in METHODS]
    per_bin=[]
    for method in METHODS:
        for kind,predicted,truth in (("calls",methods[method][0],tc),("logical_bytes",methods[method][1],tb)):
            for index in range(12):per_bin.append({"method":method,"kind":kind,"bin":index,"predicted_sum":float(predicted[:,index].sum()),"target_sum":float(truth[:,index].sum()),"absolute_error_sum":float(np.abs(predicted[:,index]-truth[:,index]).sum()),"wape":float(np.abs(predicted[:,index]-truth[:,index]).sum()/max(truth[:,index].sum(),1e-12))})
    per_unit=[]
    for method in METHODS:
        pc,pb=methods[method]
        for i,(key,row) in enumerate(zip(keys,features)):per_unit.append({"profile_id":key[0],"model":key[1],"segment":row["segment"],"method":method,**metric_bundle(pc[i:i+1],pb[i:i+1],tc[i:i+1],tb[i:i+1])})
    target_contract=spec["benchmark_contract"]["absolute_target"];direct=overall["direct_gbdt"];h0=overall["h0"];dnn=overall["h0_plus_dnn_residual"]
    overall_abs=direct["calls_histogram_wape"]<=target_contract["overall_calls_wape_max"] and direct["bytes_histogram_wape"]<=target_contract["overall_bytes_wape_max"]
    model_abs=all(value["direct_gbdt"]["calls_histogram_wape"]<=target_contract["each_model_calls_bytes_wape_max"] and value["direct_gbdt"]["bytes_histogram_wape"]<=target_contract["each_model_calls_bytes_wape_max"] for value in models.values())
    segment_abs=all(value["direct_gbdt"]["calls_histogram_wape"]<=target_contract["each_segment_calls_bytes_wape_max"] and value["direct_gbdt"]["bytes_histogram_wape"]<=target_contract["each_segment_calls_bytes_wape_max"] for value in segments.values())
    beats_h0=direct["calls_histogram_wape"]<h0["calls_histogram_wape"] and direct["bytes_histogram_wape"]<h0["bytes_histogram_wape"]
    beats_dnn=direct["calls_histogram_wape"]<dnn["calls_histogram_wape"] and direct["bytes_histogram_wape"]<dnn["bytes_histogram_wape"]
    if overall_abs and model_abs and segment_abs and beats_dnn:outcome="TARGET_OPEN_DIRECT_GBDT_MEETS_ABSOLUTE_GATES_AND_BEATS_H0_DNN"
    elif beats_dnn:outcome="TARGET_OPEN_DIRECT_GBDT_BEATS_H0_DNN_BUT_ABSOLUTE_GATES_NOT_MET"
    elif beats_h0:outcome="TARGET_OPEN_DIRECT_GBDT_BEATS_H0_NOT_H0_DNN"
    else:outcome="TARGET_OPEN_DIRECT_GBDT_DOES_NOT_BEAT_H0"
    return {"keys":keys,"methods":methods,"ordered_targets":ordered_targets,"prediction_rows":prediction_rows,"aggregate":aggregate,"models":models,"segments":segments,"per_bin":per_bin,"per_unit":per_unit,"decision":{"scientific_outcome":outcome,"classification":"target-open fixed benchmark; not fresh blind","overall_absolute_gate":overall_abs,"all_models_absolute_gate":model_abs,"all_segments_absolute_gate":segment_abs,"absolute_target_met":bool(overall_abs and model_abs and segment_abs),"beats_h0_both_histogram_wapes":beats_h0,"beats_h0_plus_dnn_both_histogram_wapes":beats_dnn}}


def bootstrap(per_unit:list[dict[str,Any]],draws:int=10000,seed:int=730073)->dict[str,Any]:
    grouped=defaultdict(dict)
    for row in per_unit:grouped[row["method"]].setdefault(row["profile_id"],[]).append(row)
    profiles=sorted(grouped["direct_gbdt"]);rng=np.random.default_rng(seed);result={"schema_version":"phase73-profile-cluster-bootstrap-v1","draws":draws,"seed":seed,"cluster":"profile; six models remain together","difference":"baseline minus Direct-GBDT; positive favors Direct-GBDT"}
    for baseline in ("h0","h0_plus_dnn_residual"):
        result[baseline]={}
        for metric in ("mean_profile_calls_l1","mean_profile_bytes_l1"):
            paired=np.asarray([np.mean([float(row[metric]) for row in grouped[baseline][profile]])-np.mean([float(row[metric]) for row in grouped["direct_gbdt"][profile]]) for profile in profiles]);sample=paired[rng.integers(0,len(paired),size=(draws,len(paired)))].mean(1)
            result[baseline][metric]={"observed_mean_difference":float(paired.mean()),"ci95_low":float(np.quantile(sample,.025)),"ci95_high":float(np.quantile(sample,.975)),"fraction_positive":float(np.mean(sample>0))}
    return result


def svg_overall(aggregate:list[dict[str,Any]])->str:
    colors={"h0":"#98a2b3","h0_plus_dnn_residual":"#2e90fa","direct_gbdt":"#7f56d9"};labels={"h0":"H0","h0_plus_dnn_residual":"H0+DNN","direct_gbdt":"Direct-GBDT"};width=900;height=520;top=70;bottom=100;left=90;plot_h=height-top-bottom;ymax=.35
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:23px;font-weight:700}.axis{font-size:13px}.value{font-size:12px}.grid{stroke:#dfe5ec}</style>','<text x="450" y="35" text-anchor="middle" class="title">Phase73 target-open histogram benchmark</text>']
    for i in range(8):v=ymax*i/7;y=top+plot_h*(1-i/7);lines += [f'<line x1="{left}" y1="{y:.1f}" x2="850" y2="{y:.1f}" class="grid"/>',f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{v*100:.0f}%</text>']
    metrics=("calls_histogram_wape","bytes_histogram_wape");centers=(300,650)
    for mi,metric in enumerate(metrics):
        for j,row in enumerate(aggregate):
            value=float(row[metric]);x=centers[mi]-100+j*80;bar=plot_h*min(value,ymax)/ymax;y=top+plot_h-bar
            lines += [f'<rect x="{x}" y="{y:.1f}" width="55" height="{bar:.1f}" rx="3" fill="{colors[row["method"]]}"/>',f'<text x="{x+27.5}" y="{y-6:.1f}" text-anchor="middle" class="value">{value*100:.1f}%</text>']
        lines.append(f'<text x="{centers[mi]}" y="445" text-anchor="middle" class="axis">{metric.replace("_histogram_wape","").upper()}</text>')
    for j,method in enumerate(METHODS):x=200+j*190;lines += [f'<rect x="{x}" y="480" width="15" height="15" fill="{colors[method]}"/>',f'<text x="{x+22}" y="492" class="axis">{labels[method]}</text>']
    lines.append('</svg>\n');return "\n".join(lines)


def svg_models(models:dict[str,Any])->str:
    names=list(sorted(models));width=1100;height=580;left=90;top=70;plot_h=390;ymax=.45;group=980/len(names);lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.axis{font-size:12px}.note{font-size:11px;fill:#5f6b7a}.grid{stroke:#dfe5ec}</style>','<text x="550" y="34" text-anchor="middle" class="title">Per-model calls/bytes histogram WAPE</text>']
    for i in range(6):v=ymax*i/5;y=top+plot_h*(1-i/5);lines += [f'<line x1="{left}" y1="{y:.1f}" x2="1070" y2="{y:.1f}" class="grid"/>',f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{v*100:.0f}%</text>']
    for i,name in enumerate(names):
        center=left+group*(i+.5);d=models[name]
        # Four bars: DNN calls/bytes then Direct calls/bytes.
        values=[d["h0_plus_dnn_residual"]["calls_histogram_wape"],d["h0_plus_dnn_residual"]["bytes_histogram_wape"],d["direct_gbdt"]["calls_histogram_wape"],d["direct_gbdt"]["bytes_histogram_wape"]];colors=["#84adff","#2e90fa","#b692f6","#7f56d9"]
        for j,(value,color) in enumerate(zip(values,colors)):x=center-52+j*27;bar=plot_h*min(value,ymax)/ymax;y=top+plot_h-bar;lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="21" height="{bar:.1f}" fill="{color}"/>')
        short=name.replace("-instruct-v0.1","").replace("-instruct","");lines += [f'<text x="{center:.1f}" y="482" text-anchor="middle" class="axis">{short}</text>',f'<text x="{center:.1f}" y="501" text-anchor="middle" class="note">DNN {values[0]*100:.1f}/{values[1]*100:.1f} · GBDT {values[2]*100:.1f}/{values[3]*100:.1f}</text>']
    lines += ['<text x="550" y="548" text-anchor="middle" class="note">Pairs are calls/bytes WAPE. Phase50 labels were already open; this is not fresh blind.</text>','</svg>\n'];return "\n".join(lines)
