#!/usr/bin/env python3
"""Small deterministic NumPy Direct-GBDT with no H0 or pseudo-request dependency."""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

BIN_EDGES_BYTES=np.asarray([4096.0,13777.246867516858,46340.95001184158,155871.75497763665,524288.0,1763487.5990421579,5931641.601515722,19951584.63713749,67108864.0,225726412.6773962,759250124.9940125,2553802833.553599,8589934592.0],dtype=np.float64)


def read_csv_gz(path:Path)->list[dict[str,str]]:
    with gzip.open(path,"rt",newline="",encoding="utf-8") as source:return list(csv.DictReader(source))


def write_csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:raise ValueError(f"empty rows: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)


def write_csv_gz(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:raise ValueError(f"empty rows: {path}")
    buffer=io.StringIO(newline="");writer=csv.DictWriter(buffer,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as output:output.write(buffer.getvalue().encode("utf-8"))


def write_json_gz(path:Path,value:Any)->None:
    payload=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8");path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as output:output.write(payload)


def read_json_gz(path:Path)->Any:
    with gzip.open(path,"rt",encoding="utf-8") as source:return json.load(source)


def direct_feature_names(rows:list[dict[str,str]])->list[str]:
    names=sorted(name for name in rows[0] if name.startswith("feature_"))
    forbidden=[name for name in names if name.startswith(("h0_","target_","residual_"))]
    if forbidden:raise RuntimeError({"forbidden_direct_features":forbidden})
    values=np.asarray([[float(row[name]) for name in names] for row in rows],dtype=np.float64)
    if not np.isfinite(values).all():raise ValueError("non-finite direct features")
    return [name for name,std in zip(names,values.std(axis=0)) if std>1e-12]


def feature_matrix(rows:list[dict[str,str]],names:list[str])->np.ndarray:
    if any(not name.startswith("feature_") or name.startswith(("h0_","target_","residual_")) for name in names):raise RuntimeError("Direct-GBDT feature scope violation")
    value=np.asarray([[float(row[name]) for name in names] for row in rows],dtype=np.float64)
    if not np.isfinite(value).all():raise ValueError("non-finite direct feature matrix")
    return value


def histogram_arrays(rows:list[dict[str,str]],prefix:str)->tuple[np.ndarray,np.ndarray]:
    calls=np.asarray([[float(row[f"{prefix}_calls_bin_{i:02d}"]) for i in range(12)] for row in rows],dtype=np.float64)
    logical=np.asarray([[float(row[f"{prefix}_logical_bytes_bin_{i:02d}"]) for i in range(12)] for row in rows],dtype=np.float64)
    return calls,logical


def encode_histograms(calls:np.ndarray,logical:np.ndarray)->np.ndarray:
    parts=[]
    for vectors in (calls,logical):
        totals=np.maximum(vectors.sum(axis=1),0.0);smooth=np.maximum(totals,1.0)*1e-6/12
        shares=(vectors+smooth[:,None])/(totals[:,None]+12*smooth[:,None]);parts.append(np.c_[np.log1p(totals),np.log(shares)])
    return np.concatenate(parts,axis=1)


def decode_histograms(encoded:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    values=[]
    for offset in (0,13):
        total=np.expm1(np.clip(encoded[:,offset],0,40));logits=np.clip(encoded[:,offset+1:offset+13],-50,50);logits-=logits.max(axis=1,keepdims=True)
        share=np.exp(logits);share/=share.sum(axis=1,keepdims=True);values.append(total[:,None]*share)
    return project_histograms(values[0],values[1])


def project_histograms(calls:np.ndarray,logical:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    calls=np.maximum(calls,0.0);logical=np.maximum(logical,0.0)
    lower=calls*BIN_EDGES_BYTES[:-1];upper=calls*BIN_EDGES_BYTES[1:]
    logical=np.minimum(np.maximum(logical,lower),upper)
    return calls,logical


def metric_bundle(pc:np.ndarray,pb:np.ndarray,tc:np.ndarray,tb:np.ndarray)->dict[str,float]:
    ct=tc.sum(1);bt=tb.sum(1);pct=pc.sum(1);pbt=pb.sum(1);ps=pc/np.maximum(pct[:,None],1e-12);ts=tc/np.maximum(ct[:,None],1e-12)
    return {"profiles":int(len(pc)),"calls_histogram_wape":float(np.abs(pc-tc).sum()/max(tc.sum(),1e-12)),"bytes_histogram_wape":float(np.abs(pb-tb).sum()/max(tb.sum(),1e-12)),"calls_total_wape":float(np.abs(pct-ct).sum()/max(ct.sum(),1e-12)),"bytes_total_wape":float(np.abs(pbt-bt).sum()/max(bt.sum(),1e-12)),"mean_profile_calls_l1":float(np.mean(np.abs(pc-tc).sum(1)/np.maximum(ct,1e-12))),"mean_profile_bytes_l1":float(np.mean(np.abs(pb-tb).sum(1)/np.maximum(bt,1e-12))),"mean_calls_histogram_tv":float(np.mean(.5*np.abs(ps-ts).sum(1))),"mean_normalized_log_payload_emd":float(np.mean(np.abs(np.cumsum(ps,1)-np.cumsum(ts,1)).sum(1)/11))}


def _leaf(value:float)->dict[str,Any]:return {"value":float(value)}


def _build_tree(X:np.ndarray,residual:np.ndarray,indices:np.ndarray,depth:int,config:dict[str,Any],rng:np.random.Generator)->dict[str,Any]:
    value=float(residual[indices].mean())
    if depth==0 or len(indices)<2*int(config["min_leaf"]):return _leaf(value)
    parent=len(indices)*value*value;best=None;count=max(1,min(X.shape[1],int(math.ceil(X.shape[1]*float(config["feature_fraction"])))))
    for feature in rng.choice(X.shape[1],size=count,replace=False):
        column=X[indices,feature];thresholds=np.unique(np.quantile(column,config["threshold_quantiles"]))
        for threshold in thresholds:
            mask=column<=threshold;left_count=int(mask.sum());right_count=len(indices)-left_count
            if left_count<int(config["min_leaf"]) or right_count<int(config["min_leaf"]):continue
            left_indices=indices[mask];right_indices=indices[~mask];left_value=float(residual[left_indices].mean());right_value=float(residual[right_indices].mean())
            gain=left_count*left_value*left_value+right_count*right_value*right_value-parent
            candidate=(gain,int(feature),float(threshold),left_indices,right_indices)
            if best is None or candidate[:3]>best[:3]:best=candidate
    if best is None:return _leaf(value)
    gain,feature,threshold,left_indices,right_indices=best
    return {"feature":feature,"threshold":threshold,"gain":float(gain),"left":_build_tree(X,residual,left_indices,depth-1,config,rng),"right":_build_tree(X,residual,right_indices,depth-1,config,rng)}


def _predict_tree(node:dict[str,Any],X:np.ndarray,output:np.ndarray,indices:np.ndarray)->None:
    if "value" in node:output[indices]=float(node["value"]);return
    mask=X[indices,int(node["feature"])]<=float(node["threshold"]);_predict_tree(node["left"],X,output,indices[mask]);_predict_tree(node["right"],X,output,indices[~mask])


def _fit_output(X:np.ndarray,y:np.ndarray,config:dict[str,Any],seed:int)->dict[str,Any]:
    rng=np.random.default_rng(seed);prediction=np.full(len(X),float(y.mean()));trees=[];all_indices=np.arange(len(X));sample_count=max(2*int(config["min_leaf"]),int(len(X)*float(config["row_subsample"])))
    for _ in range(int(config["estimators"])):
        sample=np.sort(rng.choice(len(X),size=sample_count,replace=False));tree=_build_tree(X,y-prediction,sample,int(config["max_depth"]),config,rng);update=np.empty(len(X));_predict_tree(tree,X,update,all_indices);prediction+=float(config["learning_rate"])*update;trees.append(tree)
    return {"base":float(y.mean()),"trees":trees}


def fit_model(X:np.ndarray,encoded:np.ndarray,feature_names:list[str],config:dict[str,Any],seed:int)->dict[str,Any]:
    outputs=[_fit_output(X,encoded[:,index],config,seed+1000003*index) for index in range(encoded.shape[1])]
    return {"schema_version":"phase73-direct-gbdt-model-v1","feature_names":feature_names,"output_encoding":"calls_log_total_plus_12_log_shares_then_bytes","config":config,"seed":seed,"outputs":outputs,"h0_inputs":0,"pseudo_requests":0,"teacher_calls":0}


def predict_encoded(model:dict[str,Any],X:np.ndarray)->np.ndarray:
    result=np.empty((len(X),len(model["outputs"])),dtype=np.float64);indices=np.arange(len(X));rate=float(model["config"]["learning_rate"])
    for output_index,output_model in enumerate(model["outputs"]):
        value=np.full(len(X),float(output_model["base"]));update=np.empty(len(X))
        for tree in output_model["trees"]:_predict_tree(tree,X,update,indices);value+=rate*update
        result[:,output_index]=value
    return result


def predict_histograms(model:dict[str,Any],rows:list[dict[str,str]])->tuple[np.ndarray,np.ndarray]:
    return decode_histograms(predict_encoded(model,feature_matrix(rows,list(model["feature_names"]))))


def feature_importance(model:dict[str,Any])->list[dict[str,Any]]:
    counts=Counter();gains=Counter()
    def visit(node:dict[str,Any])->None:
        if "value" in node:return
        name=model["feature_names"][int(node["feature"])];counts[name]+=1;gains[name]+=float(node["gain"]);visit(node["left"]);visit(node["right"])
    for output in model["outputs"]:
        for tree in output["trees"]:visit(tree)
    total=max(sum(gains.values()),1e-12)
    return [{"feature":name,"split_count":counts[name],"gain":gains[name],"gain_fraction":gains[name]/total} for name in sorted(counts,key=lambda name:(-gains[name],name))]
