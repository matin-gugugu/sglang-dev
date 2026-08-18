#!/usr/bin/env python3
"""Deterministic Phase52 PD histogram convolution and placement analysis."""
from __future__ import annotations
import bisect,csv,gzip,io,json,math,statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

METHODS=("h0","h0_plus_dnn_residual");LEVELS=("L1","L2","L3")

def read_csv(path:Path)->list[dict[str,str]]:
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"rt",newline="",encoding="utf-8") as source:return list(csv.DictReader(source))
def write_csv(path:Path,rows:list[dict])->None:
    if not rows:raise RuntimeError(f"refuse empty CSV: {path}")
    fields=[]
    for row in rows:fields.extend(name for name in row if name not in fields)
    buffer=io.StringIO(newline="");writer=csv.DictWriter(buffer,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows);path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix==".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as target:target.write(buffer.getvalue().encode())
    else:path.write_text(buffer.getvalue(),encoding="utf-8")
def vector(row:dict,prefix:str)->list[float]:
    values=[float(row[f"{prefix}_bin_{i:02d}"]) for i in range(12)]
    if not all(math.isfinite(v) and v>=0 for v in values):raise RuntimeError({"invalid_vector":prefix,"profile":row.get("profile_id"),"model":row.get("model")})
    return values
def percentile(values:list[float],q:float)->float:
    rows=sorted(values);pos=(len(rows)-1)*q;lo=math.floor(pos);hi=math.ceil(pos);return rows[lo] if lo==hi else rows[lo]*(hi-pos)+rows[hi]*(pos-lo)

def validate_inputs(predictions:list[dict],targets:list[dict],curves:list[dict],spec:dict)->dict:
    expected=spec["expected_counts"];errors=[]
    if len(predictions)!=expected["prediction_rows"]:errors.append("prediction_rows")
    if len(targets)!=expected["target_rows"]:errors.append("target_rows")
    target_keys={(r["profile_id"],r["model"]) for r in targets};prediction_keys={(r["profile_id"],r["model"],r["method"]) for r in predictions}
    if len(target_keys)!=len(targets) or prediction_keys!={(p,m,method) for p,m in target_keys for method in METHODS}:errors.append("prediction_target_keys")
    models={r["model"] for r in targets};profiles={r["profile_id"] for r in targets}
    if len(models)!=6 or len(profiles)!=300:errors.append("roster")
    curve_keys={(r.get("model_id"),r.get("topology_level")) for r in curves}
    if curve_keys!={(model,level) for model in models for level in LEVELS} or len(curves)!=18:errors.append("curve_matrix")
    scalar_max=0.0
    for row in predictions:
        calls=vector(row,"predicted_calls");logical=vector(row,"predicted_logical_bytes")
        scalar_max=max(scalar_max,abs(sum(calls)-float(row["predicted_total_calls_per_1000"]))/max(1.0,abs(float(row["predicted_total_calls_per_1000"]))),abs(sum(logical)-float(row["predicted_total_logical_bytes_per_1000"]))/max(1.0,abs(float(row["predicted_total_logical_bytes_per_1000"]))))
    for row in targets:
        calls=vector(row,"target_calls");logical=vector(row,"target_logical_bytes")
        scalar_max=max(scalar_max,abs(sum(calls)-float(row["target_total_calls_per_1000"]))/max(1.0,abs(float(row["target_total_calls_per_1000"]))),abs(sum(logical)-float(row["target_total_logical_bytes_per_1000"]))/max(1.0,abs(float(row["target_total_logical_bytes_per_1000"]))))
    if scalar_max>1e-12:errors.append("scalar_vector_mismatch")
    if errors:raise RuntimeError({"phase52_inputs":errors})
    return {"prediction_rows":len(predictions),"target_rows":len(targets),"profiles":len(profiles),"models":sorted(models),"curve_matrix":sorted([list(v) for v in curve_keys]),"scalar_max_relative_difference":scalar_max}

def curve_scenarios(curve:dict)->dict[str,dict]:
    rows=[]
    for knot in curve["knots"]:
        replicas=[float(row["slower_direction_latency_us"]) for row in knot["replicas"]]
        official=float(knot["official_latency_us"])
        if len(replicas)!=2 or not math.isclose(official,max(replicas),rel_tol=1e-10,abs_tol=1e-8):raise RuntimeError({"curve_policy":curve["curve_id"],"payload":knot["payload_bytes"]})
        rows.append({"payload_bytes":float(knot["payload_bytes"]),"official":official,"lower":min(replicas)})
    if [r["payload_bytes"] for r in rows]!=sorted(r["payload_bytes"] for r in rows):raise RuntimeError(f"curve payload order: {curve['curve_id']}")
    running=0.0
    for row in rows:running=max(running,row["official"]);row["monotone_official"]=running
    return {"metadata":curve,"rows":rows}
def interpolate(curve:dict,payload:float,field:str,audit:dict)->float:
    rows=curve["rows"];xs=[r["payload_bytes"] for r in rows];ys=[r[field] for r in rows];payload=max(float(payload),1.0)
    if payload<=xs[0]:position="low" if payload<xs[0] else "inside";value=ys[0]
    elif payload>=xs[-1]:position="high" if payload>xs[-1] else "inside";value=ys[-1]
    else:
        right=bisect.bisect_right(xs,payload);left=right-1;fraction=(math.log2(payload)-math.log2(xs[left]))/(math.log2(xs[right])-math.log2(xs[left]));value=ys[left]+fraction*(ys[right]-ys[left]);position="inside"
    audit["nonempty_bins"]+=1
    if position!="inside":audit[f"{position}_clamped_bins"]+=1
    return value
def histogram_cost(calls:list[float],logical:list[float],curve:dict,field:str,audit:dict)->float:
    total=0.0
    for count,byte_count in zip(calls,logical):
        if count<=1e-12:
            if byte_count>1e-6:raise RuntimeError("positive bytes with zero calls")
            continue
        if byte_count<=0:raise RuntimeError("positive calls with zero bytes")
        payload=byte_count/count;latency=interpolate(curve,payload,field,audit);total+=count*latency;audit["logical_calls"]+=count
        if payload<curve["rows"][0]["payload_bytes"]:audit["low_clamped_calls"]+=count
        if payload>curve["rows"][-1]["payload_bytes"]:audit["high_clamped_calls"]+=count
    return total

def cost_rows(predictions:list[dict],targets:list[dict],curves:list[dict])->tuple[list[dict],dict]:
    target={(r["profile_id"],r["model"]):r for r in targets};curve_map={(r["model_id"],r["topology_level"]):curve_scenarios(r) for r in curves};audits=defaultdict(lambda:defaultdict(float));output=[]
    for prediction in predictions:
        key=(prediction["profile_id"],prediction["model"]);teacher=target[key];pc=vector(prediction,"predicted_calls");pb=vector(prediction,"predicted_logical_bytes");tc=vector(teacher,"target_calls");tb=vector(teacher,"target_logical_bytes")
        for level in LEVELS:
            curve=curve_map[(prediction["model"],level)];values={}
            for field in ("official","lower","monotone_official"):
                values[f"predicted_{field}"]=histogram_cost(pc,pb,curve,field,audits[(prediction["method"],prediction["model"],level,"prediction",field)])
                values[f"teacher_{field}"]=histogram_cost(tc,tb,curve,field,audits[(prediction["method"],prediction["model"],level,"teacher",field)])
            p=values["predicted_official"];t=values["teacher_official"]
            output.append({"profile_id":prediction["profile_id"],"source":prediction["source"],"segment":prediction["segment"],"model":prediction["model"],"method":prediction["method"],"topology_level":level,"placement_id":f"phase51_{prediction['model']}__{level.lower()}","curve_id":curve["metadata"]["curve_id"],"curve_evidence":"phase51_physical_measurement","request_count":teacher["request_count"],"predicted_cost_us_per_1000":p,"teacher_cost_us_per_1000":t,"predicted_cost_lower_us_per_1000":values["predicted_lower"],"predicted_cost_upper_us_per_1000":p,"teacher_cost_lower_us_per_1000":values["teacher_lower"],"teacher_cost_upper_us_per_1000":t,"predicted_cost_monotone_us_per_1000":values["predicted_monotone_official"],"teacher_cost_monotone_us_per_1000":values["teacher_monotone_official"],"absolute_error_us_per_1000":abs(p-t),"absolute_percentage_error":abs(p-t)/max(t,1e-12),"signed_error_us_per_1000":p-t})
    return sorted(output,key=lambda r:(r["profile_id"],r["model"],r["method"],LEVELS.index(r["topology_level"]))),{"schema_version":"phase52-interpolation-audit-v1","roles":{"/".join(key):dict(value) for key,value in sorted(audits.items())}}

def _slices(row:dict)->tuple[tuple[str,str],...]:return (("overall","all"),("model",row["model"]),("segment",row["segment"]))
def aggregate_cost(rows:list[dict])->list[dict]:
    groups=defaultdict(list)
    for row in rows:
        for kind,value in _slices(row):groups[(row["method"],row["topology_level"],kind,value)].append(row)
    output=[]
    for (method,level,kind,value),items in sorted(groups.items()):
        teacher=sum(float(r["teacher_cost_us_per_1000"]) for r in items);predicted=sum(float(r["predicted_cost_us_per_1000"]) for r in items)
        output.append({"method":method,"topology_level":level,"slice_type":kind,"slice_value":value,"cases":len(items),"cost_mape":statistics.fmean(float(r["absolute_percentage_error"]) for r in items),"cost_wape":sum(float(r["absolute_error_us_per_1000"]) for r in items)/max(teacher,1e-12),"signed_bias":(predicted-teacher)/max(teacher,1e-12),"predicted_cost_us_per_1000_sum":predicted,"teacher_cost_us_per_1000_sum":teacher})
    return output
def compare_cost(metrics:list[dict])->list[dict]:
    groups=defaultdict(dict)
    for row in metrics:groups[(row["topology_level"],row["slice_type"],row["slice_value"])][row["method"]]=row
    output=[]
    for (level,kind,value),methods in sorted(groups.items()):
        h0=methods["h0"];dnn=methods["h0_plus_dnn_residual"]
        output.append({"topology_level":level,"slice_type":kind,"slice_value":value,"cases":h0["cases"],"h0_cost_mape":h0["cost_mape"],"dnn_cost_mape":dnn["cost_mape"],"cost_mape_ratio":dnn["cost_mape"]/max(h0["cost_mape"],1e-12),"h0_cost_wape":h0["cost_wape"],"dnn_cost_wape":dnn["cost_wape"],"cost_wape_ratio":dnn["cost_wape"]/max(h0["cost_wape"],1e-12),"h0_signed_bias":h0["signed_bias"],"dnn_signed_bias":dnn["signed_bias"],"strict_mape_and_wape_improvement":dnn["cost_mape"]<h0["cost_mape"] and dnn["cost_wape"]<h0["cost_wape"]})
    return output

def placement(rows:list[dict])->tuple[list[dict],list[dict]]:
    groups=defaultdict(list)
    for row in rows:groups[(row["profile_id"],row["model"],row["method"])].append(row)
    rankings=[];decisions=[]
    for key,items in sorted(groups.items()):
        if len(items)!=3 or {r["topology_level"] for r in items}!=set(LEVELS):raise RuntimeError({"placement_candidates":key})
        tie=lambda r:(float(r["predicted_cost_us_per_1000"]),LEVELS.index(r["topology_level"]));teacher_tie=lambda r:(float(r["teacher_cost_us_per_1000"]),LEVELS.index(r["topology_level"]));pred=sorted(items,key=tie);truth=sorted(items,key=teacher_tie);p_rank={r["topology_level"]:i+1 for i,r in enumerate(pred)};t_rank={r["topology_level"]:i+1 for i,r in enumerate(truth)}
        for row in items:rankings.append({**row,"predicted_rank":p_rank[row["topology_level"]],"teacher_rank":t_rank[row["topology_level"]]})
        selected=pred[0];oracle=truth[0];selected_teacher=float(selected["teacher_cost_us_per_1000"]);oracle_cost=float(oracle["teacher_cost_us_per_1000"]);mono_selected=min(items,key=lambda r:(float(r["predicted_cost_monotone_us_per_1000"]),LEVELS.index(r["topology_level"])));mono_oracle=min(items,key=lambda r:(float(r["teacher_cost_monotone_us_per_1000"]),LEVELS.index(r["topology_level"])))
        base={name:selected[name] for name in ("profile_id","source","segment","model","method","request_count")}
        decisions.append({**base,"ranking_scope":"communication_only_fixed_p1_d1_configuration","selected_topology":selected["topology_level"],"oracle_topology":oracle["topology_level"],"agreement":selected["topology_level"]==oracle["topology_level"],"teacher_regret":(selected_teacher-oracle_cost)/max(oracle_cost,1e-12),"predicted_margin":(float(pred[1]["predicted_cost_us_per_1000"])-float(selected["predicted_cost_us_per_1000"]))/max(float(selected["predicted_cost_us_per_1000"]),1e-12),"predicted_interval_robust":float(selected["predicted_cost_upper_us_per_1000"])<=min(float(r["predicted_cost_lower_us_per_1000"]) for r in items if r is not selected),"teacher_interval_robust":float(oracle["teacher_cost_upper_us_per_1000"])<=min(float(r["teacher_cost_lower_us_per_1000"]) for r in items if r is not oracle),"monotone_selected_topology":mono_selected["topology_level"],"monotone_oracle_topology":mono_oracle["topology_level"],"monotone_selected_stable":mono_selected["topology_level"]==selected["topology_level"],"monotone_oracle_stable":mono_oracle["topology_level"]==oracle["topology_level"],"selected_predicted_cost_us_per_1000":selected["predicted_cost_us_per_1000"],"selected_teacher_cost_us_per_1000":selected_teacher,"oracle_teacher_cost_us_per_1000":oracle_cost})
    return rankings,decisions
def aggregate_placement(rows:list[dict])->list[dict]:
    groups=defaultdict(list)
    for row in rows:
        for kind,value in _slices(row):groups[(row["method"],kind,value)].append(row)
    output=[]
    for (method,kind,value),items in sorted(groups.items()):
        regrets=[float(r["teacher_regret"]) for r in items]
        output.append({"method":method,"slice_type":kind,"slice_value":value,"cases":len(items),"agreement_rate":statistics.fmean(float(r["agreement"]) for r in items),"mean_teacher_regret":statistics.fmean(regrets),"p95_teacher_regret":percentile(regrets,.95),"max_teacher_regret":max(regrets),"predicted_interval_robust_fraction":statistics.fmean(float(r["predicted_interval_robust"]) for r in items),"teacher_interval_robust_fraction":statistics.fmean(float(r["teacher_interval_robust"]) for r in items),"monotone_selected_stability":statistics.fmean(float(r["monotone_selected_stable"]) for r in items),"monotone_oracle_stability":statistics.fmean(float(r["monotone_oracle_stable"]) for r in items),**{f"selected_{level.lower()}_fraction":sum(r["selected_topology"]==level for r in items)/len(items) for level in LEVELS},**{f"oracle_{level.lower()}_fraction":sum(r["oracle_topology"]==level for r in items)/len(items) for level in LEVELS}})
    return output
def compare_placement(metrics:list[dict])->list[dict]:
    groups=defaultdict(dict)
    for row in metrics:groups[(row["slice_type"],row["slice_value"])][row["method"]]=row
    output=[]
    for (kind,value),methods in sorted(groups.items()):
        h0=methods["h0"];dnn=methods["h0_plus_dnn_residual"]
        output.append({"slice_type":kind,"slice_value":value,"cases":h0["cases"],"h0_agreement_rate":h0["agreement_rate"],"dnn_agreement_rate":dnn["agreement_rate"],"agreement_delta":dnn["agreement_rate"]-h0["agreement_rate"],"h0_mean_teacher_regret":h0["mean_teacher_regret"],"dnn_mean_teacher_regret":dnn["mean_teacher_regret"],"mean_regret_ratio":dnn["mean_teacher_regret"]/max(h0["mean_teacher_regret"],1e-12),"agreement_weakly_improved":dnn["agreement_rate"]>=h0["agreement_rate"],"mean_regret_weakly_improved":dnn["mean_teacher_regret"]<=h0["mean_teacher_regret"]})
    return output
