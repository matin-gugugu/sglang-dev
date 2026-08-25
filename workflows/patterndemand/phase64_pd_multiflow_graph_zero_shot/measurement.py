#!/usr/bin/env python3
"""Phase64 raw validator and deterministic zero-shot graph evaluation."""
from __future__ import annotations
import bisect,json,math,statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from contracts import contract,file_sha,graph,iteration_counts,layout_by_id,load_json,measurement_by_id,payload_vectors,validate_plan

def _close(a:float,b:float)->bool:return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8)
def _cv(v:list[float])->float:return statistics.stdev(v)/statistics.mean(v) if len(v)>1 and statistics.mean(v) else 0.0
def _read_jsonl(path:Path)->list[dict]:
 rows=[]
 for n,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
  if not line.strip():raise RuntimeError({"blank_line":str(path),"line":n})
  try:value=json.loads(line)
  except json.JSONDecodeError as exc:raise RuntimeError({"invalid_jsonl":str(path),"line":n}) from exc
  if not isinstance(value,dict):raise RuntimeError({"non_object":str(path),"line":n})
  rows.append(value)
 return rows
def _validate_mode(mode:dict,flow_ids:list[int],timed:int)->bool:
 samples=mode.get("flow_latency_samples_us",{});summaries=mode.get("flow_latency_us",{});wave=mode.get("wave_latency_samples_us",[]);skew=mode.get("sender_start_skew_samples_us",[]);keys={str(v) for v in flow_ids}
 if set(samples)!=keys or set(summaries)!=keys or len(wave)!=timed or len(skew)!=timed or any(len(samples[k])!=timed for k in keys):return False
 values=[float(x) for v in samples.values() for x in v]+[float(x) for x in wave]+[float(x) for x in skew]
 if any(not math.isfinite(x) or x<0 for x in values) or any(float(x)<=0 for v in samples.values() for x in v) or any(float(x)<=0 for x in wave):return False
 if any(not _close(float(summaries[k].get("median",-1)),statistics.median(float(x) for x in samples[k])) for k in keys):return False
 for i in range(timed):
  if not _close(float(wave[i]),max(float(samples[str(fid)][i]) for fid in flow_ids)):return False
 return _close(float(mode.get("wave_latency_us",{}).get("median",-1)),statistics.median(float(x) for x in wave)) and mode.get("return_codes_all_zero") is True and mode.get("data_validation_pass") is True
def validate_raw(plan:dict,raw_dir:Path,*,require_complete:bool)->dict[str,Any]:
 audit=validate_plan(plan);spec=contract();mc=spec["measurement_contract"];raw_dir=raw_dir.expanduser().resolve()
 if not raw_dir.is_dir():raise RuntimeError(f"raw missing: {raw_dir}")
 files=sorted(p for p in raw_dir.rglob("*") if p.is_file());unexpected=[str(p.relative_to(raw_dir)) for p in files if p.suffix!=".jsonl"]
 if unexpected:raise RuntimeError({"unexpected_raw_files":unexpected})
 by:dict[str,dict[int,list[dict]]]=defaultdict(dict);manifest=[]
 for path in files:
  rel=path.relative_to(raw_dir);parts=rel.parts
  if len(parts)!=2 or not parts[1].startswith("repeat_") or not parts[1].endswith(".jsonl"):raise RuntimeError({"raw_path":str(rel)})
  try:rid=int(parts[1][7:-6])
  except ValueError as exc:raise RuntimeError({"repeat_path":str(rel)}) from exc
  if parts[1]!=f"repeat_{rid:02d}.jsonl" or rid in by[parts[0]]:raise RuntimeError({"duplicate_repeat":str(rel)})
  rows=_read_jsonl(path);by[parts[0]][rid]=rows;manifest.append({"path":str(rel),"sha256":file_sha(path),"bytes":path.stat().st_size,"records":len(rows)})
 expected_ids={m["measurement_id"] for m in plan["measurements"]}
 if set(by)-expected_ids:raise RuntimeError({"unknown_measurements":sorted(set(by)-expected_ids)})
 missing=[];needs=[];variance=[];quality=[];record_count=0;allowed=[int(mc["minimum_independent_repeats"]),int(mc["minimum_independent_repeats"])+int(mc["extra_repeats_per_round"]),int(mc["maximum_independent_repeats"])]
 for mid in sorted(expected_ids):
  m=measurement_by_id(plan,mid);vectors=payload_vectors(m["model_id"],m["configuration"]);vmap={v["vector_id"]:v for v in vectors};repeats=by.get(mid,{});rids=sorted(repeats)
  if rids and rids!=list(range(len(rids))):raise RuntimeError({"noncontiguous_repeats":mid,"actual":rids})
  if len(rids)>allowed[-1]:raise RuntimeError({"too_many_repeats":mid})
  if len(rids)<allowed[0]:missing.append({"measurement_id":mid,"have":len(rids),"need":allowed[0]-len(rids),"target_total":allowed[0]})
  elif len(rids) not in allowed:
   target=next(v for v in allowed if v>len(rids));missing.append({"measurement_id":mid,"have":len(rids),"need":target-len(rids),"target_total":target})
  medians=defaultdict(list)
  for rid,rows in sorted(repeats.items()):
   record_count+=len(rows);ids=[r.get("vector_id") for r in rows]
   if len(ids)!=10 or set(ids)!=set(vmap) or len(set(ids))!=10:raise RuntimeError({"vector_coverage":mid,"repeat":rid,"ids":ids})
   for row in rows:
    vector=vmap[row["vector_id"]];flows=vector["flows"];warm,timed=iteration_counts(sum(f["payload_bytes"] for f in flows));endpoints=row.get("runtime_endpoints",[])
    endpoint_ok=len(endpoints)==m["world_size"] and all(int(endpoints[i].get("rank",-1))==i and endpoints[i].get("role")==m["ranks"][i]["role"] and endpoints[i].get("expected_host")==m["ranks"][i]["host"] and int(endpoints[i].get("physical_gpu",-1))==int(m["ranks"][i]["physical_gpu"]) and endpoints[i].get("ib_device")==m["ranks"][i]["ib_device"] and endpoints[i].get("mooncake_protocol")=="rdma" and endpoints[i].get("with_nvidia_peermem")=="0" for i in range(m["world_size"]))
    solo=row.get("solo_flows",{});checks={"schema":row.get("schema_version")=="phase64-mooncake-multiflow-raw-v1","binding":row.get("workflow_commit")==plan["workflow_commit"] and row.get("plan_sha256")==audit["plan_sha256"] and row.get("measurement_id")==mid and row.get("measurement_sha256")==m["measurement_sha256"],"identity":row.get("model_id")==m["model_id"] and row.get("configuration")==m["configuration"] and row.get("topology_level")==m["topology_level"] and row.get("replica_id")==m["replica_id"] and row.get("placement_id")==m["placement_id"] and row.get("repeat_id")==rid,"vector":row.get("pages")==vector["pages"] and row.get("flows")==flows,"layout":row.get("descriptor_layout")==layout_by_id(m["model_id"])["descriptor_layout"] and int(row.get("descriptor_count",-1))==int(layout_by_id(m["model_id"])["descriptor_count"]),"operation":row.get("op")=="MooncakeTransferEngine.batch_transfer_sync" and row.get("transport")=="rdma" and row.get("wave_admission")=="gloo_barrier_then_all_graph_edges_synchronous_release","iterations":row.get("warmup_iterations")==warm and row.get("timed_iterations")==timed,"solo_keys":set(solo)=={str(f["flow_id"]) for f in flows},"solo_modes":all(_validate_mode(solo.get(str(f["flow_id"]),{}),[f["flow_id"]],timed) for f in flows),"concurrent":_validate_mode(row.get("concurrent_wave",{}),[f["flow_id"] for f in flows],timed),"endpoints":endpoint_ok,"timestamp":isinstance(row.get("timestamp_utc"),str) and row["timestamp_utc"]>=plan["generated_at_utc"]}
    if not all(checks.values()):raise RuntimeError({"invalid_raw_record":mid,"repeat":rid,"vector":row.get("vector_id"),"checks":checks})
    medians[row["vector_id"]].append(float(row["concurrent_wave"]["wave_latency_us"]["median"]))
  cvs={k:_cv(v) for k,v in sorted(medians.items())};max_cv=max(cvs.values(),default=0.0)
  if len(rids) in allowed and max_cv>float(mc["repeat_median_cv_threshold"]):
   if len(rids)<allowed[-1]:needs.append({"measurement_id":mid,"have":len(rids),"add":int(mc["extra_repeats_per_round"]),"max_cv":max_cv})
   else:variance.append({"measurement_id":mid,"repeats":len(rids),"max_cv":max_cv})
  quality.append({"measurement_id":mid,"model_id":m["model_id"],"configuration":m["configuration"],"topology_level":m["topology_level"],"replica_id":m["replica_id"],"world_size":m["world_size"],"flow_count":m["flow_count"],"repeat_count":len(rids),"record_count":sum(len(v) for v in repeats.values()),"max_repeat_median_cv":max_cv,"repeat_median_cv":cvs})
 complete=not missing and not needs
 if require_complete and not complete:raise RuntimeError({"raw_not_complete":True,"missing":missing,"needs_extra":needs})
 return {"schema_version":"phase64-raw-audit-v1","raw_dir":str(raw_dir),"plan_sha256":audit["plan_sha256"],"complete":complete,"counts":{"files":len(manifest),"records":record_count,"measurements_with_data":len(by),"expected_measurements":48},"missing":missing,"needs_extra":needs,"final_runtime_variance":variance,"files":manifest,"measurements":quality,"records":by}
def _curve_map()->dict:
 root=Path(__file__).resolve().parents[3];rows=load_json(root/"experiment-results/phase51_pd_l1_l3_physical_curve_library/curves/pd_mooncake_physical_curves.json")["curves"];return {(r["model_id"],r["topology_level"]):r for r in rows}
def _interpolate(curve:dict,payload:int)->float:
 knots=curve["knots"];xs=[int(r["payload_bytes"]) for r in knots];ys=[float(r["official_latency_us"]) for r in knots]
 if payload<xs[0] or payload>xs[-1]:raise RuntimeError({"payload_outside_curve":payload})
 if payload in xs:return ys[xs.index(payload)]
 right=bisect.bisect_right(xs,payload);left=right-1;fraction=(math.log2(payload)-math.log2(xs[left]))/(math.log2(xs[right])-math.log2(xs[left]));return ys[left]+fraction*(ys[right]-ys[left])
def _predict(edge_costs:list[float],flows:list[dict])->tuple[float,float,float]:
 coeff=contract()["graph_formula_contract"]["coefficients"];out=defaultdict(float);inc=defaultdict(float)
 for cost,flow in zip(edge_costs,flows):out[int(flow["sender_rank"])]+=cost;inc[int(flow["receiver_rank"])]+=cost
 m=max(edge_costs);b=max([*out.values(),*inc.values()]);pred=max(1.0,float(coeff["intercept_us"])+(float(coeff["beta_max"])-float(coeff["beta_min"]))*m+float(coeff["beta_min"])*b);return pred,m,b
def _metric(rows:list[dict],kind:str,value:str)->dict:
 actual=math.fsum(r["actual_concurrent_wave_us"] for r in rows);baseline=math.fsum(r["max_edge_baseline_us"] for r in rows);pred=math.fsum(r["graph_prediction_us"] for r in rows)
 return {"slice_type":kind,"slice_value":value,"points":len(rows),"max_edge_wape":math.fsum(abs(r["max_edge_baseline_us"]-r["actual_concurrent_wave_us"]) for r in rows)/actual,"graph_wape":math.fsum(abs(r["graph_prediction_us"]-r["actual_concurrent_wave_us"]) for r in rows)/actual,"max_edge_signed_bias":(baseline-actual)/actual,"graph_signed_bias":(pred-actual)/actual}
def build_analysis(plan:dict,raw:dict)->dict[str,Any]:
 if not raw["complete"]:raise RuntimeError("incomplete raw")
 curves=_curve_map();points=[];replica_points=[];spreads=[]
 for model in contract()["selected_models"]:
  for config in contract()["research_scope"]["fixed_configurations"]:
   for level in ("L1","L2","L3"):
    curve=curves[(model,level)]
    for vector in payload_vectors(model,config):
     replicas=[]
     for replica in (0,1):
      mid=f"{model}__{config.lower()}__{level.lower()}__r{replica}";selected=[next(r for r in rows if r["vector_id"]==vector["vector_id"]) for rows in raw["records"][mid].values()];concurrent=[float(r["concurrent_wave"]["wave_latency_us"]["median"]) for r in selected];solo_by_flow={str(f["flow_id"]):statistics.median(float(r["solo_flows"][str(f["flow_id"])]["wave_latency_us"]["median"]) for r in selected) for f in vector["flows"]};item={"model_id":model,"configuration":config,"topology_level":level,"vector_id":vector["vector_id"],"replica_id":replica,"measurement_id":mid,"repeat_count":len(selected),"concurrent_wave_us":statistics.median(concurrent),"matched_solo_max_us":max(solo_by_flow.values()),"concurrent_repeat_cv":_cv(concurrent)};replicas.append(item);replica_points.append(item)
     actual=max(r["concurrent_wave_us"] for r in replicas);edge=[_interpolate(curve,f["payload_bytes"]) for f in vector["flows"]];prediction,m,b=_predict(edge,vector["flows"]);spread=(max(r["concurrent_wave_us"] for r in replicas)-min(r["concurrent_wave_us"] for r in replicas))/actual if actual else 0.0;point={"model_id":model,"configuration":config,"topology_level":level,"vector_id":vector["vector_id"],"pages":"|".join(map(str,vector["pages"])),"flow_count":len(edge),"total_payload_bytes":sum(f["payload_bytes"] for f in vector["flows"]),"edge_costs_us":"|".join(f"{v:.12g}" for v in edge),"max_edge_baseline_us":m,"sum_edge_baseline_us":sum(edge),"busiest_endpoint_sum_us":b,"graph_prediction_us":prediction,"matched_solo_max_us":max(r["matched_solo_max_us"] for r in replicas),"actual_concurrent_wave_us":actual,"max_edge_absolute_error_us":abs(m-actual),"graph_absolute_error_us":abs(prediction-actual),"graph_signed_error_us":prediction-actual,"cross_replica_relative_spread":spread};points.append(point);spreads.append({"model_id":model,"configuration":config,"topology_level":level,"vector_id":vector["vector_id"],"replica0_us":replicas[0]["concurrent_wave_us"],"replica1_us":replicas[1]["concurrent_wave_us"],"official_us":actual,"relative_spread":spread,"above_threshold":spread>float(contract()["measurement_contract"]["cross_replica_relative_spread_diagnostic_threshold"])})
 groups=defaultdict(list)
 for r in points:
  for key in [("overall","all"),("model",r["model_id"]),("configuration",r["configuration"]),("topology",r["topology_level"]),("configuration_topology",f"{r['configuration']}/{r['topology_level']}"),("model_configuration",f"{r['model_id']}/{r['configuration']}")]:groups[key].append(r)
 metrics=[_metric(rows,k,v) for (k,v),rows in sorted(groups.items())];gate=contract()["zero_shot_gate"];overall=next(r for r in metrics if r["slice_type"]=="overall");models=[r for r in metrics if r["slice_type"]=="model"];configs=[r for r in metrics if r["slice_type"]=="configuration"];fine=[r for r in metrics if r["slice_type"]=="configuration_topology"]
 checks={"overall_wape":overall["graph_wape"]<=gate["overall_wape_max"],"each_model_wape":all(r["graph_wape"]<=gate["each_model_wape_max"] for r in models),"each_configuration_wape":all(r["graph_wape"]<=gate["each_configuration_wape_max"] for r in configs),"each_configuration_topology_wape":all(r["graph_wape"]<=gate["each_configuration_topology_wape_max"] for r in fine),"overall_signed_bias":abs(overall["graph_signed_bias"])<=gate["overall_absolute_signed_bias_max"],"each_model_signed_bias":all(abs(r["graph_signed_bias"])<=gate["each_model_absolute_signed_bias_max"] for r in models),"each_configuration_signed_bias":all(abs(r["graph_signed_bias"])<=gate["each_configuration_absolute_signed_bias_max"] for r in configs),"each_configuration_topology_signed_bias":all(abs(r["graph_signed_bias"])<=gate["each_configuration_topology_absolute_signed_bias_max"] for r in fine),"all_predictions_positive":all(r["graph_prediction_us"]>0 for r in points),"strictly_improves_max_edge_overall":overall["graph_wape"]<overall["max_edge_wape"],"strictly_improves_max_edge_each_configuration":all(r["graph_wape"]<r["max_edge_wape"] for r in configs)};passed=all(checks.values())
 return {"points":points,"replica_points":replica_points,"spreads":spreads,"metrics":metrics,"decision":{"scientific_outcome":"MULTIFLOW_GRAPH_ZERO_SHOT_PASS" if passed else "MULTIFLOW_GRAPH_ZERO_SHOT_FAIL_RETAIN_FOR_DEVELOPMENT","zero_shot_gate_pass":passed,"checks":checks,"thresholds":gate,"training_performed":False,"recalibration_performed":False,"phase64_labels_used_for_fitting":False}}
