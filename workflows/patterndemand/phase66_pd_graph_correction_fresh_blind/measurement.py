#!/usr/bin/env python3
"""Phase66 raw validation and deterministic frozen-model fresh-blind evaluation."""
from __future__ import annotations
import bisect,json,math,statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from contracts import contract,file_sha,frozen_model,iteration_counts,layout_by_id,load_json,measurement_by_id,payload_vectors,r61_baseline,validate_plan

ROOT=Path(__file__).resolve().parents[3]
def _close(left:float,right:float)->bool:return math.isclose(left,right,rel_tol=1e-9,abs_tol=1e-8)
def _cv(values:list[float])->float:return statistics.stdev(values)/statistics.mean(values) if len(values)>1 and statistics.mean(values) else 0.0
def _read_jsonl(path:Path)->list[dict]:
 rows=[]
 for number,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
  if not line.strip():raise RuntimeError({"blank_line":str(path),"line":number})
  try:value=json.loads(line)
  except json.JSONDecodeError as exc:raise RuntimeError({"invalid_jsonl":str(path),"line":number}) from exc
  if not isinstance(value,dict):raise RuntimeError({"non_object":str(path),"line":number})
  rows.append(value)
 return rows
def _validate_mode(mode:dict,flow_ids:list[int],timed:int)->bool:
 samples=mode.get("flow_latency_samples_us",{});summaries=mode.get("flow_latency_us",{});waves=mode.get("wave_latency_samples_us",[]);skews=mode.get("sender_start_skew_samples_us",[]);keys={str(value) for value in flow_ids}
 if set(samples)!=keys or set(summaries)!=keys or len(waves)!=timed or len(skews)!=timed or any(len(samples[key])!=timed for key in keys):return False
 values=[float(value) for group in samples.values() for value in group]+[float(value) for value in waves]+[float(value) for value in skews]
 if any(not math.isfinite(value) or value<0 for value in values) or any(float(value)<=0 for group in samples.values() for value in group) or any(float(value)<=0 for value in waves):return False
 if any(not _close(float(summaries[key].get("median",-1)),statistics.median(float(value) for value in samples[key])) for key in keys):return False
 for index in range(timed):
  if not _close(float(waves[index]),max(float(samples[str(flow_id)][index]) for flow_id in flow_ids)):return False
 return _close(float(mode.get("wave_latency_us",{}).get("median",-1)),statistics.median(float(value) for value in waves)) and mode.get("return_codes_all_zero") is True and mode.get("data_validation_pass") is True
def validate_raw(plan:dict,raw_dir:Path,*,require_complete:bool)->dict[str,Any]:
 audit=validate_plan(plan);spec=contract();measurement_contract=spec["measurement_contract"];raw_dir=raw_dir.expanduser().resolve()
 if not raw_dir.is_dir():raise RuntimeError(f"raw missing: {raw_dir}")
 files=sorted(path for path in raw_dir.rglob("*") if path.is_file());unexpected=[str(path.relative_to(raw_dir)) for path in files if path.suffix!=".jsonl"]
 if unexpected:raise RuntimeError({"unexpected_raw_files":unexpected})
 by:dict[str,dict[int,list[dict]]]=defaultdict(dict);manifest=[]
 for path in files:
  relative=path.relative_to(raw_dir);parts=relative.parts
  if len(parts)!=2 or not parts[1].startswith("repeat_") or not parts[1].endswith(".jsonl"):raise RuntimeError({"raw_path":str(relative)})
  try:repeat_id=int(parts[1][7:-6])
  except ValueError as exc:raise RuntimeError({"repeat_path":str(relative)}) from exc
  if parts[1]!=f"repeat_{repeat_id:02d}.jsonl" or repeat_id in by[parts[0]]:raise RuntimeError({"duplicate_repeat":str(relative)})
  rows=_read_jsonl(path);by[parts[0]][repeat_id]=rows;manifest.append({"path":str(relative),"sha256":file_sha(path),"bytes":path.stat().st_size,"records":len(rows)})
 expected_ids={measurement["measurement_id"] for measurement in plan["measurements"]}
 if set(by)-expected_ids:raise RuntimeError({"unknown_measurements":sorted(set(by)-expected_ids)})
 missing=[];needs=[];variance=[];quality=[];record_count=0;allowed=[int(measurement_contract["minimum_independent_repeats"]),int(measurement_contract["minimum_independent_repeats"])+int(measurement_contract["extra_repeats_per_round"]),int(measurement_contract["maximum_independent_repeats"])]
 for measurement_id in sorted(expected_ids):
  measurement=measurement_by_id(plan,measurement_id);vectors=payload_vectors(measurement["model_id"],measurement["configuration"]);vector_map={vector["vector_id"]:vector for vector in vectors};repeats=by.get(measurement_id,{});repeat_ids=sorted(repeats)
  if repeat_ids and repeat_ids!=list(range(len(repeat_ids))):raise RuntimeError({"noncontiguous_repeats":measurement_id,"actual":repeat_ids})
  if len(repeat_ids)>allowed[-1]:raise RuntimeError({"too_many_repeats":measurement_id})
  if len(repeat_ids)<allowed[0]:missing.append({"measurement_id":measurement_id,"have":len(repeat_ids),"need":allowed[0]-len(repeat_ids),"target_total":allowed[0]})
  elif len(repeat_ids) not in allowed:
   target=next(value for value in allowed if value>len(repeat_ids));missing.append({"measurement_id":measurement_id,"have":len(repeat_ids),"need":target-len(repeat_ids),"target_total":target})
  medians=defaultdict(list)
  for repeat_id,rows in sorted(repeats.items()):
   record_count+=len(rows);vector_ids=[row.get("vector_id") for row in rows]
   if len(vector_ids)!=10 or set(vector_ids)!=set(vector_map) or len(set(vector_ids))!=10:raise RuntimeError({"vector_coverage":measurement_id,"repeat":repeat_id,"ids":vector_ids})
   for row in rows:
    vector=vector_map[row["vector_id"]];flows=vector["flows"];warmup,timed=iteration_counts(sum(flow["payload_bytes"] for flow in flows));endpoints=row.get("runtime_endpoints",[])
    endpoint_ok=len(endpoints)==measurement["world_size"] and all(int(endpoints[index].get("rank",-1))==index and endpoints[index].get("role")==measurement["ranks"][index]["role"] and endpoints[index].get("expected_host")==measurement["ranks"][index]["host"] and int(endpoints[index].get("physical_gpu",-1))==int(measurement["ranks"][index]["physical_gpu"]) and endpoints[index].get("ib_device")==measurement["ranks"][index]["ib_device"] and endpoints[index].get("mooncake_protocol")=="rdma" and endpoints[index].get("with_nvidia_peermem")=="0" for index in range(measurement["world_size"]))
    solo=row.get("solo_flows",{});checks={"schema":row.get("schema_version")=="phase66-mooncake-multiflow-raw-v1","binding":row.get("workflow_commit")==plan["workflow_commit"] and row.get("plan_sha256")==audit["plan_sha256"] and row.get("measurement_id")==measurement_id and row.get("measurement_sha256")==measurement["measurement_sha256"],"identity":row.get("model_id")==measurement["model_id"] and row.get("configuration")==measurement["configuration"] and row.get("topology_level")==measurement["topology_level"] and row.get("replica_id")==measurement["replica_id"] and row.get("placement_id")==measurement["placement_id"] and row.get("repeat_id")==repeat_id,"vector":row.get("pages")==vector["pages"] and row.get("flows")==flows,"layout":row.get("descriptor_layout")==layout_by_id(measurement["model_id"])["descriptor_layout"] and int(row.get("descriptor_count",-1))==int(layout_by_id(measurement["model_id"])["descriptor_count"]),"operation":row.get("op")=="MooncakeTransferEngine.batch_transfer_sync" and row.get("transport")=="rdma" and row.get("wave_admission")=="gloo_barrier_then_all_graph_edges_synchronous_release","iterations":row.get("warmup_iterations")==warmup and row.get("timed_iterations")==timed,"solo_keys":set(solo)=={str(flow["flow_id"]) for flow in flows},"solo_modes":all(_validate_mode(solo.get(str(flow["flow_id"]),{}),[flow["flow_id"]],timed) for flow in flows),"concurrent":_validate_mode(row.get("concurrent_wave",{}),[flow["flow_id"] for flow in flows],timed),"endpoints":endpoint_ok,"timestamp":isinstance(row.get("timestamp_utc"),str) and row["timestamp_utc"]>=plan["generated_at_utc"]}
    if not all(checks.values()):raise RuntimeError({"invalid_raw_record":measurement_id,"repeat":repeat_id,"vector":row.get("vector_id"),"checks":checks})
    medians[row["vector_id"]].append(float(row["concurrent_wave"]["wave_latency_us"]["median"]))
  cvs={key:_cv(values) for key,values in sorted(medians.items())};maximum_cv=max(cvs.values(),default=0.0)
  if len(repeat_ids) in allowed and maximum_cv>float(measurement_contract["repeat_median_cv_threshold"]):
   if len(repeat_ids)<allowed[-1]:needs.append({"measurement_id":measurement_id,"have":len(repeat_ids),"add":int(measurement_contract["extra_repeats_per_round"]),"max_cv":maximum_cv})
   else:variance.append({"measurement_id":measurement_id,"repeats":len(repeat_ids),"max_cv":maximum_cv})
  quality.append({"measurement_id":measurement_id,"model_id":measurement["model_id"],"configuration":measurement["configuration"],"topology_level":measurement["topology_level"],"replica_id":measurement["replica_id"],"world_size":measurement["world_size"],"flow_count":measurement["flow_count"],"repeat_count":len(repeat_ids),"record_count":sum(len(values) for values in repeats.values()),"max_repeat_median_cv":maximum_cv,"repeat_median_cv":cvs})
 complete=not missing and not needs
 if require_complete and not complete:raise RuntimeError({"raw_not_complete":True,"missing":missing,"needs_extra":needs})
 return {"schema_version":"phase66-raw-audit-v1","raw_dir":str(raw_dir),"plan_sha256":audit["plan_sha256"],"complete":complete,"counts":{"files":len(manifest),"records":record_count,"measurements_with_data":len(by),"expected_measurements":48},"missing":missing,"needs_extra":needs,"final_runtime_variance":variance,"files":manifest,"measurements":quality,"records":by}
def _curve_map()->dict:
 rows=load_json(ROOT/"experiment-results/phase51_pd_l1_l3_physical_curve_library/curves/pd_mooncake_physical_curves.json")["curves"];return {(row["model_id"],row["topology_level"]):row for row in rows}
def _interpolate(curve:dict,payload:int)->float:
 knots=curve["knots"];xs=[int(row["payload_bytes"]) for row in knots];ys=[float(row["official_latency_us"]) for row in knots]
 if payload<xs[0] or payload>xs[-1]:raise RuntimeError({"payload_outside_curve":payload})
 if payload in xs:return ys[xs.index(payload)]
 right=bisect.bisect_right(xs,payload);left=right-1;fraction=(math.log2(payload)-math.log2(xs[left]))/(math.log2(xs[right])-math.log2(xs[left]));return ys[left]+fraction*(ys[right]-ys[left])
def _graph_features(edge_costs:list[float],flows:list[dict])->tuple[float,float,float]:
 outbound=defaultdict(float);inbound=defaultdict(float)
 for cost,flow in zip(edge_costs,flows):outbound[int(flow["sender_rank"])]+=cost;inbound[int(flow["receiver_rank"])]+=cost
 return max(edge_costs),max([*outbound.values(),*inbound.values()]),math.fsum(edge_costs)
def _predict_phase65(model_id:str,configuration:str,m:float,b:float,s:float)->float:
 model=frozen_model();coefficients=model["groups"][f"{model_id}|{configuration}"];value=float(coefficients["intercept_us"])+float(coefficients["beta_M"])*m+float(coefficients["beta_busy"])*max(0.0,b-m)+float(coefficients["beta_nonbusy"])*max(0.0,s-b);return max(float(model["prediction_floor_us"]),value)
def _predict_r61(m:float,b:float)->float:
 model=r61_baseline();coefficients=model["groups"]["__global__"];return max(float(model["prediction_floor_us"]),float(coefficients["intercept_us"])+(float(coefficients["beta_max"])-float(coefficients["beta_min"]))*m+float(coefficients["beta_min"])*b)
def _metric(rows:list[dict],kind:str,value:str)->dict:
 actual=math.fsum(row["actual_concurrent_wave_us"] for row in rows);max_edge=math.fsum(row["max_edge_baseline_us"] for row in rows);r61=math.fsum(row["r61_graph_prediction_us"] for row in rows);phase65=math.fsum(row["phase65_prediction_us"] for row in rows)
 return {"slice_type":kind,"slice_value":value,"points":len(rows),"max_edge_wape":math.fsum(abs(row["max_edge_baseline_us"]-row["actual_concurrent_wave_us"]) for row in rows)/actual,"r61_graph_wape":math.fsum(abs(row["r61_graph_prediction_us"]-row["actual_concurrent_wave_us"]) for row in rows)/actual,"phase65_wape":math.fsum(abs(row["phase65_prediction_us"]-row["actual_concurrent_wave_us"]) for row in rows)/actual,"max_edge_signed_bias":(max_edge-actual)/actual,"r61_graph_signed_bias":(r61-actual)/actual,"phase65_signed_bias":(phase65-actual)/actual}
def build_analysis(plan:dict,raw:dict)->dict[str,Any]:
 if not raw["complete"]:raise RuntimeError("incomplete raw")
 curves=_curve_map();points=[];replica_points=[];spreads=[]
 for model in contract()["selected_models"]:
  for configuration in contract()["research_scope"]["fixed_configurations"]:
   for level in ("L1","L2","L3"):
    curve=curves[(model,level)]
    for vector in payload_vectors(model,configuration):
     replicas=[]
     for replica in (0,1):
      measurement_id=f"{model}__{configuration.lower()}__{level.lower()}__r{replica}";selected=[next(row for row in rows if row["vector_id"]==vector["vector_id"]) for rows in raw["records"][measurement_id].values()];concurrent=[float(row["concurrent_wave"]["wave_latency_us"]["median"]) for row in selected];solo_by_flow={str(flow["flow_id"]):statistics.median(float(row["solo_flows"][str(flow["flow_id"])]["wave_latency_us"]["median"]) for row in selected) for flow in vector["flows"]};item={"model_id":model,"configuration":configuration,"topology_level":level,"vector_id":vector["vector_id"],"replica_id":replica,"measurement_id":measurement_id,"repeat_count":len(selected),"concurrent_wave_us":statistics.median(concurrent),"matched_solo_max_us":max(solo_by_flow.values()),"concurrent_repeat_cv":_cv(concurrent)};replicas.append(item);replica_points.append(item)
     actual=max(row["concurrent_wave_us"] for row in replicas);edge_costs=[_interpolate(curve,flow["payload_bytes"]) for flow in vector["flows"]];m,b,s=_graph_features(edge_costs,vector["flows"]);phase65=_predict_phase65(model,configuration,m,b,s);r61=_predict_r61(m,b);spread=(max(row["concurrent_wave_us"] for row in replicas)-min(row["concurrent_wave_us"] for row in replicas))/actual if actual else 0.0;point={"model_id":model,"configuration":configuration,"topology_level":level,"vector_id":vector["vector_id"],"pages":"|".join(map(str,vector["pages"])),"flow_count":len(edge_costs),"total_payload_bytes":sum(flow["payload_bytes"] for flow in vector["flows"]),"edge_costs_us":"|".join(f"{cost:.12g}" for cost in edge_costs),"max_edge_baseline_us":m,"sum_edge_baseline_us":s,"busiest_endpoint_sum_us":b,"r61_graph_prediction_us":r61,"phase65_prediction_us":phase65,"matched_solo_max_us":max(row["matched_solo_max_us"] for row in replicas),"actual_concurrent_wave_us":actual,"max_edge_absolute_error_us":abs(m-actual),"r61_graph_absolute_error_us":abs(r61-actual),"phase65_absolute_error_us":abs(phase65-actual),"phase65_signed_error_us":phase65-actual,"cross_replica_relative_spread":spread};points.append(point);spreads.append({"model_id":model,"configuration":configuration,"topology_level":level,"vector_id":vector["vector_id"],"replica0_us":replicas[0]["concurrent_wave_us"],"replica1_us":replicas[1]["concurrent_wave_us"],"official_us":actual,"relative_spread":spread,"above_threshold":spread>float(contract()["measurement_contract"]["cross_replica_relative_spread_diagnostic_threshold"])})
 groups=defaultdict(list)
 for row in points:
  for key in [("overall","all"),("model",row["model_id"]),("configuration",row["configuration"]),("topology",row["topology_level"]),("configuration_topology",f"{row['configuration']}/{row['topology_level']}"),("model_configuration",f"{row['model_id']}/{row['configuration']}")]:groups[key].append(row)
 metrics=[_metric(rows,kind,value) for (kind,value),rows in sorted(groups.items())];gate=contract()["fresh_blind_acceptance_gate"];overall=next(row for row in metrics if row["slice_type"]=="overall");models=[row for row in metrics if row["slice_type"]=="model"];configurations=[row for row in metrics if row["slice_type"]=="configuration"];fine=[row for row in metrics if row["slice_type"]=="configuration_topology"]
 checks={"overall_wape":overall["phase65_wape"]<=gate["overall_wape_max"],"each_model_wape":all(row["phase65_wape"]<=gate["each_model_wape_max"] for row in models),"each_configuration_wape":all(row["phase65_wape"]<=gate["each_configuration_wape_max"] for row in configurations),"each_configuration_topology_wape":all(row["phase65_wape"]<=gate["each_configuration_topology_wape_max"] for row in fine),"overall_signed_bias":abs(overall["phase65_signed_bias"])<=gate["overall_absolute_signed_bias_max"],"each_model_signed_bias":all(abs(row["phase65_signed_bias"])<=gate["each_model_absolute_signed_bias_max"] for row in models),"each_configuration_signed_bias":all(abs(row["phase65_signed_bias"])<=gate["each_configuration_absolute_signed_bias_max"] for row in configurations),"each_configuration_topology_signed_bias":all(abs(row["phase65_signed_bias"])<=gate["each_configuration_topology_absolute_signed_bias_max"] for row in fine),"all_predictions_positive":all(row["phase65_prediction_us"]>0 for row in points),"strictly_improves_both_baselines_overall":overall["phase65_wape"]<overall["max_edge_wape"] and overall["phase65_wape"]<overall["r61_graph_wape"],"strictly_improves_best_baseline_each_configuration":all(row["phase65_wape"]<min(row["max_edge_wape"],row["r61_graph_wape"]) for row in configurations)};passed=all(checks.values());outcome="MULTIFLOW_GRAPH_CORRECTION_FRESH_BLIND_PASS" if passed else "MULTIFLOW_GRAPH_CORRECTION_FRESH_BLIND_FAIL_RETAIN_AS_BLIND_EVIDENCE"
 return {"points":points,"replica_points":replica_points,"spreads":spreads,"metrics":metrics,"decision":{"scientific_outcome":outcome,"fresh_blind_gate_pass":passed,"checks":checks,"thresholds":gate,"training_performed":False,"recalibration_performed":False,"threshold_tuning_performed":False,"phase66_labels_used_for_fitting":False}}
