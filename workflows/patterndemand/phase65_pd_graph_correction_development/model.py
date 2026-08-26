#!/usr/bin/env python3
"""Deterministic lightweight graph-correction candidates for Phase65."""
from __future__ import annotations
import csv,math,re
from collections import defaultdict
from pathlib import Path
from typing import Any

CANDIDATES=[
 {"candidate_id":"global_affine_graph","complexity_rank":1,"scope":[]},
 {"candidate_id":"graph_class_affine_graph","complexity_rank":2,"scope":["graph_class"]},
 {"candidate_id":"model_affine_graph","complexity_rank":3,"scope":["model_id"]},
 {"candidate_id":"model_graph_class_affine_graph","complexity_rank":4,"scope":["model_id","graph_class"]},
 {"candidate_id":"model_configuration_affine_graph","complexity_rank":5,"scope":["model_id","configuration"]},
]
BASELINES=("max_edge","r61_graph_extension")
NUMERIC_FIELDS=("flow_count","total_payload_bytes","max_edge_baseline_us","sum_edge_baseline_us","busiest_endpoint_sum_us","graph_prediction_us","matched_solo_max_us","actual_concurrent_wave_us")

def graph_class(configuration:str)->str:
 if configuration in ("P1D4","P4D1"):return "shared_endpoint_star"
 if configuration=="P2D2_MATCHING":return "matching"
 if configuration=="P2D2_ALL_TO_ALL":return "all_to_all"
 raise RuntimeError({"unknown_configuration":configuration})
def read_points(path:Path)->list[dict[str,Any]]:
 with path.open(encoding="utf-8",newline="") as stream:rows=list(csv.DictReader(stream))
 for row in rows:
  for field in NUMERIC_FIELDS:row[field]=float(row[field])
  match=re.search(r"__v(\d+)$",row["vector_id"])
  if match is None:raise RuntimeError({"invalid_vector_id":row["vector_id"]})
  row["vector_index"]=int(match.group(1));row["graph_class"]=graph_class(row["configuration"])
  m=float(row["max_edge_baseline_us"]);b=float(row["busiest_endpoint_sum_us"]);s=float(row["sum_edge_baseline_us"])
  tolerance=1e-12*max(1.0,abs(m),abs(b),abs(s))
  if not (m>0 and b+tolerance>=m and s+tolerance>=b and float(row["actual_concurrent_wave_us"])>0):raise RuntimeError({"invalid_graph_costs":row["vector_id"],"M":m,"B":b,"S":s})
 return rows
def features(row:dict[str,Any])->list[float]:
 m=float(row["max_edge_baseline_us"]);b=float(row["busiest_endpoint_sum_us"]);s=float(row["sum_edge_baseline_us"]);return [1.0,m,max(0.0,b-m),max(0.0,s-b)]
def _scope_key(row:dict[str,Any],scope:list[str])->str:return "__global__" if not scope else "|".join(str(row[field]) for field in scope)
def _solve(matrix:list[list[float]],vector:list[float])->list[float]:
 n=len(vector);aug=[list(matrix[i])+[float(vector[i])] for i in range(n)]
 for column in range(n):
  pivot=max(range(column,n),key=lambda i:abs(aug[i][column]))
  if abs(aug[pivot][column])<1e-15:raise RuntimeError("singular Phase65 normal equation after ridge")
  aug[column],aug[pivot]=aug[pivot],aug[column];scale=aug[column][column];aug[column]=[value/scale for value in aug[column]]
  for i in range(n):
   if i==column:continue
   factor=aug[i][column];aug[i]=[value-factor*pivot_value for value,pivot_value in zip(aug[i],aug[column])]
 return [aug[i][-1] for i in range(n)]
def _fit_group(rows:list[dict[str,Any]],ridge_relative:float)->dict[str,Any]:
 x=[features(row) for row in rows];y=[float(row["actual_concurrent_wave_us"]) for row in rows];n=len(x[0]);normal=[[math.fsum(row[i]*row[j] for row in x) for j in range(n)] for i in range(n)];target=[math.fsum(row[i]*value for row,value in zip(x,y)) for i in range(n)];trace=math.fsum(normal[i][i] for i in range(n));ridge=trace*ridge_relative/n
 for i in range(n):normal[i][i]+=ridge
 values=_solve(normal,target)
 return {"intercept_us":values[0],"beta_M":values[1],"beta_busy":values[2],"beta_nonbusy":values[3],"ridge_absolute":ridge,"training_rows":len(rows)}
def fit_model(rows:list[dict[str,Any]],candidate:dict[str,Any],ridge_relative:float=1e-10,floor_us:float=1.0)->dict[str,Any]:
 grouped:dict[str,list[dict[str,Any]]]=defaultdict(list)
 for row in rows:grouped[_scope_key(row,candidate["scope"])].append(row)
 groups={key:_fit_group(values,ridge_relative) for key,values in sorted(grouped.items())}
 return {"schema_version":"phase65-multiflow-graph-correction-v1","candidate_id":candidate["candidate_id"],"complexity_rank":candidate["complexity_rank"],"family":"affine_graph","scope":candidate["scope"],"feature_names":["intercept","M","B_minus_M","S_minus_B"],"ridge_relative":ridge_relative,"prediction_floor_us":floor_us,"required_runtime_inputs":["model_id","configuration","max_edge_baseline_us","busiest_endpoint_sum_us","sum_edge_baseline_us"],"groups":groups}
def predict(model:dict[str,Any],row:dict[str,Any])->float:
 key=_scope_key(row,list(model["scope"]));co=model["groups"][key];x=features(row);value=float(co["intercept_us"])+float(co["beta_M"])*x[1]+float(co["beta_busy"])*x[2]+float(co["beta_nonbusy"])*x[3];return max(float(model["prediction_floor_us"]),value)
def baseline_value(candidate_id:str,row:dict[str,Any])->float:
 if candidate_id=="max_edge":return float(row["max_edge_baseline_us"])
 if candidate_id=="r61_graph_extension":return float(row["graph_prediction_us"])
 raise RuntimeError({"unknown_baseline":candidate_id})
def prediction_rows(rows:list[dict[str,Any]],candidate_id:str,values:list[float],scheme:str,fold:str)->list[dict[str,Any]]:
 output=[]
 for row,value in zip(rows,values):
  actual=float(row["actual_concurrent_wave_us"]);output.append({"candidate_id":candidate_id,"oof_scheme":scheme,"fold":fold,"model_id":row["model_id"],"configuration":row["configuration"],"graph_class":row["graph_class"],"topology_level":row["topology_level"],"vector_id":row["vector_id"],"vector_index":row["vector_index"],"max_edge_baseline_us":row["max_edge_baseline_us"],"busiest_endpoint_sum_us":row["busiest_endpoint_sum_us"],"sum_edge_baseline_us":row["sum_edge_baseline_us"],"predicted_concurrent_wave_us":value,"actual_concurrent_wave_us":actual,"absolute_error_us":abs(value-actual),"signed_error_us":value-actual})
 return output
def slice_metrics(predictions:list[dict[str,Any]],candidate_id:str,scheme:str)->list[dict[str,Any]]:
 groups:dict[tuple[str,str],list[dict[str,Any]]]=defaultdict(list)
 for row in predictions:
  keys=[("overall","all"),("model",row["model_id"]),("configuration",row["configuration"]),("topology",row["topology_level"]),("configuration_topology",f"{row['configuration']}/{row['topology_level']}"),("model_configuration",f"{row['model_id']}/{row['configuration']}")]
  for key in keys:groups[key].append(row)
 output=[]
 for (kind,value),rows in sorted(groups.items()):
  actual=math.fsum(float(row["actual_concurrent_wave_us"]) for row in rows);predicted=math.fsum(float(row["predicted_concurrent_wave_us"]) for row in rows);absolute=math.fsum(float(row["absolute_error_us"]) for row in rows);output.append({"candidate_id":candidate_id,"oof_scheme":scheme,"slice_type":kind,"slice_value":value,"points":len(rows),"wape":absolute/actual,"signed_bias":(predicted-actual)/actual})
 return output
def _fold_values(rows:list[dict[str,Any]],scheme:str)->list[Any]:
 field="vector_index" if scheme=="payload" else "topology_level";return sorted({row[field] for row in rows})
def evaluate_oof(rows:list[dict[str,Any]],candidate:dict[str,Any],scheme:str)->dict[str,Any]:
 field="vector_index" if scheme=="payload" else "topology_level";predictions=[]
 for held_value in _fold_values(rows,scheme):
  training=[row for row in rows if row[field]!=held_value];held=[row for row in rows if row[field]==held_value];model=fit_model(training,candidate);predictions.extend(prediction_rows(held,candidate["candidate_id"],[predict(model,row) for row in held],scheme,str(held_value)))
 predictions.sort(key=lambda row:(row["model_id"],row["configuration"],row["topology_level"],row["vector_index"]));return {"predictions":predictions,"slices":slice_metrics(predictions,candidate["candidate_id"],scheme),"folds":len(_fold_values(rows,scheme))}
def baseline_evaluation(rows:list[dict[str,Any]],candidate_id:str,scheme:str)->dict[str,Any]:
 values=[baseline_value(candidate_id,row) for row in rows];predictions=prediction_rows(rows,candidate_id,values,scheme,"not_applicable");return {"predictions":predictions,"slices":slice_metrics(predictions,candidate_id,scheme),"folds":0}
def _slice(slices:list[dict[str,Any]],kind:str)->list[dict[str,Any]]:return [row for row in slices if row["slice_type"]==kind]
def gate_scheme(candidate_predictions:list[dict[str,Any]],candidate_slices:list[dict[str,Any]],baseline_slices:dict[str,list[dict[str,Any]]],contract:dict[str,Any])->dict[str,Any]:
 a=contract["acceptance_gate"];overall=_slice(candidate_slices,"overall")[0];models=_slice(candidate_slices,"model");configs=_slice(candidate_slices,"configuration");fine=_slice(candidate_slices,"configuration_topology");baseline_overall={name:_slice(values,"overall")[0] for name,values in baseline_slices.items()};baseline_configs={name:{row["slice_value"]:row for row in _slice(values,"configuration")} for name,values in baseline_slices.items()}
 checks={"overall_wape":float(overall["wape"])<=a["each_oof_scheme_overall_wape_max"],"each_model_wape":all(float(row["wape"])<=a["each_oof_scheme_each_model_wape_max"] for row in models),"each_configuration_wape":all(float(row["wape"])<=a["each_oof_scheme_each_configuration_wape_max"] for row in configs),"each_configuration_topology_wape":all(float(row["wape"])<=a["each_oof_scheme_each_configuration_topology_wape_max"] for row in fine),"overall_bias":abs(float(overall["signed_bias"]))<=a["each_oof_scheme_overall_absolute_signed_bias_max"],"each_model_bias":all(abs(float(row["signed_bias"]))<=a["each_oof_scheme_each_model_absolute_signed_bias_max"] for row in models),"each_configuration_bias":all(abs(float(row["signed_bias"]))<=a["each_oof_scheme_each_configuration_absolute_signed_bias_max"] for row in configs),"each_configuration_topology_bias":all(abs(float(row["signed_bias"]))<=a["each_oof_scheme_each_configuration_topology_absolute_signed_bias_max"] for row in fine),"positive_predictions":all(float(row["predicted_concurrent_wave_us"])>0 for row in candidate_predictions),"improves_both_baselines_overall":all(float(overall["wape"])<float(row["wape"]) for row in baseline_overall.values()),"improves_best_baseline_each_configuration":all(float(row["wape"])<min(float(baseline_configs[name][row["slice_value"]]["wape"]) for name in BASELINES) for row in configs)}
 return {"pass":all(checks.values()),"checks":checks,"overall_wape":float(overall["wape"]),"overall_signed_bias":float(overall["signed_bias"]),"max_model_wape":max(float(row["wape"]) for row in models),"max_configuration_wape":max(float(row["wape"]) for row in configs),"max_configuration_topology_wape":max(float(row["wape"]) for row in fine),"max_model_absolute_signed_bias":max(abs(float(row["signed_bias"])) for row in models),"max_configuration_absolute_signed_bias":max(abs(float(row["signed_bias"])) for row in configs),"max_configuration_topology_absolute_signed_bias":max(abs(float(row["signed_bias"])) for row in fine)}
def evaluate_candidates(rows:list[dict[str,Any]],contract:dict[str,Any])->dict[str,Any]:
 evaluations={};all_predictions=[];all_slices=[]
 for baseline in BASELINES:
  evaluations[baseline]={}
  for scheme in ("payload","topology"):
   value=baseline_evaluation(rows,baseline,scheme);evaluations[baseline][scheme]=value;all_predictions.extend(value["predictions"]);all_slices.extend(value["slices"])
 candidate_values=[]
 for candidate in CANDIDATES:
  item={**candidate,"schemes":{}}
  for scheme in ("payload","topology"):
   value=evaluate_oof(rows,candidate,scheme);baselines={name:evaluations[name][scheme]["slices"] for name in BASELINES};value["gate"]=gate_scheme(value["predictions"],value["slices"],baselines,contract);item["schemes"][scheme]=value;all_predictions.extend(value["predictions"]);all_slices.extend(value["slices"])
  item["target_guard"]=all(item["schemes"][scheme]["gate"]["pass"] for scheme in ("payload","topology"));candidate_values.append(item)
 selected=next((item for item in candidate_values if item["target_guard"]),None)
 return {"baselines":evaluations,"candidates":candidate_values,"selected":selected,"all_oof_predictions":all_predictions,"all_oof_slices":all_slices,"payload_folds":10,"topology_folds":3}
def candidate_metric_rows(evaluation:dict[str,Any])->list[dict[str,Any]]:
 output=[];selected_id=None if evaluation["selected"] is None else evaluation["selected"]["candidate_id"]
 for baseline in BASELINES:
  row={"candidate_id":baseline,"complexity_rank":0,"scope":"baseline","target_guard":False,"selected":False}
  for scheme in ("payload","topology"):
   slices=evaluation["baselines"][baseline][scheme]["slices"];overall=_slice(slices,"overall")[0];row[f"{scheme}_overall_wape"]=overall["wape"];row[f"{scheme}_overall_signed_bias"]=overall["signed_bias"];row[f"{scheme}_max_model_wape"]=max(v["wape"] for v in _slice(slices,"model"));row[f"{scheme}_max_configuration_wape"]=max(v["wape"] for v in _slice(slices,"configuration"));row[f"{scheme}_max_configuration_topology_wape"]=max(v["wape"] for v in _slice(slices,"configuration_topology"))
  output.append(row)
 for candidate in evaluation["candidates"]:
  row={"candidate_id":candidate["candidate_id"],"complexity_rank":candidate["complexity_rank"],"scope":"+".join(candidate["scope"]) or "global","target_guard":candidate["target_guard"],"selected":candidate["candidate_id"]==selected_id}
  for scheme in ("payload","topology"):
   gate=candidate["schemes"][scheme]["gate"]
   for key,value in gate.items():
    if key!="checks":row[f"{scheme}_{key}"]=value
  output.append(row)
 return output
