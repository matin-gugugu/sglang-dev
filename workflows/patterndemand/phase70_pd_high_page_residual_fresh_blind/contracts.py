#!/usr/bin/env python3
"""Phase70 reserved payload, frozen model, fresh placement and plan contracts."""
from __future__ import annotations
import copy,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
from typing import Any

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
P51=ROOT/"experiment-results/phase51_pd_l1_l3_physical_curve_library"
P64=ROOT/"experiment-results/phase64_pd_multiflow_graph_zero_shot"
P65=ROOT/"experiment-results/phase65_pd_graph_correction_development"
P66=ROOT/"experiment-results/phase66_pd_graph_correction_fresh_blind"
P67=ROOT/"experiment-results/phase67_pd_graph_page_shape_refinement"
P68=ROOT/"experiment-results/phase68_pd_graph_page_shape_fresh_blind"
P69=ROOT/"experiment-results/phase69_pd_high_page_residual_refinement"

def load_json(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))
def file_sha(path:Path)->str:
 digest=hashlib.sha256()
 with path.open("rb") as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
 return digest.hexdigest()
def canonical_sha(value:Any)->str:return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def contract()->dict[str,Any]:return load_json(HERE/"experiment.json")
def reserved_grid()->dict[str,Any]:return load_json(P69/"contracts/phase70_reserved_blind_grid.json")
def phase64_graph_grid()->dict[str,Any]:return load_json(P64/"contracts/graph_payload_grid.json")
def frozen_model()->dict[str,Any]:return load_json(P69/"model/multiflow_high_page_residual.json")
def frozen_r67_model()->dict[str,Any]:return load_json(P67/"model/multiflow_graph_page_correction.json")
def r61_baseline()->dict[str,Any]:return load_json(P64/"contracts/frozen_contention_correction.json")
def r65_baseline()->dict[str,Any]:return load_json(P65/"model/multiflow_graph_correction.json")
def layouts(spec:dict[str,Any]|None=None)->list[dict[str,Any]]:
 spec=spec or contract();source=load_json(P51/"contracts/model_transfer_layouts.json")["layouts"];by={row["model_id"]:row for row in source};return [copy.deepcopy(by[model]) for model in spec["selected_models"]]
def layout_by_id(model_id:str)->dict[str,Any]:
 rows=[row for row in layouts() if row["model_id"]==model_id]
 if len(rows)!=1:raise RuntimeError({"unknown_model":model_id})
 return rows[0]
def graph(configuration:str)->dict[str,Any]:
 structure=phase64_graph_grid()["configurations"].get(configuration);pages=reserved_grid()["configurations"].get(configuration)
 if not isinstance(structure,dict) or not isinstance(pages,list):raise RuntimeError({"unknown_configuration":configuration})
 return {"roles":copy.deepcopy(structure["roles"]),"edges":copy.deepcopy(structure["edges"]),"page_vectors":copy.deepcopy(pages)}
def _page_sizes(layout:dict[str,Any],page:int)->tuple[int,int]:
 knots={int(row["page_count"]):row for row in layout["knots"]};one=knots[1]
 if page<min(knots) or page>max(knots):raise RuntimeError({"page_outside_layout":page})
 return int(one["payload_bytes"])*page,int(one["descriptor_bytes"])*page
def payload_vectors(model_id:str,configuration:str)->list[dict[str,Any]]:
 layout=layout_by_id(model_id);value=graph(configuration);output=[]
 for index,pages in enumerate(value["page_vectors"]):
  flows=[]
  for flow_id,(page,(sender,receiver)) in enumerate(zip(pages,value["edges"])):
   payload,descriptor=_page_sizes(layout,int(page));flows.append({"flow_id":flow_id,"sender_rank":sender,"receiver_rank":receiver,"page_count":int(page),"payload_bytes":payload,"descriptor_bytes":descriptor})
  output.append({"vector_id":f"{model_id}__{configuration.lower()}__third_blind_v{index:02d}","pages":[int(page) for page in pages],"flows":flows})
 return output
def validate_blind_contract(spec:dict[str,Any]|None=None)->dict[str,Any]:
 spec=spec or contract();grid=reserved_grid();base={key:value for key,value in grid.items() if key!="grid_sha256"};old=phase64_graph_grid();prior66=load_json(P66/"contracts/reserved_blind_grid.json");prior68=load_json(P68/"contracts/reserved_blind_grid.json");errors=[]
 expected_configs=spec["research_scope"]["fixed_configurations"]
 if grid.get("schema_version")!="phase70-reserved-multiflow-high-page-blind-grid-v1" or grid.get("grid_sha256")!=canonical_sha(base) or grid.get("grid_sha256")!=spec["blind_payload_contract"]["canonical_grid_sha256"]:errors.append("grid_identity")
 if grid.get("frozen_before_phase69_fit") is not True or grid.get("phase70_targets_opened") is not False:errors.append("blind_not_closed")
 if grid.get("models")!=spec["selected_models"] or set(grid.get("configurations",{}))!=set(expected_configs):errors.append("scope")
 prior_pages={int(page) for value in old["configurations"].values() for vector in value["page_vectors"] for page in vector}|{int(page) for values in prior66["configurations"].values() for vector in values for page in vector}|{int(page) for values in prior68["configurations"].values() for vector in values for page in vector};new_pages={int(page) for values in grid.get("configurations",{}).values() for vector in values for page in vector}
 if prior_pages!=set(spec["blind_payload_contract"]["prior_development_pages"]) or new_pages!=set(spec["blind_payload_contract"]["reserved_pages"]) or prior_pages&new_pages:errors.append("page_boundary")
 expected_edges={"P1D4":4,"P4D1":4,"P2D2_MATCHING":2,"P2D2_ALL_TO_ALL":4};ids=[]
 for model in spec["selected_models"]:
  model_layout=layout_by_id(model);minimum=min(int(knot["page_count"]) for knot in model_layout["knots"]);maximum=max(int(knot["page_count"]) for knot in model_layout["knots"])
  for configuration,count in expected_edges.items():
   value=graph(configuration);vectors=payload_vectors(model,configuration);ids.extend(row["vector_id"] for row in vectors)
   if len(value["page_vectors"])!=10 or len(value["edges"])!=count or len(set(map(tuple,value["edges"])))!=count:errors.append(f"graph:{configuration}")
   if any(len(row["pages"])!=count or any(page<minimum or page>maximum for page in row["pages"]) or any(flow["payload_bytes"]!=int(model_layout["knots"][0]["payload_bytes"])*flow["page_count"] or flow["descriptor_bytes"]!=int(model_layout["knots"][0]["descriptor_bytes"])*flow["page_count"] for flow in row["flows"]) for row in vectors):errors.append(f"payload:{model}:{configuration}")
 model_value=frozen_model()
 if file_sha(P69/"model/multiflow_high_page_residual.json")!=spec["frozen_correction_contract"]["sha256"] or model_value.get("candidate_id")!=spec["frozen_correction_contract"]["candidate_id"] or model_value.get("feature_family")!=spec["frozen_correction_contract"]["feature_family"] or model_value.get("activation_page_threshold")!=spec["frozen_correction_contract"]["activation_page_threshold"] or model_value.get("frozen_r67_model_sha256")!=spec["frozen_correction_contract"]["anchor_r67_sha256"] or len(model_value.get("groups",{}))!=spec["frozen_correction_contract"]["expected_groups"] or len(model_value.get("feature_names",[]))!=spec["frozen_correction_contract"]["expected_features"]:errors.append("frozen_model")
 if len(ids)!=80 or len(set(ids))!=80:errors.append("vector_cardinality")
 if errors:raise RuntimeError({"invalid_phase70_blind_contract":errors})
 return {"ok":True,"grid_sha256":grid["grid_sha256"],"vectors":80,"reserved_pages":sorted(new_pages),"development_pages":sorted(prior_pages),"configuration_flow_counts":expected_edges,"frozen_model_sha256":file_sha(P69/"model/multiflow_high_page_residual.json"),"frozen_r67_model_sha256":file_sha(P67/"model/multiflow_graph_page_correction.json")}
def iteration_counts(total_payload:int,spec:dict[str,Any]|None=None)->tuple[int,int]:
 measurement=(spec or contract())["measurement_contract"];timed=max(int(measurement["timed_iterations_min"]),min(int(measurement["timed_iterations_max"]),int(measurement["target_bytes_per_mode_block"])//max(total_payload,1)));warm=max(int(measurement["warmup_iterations_min"]),min(int(measurement["warmup_iterations_max"]),timed//10));return warm,timed
def _endpoint_key(endpoint:dict[str,Any])->tuple[str,int,str]:return str(endpoint["host"]),int(endpoint["physical_gpu"]),str(endpoint["ib_device"])
def _normalize_endpoint(endpoint:dict[str,Any])->dict[str,Any]:
 required=("host","host_aliases","physical_gpu","ib_device","transfer_hostname")
 if any(key not in endpoint for key in required) or not isinstance(endpoint["host_aliases"],list) or not endpoint["host_aliases"]:raise RuntimeError({"invalid_endpoint":endpoint})
 return {key:copy.deepcopy(endpoint[key]) for key in required}
def _prior_plans()->list[dict[str,Any]]:return [load_json(P64/"contracts/topology_plan.json"),load_json(P66/"contracts/topology_plan.json"),load_json(P68/"contracts/topology_plan.json")]
def prior_endpoint_keys()->set[tuple[str,int,str]]:return {_endpoint_key(endpoint) for plan in _prior_plans() for measurement in plan["measurements"] for endpoint in measurement["ranks"]}
def prior_host_signatures()->dict[str,set[tuple[str,...]]]:
 output:dict[str,set[tuple[str,...]]]=defaultdict(set)
 for plan in _prior_plans():
  for measurement in plan["measurements"]:output[measurement["topology_level"]].add(tuple(sorted({row["host"] for row in measurement["ranks"]})))
 return output
def validate_inventory(inventory:dict[str,Any],spec:dict[str,Any]|None=None)->dict[str,Any]:
 spec=spec or contract();errors=[]
 if inventory.get("schema_version")!="phase70-topology-inventory-v1" or inventory.get("blind_inventory_frozen_before_phase70_raw") is not True or inventory.get("selection_uses_phase70_latency_prediction_or_error") is not False:errors.append("inventory_header")
 expected_freshness={"all_endpoint_tuples_absent_from_phase64_phase66_phase68":True,"minimum_new_host_signatures_per_topology":int(spec["fresh_placement_contract"]["minimum_new_host_signatures_per_topology"]),"phase64_plan_sha256":file_sha(P64/"contracts/topology_plan.json"),"phase66_plan_sha256":file_sha(P66/"contracts/topology_plan.json"),"phase68_plan_sha256":file_sha(P68/"contracts/topology_plan.json")}
 if inventory.get("freshness_contract")!=expected_freshness:errors.append("freshness_contract")
 peak=inventory.get("phase70_peak_allocation_contract",{});expected_peak={"preferred_scheduler_reserved_nodes":4,"maximum_scheduler_reserved_nodes":4,"maximum_active_measurement_nodes_per_shard":2,"global_peak_simultaneous_gpu_processes":5,"maximum_concurrent_measurement_shards":1,"four_node_scheduler_reservation_permitted":True,"two_measurement_shards_concurrent_forbidden":True,"inventory_slots_are_not_nodes":True}
 if peak!=expected_peak:errors.append("peak_contract")
 reservation_mode=inventory.get("scheduler_reservation_mode")
 if reservation_mode not in ("FOUR_NODE_SINGLE_ALLOCATION","SEQUENTIAL_TOPOLOGY_EPOCHS"):errors.append("scheduler_reservation_mode")
 expected_keys={(level,replica) for level in ("L1","L2","L3") for replica in (0,1)};normalized=[];seen=set();endpoint_signatures={};new_host_counts=Counter();old_endpoints=prior_endpoint_keys();old_hosts=prior_host_signatures()
 for row in inventory.get("placements",[]):
  try:
   level=str(row["topology_level"]);replica=int(row["replica_id"]);key=(level,replica);sides={side:[_normalize_endpoint(endpoint) for endpoint in row["sides"][side]] for side in ("A","B")}
   if key in seen or key not in expected_keys or any(len(sides[side])!=4 for side in sides):raise ValueError("placement key/side count")
   endpoints=sides["A"]+sides["B"];keys=[_endpoint_key(endpoint) for endpoint in endpoints]
   if len(set(keys))!=8:raise ValueError("endpoint slots must be distinct")
   overlap=sorted(set(keys)&old_endpoints)
   if overlap:raise ValueError(f"Phase64/66/68 endpoint tuple reuse: {overlap}")
   a_hosts={endpoint["host"] for endpoint in sides["A"]};b_hosts={endpoint["host"] for endpoint in sides["B"]};all_hosts=a_hosts|b_hosts
   if level=="L1" and len(all_hosts)!=1:raise ValueError("L1 must use one host")
   if level in ("L2","L3") and (len(a_hosts)!=1 or len(b_hosts)!=1 or len(all_hosts)!=2):raise ValueError("L2/L3 must use exactly two hosts")
   evidence=copy.deepcopy(row["classification_evidence"])
   if evidence.get("classification_source") not in ("scheduler_asset_metadata","cluster_inventory_metadata","operator_verified_metadata"):raise ValueError("classification source")
   if level=="L2" and evidence.get("same_rack") is not True:raise ValueError("L2 rack")
   if level=="L3" and evidence.get("same_rack") is not False:raise ValueError("L3 rack")
   host_signature=tuple(sorted(all_hosts));host_fresh=host_signature not in old_hosts[level]
   if host_fresh:new_host_counts[level]+=1
   endpoint_signatures[key]=tuple(sorted(keys));seen.add(key);normalized.append({"placement_id":str(row["placement_id"]),"topology_level":level,"replica_id":replica,"classification_evidence":evidence,"sides":sides,"freshness":{"prior_endpoint_overlap_count":0,"all_eight_endpoint_tuples_fresh":True,"host_signature":list(host_signature),"host_signature_fresh_for_topology":host_fresh}})
  except (KeyError,TypeError,ValueError,RuntimeError) as exc:errors.append({"placement":row.get("placement_id"),"error":str(exc)})
 if seen!=expected_keys:errors.append({"placement_keys":sorted(seen),"expected":sorted(expected_keys)})
 for level in ("L1","L2","L3"):
  if (level,0) in endpoint_signatures and endpoint_signatures[(level,0)]==endpoint_signatures.get((level,1)):errors.append(f"replica_signature:{level}")
  if new_host_counts[level]<expected_freshness["minimum_new_host_signatures_per_topology"]:errors.append({"insufficient_new_host_signatures":level,"actual":new_host_counts[level]})
 unique_hosts={endpoint["host"] for placement in normalized for side in ("A","B") for endpoint in placement["sides"][side]}
 if reservation_mode=="FOUR_NODE_SINGLE_ALLOCATION" and len(unique_hosts)>4:errors.append({"four_node_pool_host_count":len(unique_hosts),"maximum":4})
 if errors:raise RuntimeError({"invalid_phase70_inventory":errors})
 return {"ok":True,"placements":6,"endpoint_slots":48,"scheduler_reservation_mode":reservation_mode,"inventory_unique_hosts":len(unique_hosts),"maximum_scheduler_reserved_nodes":4,"maximum_active_measurement_nodes":2,"maximum_simultaneous_gpu_processes":5,"prior_endpoint_overlap_count":0,"new_host_signatures_by_topology":dict(new_host_counts),"normalized_placements":sorted(normalized,key=lambda row:(row["topology_level"],row["replica_id"]))}
def _measurement_ranks(placement:dict[str,Any],configuration:str)->list[dict[str,Any]]:
 sides=placement["sides"];mapping={"P1D4":[sides["A"][0],*sides["B"]],"P4D1":[*sides["A"],sides["B"][0]],"P2D2_MATCHING":[sides["A"][0],sides["A"][1],sides["B"][0],sides["B"][1]],"P2D2_ALL_TO_ALL":[sides["A"][0],sides["A"][1],sides["B"][0],sides["B"][1]]};roles=graph(configuration)["roles"];return [{**copy.deepcopy(endpoint),"rank":rank,"role":roles[rank]} for rank,endpoint in enumerate(mapping[configuration])]
def expand_plan(inventory:dict[str,Any],inventory_sha256:str,generated_at_utc:str,workflow_commit:str,spec:dict[str,Any]|None=None)->dict[str,Any]:
 spec=spec or contract();inventory_audit=validate_inventory(inventory,spec);blind_audit=validate_blind_contract(spec);measurements=[]
 for model in spec["selected_models"]:
  for configuration in spec["research_scope"]["fixed_configurations"]:
   for placement in inventory_audit["normalized_placements"]:
    ranks=_measurement_ranks(placement,configuration);base={"measurement_id":f"{model}__{configuration.lower()}__{placement['topology_level'].lower()}__r{placement['replica_id']}","model_id":model,"configuration":configuration,"topology_level":placement["topology_level"],"replica_id":placement["replica_id"],"placement_id":placement["placement_id"],"classification_evidence":placement["classification_evidence"],"freshness":placement["freshness"],"world_size":len(ranks),"flow_count":len(graph(configuration)["edges"]),"op":"sglang_mooncake_multiflow_graph_batch_transfer_sync","ranks":ranks,"model_layout_sha256":canonical_sha(layout_by_id(model)),"payload_vectors_sha256":canonical_sha(payload_vectors(model,configuration))};measurements.append({**base,"measurement_sha256":canonical_sha(base)})
 base={"schema_version":"phase70-topology-plan-v1","workflow_commit":workflow_commit,"generated_at_utc":generated_at_utc,"inventory_file_sha256":inventory_sha256,"reserved_grid_sha256":blind_audit["grid_sha256"],"frozen_model_sha256":blind_audit["frozen_model_sha256"],"frozen_anchor_model_sha256":blind_audit["frozen_r67_model_sha256"],"placement_summary":{key:value for key,value in inventory_audit.items() if key not in ("ok","normalized_placements")},"resource_contract":spec["resource_allocation_contract"],"measurements":measurements};return {**base,"plan_sha256":canonical_sha(base)}
def validate_plan(plan:dict[str,Any])->dict[str,Any]:
 errors=[]
 if plan.get("schema_version")!="phase70-topology-plan-v1" or not isinstance(plan.get("workflow_commit"),str) or len(plan["workflow_commit"])!=40:errors.append("header")
 base={key:value for key,value in plan.items() if key!="plan_sha256"}
 if plan.get("plan_sha256")!=canonical_sha(base):errors.append("plan_sha")
 blind=validate_blind_contract()
 if plan.get("reserved_grid_sha256")!=blind["grid_sha256"] or plan.get("frozen_model_sha256")!=blind["frozen_model_sha256"] or plan.get("frozen_anchor_model_sha256")!=blind["frozen_r67_model_sha256"]:errors.append("blind_binding")
 rows=plan.get("measurements",[]);ids=[]
 for row in rows:
  ids.append(row.get("measurement_id"));value={key:item for key,item in row.items() if key!="measurement_sha256"}
  if row.get("measurement_sha256")!=canonical_sha(value):errors.append(f"measurement_sha:{row.get('measurement_id')}")
  configuration=row.get("configuration");world=contract()["resource_allocation_contract"]["world_size_by_configuration"].get(configuration)
  if row.get("world_size")!=world or len(row.get("ranks",[]))!=world or row.get("flow_count")!=len(graph(configuration)["edges"]):errors.append(f"shape:{row.get('measurement_id')}")
  if row.get("freshness",{}).get("all_eight_endpoint_tuples_fresh") is not True:errors.append(f"freshness:{row.get('measurement_id')}")
 if len(rows)!=48 or len(set(ids))!=48 or Counter(row["configuration"] for row in rows)!=Counter({configuration:12 for configuration in contract()["research_scope"]["fixed_configurations"]}):errors.append("coverage")
 if plan.get("placement_summary",{}).get("prior_endpoint_overlap_count")!=0 or any(plan.get("placement_summary",{}).get("new_host_signatures_by_topology",{}).get(level,0)<1 for level in ("L1","L2","L3")):errors.append("fresh_placement_summary")
 if errors:raise RuntimeError({"invalid_phase70_plan":errors})
 return {"ok":True,"plan_sha256":plan["plan_sha256"],"measurements":48,"endpoint_slots":48,"maximum_world_size":5,"prior_endpoint_overlap_count":0}
def measurement_by_id(plan:dict[str,Any],measurement_id:str)->dict[str,Any]:
 rows=[row for row in plan["measurements"] if row["measurement_id"]==measurement_id]
 if len(rows)!=1:raise RuntimeError({"unknown_measurement":measurement_id})
 return rows[0]
