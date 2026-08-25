#!/usr/bin/env python3
"""Phase64 frozen graph, payload, topology and plan contracts."""
from __future__ import annotations
import copy,hashlib,json
from collections import Counter
from pathlib import Path
from typing import Any

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
P51=ROOT/"experiment-results/phase51_pd_l1_l3_physical_curve_library"

def load_json(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))
def file_sha(path:Path)->str:
 d=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):d.update(chunk)
 return d.hexdigest()
def canonical_sha(value:Any)->str:return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def contract()->dict[str,Any]:return load_json(HERE/"experiment.json")
def layouts(spec:dict[str,Any]|None=None)->list[dict[str,Any]]:
 spec=spec or contract();source=load_json(P51/"contracts/model_transfer_layouts.json")["layouts"];by={r["model_id"]:r for r in source}
 return [copy.deepcopy(by[m]) for m in spec["selected_models"]]
def layout_by_id(model_id:str)->dict[str,Any]:
 rows=[r for r in layouts() if r["model_id"]==model_id]
 if len(rows)!=1:raise RuntimeError({"unknown_model":model_id})
 return rows[0]
def graph_grid()->dict[str,Any]:return load_json(HERE/"graph_payload_grid.json")
def graph(configuration:str)->dict[str,Any]:
 value=graph_grid()["configurations"].get(configuration)
 if not isinstance(value,dict):raise RuntimeError({"unknown_configuration":configuration})
 return copy.deepcopy(value)
def payload_vectors(model_id:str,configuration:str)->list[dict[str,Any]]:
 layout=layout_by_id(model_id);knots={int(r["page_count"]):r for r in layout["knots"]};g=graph(configuration);rows=[]
 for index,pages in enumerate(g["page_vectors"]):
  flows=[]
  for flow_id,(page,(sender,receiver)) in enumerate(zip(pages,g["edges"])):
   knot=knots[int(page)];flows.append({"flow_id":flow_id,"sender_rank":sender,"receiver_rank":receiver,"page_count":int(page),"payload_bytes":int(knot["payload_bytes"]),"descriptor_bytes":int(knot["descriptor_bytes"])})
  rows.append({"vector_id":f"{model_id}__{configuration.lower()}__v{index:02d}","pages":[int(v) for v in pages],"flows":flows})
 return rows
def validate_graph_contract(spec:dict[str,Any]|None=None)->dict[str,Any]:
 spec=spec or contract();grid=graph_grid();errors=[];base={k:v for k,v in grid.items() if k!="grid_sha256"}
 if grid.get("schema_version")!="phase64-pd-multiflow-graph-grid-v1" or grid.get("grid_sha256")!=canonical_sha(base):errors.append("grid_identity")
 if grid.get("selection_frozen_before_phase64_raw") is not True or grid.get("selection_uses_phase64_targets") is not False:errors.append("selection_freeze")
 if grid.get("models")!=spec["selected_models"] or list(grid.get("configurations",{}))!=spec["research_scope"]["fixed_configurations"]:errors.append("scope")
 expected_edges={"P1D4":4,"P4D1":4,"P2D2_MATCHING":2,"P2D2_ALL_TO_ALL":4};ids=[]
 for model in spec["selected_models"]:
  valid_pages={int(k["page_count"]) for k in layout_by_id(model)["knots"]}
  for config,count in expected_edges.items():
   g=graph(config);vectors=payload_vectors(model,config);ids.extend(v["vector_id"] for v in vectors)
   if len(g.get("page_vectors",[]))!=10 or len(g.get("edges",[]))!=count or len(set(map(tuple,g["edges"])))!=count:errors.append(f"graph:{config}")
   if any(len(v["pages"])!=count or not set(v["pages"])<=valid_pages for v in vectors):errors.append(f"payload:{model}:{config}")
 if len(ids)!=80 or len(set(ids))!=80:errors.append("vector_cardinality")
 if errors:raise RuntimeError({"invalid_phase64_graph_contract":errors})
 return {"ok":True,"grid_sha256":grid["grid_sha256"],"vectors":80,"configuration_flow_counts":expected_edges}
def iteration_counts(total_payload:int,spec:dict[str,Any]|None=None)->tuple[int,int]:
 m=(spec or contract())["measurement_contract"];timed=max(int(m["timed_iterations_min"]),min(int(m["timed_iterations_max"]),int(m["target_bytes_per_mode_block"])//max(total_payload,1)));warm=max(int(m["warmup_iterations_min"]),min(int(m["warmup_iterations_max"]),timed//10));return warm,timed
def _endpoint_key(e:dict[str,Any])->tuple[str,int,str]:return str(e["host"]),int(e["physical_gpu"]),str(e["ib_device"])
def _normalize_endpoint(e:dict[str,Any])->dict[str,Any]:
 required=("host","host_aliases","physical_gpu","ib_device","transfer_hostname")
 if any(k not in e for k in required) or not isinstance(e["host_aliases"],list) or not e["host_aliases"]:raise RuntimeError({"invalid_endpoint":e})
 return {k:copy.deepcopy(e[k]) for k in required}
def validate_inventory(inventory:dict[str,Any],spec:dict[str,Any]|None=None)->dict[str,Any]:
 spec=spec or contract();errors=[]
 if inventory.get("schema_version")!="phase64-topology-inventory-v1" or inventory.get("inventory_frozen_before_phase64_raw") is not True or inventory.get("selection_uses_phase64_latency_or_error") is not False:errors.append("inventory_header")
 peak=inventory.get("phase64_peak_allocation_contract",{});expected={"preferred_scheduler_reserved_nodes":4,"maximum_scheduler_reserved_nodes":4,"maximum_active_measurement_nodes_per_shard":2,"global_peak_simultaneous_gpu_processes":5,"maximum_concurrent_measurement_shards":1,"four_node_scheduler_reservation_permitted":True,"two_measurement_shards_concurrent_forbidden":True,"inventory_slots_are_not_nodes":True}
 if peak!=expected:errors.append("peak_contract")
 reservation_mode=inventory.get("scheduler_reservation_mode")
 if reservation_mode not in ("FOUR_NODE_SINGLE_ALLOCATION","SEQUENTIAL_TOPOLOGY_EPOCHS"):errors.append("scheduler_reservation_mode")
 rows=inventory.get("placements",[]);expected_keys={(l,r) for l in ("L1","L2","L3") for r in (0,1)};normalized=[];seen=set();signatures={}
 for row in rows:
  try:
   level=str(row["topology_level"]);rep=int(row["replica_id"]);key=(level,rep);sides={side:[_normalize_endpoint(e) for e in row["sides"][side]] for side in ("A","B")}
   if key in seen or key not in expected_keys or any(len(sides[s])!=4 for s in sides):raise ValueError("placement key/side count")
   endpoints=sides["A"]+sides["B"];keys=[_endpoint_key(e) for e in endpoints]
   if len(set(keys))!=8:raise ValueError("endpoint slots must be distinct")
   ah={e["host"] for e in sides["A"]};bh={e["host"] for e in sides["B"]};allh=ah|bh
   if level=="L1" and len(allh)!=1:raise ValueError("L1 must use one host")
   if level in ("L2","L3") and (len(ah)!=1 or len(bh)!=1 or len(allh)!=2):raise ValueError("L2/L3 must use exactly two hosts")
   evidence=copy.deepcopy(row["classification_evidence"])
   if evidence.get("classification_source") not in ("scheduler_asset_metadata","cluster_inventory_metadata","operator_verified_metadata"):raise ValueError("classification source")
   if level=="L2" and evidence.get("same_rack") is not True:raise ValueError("L2 rack")
   if level=="L3" and evidence.get("same_rack") is not False:raise ValueError("L3 rack")
   signature=tuple(sorted(keys));signatures[key]=signature;seen.add(key);normalized.append({"placement_id":str(row["placement_id"]),"topology_level":level,"replica_id":rep,"classification_evidence":evidence,"sides":sides})
  except (KeyError,TypeError,ValueError,RuntimeError) as exc:errors.append({"placement":row.get("placement_id"),"error":str(exc)})
 if seen!=expected_keys:errors.append({"placement_keys":sorted(seen),"expected":sorted(expected_keys)})
 for level in ("L1","L2","L3"):
  if (level,0) in signatures and signatures[(level,0)]==signatures.get((level,1)):errors.append(f"replica_signature:{level}")
 unique_hosts={e["host"] for placement in normalized for side in ("A","B") for e in placement["sides"][side]}
 if reservation_mode=="FOUR_NODE_SINGLE_ALLOCATION" and len(unique_hosts)>4:errors.append({"four_node_pool_host_count":len(unique_hosts),"maximum":4})
 if errors:raise RuntimeError({"invalid_phase64_inventory":errors})
 return {"ok":True,"placements":6,"endpoint_slots":48,"scheduler_reservation_mode":reservation_mode,"inventory_unique_hosts":len(unique_hosts),"maximum_scheduler_reserved_nodes":4,"maximum_active_measurement_nodes":2,"maximum_simultaneous_gpu_processes":5,"normalized_placements":sorted(normalized,key=lambda r:(r["topology_level"],r["replica_id"]))}
def _measurement_ranks(placement:dict[str,Any],configuration:str)->list[dict[str,Any]]:
 s=placement["sides"];mapping={"P1D4":[s["A"][0],*s["B"]],"P4D1":[*s["A"],s["B"][0]],"P2D2_MATCHING":[s["A"][0],s["A"][1],s["B"][0],s["B"][1]],"P2D2_ALL_TO_ALL":[s["A"][0],s["A"][1],s["B"][0],s["B"][1]]};roles=graph(configuration)["roles"]
 return [{**copy.deepcopy(e),"rank":rank,"role":roles[rank]} for rank,e in enumerate(mapping[configuration])]
def expand_plan(inventory:dict[str,Any],inventory_sha256:str,generated_at_utc:str,workflow_commit:str,spec:dict[str,Any]|None=None)->dict[str,Any]:
 spec=spec or contract();inv=validate_inventory(inventory,spec);ga=validate_graph_contract(spec);measurements=[]
 for model in spec["selected_models"]:
  for config in spec["research_scope"]["fixed_configurations"]:
   for placement in inv["normalized_placements"]:
    ranks=_measurement_ranks(placement,config);base={"measurement_id":f"{model}__{config.lower()}__{placement['topology_level'].lower()}__r{placement['replica_id']}","model_id":model,"configuration":config,"topology_level":placement["topology_level"],"replica_id":placement["replica_id"],"placement_id":placement["placement_id"],"classification_evidence":placement["classification_evidence"],"world_size":len(ranks),"flow_count":len(graph(config)["edges"]),"op":"sglang_mooncake_multiflow_graph_batch_transfer_sync","ranks":ranks,"model_layout_sha256":canonical_sha(layout_by_id(model)),"payload_vectors_sha256":canonical_sha(payload_vectors(model,config))};measurements.append({**base,"measurement_sha256":canonical_sha(base)})
 base={"schema_version":"phase64-topology-plan-v1","workflow_commit":workflow_commit,"generated_at_utc":generated_at_utc,"inventory_file_sha256":inventory_sha256,"graph_grid_sha256":ga["grid_sha256"],"placement_summary":{k:v for k,v in inv.items() if k not in ("ok","normalized_placements")},"resource_contract":spec["resource_allocation_contract"],"measurements":measurements};return {**base,"plan_sha256":canonical_sha(base)}
def validate_plan(plan:dict[str,Any])->dict[str,Any]:
 errors=[]
 if plan.get("schema_version")!="phase64-topology-plan-v1" or not isinstance(plan.get("workflow_commit"),str) or len(plan["workflow_commit"])!=40:errors.append("header")
 base={k:v for k,v in plan.items() if k!="plan_sha256"}
 if plan.get("plan_sha256")!=canonical_sha(base):errors.append("plan_sha")
 rows=plan.get("measurements",[]);ids=[]
 for row in rows:
  ids.append(row.get("measurement_id"));value={k:v for k,v in row.items() if k!="measurement_sha256"}
  if row.get("measurement_sha256")!=canonical_sha(value):errors.append(f"measurement_sha:{row.get('measurement_id')}")
  config=row.get("configuration");world=contract()["resource_allocation_contract"]["world_size_by_configuration"].get(config)
  if row.get("world_size")!=world or len(row.get("ranks",[]))!=world or row.get("flow_count")!=len(graph(config)["edges"]):errors.append(f"shape:{row.get('measurement_id')}")
 if len(rows)!=48 or len(set(ids))!=48 or Counter(r["configuration"] for r in rows)!=Counter({c:12 for c in contract()["research_scope"]["fixed_configurations"]}):errors.append("coverage")
 if errors:raise RuntimeError({"invalid_phase64_plan":errors})
 return {"ok":True,"plan_sha256":plan["plan_sha256"],"measurements":48,"endpoint_slots":48,"maximum_world_size":5}
def measurement_by_id(plan:dict[str,Any],measurement_id:str)->dict[str,Any]:
 rows=[r for r in plan["measurements"] if r["measurement_id"]==measurement_id]
 if len(rows)!=1:raise RuntimeError({"unknown_measurement":measurement_id})
 return rows[0]
