#!/usr/bin/env python3
from __future__ import annotations
import json,math,tempfile,unittest
from pathlib import Path
from contracts import contract,expand_plan,iteration_counts,layout_by_id,payload_vectors,validate_graph_contract,validate_inventory,validate_plan
from measurement import _predict,build_analysis,validate_raw

def endpoint(host:str,gpu:int)->dict:return {"host":host,"host_aliases":[host],"physical_gpu":gpu,"ib_device":"mlx5_0","transfer_hostname":host}
def inventory()->dict:
 rows=[]
 for level in ("L1","L2","L3"):
  for replica in (0,1):
   if level=="L1":a=b=f"l1host{replica}"
   else:a=f"{level.lower()}a{replica}";b=f"{level.lower()}b{replica}"
   rows.append({"placement_id":f"{level.lower()}-r{replica}","topology_level":level,"replica_id":replica,"classification_evidence":{"classification_source":"scheduler_asset_metadata","same_rack":level!="L3","fabric_domain":"f0"},"sides":{"A":[endpoint(a,i) for i in range(4)],"B":[endpoint(b,i+4) for i in range(4)]}})
 return {"schema_version":"phase64-topology-inventory-v1","inventory_frozen_before_phase64_raw":True,"selection_uses_phase64_latency_or_error":False,"scheduler_reservation_mode":"SEQUENTIAL_TOPOLOGY_EPOCHS","phase64_peak_allocation_contract":{"preferred_scheduler_reserved_nodes":4,"maximum_scheduler_reserved_nodes":4,"maximum_active_measurement_nodes_per_shard":2,"global_peak_simultaneous_gpu_processes":5,"maximum_concurrent_measurement_shards":1,"four_node_scheduler_reservation_permitted":True,"two_measurement_shards_concurrent_forbidden":True,"inventory_slots_are_not_nodes":True},"placements":rows}
def four_node_inventory()->dict:
 value=inventory();value["scheduler_reservation_mode"]="FOUR_NODE_SINGLE_ALLOCATION";mapping={("L1",0):("n0","n0"),("L1",1):("n1","n1"),("L2",0):("n0","n1"),("L2",1):("n2","n3"),("L3",0):("n0","n2"),("L3",1):("n1","n3")}
 for placement in value["placements"]:
  hosts=mapping[(placement["topology_level"],placement["replica_id"])]
  for side,host in zip(("A","B"),hosts):
   for endpoint_value in placement["sides"][side]:endpoint_value.update({"host":host,"host_aliases":[host],"transfer_hostname":host})
 return value
class Phase64ContractsTest(unittest.TestCase):
 def test_graph_and_payload_freeze(self)->None:
  audit=validate_graph_contract();self.assertEqual(audit["vectors"],80);self.assertEqual(len(payload_vectors("qwen3-8b","P1D4")),10);self.assertEqual(len(payload_vectors("deepseek-v2-lite","P2D2_MATCHING")[0]["flows"]),2)
 def test_inventory_and_plan(self)->None:
  audit=validate_inventory(inventory());self.assertEqual(audit["endpoint_slots"],48);plan=expand_plan(inventory(),"a"*64,"2026-08-25T00:00:00+00:00","b"*40);pa=validate_plan(plan);self.assertEqual(pa["measurements"],48);self.assertEqual(max(r["world_size"] for r in plan["measurements"]),5);self.assertTrue(all(len({e["host"] for e in r["ranks"]})<=2 for r in plan["measurements"]))
 def test_four_node_single_allocation_pool(self)->None:
  audit=validate_inventory(four_node_inventory());self.assertEqual(audit["scheduler_reservation_mode"],"FOUR_NODE_SINGLE_ALLOCATION");self.assertEqual(audit["inventory_unique_hosts"],4)
  value=inventory();value["scheduler_reservation_mode"]="FOUR_NODE_SINGLE_ALLOCATION"
  with self.assertRaises(RuntimeError):validate_inventory(value)
 def test_more_than_two_active_nodes_in_one_placement_rejected(self)->None:
  value=inventory();value["placements"][2]["sides"]["A"][1]["host"]="third";value["placements"][2]["sides"]["A"][1]["host_aliases"]=["third"]
  with self.assertRaises(RuntimeError):validate_inventory(value)
 def test_graph_formula_reduces_to_r61_two_flow(self)->None:
  costs=[1000.0,300.0];flows=[{"sender_rank":0,"receiver_rank":1},{"sender_rank":0,"receiver_rank":2}];pred,_,_=_predict(costs,flows);c=contract()["graph_formula_contract"]["coefficients"];expected=max(1.0,c["intercept_us"]+c["beta_max"]*max(costs)+c["beta_min"]*min(costs));self.assertTrue(math.isclose(pred,expected,rel_tol=1e-12))
 def test_full_synthetic_raw_to_analysis(self)->None:
  plan=expand_plan(inventory(),"a"*64,"2026-08-25T00:00:00+00:00","b"*40)
  def mode(flow_ids:list[int],timed:int,base:float)->dict:
   samples={str(fid):[base+fid]*timed for fid in flow_ids};waves=[max(samples[str(fid)][i] for fid in flow_ids) for i in range(timed)];skew=[0.0]*timed
   return {"flow_latency_samples_us":samples,"flow_latency_us":{key:{"min":values[0],"median":values[0],"p95":values[0],"max":values[0]} for key,values in samples.items()},"wave_latency_samples_us":waves,"wave_latency_us":{"min":waves[0],"median":waves[0],"p95":waves[0],"max":waves[0]},"sender_start_skew_samples_us":skew,"sender_start_skew_us":{"min":0.0,"median":0.0,"p95":0.0,"max":0.0},"return_codes_all_zero":True,"data_validation_pass":True}
  with tempfile.TemporaryDirectory() as tmp:
   raw_dir=Path(tmp)
   for measurement in plan["measurements"]:
    target=raw_dir/measurement["measurement_id"];target.mkdir()
    layout=layout_by_id(measurement["model_id"])
    endpoints=[{"rank":e["rank"],"role":e["role"],"expected_host":e["host"],"physical_gpu":e["physical_gpu"],"ib_device":e["ib_device"],"mooncake_protocol":"rdma","with_nvidia_peermem":"0"} for e in measurement["ranks"]]
    vectors=payload_vectors(measurement["model_id"],measurement["configuration"])
    for repeat_id in range(5):
     rows=[]
     for vector in vectors:
      flows=vector["flows"];warm,timed=iteration_counts(sum(f["payload_bytes"] for f in flows));flow_ids=[f["flow_id"] for f in flows];base=1000.0+10.0*len(flows)
      rows.append({"schema_version":"phase64-mooncake-multiflow-raw-v1","workflow_commit":plan["workflow_commit"],"plan_sha256":plan["plan_sha256"],"measurement_sha256":measurement["measurement_sha256"],"measurement_id":measurement["measurement_id"],"model_id":measurement["model_id"],"configuration":measurement["configuration"],"topology_level":measurement["topology_level"],"replica_id":measurement["replica_id"],"placement_id":measurement["placement_id"],"repeat_id":repeat_id,"vector_id":vector["vector_id"],"pages":vector["pages"],"flows":flows,"descriptor_layout":layout["descriptor_layout"],"descriptor_count":layout["descriptor_count"],"op":"MooncakeTransferEngine.batch_transfer_sync","transport":"rdma","wave_admission":"gloo_barrier_then_all_graph_edges_synchronous_release","warmup_iterations":warm,"timed_iterations":timed,"solo_flows":{str(fid):mode([fid],timed,base/2) for fid in flow_ids},"concurrent_wave":mode(flow_ids,timed,base),"runtime_endpoints":endpoints,"timestamp_utc":"2026-08-25T00:01:00+00:00"})
     (target/f"repeat_{repeat_id:02d}.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rows),encoding="utf-8")
   raw=validate_raw(plan,raw_dir,require_complete=True);self.assertTrue(raw["complete"]);self.assertEqual(raw["counts"]["records"],2400);analysis=build_analysis(plan,raw);self.assertEqual(len(analysis["points"]),240);self.assertEqual(len(analysis["replica_points"]),480)
if __name__=="__main__":unittest.main()
