#!/usr/bin/env python3
from __future__ import annotations
import json,math,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from contracts import P64,P66,P68,expand_plan,iteration_counts,layout_by_id,load_json,payload_vectors,prior_endpoint_keys,validate_blind_contract,validate_inventory,validate_plan
from measurement import _curve_map,_graph_features,_interpolate,_predict_phase67,_predict_phase69,build_analysis,validate_raw

def endpoint(host:str,gpu:int)->dict:return {"host":host,"host_aliases":[host],"physical_gpu":gpu,"ib_device":f"mlx5_{gpu}","transfer_hostname":host}
def inventory()->dict:
 mapping={("L1",0):("fresh0","fresh0"),("L1",1):("fresh1","fresh1"),("L2",0):("fresh0","fresh1"),("L2",1):("fresh2","fresh3"),("L3",0):("fresh0","fresh2"),("L3",1):("fresh1","fresh3")};placements=[]
 for level in ("L1","L2","L3"):
  for replica in (0,1):
   a,b=mapping[(level,replica)];placements.append({"placement_id":f"{level.lower()}-r{replica}","topology_level":level,"replica_id":replica,"classification_evidence":{"classification_source":"scheduler_asset_metadata","same_rack":level!="L3","fabric_domain":"f0"},"sides":{"A":[endpoint(a,index) for index in range(4)],"B":[endpoint(b,index) for index in range(4,8)]}})
 return {"schema_version":"phase70-topology-inventory-v1","blind_inventory_frozen_before_phase70_raw":True,"selection_uses_phase70_latency_prediction_or_error":False,"scheduler_reservation_mode":"FOUR_NODE_SINGLE_ALLOCATION","freshness_contract":{"all_endpoint_tuples_absent_from_phase64_phase66_phase68":True,"minimum_new_host_signatures_per_topology":1,"phase64_plan_sha256":"5c05154e13866deb0a30cbad32895f6c1db3936ea6359961e3297f4bbfac1819","phase66_plan_sha256":"1898e3615761c1d7d6230a9e5ea4943f8cd024d60bb98e6e1d24faed4b5d62ba","phase68_plan_sha256":"bb641a0d824354c1d27f38ca4c3b03495be65e7e7ecb993c70ca0a47120f3437"},"phase70_peak_allocation_contract":{"preferred_scheduler_reserved_nodes":4,"maximum_scheduler_reserved_nodes":4,"maximum_active_measurement_nodes_per_shard":2,"global_peak_simultaneous_gpu_processes":5,"maximum_concurrent_measurement_shards":1,"four_node_scheduler_reservation_permitted":True,"two_measurement_shards_concurrent_forbidden":True,"inventory_slots_are_not_nodes":True},"placements":placements}
def mode(flow_ids:list[int],timed:int,base:float)->dict:
 samples={str(flow_id):[base+flow_id]*timed for flow_id in flow_ids};waves=[max(samples[str(flow_id)][index] for flow_id in flow_ids) for index in range(timed)];skew=[0.0]*timed;return {"flow_latency_samples_us":samples,"flow_latency_us":{key:{"min":values[0],"median":values[0],"p95":values[0],"max":values[0]} for key,values in samples.items()},"wave_latency_samples_us":waves,"wave_latency_us":{"min":waves[0],"median":waves[0],"p95":waves[0],"max":waves[0]},"sender_start_skew_samples_us":skew,"sender_start_skew_us":{"min":0.0,"median":0.0,"p95":0.0,"max":0.0},"return_codes_all_zero":True,"data_validation_pass":True}

class Phase70ContractsTest(unittest.TestCase):
 def test_reserved_payload_and_frozen_model(self)->None:
  audit=validate_blind_contract();self.assertEqual(audit["vectors"],80);self.assertEqual(audit["reserved_pages"],[34,38,44,52,60]);self.assertTrue(set(audit["reserved_pages"]).isdisjoint(audit["development_pages"]));vector=payload_vectors("deepseek-v2-lite","P2D2_MATCHING")[0];self.assertEqual(vector["pages"],[34,34]);self.assertEqual(vector["flows"][0]["payload_bytes"],1990656*34);self.assertEqual(vector["flows"][0]["descriptor_bytes"],73728*34)
 def test_fresh_inventory_and_plan(self)->None:
  audit=validate_inventory(inventory());self.assertEqual(audit["prior_endpoint_overlap_count"],0);self.assertEqual(audit["inventory_unique_hosts"],4);plan=expand_plan(inventory(),"a"*64,"2026-08-26T00:00:00+00:00","b"*40);plan_audit=validate_plan(plan);self.assertEqual(plan_audit["measurements"],48);self.assertEqual(plan_audit["maximum_world_size"],5);self.assertTrue(all(len({endpoint["host"] for endpoint in measurement["ranks"]})<=2 for measurement in plan["measurements"]))
 def test_prior_endpoint_reuse_is_rejected(self)->None:
  prior=prior_endpoint_keys()
  for root in (P64,P66,P68):
   endpoint=load_json(root/"contracts/topology_plan.json")["measurements"][0]["ranks"][0];old=(endpoint["host"],int(endpoint["physical_gpu"]),endpoint["ib_device"]);self.assertIn(old,prior);value=inventory();target=value["placements"][0]["sides"]["A"][0];target.update({"host":old[0],"host_aliases":[old[0]],"physical_gpu":old[1],"ib_device":old[2],"transfer_hostname":old[0]})
   with self.assertRaises(RuntimeError):validate_inventory(value)
 def test_frozen_prediction_is_positive(self)->None:
  vector=payload_vectors("qwen3-8b","P1D4")[0];curve=_curve_map()[("qwen3-8b","L1")];costs=[_interpolate(curve,flow["payload_bytes"]) for flow in vector["flows"]];m,b,s=_graph_features(costs,vector["flows"]);phase67=_predict_phase67("qwen3-8b","P1D4",m,b,s,vector["pages"]);prediction=_predict_phase69("qwen3-8b","P1D4",phase67,vector["pages"]);self.assertGreater(prediction,0);self.assertTrue(math.isfinite(prediction))
 def test_full_synthetic_raw_to_analysis(self)->None:
  plan=expand_plan(inventory(),"a"*64,"2026-08-26T00:00:00+00:00","b"*40)
  with tempfile.TemporaryDirectory() as temporary:
   raw_dir=Path(temporary)
   for measurement in plan["measurements"]:
    target=raw_dir/measurement["measurement_id"];target.mkdir();layout=layout_by_id(measurement["model_id"]);runtime_endpoints=[{"rank":endpoint["rank"],"role":endpoint["role"],"expected_host":endpoint["host"],"physical_gpu":endpoint["physical_gpu"],"ib_device":endpoint["ib_device"],"mooncake_protocol":"rdma","with_nvidia_peermem":"0"} for endpoint in measurement["ranks"]]
    for repeat_id in range(5):
     rows=[]
     for vector in payload_vectors(measurement["model_id"],measurement["configuration"]):
      flows=vector["flows"];warmup,timed=iteration_counts(sum(flow["payload_bytes"] for flow in flows));flow_ids=[flow["flow_id"] for flow in flows];base=1000.0+10.0*len(flows);rows.append({"schema_version":"phase70-mooncake-multiflow-raw-v1","workflow_commit":plan["workflow_commit"],"plan_sha256":plan["plan_sha256"],"measurement_sha256":measurement["measurement_sha256"],"measurement_id":measurement["measurement_id"],"model_id":measurement["model_id"],"configuration":measurement["configuration"],"topology_level":measurement["topology_level"],"replica_id":measurement["replica_id"],"placement_id":measurement["placement_id"],"repeat_id":repeat_id,"vector_id":vector["vector_id"],"pages":vector["pages"],"flows":flows,"descriptor_layout":layout["descriptor_layout"],"descriptor_count":layout["descriptor_count"],"op":"MooncakeTransferEngine.batch_transfer_sync","transport":"rdma","wave_admission":"gloo_barrier_then_all_graph_edges_synchronous_release","warmup_iterations":warmup,"timed_iterations":timed,"solo_flows":{str(flow_id):mode([flow_id],timed,base/2) for flow_id in flow_ids},"concurrent_wave":mode(flow_ids,timed,base),"runtime_endpoints":runtime_endpoints,"timestamp_utc":"2026-08-26T00:01:00+00:00"})
     (target/f"repeat_{repeat_id:02d}.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rows),encoding="utf-8")
   raw=validate_raw(plan,raw_dir,require_complete=True);self.assertTrue(raw["complete"]);self.assertEqual(raw["counts"]["records"],2400);analysis=build_analysis(plan,raw);self.assertEqual(len(analysis["points"]),240);self.assertEqual(len(analysis["replica_points"]),480);self.assertEqual(len(analysis["metrics"]),30);self.assertEqual(sum(row["slice_type"]=="model_configuration" for row in analysis["metrics"]),8);self.assertTrue(all(point["phase69_prediction_us"]>0 for point in analysis["points"]));self.assertTrue(all(math.isclose(point["phase69_prediction_us"],point["phase67_prediction_us"],rel_tol=0.0,abs_tol=1e-12) for point in analysis["points"] if point["configuration"]=="P2D2_MATCHING"));self.assertIn("each_model_configuration_wape",analysis["decision"]["checks"]);self.assertIn("strictly_preserves_r67_for_p2d2_matching",analysis["decision"]["checks"])

if __name__=="__main__":unittest.main()
