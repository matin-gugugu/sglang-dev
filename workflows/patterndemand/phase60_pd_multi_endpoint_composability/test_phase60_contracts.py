#!/usr/bin/env python3
"""CPU-only Phase60 contract tests."""
from __future__ import annotations
import json,sys,unittest
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from contracts import canonical_sha,expand_plan,iteration_counts,payload_pairs,selected_layouts,validate_inventory,validate_pair_contract,validate_plan  # noqa:E402
from measurement import build_analysis  # noqa:E402

def valid_inventory()->dict:
    inv=json.loads((HERE/"topology_inventory.example.json").read_text());inv.update({"created_at_utc":datetime.now(timezone.utc).isoformat(),"created_by":"unit-test","classification_source":"scheduler rack and fabric metadata","fabric_notes":"one declared rdma fabric"})
    for p in inv["placements"]:
        level=p["topology_level"];replica=p["replica_id"];p["placement_id"]=f"{level.lower()}-r{replica}";p["evidence"]=f"asset metadata {level} replica {replica}"
        for side in ("A","B"):
            for slot,e in enumerate(p["sides"][side]):
                if level=="L1":host=f"host-l1-{replica}";rack=f"rack-l1-{replica}"
                else:host=f"host-{level.lower()}-{replica}-{side.lower()}";rack=f"rack-{replica}" if level=="L2" else f"rack-{replica}-{side.lower()}"
                e.update({"host":host,"host_aliases":[host],"transfer_hostname":host,"rack_id":rack,"network_domain":"fabric-a","physical_gpu":slot+(0 if side=="A" else 2),"ib_device":f"mlx5_{slot+(0 if side=='A' else 2)}"})
    return inv
class TestPhase60(unittest.TestCase):
    def test_layout_and_pairs(self):
        self.assertEqual([row["model_id"] for row in selected_layouts()],["qwen3-8b","deepseek-v2-lite"]);audit=validate_pair_contract();self.assertEqual(audit["counts"],{"qwen3-8b":{"development":10,"reserved_future_blind":10},"deepseek-v2-lite":{"development":10,"reserved_future_blind":10}})
        for model in ("qwen3-8b","deepseek-v2-lite"):self.assertFalse({(r["page_count0"],r["page_count1"]) for r in payload_pairs(model)} & {(r["page_count0"],r["page_count1"]) for r in payload_pairs(model,"reserved_future_blind_pairs")})
    def test_inventory_and_plan(self):
        inv=valid_inventory();inventory_audit=validate_inventory(inv);self.assertEqual((inventory_audit["placements"],inventory_audit["max_simultaneous_nodes_per_shard"],inventory_audit["simultaneous_gpu_processes_per_shard"]),(6,2,3));plan=expand_plan(inv,canonical_sha(inv),datetime.now(timezone.utc).isoformat(),"a"*40);audit=validate_plan(plan);self.assertEqual((audit["measurements"],audit["world_size_per_shard"],audit["max_simultaneous_nodes_per_shard"],audit["development_points"],audit["replica_points"]),(24,3,2,120,240));self.assertEqual(len({row["measurement_sha256"] for row in plan["measurements"]}),24)
        p1=next(row for row in plan["measurements"] if row["configuration"]=="P1D2");p2=next(row for row in plan["measurements"] if row["configuration"]=="P2D1");self.assertEqual([r["role"] for r in p1["ranks"]],["P0","D0","D1"]);self.assertEqual([r["role"] for r in p2["ranks"]],["P0","P1","D0"])
    def test_placeholder_and_posthoc_rejected(self):
        self.assertRaises(RuntimeError,validate_inventory,json.loads((HERE/"topology_inventory.example.json").read_text()));inv=valid_inventory();inv["classification_source"]="latency benchmark";self.assertRaises(RuntimeError,validate_inventory,inv)
    def test_iteration_bounds(self):
        self.assertEqual(iteration_counts(1),(3,30));self.assertEqual(iteration_counts(10**12),(2,5))
    def test_analysis_cardinality(self):
        inv=valid_inventory();plan=expand_plan(inv,canonical_sha(inv),datetime.now(timezone.utc).isoformat(),"a"*40);records={}
        for measurement in plan["measurements"]:
            rows=[]
            for pair in payload_pairs(measurement["model_id"]):rows.append({"pair_id":pair["pair_id"],"concurrent_wave":{"wave_latency_us":{"median":120.0}},"solo_flow0":{"wave_latency_us":{"median":100.0}},"solo_flow1":{"wave_latency_us":{"median":110.0}}})
            records[measurement["measurement_id"]]={repeat:[dict(row) for row in rows] for repeat in range(5)}
        result=build_analysis(plan,{"complete":True,"records":records});self.assertEqual((len(result["points"]),len(result["replica_points"]),len(result["spreads"]),len(result["metrics"])),(120,240,120,14));self.assertIn(result["decision"]["scientific_outcome"],{"P1D1_DIRECTLY_COMPOSABLE_DEVELOPMENT","P1D1_CURVE_TRANSFER_DRIFT_REQUIRES_REVIEW","CONTENTION_CORRECTION_CANDIDATE"})
if __name__=="__main__":unittest.main()
