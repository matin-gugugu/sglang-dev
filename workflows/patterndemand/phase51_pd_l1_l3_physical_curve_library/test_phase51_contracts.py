#!/usr/bin/env python3
"""CPU-only Phase51 contract tests."""
from __future__ import annotations
import json,sys,unittest
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from contracts import canonical_sha,expand_plan,iteration_counts,model_layouts,validate_inventory,validate_plan  # noqa:E402

def valid_inventory()->dict:
    inventory=json.loads((HERE/"topology_inventory.example.json").read_text());inventory.update({"created_at_utc":datetime.now(timezone.utc).isoformat(),"created_by":"unit-test","classification_source":"scheduler rack and fabric metadata","fabric_notes":"one declared rdma fabric"})
    for p in inventory["placements"]:
        level=p["topology_level"];r=p["replica_id"];p["placement_id"]=f"{level.lower()}-r{r}";p["evidence"]=f"asset metadata {level} replica {r}"
        for e in p["endpoints"]:
            rank=e["rank"]
            host=f"host-{level.lower()}-{r}-{rank}" if level!="L1" else f"host-l1-{r}"
            rack=f"rack-{r}" if level!="L3" else f"rack-{r}-{rank}"
            e.update({"host":host,"host_aliases":[host],"transfer_hostname":host,"rack_id":rack,"network_domain":"fabric-a","physical_gpu":rank+2*r,"ib_device":f"mlx5_{rank}"})
    return inventory
class TestPhase51(unittest.TestCase):
    def test_layouts(self):
        rows=model_layouts();self.assertEqual(len(rows),6);self.assertEqual(sum(len(r["knots"]) for r in rows)*3,396);self.assertEqual({r["model_id"]:r["descriptor_count"] for r in rows},{"qwen3-8b":72,"deepseek-v2-lite":27,"qwen3-30b-a3b":96,"llama-3.2-3b-instruct":56,"qwen2.5-14b-instruct":96,"mixtral-8x7b-instruct-v0.1":64});self.assertTrue(all(k["payload_bytes"]==k["descriptor_bytes"]*r["descriptor_count"] for r in rows for k in r["knots"]))
    def test_inventory_and_plan(self):
        inv=valid_inventory();self.assertEqual(validate_inventory(inv)["placements"],6);plan=expand_plan(inv,canonical_sha(inv),datetime.now(timezone.utc).isoformat(),"a"*40);audit=validate_plan(plan);self.assertEqual((audit["measurements"],audit["curves"],audit["curve_knots"]),(36,18,396));self.assertEqual(len({r["measurement_sha256"] for r in plan["measurements"]}),36)
    def test_posthoc_and_placeholder_rejected(self):
        inv=valid_inventory();inv["classification_source"]="latency benchmark";self.assertRaises(RuntimeError,validate_inventory,inv);self.assertRaises(RuntimeError,validate_inventory,json.loads((HERE/"topology_inventory.example.json").read_text()))
    def test_iteration_bounds(self):
        self.assertEqual(iteration_counts(1),(5,100));self.assertEqual(iteration_counts(10**12),(2,5))
if __name__=="__main__":unittest.main()
