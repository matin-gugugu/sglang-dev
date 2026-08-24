#!/usr/bin/env python3
"""CPU-only Phase63 contract tests."""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from contracts import canonical_sha, expand_plan, iteration_counts, load_json, payload_pairs, validate_inventory, validate_pair_contract, validate_plan  # noqa: E402
from measurement import build_external_analysis  # noqa: E402


def valid_inventory() -> dict:
    inventory = load_json(HERE / "topology_inventory.example.json")
    inventory.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "phase63-unit-test",
        "classification_source": "scheduler rack and fabric asset metadata",
        "fabric_notes": "one declared RDMA fabric",
    })
    for placement in inventory["placements"]:
        level = placement["topology_level"]
        replica = placement["replica_id"]
        placement["placement_id"] = f"external-{level.lower()}-r{replica}"
        placement["evidence"] = f"asset metadata for external {level} replica {replica}"
        for side in ("A", "B"):
            for slot, endpoint in enumerate(placement["sides"][side]):
                if level == "L1":
                    host = f"phase63-l1-r{replica}"
                    rack = f"phase63-rack-l1-r{replica}"
                else:
                    host = f"phase63-{level.lower()}-r{replica}-{side.lower()}"
                    rack = f"phase63-rack-l2-r{replica}" if level == "L2" else f"phase63-rack-l3-r{replica}-{side.lower()}"
                gpu = slot + (0 if side == "A" else 2)
                endpoint.update({
                    "host": host,
                    "host_aliases": [host],
                    "transfer_hostname": host,
                    "rack_id": rack,
                    "network_domain": "phase63-test-fabric",
                    "physical_gpu": gpu,
                    "ib_device": f"mlx5_{gpu}",
                })
    return inventory


class TestPhase63(unittest.TestCase):
    def test_four_held_out_models_and_pairs(self) -> None:
        audit = validate_pair_contract()
        self.assertEqual(audit["pairs"], 40)
        self.assertEqual(set(audit["counts"]), {
            "qwen3-30b-a3b", "llama-3.2-3b-instruct",
            "qwen2.5-14b-instruct", "mixtral-8x7b-instruct-v0.1",
        })
        self.assertTrue(all("__ext_" in row["pair_id"] for model in audit["counts"] for row in payload_pairs(model)))

    def test_inventory_and_48_shard_plan(self) -> None:
        inventory = valid_inventory()
        audit = validate_inventory(inventory)
        self.assertEqual((audit["placements"], audit["endpoint_slots"], audit["maximum_simultaneous_nodes_per_shard"], audit["global_peak_simultaneous_nodes"], audit["maximum_concurrent_measurement_shards"]), (6, 24, 2, 2, 1))
        plan = expand_plan(inventory, canonical_sha(inventory), datetime.now(timezone.utc).isoformat(), "a" * 40)
        plan_audit = validate_plan(plan)
        self.assertEqual((plan_audit["measurements"], plan_audit["official_points"], plan_audit["replica_points"]), (48, 240, 480))
        self.assertEqual((plan_audit["global_peak_simultaneous_nodes"], plan_audit["maximum_concurrent_measurement_shards"]), (2, 1))
        p1d2 = next(row for row in plan["measurements"] if row["configuration"] == "P1D2")
        p2d1 = next(row for row in plan["measurements"] if row["configuration"] == "P2D1")
        self.assertEqual([row["role"] for row in p1d2["ranks"]], ["P0", "D0", "D1"])
        self.assertEqual([row["role"] for row in p2d1["ranks"]], ["P0", "P1", "D0"])

    def test_placeholder_inventory_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_inventory(load_json(HERE / "topology_inventory.example.json"))

    def test_target_influenced_selection_rejected(self) -> None:
        inventory = valid_inventory()
        inventory["selection_uses_phase63_latency_or_error"] = True
        with self.assertRaises(RuntimeError):
            validate_inventory(inventory)

    def test_same_l2_node_pair_can_serve_replicas_sequentially(self) -> None:
        inventory = valid_inventory()
        r0 = next(row for row in inventory["placements"] if row["topology_level"] == "L2" and row["replica_id"] == 0)
        r1 = next(row for row in inventory["placements"] if row["topology_level"] == "L2" and row["replica_id"] == 1)
        for side in ("A", "B"):
            for slot, endpoint in enumerate(r1["sides"][side]):
                source = r0["sides"][side][slot]
                endpoint.update({
                    "host": source["host"],
                    "host_aliases": [source["host"]],
                    "transfer_hostname": source["transfer_hostname"],
                    "rack_id": source["rack_id"],
                    "network_domain": source["network_domain"],
                    "physical_gpu": int(source["physical_gpu"]) + 4,
                    "ib_device": f"mlx5_{int(source['physical_gpu']) + 4}",
                })
        audit = validate_inventory(inventory)
        self.assertEqual((audit["global_peak_simultaneous_nodes"], audit["maximum_concurrent_measurement_shards"]), (2, 1))

    def test_four_node_peak_contract_rejected(self) -> None:
        inventory = valid_inventory()
        inventory["phase63_peak_allocation_contract"]["global_peak_simultaneous_nodes"] = 4
        with self.assertRaises(RuntimeError):
            validate_inventory(inventory)

    def test_external_analysis_cardinality(self) -> None:
        inventory = valid_inventory()
        plan = expand_plan(inventory, canonical_sha(inventory), datetime.now(timezone.utc).isoformat(), "a" * 40)
        records = {}
        for measurement in plan["measurements"]:
            rows = []
            for pair in payload_pairs(measurement["model_id"]):
                rows.append({
                    "pair_id": pair["pair_id"],
                    "concurrent_wave": {"wave_latency_us": {"median": 120.0}},
                    "solo_flow0": {"wave_latency_us": {"median": 90.0}},
                    "solo_flow1": {"wave_latency_us": {"median": 95.0}},
                })
            records[measurement["measurement_id"]] = {repeat: [dict(row) for row in rows] for repeat in range(5)}
        analysis = build_external_analysis(plan, {"complete": True, "records": records})
        self.assertEqual(
            (len(analysis["points"]), len(analysis["replica_points"]), len(analysis["spreads"]), len(analysis["metrics"]), len(analysis["combined_six_model_metrics"])),
            (240, 480, 240, 34, 7),
        )
        self.assertIn(analysis["decision"]["scientific_outcome"], {"FROZEN_CONTENTION_CORRECTION_SIX_MODEL_PASS", "FROZEN_CONTENTION_CORRECTION_FOUR_MODEL_EXTERNAL_FAIL"})

    def test_iteration_bounds(self) -> None:
        self.assertEqual(iteration_counts(1), (3, 30))
        self.assertEqual(iteration_counts(10**12), (2, 5))


if __name__ == "__main__":
    unittest.main()
