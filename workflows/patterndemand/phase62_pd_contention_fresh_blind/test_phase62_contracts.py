#!/usr/bin/env python3
"""CPU-only Phase62 contract tests."""
from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from contracts import canonical_sha, expand_plan, file_sha, iteration_counts, load_json, payload_pairs, phase60_endpoint_keys, validate_inventory, validate_pair_contract, validate_plan  # noqa: E402
from measurement import build_blind_analysis  # noqa: E402


def valid_inventory() -> dict:
    inventory = load_json(HERE / "topology_inventory.example.json")
    inventory.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "phase62-unit-test",
        "classification_source": "scheduler rack and fabric asset metadata",
        "fabric_notes": "one declared RDMA fabric",
    })
    for placement in inventory["placements"]:
        level = placement["topology_level"]
        replica = placement["replica_id"]
        placement["placement_id"] = f"blind-{level.lower()}-r{replica}"
        placement["evidence"] = f"asset metadata for blind {level} replica {replica}"
        for side in ("A", "B"):
            for slot, endpoint in enumerate(placement["sides"][side]):
                if level == "L1":
                    host = f"new-l1-r{replica}"
                    rack = f"new-rack-l1-r{replica}"
                else:
                    host = f"new-{level.lower()}-r{replica}-{side.lower()}"
                    rack = f"new-rack-l2-r{replica}" if level == "L2" else f"new-rack-l3-r{replica}-{side.lower()}"
                gpu = slot + (0 if side == "A" else 2)
                endpoint.update({
                    "host": host,
                    "host_aliases": [host],
                    "transfer_hostname": host,
                    "rack_id": rack,
                    "network_domain": "new-test-fabric",
                    "physical_gpu": gpu,
                    "ib_device": f"mlx5_{gpu}",
                })
    return inventory


class TestPhase62(unittest.TestCase):
    def test_reserved_pairs_only(self) -> None:
        audit = validate_pair_contract()
        self.assertEqual(audit["counts"], {"qwen3-8b": 10, "deepseek-v2-lite": 10})
        self.assertTrue(all("__res_" in row["pair_id"] for model in ("qwen3-8b", "deepseek-v2-lite") for row in payload_pairs(model)))

    def test_fresh_inventory_and_plan(self) -> None:
        inventory = valid_inventory()
        audit = validate_inventory(inventory)
        self.assertEqual((audit["placements"], audit["fresh_endpoint_slots"], audit["max_simultaneous_nodes_per_shard"]), (6, 24, 2))
        plan = expand_plan(inventory, canonical_sha(inventory), datetime.now(timezone.utc).isoformat(), "a" * 40)
        plan_audit = validate_plan(plan)
        self.assertEqual((plan_audit["measurements"], plan_audit["official_points"], plan_audit["replica_points"]), (24, 120, 240))
        p1d2 = next(row for row in plan["measurements"] if row["configuration"] == "P1D2")
        p2d1 = next(row for row in plan["measurements"] if row["configuration"] == "P2D1")
        self.assertEqual([row["role"] for row in p1d2["ranks"]], ["P0", "D0", "D1"])
        self.assertEqual([row["role"] for row in p2d1["ranks"]], ["P0", "P1", "D0"])

    def test_phase60_endpoint_reuse_rejected(self) -> None:
        inventory = valid_inventory()
        old_host, old_gpu, old_ib = next(iter(phase60_endpoint_keys()))
        placement = next(row for row in inventory["placements"] if row["topology_level"] == "L1")
        for side in ("A", "B"):
            for slot, endpoint in enumerate(placement["sides"][side]):
                endpoint.update({
                    "host": old_host,
                    "host_aliases": [old_host],
                    "transfer_hostname": old_host,
                    "rack_id": "old-host-test-rack",
                })
                if side == "A" and slot == 0:
                    endpoint["physical_gpu"] = old_gpu
                    endpoint["ib_device"] = old_ib
        with self.assertRaises(RuntimeError):
            validate_inventory(inventory)

    def test_placeholder_inventory_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_inventory(load_json(HERE / "topology_inventory.example.json"))

    def test_blind_analysis_cardinality(self) -> None:
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
        analysis = build_blind_analysis(plan, {"complete": True, "records": records})
        self.assertEqual((len(analysis["points"]), len(analysis["replica_points"]), len(analysis["spreads"]), len(analysis["metrics"])), (120, 240, 120, 14))
        self.assertIn(analysis["decision"]["scientific_outcome"], {"FROZEN_CONTENTION_CORRECTION_FRESH_BLIND_PASS", "FROZEN_CONTENTION_CORRECTION_FRESH_BLIND_FAIL"})

    def test_iteration_bounds(self) -> None:
        self.assertEqual(iteration_counts(1), (3, 30))
        self.assertEqual(iteration_counts(10**12), (2, 5))


if __name__ == "__main__":
    unittest.main()
