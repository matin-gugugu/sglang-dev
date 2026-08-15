#!/usr/bin/env python3
"""CPU-only full-matrix synthetic test; never creates a formal Phase39 result."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from analysis import (
    aggregate_cost_metrics,
    combined_cost_rows,
    compare_histogram_metrics,
    compare_phase35,
    cost_phase_rows,
    frozen_histogram_metrics,
    input_rows,
    placement_decisions,
    read_csv,
)
from contracts import contract, expand_plan, measurement_sha, validate_plan
from measurement import build_curves, validate_raw

ROOT = Path(__file__).resolve().parents[3]


def inventory() -> dict:
    placements = []
    definitions = {
        "L1": [
            [("l1-a", "rack-a", list(range(8)))],
            [("l1-b", "rack-b", list(range(8)))],
        ],
        "L2": [
            [("l2-a0", "rack-c", list(range(4))), ("l2-a1", "rack-c", list(range(4)))],
            [("l2-b0", "rack-d", list(range(4))), ("l2-b1", "rack-d", list(range(4)))],
        ],
        "L3": [
            [("l3-a0", "rack-e", list(range(4))), ("l3-a1", "rack-f", list(range(4)))],
            [("l3-b0", "rack-g", list(range(4))), ("l3-b1", "rack-h", list(range(4)))],
        ],
    }
    for level, replicas in definitions.items():
        for replica_id, nodes in enumerate(replicas):
            placements.append({
                "topology_level": level,
                "replica_id": replica_id,
                "placement_id": f"{level.lower()}_replica{replica_id}",
                "evidence": f"synthetic pre-measurement {level} allocation metadata",
                "nodes": [{
                    "host": host, "host_aliases": [host], "rack_id": rack,
                    "network_domain": "synthetic-fabric", "nic_ids": ["nic0"], "gpu_ids": gpus,
                } for host, rack, gpus in nodes],
            })
    return {
        "schema_version": "phase39-topology-inventory-v1",
        "created_at_utc": "2026-08-15T00:00:00+00:00",
        "created_by": "phase39_cpu_synthetic_test",
        "classification_source": "synthetic scheduler allocation metadata",
        "classification_frozen_before_measurement": True,
        "classification_not_inferred_from_benchmark": True,
        "fabric_notes": "synthetic contract test only",
        "placements": placements,
    }


def record(plan: dict, measurement: dict, repeat: int, payload: int, direction: str, latency: float, iterations: int) -> dict:
    measurement_contract = contract()["measurement_contract"]
    samples = [latency * (1.0 + ((index % 3) - 1) * 0.0001) for index in range(iterations)]
    row = {
        "schema_version": "phase39-distributed-raw-v1",
        "timestamp_utc": "2026-08-15T00:01:00+00:00",
        "shard_started_at_utc": "2026-08-15T00:01:00+00:00",
        "workflow_commit": plan["workflow_commit"],
        "plan_sha256": plan["plan_sha256"],
        "measurement_sha256": measurement_sha(measurement),
        "measurement_id": measurement["measurement_id"],
        "case_key": measurement["case_key"],
        "replica_id": measurement["replica_id"],
        "placement_id": measurement["placement_id"],
        "parallelism": measurement["parallelism"],
        "topology_level": measurement["topology_level"],
        "classification_evidence": measurement["classification_evidence"],
        "rank_mapping": measurement["ranks"],
        "repeat_id": repeat,
        "payload_bytes": payload,
        "dtype": "bfloat16",
        "warmup_iterations": 30,
        "timed_iterations": iterations,
        "op": "p2p_send_tensor" if measurement["parallelism"] == "pp" else "sglang_tp_all_reduce",
        "backend": measurement_contract["pp_backend" if measurement["parallelism"] == "pp" else "tp_backend"],
        "measurement_scope": "synthetic contract test",
        "direction": direction,
        "latency_us": {"min": min(samples), "median": latency, "mean": latency, "p95": max(samples), "p99": max(samples), "max": max(samples)},
        "completion_cuda_samples_us": samples,
        "rank_cuda_samples_us": [samples for _ in range(int(measurement["world_size"]))],
        "rank_wall_samples_us": [samples for _ in range(int(measurement["world_size"]))],
        "data_validation_pass": True,
        "environment": {"torch": "synthetic", "cuda": "synthetic", "nccl": [0, 0, 0]},
    }
    if measurement["parallelism"] == "tp":
        row["sglang_dispatch_components"] = {"multi_node": measurement["topology_level"] != "L1"}
    return row


def main() -> None:
    spec = contract()
    plan = expand_plan(inventory(), "a" * 64, "2026-08-15T00:00:30+00:00", "f" * 40, spec)
    audit = validate_plan(plan, spec)
    assert audit["measurements"] == 24
    with tempfile.TemporaryDirectory(prefix="phase39-synthetic-raw-") as temporary:
        raw = Path(temporary)
        payloads = [int(value) for value in spec["measurement_contract"]["payload_bytes"]]
        iterations = int(spec["measurement_contract"]["timed_iterations"])
        level_factor = {"L1": 1.0, "L2": 2.0, "L3": 4.0}
        for measurement in plan["measurements"]:
            for repeat in range(5):
                path = raw / measurement["measurement_id"] / f"repeat_{repeat:02d}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = []
                for payload in payloads:
                    base = 5.0 * level_factor[measurement["topology_level"]] + payload / 1e6
                    base *= 1.0 + 0.05 * int(measurement["replica_id"]) + 0.001 * repeat
                    directions = ("rank0_to_rank1", "rank1_to_rank0") if measurement["parallelism"] == "pp" else ("collective",)
                    for direction_index, direction in enumerate(directions):
                        rows.append(record(plan, measurement, repeat, payload, direction, base * (1 + 0.01 * direction_index), iterations))
                path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        raw_audit = validate_raw(plan, raw, spec)
        assert raw_audit["complete"] and not raw_audit["final_high_variance"]
        assert raw_audit["raw_manifest"]["file_count"] == 120
        assert raw_audit["raw_manifest"]["record_count"] == 4050
        curves_payload, registry = build_curves(plan, raw_audit, spec)
        assert len(curves_payload["curves"]) == 12
        assert not registry["high_cross_replica_spread"]

        predictions, targets = input_rows(ROOT, spec)
        phase_rows, _ = cost_phase_rows(predictions, targets, curves_payload["curves"])
        total_rows = combined_cost_rows(phase_rows)
        metrics = aggregate_cost_metrics([*phase_rows, *total_rows])
        histogram = frozen_histogram_metrics(predictions, targets, spec["phase34_bin_edges_bytes"])
        paths = {item["name"]: ROOT / item["path"] for item in spec["pinned_inputs"]}
        invariant = compare_histogram_metrics(histogram, read_csv(paths["phase34d_histogram_metrics"]))
        proxy = compare_phase35(metrics, read_csv(paths["phase35_cost_metrics"]))
        rankings, decisions, decision_metrics = placement_decisions(total_rows)
        assert len(predictions) == 2592 and len(targets) == 2592
        assert len(phase_rows) == 7776 and len(total_rows) == 3888
        assert len(metrics) == 234 and invariant["ok"] and len(histogram) == 84
        observed = (len(proxy), len(rankings), len(decisions), len(decision_metrics))
        assert observed == (180, 3888, 1296, 24), observed
    print(json.dumps({
        "status": "PASS",
        "measurements": 24,
        "minimum_raw_files": 120,
        "minimum_raw_records": 4050,
        "curves": 12,
        "phase_cost_rows": 7776,
        "total_cost_rows": 3888,
        "cost_metrics": 234,
        "placement_decisions": 1296,
    }, indent=2))


if __name__ == "__main__":
    main()
