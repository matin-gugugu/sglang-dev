#!/usr/bin/env python3
"""Render three exact Phase63 rank commands for one shard/repeat."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts import load_json, measurement_by_id, validate_plan

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology-plan", type=Path, required=True)
    parser.add_argument("--measurement-id", required=True)
    parser.add_argument("--repeat-id", type=int, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    args = parser.parse_args()
    plan_path = args.topology_plan.expanduser().resolve()
    plan = load_json(plan_path)
    validate_plan(plan)
    measurement = measurement_by_id(plan, args.measurement_id)
    output = args.raw_dir.expanduser().resolve() / measurement["measurement_id"] / f"repeat_{args.repeat_id:02d}.jsonl"
    commands = []
    for endpoint in measurement["ranks"]:
        argv = [
            "env", "-u", "MC_FORCE_TCP", "-u", "MC_FORCE_MNNVL", "-u", "MC_INTRANODE_NVLINK", "-u", "SGLANG_MOONCAKE_CUSTOM_MEM_POOL",
            f"CUDA_VISIBLE_DEVICES={endpoint['physical_gpu']}",
            f"PYTHONPATH={ROOT / 'python'}",
            "MOONCAKE_PROTOCOL=rdma", "WITH_NVIDIA_PEERMEM=0", "SGLANG_DISAGG_STAGING_BUFFER=0",
            "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
            f"RANK={endpoint['rank']}", "WORLD_SIZE=3", "LOCAL_RANK=0",
            f"MASTER_ADDR={args.master_addr}", f"MASTER_PORT={args.master_port}",
            "python3", str(HERE / "benchmark_mooncake_external.py"),
            "--expected-workflow-commit", plan["workflow_commit"],
            "--topology-plan", str(plan_path),
            "--measurement-id", measurement["measurement_id"],
            "--repeat-id", str(args.repeat_id),
            "--output", str(output),
        ]
        commands.append({
            "rank": endpoint["rank"],
            "role": endpoint["role"],
            "host": endpoint["host"],
            "host_aliases": endpoint["host_aliases"],
            "physical_gpu": endpoint["physical_gpu"],
            "ib_device": endpoint["ib_device"],
            "argv": argv,
        })
    node_count = len({row["host"] for row in commands})
    print(json.dumps({
        "schema_version": "phase63-launch-command-set-v1",
        "measurement_id": measurement["measurement_id"],
        "model_id": measurement["model_id"],
        "configuration": measurement["configuration"],
        "topology_level": measurement["topology_level"],
        "repeat_id": args.repeat_id,
        "phase62_comparability": measurement["phase62_comparability"],
        "resource_contract": {
            "simultaneous_gpu_processes": 3,
            "simultaneous_nodes": node_count,
            "maximum_simultaneous_nodes": 2,
            "global_peak_simultaneous_nodes": 2,
            "maximum_concurrent_measurement_shards": 1,
            "must_finish_this_shard_before_starting_any_other_shard": True,
            "replica_and_topology_allocations_are_sequential": True,
            "four_node_allocation_required": False,
            "four_node_simultaneous_allocation_forbidden": True,
            "unused_inventory_slot_not_launched": True,
        },
        "world_size": 3,
        "commands_must_start_concurrently": True,
        "one_process_per_command": True,
        "shared_output_must_not_exist": str(output),
        "commands": commands,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
