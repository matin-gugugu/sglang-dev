#!/usr/bin/env python3
"""Render exact per-node torchrun argv arrays; execution remains environment-owned."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from contracts import load_json, measurement_by_id, validate_plan

HERE = Path(__file__).resolve().parent


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
    raw_dir = args.raw_dir.expanduser().resolve()
    output = raw_dir / measurement["measurement_id"] / f"repeat_{args.repeat_id:02d}.jsonl"
    by_host = defaultdict(list)
    for rank in measurement["ranks"]:
        by_host[rank["host"]].append(rank)
    hosts = list(by_host)
    commands = []
    for node_rank, host in enumerate(hosts):
        ranks = by_host[host]
        cuda_visible = ",".join(str(row["physical_gpu"]) for row in ranks)
        argv = [
            "env", f"CUDA_VISIBLE_DEVICES={cuda_visible}", "torchrun",
            "--nnodes", str(len(hosts)),
            "--nproc-per-node", str(len(ranks)),
            "--node-rank", str(node_rank),
            "--master-addr", args.master_addr,
            "--master-port", str(args.master_port),
            str(HERE / "benchmark_distributed.py"),
            "--expected-workflow-commit", plan["workflow_commit"],
            "--topology-plan", str(plan_path),
            "--measurement-id", measurement["measurement_id"],
            "--repeat-id", str(args.repeat_id),
            "--output", str(output),
        ]
        commands.append({
            "host": host,
            "host_aliases": ranks[0]["host_aliases"],
            "node_rank": node_rank,
            "global_ranks": [row["rank"] for row in ranks],
            "physical_gpus": [row["physical_gpu"] for row in ranks],
            "argv": argv,
        })
    print(json.dumps({
        "schema_version": "phase39-launch-command-set-v1",
        "measurement_id": measurement["measurement_id"],
        "repeat_id": args.repeat_id,
        "world_size": measurement["world_size"],
        "commands_must_start_concurrently": True,
        "commands": commands,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
