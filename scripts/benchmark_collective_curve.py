#!/usr/bin/env python3
import argparse
import json
import math
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-id", type=int, required=True)
    parser.add_argument(
        "--op", choices=("all_reduce", "all_gather"), default="all_reduce"
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--min-bytes", type=int, default=8 * 1024)
    parser.add_argument("--max-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument(
        "--extra-bytes",
        type=int,
        nargs="*",
        default=[48 * 1024],
        help="Extra observed payload sizes added to the power-of-two sweep.",
    )
    parser.add_argument(
        "--topology",
        default="single-node-nvlink",
        help=(
            "Physical placement label recorded in every row, for example "
            "single-node-nvlink or cross-node-same-rdma-pod."
        ),
    )
    parser.add_argument(
        "--timing-mode",
        choices=("steady_state", "rendezvous"),
        default="steady_state",
        help=(
            "steady_state preserves the original back-to-back measurement. "
            "rendezvous synchronizes all ranks before each isolated collective."
        ),
    )
    parser.add_argument(
        "--transport-label",
        default="auto",
        help="Human-readable transport label such as nvlink, roce, or tcp.",
    )
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_sizes(min_bytes, max_bytes, extra_bytes):
    if min_bytes <= 0 or max_bytes < min_bytes:
        raise ValueError("invalid payload range")
    sizes = set(extra_bytes)
    value = 1 << math.ceil(math.log2(min_bytes))
    while value <= max_bytes:
        sizes.add(value)
        value *= 2
    return sorted(size for size in sizes if min_bytes <= size <= max_bytes)


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    dtype = torch.bfloat16
    element_size = torch.empty((), dtype=dtype).element_size()
    sizes = build_sizes(args.min_bytes, args.max_bytes, args.extra_bytes)
    divisor = element_size if args.op == "all_reduce" else element_size * world_size
    if any(size % divisor for size in sizes):
        raise ValueError(
            f"all {args.op} payload sizes must be divisible by {divisor} bytes"
        )

    max_numel = max(sizes) // element_size
    if args.op == "all_reduce":
        input_storage = torch.zeros(max_numel, dtype=dtype, device="cuda")
        output_storage = None
        payload_scope = "representative-rank-logical-input"
        ring_factor = 2 * (world_size - 1) / world_size
    else:
        input_storage = torch.zeros(max_numel // world_size, dtype=dtype, device="cuda")
        output_storage = torch.empty(max_numel, dtype=dtype, device="cuda")
        payload_scope = "logical-gathered-output"
        ring_factor = (world_size - 1) / world_size
    device_name = torch.cuda.get_device_name(local_rank)
    device_uuid = torch.cuda.get_device_properties(local_rank).uuid
    commit = git_commit()
    local_rank_metadata = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "device_name": device_name,
        "device_uuid": str(device_uuid),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    rank_metadata = [None for _ in range(world_size)]
    dist.all_gather_object(rank_metadata, local_rank_metadata)
    hostnames = sorted({metadata["hostname"] for metadata in rank_metadata})

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    for payload_bytes in sizes:
        payload_numel = payload_bytes // element_size
        if args.op == "all_reduce":
            input_tensor = input_storage[:payload_numel]
            output_tensor = None
        else:
            input_tensor = input_storage[: payload_numel // world_size]
            output_tensor = output_storage[:payload_numel]

        def run_collective():
            if args.op == "all_reduce":
                dist.all_reduce(input_tensor)
            else:
                dist.all_gather_into_tensor(output_tensor, input_tensor)

        dist.barrier()
        for _ in range(args.warmup):
            if args.timing_mode == "rendezvous":
                dist.barrier(device_ids=[local_rank])
                torch.cuda.synchronize()
            run_collective()
        torch.cuda.synchronize()
        dist.barrier()

        if args.timing_mode == "steady_state":
            starts = [
                torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)
            ]
            ends = [
                torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)
            ]
            for start, end in zip(starts, ends):
                start.record()
                run_collective()
                end.record()
            torch.cuda.synchronize()
            local_samples_us = [
                float(start.elapsed_time(end) * 1000.0)
                for start, end in zip(starts, ends)
            ]
        else:
            local_samples_us = []
            for _ in range(args.iterations):
                dist.barrier(device_ids=[local_rank])
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run_collective()
                end.record()
                torch.cuda.synchronize()
                local_samples_us.append(float(start.elapsed_time(end) * 1000.0))

        gathered_samples = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_samples, local_samples_us)
        dist.barrier()

        if rank == 0:
            completion_samples_us = [
                max(rank_samples[index] for rank_samples in gathered_samples)
                for index in range(args.iterations)
            ]
            median_us = float(np.median(completion_samples_us))
            seconds = median_us / 1_000_000.0
            algorithmic_gbps = payload_bytes / seconds / 1e9
            record = {
                "schema_version": "collective-cost-v2",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "repeat_id": args.repeat_id,
                "hostname": socket.gethostname(),
                "hostnames": hostnames,
                "node_count": len(hostnames),
                "rank_layout": rank_metadata,
                "topology": args.topology,
                "transport_label": args.transport_label,
                "timing_mode": args.timing_mode,
                "op": args.op,
                "backend": "nccl",
                "group_size": world_size,
                "rank_scope": (
                    "max-completion-across-ranks"
                    if args.timing_mode == "steady_state"
                    else "max-rank-local-envelope-after-rendezvous"
                ),
                "payload_scope": payload_scope,
                "payload_bytes": payload_bytes,
                "input_payload_bytes_per_rank": (
                    payload_bytes
                    if args.op == "all_reduce"
                    else payload_bytes // world_size
                ),
                "dtype": str(dtype).removeprefix("torch."),
                "warmup_iterations": args.warmup,
                "timed_iterations": args.iterations,
                "latency_us": {
                    "min": float(min(completion_samples_us)),
                    "median": median_us,
                    "mean": float(np.mean(completion_samples_us)),
                    "p95": percentile(completion_samples_us, 95),
                    "p99": percentile(completion_samples_us, 99),
                    "max": float(max(completion_samples_us)),
                },
                "algorithmic_bandwidth_GBps": algorithmic_gbps,
                "ring_equivalent_factor": ring_factor,
                "ring_equivalent_bytes": payload_bytes * ring_factor,
                "ring_equivalent_bus_bandwidth_GBps": (algorithmic_gbps * ring_factor),
                "samples_us": completion_samples_us,
                "rank_samples_us": gathered_samples,
                "environment": {
                    "git_commit": commit,
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "nccl_version": list(torch.cuda.nccl.version()),
                    "device_name": device_name,
                    "device_uuid_rank0": str(device_uuid),
                    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "master_addr": os.environ.get("MASTER_ADDR"),
                    "master_port": os.environ.get("MASTER_PORT"),
                    "nccl_env": {
                        key: os.environ.get(key)
                        for key in (
                            "NCCL_ALGO",
                            "NCCL_CROSS_NIC",
                            "NCCL_DEBUG",
                            "NCCL_IB_DISABLE",
                            "NCCL_IB_GID_INDEX",
                            "NCCL_IB_HCA",
                            "NCCL_NET_GDR_LEVEL",
                            "NCCL_SOCKET_IFNAME",
                        )
                        if os.environ.get(key) is not None
                    },
                },
            }
            with args.output.open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"repeat={args.repeat_id} op={args.op} payload={payload_bytes} "
                f"median={median_us:.3f} us p95={record['latency_us']['p95']:.3f} us "
                f"alg_bw={algorithmic_gbps:.2f} GB/s",
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
