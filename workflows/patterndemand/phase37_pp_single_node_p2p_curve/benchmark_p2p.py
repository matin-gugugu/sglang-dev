#!/usr/bin/env python3
"""测量SGLang PP异步GPU tensor P2P原语；由torchrun以两进程启动。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-id", type=int, required=True)
    parser.add_argument("--topology-category", required=True)
    parser.add_argument("--raw-link", required=True)
    parser.add_argument("--physical-gpus", required=True)
    parser.add_argument("--payload-bytes", type=int, nargs="+", required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def nccl_version() -> object:
    try:
        return list(torch.cuda.nccl.version())
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("Phase37 benchmark固定为两个进程和一个有向PP boundary")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method="env://")
    cpu_group = dist.new_group(ranks=[0, 1], backend="gloo")
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    element_size = torch.empty((), dtype=dtype).element_size()
    sizes = sorted(set(args.payload_bytes))
    if any(size <= 0 or size % element_size for size in sizes):
        raise RuntimeError("payload必须是正数且能被bfloat16元素大小整除")
    physical_gpus = [int(value) for value in args.physical_gpus.split(",")]
    maximum_elements = max(sizes) // element_size
    storage = torch.zeros((maximum_elements,), dtype=dtype, device=device)
    peer_access = torch.cuda.can_device_access_peer(local_rank, 1 - local_rank)

    for payload_bytes in sizes:
        tensor = storage[: payload_bytes // element_size]
        for sender_rank, receiver_rank in ((0, 1), (1, 0)):
            expected_value = 1.25 if sender_rank == 0 else 2.5
            if rank == sender_rank:
                tensor.fill_(expected_value)
            else:
                tensor.zero_()
            torch.cuda.synchronize()
            dist.barrier(group=cpu_group)
            for _ in range(args.warmup):
                dist.barrier(group=cpu_group)
                work = dist.isend(tensor, dst=receiver_rank) if rank == sender_rank else dist.irecv(tensor, src=sender_rank)
                work.wait()
                torch.cuda.synchronize()
            dist.barrier(group=cpu_group)

            starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
            local_cuda_us = []
            local_wall_us = []
            for start, end in zip(starts, ends):
                dist.barrier(group=cpu_group)
                start.record()
                wall_start = time.perf_counter_ns()
                work = dist.isend(tensor, dst=receiver_rank) if rank == sender_rank else dist.irecv(tensor, src=sender_rank)
                work.wait()
                wall_end = time.perf_counter_ns()
                end.record()
                torch.cuda.synchronize()
                local_cuda_us.append(float(start.elapsed_time(end) * 1000.0))
                local_wall_us.append(float((wall_end - wall_start) / 1000.0))
            gathered_cuda = [None, None]
            gathered_wall = [None, None]
            dist.all_gather_object(gathered_cuda, local_cuda_us, group=cpu_group)
            dist.all_gather_object(gathered_wall, local_wall_us, group=cpu_group)
            valid = True
            if rank == receiver_rank:
                valid = bool(torch.all(tensor[: min(tensor.numel(), 64)] == torch.tensor(expected_value, dtype=dtype, device=device)).item())
            validations = [None, None]
            dist.all_gather_object(validations, valid, group=cpu_group)

            if rank != 0:
                dist.barrier(group=cpu_group)
                continue
            completion_cuda_us = [max(gathered_cuda[0][index], gathered_cuda[1][index]) for index in range(args.iterations)]
            completion_wall_us = [max(gathered_wall[0][index], gathered_wall[1][index]) for index in range(args.iterations)]
            median_us = statistics.median(completion_cuda_us)
            record = {
                "schema_version": "phase37-pp-p2p-raw-v1",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "repeat_id": args.repeat_id,
                "hostname": platform.node(),
                "topology_category": args.topology_category,
                "raw_link": args.raw_link,
                "physical_gpus": physical_gpus,
                "direction": f"gpu{physical_gpus[sender_rank]}_to_gpu{physical_gpus[receiver_rank]}",
                "sender_physical_gpu": physical_gpus[sender_rank],
                "receiver_physical_gpu": physical_gpus[receiver_rank],
                "op": "p2p_send_tensor",
                "backend": "torch.distributed.isend_irecv_nccl_device_group",
                "production_contract": "SGLang GroupCoordinator.send_tensor_dict async GPU tensor primitive; CPU metadata excluded",
                "payload_scope": "sender-counted logical tensor bytes",
                "payload_bytes": payload_bytes,
                "dtype": "bfloat16",
                "warmup_iterations": args.warmup,
                "timed_iterations": args.iterations,
                "rank_scope": "max-completion-across-sender-and-receiver",
                "latency_us": {
                    "min": min(completion_cuda_us),
                    "median": median_us,
                    "mean": statistics.fmean(completion_cuda_us),
                    "p95": percentile(completion_cuda_us, 0.95),
                    "p99": percentile(completion_cuda_us, 0.99),
                    "max": max(completion_cuda_us)
                },
                "diagnostic_wall_latency_us": {
                    "median": statistics.median(completion_wall_us),
                    "p95": percentile(completion_wall_us, 0.95)
                },
                "algorithmic_bandwidth_GBps": payload_bytes / (median_us / 1e6) / 1e9,
                "completion_cuda_samples_us": completion_cuda_us,
                "rank_cuda_samples_us": gathered_cuda,
                "rank_wall_samples_us": gathered_wall,
                "data_validation_pass": all(validations),
                "cuda_peer_access": peer_access,
                "environment": {
                    "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "nccl": nccl_version(),
                    "device_name": torch.cuda.get_device_name(local_rank),
                    "device_uuid_rank0": str(torch.cuda.get_device_properties(local_rank).uuid),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")
                }
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("a", encoding="utf-8") as target:
                target.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            print(f"repeat={args.repeat_id} topology={args.topology_category} direction={record['direction']} payload={payload_bytes} median={median_us:.3f}us p95={record['latency_us']['p95']:.3f}us", flush=True)
            dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
