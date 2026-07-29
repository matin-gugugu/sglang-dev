#!/usr/bin/env python3
"""Measure pure GPU-kernel cost for SGLang CustomAllReduceV2."""

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
from torch.autograd import DeviceType

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce
from sglang.srt.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    get_tensor_model_parallel_group,
    initialize_model_parallel,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-id", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--min-bytes", type=int, default=8 * 1024)
    parser.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument(
        "--extra-bytes",
        type=int,
        nargs="*",
        default=[48 * 1024],
        help="Extra observed payload sizes added to the power-of-two sweep.",
    )
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def coefficient_of_variation(values):
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    return float(np.std(array) / mean) if mean else 0.0


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


def is_custom_allreduce_kernel(name):
    lowered = name.lower()
    return any(
        algorithm in lowered
        for algorithm in (
            "all_reduce_one_shot_push_kernel",
            "all_reduce_one_shot_pull_kernel",
            "all_reduce_one_shot_kernel",
            "all_reduce_two_shot_push_kernel",
            "all_reduce_two_shot_pull_kernel",
        )
    )


def kernel_durations_us(profiler):
    matched = []
    names = {}
    for event in profiler.events():
        if event.device_type != DeviceType.CUDA:
            continue
        if not is_custom_allreduce_kernel(event.name):
            continue
        duration_us = float(event.time_range.elapsed_us())
        matched.append(duration_us)
        names[event.name] = names.get(event.name, 0) + 1
    return matched, names


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
    )
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    communicator = get_tensor_model_parallel_group()
    cpu_group = communicator.cpu_group
    custom_communicator = communicator.ca_comm
    if custom_communicator is None or custom_communicator.disabled:
        raise RuntimeError("SGLang custom AllReduce is disabled for this topology")

    dtype = torch.bfloat16
    element_size = torch.empty((), dtype=dtype).element_size()
    sizes = build_sizes(args.min_bytes, args.max_bytes, args.extra_bytes)
    if max(sizes) > custom_communicator.max_size:
        raise ValueError(
            f"requested {max(sizes)} bytes exceeds custom backend maximum "
            f"{custom_communicator.max_size}"
        )
    if any(size % element_size for size in sizes):
        raise ValueError("all payload sizes must be divisible by dtype element size")

    storage = torch.zeros(max(sizes) // element_size, dtype=dtype, device=device)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    for payload_bytes in sizes:
        tensor = storage[: payload_bytes // element_size]
        if not custom_communicator.should_custom_ar(tensor):
            raise RuntimeError(f"custom backend rejected payload {payload_bytes}")
        algorithm = custom_communicator._determine_algo(tensor).name

        dist.barrier(group=cpu_group)
        for _ in range(args.warmup):
            output_tensor = tensor_model_parallel_all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            acc_events=True,
        ) as profiler:
            for _ in range(args.iterations):
                dist.barrier(group=cpu_group)
                output_tensor = tensor_model_parallel_all_reduce(tensor)
                torch.cuda.synchronize()
        del output_tensor

        local_samples_us, local_kernel_names = kernel_durations_us(profiler)
        if len(local_samples_us) != args.iterations:
            raise RuntimeError(
                f"rank {rank} payload {payload_bytes}: expected {args.iterations} "
                f"custom kernels, found {len(local_samples_us)}; "
                f"matched_names={local_kernel_names}"
            )

        gathered_samples = [None for _ in range(world_size)]
        gathered_names = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_samples, local_samples_us, group=cpu_group)
        dist.all_gather_object(gathered_names, local_kernel_names, group=cpu_group)

        if rank == 0:
            intrinsic_samples_us = [
                min(rank_samples[index] for rank_samples in gathered_samples)
                for index in range(args.iterations)
            ]
            completion_samples_us = [
                max(rank_samples[index] for rank_samples in gathered_samples)
                for index in range(args.iterations)
            ]
            rank_skew_samples_us = [
                completion - intrinsic
                for intrinsic, completion in zip(
                    intrinsic_samples_us, completion_samples_us
                )
            ]
            median_us = float(np.median(intrinsic_samples_us))
            ring_factor = 2 * (world_size - 1) / world_size
            record = {
                "schema_version": "collective-kernel-cost-v1",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "repeat_id": args.repeat_id,
                "hostname": socket.gethostname(),
                "topology": "single-node-nvlink",
                "op": "all_reduce",
                "backend": "sglang_custom_all_reduce_v2",
                "algorithm": algorithm,
                "group_size": world_size,
                "latency_scope": ("skew-free-intrinsic-lower-envelope-across-ranks"),
                "payload_scope": "representative-rank-logical-input",
                "payload_bytes": payload_bytes,
                "dtype": str(dtype).removeprefix("torch."),
                "warmup_iterations": args.warmup,
                "timed_iterations": args.iterations,
                "latency_us": {
                    "min": float(min(intrinsic_samples_us)),
                    "median": median_us,
                    "mean": float(np.mean(intrinsic_samples_us)),
                    "p95": percentile(intrinsic_samples_us, 95),
                    "p99": percentile(intrinsic_samples_us, 99),
                    "max": float(max(intrinsic_samples_us)),
                    "cv": coefficient_of_variation(intrinsic_samples_us),
                },
                "completion_latency_us": {
                    "min": float(min(completion_samples_us)),
                    "median": float(np.median(completion_samples_us)),
                    "mean": float(np.mean(completion_samples_us)),
                    "p95": percentile(completion_samples_us, 95),
                    "p99": percentile(completion_samples_us, 99),
                    "max": float(max(completion_samples_us)),
                    "cv": coefficient_of_variation(completion_samples_us),
                },
                "rank_skew_us": {
                    "min": float(min(rank_skew_samples_us)),
                    "median": float(np.median(rank_skew_samples_us)),
                    "mean": float(np.mean(rank_skew_samples_us)),
                    "p95": percentile(rank_skew_samples_us, 95),
                    "p99": percentile(rank_skew_samples_us, 99),
                    "max": float(max(rank_skew_samples_us)),
                },
                "ring_equivalent_factor": ring_factor,
                "ring_equivalent_bytes": payload_bytes * ring_factor,
                "algorithmic_bandwidth_GBps": (
                    payload_bytes / (median_us / 1_000_000.0) / 1e9
                ),
                "ring_equivalent_bus_bandwidth_GBps": (
                    payload_bytes * ring_factor / (median_us / 1_000_000.0) / 1e9
                ),
                "samples_us": intrinsic_samples_us,
                "completion_samples_us": completion_samples_us,
                "rank_skew_samples_us": rank_skew_samples_us,
                "rank_samples_us": gathered_samples,
                "rank_kernel_name_counts": gathered_names,
                "backend_limits": {
                    "max_push_size": custom_communicator.max_push_size,
                    "max_pull_size": custom_communicator.max_pull_size,
                    "max_size": custom_communicator.max_size,
                    "one_shot_push_threshold": (
                        custom_communicator.config.one_shot_push_threshold
                    ),
                    "one_shot_pull_threshold": (
                        custom_communicator.config.one_shot_pull_threshold
                    ),
                },
                "environment": {
                    "git_commit": git_commit(),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "nccl_version": list(torch.cuda.nccl.version()),
                    "device_name": torch.cuda.get_device_name(local_rank),
                    "device_uuid_rank0": str(
                        torch.cuda.get_device_properties(local_rank).uuid
                    ),
                    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
            }
            with args.output.open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"repeat={args.repeat_id} tp={world_size} payload={payload_bytes} "
                f"algorithm={algorithm} kernel_median={median_us:.3f} us "
                f"p95={record['latency_us']['p95']:.3f} us",
                flush=True,
            )

    dist.barrier(group=cpu_group)
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
