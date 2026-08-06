#!/usr/bin/env python3
"""Measure the inference communication paths used by Phase 14C workloads."""

import argparse
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.autograd import DeviceType

from extract_inference_comm_trace import classify_collective_kernel
from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce
from sglang.srt.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    get_tensor_model_parallel_group,
    initialize_model_parallel,
)
from sglang.srt.layers.flashinfer_comm_fusion import (
    flashinfer_allreduce_residual_rmsnorm,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-id", type=int, required=True)
    parser.add_argument(
        "--op",
        choices=("all_reduce", "fused_allreduce_residual_rmsnorm"),
        required=True,
    )
    parser.add_argument("--payload-bytes", type=int, nargs="+", required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=2048)
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summary(values):
    return {
        "min": float(min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": float(max(values)),
    }


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def matching_kernel_samples(profiler):
    samples = []
    names = {}
    for event in profiler.profiler.kineto_results.events():
        if event.device_type() != DeviceType.CUDA:
            continue
        backend = classify_collective_kernel(event.name())
        if backend is None:
            continue
        samples.append(
            (
                backend,
                float(event.start_ns()) / 1000.0,
                float(event.end_ns()) / 1000.0,
            )
        )
        names[event.name()] = names.get(event.name(), 0) + 1
    return samples, names


def main():
    args = parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup nonnegative")
    payloads = sorted(set(args.payload_bytes))
    if any(payload <= 0 or payload % 2 for payload in payloads):
        raise ValueError("all payloads must be positive and divisible by bf16 bytes")

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
    set_global_server_args_for_scheduler(
        ServerArgs(
            model_path="dummy",
            flashinfer_allreduce_fusion_backend="mnnvl",
        )
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    communicator = get_tensor_model_parallel_group()
    cpu_group = communicator.cpu_group
    dtype = torch.bfloat16
    element_size = torch.empty((), dtype=dtype).element_size()

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    max_payload = max(payloads)
    all_reduce_storage = torch.zeros(
        max_payload // element_size, dtype=dtype, device=device
    )
    weight = torch.ones(args.hidden_size, dtype=dtype, device=device)

    for payload_bytes in payloads:
        if args.op == "fused_allreduce_residual_rmsnorm":
            row_bytes = args.hidden_size * element_size
            if payload_bytes % row_bytes:
                raise ValueError(
                    f"fused payload {payload_bytes} is not divisible by "
                    f"hidden_size*dtype_bytes={row_bytes}"
                )
            token_count = payload_bytes // row_bytes
            input_tensor = torch.zeros(
                (token_count, args.hidden_size), dtype=dtype, device=device
            )
            residual = torch.zeros_like(input_tensor)
            backend_proxy = "flashinfer_mnnvl:auto_one_or_two_shot"

            def run_collective():
                output = flashinfer_allreduce_residual_rmsnorm(
                    input_tensor=input_tensor,
                    residual=residual,
                    weight=weight,
                    eps=1e-6,
                    max_token_num=max(token_count, 2048),
                )
                if output[0] is None:
                    raise RuntimeError("FlashInfer fused AllReduce path was unavailable")
                return output

        else:
            token_count = None
            input_tensor = all_reduce_storage[: payload_bytes // element_size]
            residual = None
            custom = communicator.ca_comm
            if (
                custom is not None
                and not custom.disabled
                and custom.should_custom_ar(input_tensor)
            ):
                backend_proxy = f"sglang_custom:{custom._determine_algo(input_tensor).name.lower()}"
            else:
                backend_proxy = "nccl:all_reduce"

            def run_collective():
                return tensor_model_parallel_all_reduce(input_tensor)

        dist.barrier(group=cpu_group)
        for _ in range(args.warmup):
            run_collective()
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
                run_collective()
                torch.cuda.synchronize()

        local_samples, local_names = matching_kernel_samples(profiler)
        if len(local_samples) != args.iterations:
            raise RuntimeError(
                f"rank={rank} op={args.op} payload={payload_bytes}: expected "
                f"{args.iterations} communication kernels, found {len(local_samples)}; "
                f"kernel_names={local_names}"
            )

        gathered_samples = [None for _ in range(world_size)]
        gathered_names = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_samples, local_samples, group=cpu_group)
        dist.all_gather_object(gathered_names, local_names, group=cpu_group)

        if rank == 0:
            backend_sequences = [
                [backend for backend, _, _ in rank_samples]
                for rank_samples in gathered_samples
            ]
            if any(sequence != backend_sequences[0] for sequence in backend_sequences):
                raise RuntimeError(
                    f"backend sequence differs across ranks for {args.op}/{payload_bytes}"
                )
            if len(set(backend_sequences[0])) != 1:
                raise RuntimeError(
                    f"backend changed across iterations for {args.op}/{payload_bytes}: "
                    f"{sorted(set(backend_sequences[0]))}"
                )
            rank_durations = [
                [end_us - start_us for _, start_us, end_us in rank_samples]
                for rank_samples in gathered_samples
            ]
            rank_starts = [
                [start_us for _, start_us, _ in rank_samples]
                for rank_samples in gathered_samples
            ]
            rank_ends = [
                [end_us for _, _, end_us in rank_samples]
                for rank_samples in gathered_samples
            ]
            intrinsic = [
                min(rank_samples[index] for rank_samples in rank_durations)
                for index in range(args.iterations)
            ]
            completion = [
                max(rank_samples[index] for rank_samples in rank_durations)
                for index in range(args.iterations)
            ]
            post_rendezvous = [
                max(rank_samples[index] for rank_samples in rank_ends)
                - max(rank_samples[index] for rank_samples in rank_starts)
                for index in range(args.iterations)
            ]
            if any(value < 0 for value in post_rendezvous):
                raise RuntimeError("negative post-rendezvous interval")
            record = {
                "schema_version": "phase14f-backend-cost-v2",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "repeat_id": args.repeat_id,
                "hostname": socket.gethostname(),
                "topology": "single-node-nvlink",
                "op": args.op,
                "group_size": world_size,
                "payload_scope": "representative-rank-logical-input",
                "payload_bytes": payload_bytes,
                "hidden_size": args.hidden_size if token_count is not None else None,
                "token_count": token_count,
                "dtype": "bfloat16",
                "backend_proxy_pre_run": backend_proxy,
                "observed_backend_audit_only": backend_sequences[0][0],
                "latency_scope": "collective-kernel-duration-across-aligned-ranks",
                "intrinsic_latency_us": summary(intrinsic),
                "completion_latency_us": summary(completion),
                "post_rendezvous_latency_us": summary(post_rendezvous),
                "rank_skew_us": summary(
                    [right - left for left, right in zip(intrinsic, completion)]
                ),
                "warmup_iterations": args.warmup,
                "timed_iterations": args.iterations,
                "intrinsic_samples_us": intrinsic,
                "completion_samples_us": completion,
                "post_rendezvous_samples_us": post_rendezvous,
                "rank_samples_us": rank_durations,
                "rank_kernel_name_counts": gathered_names,
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
                f"repeat={args.repeat_id} tp={world_size} op={args.op} "
                f"payload={payload_bytes} proxy={backend_proxy} "
                f"observed={backend_sequences[0][0]} "
                f"post_median={record['post_rendezvous_latency_us']['median']:.3f} us",
                flush=True,
            )

        dist.barrier(group=cpu_group)

    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
