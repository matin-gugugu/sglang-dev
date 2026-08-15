#!/usr/bin/env python3
"""Distributed Phase39 worker for PP P2P or SGLang TP all-reduce raw shards."""

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

from contracts import contract, load_json, measurement_by_id, measurement_sha, validate_plan

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--topology-plan", type=Path, required=True)
    parser.add_argument("--measurement-id", required=True)
    parser.add_argument("--repeat-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def hostname_matches(actual: str, aliases: list[str]) -> bool:
    candidates = {actual, actual.split(".", 1)[0]}
    expected = {value for alias in aliases for value in (alias, alias.split(".", 1)[0])}
    return bool(candidates & expected)


def nccl_version(torch_module) -> object:
    try:
        return list(torch_module.cuda.nccl.version())
    except Exception:
        return None


def append_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    spec = contract()
    plan_path = args.topology_plan.expanduser().resolve()
    plan = load_json(plan_path)
    validate_plan(plan, spec)
    if plan["workflow_commit"] != args.expected_workflow_commit:
        raise RuntimeError("topology plan workflow commit与命令不一致")
    actual_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual_head != args.expected_workflow_commit:
        raise RuntimeError(f"benchmark HEAD不等于W39：{actual_head}")
    measurement = measurement_by_id(plan, args.measurement_id)
    repeat_id = args.repeat_id
    measurement_contract = spec["measurement_contract"]
    allowed_repeats = range(int(measurement_contract["maximum_independent_repeats"]))
    if repeat_id not in allowed_repeats:
        raise RuntimeError(f"repeat-id超出合同：{repeat_id}")
    output = args.output.expanduser().resolve()
    if ROOT.resolve() in output.parents:
        raise RuntimeError("raw output必须在Git仓库外")
    expected_name = f"repeat_{repeat_id:02d}.jsonl"
    if output.parent.name != measurement["measurement_id"] or output.name != expected_name:
        raise RuntimeError(f"raw路径必须为.../{measurement['measurement_id']}/{expected_name}")
    if output.exists():
        raise RuntimeError(f"拒绝覆盖已有raw shard：{output}")

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != int(measurement["world_size"]):
        raise RuntimeError({"world_size": world_size, "expected": measurement["world_size"]})
    rank_spec = measurement["ranks"][rank]
    if rank_spec["local_rank"] != local_rank:
        raise RuntimeError({"local_rank": local_rank, "expected": rank_spec["local_rank"]})
    actual_host = platform.node()
    if not hostname_matches(actual_host, rank_spec["host_aliases"]):
        raise RuntimeError({"hostname": actual_host, "expected_aliases": rank_spec["host_aliases"]})
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if len(visible) <= local_rank or visible[local_rank] != str(rank_spec["physical_gpu"]):
        raise RuntimeError({"CUDA_VISIBLE_DEVICES": visible, "rank_spec": rank_spec})
    torch.cuda.set_device(local_rank)

    tp_group = None
    if measurement["parallelism"] == "tp":
        from sglang.srt.distributed.parallel_state import (
            get_tp_group,
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method="env://",
            local_rank=local_rank,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1, backend="nccl")
        tp_group = get_tp_group()
    else:
        dist.init_process_group("nccl", init_method="env://")
    cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    payloads = [int(value) for value in measurement_contract["payload_bytes"]]
    element_size = torch.empty((), dtype=dtype).element_size()
    if any(value <= 0 or value % element_size for value in payloads):
        raise RuntimeError("payload grid不能被bfloat16元素大小整除")
    storage = torch.empty((max(payloads) // element_size,), dtype=dtype, device=device)
    plan_sha = plan["plan_sha256"]
    shard_started = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    warmup = int(measurement_contract["warmup_iterations"])
    iterations = int(measurement_contract["timed_iterations"])

    def base_record(payload_bytes: int) -> dict:
        return {
            "schema_version": "phase39-distributed-raw-v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "shard_started_at_utc": shard_started,
            "workflow_commit": actual_head,
            "plan_sha256": plan_sha,
            "measurement_sha256": measurement_sha(measurement),
            "measurement_id": measurement["measurement_id"],
            "case_key": measurement["case_key"],
            "replica_id": measurement["replica_id"],
            "placement_id": measurement["placement_id"],
            "parallelism": measurement["parallelism"],
            "topology_level": measurement["topology_level"],
            "classification_evidence": measurement["classification_evidence"],
            "rank_mapping": measurement["ranks"],
            "repeat_id": repeat_id,
            "payload_bytes": payload_bytes,
            "dtype": "bfloat16",
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": nccl_version(torch),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "rank0_device_name": torch.cuda.get_device_name(local_rank) if rank == 0 else None,
            },
        }

    for payload_bytes in payloads:
        tensor = storage[: payload_bytes // element_size]
        if measurement["parallelism"] == "pp":
            directions = ((0, 1), (1, 0))
            for sender, receiver in directions:
                expected_value = 1.25 if sender == 0 else 2.5
                if rank == sender:
                    tensor.fill_(expected_value)
                else:
                    tensor.zero_()
                torch.cuda.synchronize()
                for _ in range(warmup):
                    dist.barrier(group=cpu_group)
                    work = dist.isend(tensor, dst=receiver) if rank == sender else dist.irecv(tensor, src=sender)
                    work.wait()
                    torch.cuda.synchronize()
                local_cuda_us = []
                local_wall_us = []
                for _ in range(iterations):
                    dist.barrier(group=cpu_group)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    wall_start = time.perf_counter_ns()
                    work = dist.isend(tensor, dst=receiver) if rank == sender else dist.irecv(tensor, src=sender)
                    work.wait()
                    wall_end = time.perf_counter_ns()
                    end.record()
                    torch.cuda.synchronize()
                    local_cuda_us.append(float(start.elapsed_time(end) * 1000.0))
                    local_wall_us.append(float((wall_end - wall_start) / 1000.0))
                gathered_cuda = [None] * world_size
                gathered_wall = [None] * world_size
                dist.all_gather_object(gathered_cuda, local_cuda_us, group=cpu_group)
                dist.all_gather_object(gathered_wall, local_wall_us, group=cpu_group)
                valid = True
                if rank == receiver:
                    valid = bool(torch.all(tensor[: min(tensor.numel(), 64)] == torch.tensor(expected_value, dtype=dtype, device=device)).item())
                validations = [None] * world_size
                dist.all_gather_object(validations, valid, group=cpu_group)
                if rank == 0:
                    completion = [max(gathered_cuda[r][index] for r in range(world_size)) for index in range(iterations)]
                    completion_wall = [max(gathered_wall[r][index] for r in range(world_size)) for index in range(iterations)]
                    record = {
                        **base_record(payload_bytes),
                        "op": "p2p_send_tensor",
                        "backend": measurement_contract["pp_backend"],
                        "measurement_scope": measurement_contract["pp_scope"],
                        "direction": f"rank{sender}_to_rank{receiver}",
                        "sender_rank": sender,
                        "receiver_rank": receiver,
                        "latency_us": {
                            "min": min(completion), "median": statistics.median(completion),
                            "mean": statistics.fmean(completion), "p95": percentile(completion, 0.95),
                            "p99": percentile(completion, 0.99), "max": max(completion),
                        },
                        "diagnostic_wall_latency_us": {"median": statistics.median(completion_wall), "p95": percentile(completion_wall, 0.95)},
                        "algorithmic_bandwidth_GBps": payload_bytes / (statistics.median(completion) / 1e6) / 1e9,
                        "completion_cuda_samples_us": completion,
                        "rank_cuda_samples_us": gathered_cuda,
                        "rank_wall_samples_us": gathered_wall,
                        "data_validation_pass": all(validations),
                    }
                    append_record(output, record)
        else:
            assert tp_group is not None
            for _ in range(warmup):
                tensor.fill_(float(rank + 1))
                dist.barrier(group=cpu_group)
                result = tp_group.all_reduce(tensor)
                torch.cuda.synchronize()
            local_cuda_us = []
            local_wall_us = []
            result = tensor
            for _ in range(iterations):
                tensor.fill_(float(rank + 1))
                dist.barrier(group=cpu_group)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                wall_start = time.perf_counter_ns()
                result = tp_group.all_reduce(tensor)
                wall_end = time.perf_counter_ns()
                end.record()
                torch.cuda.synchronize()
                local_cuda_us.append(float(start.elapsed_time(end) * 1000.0))
                local_wall_us.append(float((wall_end - wall_start) / 1000.0))
            gathered_cuda = [None] * world_size
            gathered_wall = [None] * world_size
            dist.all_gather_object(gathered_cuda, local_cuda_us, group=cpu_group)
            dist.all_gather_object(gathered_wall, local_wall_us, group=cpu_group)
            expected_sum = float(world_size * (world_size + 1) // 2)
            valid = bool(torch.all(result[: min(result.numel(), 64)] == torch.tensor(expected_sum, dtype=dtype, device=device)).item())
            validations = [None] * world_size
            dist.all_gather_object(validations, valid, group=cpu_group)
            dispatch = {
                "ca_comm_available": getattr(tp_group, "ca_comm", None) is not None,
                "qr_comm_available": getattr(tp_group, "qr_comm", None) is not None,
                "pynccl_comm_available": getattr(tp_group, "pynccl_comm", None) is not None,
                "torch_symm_mem_available": getattr(tp_group, "torch_symm_mem_comm", None) is not None,
                "multi_node": len({row["host"] for row in measurement["ranks"]}) > 1,
            }
            if rank == 0:
                completion = [max(gathered_cuda[r][index] for r in range(world_size)) for index in range(iterations)]
                completion_wall = [max(gathered_wall[r][index] for r in range(world_size)) for index in range(iterations)]
                record = {
                    **base_record(payload_bytes),
                    "op": "sglang_tp_all_reduce",
                    "backend": measurement_contract["tp_backend"],
                    "measurement_scope": measurement_contract["tp_scope"],
                    "direction": "collective",
                    "sglang_dispatch_components": dispatch,
                    "latency_us": {
                        "min": min(completion), "median": statistics.median(completion),
                        "mean": statistics.fmean(completion), "p95": percentile(completion, 0.95),
                        "p99": percentile(completion, 0.99), "max": max(completion),
                    },
                    "diagnostic_wall_latency_us": {"median": statistics.median(completion_wall), "p95": percentile(completion_wall, 0.95)},
                    "algorithmic_bandwidth_GBps": payload_bytes / (statistics.median(completion) / 1e6) / 1e9,
                    "completion_cuda_samples_us": completion,
                    "rank_cuda_samples_us": gathered_cuda,
                    "rank_wall_samples_us": gathered_wall,
                    "data_validation_pass": all(validations),
                }
                append_record(output, record)
        dist.barrier(group=cpu_group)
        if rank == 0:
            print(f"measurement={measurement['measurement_id']} repeat={repeat_id} payload={payload_bytes} complete", flush=True)
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
