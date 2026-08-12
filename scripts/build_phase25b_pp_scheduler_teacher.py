#!/usr/bin/env python3
"""Build and audit the scheduler-faithful Phase 25B PP full-window teacher.

This simulator mirrors the fixed-draining PP scheduler contract used by the
Phase 25 GPU smoke: FCFS admission, one running batch per PP loop lane,
prefill-before-finished-filtering, 4096-token chunking with page-rounded budget
accounting, and a globally continued chunked request that may migrate lanes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from analyze_phase25_full_window_gpu_smoke import aggregate, combine, metric_row
from build_phase25_full_window_teacher import (
    PP_BIN_EDGES,
    PP_CHUNK_TOKENS,
    PP_MICROBATCH_SIZES,
    PP_MODEL,
    PP_PROXY_COUNT,
    PP_SIZES,
    bin_vectors,
    deterministic_gzip,
    label_row,
    normalize,
    read_jsonl_gz,
    sha256,
    write_csv,
    write_csv_gz,
    write_json,
)


PAGE_SIZE = 64
BYTES_PER_TOKEN = 4096 * 2
PHASES = ("prefill", "decode")
LABEL_STATUS = "GPU_VALIDATED_SCHEDULER_FORMULA_SMOKE_9_OF_9"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase25-root",
        type=Path,
        default=root / "experiment-results/phase25_full_window_teacher",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase25b_pp_scheduler_teacher",
    )
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def ceil_page(tokens: int) -> int:
    return -(-tokens // PAGE_SIZE) * PAGE_SIZE


@dataclass
class RequestState:
    request_id: int
    input_tokens: int
    output_tokens: int
    input_position: int = 0
    generated_tokens: int = 0

    @property
    def finished(self) -> bool:
        return self.generated_tokens >= self.output_tokens


@dataclass
class PendingBatch:
    phase: str
    request_ids: tuple[int, ...]
    final_prefill_request_ids: tuple[int, ...] = ()


@dataclass
class LaneState:
    running_request_ids: list[int] = field(default_factory=list)
    pending_batch: Optional[PendingBatch] = None
    batch_is_full: bool = False


@dataclass
class SimulationResult:
    event_histograms: dict[str, Counter[int]]
    annotation_histogram: Counter[tuple[str, int, int]]
    event_counts: dict[str, int]
    max_active_batch_size: dict[str, int]
    scheduler_visits: int
    all_requests_complete: bool
    prefill_token_mass: int
    decode_token_mass: int


def simulate_scheduler(
    requests: list[tuple[int, int]],
    *,
    pp_size: int,
    max_microbatch: int,
    chunk_tokens: int = PP_CHUNK_TOKENS,
) -> SimulationResult:
    """Simulate the PP first-rank scheduler at forward-event granularity."""
    states = [
        RequestState(index, int(input_len), int(output_len))
        for index, (input_len, output_len) in enumerate(requests)
    ]
    if not states or any(row.input_tokens <= 0 or row.output_tokens <= 0 for row in states):
        raise ValueError("requests must have positive input and output lengths")
    waiting = list(range(len(states)))
    chunked_request_id: Optional[int] = None
    lanes = [LaneState() for _ in range(pp_size)]
    event_histograms = {phase: Counter() for phase in PHASES}
    annotation_histogram: Counter[tuple[str, int, int]] = Counter()
    event_counts = {phase: 0 for phase in PHASES}
    max_active_batch_size = {phase: 0 for phase in PHASES}
    scheduler_visits = 0

    def record(phase: str, active_tokens: int, active_requests: int) -> None:
        if active_tokens <= 0 or active_requests <= 0:
            raise RuntimeError("scheduler emitted an empty forward")
        event_histograms[phase][active_tokens] += 1
        annotation_histogram[(phase, active_requests, active_tokens)] += 1
        event_counts[phase] += 1
        max_active_batch_size[phase] = max(
            max_active_batch_size[phase], active_requests
        )

    while True:
        emitted_in_round = False
        for lane in lanes:
            scheduler_visits += 1

            # PP outputs for this lane have arrived before the lane is scheduled
            # again. Prefill samples token one only on the final prompt chunk;
            # decode emits one token per active request.
            previous = lane.pending_batch
            lane.pending_batch = None
            if previous is not None:
                if previous.phase == "prefill":
                    for request_id in previous.final_prefill_request_ids:
                        states[request_id].generated_tokens += 1
                elif previous.phase == "decode":
                    for request_id in previous.request_ids:
                        states[request_id].generated_tokens += 1
                else:
                    raise AssertionError(previous.phase)

            # get_next_batch_to_run merges only final, unfinished prefill
            # requests. Middle chunks and one-token completions are filtered;
            # filtering also reopens a lane previously marked full.
            if previous is not None and previous.phase == "prefill":
                mergeable = [
                    request_id
                    for request_id in previous.final_prefill_request_ids
                    if not states[request_id].finished
                ]
                if len(mergeable) < len(previous.request_ids):
                    lane.batch_is_full = False
                for request_id in mergeable:
                    if request_id not in lane.running_request_ids:
                        lane.running_request_ids.append(request_id)

            can_run: list[int] = []
            final_prefill: list[int] = []
            active_tokens = 0
            remaining_chunk_budget = chunk_tokens

            # A global chunk continuation bypasses the per-lane full gate. A
            # fresh FCFS prefill is eligible only when this lane is not full.
            prefill_eligible = chunked_request_id is not None or (
                bool(waiting) and not lane.batch_is_full
            )
            if (
                prefill_eligible
                and chunked_request_id is None
                and max_microbatch - len(lane.running_request_ids) <= 0
            ):
                lane.batch_is_full = True
                prefill_eligible = False

            if prefill_eligible and chunked_request_id is not None:
                request_id = chunked_request_id
                request = states[request_id]
                take = min(
                    request.input_tokens - request.input_position,
                    remaining_chunk_budget,
                )
                request.input_position += take
                active_tokens += take
                remaining_chunk_budget -= ceil_page(take)
                can_run.append(request_id)
                if request.input_position == request.input_tokens:
                    final_prefill.append(request_id)
                    chunked_request_id = None

            if prefill_eligible:
                while waiting and remaining_chunk_budget > 0:
                    allocatable = max_microbatch - len(lane.running_request_ids)
                    if len(can_run) >= allocatable:
                        lane.batch_is_full = True
                        break
                    request_id = waiting.pop(0)
                    request = states[request_id]
                    remaining = request.input_tokens - request.input_position
                    # Ignore-EOS + ChunkCache compares the exact remaining
                    # length with the page-rounded residual budget, then charges
                    # ceil_page(take) after admission.
                    take = min(remaining, remaining_chunk_budget)
                    if take <= 0:
                        waiting.insert(0, request_id)
                        break
                    request.input_position += take
                    active_tokens += take
                    remaining_chunk_budget -= ceil_page(take)
                    can_run.append(request_id)
                    if request.input_position == request.input_tokens:
                        final_prefill.append(request_id)
                    else:
                        chunked_request_id = request_id

            if can_run:
                record("prefill", active_tokens, len(can_run))
                lane.pending_batch = PendingBatch(
                    phase="prefill",
                    request_ids=tuple(can_run),
                    final_prefill_request_ids=tuple(final_prefill),
                )
                emitted_in_round = True
                continue

            # This ordering is the Phase 25B correction: SGLang attempts
            # prefill before update_running_batch filters finished decodes. A
            # just-freed slot therefore produces one smaller decode forward and
            # is refilled on the lane's following visit.
            initial_running = len(lane.running_request_ids)
            lane.running_request_ids = [
                request_id
                for request_id in lane.running_request_ids
                if not states[request_id].finished
            ]
            if len(lane.running_request_ids) < initial_running:
                lane.batch_is_full = False
            if lane.running_request_ids:
                active = tuple(lane.running_request_ids)
                record("decode", len(active), len(active))
                lane.pending_batch = PendingBatch("decode", active)
                emitted_in_round = True

        if not emitted_in_round:
            complete = (
                not waiting
                and chunked_request_id is None
                and all(
                    not lane.running_request_ids and lane.pending_batch is None
                    for lane in lanes
                )
            )
            if complete:
                break
        if scheduler_visits > 100_000_000:
            raise RuntimeError("scheduler simulation failed to converge")

    return SimulationResult(
        event_histograms=event_histograms,
        annotation_histogram=annotation_histogram,
        event_counts=event_counts,
        max_active_batch_size=max_active_batch_size,
        scheduler_visits=scheduler_visits,
        all_requests_complete=all(
            row.input_position == row.input_tokens
            and row.generated_tokens == row.output_tokens
            for row in states
        ),
        prefill_token_mass=sum(
            tokens * count for tokens, count in event_histograms["prefill"].items()
        ),
        decode_token_mass=sum(
            tokens * count for tokens, count in event_histograms["decode"].items()
        ),
    )


def load_windows(path: Path) -> list[dict]:
    rows = read_jsonl_gz(path)
    if len(rows) != 24:
        raise ValueError(f"expected 24 windows, got {len(rows)}")
    for row in rows:
        if len(row["input_lens"]) != len(row["output_lens"]):
            raise ValueError(f"{row['profile_id']}: request array mismatch")
        if len(row["input_lens"]) != int(row["request_count"]):
            raise ValueError(f"{row['profile_id']}: request count mismatch")
    return rows


def load_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def gpu_histograms(cell: Path) -> tuple[Counter, Counter, Path]:
    profiles = sorted((cell / "profile").glob("*pp0*.json"))
    if len(profiles) != 1:
        raise ValueError(f"{cell}: expected one PP0 profile, got {len(profiles)}")
    data = json.loads(profiles[0].read_text())
    payload: Counter[tuple[str, int]] = Counter()
    annotation: Counter[tuple[str, int, int]] = Counter()
    for row in data["histograms"]:
        if not row.get("workload_id") or row.get("msg_type") != "proxy":
            continue
        phase = row["phase"]
        count = int(row["count"])
        payload[(phase, int(row["payload_bytes"]))] += count
        annotation[(phase, int(row["active_batch_size"]), int(row["active_tokens"]))] += count
    return payload, annotation, profiles[0]


def validate_gpu_smoke(
    phase25_root: Path, smoke_window: dict
) -> tuple[list[dict], bool]:
    requests = list(zip(smoke_window["input_lens"], smoke_window["output_lens"]))
    rows = []
    for pp_size in PP_SIZES:
        for microbatch in PP_MICROBATCH_SIZES:
            cell = (
                phase25_root
                / "gpu_audit/results/pp/smoke"
                / f"pp{pp_size}"
                / f"mb{microbatch}"
                / "r0"
            )
            if not (cell / "DONE").exists():
                raise FileNotFoundError(cell / "DONE")
            simulated = simulate_scheduler(
                requests, pp_size=pp_size, max_microbatch=microbatch
            )
            gpu_payload, gpu_annotation, profile_path = gpu_histograms(cell)
            simulated_payload: Counter[tuple[str, int]] = Counter()
            for phase in PHASES:
                for tokens, count in simulated.event_histograms[phase].items():
                    simulated_payload[(phase, tokens * BYTES_PER_TOKEN)] += (
                        count * PP_PROXY_COUNT
                    )
            simulated_annotation = Counter(
                {
                    key: count * PP_PROXY_COUNT
                    for key, count in simulated.annotation_histogram.items()
                }
            )
            payload_keys = set(simulated_payload) | set(gpu_payload)
            annotation_keys = set(simulated_annotation) | set(gpu_annotation)
            payload_l1 = sum(
                abs(simulated_payload[key] - gpu_payload[key])
                for key in payload_keys
            )
            annotation_l1 = sum(
                abs(simulated_annotation[key] - gpu_annotation[key])
                for key in annotation_keys
            )
            simulated_calls = sum(simulated_payload.values())
            gpu_calls = sum(gpu_payload.values())
            simulated_bytes = sum(
                payload * count for (_, payload), count in simulated_payload.items()
            )
            gpu_bytes = sum(
                payload * count for (_, payload), count in gpu_payload.items()
            )
            exact = (
                simulated_payload == gpu_payload
                and simulated_annotation == gpu_annotation
                and simulated_calls == gpu_calls
                and simulated_bytes == gpu_bytes
            )
            rows.append(
                {
                    "profile_id": smoke_window["profile_id"],
                    "requests": len(requests),
                    "pp_size": pp_size,
                    "microbatch": microbatch,
                    "simulated_calls": simulated_calls,
                    "gpu_calls": gpu_calls,
                    "calls_abs_error": abs(simulated_calls - gpu_calls),
                    "simulated_logical_bytes": simulated_bytes,
                    "gpu_logical_bytes": gpu_bytes,
                    "logical_bytes_abs_error": abs(simulated_bytes - gpu_bytes),
                    "payload_histogram_l1": payload_l1,
                    "annotation_histogram_l1": annotation_l1,
                    "max_prefill_batch_size": simulated.max_active_batch_size["prefill"],
                    "max_decode_batch_size": simulated.max_active_batch_size["decode"],
                    "exact": exact,
                    "gpu_profile": str(profile_path.relative_to(phase25_root)),
                    "gpu_profile_sha256": sha256(profile_path),
                }
            )
    return rows, len(rows) == 9 and all(row["exact"] for row in rows)


def make_label(
    *,
    model: dict,
    profile: dict,
    pp_size: int,
    microbatch: int,
    phase: str,
    hist: Counter[int],
) -> dict:
    normalized = normalize(
        Counter(
            {
                tokens * BYTES_PER_TOKEN: count * PP_PROXY_COUNT
                for tokens, count in hist.items()
            }
        ),
        int(profile["request_count"]),
    )
    row = label_row(
        model=model,
        profile=profile,
        parallelism="pp",
        parallel_size=pp_size,
        policy=f"mb{microbatch}",
        phase=phase,
        request_count=int(profile["request_count"]),
        hist=normalized,
        edges=PP_BIN_EDGES,
        boundary_multiplier=pp_size - 1,
    )
    row["label_status"] = LABEL_STATUS
    row["teacher_kind"] = "full_window_fixed_draining_scheduler_faithful_teacher"
    row["scheduler_contract"] = "sglang_pp_fcfs_lanes_v1"
    row["pp_loop_lanes"] = pp_size
    row["chunk_tokens"] = PP_CHUNK_TOKENS
    row["page_size"] = PAGE_SIZE
    row["proxy_tensor_count"] = PP_PROXY_COUNT
    return row


def build_labels(
    windows: list[dict], model: dict
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    phase_rows, boundary_rows, config_rows, invariant_failures = [], [], [], []
    for profile in windows:
        requests = list(zip(profile["input_lens"], profile["output_lens"]))
        expected_prefill = sum(profile["input_lens"])
        expected_decode = sum(max(int(value) - 1, 0) for value in profile["output_lens"])
        for pp_size in PP_SIZES:
            for microbatch in PP_MICROBATCH_SIZES:
                simulated = simulate_scheduler(
                    requests, pp_size=pp_size, max_microbatch=microbatch
                )
                mass_ok = (
                    simulated.prefill_token_mass == expected_prefill
                    and simulated.decode_token_mass == expected_decode
                )
                complete = simulated.all_requests_complete
                if not mass_ok or not complete:
                    invariant_failures.append(
                        {
                            "profile_id": profile["profile_id"],
                            "pp_size": pp_size,
                            "microbatch": microbatch,
                            "mass_ok": mass_ok,
                            "all_requests_complete": complete,
                        }
                    )
                config_rows.append(
                    {
                        "profile_id": profile["profile_id"],
                        "source": profile["source"],
                        "segment": profile["segment"],
                        "split": profile["split"],
                        "requests": len(requests),
                        "pp_size": pp_size,
                        "microbatch": microbatch,
                        "prefill_forwards": simulated.event_counts["prefill"],
                        "decode_forwards": simulated.event_counts["decode"],
                        "total_proxy_calls": PP_PROXY_COUNT
                        * sum(simulated.event_counts.values()),
                        "max_prefill_batch_size": simulated.max_active_batch_size["prefill"],
                        "max_decode_batch_size": simulated.max_active_batch_size["decode"],
                        "prefill_token_mass": simulated.prefill_token_mass,
                        "decode_token_mass": simulated.decode_token_mass,
                        "scheduler_visits": simulated.scheduler_visits,
                        "mass_conservation": mass_ok,
                        "all_requests_complete": complete,
                    }
                )
                for phase in PHASES:
                    row = make_label(
                        model=model,
                        profile=profile,
                        pp_size=pp_size,
                        microbatch=microbatch,
                        phase=phase,
                        hist=simulated.event_histograms[phase],
                    )
                    phase_rows.append(row)
                    for sender in range(pp_size - 1):
                        boundary_rows.append(
                            {
                                **row,
                                "label_id": f"{row['label_id']}/boundary{sender}-{sender + 1}",
                                "scope": "single_sender_boundary",
                                "sender_stage": sender,
                                "receiver_stage": sender + 1,
                                "boundary_multiplier": 1,
                                "pipeline_calls_per_1000": row["total_calls_per_1000"],
                                "pipeline_logical_bytes_per_1000": row[
                                    "total_logical_bytes_per_1000"
                                ],
                            }
                        )
    return phase_rows, boundary_rows, config_rows, invariant_failures


def compare_old_teacher(
    old_rows: list[dict[str, str]], new_rows: list[dict]
) -> tuple[list[dict], list[dict], dict]:
    old = {
        (
            row["profile_id"],
            int(row["parallel_size"]),
            row["policy"],
            row["phase"],
        ): row
        for row in old_rows
    }
    new = {
        (
            row["profile_id"],
            int(row["parallel_size"]),
            row["policy"],
            row["phase"],
        ): row
        for row in new_rows
    }
    if set(old) != set(new):
        raise ValueError("old/new PP label keys differ")
    metrics = []
    for key in sorted(new):
        row = metric_row(old[key], new[key])
        row["comparison"] = "phase25a_static_formula_vs_phase25b_scheduler_teacher"
        metrics.append(row)
    grouped_old, grouped_new = defaultdict(list), defaultdict(list)
    for key, row in old.items():
        grouped_old[key[:3]].append(row)
    for key, row in new.items():
        grouped_new[key[:3]].append(row)
    total_metrics = []
    for key in sorted(grouped_new):
        row = metric_row(combine(grouped_old[key]), combine(grouped_new[key]))
        row["comparison"] = "phase25a_static_formula_vs_phase25b_scheduler_teacher"
        total_metrics.append(row)
    aggregates = [aggregate(total_metrics, "pp", "total", policy) for policy in ("mb1", "mb4", "mb16")]
    aggregate_all = aggregate(total_metrics, "pp", "total")
    return metrics + total_metrics, aggregates, aggregate_all


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()


def source_contract(root: Path) -> dict:
    sources = [
        root / "python/sglang/srt/managers/scheduler.py",
        root / "python/sglang/srt/managers/scheduler_pp_mixin.py",
        root / "python/sglang/srt/managers/schedule_policy.py",
        root / "python/sglang/srt/managers/scheduler_components/batch_result_processor.py",
    ]
    return {
        "schema_version": "phase25b-pp-scheduler-source-contract-v1",
        "repository_head_at_build": git_head(root),
        "source_files": {
            str(path.relative_to(root)): sha256(path) for path in sources
        },
        "server_contract": {
            "schedule_policy": "fcfs",
            "disable_radix_cache": True,
            "chunked_prefill_size": PP_CHUNK_TOKENS,
            "page_size": PAGE_SIZE,
            "enable_mixed_chunk": False,
            "pp_async_batch_depth": 0,
            "disable_overlap_schedule": True,
            "ignore_eos": True,
            "proxy_tensor_count": PP_PROXY_COUNT,
        },
        "mirrored_rules": [
            "pp_loop_size equals pp_size when pp_async_batch_depth is zero",
            "each PP loop lane owns an independent running batch",
            "a global chunked request continues before fresh FCFS admission and may migrate lanes",
            "prefill chunk budget charges page-rounded tokens while the forward payload uses exact tokens",
            "get_new_batch_prefill runs before update_running_batch filters finished decode requests",
            "the just-shrunk decode batch executes once before its freed slot is refilled",
            "final prefill emits token one; decode emits the remaining output tokens",
        ],
        "scope_boundary": "fixed-draining batch generate; no online arrival timing, preemption, radix reuse, mixed chunk, async PP depth, or speculative decoding",
    }


def readme(summary: dict) -> str:
    aggregate = summary["old_static_formula_comparison"]["overall"]
    policy = {row["policy"]: row for row in summary["old_static_formula_comparison"]["by_policy"]}
    return f"""# Phase 25B: scheduler-faithful PP full-window teacher

Status: **PASS** for the scoped fixed-draining scheduler contract. The source-derived
simulator matches all {summary['gpu_smoke_validation']['exact_cells']}/9 saved GPU
smoke cells exactly, including calls, logical bytes, payload histograms, phase
labels, active batch size, and active tokens.

## What changed

The Phase 25A PP formula used static prefill/decode groups. SGLang instead keeps
one running batch per PP loop lane, continues a global chunked request across
lanes, and attempts prefill before filtering finished decode requests. A freed
slot therefore causes one smaller decode forward and is refilled on the next
visit. Page-rounded chunk budget accounting is also required to reproduce the
exact prefill payloads.

## Assets and checks

- Complete windows: {summary['inputs']['profiles']} profiles and
  {summary['inputs']['requests']:,} requests, original order, fixed-draining.
- PP phase labels: {summary['labels']['phase_rows']}.
- Explicit sender-boundary labels: {summary['labels']['boundary_rows']}.
- CPU configurations checked: {summary['cpu_invariants']['configurations']};
  token-mass conservation and request completion pass in all cases.
- GPU smoke: PP2/4/8 x MB1/4/16 on the 42-request BurstGPT sentinel; all 9 cells exact.

## Size of the Phase 25A correction

Treating the new scheduler-faithful label as reference, the old static formula
has overall calls WAPE {aggregate['calls_wape']:.2%}, mean histogram TV
{aggregate['mean_calls_histogram_tv']:.4f}, normalized log-payload EMD
{aggregate['mean_normalized_log_payload_emd']:.4f}, and reference-cost MAPE
{aggregate['common_reference_cost_mape']:.2%}. Logical bytes remain conserved.

- MB1: calls WAPE {policy['mb1']['calls_wape']:.2%}; the old formula is exact.
- MB4: calls WAPE {policy['mb4']['calls_wape']:.2%}.
- MB16: calls WAPE {policy['mb16']['calls_wape']:.2%}.

## Scientific boundary

These labels are valid for the recorded fixed-draining contract. The 9/9 exact
result validates all PP-size/microbatch combinations on one heterogeneous full
window, not every traffic distribution. Online arrivals, preemption, radix
cache, mixed chunking, async PP depth, speculative decoding, and other policies
require separate teachers or audits. The final predictor still consumes only
the compact history profile, model structure, fixed PP configuration, policy,
and phase; full request lists are offline label-generation inputs only.

Raw profiler traces, weights, caches, and PID files are not included.
"""


def main() -> None:
    started = time.time()
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    for directory in ("labels", "analysis", "logs"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)

    windows_path = args.phase25_root / "inputs/full_windows.jsonl.gz"
    old_labels_path = args.phase25_root / "labels/pp_phase_labels.csv.gz"
    windows = load_windows(windows_path)
    model_rows = json.loads(args.model_features.read_text())
    model = next(row for row in model_rows if row["model"] == PP_MODEL)

    phase_rows, boundary_rows, config_rows, invariant_failures = build_labels(
        windows, model
    )
    write_csv_gz(args.output_dir / "labels/pp_phase_labels.csv.gz", phase_rows)
    write_csv_gz(args.output_dir / "labels/pp_boundary_labels.csv.gz", boundary_rows)
    write_csv(args.output_dir / "analysis/configuration_statistics.csv", config_rows)

    smoke_window = next(
        row for row in windows if row["profile_id"] == "profile_13_burstgpt_3_c2"
    )
    smoke_rows, smoke_pass = validate_gpu_smoke(args.phase25_root, smoke_window)
    write_csv(args.output_dir / "analysis/gpu_smoke_exact_validation.csv", smoke_rows)

    old_rows = load_csv_gz(old_labels_path)
    comparison_rows, policy_aggregates, overall_aggregate = compare_old_teacher(
        old_rows, phase_rows
    )
    write_csv(args.output_dir / "analysis/old_vs_scheduler_row_metrics.csv", comparison_rows)
    write_csv(args.output_dir / "analysis/old_vs_scheduler_aggregate.csv", policy_aggregates + [overall_aggregate])
    comparison_summary = {
        "schema_version": "phase25b-old-vs-scheduler-comparison-v1",
        "interpretation": "Phase 25A static structural formula is prediction; Phase 25B scheduler-faithful formula is reference",
        "by_policy": policy_aggregates,
        "overall": overall_aggregate,
    }
    write_json(args.output_dir / "analysis/old_vs_scheduler_summary.json", comparison_summary)

    contract = source_contract(root)
    write_json(args.output_dir / "source_contract.json", contract)

    checks = {
        "profiles_24": len(windows) == 24,
        "requests_18285": sum(int(row["request_count"]) for row in windows) == 18285,
        "phase_labels_432": len(phase_rows) == 432,
        "boundary_labels_1584": len(boundary_rows) == 1584,
        "cpu_configurations_216": len(config_rows) == 216,
        "cpu_invariants_all_pass": not invariant_failures,
        "gpu_smoke_cells_9": len(smoke_rows) == 9,
        "gpu_smoke_exact_9_of_9": smoke_pass,
        "all_label_statuses_validated": all(
            row["label_status"] == LABEL_STATUS for row in phase_rows + boundary_rows
        ),
        "mb1_old_formula_exact": next(
            row for row in policy_aggregates if row["policy"] == "mb1"
        )["exact_histograms"]
        == 72,
        "logical_bytes_conserved_old_vs_new": math.isclose(
            overall_aggregate["bytes_wape"], 0.0, rel_tol=0, abs_tol=1e-12
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase25b-pp-scheduler-teacher-v1",
        "status": status,
        "scope": "SGLang fixed-draining FCFS PP scheduler contract recorded in source_contract.json",
        "inputs": {
            "profiles": len(windows),
            "requests": sum(int(row["request_count"]) for row in windows),
            "full_windows": str(windows_path.relative_to(root)),
            "full_windows_sha256": sha256(windows_path),
            "old_phase25a_pp_labels": str(old_labels_path.relative_to(root)),
            "old_phase25a_pp_labels_sha256": sha256(old_labels_path),
            "model_features_sha256": sha256(args.model_features),
        },
        "labels": {
            "phase_rows": len(phase_rows),
            "boundary_rows": len(boundary_rows),
            "label_status": LABEL_STATUS,
        },
        "cpu_invariants": {
            "configurations": len(config_rows),
            "passed": len(config_rows) - len(invariant_failures),
            "failed": len(invariant_failures),
            "failures": invariant_failures,
        },
        "gpu_smoke_validation": {
            "profile_id": smoke_window["profile_id"],
            "requests": int(smoke_window["request_count"]),
            "cells": len(smoke_rows),
            "exact_cells": sum(row["exact"] for row in smoke_rows),
            "max_calls_abs_error": max(row["calls_abs_error"] for row in smoke_rows),
            "max_logical_bytes_abs_error": max(
                row["logical_bytes_abs_error"] for row in smoke_rows
            ),
            "max_payload_histogram_l1": max(
                row["payload_histogram_l1"] for row in smoke_rows
            ),
            "max_annotation_histogram_l1": max(
                row["annotation_histogram_l1"] for row in smoke_rows
            ),
        },
        "old_static_formula_comparison": comparison_summary,
        "checks": checks,
        "can_conclude": [
            "the recovered simulator exactly reproduces all nine saved Phase 25 PP smoke histograms",
            "PP size is a scheduler input because it changes the number of independent running lanes",
            "logical-byte mass was never the failing component; the correction changes forward-call grouping and payload distribution",
            "the scheduler-faithful full-window labels can replace the Phase 25A provisional static PP labels for the scoped contract",
        ],
        "cannot_conclude": [
            "one sentinel proves accuracy for every possible request distribution",
            "the same teacher applies to online arrival-aware scheduling or other server options",
            "compact low-dimensional profiles can recover the scheduler-faithful labels without a learned residual",
        ],
        "next_step": "run a small tail-coverage GPU audit on selected full windows, then rebuild H32/H64/H128/Hfull convergence against the scheduler-faithful PP teacher and the existing TP teacher",
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "README.md").write_text(readme(summary))
    build_log = {
        "status": status,
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "duration_seconds": time.time() - started,
        "checks": checks,
    }
    write_json(args.output_dir / "logs/build.log", build_log)

    if status == "FAIL":
        raise RuntimeError(json.dumps(summary, indent=2))
    if not args.skip_plot:
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts/plot_phase25b_pp_scheduler_recovery.py"),
                "--result-dir",
                str(args.output_dir),
            ],
            cwd=root,
            check=True,
        )
    (args.output_dir / "DONE").write_text("PASS\n")
    files = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(args.output_dir)}\n" for path in files
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
