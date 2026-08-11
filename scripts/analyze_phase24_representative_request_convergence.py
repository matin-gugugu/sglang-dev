#!/usr/bin/env python3
"""Compare H32/H64/H128 with full-window fixed-draining TP/PP H0 labels.

The requested sample-size experiment is deliberately CPU-only.  Phase 16 and
Phase 23 already validated the exact-workload TP and PP structural formulas
against GPU histogram-only labels.  This script therefore isolates the error
introduced by representative-request sampling and by compact-profile
reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_phase21b_pp_h0 import (
    decode_events as pp_decode_events,
    histogram as pp_histogram,
    prefill_events as pp_prefill_events,
    pseudo_requests,
)
from build_profiledemand_replay_plans import STRATEGIES, stratified_indices
from build_profiledemand_service_profiles import (
    HISTORY_SECONDS,
    INPUT_CAP,
    OUTPUT_CAP,
    representative_indices,
)
from prepare_phase15_trace_windows import (
    BURST_FILES,
    MOONCAKE_FILES,
    load_segment,
)


SAMPLE_LABELS = ("h32", "h64", "h128", "hfull", "compact32")
REQUESTED_ESTIMATORS = ("h32", "h64", "h128")
TPS = (2, 4, 8)
PP_SIZES = (2, 4, 8)
PP_MICROBATCH_SIZES = (1, 4, 16)
PHASES = ("prefill", "decode")
TP_CALLS_PER_FORWARD = 73
PAYLOAD_BYTES_PER_TOKEN = 4096 * 2
PP_PROXY_TENSOR_COUNT = 2
PP_CHUNK_TOKENS = 4096
MIN_PAYLOAD = 4 * 1024
MAX_PAYLOAD = 8 * 1024 * 1024 * 1024
COMMON_REFERENCE_LAUNCH_US = 5.0
COMMON_REFERENCE_BANDWIDTH_GBPS = 100.0
CONVERGENCE_THRESHOLDS = {
    "calls_mape": 0.05,
    "calls_wape": 0.05,
    "bytes_mape": 0.05,
    "bytes_wape": 0.05,
    "histogram_tv": 0.05,
    "normalized_log_payload_emd": 0.02,
    "common_reference_cost_mape": 0.05,
    "p95_calls_ape": 0.15,
    "p95_bytes_ape": 0.15,
    "p95_common_reference_cost_ape": 0.15,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", type=Path, default=root / "data/phase15_traces/raw"
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=root
        / "experiment-results/phase16_service_profiles/service_profiles.csv",
    )
    parser.add_argument(
        "--representatives",
        type=Path,
        default=root
        / "experiment-results/phase16_service_profiles/representative_requests.jsonl",
    )
    parser.add_argument(
        "--phase16-plan",
        type=Path,
        default=root
        / "experiment-results/phase16_profiledemand_plans/full_replay_plan.jsonl",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root
        / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--l1-curve-root",
        type=Path,
        default=root / "experiment-results/phase14f_post_rendezvous/curve",
    )
    parser.add_argument(
        "--l1-curve-extension",
        type=Path,
        default=root
        / "experiment-results/phase15_l1_curve_extension/curve_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "experiment-results/phase24_representative_request_convergence",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(text.encode("utf-8"))


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    deterministic_gzip(
        path,
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows),
    )


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def load_profiles(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 24:
        raise ValueError(f"expected 24 service profiles, got {len(rows)}")
    return rows


def load_representatives(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[row["profile_id"]].append(row)
    for profile_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["request_index"]))
        if len(rows) != 128:
            raise ValueError(f"{profile_id}: expected 128 representatives, got {len(rows)}")
    return dict(grouped)


def load_phase16_h32_plan(path: Path) -> dict[tuple[str, str], list[tuple[int, int]]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        if int(row["repeat"]) == 0:
            grouped[(row["profile_id"], row["strategy"])].append(row)
    result = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["batch_index"]))
        requests = []
        for row in rows:
            requests.extend(
                zip(
                    map(int, row["input_lens_per_request"]),
                    map(int, row["output_lens_per_request"]),
                )
            )
        result[key] = list(requests)
    return result


def segment_file_map(raw_dir: Path) -> dict[str, Path]:
    mapping = {}
    for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items():
        mapping[segment] = raw_dir / name
    return mapping


def verify_raw_sources(raw_dir: Path) -> tuple[dict[str, bool], dict[str, str]]:
    manifest_path = raw_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checks = {}
    hashes = {}
    for row in manifest["sources"]:
        path = raw_dir / row["name"]
        actual = sha256(path)
        hashes[row["name"]] = actual
        checks[row["name"]] = (
            path.is_file()
            and path.stat().st_size == int(row["actual_size"])
            and actual == row["sha256"]
        )
    return checks, hashes


def materialize_request_sets(
    profiles: list[dict[str, str]],
    representatives: dict[str, list[dict]],
    phase16_h32: dict[tuple[str, str], list[tuple[int, int]]],
    raw_dir: Path,
) -> tuple[dict[str, dict[str, list[tuple[int, int]]]], list[dict], dict]:
    arrays = {
        segment: load_segment(path)
        for segment, path in segment_file_map(raw_dir).items()
    }
    request_sets = {}
    rows_for_archive = []
    count_checks = []
    h128_checks = []
    h32_plan_checks = []
    total_full_requests = 0
    for profile in profiles:
        profile_id = profile["profile_id"]
        timestamps, raw_inputs, raw_outputs = arrays[profile["segment"]]
        cutoff = int(profile["cutoff_ms"])
        left = int(
            np.searchsorted(
                timestamps, cutoff - HISTORY_SECONDS * 1000, side="left"
            )
        )
        right = int(np.searchsorted(timestamps, cutoff, side="left"))
        window_timestamps = timestamps[left:right]
        window_inputs_raw = raw_inputs[left:right]
        window_outputs_raw = raw_outputs[left:right]
        window_inputs = np.clip(window_inputs_raw, 1, INPUT_CAP).astype(int)
        window_outputs = np.clip(window_outputs_raw, 1, OUTPUT_CAP).astype(int)
        full = list(zip(window_inputs.tolist(), window_outputs.tolist()))
        total_full_requests += len(full)
        count_checks.append(len(full) == int(profile["request_count"]))

        regenerated_indices = representative_indices(
            window_inputs_raw, window_outputs_raw, 128
        )
        regenerated_h128 = [
            (int(window_inputs[index]), int(window_outputs[index]))
            for index in regenerated_indices
        ]
        stored_h128 = [
            (int(row["input_len_capped"]), int(row["output_len_capped"]))
            for row in representatives[profile_id]
        ]
        h128_checks.append(regenerated_h128 == stored_h128)

        exact_sets = {"h128": stored_h128, "hfull": full}
        for count in (32, 64):
            selected = stratified_indices(representatives[profile_id], count)
            exact_sets[f"h{count}"] = [stored_h128[index] for index in selected]
        exact_sets["compact32"] = [tuple(map(int, row)) for row in pseudo_requests(profile)]
        request_sets[profile_id] = exact_sets

        for strategy in STRATEGIES:
            h32_plan_checks.append(
                phase16_h32[(profile_id, strategy)] == exact_sets["h32"]
            )
        for sample_label in SAMPLE_LABELS:
            sample = exact_sets[sample_label]
            rows_for_archive.append(
                {
                    "profile_id": profile_id,
                    "source": profile["source"],
                    "segment": profile["segment"],
                    "split": profile["split"],
                    "window_id": profile["window_id"],
                    "cutoff_ms": cutoff,
                    "sample_label": sample_label,
                    "sampling_contract": (
                        "phase16_128_pool_then_4x4_stratified"
                        if sample_label in {"h32", "h64"}
                        else "phase16_4x4_representative_pool"
                        if sample_label == "h128"
                        else "full_window_original_order_capped"
                        if sample_label == "hfull"
                        else "compact_profile_4x4_canonical_reconstruction"
                    ),
                    "request_count": len(sample),
                    "full_request_count": len(full),
                    "full_ge_128": len(full) >= 128,
                    "h128_uses_replacement": len(full) < 128,
                    "input_lens": [row[0] for row in sample],
                    "output_lens": [row[1] for row in sample],
                }
            )
    audit = {
        "full_request_counts_match_profiles": all(count_checks),
        "stored_h128_matches_regenerated_phase16_sampling": all(h128_checks),
        "h32_matches_all_phase16_strategy_replay_plans": all(h32_plan_checks),
        "h32_plan_comparisons": len(h32_plan_checks),
        "total_full_requests": total_full_requests,
        "profiles_with_at_least_128_requests": sum(
            int(profile["request_count"]) >= 128 for profile in profiles
        ),
    }
    return request_sets, rows_for_archive, audit


def tp_batches(
    requests: list[tuple[int, int]], max_batch_size: int, max_prefill_tokens: int
) -> list[list[tuple[int, int]]]:
    batches = []
    current = []
    current_tokens = 0
    for request in requests:
        would_exceed = current and (
            len(current) >= max_batch_size
            or current_tokens + request[0] > max_prefill_tokens
        )
        if would_exceed:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(request)
        current_tokens += request[0]
    if current:
        batches.append(current)
    return batches


def tp_histograms(
    requests: list[tuple[int, int]], strategy: dict[str, int]
) -> dict[str, Counter[int]]:
    histograms = {"prefill": Counter(), "decode": Counter()}
    batches = tp_batches(
        requests,
        int(strategy["max_batch_size"]),
        int(strategy["max_prefill_tokens"]),
    )
    for batch in batches:
        prefill_tokens = sum(row[0] for row in batch)
        histograms["prefill"][prefill_tokens * PAYLOAD_BYTES_PER_TOKEN] += (
            TP_CALLS_PER_FORWARD
        )
        max_output = max(row[1] for row in batch)
        for step in range(1, max_output):
            active = sum(row[1] > step for row in batch)
            if active:
                histograms["decode"][active * PAYLOAD_BYTES_PER_TOKEN] += (
                    TP_CALLS_PER_FORWARD
                )
    return histograms


def pp_histograms(
    requests: list[tuple[int, int]], max_microbatch: int
) -> dict[str, Counter[int]]:
    prefill = pp_prefill_events(requests, max_microbatch, PP_CHUNK_TOKENS)
    decode = pp_decode_events(requests, max_microbatch, PP_CHUNK_TOKENS)
    return {
        "prefill": pp_histogram(
            prefill, PAYLOAD_BYTES_PER_TOKEN, PP_PROXY_TENSOR_COUNT
        ),
        "decode": pp_histogram(
            decode, PAYLOAD_BYTES_PER_TOKEN, PP_PROXY_TENSOR_COUNT
        ),
    }


def normalize_histogram(histogram: Counter[int], requests: int) -> dict[int, float]:
    scale = 1000.0 / requests
    return {int(payload): float(calls) * scale for payload, calls in sorted(histogram.items())}


def load_measured_l1_curves(root: Path, extension: Path) -> dict[int, list[tuple[int, float]]]:
    samples: dict[tuple[int, int], list[float]] = defaultdict(list)
    for path in sorted(root.glob("tp*/all_reduce/r*/curve.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            samples[(int(row["group_size"]), int(row["payload_bytes"]))].extend(
                map(float, row["post_rendezvous_samples_us"])
            )
    points: dict[int, dict[int, float]] = defaultdict(dict)
    for (tp, payload), values in samples.items():
        points[tp][payload] = float(np.median(values))
    with extension.open(newline="") as source:
        for row in csv.DictReader(source):
            points[int(row["tp"])][int(row["payload_bytes"])] = float(
                row["median_post_rendezvous_us"]
            )
    curves = {tp: sorted(points[tp].items()) for tp in TPS}
    for tp, curve in curves.items():
        if curve[0][0] > MIN_PAYLOAD or curve[-1][0] < 512 * 1024 * 1024:
            raise ValueError(f"incomplete measured L1 curve for TP{tp}")
    return curves


def measured_interpolate(points: list[tuple[int, float]], payload: int) -> float:
    clipped = float(np.clip(payload, points[0][0], points[-1][0]))
    xs = np.log2(np.asarray([row[0] for row in points], dtype=np.float64))
    ys = np.asarray([row[1] for row in points], dtype=np.float64)
    return max(float(np.interp(math.log2(clipped), xs, ys)), 1e-9)


def common_reference_cost_us(payload: int) -> float:
    data_us = (
        float(payload) / (COMMON_REFERENCE_BANDWIDTH_GBPS * 1e9) * 1e6
    )
    return COMMON_REFERENCE_LAUNCH_US + data_us


def histogram_cost(
    histogram: dict[int, float], cost_function
) -> float:
    return float(
        sum(float(calls) * float(cost_function(payload)) for payload, calls in histogram.items())
    )


def label_rows(
    profiles: list[dict[str, str]],
    request_sets: dict[str, dict[str, list[tuple[int, int]]]],
    measured_curves: dict[int, list[tuple[int, float]]],
) -> tuple[list[dict], dict[tuple, dict[int, float]]]:
    rows = []
    lookup = {}
    for profile in profiles:
        profile_id = profile["profile_id"]
        full_count = len(request_sets[profile_id]["hfull"])
        for sample_label in SAMPLE_LABELS:
            requests = request_sets[profile_id][sample_label]
            estimator_kind = (
                "compact_profile" if sample_label == "compact32" else "exact_representative"
            )
            for tp in TPS:
                for policy, strategy in STRATEGIES.items():
                    histograms = tp_histograms(requests, strategy)
                    for phase in PHASES:
                        normalized = normalize_histogram(histograms[phase], len(requests))
                        key = (profile_id, "tp", tp, policy, sample_label, phase)
                        lookup[key] = normalized
                        common_cost = histogram_cost(normalized, common_reference_cost_us)
                        measured_cost = histogram_cost(
                            normalized,
                            lambda payload, curve=measured_curves[tp]: measured_interpolate(
                                curve, payload
                            ),
                        )
                        rows.append(
                            make_label_row(
                                profile,
                                "tp",
                                tp,
                                policy,
                                estimator_kind,
                                sample_label,
                                len(requests),
                                full_count,
                                phase,
                                normalized,
                                common_cost,
                                measured_cost,
                                boundary_multiplier=1,
                            )
                        )
            for pp_size in PP_SIZES:
                for max_microbatch in PP_MICROBATCH_SIZES:
                    policy = f"mb{max_microbatch}"
                    histograms = pp_histograms(requests, max_microbatch)
                    for phase in PHASES:
                        normalized = normalize_histogram(histograms[phase], len(requests))
                        key = (
                            profile_id,
                            "pp",
                            pp_size,
                            policy,
                            sample_label,
                            phase,
                        )
                        lookup[key] = normalized
                        common_cost = histogram_cost(normalized, common_reference_cost_us)
                        rows.append(
                            make_label_row(
                                profile,
                                "pp",
                                pp_size,
                                policy,
                                estimator_kind,
                                sample_label,
                                len(requests),
                                full_count,
                                phase,
                                normalized,
                                common_cost,
                                None,
                                boundary_multiplier=pp_size - 1,
                            )
                        )
    return rows, lookup


def make_label_row(
    profile: dict[str, str],
    parallelism: str,
    parallel_size: int,
    policy: str,
    estimator_kind: str,
    sample_label: str,
    sample_requests: int,
    full_requests: int,
    phase: str,
    histogram: dict[int, float],
    common_cost: float,
    measured_cost: float | None,
    boundary_multiplier: int,
) -> dict:
    logical_bytes = {
        str(payload): float(payload) * calls for payload, calls in histogram.items()
    }
    return {
        "label_id": (
            f"qwen3-8b/{parallelism}{parallel_size}/{policy}/"
            f"{profile['profile_id']}/{sample_label}/{phase}"
        ),
        "profile_id": profile["profile_id"],
        "source": profile["source"],
        "segment": profile["segment"],
        "split": profile["split"],
        "parallelism": parallelism,
        "parallel_size": parallel_size,
        "policy": policy,
        "estimator_kind": estimator_kind,
        "sample_label": sample_label,
        "sample_requests": sample_requests,
        "full_requests": full_requests,
        "full_ge_128": full_requests >= 128,
        "phase": phase,
        "normalization_requests": 1000,
        "scope": "group-level TP collective" if parallelism == "tp" else "single sender boundary",
        "boundary_multiplier": boundary_multiplier,
        "total_calls_per_1000": float(sum(histogram.values())),
        "total_logical_bytes_per_1000": float(
            sum(payload * calls for payload, calls in histogram.items())
        ),
        "pipeline_calls_per_1000": float(sum(histogram.values())) * boundary_multiplier,
        "pipeline_logical_bytes_per_1000": float(
            sum(payload * calls for payload, calls in histogram.items())
        )
        * boundary_multiplier,
        "exact_calls_histogram_per_1000_json": json.dumps(
            {str(key): value for key, value in histogram.items()}, separators=(",", ":")
        ),
        "exact_logical_bytes_histogram_per_1000_json": json.dumps(
            logical_bytes, separators=(",", ":")
        ),
        "common_reference_cost_us_per_1000": common_cost,
        "pipeline_common_reference_cost_us_per_1000": common_cost * boundary_multiplier,
        "tp_measured_b200_l1_cost_us_per_1000": (
            "" if measured_cost is None else measured_cost
        ),
    }


def l1_tv(predicted: dict, truth: dict) -> tuple[float, float]:
    predicted_total = max(float(sum(predicted.values())), 1e-12)
    truth_total = max(float(sum(truth.values())), 1e-12)
    keys = set(predicted) | set(truth)
    l1 = sum(
        abs(predicted.get(key, 0.0) / predicted_total - truth.get(key, 0.0) / truth_total)
        for key in keys
    )
    return float(l1), float(l1 / 2.0)


def log_payload_emd(predicted: dict[int, float], truth: dict[int, float]) -> tuple[float, float]:
    supports = sorted(set(predicted) | set(truth))
    if len(supports) <= 1:
        return 0.0, 0.0
    predicted_total = max(float(sum(predicted.values())), 1e-12)
    truth_total = max(float(sum(truth.values())), 1e-12)
    cdf_predicted = 0.0
    cdf_truth = 0.0
    emd = 0.0
    for left, right in zip(supports[:-1], supports[1:]):
        cdf_predicted += predicted.get(left, 0.0) / predicted_total
        cdf_truth += truth.get(left, 0.0) / truth_total
        emd += abs(cdf_predicted - cdf_truth) * (
            math.log2(right) - math.log2(left)
        )
    normalized = emd / (math.log2(MAX_PAYLOAD) - math.log2(MIN_PAYLOAD))
    return float(emd), float(normalized)


def combine_phase_histograms(
    phase_histograms: dict[str, dict[int, float]], phase: str
) -> tuple[dict, dict[int, float]]:
    if phase in PHASES:
        histogram = phase_histograms[phase]
        return dict(histogram), dict(histogram)
    phase_aware = {
        (current_phase, payload): calls
        for current_phase, histogram in phase_histograms.items()
        for payload, calls in histogram.items()
    }
    pooled: Counter[int] = Counter()
    for histogram in phase_histograms.values():
        pooled.update(histogram)
    return phase_aware, dict(pooled)


def metric_row(
    profile: dict[str, str],
    parallelism: str,
    parallel_size: int,
    policy: str,
    sample_label: str,
    estimator_kind: str,
    phase: str,
    predicted_by_phase: dict[str, dict[int, float]],
    truth_by_phase: dict[str, dict[int, float]],
    measured_curves: dict[int, list[tuple[int, float]]],
    reference_label: str,
) -> dict:
    predicted_aware, predicted_pooled = combine_phase_histograms(predicted_by_phase, phase)
    truth_aware, truth_pooled = combine_phase_histograms(truth_by_phase, phase)
    calls_predicted = float(sum(predicted_aware.values()))
    calls_truth = float(sum(truth_aware.values()))
    bytes_predicted = float(
        sum((key[1] if isinstance(key, tuple) else key) * value for key, value in predicted_aware.items())
    )
    bytes_truth = float(
        sum((key[1] if isinstance(key, tuple) else key) * value for key, value in truth_aware.items())
    )
    calls_l1, calls_tv = l1_tv(predicted_aware, truth_aware)
    predicted_bytes_distribution = {
        key: (key[1] if isinstance(key, tuple) else key) * value
        for key, value in predicted_aware.items()
    }
    truth_bytes_distribution = {
        key: (key[1] if isinstance(key, tuple) else key) * value
        for key, value in truth_aware.items()
    }
    bytes_l1, bytes_tv = l1_tv(
        predicted_bytes_distribution, truth_bytes_distribution
    )
    emd, normalized_emd = log_payload_emd(predicted_pooled, truth_pooled)
    common_predicted = histogram_cost(predicted_pooled, common_reference_cost_us)
    common_truth = histogram_cost(truth_pooled, common_reference_cost_us)
    if parallelism == "tp":
        measured_predicted = histogram_cost(
            predicted_pooled,
            lambda payload: measured_interpolate(measured_curves[parallel_size], payload),
        )
        measured_truth = histogram_cost(
            truth_pooled,
            lambda payload: measured_interpolate(measured_curves[parallel_size], payload),
        )
        measured_ape = abs(measured_predicted - measured_truth) / max(
            measured_truth, 1e-12
        )
    else:
        measured_predicted = ""
        measured_truth = ""
        measured_ape = ""
    return {
        "profile_id": profile["profile_id"],
        "source": profile["source"],
        "segment": profile["segment"],
        "split": profile["split"],
        "parallelism": parallelism,
        "parallel_size": parallel_size,
        "policy": policy,
        "phase": phase,
        "estimator_kind": estimator_kind,
        "sample_label": sample_label,
        "reference_label": reference_label,
        "sample_requests": 32 if sample_label == "compact32" else int(sample_label[1:]) if sample_label.startswith("h") and sample_label != "hfull" else int(profile["request_count"]),
        "full_requests": int(profile["request_count"]),
        "full_ge_128": int(profile["request_count"]) >= 128,
        "calls_predicted": calls_predicted,
        "calls_truth": calls_truth,
        "calls_abs_error": abs(calls_predicted - calls_truth),
        "calls_ape": abs(calls_predicted - calls_truth) / max(calls_truth, 1e-12),
        "bytes_predicted": bytes_predicted,
        "bytes_truth": bytes_truth,
        "bytes_abs_error": abs(bytes_predicted - bytes_truth),
        "bytes_ape": abs(bytes_predicted - bytes_truth) / max(bytes_truth, 1e-12),
        "calls_histogram_l1": calls_l1,
        "calls_histogram_tv": calls_tv,
        "bytes_histogram_l1": bytes_l1,
        "bytes_histogram_tv": bytes_tv,
        "log_payload_emd_log2_bytes": emd,
        "normalized_log_payload_emd": normalized_emd,
        "common_reference_cost_predicted_us": common_predicted,
        "common_reference_cost_truth_us": common_truth,
        "common_reference_cost_abs_error_us": abs(common_predicted - common_truth),
        "common_reference_cost_ape": abs(common_predicted - common_truth)
        / max(common_truth, 1e-12),
        "tp_measured_l1_cost_predicted_us": measured_predicted,
        "tp_measured_l1_cost_truth_us": measured_truth,
        "tp_measured_l1_cost_ape": measured_ape,
    }


def build_metrics(
    profiles: list[dict[str, str]],
    lookup: dict[tuple, dict[int, float]],
    measured_curves: dict[int, list[tuple[int, float]]],
) -> tuple[list[dict], list[dict]]:
    requested_rows = []
    decomposition_rows = []
    configs = []
    for tp in TPS:
        for policy in STRATEGIES:
            configs.append(("tp", tp, policy))
    for pp_size in PP_SIZES:
        for max_microbatch in PP_MICROBATCH_SIZES:
            configs.append(("pp", pp_size, f"mb{max_microbatch}"))
    for profile in profiles:
        profile_id = profile["profile_id"]
        for parallelism, parallel_size, policy in configs:
            truth_by_phase = {
                phase: lookup[
                    (
                        profile_id,
                        parallelism,
                        parallel_size,
                        policy,
                        "hfull",
                        phase,
                    )
                ]
                for phase in PHASES
            }
            for sample_label in (*REQUESTED_ESTIMATORS, "compact32"):
                predicted_by_phase = {
                    phase: lookup[
                        (
                            profile_id,
                            parallelism,
                            parallel_size,
                            policy,
                            sample_label,
                            phase,
                        )
                    ]
                    for phase in PHASES
                }
                for phase in (*PHASES, "total"):
                    requested_rows.append(
                        metric_row(
                            profile,
                            parallelism,
                            parallel_size,
                            policy,
                            sample_label,
                            "compact_profile"
                            if sample_label == "compact32"
                            else "exact_representative",
                            phase,
                            predicted_by_phase,
                            truth_by_phase,
                            measured_curves,
                            "hfull",
                        )
                    )
            compact_by_phase = {
                phase: lookup[
                    (
                        profile_id,
                        parallelism,
                        parallel_size,
                        policy,
                        "compact32",
                        phase,
                    )
                ]
                for phase in PHASES
            }
            exact_h32_by_phase = {
                phase: lookup[
                    (
                        profile_id,
                        parallelism,
                        parallel_size,
                        policy,
                        "h32",
                        phase,
                    )
                ]
                for phase in PHASES
            }
            for phase in (*PHASES, "total"):
                decomposition_rows.append(
                    metric_row(
                        profile,
                        parallelism,
                        parallel_size,
                        policy,
                        "compact32",
                        "compact_profile",
                        phase,
                        compact_by_phase,
                        exact_h32_by_phase,
                        measured_curves,
                        "h32",
                    )
                )
    return requested_rows, decomposition_rows


def aggregate_group(rows: list[dict], scope: str, dimensions: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[dimension] for dimension in dimensions)].append(row)
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        result = {
            "aggregation_scope": scope,
            "parallelism": "",
            "sample_label": "",
            "estimator_kind": "",
            "reference_label": "",
            "phase": "",
            "policy": "",
            "parallel_size": "",
            "source": "",
            "full_ge_128": "",
        }
        result.update(dict(zip(dimensions, key)))
        calls_apes = np.asarray([float(row["calls_ape"]) for row in values])
        bytes_apes = np.asarray([float(row["bytes_ape"]) for row in values])
        cost_apes = np.asarray(
            [float(row["common_reference_cost_ape"]) for row in values]
        )
        measured_apes = np.asarray(
            [
                float(row["tp_measured_l1_cost_ape"])
                for row in values
                if row["tp_measured_l1_cost_ape"] != ""
            ]
        )
        result.update(
            {
                "cases": len(values),
                "calls_mape": float(np.mean(calls_apes)),
                "calls_wape": sum(float(row["calls_abs_error"]) for row in values)
                / max(sum(float(row["calls_truth"]) for row in values), 1e-12),
                "p95_calls_ape": float(np.quantile(calls_apes, 0.95)),
                "bytes_mape": float(np.mean(bytes_apes)),
                "bytes_wape": sum(float(row["bytes_abs_error"]) for row in values)
                / max(sum(float(row["bytes_truth"]) for row in values), 1e-12),
                "p95_bytes_ape": float(np.quantile(bytes_apes, 0.95)),
                "mean_calls_histogram_l1": float(
                    np.mean([float(row["calls_histogram_l1"]) for row in values])
                ),
                "mean_calls_histogram_tv": float(
                    np.mean([float(row["calls_histogram_tv"]) for row in values])
                ),
                "mean_bytes_histogram_l1": float(
                    np.mean([float(row["bytes_histogram_l1"]) for row in values])
                ),
                "mean_bytes_histogram_tv": float(
                    np.mean([float(row["bytes_histogram_tv"]) for row in values])
                ),
                "mean_log_payload_emd_log2_bytes": float(
                    np.mean([float(row["log_payload_emd_log2_bytes"]) for row in values])
                ),
                "mean_normalized_log_payload_emd": float(
                    np.mean([float(row["normalized_log_payload_emd"]) for row in values])
                ),
                "common_reference_cost_mape": float(np.mean(cost_apes)),
                "common_reference_cost_wape": sum(
                    float(row["common_reference_cost_abs_error_us"]) for row in values
                )
                / max(
                    sum(float(row["common_reference_cost_truth_us"]) for row in values),
                    1e-12,
                ),
                "p95_common_reference_cost_ape": float(np.quantile(cost_apes, 0.95)),
                "tp_measured_l1_cost_mape": (
                    "" if not len(measured_apes) else float(np.mean(measured_apes))
                ),
                "p95_tp_measured_l1_cost_ape": (
                    "" if not len(measured_apes) else float(np.quantile(measured_apes, 0.95))
                ),
            }
        )
        output.append(result)
    return output


def build_aggregates(rows: list[dict]) -> list[dict]:
    base = (
        "parallelism",
        "sample_label",
        "estimator_kind",
        "reference_label",
        "phase",
    )
    output = []
    output.extend(aggregate_group(rows, "overall", base))
    output.extend(aggregate_group(rows, "policy", base + ("policy",)))
    output.extend(
        aggregate_group(rows, "parallel_size", base + ("parallel_size",))
    )
    output.extend(aggregate_group(rows, "source", base + ("source",)))
    output.extend(aggregate_group(rows, "cohort", base + ("full_ge_128",)))
    return output


def overall_row(
    aggregates: list[dict],
    parallelism: str,
    sample_label: str,
    phase: str = "total",
    reference_label: str = "hfull",
) -> dict:
    matches = [
        row
        for row in aggregates
        if row["aggregation_scope"] == "overall"
        and row["parallelism"] == parallelism
        and row["sample_label"] == sample_label
        and row["phase"] == phase
        and row["reference_label"] == reference_label
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one aggregate row for {parallelism}/{sample_label}/{phase}/{reference_label}, got {len(matches)}"
        )
    return matches[0]


def meets_thresholds(row: dict) -> bool:
    return all(
        [
            float(row["calls_mape"]) <= CONVERGENCE_THRESHOLDS["calls_mape"],
            float(row["calls_wape"]) <= CONVERGENCE_THRESHOLDS["calls_wape"],
            float(row["bytes_mape"]) <= CONVERGENCE_THRESHOLDS["bytes_mape"],
            float(row["bytes_wape"]) <= CONVERGENCE_THRESHOLDS["bytes_wape"],
            float(row["mean_calls_histogram_tv"])
            <= CONVERGENCE_THRESHOLDS["histogram_tv"],
            float(row["mean_normalized_log_payload_emd"])
            <= CONVERGENCE_THRESHOLDS["normalized_log_payload_emd"],
            float(row["common_reference_cost_mape"])
            <= CONVERGENCE_THRESHOLDS["common_reference_cost_mape"],
            float(row["p95_calls_ape"])
            <= CONVERGENCE_THRESHOLDS["p95_calls_ape"],
            float(row["p95_bytes_ape"])
            <= CONVERGENCE_THRESHOLDS["p95_bytes_ape"],
            float(row["p95_common_reference_cost_ape"])
            <= CONVERGENCE_THRESHOLDS["p95_common_reference_cost_ape"],
        ]
    )


def convergence_decisions(aggregates: list[dict]) -> dict:
    decisions = {}
    for parallelism in ("tp", "pp"):
        rows = [overall_row(aggregates, parallelism, label) for label in REQUESTED_ESTIMATORS]
        sufficient = [
            label
            for label, row in zip(REQUESTED_ESTIMATORS, rows)
            if meets_thresholds(row)
        ]
        decisions[parallelism] = {
            "minimum_sample_meeting_all_preregistered_thresholds": (
                sufficient[0] if sufficient else None
            ),
            "by_sample": {
                label: {
                    "meets_all": meets_thresholds(row),
                    "calls_mape": row["calls_mape"],
                    "calls_wape": row["calls_wape"],
                    "bytes_mape": row["bytes_mape"],
                    "bytes_wape": row["bytes_wape"],
                    "histogram_tv": row["mean_calls_histogram_tv"],
                    "normalized_log_payload_emd": row[
                        "mean_normalized_log_payload_emd"
                    ],
                    "common_reference_cost_mape": row[
                        "common_reference_cost_mape"
                    ],
                    "p95_calls_ape": row["p95_calls_ape"],
                }
                for label, row in zip(REQUESTED_ESTIMATORS, rows)
            },
        }
    return decisions


def pp_error_decomposition(
    aggregates: list[dict], decomposition_aggregates: list[dict]
) -> dict:
    exact_h32 = overall_row(aggregates, "pp", "h32")
    exact_h64 = overall_row(aggregates, "pp", "h64")
    exact_h128 = overall_row(aggregates, "pp", "h128")
    compact_full = overall_row(aggregates, "pp", "compact32")
    compact_exact = overall_row(
        decomposition_aggregates,
        "pp",
        "compact32",
        reference_label="h32",
    )
    size_error = float(exact_h32["calls_mape"])
    h64_size_error = float(exact_h64["calls_mape"])
    residual_size_error = float(exact_h128["calls_mape"])
    reconstruction_error = float(compact_exact["calls_mape"])
    compact_full_error = float(compact_full["calls_mape"])
    if reconstruction_error > size_error:
        inference = (
            "both finite-sample and compact-profile reconstruction errors are material; "
            "compact32 differs from exact H32 more than exact H32 differs from Hfull, "
            "but partial cancellation against Hfull makes the two components non-additive"
        )
    else:
        inference = (
            "both finite-sample and compact-profile reconstruction errors are material; "
            "finite-sample error is at least as large, and the two components are non-additive"
        )
    return {
        "exact_h32_vs_hfull_calls_mape": size_error,
        "exact_h64_vs_hfull_calls_mape": h64_size_error,
        "exact_h128_vs_hfull_calls_mape": residual_size_error,
        "compact32_vs_exact_h32_calls_mape": reconstruction_error,
        "compact32_vs_hfull_calls_mape": compact_full_error,
        "comparison_is_non_additive": True,
        "full_window_teacher_recommended": True,
        "inference": inference,
    }


def pct(value) -> str:
    return f"{100 * float(value):.2f}%"


def make_svg(path: Path, aggregates: list[dict]) -> None:
    metrics = [
        ("calls_mape", "Calls MAPE", 100.0),
        ("bytes_mape", "Logical bytes MAPE", 100.0),
        ("mean_calls_histogram_tv", "Histogram TV", 1.0),
        ("mean_normalized_log_payload_emd", "Normalized log-payload EMD", 1.0),
        ("common_reference_cost_mape", "Reference cost MAPE", 100.0),
        ("p95_calls_ape", "P95 calls APE", 100.0),
    ]
    width, height = 1080, 650
    panel_w, panel_h = 330, 250
    lefts = [55, 375, 695]
    tops = [55, 360]
    sample_x = {"h32": 0, "h64": 1, "h128": 2}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">TP and PP representative-request convergence</title>',
        '<desc id="desc">Six small multiples compare H32, H64, and H128 against full-window fixed-draining labels for TP and PP.</desc>',
        '<style>text{font-family:system-ui,sans-serif;fill:currentColor;font-size:12px}.title{font-size:14px;font-weight:500}.axis,.grid,.series{fill:none;stroke:currentColor}.axis{stroke-width:1}.grid{stroke-width:1;opacity:.16}.series{stroke-width:2}.pp{stroke-dasharray:6 4}.marker{fill:var(--background,transparent);stroke:currentColor;stroke-width:2}</style>',
    ]
    for index, (field, title, scale) in enumerate(metrics):
        x0 = lefts[index % 3]
        y0 = tops[index // 3]
        plot_left, plot_top = x0 + 50, y0 + 25
        plot_width, plot_height = panel_w - 80, panel_h - 55
        values = []
        series = {}
        for parallelism in ("tp", "pp"):
            current = []
            for sample in REQUESTED_ESTIMATORS:
                value = float(overall_row(aggregates, parallelism, sample)[field]) * scale
                current.append(value)
                values.append(value)
            series[parallelism] = current
        ymax = max(values) * 1.12 if max(values) > 0 else 1.0
        parts.append(f'<text class="title" x="{x0}" y="{y0 + 10}">{title}</text>')
        for tick in range(5):
            value = ymax * tick / 4
            y = plot_top + plot_height - plot_height * tick / 4
            parts.append(
                f'<line class="grid" x1="{plot_left}" y1="{y:.1f}" x2="{plot_left + plot_width}" y2="{y:.1f}"/>'
            )
            parts.append(
                f'<text x="{plot_left - 7}" y="{y + 4:.1f}" text-anchor="end">{value:.1f}</text>'
            )
        parts.append(
            f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}"/>'
        )
        parts.append(
            f'<line class="axis" x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}"/>'
        )
        for sample, position in sample_x.items():
            x = plot_left + position * plot_width / 2
            parts.append(
                f'<text x="{x:.1f}" y="{plot_top + plot_height + 20}" text-anchor="middle">{sample.upper()}</text>'
            )
        for parallelism, current in series.items():
            points = []
            for position, value in enumerate(current):
                x = plot_left + position * plot_width / 2
                y = plot_top + plot_height - value / ymax * plot_height
                points.append((x, y))
            point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            css = "series pp" if parallelism == "pp" else "series"
            parts.append(f'<polyline class="{css}" points="{point_text}"/>')
            for x, y in points:
                if parallelism == "pp":
                    parts.append(
                        f'<rect class="marker" x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8"/>'
                    )
                else:
                    parts.append(f'<circle class="marker" cx="{x:.1f}" cy="{y:.1f}" r="4"/>')
            label_x, label_y = points[-1]
            parts.append(
                f'<text x="{label_x + 8:.1f}" y="{label_y + (4 if parallelism == "tp" else 16):.1f}">{parallelism.upper()}</text>'
            )
    parts.append("</svg>\n")
    path.write_text("".join(parts))


def readme_text(summary: dict, aggregates: list[dict]) -> str:
    table_rows = []
    for parallelism in ("tp", "pp"):
        for sample in REQUESTED_ESTIMATORS:
            row = overall_row(aggregates, parallelism, sample)
            table_rows.append(
                "| {par} | {sample} | {calls_mape} | {calls_wape} | {bytes_mape} | {bytes_wape} | {tv:.4f} | {emd:.4f} | {cost} | {p95} |".format(
                    par=parallelism.upper(),
                    sample=sample.upper(),
                    calls_mape=pct(row["calls_mape"]),
                    calls_wape=pct(row["calls_wape"]),
                    bytes_mape=pct(row["bytes_mape"]),
                    bytes_wape=pct(row["bytes_wape"]),
                    tv=float(row["mean_calls_histogram_tv"]),
                    emd=float(row["mean_normalized_log_payload_emd"]),
                    cost=pct(row["common_reference_cost_mape"]),
                    p95=pct(row["p95_calls_ape"]),
                )
            )
    decomposition = summary["pp_error_decomposition"]
    decisions = summary["convergence_decisions"]
    return f"""# Phase 24：代表请求规模 H32/H64/H128/Hfull 收敛

本实验复用 Phase 16 固定的24个BurstGPT/Mooncake medoid历史窗口，在相同
fixed-draining语义下用GPU验证过的CPU结构公式生成Qwen3-8B TP和PP消息直方图。
Hfull是唯一参考；完整请求列表只用于离线teacher label，不是部署时预测器输入。

## 输入与策略

- 完整窗口：24个，合计{summary['input_windows']['total_full_requests']:,}条请求；
- H128：Phase 16的固定4×4联合长度分层代表池；
- H32/H64：从同一H128池再次确定性分层选择，H32逐项匹配既有Phase 16 replay plan；
- compact32：仅由低维画像重建的32个伪请求，用于分离画像重建误差；
- TP：TP2/4/8 × latency/balanced/throughput；
- PP：PP2/4/8 × `pp_max_micro_batch_size=1/4/16`，每条sender boundary计数一次；
- 所有结果归一化到每1000请求，Prefill/Decode另存并提供total汇总。

## 主要结果（phase=total，Hfull为真值）

| 并行 | 样本 | calls MAPE | calls WAPE | bytes MAPE | bytes WAPE | hist TV | norm EMD | common cost MAPE | P95 calls APE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

预注册门槛下最小充分规模：TP=`{decisions['tp']['minimum_sample_meeting_all_preregistered_thresholds']}`，
PP=`{decisions['pp']['minimum_sample_meeting_all_preregistered_thresholds']}`。门槛和逐配置结果见
`summary.json`与`analysis/aggregate_metrics.csv`，不能只按平均calls MAPE挑选样本规模。

## PP误差分解

- exact H32→Hfull calls MAPE：{pct(decomposition['exact_h32_vs_hfull_calls_mape'])}；
- exact H64→Hfull calls MAPE：{pct(decomposition['exact_h64_vs_hfull_calls_mape'])}；
- exact H128→Hfull calls MAPE：{pct(decomposition['exact_h128_vs_hfull_calls_mape'])}；
- compact32→exact H32 calls MAPE：{pct(decomposition['compact32_vs_exact_h32_calls_mape'])}；
- compact32→Hfull calls MAPE：{pct(decomposition['compact32_vs_hfull_calls_mape'])}；
- 诊断：{decomposition['inference']}。

因没有任一规模同时满足均值、尾部与直方图门槛，首版teacher label应使用
full-window fixed-draining H0；H64/H128可继续作为低成本近似，但不应直接替代teacher。

## Cost口径与结论边界

跨TP/PP的`common_reference_cost`统一使用显式参考曲线
`5 us + payload / 100 GB/s`，只评价同一曲线下的直方图收敛，不是PP物理时延。
TP另在逐样本与聚合文件中报告B200 L1 AllReduce实测连续曲线传播误差。PP P2P物理曲线
尚未测量，因此不能从本实验报告PP真实通信时间MAPE。

本实验可以判断代表请求规模是否逼近full-window fixed-draining teacher，以及画像重建
相对精确代表样本造成多少额外误差；不能证明online arrival期望直方图，也不能替代未来
的PP P2P曲线、多模型PP或预测器留出评测。

## 正式资产

- `input_windows/selected_requests.jsonl.gz`：H32/H64/H128/Hfull与compact32精确长度列表；
- `labels/histogram_labels.jsonl.gz`：phase×配置×payload的每1000请求精确标签；
- `analysis/per_case_metrics.csv.gz`：逐窗口、配置、phase/total误差；
- `analysis/decomposition_metrics.csv.gz`：compact32相对exact H32的画像重建误差；
- `analysis/aggregate_metrics.csv`：MAPE/WAPE/P95/L1/TV/EMD/cost汇总；
- `figures/convergence.svg`：TP/PP收敛曲线；
- `summary.json`、`run.log`、`DONE`、`PIPELINE_DONE`与`manifest.sha256`：审计和完整性。
"""


def main() -> None:
    started = time.time()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("input_windows", "labels", "analysis", "figures"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)

    profiles = load_profiles(args.profiles)
    representatives = load_representatives(args.representatives)
    phase16_h32 = load_phase16_h32_plan(args.phase16_plan)
    raw_checks, raw_hashes = verify_raw_sources(args.raw_dir)
    request_sets, archived_requests, input_audit = materialize_request_sets(
        profiles, representatives, phase16_h32, args.raw_dir
    )
    write_jsonl_gz(
        args.output_dir / "input_windows/selected_requests.jsonl.gz",
        archived_requests,
    )

    model_features = {
        row["model"]: row for row in json.loads(args.model_features.read_text())
    }["qwen3-8b"]
    model_checks = {
        "logical_collectives_per_forward_is_73": int(
            model_features["logical_collectives_per_forward_prior"]
        )
        == TP_CALLS_PER_FORWARD,
        "payload_bytes_per_token_is_8192": int(
            model_features["payload_bytes_per_active_token_prior"]
        )
        == PAYLOAD_BYTES_PER_TOKEN,
        "hidden_size_is_4096": int(model_features["hidden_size"]) == 4096,
        "dtype_bytes_is_2": int(model_features["dtype_bytes"]) == 2,
    }
    measured_curves = load_measured_l1_curves(
        args.l1_curve_root, args.l1_curve_extension
    )
    labels, lookup = label_rows(profiles, request_sets, measured_curves)
    write_jsonl_gz(
        args.output_dir / "labels/histogram_labels.jsonl.gz", labels
    )
    metrics, decomposition = build_metrics(profiles, lookup, measured_curves)
    write_csv_gz(args.output_dir / "analysis/per_case_metrics.csv.gz", metrics)
    write_csv_gz(
        args.output_dir / "analysis/decomposition_metrics.csv.gz", decomposition
    )
    aggregates = build_aggregates(metrics)
    decomposition_aggregates = build_aggregates(decomposition)
    all_aggregates = aggregates + decomposition_aggregates
    write_csv(
        args.output_dir / "analysis/aggregate_metrics.csv", all_aggregates
    )

    decisions = convergence_decisions(aggregates)
    error_decomposition = pp_error_decomposition(
        aggregates, decomposition_aggregates
    )
    make_svg(args.output_dir / "figures/convergence.svg", aggregates)

    finite_metrics = all(
        math.isfinite(float(row[field]))
        for row in metrics
        for field in (
            "calls_ape",
            "bytes_ape",
            "calls_histogram_l1",
            "normalized_log_payload_emd",
            "common_reference_cost_ape",
        )
    )
    checks = {
        "raw_source_hashes_match_manifest": all(raw_checks.values()),
        "profiles_24": len(profiles) == 24,
        "full_request_counts_match_profiles": input_audit[
            "full_request_counts_match_profiles"
        ],
        "stored_h128_matches_regenerated_phase16_sampling": input_audit[
            "stored_h128_matches_regenerated_phase16_sampling"
        ],
        "h32_matches_all_72_phase16_strategy_plans": input_audit[
            "h32_matches_all_phase16_strategy_replay_plans"
        ]
        and input_audit["h32_plan_comparisons"] == 72,
        "model_structure_matches_qwen3_8b_contract": all(model_checks.values()),
        "histogram_label_rows_4320": len(labels) == 24 * 18 * 5 * 2,
        "per_case_metric_rows_5184": len(metrics) == 24 * 18 * 4 * 3,
        "decomposition_rows_1296": len(decomposition) == 24 * 18 * 3,
        "all_metrics_finite": finite_metrics,
        "all_histograms_positive": all(
            float(row["total_calls_per_1000"]) > 0
            and float(row["total_logical_bytes_per_1000"]) > 0
            for row in labels
        ),
    }
    summary = {
        "schema_version": "phase24-representative-request-convergence-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "objective": (
            "Compare exact representative H32/H64/H128 and compact-profile H32 "
            "with full-window fixed-draining TP/PP H0 labels, all normalized per 1000 requests."
        ),
        "teacher_contract": (
            "Hfull uses complete capped request lengths in original order under fixed-draining; "
            "complete request lists are offline label-only inputs."
        ),
        "input_windows": {
            **input_audit,
            "raw_source_hashes": raw_hashes,
            "raw_manifest_sha256": sha256(args.raw_dir / "source_manifest.json"),
            "profiles_sha256": sha256(args.profiles),
            "representatives_sha256": sha256(args.representatives),
            "phase16_plan_sha256": sha256(args.phase16_plan),
        },
        "model": "qwen3-8b",
        "model_contract": model_features,
        "fixed_policies": {
            "tp": STRATEGIES,
            "pp_sizes": PP_SIZES,
            "pp_max_micro_batch_sizes": PP_MICROBATCH_SIZES,
            "pp_chunk_tokens": PP_CHUNK_TOKENS,
            "arrival_semantics": "fixed-draining; simultaneous within each canonical workload",
        },
        "cost_contract": {
            "common_reference": {
                "kind": "parameterized convergence reference, not physical PP measurement",
                "formula": "launch_us + payload_bytes / bandwidth_bytes_per_second * 1e6",
                "launch_us": COMMON_REFERENCE_LAUNCH_US,
                "bandwidth_gbps": COMMON_REFERENCE_BANDWIDTH_GBPS,
            },
            "tp_additional_curve": (
                "measured B200 L1 AllReduce post-rendezvous curve with Phase 15 160-512 MiB extension"
            ),
            "pp_physical_curve_available": False,
        },
        "normalization_requests": 1000,
        "label_rows": len(labels),
        "per_case_metric_rows": len(metrics),
        "decomposition_metric_rows": len(decomposition),
        "aggregate_rows": len(all_aggregates),
        "preregistered_convergence_thresholds": CONVERGENCE_THRESHOLDS,
        "convergence_decisions": decisions,
        "pp_error_decomposition": error_decomposition,
        "checks": checks,
        "conclusion_boundary": {
            "can_conclude": (
                "representative-size convergence relative to full-window fixed-draining structural labels"
            ),
            "cannot_conclude": [
                "PP physical communication-time accuracy because no PP P2P curve is measured",
                "online arrival-aware expected histograms",
                "multi-model PP generalization",
                "prediction from compact profiles without a separate holdout evaluation",
            ],
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "README.md").write_text(readme_text(summary, aggregates))
    run_log = {
        "status": summary["status"],
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "duration_seconds": time.time() - started,
        "checks": checks,
        "output_dir": str(args.output_dir),
    }
    write_json(args.output_dir / "run.log", run_log)
    if summary["status"] != "PASS":
        raise RuntimeError(summary)
    (args.output_dir / "DONE").write_text("PASS\n")
    (args.output_dir / "PIPELINE_DONE").write_text("PASS\n")
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
