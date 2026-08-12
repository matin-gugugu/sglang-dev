#!/usr/bin/env python3
"""Build Phase 27 PP low-dimensional profiles and scheduler-faithful Hfull labels."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_phase21b_pp_h0 import pseudo_requests
from build_phase25_full_window_teacher import PP_BIN_EDGES
from build_phase25b_pp_scheduler_teacher import (
    BYTES_PER_TOKEN,
    PAGE_SIZE,
    PP_CHUNK_TOKENS,
    PP_PROXY_COUNT,
    simulate_scheduler,
)
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


HISTORY_SECONDS = 300
INPUT_CAP = 8192
OUTPUT_CAP = 128
PP_SIZES = (2, 4, 8)
MICROBATCH_SIZES = (1, 4, 16)
PHASES = ("prefill", "decode")
INPUT_EDGES = np.asarray([0, 128, 512, 2048, np.inf], dtype=np.float64)
OUTPUT_EDGES = np.asarray([0, 16, 32, 64, np.inf], dtype=np.float64)
OUTPUT_BUCKET_UPPER = np.asarray([8, 16, 32, 64], dtype=np.int64)
COMMON_REFERENCE_LAUNCH_US = 5.0
COMMON_REFERENCE_BANDWIDTH_GBPS = 100.0
LABEL_STATUS = "GPU_VALIDATED_PHASE25B_9_OF_9_PLUS_PHASE25C_3_OF_3"
TEACHER_KIND = "sglang_pp_fcfs_lanes_v1_hfull"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=root
        / "experiment-results/phase27a_pp_feature_and_holdout_contract/selection/selected_windows.csv",
    )
    parser.add_argument(
        "--phase27a-summary",
        type=Path,
        default=root
        / "experiment-results/phase27a_pp_feature_and_holdout_contract/summary.json",
    )
    parser.add_argument(
        "--feature-contract",
        type=Path,
        default=root
        / "experiment-results/phase27a_pp_feature_and_holdout_contract/feature_contract.json",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase27b_pp_hfull_dataset",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def safe_cv(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    mean = float(np.mean(values))
    return float(np.std(values) / mean) if mean else 0.0


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else 0.0


def rolling_fraction_max(values: np.ndarray, width: int) -> float:
    if not len(values):
        return 0.0
    if len(values) <= width:
        return float(np.mean(values))
    sums = np.convolve(values.astype(np.float64), np.ones(width), mode="valid")
    return float(np.max(sums) / width)


def positive_run_lengths(values: np.ndarray) -> list[int]:
    runs = []
    current = 0
    for value in values:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def summarize_profile(
    selection: dict[str, str], timestamps: np.ndarray, inputs: np.ndarray, outputs: np.ndarray
) -> tuple[dict, list[tuple[int, int]]]:
    clipped_l = np.clip(inputs, 1, INPUT_CAP).astype(np.int64)
    clipped_m = np.clip(outputs, 1, OUTPUT_CAP).astype(np.int64)
    if not len(clipped_l):
        raise ValueError(f"{selection['phase27_profile_id']}: empty window")

    joint, _, _ = np.histogram2d(clipped_l, clipped_m, bins=(INPUT_EDGES, OUTPUT_EDGES))
    joint = joint / float(np.sum(joint))
    interarrival = np.diff(timestamps) / 1000.0
    cutoff = int(selection["cutoff_ms"])
    seconds = np.clip(
        ((timestamps - (cutoff - HISTORY_SECONDS * 1000)) // 1000).astype(int),
        0,
        HISTORY_SECONDS - 1,
    )
    per_second = np.bincount(seconds, minlength=HISTORY_SECONDS)
    mean_per_second = float(np.mean(per_second))
    peak_to_mean = float(np.max(per_second) / mean_per_second) if mean_per_second else 0.0
    fano = float(np.var(per_second) / mean_per_second) if mean_per_second else 0.0
    correlation = (
        float(np.corrcoef(clipped_l, clipped_m)[0, 1])
        if len(clipped_l) > 1 and np.std(clipped_l) and np.std(clipped_m)
        else 0.0
    )

    chunk_count = np.ceil(clipped_l / PP_CHUNK_TOKENS).astype(np.int64)
    chunk_class = (chunk_count > 1).astype(np.int64)
    output_bucket = np.searchsorted(OUTPUT_BUCKET_UPPER, clipped_m, side="left")
    chunk_output_joint = np.bincount(
        chunk_class * 5 + output_bucket, minlength=10
    ).astype(np.float64) / len(clipped_l)
    if len(chunk_class) > 1:
        transitions = np.bincount(
            chunk_class[:-1] * 2 + chunk_class[1:], minlength=4
        ).astype(np.float64) / (len(chunk_class) - 1)
        transition_rate = float(np.mean(chunk_class[:-1] != chunk_class[1:]))
    else:
        transitions = np.zeros(4, dtype=np.float64)
        transition_rate = 0.0
    runs = positive_run_lengths(chunk_class)
    chunk_output_work = chunk_count * clipped_m

    profile = {
        "profile_id": selection["phase27_profile_id"],
        "phase27_role": selection["phase27_role"],
        "source": selection["source"],
        "segment": selection["segment"],
        "source_split": selection["source_split"],
        "window_id": selection["window_id"],
        "cutoff_ms": cutoff,
        "request_count": len(clipped_l),
        "rps": len(clipped_l) / HISTORY_SECONDS,
        "interarrival_cv": safe_cv(interarrival),
        "peak_to_mean_1s": peak_to_mean,
        "fano_1s": fano,
        "input_mean_raw": float(np.mean(inputs)),
        "input_p50_raw": quantile(inputs, 0.50),
        "input_p90_raw": quantile(inputs, 0.90),
        "input_p99_raw": quantile(inputs, 0.99),
        "input_mean_capped": float(np.mean(clipped_l)),
        "input_p50_capped": quantile(clipped_l, 0.50),
        "input_p90_capped": quantile(clipped_l, 0.90),
        "input_p99_capped": quantile(clipped_l, 0.99),
        "output_mean_raw": float(np.mean(outputs)),
        "output_p50_raw": quantile(outputs, 0.50),
        "output_p90_raw": quantile(outputs, 0.90),
        "output_p99_raw": quantile(outputs, 0.99),
        "output_mean_capped": float(np.mean(clipped_m)),
        "output_p50_capped": quantile(clipped_m, 0.50),
        "output_p90_capped": quantile(clipped_m, 0.90),
        "output_p99_capped": quantile(clipped_m, 0.99),
        "lm_correlation_capped": correlation,
        "survival_m_gt_1": float(np.mean(clipped_m > 1)),
        "survival_m_gt_8": float(np.mean(clipped_m > 8)),
        "survival_m_gt_16": float(np.mean(clipped_m > 16)),
        "survival_m_gt_32": float(np.mean(clipped_m > 32)),
        "survival_m_gt_64": float(np.mean(clipped_m > 64)),
        "input_multichunk_fraction": float(np.mean(chunk_class)),
        "chunk_count_mean": float(np.mean(chunk_count)),
        "chunk_count_p50": quantile(chunk_count, 0.50),
        "chunk_count_p90": quantile(chunk_count, 0.90),
        "chunk_count_p99": quantile(chunk_count, 0.99),
        "chunk_output_work_mean": float(np.mean(chunk_output_work)),
        "chunk_output_work_p90": quantile(chunk_output_work, 0.90),
        "chunk_output_work_p99": quantile(chunk_output_work, 0.99),
        "multichunk_transition_rate": transition_rate,
        "multichunk_run_length_mean": float(statistics.fmean(runs)) if runs else 0.0,
        "multichunk_run_length_p90": float(np.quantile(runs, 0.90)) if runs else 0.0,
        "multichunk_run_length_max": max(runs) if runs else 0,
        "rolling_multichunk_fraction_max_4": rolling_fraction_max(chunk_class, 4),
        "rolling_multichunk_fraction_max_16": rolling_fraction_max(chunk_class, 16),
        "rolling_multichunk_fraction_max_32": rolling_fraction_max(chunk_class, 32),
        "joint_lm_4x4_json": json.dumps(joint.reshape(-1).tolist(), separators=(",", ":")),
        "chunk_output_bucket_joint_2x5_json": json.dumps(
            chunk_output_joint.tolist(), separators=(",", ":")
        ),
        "chunk_class_transition_2x2_json": json.dumps(
            transitions.tolist(), separators=(",", ":")
        ),
    }
    return profile, list(zip(clipped_l.tolist(), clipped_m.tolist()))


def exact_histogram(
    requests: list[tuple[int, int]], pp_size: int, microbatch: int
) -> tuple[dict[str, dict[int, float]], dict]:
    simulated = simulate_scheduler(
        requests, pp_size=pp_size, max_microbatch=microbatch
    )
    scale = 1000.0 / len(requests)
    histograms = {}
    for phase in PHASES:
        histograms[phase] = {
            int(tokens) * BYTES_PER_TOKEN: float(events * PP_PROXY_COUNT) * scale
            for tokens, events in sorted(simulated.event_histograms[phase].items())
        }
    audit = {
        "all_requests_complete": simulated.all_requests_complete,
        "prefill_token_mass": simulated.prefill_token_mass,
        "decode_token_mass": simulated.decode_token_mass,
        "scheduler_visits": simulated.scheduler_visits,
        "max_prefill_batch": simulated.max_active_batch_size["prefill"],
        "max_decode_batch": simulated.max_active_batch_size["decode"],
    }
    return histograms, audit


def bin_vectors(histogram: dict[int, float]) -> tuple[list[float], list[float]]:
    calls = np.zeros(12, dtype=np.float64)
    logical_bytes = np.zeros(12, dtype=np.float64)
    for payload, count in histogram.items():
        index = int(np.clip(np.searchsorted(PP_BIN_EDGES, payload, side="right") - 1, 0, 11))
        calls[index] += count
        logical_bytes[index] += payload * count
    return calls.tolist(), logical_bytes.tolist()


def reference_cost(histogram: dict[int, float]) -> float:
    return float(
        sum(
            calls
            * (
                COMMON_REFERENCE_LAUNCH_US
                + payload / (COMMON_REFERENCE_BANDWIDTH_GBPS * 1e9) * 1e6
            )
            for payload, calls in histogram.items()
        )
    )


def label_row(
    profile: dict,
    pp_size: int,
    microbatch: int,
    phase: str,
    histogram: dict[int, float],
) -> dict:
    calls_bins, bytes_bins = bin_vectors(histogram)
    total_calls = float(sum(histogram.values()))
    total_bytes = float(sum(payload * calls for payload, calls in histogram.items()))
    return {
        "label_id": f"qwen3-8b/pp{pp_size}/mb{microbatch}/{profile['profile_id']}/hfull/{phase}",
        "label_status": LABEL_STATUS,
        "teacher_kind": TEACHER_KIND,
        "model": "qwen3-8b",
        "profile_id": profile["profile_id"],
        "phase27_role": profile["phase27_role"],
        "source": profile["source"],
        "segment": profile["segment"],
        "window_id": profile["window_id"],
        "parallelism": "pp",
        "parallel_size": pp_size,
        "policy": f"mb{microbatch}",
        "phase": phase,
        "requests": profile["request_count"],
        "normalization_requests": 1000,
        "boundary_multiplier": pp_size - 1,
        "bin_schema_id": "pp_native_12bin_4k_8g_v1",
        "bin_edges_bytes_json": json.dumps(PP_BIN_EDGES.tolist(), separators=(",", ":")),
        "total_calls_per_1000": total_calls,
        "total_logical_bytes_per_1000": total_bytes,
        "common_reference_cost_us_per_1000": reference_cost(histogram),
        "pipeline_calls_per_1000": total_calls * (pp_size - 1),
        "pipeline_logical_bytes_per_1000": total_bytes * (pp_size - 1),
        "calls_by_12bin_json": json.dumps(calls_bins, separators=(",", ":")),
        "logical_bytes_by_12bin_json": json.dumps(bytes_bins, separators=(",", ":")),
        "exact_calls_histogram_per_1000_json": json.dumps(
            {str(key): value for key, value in histogram.items()}, separators=(",", ":")
        ),
        "exact_logical_bytes_histogram_per_1000_json": json.dumps(
            {str(key): key * value for key, value in histogram.items()}, separators=(",", ":")
        ),
        "scheduler_contract": "sglang_pp_fcfs_lanes_v1",
        "pp_loop_lanes": pp_size,
        "chunk_tokens": PP_CHUNK_TOKENS,
        "page_size": PAGE_SIZE,
        "proxy_tensor_count": PP_PROXY_COUNT,
    }


def scalar_profile_features(profile: dict) -> dict:
    metadata = {
        "profile_id",
        "phase27_role",
        "source",
        "segment",
        "source_split",
        "window_id",
        "cutoff_ms",
        "joint_lm_4x4_json",
        "chunk_output_bucket_joint_2x5_json",
        "chunk_class_transition_2x2_json",
    }
    result = {
        f"feature_profile_{key}": value
        for key, value in profile.items()
        if key not in metadata
    }
    arrays = {
        "joint_lm": json.loads(profile["joint_lm_4x4_json"]),
        "chunk_output_bucket_joint": json.loads(
            profile["chunk_output_bucket_joint_2x5_json"]
        ),
        "chunk_class_transition": json.loads(profile["chunk_class_transition_2x2_json"]),
    }
    for prefix, values in arrays.items():
        for index, value in enumerate(values):
            result[f"feature_profile_{prefix}_{index}"] = value
    return result


def model_features(path: Path) -> tuple[dict, dict]:
    models = {row["model"]: row for row in json.loads(path.read_text())}
    model = models["qwen3-8b"]
    excluded = {
        "model",
        "config_path",
        "architecture_audit_only",
        "model_type_audit_only",
        "raw_op_template_audit_only",
        "config_sha256",
    }
    features = {
        f"feature_model_{key}": value
        for key, value in model.items()
        if key not in excluded
    }
    return model, features


def training_features(
    profile: dict, model: dict, pp_size: int, microbatch: int, phase: str
) -> dict:
    pressure = microbatch / pp_size
    features = {
        **scalar_profile_features(profile),
        **model,
        "feature_parallelism_pp": 1,
        "feature_parallel_size_log2": math.log2(pp_size),
        "feature_phase_prefill": int(phase == "prefill"),
        "feature_phase_decode": int(phase == "decode"),
        "feature_pp_max_microbatch_size": microbatch,
        "feature_pp_chunk_tokens": PP_CHUNK_TOKENS,
        "feature_pp_page_size": PAGE_SIZE,
        "feature_pp_proxy_tensor_count": PP_PROXY_COUNT,
        "feature_policy_pressure": pressure,
        "feature_multichunk_policy_pressure": profile["input_multichunk_fraction"] * pressure,
    }
    for threshold in (1, 8, 16, 32, 64):
        features[f"feature_survival_m_gt_{threshold}_policy_pressure"] = (
            profile[f"survival_m_gt_{threshold}"] * pressure
        )
    for width in (4, 16, 32):
        features[f"feature_rolling_multichunk_max_{width}_policy_pressure"] = (
            profile[f"rolling_multichunk_fraction_max_{width}"] * pressure
        )
    return features


def inventory_rows(profiles: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for profile in profiles:
        grouped[(profile["phase27_role"], profile["segment"])].append(profile)
    rows = []
    for (role, segment), group in sorted(grouped.items()):
        counts = [int(row["request_count"]) for row in group]
        multi = [float(row["input_multichunk_fraction"]) for row in group]
        rows.append(
            {
                "phase27_role": role,
                "segment": segment,
                "profiles": len(group),
                "requests_total": sum(counts),
                "requests_min": min(counts),
                "requests_median": statistics.median(counts),
                "requests_max": max(counts),
                "multichunk_fraction_mean": statistics.fmean(multi),
                "multichunk_fraction_max": max(multi),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    for name in ("profiles", "labels", "dataset", "analysis", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    phase27a = json.loads(args.phase27a_summary.read_text())
    if phase27a["label_state_at_freeze"] != "no_phase27_hfull_labels_generated":
        raise RuntimeError("Phase 27A was not frozen before labels")
    selection = read_csv(args.selection)
    if len(selection) != 60:
        raise ValueError(f"expected 60 selected windows, got {len(selection)}")

    raw_manifest_path = args.raw_dir / "source_manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text())
    raw_checks = {}
    for row in raw_manifest["sources"]:
        path = args.raw_dir / row["name"]
        actual_hash = sha256(path)
        raw_checks[row["name"]] = (
            path.stat().st_size == int(row["actual_size"])
            and actual_hash == row["sha256"]
        )
    if not all(raw_checks.values()):
        raise RuntimeError({"raw_source_checks": raw_checks})

    file_by_segment = {
        segment: args.raw_dir / name
        for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()
    }
    arrays = {segment: load_segment(path) for segment, path in file_by_segment.items()}
    profiles = []
    windows = {}
    count_matches = []
    for selected in selection:
        timestamps, inputs, outputs = arrays[selected["segment"]]
        cutoff = int(selected["cutoff_ms"])
        left = int(np.searchsorted(timestamps, cutoff - HISTORY_SECONDS * 1000, side="left"))
        right = int(np.searchsorted(timestamps, cutoff, side="left"))
        profile, requests = summarize_profile(
            selected, timestamps[left:right], inputs[left:right], outputs[left:right]
        )
        count_matches.append(len(requests) == int(selected["history_count"]))
        profiles.append(profile)
        windows[profile["profile_id"]] = requests

    model_meta, model_feature_values = model_features(args.model_features)
    target_rows = []
    baseline_rows = []
    development_examples = []
    confirmation_features = []
    simulation_checks = []
    for profile in profiles:
        full_requests = windows[profile["profile_id"]]
        compact_requests = pseudo_requests(profile)
        for pp_size in PP_SIZES:
            for microbatch in MICROBATCH_SIZES:
                target_histograms, target_audit = exact_histogram(
                    full_requests, pp_size, microbatch
                )
                h0_histograms, h0_audit = exact_histogram(
                    compact_requests, pp_size, microbatch
                )
                simulation_checks.append(
                    {
                        "profile_id": profile["profile_id"],
                        "pp_size": pp_size,
                        "microbatch": microbatch,
                        "target_complete": target_audit["all_requests_complete"],
                        "target_prefill_mass_exact": target_audit["prefill_token_mass"]
                        == sum(row[0] for row in full_requests),
                        "target_decode_mass_exact": target_audit["decode_token_mass"]
                        == sum(row[1] - 1 for row in full_requests),
                        "h0_complete": h0_audit["all_requests_complete"],
                        "h0_prefill_mass_exact": h0_audit["prefill_token_mass"]
                        == sum(row[0] for row in compact_requests),
                        "h0_decode_mass_exact": h0_audit["decode_token_mass"]
                        == sum(row[1] - 1 for row in compact_requests),
                    }
                )
                for phase in PHASES:
                    target = label_row(
                        profile, pp_size, microbatch, phase, target_histograms[phase]
                    )
                    h0 = label_row(
                        {**profile, "request_count": len(compact_requests)},
                        pp_size,
                        microbatch,
                        phase,
                        h0_histograms[phase],
                    )
                    h0["label_id"] = target["label_id"].replace("/hfull/", "/compact32_h0/")
                    h0["label_status"] = "PARAMETER_FREE_LOW_DIMENSIONAL_BASELINE"
                    h0["teacher_kind"] = "compact32_reconstruction_plus_sglang_pp_fcfs_lanes_v1"
                    h0["full_window_requests_audit_only"] = profile["request_count"]
                    target_rows.append(target)
                    baseline_rows.append(h0)
                    identifiers = {
                        "training_id": target["label_id"],
                        "profile_id": profile["profile_id"],
                        "phase27_role": profile["phase27_role"],
                        "source": profile["source"],
                        "segment": profile["segment"],
                        "window_id": profile["window_id"],
                        "model": "qwen3-8b",
                        "parallelism": "pp",
                        "parallel_size": pp_size,
                        "policy": f"mb{microbatch}",
                        "phase": phase,
                    }
                    features = training_features(
                        profile, model_feature_values, pp_size, microbatch, phase
                    )
                    h0_fields = {
                        "h0_total_calls_per_1000": h0["total_calls_per_1000"],
                        "h0_total_logical_bytes_per_1000": h0["total_logical_bytes_per_1000"],
                        "h0_common_reference_cost_us_per_1000": h0[
                            "common_reference_cost_us_per_1000"
                        ],
                        "h0_calls_by_12bin_json": h0["calls_by_12bin_json"],
                        "h0_logical_bytes_by_12bin_json": h0[
                            "logical_bytes_by_12bin_json"
                        ],
                    }
                    if profile["phase27_role"] == "independent_confirmation":
                        confirmation_features.append({**identifiers, **features, **h0_fields})
                    else:
                        development_examples.append(
                            {
                                **identifiers,
                                **features,
                                **h0_fields,
                                "target_total_calls_per_1000": target[
                                    "total_calls_per_1000"
                                ],
                                "target_total_logical_bytes_per_1000": target[
                                    "total_logical_bytes_per_1000"
                                ],
                                "target_common_reference_cost_us_per_1000": target[
                                    "common_reference_cost_us_per_1000"
                                ],
                                "target_calls_by_12bin_json": target[
                                    "calls_by_12bin_json"
                                ],
                                "target_logical_bytes_by_12bin_json": target[
                                    "logical_bytes_by_12bin_json"
                                ],
                            }
                        )

    profile_rows = []
    for profile in profiles:
        profile_rows.append({**profile, **scalar_profile_features(profile)})
    write_csv_gz(args.output_dir / "profiles/low_dimensional_profiles.csv.gz", profile_rows)
    development_targets = [
        row for row in target_rows if row["phase27_role"] != "independent_confirmation"
    ]
    confirmation_targets = [
        row for row in target_rows if row["phase27_role"] == "independent_confirmation"
    ]
    write_csv_gz(args.output_dir / "labels/development_hfull_targets.csv.gz", development_targets)
    write_csv_gz(
        args.output_dir / "labels/independent_confirmation_hfull_targets.csv.gz",
        confirmation_targets,
    )
    write_csv_gz(args.output_dir / "labels/compact32_h0_baselines.csv.gz", baseline_rows)
    write_csv_gz(args.output_dir / "dataset/development_examples.csv.gz", development_examples)
    write_csv_gz(
        args.output_dir / "dataset/independent_confirmation_features.csv.gz",
        confirmation_features,
    )
    write_csv(args.output_dir / "analysis/profile_inventory.csv", inventory_rows(profiles))
    label_inventory = [
        {
            "artifact": "development_hfull_targets",
            "profiles": 42,
            "phase_rows": len(development_targets),
            "contains_target": True,
            "allowed_training_access": True,
        },
        {
            "artifact": "independent_confirmation_hfull_targets",
            "profiles": 18,
            "phase_rows": len(confirmation_targets),
            "contains_target": True,
            "allowed_training_access": False,
        },
        {
            "artifact": "independent_confirmation_features",
            "profiles": 18,
            "phase_rows": len(confirmation_features),
            "contains_target": False,
            "allowed_training_access": True,
        },
    ]
    write_csv(args.output_dir / "analysis/label_inventory.csv", label_inventory)

    feature_columns = [
        name for name in development_examples[0] if name.startswith("feature_")
    ]
    write_json(
        args.output_dir / "feature_columns.json",
        {
            "schema_version": "phase27b-pp-feature-columns-v1",
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "confirmation_target_columns_absent_from_feature_artifact": True,
        },
    )

    all_simulations_exact = all(
        all(value for key, value in row.items() if key.endswith("_exact") or key.endswith("_complete"))
        for row in simulation_checks
    )
    role_counts = Counter(profile["phase27_role"] for profile in profiles)
    summary = {
        "schema_version": "phase27b-pp-hfull-dataset-v1",
        "status": "PASS",
        "profiles": len(profiles),
        "profile_role_counts": dict(role_counts),
        "full_window_requests": sum(profile["request_count"] for profile in profiles),
        "feature_columns": len(feature_columns),
        "target_phase_rows": len(target_rows),
        "development_phase_rows": len(development_targets),
        "independent_confirmation_phase_rows": len(confirmation_targets),
        "baseline_phase_rows": len(baseline_rows),
        "scheduler_simulations": len(simulation_checks) * 2,
        "teacher_status": LABEL_STATUS,
        "teacher_kind": TEACHER_KIND,
        "model": "qwen3-8b",
        "model_config_sha256": model_meta["config_sha256"],
        "inputs": {
            "raw_manifest_sha256": sha256(raw_manifest_path),
            "phase27a_selection_sha256": sha256(args.selection),
            "phase27a_summary_sha256": sha256(args.phase27a_summary),
            "feature_contract_sha256": sha256(args.feature_contract),
            "model_features_sha256": sha256(args.model_features),
        },
        "raw_source_checks": raw_checks,
        "confirmation_outcome_metrics_computed": False,
    }
    write_json(args.output_dir / "summary.json", summary)

    forbidden_request_fields = {
        "input_lens",
        "output_lens",
        "requests",
        "full_request_list",
        "representative_request_list",
    }
    checks = {
        "raw_source_hashes_6_of_6": len(raw_checks) == 6 and all(raw_checks.values()),
        "profiles_60": len(profiles) == 60,
        "role_counts_30_12_18": role_counts
        == Counter(
            {
                "development_train": 30,
                "development_validation": 12,
                "independent_confirmation": 18,
            }
        ),
        "history_counts_match_selection_60_of_60": all(count_matches),
        "target_rows_1080": len(target_rows) == 60 * 3 * 3 * 2,
        "development_rows_756": len(development_examples) == 42 * 3 * 3 * 2,
        "confirmation_rows_324": len(confirmation_features) == 18 * 3 * 3 * 2,
        "confirmation_features_have_no_targets": not any(
            name.startswith("target_") for name in confirmation_features[0]
        ),
        "no_request_lists_in_profiles_or_examples": not (
            forbidden_request_fields
            & (set(profile_rows[0]) | set(development_examples[0]) | set(confirmation_features[0]))
        ),
        "scheduler_mass_and_completion_exact": all_simulations_exact,
        "confirmation_outcome_metrics_not_computed": not summary[
            "confirmation_outcome_metrics_computed"
        ],
    }
    audit = {
        "schema_version": "phase27b-pp-hfull-dataset-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    write_json(args.output_dir / "audit_summary.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(audit)

    readme = f"""# Phase 27B：新窗口 PP Hfull 数据集

本阶段按照 Phase 27A 已提交的事前合同，处理 60 个此前未使用的 300 秒历史窗口：
30 个开发训练、12 个开发验证、18 个独立确认。完整请求列表只在本脚本内存中用于两件事：
聚合低维画像，以及通过 Phase 25B/25C 已经 GPU 验证的 SGLang PP fixed-draining
调度公式生成 Hfull teacher；结果中不保存任何请求列表。

## 规模与隔离

- 完整历史请求：{summary['full_window_requests']:,} 条；
- 低维输入特征：{len(feature_columns)} 列；
- Hfull 目标：{len(target_rows):,} 个 phase rows（PP2/4/8 × MB1/4/16 × prefill/decode）；
- 开发目标：{len(development_targets):,} rows；独立确认目标：{len(confirmation_targets):,} rows；
- `dataset/independent_confirmation_features.csv.gz` 明确不含 target 列；训练阶段不得读取
  `labels/independent_confirmation_hfull_targets.csv.gz`。

## 证据与口径

六个公开 trace 文件的大小和 SHA-256 全部匹配 Phase 15 manifest。共执行
{summary['scheduler_simulations']:,} 次 Hfull/compact32 调度模拟；每次都验证请求完成、prefill
token mass 和 decode token mass 精确守恒。teacher 是每 1000 请求归一化的、单 PP boundary
拓扑无关消息直方图；`pipeline_*` 仅通过 `pp_size-1` 给出链路边界总量审计。

本阶段没有计算独立确认集的 H0 或学习器误差，因此仍不能声称新增特征改善了 PP。下一步应
只读取开发集训练并冻结 checkpoint，随后由独立评测脚本加载确认集真值。
"""
    (args.output_dir / "README.md").write_text(readme)
    (args.output_dir / "DONE").write_text("PASS\n")
    (args.output_dir / "logs/build.log").write_text(
        json.dumps(
            {
                "event": "phase27b_dataset_built",
                "status": "PASS",
                "profiles": len(profiles),
                "target_phase_rows": len(target_rows),
                "confirmation_metrics_computed": False,
            },
            sort_keys=True,
        )
        + "\n"
    )
    files = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(args.output_dir)}\n" for path in files)
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "profiles": len(profiles),
                "requests": summary["full_window_requests"],
                "target_phase_rows": len(target_rows),
                "feature_columns": len(feature_columns),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
