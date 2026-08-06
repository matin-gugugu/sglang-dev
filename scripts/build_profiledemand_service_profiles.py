#!/usr/bin/env python3
"""Build fixed steady-state service profiles from BurstGPT and Mooncake traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


INPUT_EDGES = np.asarray([0, 128, 512, 2048, np.inf], dtype=np.float64)
OUTPUT_EDGES = np.asarray([0, 16, 32, 64, np.inf], dtype=np.float64)
HISTORY_SECONDS = 300
INPUT_CAP = 8192
OUTPUT_CAP = 128
REPRESENTATIVE_REQUESTS = 128
SEED = 20260806
QUOTAS = {
    "burstgpt_1": 5,
    "burstgpt_2": 5,
    "burstgpt_3": 5,
    "mooncake_conversation": 4,
    "mooncake_toolagent": 4,
    "mooncake_synthetic": 1,
}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", type=Path, default=root / "data/phase15_traces/raw"
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=root / "experiment-results/phase15_trace_data/windows.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase16_service_profiles",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_cv(values):
    if len(values) < 2:
        return 0.0
    mean = float(np.mean(values))
    return float(np.std(values) / mean) if mean else 0.0


def summarize_requests(timestamps, inputs, outputs):
    clipped_l = np.clip(inputs, 1, INPUT_CAP)
    clipped_m = np.clip(outputs, 1, OUTPUT_CAP)
    joint, _, _ = np.histogram2d(clipped_l, clipped_m, bins=(INPUT_EDGES, OUTPUT_EDGES))
    joint = joint / max(float(np.sum(joint)), 1.0)
    interarrival = np.diff(timestamps) / 1000.0
    if len(timestamps):
        origin = int(timestamps[0])
        seconds = np.clip(((timestamps - origin) // 1000).astype(int), 0, HISTORY_SECONDS - 1)
        per_second = np.bincount(seconds, minlength=HISTORY_SECONDS)
    else:
        per_second = np.zeros(HISTORY_SECONDS, dtype=int)
    mean_per_second = float(np.mean(per_second))
    peak_to_mean = float(np.max(per_second) / mean_per_second) if mean_per_second else 0.0
    fano = float(np.var(per_second) / mean_per_second) if mean_per_second else 0.0
    correlation = (
        float(np.corrcoef(clipped_l, clipped_m)[0, 1])
        if len(clipped_l) > 1 and np.std(clipped_l) and np.std(clipped_m)
        else 0.0
    )
    result = {
        "request_count": int(len(inputs)),
        "rps": float(len(inputs) / HISTORY_SECONDS),
        "interarrival_cv": safe_cv(interarrival),
        "peak_to_mean_1s": peak_to_mean,
        "fano_1s": fano,
        "input_mean_raw": float(np.mean(inputs)),
        "input_p50_raw": float(np.quantile(inputs, 0.50)),
        "input_p90_raw": float(np.quantile(inputs, 0.90)),
        "input_p99_raw": float(np.quantile(inputs, 0.99)),
        "output_mean_raw": float(np.mean(outputs)),
        "output_p50_raw": float(np.quantile(outputs, 0.50)),
        "output_p90_raw": float(np.quantile(outputs, 0.90)),
        "output_p99_raw": float(np.quantile(outputs, 0.99)),
        "input_mean_capped": float(np.mean(clipped_l)),
        "output_mean_capped": float(np.mean(clipped_m)),
        "lm_correlation_capped": correlation,
        "survival_m_gt_8": float(np.mean(clipped_m > 8)),
        "survival_m_gt_16": float(np.mean(clipped_m > 16)),
        "survival_m_gt_32": float(np.mean(clipped_m > 32)),
        "survival_m_gt_64": float(np.mean(clipped_m > 64)),
        "joint_lm_4x4": joint.reshape(-1).tolist(),
    }
    return result


def feature_vector(summary):
    return np.asarray(
        summary["joint_lm_4x4"]
        + [
            math.log1p(summary["rps"]),
            math.log1p(summary["interarrival_cv"]),
            math.log1p(summary["peak_to_mean_1s"]),
            math.log1p(summary["fano_1s"]),
            math.log1p(summary["input_mean_capped"]),
            math.log1p(summary["output_mean_capped"]),
            summary["lm_correlation_capped"],
            summary["survival_m_gt_8"],
            summary["survival_m_gt_16"],
            summary["survival_m_gt_32"],
            summary["survival_m_gt_64"],
        ],
        dtype=np.float64,
    )


def robust_scale(matrix):
    median = np.median(matrix, axis=0)
    scale = np.quantile(matrix, 0.75, axis=0) - np.quantile(matrix, 0.25, axis=0)
    scale[scale < 1e-9] = 1.0
    return (matrix - median) / scale


def assign_clusters(matrix, medoids):
    distances = np.stack(
        [np.sum((matrix - matrix[index]) ** 2, axis=1) for index in medoids], axis=1
    )
    return np.argmin(distances, axis=1), np.min(distances, axis=1)


def choose_medoids(matrix, count):
    """Deterministic farthest initialization followed by centroid-medoid refinement."""
    normalized = robust_scale(matrix)
    first = int(np.argmin(np.sum(normalized**2, axis=1)))
    medoids = [first]
    while len(medoids) < count:
        _, distance = assign_clusters(normalized, medoids)
        distance[medoids] = -1.0
        medoids.append(int(np.argmax(distance)))
    for _ in range(20):
        labels, _ = assign_clusters(normalized, medoids)
        updated = []
        for cluster in range(count):
            members = np.flatnonzero(labels == cluster)
            if not len(members):
                updated.append(medoids[cluster])
                continue
            centroid = np.mean(normalized[members], axis=0)
            updated.append(int(members[np.argmin(np.sum((normalized[members] - centroid) ** 2, axis=1))]))
        if updated == medoids:
            break
        medoids = updated
    labels, distances = assign_clusters(normalized, medoids)
    return medoids, labels, distances


def representative_indices(inputs, outputs, count):
    """Deterministic proportional stratified sample over the fixed 4x4 grid."""
    lbin = np.minimum(np.searchsorted(INPUT_EDGES, np.clip(inputs, 1, INPUT_CAP), side="right") - 1, 3)
    mbin = np.minimum(np.searchsorted(OUTPUT_EDGES, np.clip(outputs, 1, OUTPUT_CAP), side="right") - 1, 3)
    cells = lbin * 4 + mbin
    cell_counts = np.bincount(cells, minlength=16)
    exact = cell_counts * (count / len(inputs))
    allocation = np.floor(exact).astype(int)
    remainder = count - int(np.sum(allocation))
    order = np.argsort(-(exact - allocation), kind="stable")
    for cell in order:
        if remainder <= 0:
            break
        if cell_counts[cell]:
            allocation[cell] += 1
            remainder -= 1
    selected = []
    for cell, take in enumerate(allocation):
        members = np.flatnonzero(cells == cell)
        if take:
            # ``take`` may exceed the original cell count for sparse BurstGPT
            # profiles. Evenly repeated members preserve the empirical joint
            # distribution better than resizing the complete request sequence.
            positions = np.linspace(0, len(members) - 1, take, dtype=int)
            selected.extend(members[positions].tolist())
    return np.asarray(sorted(selected), dtype=int)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_columns = ["window_id", "segment", "source", "split", "cutoff_ms", "history_count"]
    windows = pd.read_csv(args.windows, usecols=window_columns)
    file_by_segment = {
        segment: args.raw_dir / name
        for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()
    }
    candidates_by_segment = defaultdict(list)
    arrays_by_segment = {}
    for segment, quota in QUOTAS.items():
        timestamps, inputs, outputs = load_segment(file_by_segment[segment])
        arrays_by_segment[segment] = (timestamps, inputs, outputs)
        segment_windows = windows[windows["segment"] == segment]
        minimum = 32 if segment.startswith("burstgpt") else 128
        segment_windows = segment_windows[segment_windows["history_count"] >= minimum]
        for row in segment_windows.itertuples(index=False):
            cutoff = int(row.cutoff_ms)
            left = int(np.searchsorted(timestamps, cutoff - HISTORY_SECONDS * 1000, side="left"))
            right = int(np.searchsorted(timestamps, cutoff, side="left"))
            summary = summarize_requests(timestamps[left:right], inputs[left:right], outputs[left:right])
            candidates_by_segment[segment].append(
                {
                    "window_id": row.window_id,
                    "segment": segment,
                    "source": row.source,
                    "split": row.split,
                    "cutoff_ms": cutoff,
                    "left": left,
                    "right": right,
                    **summary,
                }
            )
        if len(candidates_by_segment[segment]) < quota:
            raise RuntimeError(f"{segment}: only {len(candidates_by_segment[segment])} candidates")

    profiles = []
    request_rows = []
    profile_number = 0
    cluster_audit = []
    for segment, quota in QUOTAS.items():
        candidates = candidates_by_segment[segment]
        matrix = np.stack([feature_vector(row) for row in candidates])
        medoids, labels, distances = choose_medoids(matrix, quota)
        for cluster, index in enumerate(medoids):
            row = candidates[index]
            profile_number += 1
            profile_id = f"profile_{profile_number:02d}_{segment}_c{cluster}"
            members = np.flatnonzero(labels == cluster)
            timestamps, inputs, outputs = arrays_by_segment[segment]
            left, right = row["left"], row["right"]
            wt, wl, wm = timestamps[left:right], inputs[left:right], outputs[left:right]
            representative = representative_indices(wl, wm, REPRESENTATIVE_REQUESTS)
            rep_summary = summarize_requests(wt[representative], wl[representative], wm[representative])
            profile = {
                "profile_id": profile_id,
                "source": row["source"],
                "segment": segment,
                "split": row["split"],
                "window_id": row["window_id"],
                "cutoff_ms": row["cutoff_ms"],
                "cluster": cluster,
                "cluster_members": int(len(members)),
                "cluster_weight_within_segment": float(len(members) / len(candidates)),
                "distance_to_medoid_mean": float(np.mean(distances[members])),
                **{key: value for key, value in row.items() if key not in {"left", "right", "window_id", "segment", "source", "split", "cutoff_ms"}},
                "representative_requests": REPRESENTATIVE_REQUESTS,
                "representative_joint_l1": float(
                    np.sum(np.abs(np.asarray(row["joint_lm_4x4"]) - np.asarray(rep_summary["joint_lm_4x4"])))
                ),
            }
            profiles.append(profile)
            first_timestamp = int(wt[representative][0])
            for request_index, original_index in enumerate(representative):
                request_rows.append(
                    {
                        "profile_id": profile_id,
                        "request_index": request_index,
                        "arrival_offset_ms_audit_only": int(wt[original_index] - first_timestamp),
                        "input_len_raw": int(wl[original_index]),
                        "output_len_raw": int(wm[original_index]),
                        "input_len_capped": int(np.clip(wl[original_index], 1, INPUT_CAP)),
                        "output_len_capped": int(np.clip(wm[original_index], 1, OUTPUT_CAP)),
                    }
                )
            cluster_audit.append(
                {
                    "segment": segment,
                    "cluster": cluster,
                    "profile_id": profile_id,
                    "candidate_windows": len(candidates),
                    "cluster_members": len(members),
                    "mean_squared_distance": float(np.mean(distances[members])),
                    "max_squared_distance": float(np.max(distances[members])),
                }
            )

    scalar_fields = [
        key
        for key, value in profiles[0].items()
        if key != "joint_lm_4x4" and not isinstance(value, (list, dict))
    ]
    with (args.output_dir / "service_profiles.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=scalar_fields + ["joint_lm_4x4_json"], lineterminator="\n"
        )
        writer.writeheader()
        for profile in profiles:
            writer.writerow(
                {
                    **{key: profile[key] for key in scalar_fields},
                    "joint_lm_4x4_json": json.dumps(profile["joint_lm_4x4"], separators=(",", ":")),
                }
            )
    with (args.output_dir / "representative_requests.jsonl").open("w") as output:
        for row in request_rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (args.output_dir / "cluster_audit.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(cluster_audit[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(cluster_audit)

    source_manifest = json.loads((args.raw_dir / "source_manifest.json").read_text())
    summary = {
        "schema_version": "profiledemand-service-profiles-v1",
        "seed": SEED,
        "profile_count": len(profiles),
        "quotas": QUOTAS,
        "history_seconds": HISTORY_SECONDS,
        "minimum_candidate_requests": {"burstgpt": 32, "mooncake": 128},
        "joint_input_edges": [0, 128, 512, 2048, "inf"],
        "joint_output_edges": [0, 16, 32, 64, "inf"],
        "gpu_caps": {"input_len": INPUT_CAP, "actual_output_len": OUTPUT_CAP},
        "representative_requests_per_profile": REPRESENTATIVE_REQUESTS,
        "max_representative_joint_l1": max(row["representative_joint_l1"] for row in profiles),
        "source_manifest_sha256": sha256(args.raw_dir / "source_manifest.json"),
        "windows_sha256": sha256(args.windows),
        "raw_source_hashes": {row["name"]: row["sha256"] for row in source_manifest["sources"]},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = f"""# Phase 16B：ProfileDemand 服务常态画像

从固定哈希的 BurstGPT v2.0 与 Mooncake FAST'25 trace 的 300 秒窗口中选择 {len(profiles)}
个 medoid 画像。选择特征包含截断后的 4×4 `P(L,M)`、长度/Decode 生存率、RPS 和突发
摘要；各 trace segment 使用固定 quota，避免大规模 BurstGPT 完全淹没 Mooncake。

- 输入联合分布边界：0/128/512/2048/∞；
- 实际输出联合分布边界：0/16/32/64/∞；
- GPU 回放上限：输入 {INPUT_CAP}、实际输出 {OUTPUT_CAP}；
- 每画像固定 {REPRESENTATIVE_REQUESTS} 个分层代表请求；
- 最大代表样本 joint-distribution L1：{summary['max_representative_joint_l1']:.4f}。

`service_profiles.csv` 是模型输入画像；`representative_requests.jsonl` 是后续
histogram-only GPU 标签的固定请求集合；`cluster_audit.csv` 记录覆盖范围。到达特征当前
作为画像与后续 batching 扩展输入，但首版不得把 draining-batch 回放声称为真实 online
continuous batching。
"""
    (args.output_dir / "README.md").write_text(readme)
    checks = {
        "profiles_24": len(profiles) == 24,
        "representative_rows_3072": len(request_rows) == 24 * REPRESENTATIVE_REQUESTS,
        "joint_probabilities_sum_to_one": all(abs(sum(row["joint_lm_4x4"]) - 1.0) < 1e-9 for row in profiles),
        "representative_joint_l1_below_0_10": summary["max_representative_joint_l1"] < 0.10,
        "all_actual_outputs_positive_and_capped": all(1 <= row["output_len_capped"] <= OUTPUT_CAP for row in request_rows),
    }
    audit = {
        "schema_version": "profiledemand-service-profiles-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    (args.output_dir / "DONE").write_text("PASS\n")
    files = sorted(
        path for path in args.output_dir.iterdir() if path.is_file() and path.name not in {"manifest.sha256", "run.log"}
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    print(json.dumps({"profiles": len(profiles), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
