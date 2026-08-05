#!/usr/bin/env python3
"""Build the Phase 14C three-model, Decode-expanded TP timing dataset."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


MODELS = ("qwen3-8b", "qwen3-30b-a3b", "deepseek-v2-lite")
TPS = (2, 4, 8)
BASE_PROFILES = {"balanced", "staircase", "bimodal"}
EXTRA_PROFILES = {"uniform_b4", "uniform_b16", "long_tail"}
CHUNK_INPUTS = {
    "c1024": {1023, 1024, 1025},
    "c4096": {4095, 4096, 4097},
}
CHUNK_BATCHES = {1, 4}
TARGET_FIELD = "post_rendezvous_completion_kernel_time_us"


def parse_args():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase14-root",
        type=Path,
        default=repo / "experiment-results/phase14/tp_group_size_timing_ground_truth",
    )
    parser.add_argument(
        "--phase14c-root",
        type=Path,
        default=repo / "experiment-results/phase14c",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "experiment-results/phase14c/extended_dataset_analysis",
    )
    return parser.parse_args()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def op_histogram(pattern):
    entries = pattern.get("calls_by_raw_op_and_input_payload_bytes")
    if entries is None:
        return {
            ("all_reduce", int(payload)): int(count)
            for payload, count in pattern["calls_by_input_payload_bytes"].items()
        }
    result = {
        (entry["raw_op"], int(entry["input_payload_bytes"])): int(entry["count"])
        for entry in entries
    }
    if len(result) != len(entries):
        raise ValueError("duplicate raw-op/payload entries")
    if any(entry["collective_family"] != "all_reduce" for entry in entries):
        raise ValueError("Phase 14C only supports the AllReduce family")
    return result


def selected(record, mode, label, suite):
    workload = record["workload"]
    if suite == "extra_decode":
        return mode == "mixed_same_coarse" and label in EXTRA_PROFILES
    if mode == "mixed_same_coarse":
        return label in BASE_PROFILES
    return (
        label in CHUNK_INPUTS
        and int(workload["input_len"]) in CHUNK_INPUTS[label]
        and int(workload["batch_size"]) in CHUNK_BATCHES
    )


def workload_key(model, tp, mode, label, record):
    workload = record["workload"]
    return (
        model,
        tp,
        mode,
        label,
        int(workload["batch_size"]),
        int(workload["input_len"]),
        int(workload["output_len"]),
        tuple(int(value) for value in workload["output_lens_per_request"]),
        int(workload["prefill_chunk_size"]),
    )


def workload_id(key):
    model, _, mode, label, batch, input_len, output_len, _, chunk = key
    if mode == "mixed_same_coarse":
        return f"{model}-mixed-{label}"
    return f"{model}-chunk-c{chunk}-b{batch}-l{input_len}-m{output_len}"


def roots(args):
    repo = Path(__file__).resolve().parents[1]
    tp2 = {
        "qwen3-8b": repo / "experiment-results/phase11/multiscale_timing_ground_truth/qwen3-8b",
        "qwen3-30b-a3b": repo / "experiment-results/phase13/multiscale_timing_ground_truth/qwen3-30b-a3b",
        "deepseek-v2-lite": repo / "experiment-results/phase11/multiscale_timing_ground_truth/deepseek-v2-lite",
    }
    output = []
    for model in MODELS:
        output.append((2, model, "baseline", tp2[model]))
        for tp in (4, 8):
            root = (
                args.phase14c_root / "deepseek_tp_extension" / f"tp{tp}" / model
                if model == "deepseek-v2-lite"
                else args.phase14_root / f"tp{tp}" / model
            )
            output.append((tp, model, "baseline", root))
    for tp in TPS:
        for model in MODELS:
            output.append(
                (
                    tp,
                    model,
                    "extra_decode",
                    args.phase14c_root / "decode_extension" / f"tp{tp}" / model,
                )
            )
    return output


def aggregate(args):
    grouped = defaultdict(list)
    manifest = []
    selected_raw_rows = 0
    for tp, model, suite, root in roots(args):
        paths = sorted(root.glob("*/*/r*/all_rank_ground_truth.jsonl"))
        if not paths:
            raise ValueError(f"no all-rank labels under {root}")
        for path in paths:
            mode, label, repeat_dir, _ = path.relative_to(root).parts
            records = [record for record in read_jsonl(path) if selected(record, mode, label, suite)]
            if not records:
                continue
            manifest.append(
                {
                    "suite": suite,
                    "model": model,
                    "tp": tp,
                    "mode": mode,
                    "case_label": label,
                    "repeat_dir": repeat_dir,
                    "selected_rows": len(records),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "path": str(path),
                }
            )
            for record in records:
                selected_raw_rows += 1
                record["_source"] = str(path)
                grouped[workload_key(model, tp, mode, label, record)].append(record)

    rows = []
    for key, repeats in sorted(grouped.items(), key=lambda item: str(item[0])):
        if len(repeats) != 3 or sorted(int(row["repeat_id"]) for row in repeats) != [0, 1, 2]:
            raise ValueError(f"{key}: expected repeat ids 0,1,2 exactly once")
        model, tp, mode, label, batch, input_len, output_len, output_lens, chunk = key
        patterns = [row["full_phase_pattern_demand"] for row in repeats]
        if any(pattern != patterns[0] for pattern in patterns[1:]):
            raise ValueError(f"{key}: PatternDemand changed across repeats")
        for record in repeats:
            if int(record["full_phase_pattern_demand"]["group_size"]) != tp:
                raise ValueError(f"{record['_source']}: group size mismatch")
            alignment = record["alignment"]
            required = (
                "exact_count_on_every_rank",
                "identical_backend_sequence",
                "identical_profiled_pattern_demand_on_every_rank",
                "identical_full_phase_pattern_demand_on_every_rank",
            )
            if not all(alignment[field] for field in required):
                raise ValueError(f"{record['_source']}: all-rank alignment failed")
            estimate = record["all_rank_ground_truth"]["full_phase_estimate"]
            if not math.isclose(float(estimate["profiled_to_full_call_scale"]), 1.0, abs_tol=1e-12):
                raise ValueError(f"{record['_source']}: non-unit phase scale")

        pattern = patterns[0]
        by_op_payload = op_histogram(pattern)
        by_payload = defaultdict(int)
        for (_, payload), count in by_op_payload.items():
            by_payload[payload] += count
        serialized = {
            int(payload): int(count)
            for payload, count in pattern["calls_by_input_payload_bytes"].items()
        }
        if dict(sorted(by_payload.items())) != dict(sorted(serialized.items())):
            raise ValueError(f"{key}: raw-op marginal mismatch")
        calls = sum(by_payload.values())
        logical_bytes = sum(payload * count for payload, count in by_payload.items())
        if calls != int(pattern["all_reduce_calls"]) or logical_bytes != int(pattern["input_payload_bytes"]):
            raise ValueError(f"{key}: PatternDemand total mismatch")

        estimates = [row["all_rank_ground_truth"]["full_phase_estimate"] for row in repeats]
        targets = [float(estimate[TARGET_FIELD]) for estimate in estimates]
        signatures = {
            row["all_rank_ground_truth"]["backend_sequence_signature"] for row in repeats
        }
        if len(signatures) != 1:
            raise ValueError(f"{key}: backend changed across repeats")
        target = float(statistics.median(targets))
        rows.append(
            {
                "workload_id": workload_id(key),
                "model": model,
                "tp": tp,
                "phase": "decode" if mode == "mixed_same_coarse" else "prefill",
                "mode": mode,
                "case_label": label,
                "batch_size": batch,
                "input_len": input_len,
                "output_len": output_len,
                "output_lens_json": json.dumps(output_lens, separators=(",", ":")),
                "prefill_chunk_size": chunk,
                "repeat_count": 3,
                "calls": calls,
                "logical_payload_bytes": logical_bytes,
                "calls_by_payload_json": json.dumps(
                    {str(payload): count for payload, count in sorted(by_payload.items())},
                    separators=(",", ":"),
                ),
                "calls_by_op_payload_json": json.dumps(
                    {
                        f"{op}:{payload}": count
                        for (op, payload), count in sorted(by_op_payload.items())
                    },
                    separators=(",", ":"),
                ),
                "backend_signature": next(iter(signatures)),
                "target_post_us": target,
                "post_repeat_iqr_fraction": (
                    percentile(targets, 75) - percentile(targets, 25)
                ) / target,
            }
        )

    expected_rows = len(MODELS) * len(TPS) * 18
    expected_raw = expected_rows * 3
    if len(rows) != expected_rows or selected_raw_rows != expected_raw:
        raise ValueError(
            f"expected {expected_raw} raw/{expected_rows} aggregate rows, "
            f"got {selected_raw_rows}/{len(rows)}"
        )
    groups = defaultdict(list)
    for row in rows:
        groups[row["workload_id"]].append(row)
    if len(groups) != expected_rows // len(TPS):
        raise ValueError(f"expected 54 TP-linked workloads, got {len(groups)}")
    for identifier, variants in groups.items():
        if {row["tp"] for row in variants} != set(TPS):
            raise ValueError(f"{identifier}: missing TP variant")
        structural = {
            (
                row["calls"],
                row["logical_payload_bytes"],
                row["calls_by_payload_json"],
                row["calls_by_op_payload_json"],
            )
            for row in variants
        }
        if len(structural) != 1:
            raise ValueError(f"{identifier}: logical PatternDemand changed across TP")
    return rows, manifest


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest = aggregate(args)
    write_csv(args.output_dir / "aggregated_configurations.csv", rows)
    write_csv(args.output_dir / "source_manifest.csv", manifest)
    iqr = [float(row["post_repeat_iqr_fraction"]) for row in rows]
    summary = {
        "schema_version": "phase14c-extended-dataset-v1",
        "models": list(MODELS),
        "tensor_parallel_sizes": list(TPS),
        "workload_groups": len({row["workload_id"] for row in rows}),
        "aggregated_configurations": len(rows),
        "raw_label_rows": len(rows) * 3,
        "phase_counts": {
            phase: sum(row["phase"] == phase for row in rows)
            for phase in ("prefill", "decode")
        },
        "repeat_stability": {
            "median_iqr_fraction": float(np.median(iqr)),
            "p95_iqr_fraction": percentile(iqr, 95),
            "max_iqr_fraction": max(iqr),
        },
        "integrity": {
            "three_repeats_per_configuration": True,
            "all_rank_alignment": True,
            "full_phase_scale_one": True,
            "logical_pattern_invariant_across_tp": True,
            "source_files_hashed": len(manifest),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Phase 14C extended timing dataset\n\n"
        "This dataset adds DeepSeek-V2-Lite TP4/8 labels and three Decode "
        "profiles (`uniform_b4`, `uniform_b16`, and `long_tail`) for all "
        "three models at TP2/4/8. It contains 54 TP-linked workload groups, "
        "162 aggregate configurations, and 486 raw repeat labels.\n"
    )
    print(
        f"built {len(rows)} aggregate configurations from {len(manifest)} "
        f"validated source files"
    )


if __name__ == "__main__":
    main()
