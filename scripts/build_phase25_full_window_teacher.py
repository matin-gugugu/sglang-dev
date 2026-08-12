#!/usr/bin/env python3
"""Build provisional full-window fixed-draining TP/PP structural teachers."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_phase21b_pp_h0 import decode_events, histogram, prefill_events
from build_profiledemand_replay_plans import STRATEGIES


TPS = (2, 4, 8)
PP_SIZES = (2, 4, 8)
PP_MICROBATCH_SIZES = (1, 4, 16)
PHASES = ("prefill", "decode")
TP_BIN_EDGES = np.geomspace(4 * 1024, 512 * 1024 * 1024, 13)
PP_BIN_EDGES = np.geomspace(4 * 1024, 8 * 1024 * 1024 * 1024, 13)
PP_MODEL = "qwen3-8b"
PP_CHUNK_TOKENS = 4096
PP_PROXY_COUNT = 2


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase24-requests",
        type=Path,
        default=root
        / "experiment-results/phase24_representative_request_convergence/input_windows/selected_requests.jsonl.gz",
    )
    parser.add_argument(
        "--phase24-labels",
        type=Path,
        default=root
        / "experiment-results/phase24_representative_request_convergence/labels/histogram_labels.jsonl.gz",
    )
    parser.add_argument(
        "--phase24-summary",
        type=Path,
        default=root
        / "experiment-results/phase24_representative_request_convergence/summary.json",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=root / "experiment-results/phase16_service_profiles/service_profiles.csv",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase25_full_window_teacher",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode())


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    deterministic_gzip(
        path,
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def load_profiles(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 24:
        raise ValueError(f"expected 24 profiles, got {len(rows)}")
    return rows


def load_full_windows(path: Path) -> dict[str, dict]:
    rows = [row for row in read_jsonl_gz(path) if row["sample_label"] == "hfull"]
    if len(rows) != 24:
        raise ValueError(f"expected 24 Hfull windows, got {len(rows)}")
    result = {}
    for row in rows:
        inputs = list(map(int, row["input_lens"]))
        outputs = list(map(int, row["output_lens"]))
        if len(inputs) != len(outputs) or len(inputs) != int(row["request_count"]):
            raise ValueError(f"{row['profile_id']}: invalid request arrays")
        result[row["profile_id"]] = {**row, "requests": list(zip(inputs, outputs))}
    return result


def tp_batches(requests: list[tuple[int, int]], strategy: dict) -> list[list[tuple[int, int]]]:
    batches, current, current_tokens = [], [], 0
    for request in requests:
        if current and (
            len(current) >= int(strategy["max_batch_size"])
            or current_tokens + request[0] > int(strategy["max_prefill_tokens"])
        ):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(request)
        current_tokens += request[0]
    if current:
        batches.append(current)
    return batches


def tp_histograms(requests: list[tuple[int, int]], strategy: dict, model: dict) -> dict:
    calls_per_forward = int(model["logical_collectives_per_forward_prior"])
    bytes_per_token = int(model["payload_bytes_per_active_token_prior"])
    result = {phase: Counter() for phase in PHASES}
    for batch in tp_batches(requests, strategy):
        result["prefill"][sum(row[0] for row in batch) * bytes_per_token] += calls_per_forward
        for step in range(1, max(row[1] for row in batch)):
            active = sum(row[1] > step for row in batch)
            if active:
                result["decode"][active * bytes_per_token] += calls_per_forward
    return result


def pp_histograms(requests: list[tuple[int, int]], model: dict, max_microbatch: int) -> dict:
    bytes_per_token = int(model["payload_bytes_per_active_token_prior"])
    return {
        "prefill": histogram(
            prefill_events(requests, max_microbatch, PP_CHUNK_TOKENS),
            bytes_per_token,
            PP_PROXY_COUNT,
        ),
        "decode": histogram(
            decode_events(requests, max_microbatch, PP_CHUNK_TOKENS),
            bytes_per_token,
            PP_PROXY_COUNT,
        ),
    }


def normalize(hist: Counter[int], request_count: int) -> dict[int, float]:
    scale = 1000.0 / request_count
    return {int(payload): float(calls) * scale for payload, calls in sorted(hist.items())}


def bin_vectors(hist: dict[int, float], edges: np.ndarray) -> tuple[list[float], list[float]]:
    calls = np.zeros(12, dtype=np.float64)
    logical_bytes = np.zeros(12, dtype=np.float64)
    for payload, count in hist.items():
        index = int(np.clip(np.searchsorted(edges, payload, side="right") - 1, 0, 11))
        calls[index] += count
        logical_bytes[index] += payload * count
    return calls.tolist(), logical_bytes.tolist()


def label_row(*, model: dict, profile: dict, parallelism: str, parallel_size: int,
              policy: str, phase: str, request_count: int, hist: dict[int, float],
              edges: np.ndarray, boundary_multiplier: int) -> dict:
    calls_bins, bytes_bins = bin_vectors(hist, edges)
    total_calls = float(sum(hist.values()))
    total_bytes = float(sum(payload * calls for payload, calls in hist.items()))
    return {
        "label_id": f"{model['model']}/{parallelism}{parallel_size}/{policy}/{profile['profile_id']}/hfull/{phase}",
        "label_status": "PROVISIONAL_PENDING_GPU_AUDIT",
        "teacher_kind": "full_window_fixed_draining_structural_teacher",
        "model": model["model"],
        "model_config_sha256": model["config_sha256"],
        "profile_id": profile["profile_id"],
        "source": profile["source"],
        "segment": profile["segment"],
        "split": profile["split"],
        "window_id": profile["window_id"],
        "parallelism": parallelism,
        "parallel_size": parallel_size,
        "policy": policy,
        "phase": phase,
        "requests": request_count,
        "normalization_requests": 1000,
        "boundary_multiplier": boundary_multiplier,
        "total_calls_per_1000": total_calls,
        "total_logical_bytes_per_1000": total_bytes,
        "pipeline_calls_per_1000": total_calls * boundary_multiplier,
        "pipeline_logical_bytes_per_1000": total_bytes * boundary_multiplier,
        "calls_by_12bin_json": json.dumps(calls_bins, separators=(",", ":")),
        "logical_bytes_by_12bin_json": json.dumps(bytes_bins, separators=(",", ":")),
        "exact_calls_histogram_per_1000_json": json.dumps(
            {str(payload): calls for payload, calls in hist.items()}, separators=(",", ":")
        ),
        "exact_logical_bytes_histogram_per_1000_json": json.dumps(
            {str(payload): payload * calls for payload, calls in hist.items()}, separators=(",", ":")
        ),
    }


def build_labels(profiles: list[dict], windows: dict[str, dict], models: dict[str, dict]):
    tp_rows, pp_rows, boundary_rows = [], [], []
    for profile in profiles:
        requests = windows[profile["profile_id"]]["requests"]
        for model in models.values():
            for policy, strategy in STRATEGIES.items():
                histograms = tp_histograms(requests, strategy, model)
                for tp in TPS:
                    for phase in PHASES:
                        tp_rows.append(label_row(
                            model=model, profile=profile, parallelism="tp", parallel_size=tp,
                            policy=policy, phase=phase, request_count=len(requests),
                            hist=normalize(histograms[phase], len(requests)), edges=TP_BIN_EDGES,
                            boundary_multiplier=1,
                        ))
        model = models[PP_MODEL]
        for microbatch in PP_MICROBATCH_SIZES:
            histograms = pp_histograms(requests, model, microbatch)
            for pp in PP_SIZES:
                for phase in PHASES:
                    row = label_row(
                        model=model, profile=profile, parallelism="pp", parallel_size=pp,
                        policy=f"mb{microbatch}", phase=phase, request_count=len(requests),
                        hist=normalize(histograms[phase], len(requests)), edges=PP_BIN_EDGES,
                        boundary_multiplier=pp - 1,
                    )
                    pp_rows.append(row)
                    for sender in range(pp - 1):
                        boundary_rows.append({
                            **row,
                            "label_id": f"{row['label_id']}/boundary{sender}-{sender + 1}",
                            "scope": "single_sender_boundary",
                            "sender_stage": sender,
                            "receiver_stage": sender + 1,
                            "boundary_multiplier": 1,
                            "pipeline_calls_per_1000": row["total_calls_per_1000"],
                            "pipeline_logical_bytes_per_1000": row["total_logical_bytes_per_1000"],
                        })
    return tp_rows, pp_rows, boundary_rows


def compare_phase24(tp_rows: list[dict], pp_rows: list[dict], path: Path) -> dict:
    truth = {
        (r["profile_id"], r["parallelism"], int(r["parallel_size"]), r["policy"], r["phase"]): r
        for r in read_jsonl_gz(path) if r["sample_label"] == "hfull"
    }
    candidates = [r for r in tp_rows if r["model"] == PP_MODEL] + pp_rows
    scalar_fields = (
        "total_calls_per_1000", "total_logical_bytes_per_1000",
        "pipeline_calls_per_1000", "pipeline_logical_bytes_per_1000",
    )
    hist_matches = scalar_matches = 0
    failures = []
    for row in candidates:
        key = (row["profile_id"], row["parallelism"], int(row["parallel_size"]), row["policy"], row["phase"])
        reference = truth.get(key)
        hist_ok = reference is not None and json.loads(row["exact_calls_histogram_per_1000_json"]) == json.loads(reference["exact_calls_histogram_per_1000_json"])
        scalar_ok = reference is not None and all(
            math.isclose(float(row[field]), float(reference[field]), rel_tol=0, abs_tol=1e-9)
            for field in scalar_fields
        )
        hist_matches += hist_ok
        scalar_matches += scalar_ok
        if not hist_ok or not scalar_ok:
            failures.append({"key": key, "histogram_match": hist_ok, "scalar_match": scalar_ok})
    count = len(candidates)
    return {
        "expected_comparisons": 24 * 18 * 2,
        "comparisons": count,
        "exact_histogram_matches": hist_matches,
        "scalar_matches": scalar_matches,
        "status": "PASS" if count == 24 * 18 * 2 and hist_matches == count and scalar_matches == count else "FAIL",
        "failures": failures,
    }


def window_statistics(profiles: list[dict], windows: dict[str, dict]) -> list[dict]:
    rows = []
    for profile in profiles:
        requests = windows[profile["profile_id"]]["requests"]
        inputs, outputs = [x[0] for x in requests], [x[1] for x in requests]
        rows.append({
            "profile_id": profile["profile_id"], "source": profile["source"],
            "segment": profile["segment"], "request_count": len(requests),
            "input_mean": statistics.fmean(inputs), "input_p99": float(np.quantile(inputs, .99)),
            "input_max": max(inputs), "output_mean": statistics.fmean(outputs),
            "output_p99": float(np.quantile(outputs, .99)), "output_max": max(outputs),
        })
    return rows


def select_sentinels(stats: list[dict]) -> list[dict]:
    reasons = defaultdict(set)
    by_segment = defaultdict(list)
    for row in stats:
        by_segment[row["segment"]].append(row)
    for segment, rows in sorted(by_segment.items()):
        ordered = sorted(rows, key=lambda row: (row["request_count"], row["profile_id"]))
        reasons[ordered[(len(ordered) - 1) // 2]["profile_id"]].add(f"segment_median:{segment}")
    extrema = {
        "global_min_requests": min(stats, key=lambda r: (r["request_count"], r["profile_id"])),
        "global_max_requests": max(stats, key=lambda r: (r["request_count"], r["profile_id"])),
        "global_max_input_p99": max(stats, key=lambda r: (r["input_p99"], r["profile_id"])),
        "global_max_output_p99": max(stats, key=lambda r: (r["output_p99"], r["profile_id"])),
    }
    for reason, row in extrema.items():
        reasons[row["profile_id"]].add(reason)
    median_count = statistics.median(row["request_count"] for row in stats)
    median_row = min(stats, key=lambda r: (abs(r["request_count"] - median_count), r["profile_id"]))
    reasons[median_row["profile_id"]].add("global_median_requests")
    smoke = extrema["global_min_requests"]["profile_id"]
    multimodel = {smoke, median_row["profile_id"], extrema["global_max_input_p99"]["profile_id"], extrema["global_max_output_p99"]["profile_id"]}
    lookup = {row["profile_id"]: row for row in stats}
    rows = []
    for profile_id in sorted(reasons):
        tiers = ["qwen_formal"]
        if profile_id == smoke:
            tiers.append("smoke")
        if profile_id in multimodel:
            tiers.append("tp_multimodel")
        rows.append({**lookup[profile_id], "tiers": ",".join(tiers), "selection_reasons": ",".join(sorted(reasons[profile_id]))})
    return rows


def tp_plan(sentinels: list[dict], windows: dict[str, dict], tier: str) -> list[dict]:
    selected = {r["profile_id"] for r in sentinels if tier in r["tiers"].split(",")}
    rows = []
    for profile_id in sorted(selected):
        window = windows[profile_id]
        for policy, strategy in STRATEGIES.items():
            for batch_index, batch in enumerate(tp_batches(window["requests"], strategy)):
                rows.append({
                    "workload_id": f"phase25-{tier}-{profile_id}-{policy}-batch{batch_index:04d}-r0",
                    "profile_id": profile_id, "source": window["source"], "segment": window["segment"],
                    "split": window["split"], "strategy": policy,
                    "strategy_max_batch_size": int(strategy["max_batch_size"]),
                    "strategy_max_prefill_tokens": int(strategy["max_prefill_tokens"]),
                    "repeat": 0, "batch_index": batch_index,
                    "profile_requests_replayed": len(window["requests"]),
                    "trace_replay_mode": "phase25_full_window_fixed_draining",
                    "input_lens_per_request": [x[0] for x in batch],
                    "output_lens_per_request": [x[1] for x in batch],
                    "arrival_offsets_ms_audit_only": [0] * len(batch),
                    "chunk_interaction": "fixed_strategy_token_budget",
                })
    return rows


def pp_plan(sentinels: list[dict], windows: dict[str, dict], tier: str) -> list[dict]:
    selected = {r["profile_id"] for r in sentinels if tier in r["tiers"].split(",")}
    rows = []
    for profile_id in sorted(selected):
        window = windows[profile_id]
        rows.append({
            "profile_id": profile_id, "source": window["source"], "segment": window["segment"],
            "window_id": window["window_id"], "request_count": len(window["requests"]),
            "input_lens": [x[0] for x in window["requests"]],
            "output_lens": [x[1] for x in window["requests"]],
            "arrival_offsets_ms": [0] * len(window["requests"]),
            "arrival_mode": "draining_all_at_once",
        })
    return rows


def readme(summary: dict) -> str:
    reg = summary["phase24_regression"]
    return f"""# Phase 25A: full-window fixed-draining structural teacher

Status: **{summary['status']}**. All labels remain provisional until the
full-window GPU sentinel audit passes; they are not GPU ground truth.

## Contract

- Offline teacher input: every capped request length in original order, with
  arrival timestamps removed from scheduling.
- Predictor input remains compact profile + model structure + fixed TP/PP +
  fixed policy + phase.
- Output: exact and 12-bin calls/logical-bytes histograms per 1000 requests.
- PP additionally stores one explicit label per sender boundary.

## Provisional assets

- TP labels: {summary['label_counts']['tp_phase_labels']:,}.
- PP phase labels: {summary['label_counts']['pp_phase_labels']:,}.
- PP boundary labels: {summary['label_counts']['pp_boundary_labels']:,}.
- Full-window requests represented: {summary['inputs']['total_full_requests']:,}.
- Phase 24 Qwen Hfull regression: {reg['exact_histogram_matches']}/{reg['comparisons']}
  exact histograms and {reg['scalar_matches']}/{reg['comparisons']} scalar rows match.

`gpu_audit/sentinel_profiles.csv` records deterministic source/tail coverage;
`gpu_audit/plans/` contains TP trace-replay plans and complete PP draining
request lists. GPU calls, bytes, exact histograms, 12-bin conservation, and PP
sender boundaries must pass before promotion to a formal teacher dataset.

Raw traces, model weights, caches, and PIDs are excluded.
"""


def main() -> None:
    started, args = time.time(), parse_args()
    for directory in ("inputs", "labels", "gpu_audit/plans", "gpu_audit/results", "analysis"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(args.profiles)
    windows = load_full_windows(args.phase24_requests)
    models = {row["model"]: row for row in json.loads(args.model_features.read_text())}
    if set(models) != {"qwen3-8b", "qwen3-30b-a3b", "deepseek-v2-lite"}:
        raise ValueError(f"unexpected models: {sorted(models)}")
    for profile in profiles:
        if int(profile["request_count"]) != len(windows[profile["profile_id"]]["requests"]):
            raise ValueError(f"{profile['profile_id']}: request-count mismatch")
    archived = [{k: v for k, v in windows[p["profile_id"]].items() if k != "requests"} for p in profiles]
    write_jsonl_gz(args.output_dir / "inputs/full_windows.jsonl.gz", archived)
    tp_rows, pp_rows, boundary_rows = build_labels(profiles, windows, models)
    write_csv_gz(args.output_dir / "labels/tp_phase_labels.csv.gz", tp_rows)
    write_csv_gz(args.output_dir / "labels/pp_phase_labels.csv.gz", pp_rows)
    write_csv_gz(args.output_dir / "labels/pp_boundary_labels.csv.gz", boundary_rows)
    regression = compare_phase24(tp_rows, pp_rows, args.phase24_labels)
    stats = window_statistics(profiles, windows)
    sentinels = select_sentinels(stats)
    write_csv(args.output_dir / "analysis/window_statistics.csv", stats)
    write_csv(args.output_dir / "gpu_audit/sentinel_profiles.csv", sentinels)
    tp_plans = {tier: tp_plan(sentinels, windows, tier) for tier in ("smoke", "qwen_formal", "tp_multimodel")}
    pp_plans = {tier: pp_plan(sentinels, windows, tier) for tier in ("smoke", "qwen_formal")}
    for tier, rows in tp_plans.items():
        (args.output_dir / f"gpu_audit/plans/tp_{tier}_plan.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows)
        )
    for tier, rows in pp_plans.items():
        write_jsonl_gz(args.output_dir / f"gpu_audit/plans/pp_{tier}_requests.jsonl.gz", rows)
    contract = {
        "schema_version": "phase25-full-window-teacher-v1",
        "status": "PROVISIONAL_PENDING_GPU_AUDIT",
        "teacher_input": "complete capped requests in original order; fixed-draining without arrival timestamps",
        "predictor_input": "compact profile + model structure + fixed TP/PP + policy + phase",
        "normalization_requests": 1000,
        "tp": {"models": sorted(models), "sizes": TPS, "policies": STRATEGIES, "bin_edges_bytes": TP_BIN_EDGES.tolist()},
        "pp": {"models": [PP_MODEL], "sizes": PP_SIZES, "microbatch_sizes": PP_MICROBATCH_SIZES,
               "chunk_tokens": PP_CHUNK_TOKENS, "proxy_count": PP_PROXY_COUNT, "bin_edges_bytes": PP_BIN_EDGES.tolist()},
        "promotion_gate": "sentinel GPU audit must match calls, bytes, exact histograms, 12-bin conservation, and PP sender boundaries",
        "boundary": "provisional structural labels are not full-window GPU ground truth",
    }
    write_json(args.output_dir / "contract.json", contract)
    plan_counts = {
        "tp": {tier: {"workloads": len(rows), "profiles": len({r['profile_id'] for r in rows})} for tier, rows in tp_plans.items()},
        "pp": {tier: {"profiles": len(rows), "requests": sum(r['request_count'] for r in rows),
                       "cells": len(rows) * len(PP_SIZES) * len(PP_MICROBATCH_SIZES)} for tier, rows in pp_plans.items()},
    }
    checks = {
        "profiles_24": len(profiles) == 24,
        "full_requests_18285": sum(len(w["requests"]) for w in windows.values()) == 18285,
        "tp_labels_1296": len(tp_rows) == 1296,
        "pp_labels_432": len(pp_rows) == 432,
        "pp_boundary_labels_1584": len(boundary_rows) == 1584,
        "phase24_regression_pass": regression["status"] == "PASS",
        "all_totals_positive": all(float(r["total_calls_per_1000"]) > 0 and float(r["total_logical_bytes_per_1000"]) > 0 for r in tp_rows + pp_rows),
        "all_labels_provisional": all(r["label_status"] == "PROVISIONAL_PENDING_GPU_AUDIT" for r in tp_rows + pp_rows + boundary_rows),
        "all_segments_in_sentinel_plan": {r["segment"] for r in sentinels} == {p["segment"] for p in profiles},
        "gpu_results_empty": not any((args.output_dir / "gpu_audit/results").iterdir()),
    }
    status = "PROVISIONAL_READY_FOR_GPU_AUDIT" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase25-full-window-teacher-build-v1", "status": status,
        "inputs": {"profiles": 24, "total_full_requests": 18285,
                   "phase24_requests_sha256": sha256(args.phase24_requests),
                   "phase24_labels_sha256": sha256(args.phase24_labels),
                   "phase24_summary_sha256": sha256(args.phase24_summary),
                   "profiles_sha256": sha256(args.profiles), "model_features_sha256": sha256(args.model_features)},
        "label_counts": {"tp_phase_labels": len(tp_rows), "pp_phase_labels": len(pp_rows), "pp_boundary_labels": len(boundary_rows)},
        "phase24_regression": regression, "sentinel_profiles": sentinels,
        "gpu_audit_plan": plan_counts, "checks": checks,
        "promotion_gate": contract["promotion_gate"], "next_step": "run TP/PP GPU smoke, then formal sentinel audits",
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "README.md").write_text(readme(summary))
    write_json(args.output_dir / "build.log", {"status": status, "argv": sys.argv, "python": sys.version,
               "numpy": np.__version__, "platform": platform.platform(), "duration_seconds": time.time() - started, "checks": checks})
    if status == "FAIL":
        raise RuntimeError(json.dumps(summary, indent=2))
    (args.output_dir / "PROVISIONAL").write_text("READY_FOR_GPU_AUDIT; NOT GPU GROUND TRUTH\n")
    files = sorted(path for path in args.output_dir.rglob("*") if path.is_file() and path.name != "manifest.sha256")
    (args.output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(args.output_dir)}\n" for path in files)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
