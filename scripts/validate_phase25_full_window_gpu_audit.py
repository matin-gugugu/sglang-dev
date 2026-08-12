#!/usr/bin/env python3
"""Compare Phase 25 full-window GPU sentinels with provisional teachers."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FLOAT_FIELDS = (
    "total_calls_per_1000",
    "total_logical_bytes_per_1000",
)
VECTOR_FIELDS = (
    "calls_by_12bin_json",
    "logical_bytes_by_12bin_json",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallelism", choices=("tp", "pp"), required=True)
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--parallel-size", type=int, required=True)
    parser.add_argument("--policy", help="Required for one PP cell, for example mb4.")
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=root / "experiment-results/phase25_full_window_teacher",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-7)


def json_vector_close(left: str, right: str) -> bool:
    a, b = json.loads(left), json.loads(right)
    return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))


def json_hist_close(left: str, right: str) -> bool:
    a = {int(key): float(value) for key, value in json.loads(left).items()}
    b = {int(key): float(value) for key, value in json.loads(right).items()}
    return set(a) == set(b) and all(close(a[key], b[key]) for key in a)


def bin_vectors(histogram: Counter[int], edges: list[float]) -> tuple[list[float], list[float]]:
    calls = [0.0] * 12
    logical_bytes = [0.0] * 12
    for payload, count in histogram.items():
        index = 11
        for current in range(12):
            if edges[current] <= payload < edges[current + 1]:
                index = current
                break
        calls[index] += float(count)
        logical_bytes[index] += float(payload * count)
    return calls, logical_bytes


def geomspace(start: float, stop: float, count: int) -> list[float]:
    ratio = (stop / start) ** (1.0 / (count - 1))
    return [start * ratio**index for index in range(count)]


def load_teacher(args: argparse.Namespace) -> list[dict[str, str]]:
    name = "tp_phase_labels.csv.gz" if args.parallelism == "tp" else "pp_phase_labels.csv.gz"
    rows = read_csv_gz(args.teacher_root / "labels" / name)
    return [
        row
        for row in rows
        if row["model"] == args.model
        and int(row["parallel_size"]) == args.parallel_size
        and (args.parallelism == "tp" or row["policy"] == args.policy)
    ]


def load_tp_gpu(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    source = args.gpu_dir / "phase_labels.csv"
    summary = json.loads((args.gpu_dir / "summary.json").read_text())
    rows = []
    for row in read_csv(source):
        rows.append(
            {
                "model": row["model"],
                "parallel_size": int(row["tp"]),
                "profile_id": row["profile_id"],
                "policy": row["strategy"],
                "phase": row["phase"],
                "requests": int(row["requests"]),
                "total_calls_per_1000": float(row["total_calls_per_1000"]),
                "total_logical_bytes_per_1000": float(row["total_logical_bytes_per_1000"]),
                "calls_by_12bin_json": row["calls_by_12bin_json"],
                "logical_bytes_by_12bin_json": row["logical_bytes_by_12bin_json"],
                "exact_calls_histogram_per_1000_json": row[
                    "canonical_exact_histogram_per_1000_json"
                ],
            }
        )
    return rows, {key: bool(value) for key, value in summary["checks"].items()}


def normalized_sender(snapshot: dict[str, Any]) -> dict[tuple[str, str, int], int]:
    result: dict[tuple[str, str, int], int] = defaultdict(int)
    for row in snapshot.get("histograms", []):
        workload_id = row.get("workload_id")
        if row.get("msg_type") != "proxy" or not workload_id:
            continue
        result[(workload_id, row["phase"], int(row["payload_bytes"]))] += int(row["count"])
    return dict(result)


def load_pp_gpu(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    if not args.policy or not args.policy.startswith("mb"):
        raise ValueError("--policy mbN is required for PP")
    config = json.loads((args.gpu_dir / "run_config.json").read_text())
    clients = [
        json.loads(line)
        for line in (args.gpu_dir / "client_results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    snapshots = [
        json.loads(path.read_text())
        for path in sorted((args.gpu_dir / "profile").glob("*.json"))
    ]
    senders = sorted(
        (row for row in snapshots if int(row["pp_rank"]) < args.parallel_size - 1),
        key=lambda row: int(row["pp_rank"]),
    )
    signatures = [normalized_sender(snapshot) for snapshot in senders]
    representative = signatures[0] if signatures else {}
    rows = []
    edges = geomspace(4 * 1024, 8 * 1024 * 1024 * 1024, 13)
    for client in clients:
        profile_id = client["workload"]
        workload_id = client["workload_id"]
        request_count = len(client["input_lens"])
        for phase in ("prefill", "decode"):
            exact = Counter(
                {
                    payload: count
                    for (wid, current_phase, payload), count in representative.items()
                    if wid == workload_id and current_phase == phase
                }
            )
            if not exact:
                raise ValueError(f"missing PP GPU histogram: {workload_id}/{phase}")
            scale = 1000.0 / request_count
            normalized = Counter({payload: count * scale for payload, count in exact.items()})
            calls, logical_bytes = bin_vectors(normalized, edges)
            rows.append(
                {
                    "model": "qwen3-8b",
                    "parallel_size": args.parallel_size,
                    "profile_id": profile_id,
                    "policy": args.policy,
                    "phase": phase,
                    "requests": request_count,
                    "total_calls_per_1000": sum(normalized.values()),
                    "total_logical_bytes_per_1000": sum(
                        payload * count for payload, count in normalized.items()
                    ),
                    "calls_by_12bin_json": json.dumps(calls, separators=(",", ":")),
                    "logical_bytes_by_12bin_json": json.dumps(
                        logical_bytes, separators=(",", ":")
                    ),
                    "exact_calls_histogram_per_1000_json": json.dumps(
                        {str(payload): count for payload, count in sorted(normalized.items())},
                        separators=(",", ":"),
                    ),
                }
            )
    integrity = json.loads((args.gpu_dir / "audit.json").read_text())
    checks = {key: bool(value) for key, value in integrity["checks"].items()}
    checks["forward_boundaries_identical"] = bool(signatures) and all(
        signature == signatures[0] for signature in signatures[1:]
    )
    checks["sender_boundary_count"] = len(senders) == args.parallel_size - 1
    checks["config_identity"] = (
        int(config["pp_size"]) == args.parallel_size
        and int(config["pp_max_micro_batch_size"]) == int(args.policy[2:])
    )
    return rows, checks


def main() -> None:
    args = parse_args()
    args.gpu_dir = args.gpu_dir.resolve()
    output = args.output or args.gpu_dir / "teacher_audit.json"
    gpu_rows, integrity_checks = (
        load_tp_gpu(args) if args.parallelism == "tp" else load_pp_gpu(args)
    )
    teacher_rows = load_teacher(args)
    teacher = {
        (row["profile_id"], row["policy"], row["phase"]): row for row in teacher_rows
    }
    comparisons = []
    for gpu in gpu_rows:
        key = (gpu["profile_id"], gpu["policy"], gpu["phase"])
        target = teacher.get(key)
        if target is None:
            raise ValueError(f"no teacher label for {key}")
        checks = {
            "request_count": int(gpu["requests"]) == int(target["requests"]),
            "total_calls": close(
                gpu["total_calls_per_1000"], target["total_calls_per_1000"]
            ),
            "logical_bytes": close(
                gpu["total_logical_bytes_per_1000"],
                target["total_logical_bytes_per_1000"],
            ),
            "calls_12bin": json_vector_close(
                gpu["calls_by_12bin_json"], target["calls_by_12bin_json"]
            ),
            "logical_bytes_12bin": json_vector_close(
                gpu["logical_bytes_by_12bin_json"],
                target["logical_bytes_by_12bin_json"],
            ),
            "exact_histogram": json_hist_close(
                gpu["exact_calls_histogram_per_1000_json"],
                target["exact_calls_histogram_per_1000_json"],
            ),
        }
        comparisons.append(
            {
                "profile_id": key[0],
                "policy": key[1],
                "phase": key[2],
                **{f"check_{name}": value for name, value in checks.items()},
            }
        )
    expected_keys = {(row["profile_id"], row["policy"], row["phase"]) for row in gpu_rows}
    comparison_pass = bool(comparisons) and all(
        all(value for name, value in row.items() if name.startswith("check_"))
        for row in comparisons
    )
    checks = {
        "gpu_integrity": all(integrity_checks.values()),
        "unique_gpu_keys": len(expected_keys) == len(gpu_rows),
        "teacher_comparisons_nonempty": bool(comparisons),
        "teacher_exact_match": comparison_pass,
    }
    summary = {
        "schema_version": "phase25-full-window-gpu-teacher-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "parallelism": args.parallelism,
        "model": args.model,
        "parallel_size": args.parallel_size,
        "policy": args.policy,
        "profiles": sorted({row["profile_id"] for row in gpu_rows}),
        "phase_comparisons": len(comparisons),
        "integrity_checks": integrity_checks,
        "checks": checks,
        "teacher_labels_sha256": sha256(
            args.teacher_root
            / "labels"
            / ("tp_phase_labels.csv.gz" if args.parallelism == "tp" else "pp_phase_labels.csv.gz")
        ),
    }
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(args.gpu_dir / "teacher_comparisons.csv", comparisons)
    write_csv(args.gpu_dir / "gpu_phase_labels.csv", gpu_rows)
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(summary, indent=2))
    (args.gpu_dir / "TEACHER_AUDIT_DONE").write_text("PASS\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
