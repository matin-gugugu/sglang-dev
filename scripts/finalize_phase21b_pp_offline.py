#!/usr/bin/env python3
"""Finalize the 24-profile pure-PP draining matrix into compact truth labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BIN_COUNT = 12
BIN_EDGES = np.geomspace(4 * 1024, 8 * 1024 * 1024 * 1024, BIN_COUNT + 1)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root
        / "experiment-results/phase21b_pp_offline_profiledemand/qwen3-8b-draining-v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "experiment-results/phase21b_pp_offline_profiledemand/qwen3-8b-labels-v1",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalized_sender(snapshot: dict) -> dict[tuple, int]:
    result: dict[tuple, int] = defaultdict(int)
    for row in snapshot.get("histograms", []):
        workload_id = row.get("workload_id")
        if row.get("msg_type") != "proxy" or not workload_id:
            continue
        key = (
            workload_id,
            row["phase"],
            row.get("raw_op", "send"),
            row.get("tensor_name", "unknown"),
            int(row["payload_bytes"]),
        )
        result[key] += int(row["count"])
    return dict(result)


def bin_vectors(histogram: Counter[int]) -> tuple[list[float], list[float]]:
    calls = np.zeros(BIN_COUNT, dtype=np.float64)
    logical_bytes = np.zeros(BIN_COUNT, dtype=np.float64)
    for payload, count in histogram.items():
        index = int(
            np.clip(
                np.searchsorted(BIN_EDGES, payload, side="right") - 1,
                0,
                BIN_COUNT - 1,
            )
        )
        calls[index] += count
        logical_bytes[index] += payload * count
    return calls.tolist(), logical_bytes.tolist()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (input_dir / "MATRIX_DONE").is_file():
        raise RuntimeError(f"matrix is not complete: {input_dir}")

    labels = []
    cell_audits = []
    for cell in sorted(input_dir.glob("pp*/mb*")):
        if not (cell / "DONE").is_file():
            continue
        config = json.loads((cell / "run_config.json").read_text())
        clients = [
            json.loads(line)
            for line in (cell / "client_results.jsonl").read_text().splitlines()
            if line
        ]
        snapshots = [
            json.loads(path.read_text())
            for path in sorted((cell / "profile").glob("*.json"))
        ]
        pp_size = int(config["pp_size"])
        max_microbatch = int(config["pp_max_micro_batch_size"])
        senders = sorted(
            (
                snapshot
                for snapshot in snapshots
                if int(snapshot["pp_rank"]) < pp_size - 1
            ),
            key=lambda row: int(row["pp_rank"]),
        )
        signatures = [normalized_sender(snapshot) for snapshot in senders]
        if len(senders) != pp_size - 1 or not signatures:
            raise ValueError(f"bad sender set in {cell}")
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(f"sender boundary mismatch in {cell}")
        truth = signatures[0]
        expected_ids = {row["workload_id"] for row in clients}
        labeled_ids = {key[0] for key in truth}
        if expected_ids != labeled_ids:
            raise ValueError(
                f"labeled workload mismatch in {cell}: "
                f"missing={sorted(expected_ids-labeled_ids)} extra={sorted(labeled_ids-expected_ids)}"
            )
        if len(clients) != 24:
            raise ValueError(f"expected 24 profile rows in {cell}, got {len(clients)}")
        if not all(row.get("all_output_lengths_exact") for row in clients):
            raise ValueError(f"non-exact generation length in {cell}")

        for client in clients:
            workload_id = client["workload_id"]
            for phase in ("prefill", "decode"):
                exact = Counter()
                raw = Counter()
                for (wid, row_phase, op, tensor, payload), count in truth.items():
                    if wid == workload_id and row_phase == phase:
                        exact[payload] += count
                        raw[f"{op}:{tensor}:{payload}"] += count
                if not exact:
                    raise ValueError(f"missing {phase} histogram for {workload_id}")
                calls, logical_bytes = bin_vectors(exact)
                normalization = 1000.0 / int(client["request_count"])
                calls_1000 = [value * normalization for value in calls]
                bytes_1000 = [value * normalization for value in logical_bytes]
                profile_id = client["profile_id"]
                labels.append(
                    {
                        "sample_id": (
                            f"qwen3-8b/pp{pp_size}/mb{max_microbatch}/"
                            f"{profile_id}/{phase}"
                        ),
                        "model": "qwen3-8b",
                        "profile_id": profile_id,
                        "phase": phase,
                        "pp_size": pp_size,
                        "pp_max_micro_batch_size": max_microbatch,
                        "request_count": int(client["request_count"]),
                        "per_boundary_calls": int(sum(exact.values())),
                        "per_boundary_logical_bytes": int(
                            sum(payload * count for payload, count in exact.items())
                        ),
                        "pipeline_calls": int(sum(exact.values())) * (pp_size - 1),
                        "pipeline_logical_bytes": int(
                            sum(payload * count for payload, count in exact.items())
                        )
                        * (pp_size - 1),
                        "payload_histogram_json": json.dumps(
                            dict(sorted(exact.items())), separators=(",", ":")
                        ),
                        "op_tensor_payload_histogram_json": json.dumps(
                            dict(sorted(raw.items())), separators=(",", ":")
                        ),
                        "calls_by_12bin_per_1000_json": json.dumps(
                            [value * normalization for value in calls],
                            separators=(",", ":"),
                        ),
                        "logical_bytes_by_12bin_per_1000_json": json.dumps(
                            [value * normalization for value in logical_bytes],
                            separators=(",", ":"),
                        ),
                    }
                )
        cell_audits.append(
            {
                "cell": str(cell.relative_to(input_dir)),
                "pp_size": pp_size,
                "max_microbatch": max_microbatch,
                "profiles": len(clients),
                "sender_boundaries": len(senders),
                "status": "PASS",
            }
        )

    checks = {
        "cells_9": len(cell_audits) == 9,
        "labels_432": len(labels) == 24 * 3 * 3 * 2,
        "unique_sample_ids": len({row["sample_id"] for row in labels}) == len(labels),
        "positive_calls_and_bytes": all(
            int(row["per_boundary_calls"]) > 0
            and int(row["per_boundary_logical_bytes"]) > 0
            for row in labels
        ),
    }
    write_csv(output_dir / "labels.csv", labels)
    summary = {
        "schema_version": "phase21b-pure-pp-offline-labels-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "cells": cell_audits,
        "labels": len(labels),
        "checks": checks,
        "group_level_truth": (
            "first sender boundary; all other forward boundaries are exact equality checks"
        ),
        "pipeline_expansion": "per-boundary calls and bytes multiplied by pp_size - 1",
        "input_matrix_sha256": sha256(input_dir / "matrix_summary.json"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(summary)
    (output_dir / "DONE").write_text("PASS\n")
    files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "manifest.sha256"
    )
    (output_dir / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
