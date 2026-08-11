#!/usr/bin/env python3
"""Build a balanced online PP residual label table from frozen and new rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_phase21c_pp_online_subset import PROFILES


BIN_COUNT = 12
BIN_EDGES = np.geomspace(4 * 1024, 8 * 1024 * 1024 * 1024, BIN_COUNT + 1)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=root
        / "experiment-results/phase21_pp_service_profile/qwen3-8b-formal-profiled-v1",
    )
    parser.add_argument(
        "--subset-root",
        type=Path,
        default=root / "experiment-results/phase21c_pp_online_residual/qwen3-8b-v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "experiment-results/phase21c_pp_online_residual/qwen3-8b-labels-v1",
    )
    parser.add_argument("--repeats", type=int, default=2)
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


def sender_histogram(snapshot: dict) -> dict[tuple, int]:
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


def cell_paths(frozen_root: Path, subset_root: Path) -> list[Path]:
    return [
        frozen_root / "pp2/mb1",
        frozen_root / "pp2/mb4",
        frozen_root / "pp2/mb16",
        frozen_root / "pp4/mb1",
        frozen_root / "pp4/mb4",
        subset_root / "pp4-mb16/pp4/mb16",
        subset_root / "pp8-all/pp8/mb1",
        subset_root / "pp8-all/pp8/mb4",
        subset_root / "pp8-all/pp8/mb16",
    ]


def main() -> None:
    args = parse_args()
    frozen_root = args.frozen_root.resolve()
    subset_root = args.subset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (frozen_root / "PARTIAL_FROZEN").is_file():
        raise RuntimeError("frozen online calibration pool has not passed its audit")
    if not (subset_root / "DONE").is_file():
        raise RuntimeError("targeted online subset is incomplete")

    labels = []
    cell_rows = []
    selected_profiles = set(PROFILES)
    for cell in cell_paths(frozen_root, subset_root):
        clients = [
            json.loads(line)
            for line in (cell / "client_results.jsonl").read_text().splitlines()
            if line
        ]
        clients = [
            row
            for row in clients
            if row["profile_id"] in selected_profiles
            and row["arrival_mode"] == "profiled"
            and int(row["repeat"]) < args.repeats
        ]
        if len(clients) != len(PROFILES) * args.repeats:
            raise ValueError(
                f"{cell}: expected {len(PROFILES) * args.repeats} selected rows, "
                f"got {len(clients)}"
            )
        snapshots = [
            json.loads(path.read_text())
            for path in sorted((cell / "profile").glob("*.json"))
        ]
        pp_size = int(clients[0]["pp_size"])
        max_microbatch = int(clients[0]["pp_max_micro_batch_size"])
        senders = sorted(
            (
                snapshot
                for snapshot in snapshots
                if int(snapshot["pp_rank"]) < pp_size - 1
            ),
            key=lambda row: int(row["pp_rank"]),
        )
        signatures = [sender_histogram(snapshot) for snapshot in senders]
        if len(senders) != pp_size - 1 or not signatures:
            raise ValueError(f"invalid PP senders in {cell}")
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(f"sender boundary mismatch in {cell}")
        truth = signatures[0]

        for client in sorted(
            clients, key=lambda row: (row["profile_id"], int(row["repeat"]))
        ):
            if not client.get("all_output_lengths_exact"):
                raise ValueError(f"non-exact output length: {client['workload_id']}")
            for phase in ("prefill", "decode"):
                exact = Counter()
                raw = Counter()
                for (wid, row_phase, op, tensor, payload), count in truth.items():
                    if wid == client["workload_id"] and row_phase == phase:
                        exact[payload] += count
                        raw[f"{op}:{tensor}:{payload}"] += count
                if not exact:
                    raise ValueError(
                        f"missing {phase} truth for {client['workload_id']}"
                    )
                calls, logical_bytes = bin_vectors(exact)
                normalization = 1000.0 / int(client["request_count"])
                profile_id = client["profile_id"]
                repeat = int(client["repeat"])
                h0_sample_id = (
                    f"qwen3-8b/pp{pp_size}/mb{max_microbatch}/{profile_id}/{phase}"
                )
                labels.append(
                    {
                        "sample_id": f"{h0_sample_id}/online-r{repeat}",
                        "h0_sample_id": h0_sample_id,
                        "model": "qwen3-8b",
                        "profile_id": profile_id,
                        "phase": phase,
                        "pp_size": pp_size,
                        "pp_max_micro_batch_size": max_microbatch,
                        "repeat": repeat,
                        "request_count": int(client["request_count"]),
                        "planned_arrival_span_ms": float(
                            client["planned_arrival_span_ms"]
                        ),
                        "actual_arrival_span_ms": float(
                            client["actual_arrival_span_ms"]
                        ),
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
        cell_rows.append(
            {
                "cell": str(cell),
                "pp_size": pp_size,
                "max_microbatch": max_microbatch,
                "selected_windows": len(clients),
                "phase_labels": len(clients) * 2,
                "status": "PASS",
            }
        )

    expected = len(PROFILES) * 3 * 3 * args.repeats * 2
    checks = {
        "cells_9": len(cell_rows) == 9,
        "balanced_phase_labels": len(labels) == expected,
        "unique_sample_ids": len({row["sample_id"] for row in labels}) == len(labels),
        "all_profiles_present": {row["profile_id"] for row in labels}
        == selected_profiles,
    }
    write_csv(output_dir / "labels.csv", labels)
    write_csv(output_dir / "cell_summary.csv", cell_rows)
    summary = {
        "schema_version": "phase21c-pure-pp-online-labels-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "profiles": list(PROFILES),
        "repeats": args.repeats,
        "window_rows": len(labels) // 2,
        "phase_labels": len(labels),
        "cells": cell_rows,
        "checks": checks,
        "interpretation_boundary": (
            "These labels describe a deterministic profile-derived online arrival "
            "realization. They calibrate canonical H0; they do not forecast a future window."
        ),
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
