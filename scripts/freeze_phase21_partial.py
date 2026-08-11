#!/usr/bin/env python3
"""Freeze and audit a deliberately stopped Phase-21 PP online matrix.

The script never rewrites profiler/client data.  It validates every completed
workload row, checks sender-boundary equality for the labeled PP proxy traffic,
and writes a compact immutable audit plus hashes next to the partial dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root
        / "experiment-results/phase21_pp_service_profile/qwen3-8b-formal-profiled-v1",
    )
    parser.add_argument("--planned-rows", type=int, default=1080)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


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


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)

    cells = []
    all_rows = []
    all_checks = []
    for client_path in sorted(input_dir.glob("pp*/mb*/client_results.jsonl")):
        cell = client_path.parent
        rows = read_jsonl(client_path)
        all_rows.extend(rows)
        snapshots = [
            json.loads(path.read_text())
            for path in sorted((cell / "profile").glob("*.json"))
        ]
        senders = sorted(
            (
                snapshot
                for snapshot in snapshots
                if int(snapshot["pp_rank"]) < int(snapshot["pp_size"]) - 1
            ),
            key=lambda row: int(row["pp_rank"]),
        )
        workload_ids = {row["workload_id"] for row in rows}
        signatures = [sender_histogram(snapshot) for snapshot in senders]
        labeled_ids = {
            key[0]
            for key in signatures[0]
        } if signatures else set()
        checks = {
            "rows_nonempty": bool(rows),
            "workload_ids_unique": len(workload_ids) == len(rows),
            "request_count_32": all(int(row.get("request_count", 0)) == 32 for row in rows),
            "all_output_lengths_exact": all(
                bool(row.get("all_output_lengths_exact")) for row in rows
            ),
            "sender_count_matches_pp_minus_one": bool(rows)
            and len(senders) == int(rows[0]["pp_size"]) - 1,
            "histogram_only": bool(snapshots)
            and all(
                snapshot.get("capture_mode") == "histogram-only"
                and not snapshot.get("raw_events_saved", False)
                for snapshot in snapshots
            ),
            "all_client_workloads_labeled": workload_ids <= labeled_ids,
            "sender_boundaries_identical": bool(signatures)
            and all(signature == signatures[0] for signature in signatures[1:]),
        }
        all_checks.append(all(checks.values()))
        cells.append(
            {
                "cell": str(cell.relative_to(input_dir)),
                "rows": len(rows),
                "logical_requests": sum(int(row["request_count"]) for row in rows),
                "complete_cell": (cell / "DONE").is_file(),
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )

    if not all_rows:
        raise ValueError(f"no completed client rows under {input_dir}")

    files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.name not in {"partial_manifest.sha256", "partial_audit.json", "PARTIAL_FROZEN"}
    )
    manifest = "".join(
        f"{sha256(path)}  {path.relative_to(input_dir)}\n" for path in files
    )
    (input_dir / "partial_manifest.sha256").write_text(manifest)

    audit = {
        "schema_version": "phase21-pp-online-partial-audit-v1",
        "status": "PASS" if all(all_checks) else "FAIL",
        "reason": (
            "The exhaustive online matrix was intentionally stopped and replaced by "
            "an offline main matrix plus a stratified online residual subset."
        ),
        "planned_rows": args.planned_rows,
        "frozen_rows": len(all_rows),
        "completion_fraction": len(all_rows) / args.planned_rows,
        "logical_request_executions": sum(
            int(row["request_count"]) for row in all_rows
        ),
        "unique_profiles": len({row["profile_id"] for row in all_rows}),
        "pp_sizes": sorted({int(row["pp_size"]) for row in all_rows}),
        "microbatch_sizes": sorted(
            {int(row["pp_max_micro_batch_size"]) for row in all_rows}
        ),
        "cells": cells,
        "interpretation_boundary": (
            "These rows are offline training/calibration evidence for the effect of a "
            "deterministic profile-derived arrival process.  They are not a forecast "
            "of a future request window and incomplete cells are never called a full matrix."
        ),
    }
    (input_dir / "partial_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(json.dumps(audit, indent=2))
    (input_dir / "PARTIAL_FROZEN").write_text("PASS\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
