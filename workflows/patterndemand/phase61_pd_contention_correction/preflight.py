#!/usr/bin/env python3
"""Read-only Phase61 contract and source-data audit."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import load_json, repo_root, require_clean_before_run, require_expected_head, utc_now, verify_pinned_inputs  # noqa: E402
from model import read_points  # noqa: E402


def run_checks(expected: str) -> dict[str, Any]:
    head = require_expected_head(expected)
    require_clean_before_run()
    contract = load_json(HERE / "experiment.json")
    pins = verify_pinned_inputs(contract)
    root = repo_root()
    phase60 = root / "experiment-results/phase60_pd_multi_endpoint_composability"
    summary = load_json(phase60 / "summary.json")
    if summary.get("status") != "PASS" or summary.get("scientific_outcome") != "CONTENTION_CORRECTION_CANDIDATE":
        raise RuntimeError({"unexpected_phase60_summary": summary})
    rows = read_points(phase60 / "analysis/composability_points.csv")
    pair_grid = load_json(phase60 / "contracts/payload_pair_grid.json")
    development_ids = {
        row["pair_id"]
        for values in pair_grid["development"].values()
        for row in values
    }
    reserved_ids = {
        row["pair_id"]
        for values in pair_grid["reserved_future_blind"].values()
        for row in values
    }
    actual_ids = {row["pair_id"] for row in rows}
    counts = {
        "rows": len(rows),
        "models": Counter(row["model_id"] for row in rows),
        "configurations": Counter(row["configuration"] for row in rows),
        "topologies": Counter(row["topology_level"] for row in rows),
        "pairs": Counter(row["pair_id"] for row in rows),
    }
    checks = {
        "row_count": len(rows) == 120,
        "models": counts["models"] == Counter({"qwen3-8b": 60, "deepseek-v2-lite": 60}),
        "configurations": counts["configurations"] == Counter({"P1D2": 60, "P2D1": 60}),
        "topologies": counts["topologies"] == Counter({"L1": 40, "L2": 40, "L3": 40}),
        "development_pair_identity": actual_ids == development_ids and len(actual_ids) == 20,
        "six_rows_per_pair": set(counts["pairs"].values()) == {6},
        "reserved_pair_zero_overlap": not (actual_ids & reserved_ids),
        "positive_physical_values": all(
            float(row[field]) > 0
            for row in rows
            for field in ("phase51_flow0_us", "phase51_flow1_us", "actual_concurrent_wave_us")
        ),
        "official_replica_policy_present": all("cross_replica_relative_spread" in row for row in rows),
    }
    if not all(checks.values()):
        raise RuntimeError({"phase61_preflight_checks": checks, "counts": counts})
    return {
        "schema_version": "phase61-preflight-v1",
        "status": "PASS",
        "workflow_commit": head,
        "captured_at_utc": utc_now(),
        "pinned_inputs": pins,
        "phase60": {
            "workflow_commit": summary["workflow_commit"],
            "result_status": summary["status"],
            "scientific_outcome": summary["scientific_outcome"],
            "rows": len(rows),
            "development_pair_ids": sorted(actual_ids),
            "reserved_pair_ids_read_only_for_overlap_audit": sorted(reserved_ids),
            "reserved_measurements_or_targets_read": False,
        },
        "checks": checks,
        "execution": {
            "gpu_used": False,
            "network_used": False,
            "new_physical_measurement": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(run_checks(args.expected_workflow_commit), ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
