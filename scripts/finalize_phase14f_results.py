#!/usr/bin/env python3
"""Validate and finalize the corrected Phase 14F result directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_OPS = {"all_reduce", "fused_allreduce_residual_rmsnorm"}
EXPECTED_TPS = {2, 4, 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_curve_rows(root: Path):
    rows = []
    files = sorted((root / "curve").glob("tp*/*/r*/curve.jsonl"))
    for path in files:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return files, rows


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    required = [
        root / "README.md",
        root / "DONE",
        root / "environment.json",
        root / "nvidia_topology.txt",
        root / "runner.log",
        root / "support_inventory.json",
        root / "analysis/README.md",
        root / "analysis/summary.json",
        root / "analysis/curve_summary.csv",
        root / "analysis/metrics.csv",
        root / "analysis/predictions.csv",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required Phase 14F files: {missing}")

    summary = read_json(root / "analysis/summary.json")
    files, rows = load_curve_rows(root)
    support = {}
    sample_count = 0
    for row in rows:
        key = (row["op"], int(row["group_size"]), int(row["payload_bytes"]))
        support.setdefault(key, set()).add(int(row["repeat_id"]))
        samples = row.get("post_rendezvous_samples_us", [])
        sample_count += len(samples)

    bad_ops = sorted({key[0] for key in support} - EXPECTED_OPS)
    bad_tps = sorted({key[1] for key in support} - EXPECTED_TPS)
    bad_repeats = {
        f"{op}:{tp}:{payload}": sorted(repeats)
        for (op, tp, payload), repeats in support.items()
        if repeats != set(range(5))
    }
    selected = summary["selected_workload_cv"]
    decode = summary["selected_decode_workload_cv"]
    checks = {
        "curve_files_30": len(files) == 30,
        "curve_records_525": len(rows) == 525,
        "support_points_105": len(support) == 105,
        "curve_samples_52500": sample_count == 52500,
        "only_expected_ops": not bad_ops,
        "only_expected_tps": not bad_tps,
        "five_repeats_per_support": not bad_repeats,
        "all_convergence_gates_passed": summary.get("all_gates_passed") is True,
        "overall_mape_below_10pct": float(selected["mape"]) < 0.10,
        "overall_p95_below_25pct": float(selected["p95_ape"]) < 0.25,
        "decode_mape_below_10pct": float(decode["mape"]) < 0.10,
    }
    audit = {
        "schema_version": "phase14f-post-rendezvous-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_root": str(root),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "time_contract": "max(rank kernel end) - max(rank kernel start)",
        "curve_files": len(files),
        "curve_records": len(rows),
        "support_points": len(support),
        "curve_samples": sample_count,
        "bad_ops": bad_ops,
        "bad_tps": bad_tps,
        "bad_repeats": bad_repeats,
        "selected_workload_cv": selected,
        "selected_decode_workload_cv": decode,
        "selected_lomo": summary["selected_lomo"],
        "checks": checks,
    }
    (root / "audit_summary.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    if audit["status"] != "PASS":
        raise SystemExit(json.dumps(audit, indent=2))

    manifest_path = root / "manifest.sha256"
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    )
    manifest_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(root)}\n" for path in paths)
    )
    print(json.dumps(audit, indent=2))
    print(f"manifest_files={len(paths)}")


if __name__ == "__main__":
    main()
