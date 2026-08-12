#!/usr/bin/env python3
"""Finalize the Phase 25C PP scheduler GPU tail audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=root / "experiment-results/phase25c_pp_scheduler_tail_audit",
    )
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


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, total_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    profiles = sorted({row["profile_id"] for row in total_rows})
    configs = ["PP2/MB1", "PP4/MB4", "PP8/MB16"]
    lookup = {
        (row["profile_id"], row["configuration"]): row for row in total_rows
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    colors = ("#4C78A8", "#F58518", "#E45756")
    for axis, profile in zip(axes, profiles):
        values = [float(lookup[(profile, config)]["total_calls_per_1000"]) for config in configs]
        bars = axis.bar(configs, values, color=colors)
        axis.set_yscale("log")
        axis.set_ylabel("GPU proxy calls per 1,000 requests (log scale)")
        axis.set_title(profile.replace("profile_", "Profile ").replace("_", " "))
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle("Phase 25C measured PP tail demand; Phase 25B teacher exact in every cell")
    figure.savefig(path, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def main() -> None:
    started = time.time()
    args = parse_args()
    if (args.result_dir / "RUN_DONE").read_text().strip() != "PASS":
        raise RuntimeError("tail run is not complete")
    plan = json.loads((args.result_dir / "plan.json").read_text())
    run_summary = json.loads((args.result_dir / "run_summary.json").read_text())
    cell_dirs = sorted(args.result_dir.glob("results/pp*/mb*/r0"))
    if len(cell_dirs) != 3:
        raise RuntimeError(f"expected 3 cells, got {len(cell_dirs)}")

    phase_rows = []
    comparison_rows = []
    for cell in cell_dirs:
        config = json.loads((cell / "run_config.json").read_text())
        pp_size = int(config["pp_size"])
        microbatch = int(config["pp_max_micro_batch_size"])
        for row in read_csv(cell / "gpu_phase_labels.csv"):
            phase_rows.append(
                {
                    "profile_id": row["profile_id"],
                    "source": "mooncake" if "mooncake" in row["profile_id"] else "burstgpt",
                    "pp_size": pp_size,
                    "microbatch": microbatch,
                    "configuration": f"PP{pp_size}/MB{microbatch}",
                    "phase": row["phase"],
                    "requests": int(row["requests"]),
                    "total_calls_per_1000": float(row["total_calls_per_1000"]),
                    "total_logical_bytes_per_1000": float(row["total_logical_bytes_per_1000"]),
                }
            )
        for row in read_csv(cell / "teacher_comparisons.csv"):
            comparison_rows.append(
                {
                    "profile_id": row["profile_id"],
                    "pp_size": pp_size,
                    "microbatch": microbatch,
                    "phase": row["phase"],
                    **{
                        key: value.lower() == "true"
                        for key, value in row.items()
                        if key.startswith("check_")
                    },
                }
            )

    grouped = defaultdict(list)
    for row in phase_rows:
        grouped[(row["profile_id"], row["source"], row["pp_size"], row["microbatch"], row["configuration"], row["requests"])].append(row)
    total_rows = []
    for key, rows in sorted(grouped.items()):
        total_rows.append(
            {
                "profile_id": key[0],
                "source": key[1],
                "pp_size": key[2],
                "microbatch": key[3],
                "configuration": key[4],
                "requests": key[5],
                "total_calls_per_1000": sum(row["total_calls_per_1000"] for row in rows),
                "total_logical_bytes_per_1000": sum(
                    row["total_logical_bytes_per_1000"] for row in rows
                ),
            }
        )
    analysis = args.result_dir / "analysis"
    analysis.mkdir(exist_ok=True)
    write_csv(analysis / "gpu_phase_metrics.csv", phase_rows)
    write_csv(analysis / "gpu_total_metrics.csv", total_rows)
    write_csv(analysis / "teacher_exact_checks.csv", comparison_rows)
    plot(analysis / "gpu_tail_calls.png", total_rows)

    comparison_checks = [
        value
        for row in comparison_rows
        for key, value in row.items()
        if key.startswith("check_")
    ]
    audits = [json.loads((cell / "teacher_audit.json").read_text()) for cell in cell_dirs]
    checks = {
        "run_summary_pass": run_summary["status"] == "PASS",
        "cells_3": len(cell_dirs) == 3,
        "profile_cells_6": len(total_rows) == 6,
        "phase_comparisons_12": len(comparison_rows) == 12,
        "all_gpu_integrity_pass": all(row["checks"]["gpu_integrity"] for row in audits),
        "all_teacher_exact": all(row["checks"]["teacher_exact_match"] for row in audits),
        "all_comparison_fields_pass": bool(comparison_checks) and all(comparison_checks),
        "no_pid_files": not any(args.result_dir.rglob("*.pid")),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase25c-pp-scheduler-tail-audit-v1",
        "status": status,
        "selection": plan["selection_contract"],
        "profiles": plan["profile_metadata"],
        "cells": plan["cells"],
        "profile_cells": len(total_rows),
        "phase_comparisons": len(comparison_rows),
        "gpu_validation": {
            "integrity_pass_cells": sum(row["checks"]["gpu_integrity"] for row in audits),
            "teacher_exact_cells": sum(row["checks"]["teacher_exact_match"] for row in audits),
            "all_scalar_vector_and_exact_histogram_checks": all(comparison_checks),
        },
        "checks": checks,
        "can_conclude": [
            "the Phase 25B scheduler teacher exactly matches both BurstGPT and Mooncake tail sentinels in all three diagonal PP/MB cells",
            "the recovered chunk/lane semantics generalize beyond the original 42-request smoke to long prompts and a 930-request Mooncake window",
        ],
        "cannot_conclude": [
            "three diagonal cells replace a full Cartesian tail matrix",
            "the fixed-draining teacher applies to online arrival-aware scheduling or other server contracts",
        ],
        "next_step": "recompute PP H32/H64/H128/Hfull convergence with the Phase 25B scheduler-faithful formula",
    }
    (args.result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.result_dir / "README.md").write_text(
        f"""# Phase 25C: PP scheduler teacher GPU tail audit

Status: **{status}**. The Phase 25B scheduler-faithful teacher matches all
{summary['gpu_validation']['teacher_exact_cells']}/3 measured GPU cells and all
{summary['phase_comparisons']}/12 profile-phase comparisons exactly.

The audit uses two complete fixed-draining windows: a 48-request BurstGPT window
with a 6,216-token prompt and a 930-request Mooncake conversation window with
8,192-token prompts. The diagonal cells PP2/MB1, PP4/MB4, and PP8/MB16 cover
cross-chunk continuation, different lane counts, and small/large microbatch
limits without running an expensive full Cartesian matrix.

For every cell, GPU execution integrity, sender-boundary identity, total calls,
logical bytes, 12-bin calls/bytes, and the exact payload histogram pass. Compact
GPU histograms and logs are retained; model weights, caches, raw profiler traces,
and PID files are excluded.

This supports promoting the Phase 25B formula across the audited BurstGPT and
Mooncake tails. It does not establish online-arrival semantics or replace all
nine combinations on every tail window. The next step is to recompute
H32/H64/H128/Hfull convergence under the scheduler-faithful PP teacher.
"""
    )
    logs = args.result_dir / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "finalize.log").write_text(
        json.dumps(
            {
                "status": status,
                "argv": sys.argv,
                "python": sys.version,
                "platform": platform.platform(),
                "duration_seconds": time.time() - started,
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if status != "PASS":
        raise RuntimeError(json.dumps(summary, indent=2))
    (args.result_dir / "DONE").write_text("PASS\n")
    files = sorted(
        path
        for path in args.result_dir.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    (args.result_dir / "manifest.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(args.result_dir)}\n" for path in files
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
