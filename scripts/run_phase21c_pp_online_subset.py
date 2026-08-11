#!/usr/bin/env python3
"""Run the stratified online PP residual subset after the offline matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROFILES = (
    "profile_02_burstgpt_1_c1",  # high RPS, bursty, short
    "profile_08_burstgpt_2_c2",  # low RPS, long input/output
    "profile_10_burstgpt_2_c4",  # steady, short
    "profile_16_mooncake_conversation_c0",  # long input/output, bursty
    "profile_20_mooncake_toolagent_c0",  # high RPS, very bursty
    "profile_24_mooncake_synthetic_c0",  # long and comparatively steady
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "experiment-results/phase21c_pp_online_residual/qwen3-8b-v1",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    return parser.parse_args()


def run(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as output:
        output.write("COMMAND " + json.dumps(command) + "\n")
        output.flush()
        subprocess.run(
            command,
            cwd=cwd,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    script = repo_root / "scripts/run_qwen3_8b_pp_profile_matrix.py"
    common = [
        sys.executable,
        str(script),
        "--profiles",
        *PROFILES,
        "--repeats",
        str(args.repeats),
        "--arrival-modes",
        "profiled",
        "--startup-timeout",
        str(args.startup_timeout),
    ]
    # Existing frozen data already cover PP2 and PP4/mb1.  Only fill the
    # structurally missing PP4/mb16 and all PP8 strategies.
    run(
        common
        + [
            "--output-root",
            str(output_root / "pp4-mb16"),
            "--pp-sizes",
            "4",
            "--microbatch-sizes",
            "16",
            "--port-base",
            "31700",
        ],
        repo_root,
        output_root / "pp4-mb16.log",
    )
    run(
        common
        + [
            "--output-root",
            str(output_root / "pp8-all"),
            "--pp-sizes",
            "8",
            "--microbatch-sizes",
            "1",
            "4",
            "16",
            "--port-base",
            "31800",
        ],
        repo_root,
        output_root / "pp8-all.log",
    )
    summary = {
        "schema_version": "phase21c-pp-online-subset-v1",
        "status": "PASS",
        "profiles": list(PROFILES),
        "repeats": args.repeats,
        "new_profile_windows": len(PROFILES) * args.repeats * 4,
        "cells": ["pp4/mb16", "pp8/mb1", "pp8/mb4", "pp8/mb16"],
        "reuse": (
            "Frozen Phase-21 rows supply PP2 and PP4/mb1 plus available PP4/mb4 "
            "calibration; this subset only fills missing structural cells."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_root / "DONE").write_text("PASS\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
