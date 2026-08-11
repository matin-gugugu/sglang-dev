#!/usr/bin/env python3
"""Continue PP offline finalization/training and the online residual subset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline-root",
        type=Path,
        default=root
        / "experiment-results/phase21b_pp_offline_profiledemand/qwen3-8b-draining-v1",
    )
    parser.add_argument("--offline-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-wait-hours", type=float, default=24)
    return parser.parse_args()


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def run(command: list[str], cwd: Path, log) -> None:
    log.write("COMMAND " + json.dumps(command) + "\n")
    log.flush()
    subprocess.run(
        command,
        cwd=cwd,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    offline_root = args.offline_root.resolve()
    pipeline_root = repo_root / "experiment-results/phase21b_pp_offline_profiledemand"
    pipeline_root.mkdir(parents=True, exist_ok=True)
    log_path = pipeline_root / "pipeline.log"
    with log_path.open("a") as log:
        started = time.monotonic()
        while not (offline_root / "MATRIX_DONE").is_file():
            if time.monotonic() - started > args.max_wait_hours * 3600:
                raise TimeoutError("offline PP matrix did not finish before deadline")
            if not process_exists(args.offline_pid):
                raise RuntimeError(
                    f"offline runner {args.offline_pid} exited without MATRIX_DONE"
                )
            log.write("WAIT offline matrix\n")
            log.flush()
            time.sleep(args.poll_seconds)

        run(
            [
                sys.executable,
                "scripts/finalize_phase21b_pp_offline.py",
                "--input-dir",
                str(offline_root),
                "--output-dir",
                str(pipeline_root / "qwen3-8b-labels-v1"),
            ],
            repo_root,
            log,
        )
        run(
            [
                sys.executable,
                "scripts/build_phase21b_pp_h0.py",
                "--output-dir",
                str(pipeline_root / "h0-v1"),
            ],
            repo_root,
            log,
        )
        run(
            [
                sys.executable,
                "scripts/train_phase21b_pp_predictor.py",
                "--labels",
                str(pipeline_root / "qwen3-8b-labels-v1/labels.csv"),
                "--h0",
                str(pipeline_root / "h0-v1/h0_samples.csv"),
            ],
            repo_root,
            log,
        )
        run(
            [sys.executable, "scripts/run_phase21c_pp_online_subset.py"],
            repo_root,
            log,
        )
        (pipeline_root / "PIPELINE_DONE").write_text("PASS\n")
        log.write("PASS\n")
        log.flush()


if __name__ == "__main__":
    main()
