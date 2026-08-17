#!/usr/bin/env python3
"""Phase44 selection, input and protected-raw preflight."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
P41 = HERE.parent / "phase41_pd_full_window_dataset"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(P41))
from build_selection import HISTORY_MS, PRIOR_SELECTIONS, select  # noqa: E402
from common import load_json, repo_root, require_clean_before_run, require_expected_head, verify_pinned_inputs  # noqa: E402
from prepare_bundle import raw_source_audit  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source: return list(csv.DictReader(source))


def selection_audit(contract: dict) -> dict:
    path = repo_root() / contract["selection_contract"]["path"]
    frozen = read_csv(path); reproduced = select(repo_root())
    frozen_ids = [row["window_id"] for row in frozen]; reproduced_ids = [str(row["window_id"]) for row in reproduced]
    roles = Counter((row["segment"], row["role"]) for row in frozen)
    overlaps = []
    for index, left in enumerate(frozen):
        for right in frozen[index + 1:]:
            if left["segment"] == right["segment"] and abs(int(left["cutoff_ms"]) - int(right["cutoff_ms"])) < HISTORY_MS:
                overlaps.append((left["profile_id"], right["profile_id"]))
    prior = {segment: [] for segment in contract["selection_contract"]["segments"]}
    for relative in PRIOR_SELECTIONS:
        for row in read_csv(repo_root() / relative):
            if row["segment"] in prior: prior[row["segment"]].append(int(row["cutoff_ms"]))
    embargo = [row["profile_id"] for row in frozen if any(abs(int(row["cutoff_ms"]) - old) < HISTORY_MS for old in prior[row["segment"]])]
    checks = {
        "reproduced_exactly": frozen_ids == reproduced_ids, "profiles_1200": len(frozen) == 1200,
        "requests_exact": sum(int(row["history_count"]) for row in frozen) == 486242,
        "roles_exact": all(roles[(segment, "expanded_train")] == 320 and roles[(segment, "expanded_validation")] == 80 for segment in prior),
        "pairwise_nonoverlap": not overlaps, "historical_embargo": not embargo,
        "target_free_columns": not any(name.startswith("future_") or name.startswith("target_") for name in frozen[0]),
    }
    if not all(checks.values()): raise RuntimeError({"checks": checks, "overlaps": overlaps[:10], "embargo": embargo[:10]})
    return {"checks": checks, "roles": {f"{key[0]}::{key[1]}": value for key, value in roles.items()}}


def run_checks(expected: str, raw_dir: Path) -> dict:
    contract = load_json(HERE / "experiment.json"); phase41 = load_json(P41 / "experiment.json")
    head = require_expected_head(expected); require_clean_before_run(allowed_untracked_prefixes=())
    pins = verify_pinned_inputs(contract); selection = selection_audit(contract)
    raw = raw_source_audit(phase41, raw_dir.expanduser().resolve())
    return {"status": "PASS", "workflow_commit": head, "pinned_inputs": pins, "selection_audit": selection, "raw_source_audit": raw, "raw_read_only": True, "gpu_used": False, "phase43_target_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True); parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(run_checks(args.expected_workflow_commit, args.raw_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
