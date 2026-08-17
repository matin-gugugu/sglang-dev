#!/usr/bin/env python3
"""Phase48 pinned-input, selection and protected-raw preflight."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
P41 = HERE.parent / "phase41_pd_full_window_dataset"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(P41))
from common import load_json, repo_root, require_clean_before_run, require_expected_head, sha256, verify_pinned_inputs  # noqa: E402
from prepare_bundle import raw_source_audit  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("phase48_contracts", HERE / "contracts.py")
if _SPEC is None or _SPEC.loader is None: raise RuntimeError("cannot load Phase48 contracts")
_P48 = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_P48)


def read_selection(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source: return list(csv.DictReader(source))


def selection_audit(contract: dict) -> dict:
    spec = contract["dataset_contract"]; path = repo_root() / spec["selection_path"]; rows = read_selection(path)
    roles = Counter(row["role"] for row in rows); segments = Counter(row["segment"] for row in rows)
    by_segment = {segment: sorted(int(row["cutoff_ms"]) for row in rows if row["segment"] == segment) for segment in segments}
    checks = {
        "selection_sha": sha256(path) == spec["selection_sha256"],
        "profiles_1200": len(rows) == 1200,
        "requests_486242": sum(int(row["history_count"]) for row in rows) == 486242,
        "roles_exact": roles == {"expanded_train": 960, "expanded_validation": 240},
        "segments_exact": segments == {"burstgpt_1": 400, "burstgpt_2": 400, "burstgpt_3": 400},
        "profile_ids_unique": len({row["profile_id"] for row in rows}) == 1200,
        "window_ids_unique": len({row["window_id"] for row in rows}) == 1200,
        "pairwise_300s_nonoverlap": all(right - left >= 300000 for values in by_segment.values() for left, right in zip(values, values[1:])),
        "target_free_selection": not any(name.startswith("target_") or name.startswith("future_") for name in rows[0]),
    }
    if not all(checks.values()): raise RuntimeError({"selection_checks": checks})
    return {"checks": checks, "roles": dict(roles), "segments": dict(segments)}


def run_checks(expected: str, raw_dir: Path) -> dict:
    contract = load_json(HERE / "experiment.json"); phase41 = load_json(P41 / "experiment.json")
    head = require_expected_head(expected); require_clean_before_run(allowed_untracked_prefixes=("data/",))
    return {
        "status": "PASS", "workflow_commit": head,
        "pinned_inputs": verify_pinned_inputs(contract),
        "selection_audit": selection_audit(contract),
        "teacher_contract_self_check": _P48.contract_self_check(),
        "raw_source_audit": raw_source_audit(phase41, raw_dir.expanduser().resolve()),
        "raw_read_only": True, "gpu_used": False, "network_used": False,
        "phase45_or_phase46_target_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-workflow-commit", required=True); parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(run_checks(args.expected_workflow_commit, args.raw_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
