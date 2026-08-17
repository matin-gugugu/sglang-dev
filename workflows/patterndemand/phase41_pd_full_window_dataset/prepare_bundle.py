#!/usr/bin/env python3
"""Control-side Phase41 raw reconstruction and external transfer-bundle builder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from common import (  # noqa: E402
    load_json,
    repo_root,
    require_clean_before_run,
    require_expected_head,
    sha256,
    utc_now,
    verify_pinned_inputs,
    write_json,
)
from contracts import read_csv, validate_bundle, write_bundle  # noqa: E402
from build_phase27a_pp_feature_and_holdout_contract import (  # noqa: E402
    HISTORY_ONLY_SOURCE_COLUMNS,
    choose_medoids,
    selection_vector,
)
from build_phase27b_pp_hfull_dataset import HISTORY_SECONDS, summarize_profile  # noqa: E402
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment  # noqa: E402


HISTORY_MS = 300_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-workflow-commit", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    return parser.parse_args()


def external_new_directory(path: Path, *, raw_dir: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = repo_root().resolve()
    raw = raw_dir.expanduser().resolve()
    if resolved == root or root in resolved.parents:
        raise RuntimeError("bundle directory must be outside Git")
    if resolved == raw or raw in resolved.parents or resolved in raw.parents:
        raise RuntimeError("bundle directory and protected raw directory must be disjoint")
    if resolved.exists():
        raise RuntimeError(f"bundle directory must not already exist: {resolved}")
    return resolved


def raw_source_audit(contract: dict[str, Any], raw_dir: Path) -> list[dict[str, Any]]:
    if not raw_dir.is_dir():
        raise RuntimeError(f"protected raw directory missing: {raw_dir}")
    manifest = raw_dir / "source_manifest.json"
    if not manifest.is_file() or sha256(manifest) != contract["raw_source_contract"]["manifest_sha256"]:
        raise RuntimeError("protected raw source_manifest hash mismatch")
    rows = []
    for expected in contract["raw_source_contract"]["files"]:
        path = raw_dir / expected["name"]
        actual_bytes = path.stat().st_size if path.is_file() else None
        actual_sha = sha256(path) if path.is_file() and actual_bytes == expected["bytes"] else None
        exact = actual_bytes == expected["bytes"] and actual_sha == expected["sha256"]
        rows.append(
            {
                "name": expected["name"],
                "bytes": actual_bytes,
                "sha256": actual_sha,
                "exact": exact,
            }
        )
    if not all(row["exact"] for row in rows):
        raise RuntimeError({"raw_source_checks": rows})
    return rows


def prior_selection_paths(contract: dict[str, Any]) -> list[Path]:
    names = {f"phase{phase}_selection" for phase in (27, 28, 30, 31, 32, 33, 34)}
    paths = [repo_root() / row["path"] for row in contract["pinned_inputs"] if row["name"] in names]
    if len(paths) != 7:
        raise RuntimeError(f"expected seven prior selection files, got {len(paths)}")
    return paths


def disjoint_pool(frame: pd.DataFrame) -> pd.DataFrame:
    chosen: list[int] = []
    last: int | None = None
    for index, row in frame.sort_values(["cutoff_ms", "window_id"], kind="stable").iterrows():
        cutoff = int(row["cutoff_ms"])
        if last is None or cutoff - last >= HISTORY_MS:
            chosen.append(index)
            last = cutoff
    return frame.loc[chosen].reset_index(drop=True)


def reproduce_blind_selection(contract: dict[str, Any]) -> dict[str, Any]:
    spec = contract["blind_selection_contract"]
    frozen = read_csv(repo_root() / spec["selection_file"])
    segments = tuple(spec["segments"])
    prior = {segment: [] for segment in segments}
    for path in prior_selection_paths(contract):
        for row in read_csv(path):
            if row["segment"] in prior:
                prior[row["segment"]].append(int(row["cutoff_ms"]))

    windows = pd.read_csv(
        repo_root() / "experiment-results/phase15_trace_data/windows.csv.gz",
        usecols=list(HISTORY_ONLY_SOURCE_COLUMNS),
    )
    reproduced = []
    inventory = []
    for segment in segments:
        candidates = windows[
            (windows["segment"] == segment)
            & (windows["history_count"] >= int(spec["minimum_history_count"]))
        ].copy()
        before = len(candidates)
        candidates = candidates[
            [
                not any(abs(int(cutoff) - old) < HISTORY_MS for old in prior[segment])
                for cutoff in candidates["cutoff_ms"]
            ]
        ].copy()
        matrix = np.stack([selection_vector(row) for _, row in candidates.iterrows()])
        median = np.median(matrix, axis=0)
        scale = np.quantile(matrix, 0.75, axis=0) - np.quantile(matrix, 0.25, axis=0)
        scale[scale < 1e-9] = 1.0
        candidates["normality_distance"] = np.sqrt(
            np.mean(((matrix - median) / scale) ** 2, axis=1)
        )
        threshold = float(
            np.quantile(candidates["normality_distance"], float(spec["normality_pool_quantile"]))
        )
        pool = disjoint_pool(candidates[candidates["normality_distance"] <= threshold])
        count = int(spec["profiles_per_segment"])
        pool_matrix = np.stack([selection_vector(row) for _, row in pool.iterrows()])
        medoids, labels, distances = choose_medoids(pool_matrix, count)
        chosen = []
        for cluster, index in enumerate(medoids):
            row = pool.iloc[index]
            members = np.flatnonzero(labels == cluster)
            window_id = str(row["window_id"])
            order = hashlib.sha256(f"{spec['seed']}:{window_id}".encode()).hexdigest()
            chosen.append(
                {
                    "window_id": window_id,
                    "source": str(row["source"]),
                    "segment": segment,
                    "source_split": str(row["split"]),
                    "cutoff_ms": int(row["cutoff_ms"]),
                    "history_seconds": int(row["history_seconds"]),
                    "history_count": int(row["history_count"]),
                    "normality_distance": float(row["normality_distance"]),
                    "normality_pool_quantile": float(spec["normality_pool_quantile"]),
                    "selection_cluster": cluster,
                    "selection_cluster_members": int(len(members)),
                    "selection_distance_to_medoid_mean": float(np.mean(distances[members])),
                    "role_order_sha256": order,
                    "role": "blind_confirmation",
                }
            )
        chosen.sort(key=lambda row: row["role_order_sha256"])
        for local, row in enumerate(chosen, 1):
            row["profile_id"] = f"phase41_{segment}_blind_confirmation_{local:02d}"
            reproduced.append(row)
        inventory.append(
            {
                "segment": segment,
                "eligible_before_embargo": before,
                "eligible_after_embargo": len(candidates),
                "disjoint_p95_pool": len(pool),
                "selected": len(chosen),
            }
        )
    reproduced.sort(key=lambda row: (row["segment"], row["cutoff_ms"]))
    frozen_ids = [row["window_id"] for row in frozen]
    reproduced_ids = [row["window_id"] for row in reproduced]
    pair_overlaps = [
        (left["profile_id"], right["profile_id"])
        for index, left in enumerate(reproduced)
        for right in reproduced[index + 1 :]
        if left["segment"] == right["segment"]
        and abs(int(left["cutoff_ms"]) - int(right["cutoff_ms"])) < HISTORY_MS
    ]
    checks = {
        "frozen_ids_reproduced_exactly": frozen_ids == reproduced_ids,
        "twelve_four_per_segment": len(reproduced) == 12
        and Counter(row["segment"] for row in reproduced)
        == Counter({segment: 4 for segment in segments}),
        "pairwise_disjoint": not pair_overlaps,
        "all_roles_blind": all(row["role"] == "blind_confirmation" for row in reproduced),
        "history_only_source_columns": set(HISTORY_ONLY_SOURCE_COLUMNS).issubset(windows.columns),
    }
    if not all(checks.values()):
        raise RuntimeError({"blind_selection_checks": checks, "pair_overlaps": pair_overlaps})
    return {"checks": checks, "inventory": inventory, "rows": reproduced}


def profile_difference(old: dict[str, str], new: dict[str, Any]) -> dict[str, Any]:
    numeric_differences = []
    structured_matches = []
    identifiers_match = []
    identifier_names = {"profile_id", "split_role", "source", "segment", "source_split", "window_id"}
    for key, value in new.items():
        if key not in old:
            continue
        if key in identifier_names:
            identifiers_match.append(str(value) == str(old[key]))
        elif key.endswith("_json"):
            structured_matches.append(json.loads(str(value)) == json.loads(old[key]))
        else:
            numeric_differences.append(abs(float(value) - float(old[key])))
    return {
        "identifiers_match": all(identifiers_match),
        "structured_fields_match": all(structured_matches),
        "max_absolute_difference": max(numeric_differences, default=0.0),
    }


def reconstruct_profile(
    row: dict[str, Any], arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[dict[str, Any], list[list[int]]]:
    timestamps, inputs, outputs = arrays[row["segment"]]
    cutoff = int(row["cutoff_ms"])
    left = int(np.searchsorted(timestamps, cutoff - HISTORY_SECONDS * 1000, side="left"))
    right = int(np.searchsorted(timestamps, cutoff, side="left"))
    selection = {
        "phase27_profile_id": row["profile_id"],
        "phase27_role": row["split_role"],
        "source": row["source"],
        "segment": row["segment"],
        "source_split": row["source_split"],
        "window_id": row["window_id"],
        "cutoff_ms": cutoff,
    }
    profile, requests = summarize_profile(
        selection, timestamps[left:right], inputs[left:right], outputs[left:right]
    )
    profile["split_role"] = profile.pop("phase27_role")
    return profile, [[int(left_value), int(right_value)] for left_value, right_value in requests]


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_json(HERE / "experiment.json")
    feature_contract = load_json(HERE / "feature_contract.json")
    head = require_expected_head(args.expected_workflow_commit)
    require_clean_before_run()
    pins = verify_pinned_inputs(contract)
    raw_dir = args.raw_dir.expanduser().resolve()
    bundle_dir = external_new_directory(args.bundle_dir, raw_dir=raw_dir)
    raw_audit = raw_source_audit(contract, raw_dir)
    blind_selection = reproduce_blind_selection(contract)

    file_by_segment = {
        segment: raw_dir / name
        for name, (segment, _split) in {**BURST_FILES, **MOONCAKE_FILES}.items()
    }
    saved_development = read_csv(
        repo_root()
        / "experiment-results/phase34b_six_model_hfull_dataset/profiles/low_dimensional_profiles_94.csv.gz"
    )
    blind_rows = [
        {
            **row,
            "split_role": row.pop("role"),
        }
        for row in [dict(item) for item in blind_selection["rows"]]
    ]
    required_segments = sorted(
        {row["segment"] for row in saved_development} | {row["segment"] for row in blind_rows}
    )
    arrays = {segment: load_segment(file_by_segment[segment]) for segment in required_segments}

    development = []
    reconstruction_audit = []
    for saved in saved_development:
        profile, requests = reconstruct_profile(saved, arrays)
        difference = profile_difference(saved, profile)
        difference.update(
            {
                "profile_id": profile["profile_id"],
                "request_count_match": len(requests) == int(saved["request_count"]),
            }
        )
        reconstruction_audit.append(difference)
        development.append({"profile": profile, "requests": requests})
    tolerance = float(contract["acceptance_gates"]["profile_reconstruction_max_absolute_error_lt"])
    if not all(
        row["identifiers_match"]
        and row["structured_fields_match"]
        and row["request_count_match"]
        and float(row["max_absolute_difference"]) < tolerance
        for row in reconstruction_audit
    ):
        raise RuntimeError({"development_reconstruction_failed": reconstruction_audit})

    blind_profiles = []
    for row in blind_rows:
        profile, requests = reconstruct_profile(row, arrays)
        if len(requests) != int(row["history_count"]):
            raise RuntimeError(f"blind request count mismatch: {row['profile_id']}")
        blind_profiles.append(profile)
    bundle = {
        "schema_version": "phase41-external-transfer-bundle-v1",
        "workflow_commit": head,
        "workflow_parent_result_commit": contract["workflow_parent_result_commit"],
        "feature_contract": feature_contract,
        "source_inventory": raw_audit,
        "development": development,
        "blind_features": blind_profiles,
        "blind_targets_generated": False,
        "complete_request_policy": {
            "development": "included for offline teacher only",
            "blind": "not included",
        },
    }
    bundle_audit = validate_bundle(contract, bundle)
    bundle_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = bundle_dir / "phase41_bundle.json.gz"
    write_bundle(bundle_path, bundle)
    manifest = {
        "schema_version": "phase41-external-bundle-manifest-v1",
        "created_at_utc": utc_now(),
        "workflow_commit": head,
        "bundle_file": bundle_path.name,
        "bundle_bytes": bundle_path.stat().st_size,
        "bundle_sha256": sha256(bundle_path),
        "bundle_audit": bundle_audit,
        "blind_selection_audit": {
            "checks": blind_selection["checks"],
            "inventory": blind_selection["inventory"],
        },
        "development_reconstruction": {
            "profiles": len(reconstruction_audit),
            "max_absolute_difference": max(
                float(row["max_absolute_difference"]) for row in reconstruction_audit
            ),
            "all_exact": True,
        },
        "raw_source_audit": raw_audit,
        "pinned_inputs": pins,
        "git_policy": {
            "bundle_committed_to_git": False,
            "raw_committed_to_git": False,
            "blind_complete_requests_exported": False,
        },
    }
    write_json(bundle_dir / "bundle_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    print(json.dumps(build_bundle(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
