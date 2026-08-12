#!/usr/bin/env python3
"""Build structured TP event supervision while keeping confirmations isolated."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_phase21b_pp_h0 import pseudo_requests
from build_phase25_full_window_teacher import PHASES, normalize, tp_batches, tp_histograms
from build_phase27b_pp_hfull_dataset import (
    HISTORY_SECONDS,
    scalar_profile_features,
    summarize_profile,
)
from build_phase29b_tp_hfull_dataset import (
    MODELS,
    STRATEGIES,
    TEACHER_KIND,
    TEACHER_STATUS,
    TP_SIZES,
    all_model_features,
    bin_vectors,
    feature_values,
)
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment


QWEN_MODEL = "qwen3-8b"
QWEN_BYTES_PER_TOKEN = 8192
QWEN_CALLS_PER_FORWARD = 73
FIRST_ROLE = "independent_confirmation"
SECOND_ROLE = "second_independent_confirmation"
DEVELOPMENT_ROLES = ("development_train", "development_validation")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    phase29b = root / "experiment-results/phase29b_tp_hfull_dataset"
    phase30a = root / "experiment-results/phase30a_tp_structured_event_contract"
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--phase30-selection",
        type=Path,
        default=phase30a / "selection/selected_windows.csv",
    )
    parser.add_argument(
        "--event-contract", type=Path, default=phase30a / "event_contract.json"
    )
    parser.add_argument(
        "--modeling-contract",
        type=Path,
        default=phase30a / "modeling_contract.json",
    )
    parser.add_argument(
        "--phase30a-summary", type=Path, default=phase30a / "summary.json"
    )
    parser.add_argument(
        "--phase29-feature-contract",
        type=Path,
        default=root
        / "experiment-results/phase29a_tp_aligned_contract/feature_contract.json",
    )
    parser.add_argument(
        "--phase29-development-examples",
        type=Path,
        default=phase29b / "dataset/development_examples.csv.gz",
    )
    parser.add_argument(
        "--phase29-development-targets",
        type=Path,
        default=phase29b / "labels/development_hfull_targets.csv.gz",
    )
    parser.add_argument(
        "--phase29-h0-baselines",
        type=Path,
        default=phase29b / "labels/compact32_h0_baselines.csv.gz",
    )
    parser.add_argument(
        "--phase29-profiles",
        type=Path,
        default=phase29b / "profiles/low_dimensional_profiles.csv.gz",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--phase26a-summary",
        type=Path,
        default=root
        / "experiment-results/phase26a_tp_hfull_teacher_audit/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase30b_tp_structured_event_dataset",
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


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def verify_raw(raw_dir: Path, official_manifest: Path) -> dict[str, bool]:
    raw_manifest = json.loads((raw_dir / "source_manifest.json").read_text())
    official = json.loads(official_manifest.read_text())
    expected = {row["name"]: row for row in official["sources"]}
    checks = {}
    for row in raw_manifest["sources"]:
        path = raw_dir / row["name"]
        reference = expected[row["name"]]
        checks[row["name"]] = (
            path.stat().st_size == int(reference["actual_size"])
            and sha256(path) == reference["sha256"]
        )
    return checks


def event_names(contract: dict) -> list[str]:
    groups = contract["target_groups"]
    return [
        *groups["prefill_batch_counts"],
        *groups["prefill_input_token_mass"],
        *groups["decode_active_lane_step_counts"],
    ]


def category_index(tokens: int, categories: list[dict]) -> int:
    for row in categories:
        if int(row["token_sum_min_inclusive"]) <= tokens <= int(
            row["token_sum_max_inclusive"]
        ):
            return int(row["category"])
    raise ValueError(f"token sum outside event contract: {tokens}")


def events_from_requests(
    requests: list[tuple[int, int]], strategy: dict, contract: dict
) -> tuple[dict[str, float], bool]:
    names = event_names(contract)
    events = {name: 0.0 for name in names}
    categories = contract["prefill_joint_categories"]
    scale = 1000.0 / len(requests)
    batches = tp_batches(requests, strategy)
    for batch in batches:
        tokens = sum(row[0] for row in batch)
        index = category_index(tokens, categories)
        events[f"prefill_joint_category_{index}_batch_count_per_1000"] += scale
        events[
            f"prefill_joint_category_{index}_input_token_mass_per_1000"
        ] += tokens * scale
        for step in range(1, max(row[1] for row in batch)):
            active = sum(row[1] > step for row in batch)
            if active:
                events[
                    f"decode_active_lanes_{active}_step_count_per_1000"
                ] += scale
    partition = sum(len(batch) for batch in batches) == len(requests)
    return events, partition


def events_from_qwen_exact_labels(
    phase_labels: dict[str, dict[str, str]], contract: dict
) -> dict[str, float]:
    events = {name: 0.0 for name in event_names(contract)}
    prefill = phase_labels["prefill"]
    prefill_hist = json.loads(prefill["exact_calls_histogram_per_1000_json"])
    for payload_text, calls in prefill_hist.items():
        payload = int(payload_text)
        if payload % QWEN_BYTES_PER_TOKEN:
            raise ValueError("Qwen prefill payload is not token aligned")
        tokens = payload // QWEN_BYTES_PER_TOKEN
        index = category_index(tokens, contract["prefill_joint_categories"])
        batches = float(calls) / QWEN_CALLS_PER_FORWARD
        events[f"prefill_joint_category_{index}_batch_count_per_1000"] += batches
        events[
            f"prefill_joint_category_{index}_input_token_mass_per_1000"
        ] += tokens * batches
    decode = phase_labels["decode"]
    decode_hist = json.loads(decode["exact_calls_histogram_per_1000_json"])
    for payload_text, calls in decode_hist.items():
        payload = int(payload_text)
        if payload % QWEN_BYTES_PER_TOKEN:
            raise ValueError("Qwen decode payload is not token aligned")
        lanes = payload // QWEN_BYTES_PER_TOKEN
        if not 1 <= lanes <= 16:
            raise ValueError(f"invalid active lanes {lanes}")
        events[f"decode_active_lanes_{lanes}_step_count_per_1000"] += (
            float(calls) / QWEN_CALLS_PER_FORWARD
        )
    return events


def reconstruct_message_vectors(
    events: dict[str, float], model: dict, contract: dict
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    calls_per_forward = int(model["logical_collectives_per_forward_prior"])
    bytes_per_token = int(model["payload_bytes_per_active_token_prior"])
    byte_key = f"tp_bin_for_{bytes_per_token}_bytes_per_token"
    result = {
        phase: (np.zeros(12, dtype=np.float64), np.zeros(12, dtype=np.float64))
        for phase in PHASES
    }
    prefill_calls, prefill_bytes = result["prefill"]
    for category in contract["prefill_joint_categories"]:
        index = int(category["category"])
        target_bin = int(category[byte_key])
        count = events[f"prefill_joint_category_{index}_batch_count_per_1000"]
        token_mass = events[
            f"prefill_joint_category_{index}_input_token_mass_per_1000"
        ]
        prefill_calls[target_bin] += count * calls_per_forward
        prefill_bytes[target_bin] += (
            token_mass * bytes_per_token * calls_per_forward
        )
    decode_calls, decode_bytes = result["decode"]
    for lanes in range(1, 17):
        count = events[f"decode_active_lanes_{lanes}_step_count_per_1000"]
        calls, logical_bytes = bin_vectors(
            {lanes * bytes_per_token: count * calls_per_forward}
        )
        decode_calls += calls
        decode_bytes += logical_bytes
    return result


def max_vector_error(
    predicted: dict[str, tuple[np.ndarray, np.ndarray]],
    actual: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, float, float]:
    calls = max(
        float(np.max(np.abs(predicted[phase][0] - actual[phase][0])))
        for phase in PHASES
    )
    logical_bytes = max(
        float(np.max(np.abs(predicted[phase][1] - actual[phase][1])))
        for phase in PHASES
    )
    calls_relative = max(
        float(
            np.max(
                np.abs(predicted[phase][0] - actual[phase][0])
                / np.maximum(np.abs(actual[phase][0]), 1.0)
            )
        )
        for phase in PHASES
    )
    bytes_relative = max(
        float(
            np.max(
                np.abs(predicted[phase][1] - actual[phase][1])
                / np.maximum(np.abs(actual[phase][1]), 1.0)
            )
        )
        for phase in PHASES
    )
    return calls, logical_bytes, calls_relative, bytes_relative


def target_vectors_from_labels(
    labels: dict[str, dict[str, str]]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        phase: (
            np.asarray(json.loads(labels[phase]["calls_by_12bin_json"])),
            np.asarray(json.loads(labels[phase]["logical_bytes_by_12bin_json"])),
        )
        for phase in PHASES
    }


def target_vectors_from_requests(
    requests: list[tuple[int, int]], strategy: dict, model: dict
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    histograms = tp_histograms(requests, strategy, model)
    result = {}
    for phase in PHASES:
        result[phase] = bin_vectors(normalize(histograms[phase], len(requests)))
    return result


def feature_columns(phase29_contract: dict) -> list[str]:
    removed = {
        "feature_parallelism_tp",
        "feature_parallel_size_log2",
        "feature_phase_prefill",
        "feature_phase_decode",
    }
    return [
        name
        for name in phase29_contract["enhanced_feature_columns"]
        if not name.startswith("feature_model_") and name not in removed
    ]


def identifiers(profile: dict, policy: str, provenance: str) -> dict:
    return {
        "event_id": f"{profile['profile_id']}/{policy}/hfull_events",
        "profile_id": profile["profile_id"],
        "role": profile["phase27_role"],
        "source": profile["source"],
        "segment": profile["segment"],
        "window_id": profile["window_id"],
        "policy": policy,
        "normalization_requests": 1000,
        "event_teacher_provenance": provenance,
    }


def prefix_events(prefix: str, events: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{name}": value for name, value in events.items()}


def main() -> None:
    args = parse_args()
    for directory in ("profiles", "labels", "dataset", "analysis", "logs"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    phase30a = json.loads(args.phase30a_summary.read_text())
    event_contract = json.loads(args.event_contract.read_text())
    modeling_contract = json.loads(args.modeling_contract.read_text())
    phase29_contract = json.loads(args.phase29_feature_contract.read_text())
    phase26a = json.loads(args.phase26a_summary.read_text())
    if phase30a["status"] != "PASS" or phase30a[
        "target_state_at_freeze"
    ] != "no_phase30_event_or_hfull_targets_generated":
        raise ValueError("Phase 30A is not a clean PASS contract")
    if phase26a["promoted_label_status"] != TEACHER_STATUS:
        raise ValueError("Phase 26A TP teacher is not promoted")
    raw_checks = verify_raw(
        args.raw_dir, root / "experiment-results/phase15_trace_data/source_manifest.json"
    )
    if len(raw_checks) != 6 or not all(raw_checks.values()):
        raise RuntimeError(raw_checks)

    models = all_model_features(args.model_features)
    qwen_model, qwen_model_values = models[QWEN_MODEL]
    features = feature_columns(phase29_contract)
    if len(features) != 91 or set(features) != set(
        modeling_contract["feature_views"][
            "phase29_enhanced_profile_and_policy"
        ]
    ) - {
        name
        for name in modeling_contract["feature_views"][
            "phase29_enhanced_profile_and_policy"
        ]
        if name.startswith("feature_model_")
        or name
        in {
            "feature_parallelism_tp",
            "feature_parallel_size_log2",
            "feature_phase_prefill",
            "feature_phase_decode",
        }
    }:
        raise ValueError("unexpected event-predictor feature view")
    event_columns = event_names(event_contract)
    if len(event_columns) != 62:
        raise ValueError("event target count mismatch")

    development = []
    first_features = []
    first_targets = []
    second_features = []
    development_targets = []
    h0_rows = []
    adapter_errors: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)

    phase29_examples = read_csv_gz(args.phase29_development_examples)
    phase29_targets = read_csv_gz(args.phase29_development_targets)
    phase29_h0 = read_csv_gz(args.phase29_h0_baselines)
    old_feature_lookup = {}
    for row in phase29_examples:
        if (
            row["model"] == QWEN_MODEL
            and row["parallel_size"] == "2"
            and row["phase"] == "prefill"
        ):
            old_feature_lookup[(row["profile_id"], row["policy"])] = row
    old_target_lookup = {
        (row["profile_id"], row["model"], row["parallel_size"], row["policy"], row["phase"]): row
        for row in phase29_targets
    }
    old_h0_lookup = {
        (row["profile_id"], row["model"], row["parallel_size"], row["policy"], row["phase"]): row
        for row in phase29_h0
        if row["role"] in DEVELOPMENT_ROLES
    }
    old_profiles = {
        row["profile_id"]: row
        for row in read_csv_gz(args.phase29_profiles)
        if row["phase27_role"] in DEVELOPMENT_ROLES
    }
    for profile_id, profile in sorted(old_profiles.items()):
        for policy in STRATEGIES:
            qwen_target_phases = {
                phase: old_target_lookup[
                    (profile_id, QWEN_MODEL, "2", policy, phase)
                ]
                for phase in PHASES
            }
            qwen_h0_phases = {
                phase: old_h0_lookup[(profile_id, QWEN_MODEL, "2", policy, phase)]
                for phase in PHASES
            }
            target_events = events_from_qwen_exact_labels(
                qwen_target_phases, event_contract
            )
            h0_events = events_from_qwen_exact_labels(qwen_h0_phases, event_contract)
            source = old_feature_lookup[(profile_id, policy)]
            base = {
                **identifiers(
                    profile,
                    policy,
                    "phase29_gpu_validated_exact_histogram_event_inversion",
                ),
                **{name: source[name] for name in features},
                **prefix_events("h0_event_", h0_events),
            }
            target_fields = prefix_events("target_event_", target_events)
            development.append({**base, **target_fields})
            development_targets.append({**identifiers(profile, policy, base["event_teacher_provenance"]), **target_fields})
            h0_rows.append({**identifiers(profile, policy, base["event_teacher_provenance"]), **prefix_events("h0_event_", h0_events)})
            for model_name in MODELS:
                model, _ = models[model_name]
                for tp_size in TP_SIZES:
                    target_labels = {
                        phase: old_target_lookup[
                            (profile_id, model_name, str(tp_size), policy, phase)
                        ]
                        for phase in PHASES
                    }
                    h0_labels = {
                        phase: old_h0_lookup[
                            (profile_id, model_name, str(tp_size), policy, phase)
                        ]
                        for phase in PHASES
                    }
                    adapter_errors[("phase29_development", "hfull")].append(
                        max_vector_error(
                            reconstruct_message_vectors(
                                target_events, model, event_contract
                            ),
                            target_vectors_from_labels(target_labels),
                        )
                    )
                    adapter_errors[("phase29_development", "compact32_h0")].append(
                        max_vector_error(
                            reconstruct_message_vectors(h0_events, model, event_contract),
                            target_vectors_from_labels(h0_labels),
                        )
                    )

    selection = read_csv(args.phase30_selection)
    file_by_segment = {
        segment: args.raw_dir / name
        for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()
    }
    raw_arrays = {
        segment: load_segment(path) for segment, path in file_by_segment.items()
    }
    new_profiles = []
    new_requests = {}
    count_checks = []
    for selected in selection:
        timestamps, inputs, outputs = raw_arrays[selected["segment"]]
        cutoff = int(selected["cutoff_ms"])
        left = int(
            np.searchsorted(
                timestamps, cutoff - HISTORY_SECONDS * 1000, side="left"
            )
        )
        right = int(np.searchsorted(timestamps, cutoff, side="left"))
        compatibility = {
            **selected,
            "phase27_profile_id": selected["profile_id"],
            "phase27_role": selected["role"],
        }
        profile, requests = summarize_profile(
            compatibility,
            timestamps[left:right],
            inputs[left:right],
            outputs[left:right],
        )
        count_checks.append(len(requests) == int(selected["history_count"]))
        new_profiles.append(profile)
        new_requests[profile["profile_id"]] = requests

    full_partition_checks = []
    h0_partition_checks = []
    second_full_targets_generated = False
    for profile in new_profiles:
        requests = new_requests[profile["profile_id"]]
        compact = pseudo_requests(profile)
        role = profile["phase27_role"]
        for policy, strategy in STRATEGIES.items():
            h0_events, h0_partition = events_from_requests(
                compact, strategy, event_contract
            )
            h0_partition_checks.append(h0_partition)
            values = feature_values(
                profile,
                qwen_model_values,
                2,
                policy,
                "prefill",
                phase29_contract["legacy_feature_columns"],
            )
            base = {
                **identifiers(
                    profile, policy, "phase30_raw_full_window_structured_event_teacher"
                ),
                **{name: values[name] for name in features},
                **prefix_events("h0_event_", h0_events),
            }
            h0_rows.append(
                {
                    **identifiers(
                        profile,
                        policy,
                        "phase30_raw_full_window_structured_event_teacher",
                    ),
                    **prefix_events("h0_event_", h0_events),
                }
            )
            for model_name in MODELS:
                model, _ = models[model_name]
                for tp_size in TP_SIZES:
                    actual_h0 = target_vectors_from_requests(compact, strategy, model)
                    adapter_errors[("phase30_new", "compact32_h0")].append(
                        max_vector_error(
                            reconstruct_message_vectors(h0_events, model, event_contract),
                            actual_h0,
                        )
                    )
            if role == SECOND_ROLE:
                second_features.append(base)
                continue
            target_events, full_partition = events_from_requests(
                requests, strategy, event_contract
            )
            full_partition_checks.append(full_partition)
            target_fields = prefix_events("target_event_", target_events)
            target_row = {
                **identifiers(
                    profile, policy, "phase30_raw_full_window_structured_event_teacher"
                ),
                **target_fields,
            }
            if role == FIRST_ROLE:
                first_features.append(base)
                first_targets.append(target_row)
            elif role in DEVELOPMENT_ROLES:
                development.append({**base, **target_fields})
                development_targets.append(target_row)
            else:
                raise ValueError(role)
            for model_name in MODELS:
                model, _ = models[model_name]
                for tp_size in TP_SIZES:
                    actual = target_vectors_from_requests(requests, strategy, model)
                    adapter_errors[("phase30_new", "hfull")].append(
                        max_vector_error(
                            reconstruct_message_vectors(
                                target_events, model, event_contract
                            ),
                            actual,
                        )
                    )

    profile_rows = [
        {**profile, **scalar_profile_features(profile)} for profile in new_profiles
    ]
    write_csv_gz(
        args.output_dir / "profiles/new_low_dimensional_profiles.csv.gz",
        profile_rows,
    )
    write_csv_gz(
        args.output_dir / "labels/development_event_targets.csv.gz",
        development_targets,
    )
    write_csv_gz(
        args.output_dir / "labels/first_confirmation_event_targets.csv.gz",
        first_targets,
    )
    write_csv_gz(
        args.output_dir / "labels/compact32_h0_events.csv.gz", h0_rows
    )
    write_csv_gz(
        args.output_dir / "dataset/development_examples.csv.gz", development
    )
    write_csv_gz(
        args.output_dir / "dataset/first_confirmation_features.csv.gz",
        first_features,
    )
    write_csv_gz(
        args.output_dir / "dataset/second_confirmation_features.csv.gz",
        second_features,
    )
    write_json(
        args.output_dir / "feature_columns.json",
        {
            "schema_version": "phase30b-event-predictor-feature-columns-v1",
            "feature_count": len(features),
            "feature_columns": features,
            "h0_event_count": len(event_columns),
            "h0_event_columns": [f"h0_event_{name}" for name in event_columns],
            "target_event_count": len(event_columns),
            "target_event_columns": [
                f"target_event_{name}" for name in event_columns
            ],
            "model_structure_in_deterministic_adapter_only": True,
            "tp_size_invariant_under_current_teacher": True,
        },
    )
    inventory = []
    all_profile_roles = {
        **{profile_id: profile["phase27_role"] for profile_id, profile in old_profiles.items()},
        **{profile["profile_id"]: profile["phase27_role"] for profile in new_profiles},
    }
    for role, count in sorted(Counter(all_profile_roles.values()).items()):
        inventory.append(
            {
                "role": role,
                "profiles": count,
                "profile_policy_units": count * len(STRATEGIES),
                "contains_hfull_event_target": role != SECOND_ROLE,
                "allowed_training_access": role in DEVELOPMENT_ROLES,
            }
        )
    write_csv(args.output_dir / "analysis/dataset_inventory.csv", inventory)
    adapter_rows = []
    for (provenance, label_kind), errors in sorted(adapter_errors.items()):
        adapter_rows.append(
            {
                "provenance": provenance,
                "label_kind": label_kind,
                "expanded_model_tp_configurations": len(errors),
                "expanded_model_tp_phase_rows": len(errors) * len(PHASES),
                "max_calls_bin_absolute_error": max(value[0] for value in errors),
                "max_logical_bytes_bin_absolute_error": max(
                    value[1] for value in errors
                ),
                "max_calls_bin_relative_error": max(value[2] for value in errors),
                "max_logical_bytes_bin_relative_error": max(
                    value[3] for value in errors
                ),
            }
        )
    write_csv(args.output_dir / "analysis/adapter_exactness.csv", adapter_rows)

    role_counts = Counter(all_profile_roles.values())
    event_target_absent = {
        f"target_event_{name}" for name in event_columns
    }
    checks = {
        "raw_source_hashes_6_of_6": len(raw_checks) == 6
        and all(raw_checks.values()),
        "profiles_132_roles_75_27_15_15": len(all_profile_roles) == 132
        and role_counts
        == Counter(
            {
                "development_train": 75,
                "development_validation": 27,
                FIRST_ROLE: 15,
                SECOND_ROLE: 15,
            }
        ),
        "new_history_counts_match_90_of_90": len(count_checks) == 90
        and all(count_checks),
        "features_91_h0_events_62_targets_62": len(features) == 91
        and len(event_columns) == 62,
        "development_units_306_fit_225_validation_81": len(development) == 306
        and sum(row["role"] == "development_train" for row in development)
        == 225
        and sum(
            row["role"] == "development_validation" for row in development
        )
        == 81,
        "first_features_targets_45_isolated": len(first_features) == 45
        and len(first_targets) == 45
        and not (event_target_absent & set(first_features[0])),
        "second_features_45_targets_zero": len(second_features) == 45
        and not second_full_targets_generated
        and not (event_target_absent & set(second_features[0])),
        "h0_event_units_396": len(h0_rows) == 396,
        "full_partition_checks_225_of_225": len(full_partition_checks) == 225
        and all(full_partition_checks),
        "h0_partition_checks_270_of_270": len(h0_partition_checks) == 270
        and all(h0_partition_checks),
        "adapter_exact_for_6318_hfull_expansions": sum(
            row["expanded_model_tp_phase_rows"]
            for row in adapter_rows
            if row["label_kind"] == "hfull"
        )
        == 6318,
        "adapter_exact_for_7128_h0_expansions": sum(
            row["expanded_model_tp_phase_rows"]
            for row in adapter_rows
            if row["label_kind"] == "compact32_h0"
        )
        == 7128,
        "adapter_relative_errors_below_one_part_per_billion": all(
            float(row["max_calls_bin_relative_error"]) <= 1e-9
            and float(row["max_logical_bytes_bin_relative_error"]) <= 1e-9
            for row in adapter_rows
        ),
        "no_request_arrays_saved": not any(
            name in profile_rows[0]
            for name in (
                "input_lens",
                "output_lens",
                "requests_json",
                "full_request_list",
            )
        )
        and not any(
            name in development[0]
            for name in (
                "input_lens",
                "output_lens",
                "requests_json",
                "full_request_list",
            )
        ),
        "teacher_promoted_by_phase26a": phase26a["promoted_labels"] == 1296
        and phase26a["promoted_label_status"] == TEACHER_STATUS,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    audit = {
        "schema_version": "phase30b-tp-structured-event-dataset-audit-v1",
        "status": status,
        "checks": checks,
    }
    write_json(args.output_dir / "audit_summary.json", audit)
    if status != "PASS":
        raise RuntimeError(audit)

    new_request_total = sum(int(profile["request_count"]) for profile in new_profiles)
    summary = {
        "schema_version": "phase30b-tp-structured-event-dataset-v1",
        "status": status,
        "objective": modeling_contract["objective"],
        "profiles": len(all_profile_roles),
        "profile_role_counts": dict(role_counts),
        "new_phase30_profiles": len(new_profiles),
        "new_full_window_requests": new_request_total,
        "feature_columns": len(features),
        "structured_event_targets": len(event_columns),
        "development_profile_policy_units": len(development),
        "first_confirmation_feature_units": len(first_features),
        "first_confirmation_target_units": len(first_targets),
        "second_confirmation_feature_units": len(second_features),
        "second_confirmation_target_units": 0,
        "h0_event_units": len(h0_rows),
        "expanded_hfull_adapter_audits": 6318,
        "expanded_h0_adapter_audits": 7128,
        "teacher_status": TEACHER_STATUS,
        "teacher_kind": TEACHER_KIND,
        "second_confirmation_targets_generated": False,
        "raw_source_checks": raw_checks,
        "inputs": {
            "phase30_selection_sha256": sha256(args.phase30_selection),
            "event_contract_sha256": sha256(args.event_contract),
            "modeling_contract_sha256": sha256(args.modeling_contract),
            "phase30a_summary_sha256": sha256(args.phase30a_summary),
            "phase29_feature_contract_sha256": sha256(
                args.phase29_feature_contract
            ),
            "phase29_development_examples_sha256": sha256(
                args.phase29_development_examples
            ),
            "phase29_development_targets_sha256": sha256(
                args.phase29_development_targets
            ),
            "phase29_h0_baselines_sha256": sha256(args.phase29_h0_baselines),
            "phase29_profiles_sha256": sha256(args.phase29_profiles),
            "model_features_sha256": sha256(args.model_features),
            "phase26a_summary_sha256": sha256(args.phase26a_summary),
            "raw_manifest_sha256": sha256(args.raw_dir / "source_manifest.json"),
        },
        "checks": checks,
        "next_step": (
            "train structured-event residual/direct controls on 225 fit units with 81-unit "
            "early stopping, and freeze both confirmation prediction sets before target access"
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "README.md").write_text(
        f"""# Phase 30B：TP结构化batch事件监督数据集

状态：**PASS**。本阶段把Phase 29允许复用的42个开发画像与Phase 30A的90个全新画像合并，
形成75 train、27 validation、15 first confirmation和15 second confirmation，共132个独立画像。
训练单位是画像×固定策略，不再按模型、TP size或phase重复计算独立样本。

每个单位保存91列低维画像/策略特征、compact32 H0的62维结构化事件先验。开发集306个单位
含Hfull event target，其中225用于拟合、81用于早停；第一确认45个feature与45个target分文件；
第二确认仅45个无target feature，Hfull event target尚未生成。

62维目标由23个prefill batch-count、23个prefill token-mass和16个decode active-lane step
count组成。结构适配器对Phase 29既有标签和Phase 30新teacher共做6,318次Hfull与7,128次H0
跨模型/TP展开核验，所有12桶calls与logical-bytes误差均在浮点容差内。因此DNN只需学习
scheduler事件，模型collective数与bytes/token由确定性适配器恢复。

90个新窗口共{new_request_total:,}个完整请求。请求数组只在内存中用于低维聚合、H0和离线
teacher，没有保存到profiles、labels、dataset或Git。Phase 29两批确认画像未进入本数据集；
Phase 30两批确认预测必须在读取第一确认target前同时冻结。
"""
    )
    (args.output_dir / "DONE").write_text("PASS\n")
    write_json(
        args.output_dir / "logs/build.log",
        {
            "schema_version": "phase30b-build-log-v1",
            "status": status,
            "profiles": len(all_profile_roles),
            "new_full_window_requests": new_request_total,
            "development_units": len(development),
            "first_confirmation_units": len(first_features),
            "second_confirmation_feature_units": len(second_features),
            "second_confirmation_targets_generated": False,
            "full_request_lists_saved": False,
        },
    )
    files = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    (args.output_dir / "manifest.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(args.output_dir)}\n" for path in files
        )
    )
    print(
        json.dumps(
            {
                "status": status,
                "profiles": len(all_profile_roles),
                "new_full_window_requests": new_request_total,
                "development_units": len(development),
                "first_confirmation_units": len(first_features),
                "second_confirmation_feature_units": len(second_features),
                "structured_event_targets": len(event_columns),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
