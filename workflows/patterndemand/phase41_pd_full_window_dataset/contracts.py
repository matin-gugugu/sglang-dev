#!/usr/bin/env python3
"""Phase41 bounded-wave teacher, bundle, feature and CSV contracts."""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


BIN_EDGES_BYTES = [
    4096.0,
    13777.246867516858,
    46340.95001184158,
    155871.75497763665,
    524288.0,
    1763487.5990421579,
    5931641.601515722,
    19951584.63713749,
    67108864.0,
    225726412.6773962,
    759250124.9940125,
    2553802833.553599,
    8589934592.0,
]


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows[1:]):
        raise ValueError(f"inconsistent CSV schema: {path}")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse empty CSV: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows[1:]):
        raise ValueError(f"inconsistent CSV schema: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(buffer.getvalue().encode("utf-8"))


def write_bundle(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(encoded)


def read_bundle(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise RuntimeError("Phase41 bundle root must be an object")
    return value


def partition_requests(requests: list[Any], wave_size: int) -> list[list[Any]]:
    if wave_size <= 0:
        raise ValueError("wave_size must be positive")
    return [requests[index : index + wave_size] for index in range(0, len(requests), wave_size)]


def teacher_chunks_for_wave(
    requests: list[dict[str, Any]],
    *,
    chunk_tokens: int,
    page_size_tokens: int,
    kv_bytes_per_page: int,
) -> list[dict[str, Any]]:
    """Replay Phase40's frozen FCFS token budget inside exactly one wave."""
    if not requests:
        raise ValueError("teacher wave must be non-empty")
    if chunk_tokens <= 0 or page_size_tokens <= 0 or kv_bytes_per_page <= 0:
        raise ValueError("teacher sizes must be positive")
    if chunk_tokens % page_size_tokens:
        raise ValueError("chunk_tokens must be page aligned")
    rows: list[dict[str, Any]] = []
    chunk_indices: dict[str, int] = defaultdict(int)
    request_index = 0
    token_offset = 0
    scheduler_pass = 0
    while request_index < len(requests):
        budget = chunk_tokens
        made_progress = False
        while budget > 0 and request_index < len(requests):
            request = requests[request_index]
            total = int(request["prompt_tokens"])
            if total <= 0:
                raise ValueError({"non_positive_prompt_tokens": request})
            remaining = total - token_offset
            send_tokens = min(remaining, budget)
            is_last = send_tokens == remaining
            if not is_last:
                send_tokens -= send_tokens % page_size_tokens
            if send_tokens <= 0:
                break
            page_start = token_offset // page_size_tokens
            page_count = math.ceil(send_tokens / page_size_tokens)
            rid = str(request["rid"])
            rows.append(
                {
                    "case": request["case"],
                    "repeat": int(request["repeat"]),
                    "wave_index": int(request["wave_index"]),
                    "request_index": int(request["request_index"]),
                    "wave_request_index": int(request["wave_request_index"]),
                    "rid": rid,
                    "scheduler_pass": scheduler_pass,
                    "chunk_index": chunk_indices[rid],
                    "page_start": page_start,
                    "page_end": page_start + page_count,
                    "kv_page_count": page_count,
                    "logical_bytes": page_count * kv_bytes_per_page,
                }
            )
            chunk_indices[rid] += 1
            token_offset += send_tokens
            budget -= send_tokens
            made_progress = True
            if token_offset == total:
                request_index += 1
                token_offset = 0
            else:
                break
        if not made_progress:
            raise RuntimeError("bounded-wave teacher made no progress")
        scheduler_pass += 1
    return rows


def teacher_for_requests(
    *,
    case: str,
    repeat: int,
    requests: list[tuple[int, int]],
    wave_size: int,
    chunk_tokens: int,
    page_size_tokens: int,
    kv_bytes_per_page: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    request_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for wave_index, wave in enumerate(partition_requests(requests, wave_size)):
        prefix = f"p41::{case}::rep{repeat}::w{wave_index}"
        rows = []
        for local_index, pair in enumerate(wave):
            global_index = wave_index * wave_size + local_index
            row = {
                "case": case,
                "repeat": repeat,
                "wave_index": wave_index,
                "request_index": global_index,
                "wave_request_index": local_index,
                "rid_prefix": prefix,
                "rid": f"{prefix}_{local_index}",
                "prompt_tokens": int(pair[0]),
                "output_tokens_audit_only": int(pair[1]),
            }
            rows.append(row)
            request_rows.append(row)
        event_rows.extend(
            teacher_chunks_for_wave(
                rows,
                chunk_tokens=chunk_tokens,
                page_size_tokens=page_size_tokens,
                kv_bytes_per_page=kv_bytes_per_page,
            )
        )
    return request_rows, event_rows


def synthetic_requests(contract: dict[str, Any], count: int) -> list[tuple[int, int]]:
    cycle = [int(value) for value in contract["gpu_sentinel_contract"]["synthetic_prompt_cycle"]]
    return [(cycle[index % len(cycle)], 2) for index in range(count)]


def sentinel_workload(
    contract: dict[str, Any], bundle: dict[str, Any], kv_bytes_per_page: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    measurement = contract["measurement_contract"]
    sentinel = contract["gpu_sentinel_contract"]
    by_profile = {
        row["profile"]["profile_id"]: [tuple(map(int, pair)) for pair in row["requests"]]
        for row in bundle["development"]
    }
    cases: list[dict[str, Any]] = []
    for item in sentinel["synthetic_cases"]:
        cases.append(
            {
                "name": item["name"],
                "requests": synthetic_requests(contract, int(item["request_count"])),
                "repeats": int(item["repeats"]),
                "kind": "synthetic_boundary",
            }
        )
    for item in sentinel["real_full_window_cases"]:
        profile_id = item["profile_id"]
        if profile_id not in by_profile:
            raise RuntimeError(f"sentinel profile absent from bundle: {profile_id}")
        if len(by_profile[profile_id]) != int(item["request_count"]):
            raise RuntimeError(f"sentinel request count mismatch: {profile_id}")
        cases.append(
            {
                "name": item["name"],
                "profile_id": profile_id,
                "requests": by_profile[profile_id],
                "repeats": int(item["repeats"]),
                "kind": "real_complete_window",
            }
        )
    request_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    case_inventory: list[dict[str, Any]] = []
    for case in cases:
        case_requests = []
        case_events = []
        for repeat in range(case["repeats"]):
            requests, events = teacher_for_requests(
                case=case["name"],
                repeat=repeat,
                requests=case["requests"],
                wave_size=int(measurement["wave_size"]),
                chunk_tokens=int(measurement["chunked_prefill_tokens"]),
                page_size_tokens=int(measurement["page_size_tokens"]),
                kv_bytes_per_page=kv_bytes_per_page,
            )
            case_requests.extend(requests)
            case_events.extend(events)
        request_rows.extend(case_requests)
        teacher_rows.extend(case_events)
        case_inventory.append(
            {
                "case": case["name"],
                "kind": case["kind"],
                "profile_id": case.get("profile_id", ""),
                "requests_per_repeat": len(case["requests"]),
                "repeats": case["repeats"],
                "total_requests": len(case_requests),
                "waves_per_repeat": math.ceil(len(case["requests"]) / int(measurement["wave_size"])),
                "total_waves": len({(row["repeat"], row["wave_index"]) for row in case_requests}),
                "teacher_chunks": len(case_events),
            }
        )
    return request_rows, teacher_rows, case_inventory


def bin_index(payload_bytes: int) -> int:
    for index in range(12):
        if payload_bytes < BIN_EDGES_BYTES[index + 1]:
            return index
    return 11


def histogram_vectors(
    events: Iterable[dict[str, Any]], request_count: int
) -> tuple[list[float], list[float]]:
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    calls = [0.0] * 12
    logical_bytes = [0.0] * 12
    scale = 1000.0 / request_count
    for event in events:
        payload = int(event["logical_bytes"])
        index = bin_index(payload)
        calls[index] += scale
        logical_bytes[index] += payload * scale
    return calls, logical_bytes


def histogram_fields(prefix: str, events: Iterable[dict[str, Any]], request_count: int) -> dict[str, float]:
    calls, logical_bytes = histogram_vectors(events, request_count)
    output: dict[str, float] = {
        f"{prefix}_total_calls_per_1000": sum(calls),
        f"{prefix}_total_logical_bytes_per_1000": sum(logical_bytes),
    }
    for index, value in enumerate(calls):
        output[f"{prefix}_calls_bin_{index:02d}"] = value
    for index, value in enumerate(logical_bytes):
        output[f"{prefix}_logical_bytes_bin_{index:02d}"] = value
    return output


def _allocate_counts(probabilities: list[float], total: int) -> list[int]:
    values = [max(float(value), 0.0) for value in probabilities]
    mass = sum(values)
    if mass <= 0:
        raise ValueError("profile joint distribution has zero mass")
    values = [value / mass for value in values]
    raw = [value * total for value in values]
    result = [math.floor(value) for value in raw]
    order = sorted(range(len(raw)), key=lambda index: (-(raw[index] - result[index]), index))
    for index in order[: total - sum(result)]:
        result[index] += 1
    return result


def _scaled_representatives(
    base: list[float], weights: list[float], target: float, lower: float, upper: float
) -> list[int]:
    mass = sum(weights)
    normalized = [value / mass for value in weights] if mass > 0 else [1.0 / len(weights)] * len(weights)
    low, high = 0.001, 256.0
    for _ in range(80):
        scale = (low + high) / 2
        mean = sum(weight * min(max(value * scale, lower), upper) for value, weight in zip(base, normalized))
        if mean < target:
            low = scale
        else:
            high = scale
    scale = (low + high) / 2
    return [int(round(min(max(value * scale, lower), upper))) for value in base]


def pseudo_requests(profile: dict[str, Any]) -> list[tuple[int, int]]:
    """Exact stdlib translation of the frozen Phase21B 32-request H0."""
    joint_flat = [float(value) for value in json.loads(profile["joint_lm_4x4_json"])]
    if len(joint_flat) != 16:
        raise ValueError("joint_lm_4x4_json must contain 16 values")
    joint = [joint_flat[index : index + 4] for index in range(0, 16, 4)]
    counts = _allocate_counts(joint_flat, 32)
    input_weights = [sum(row) for row in joint]
    output_weights = [sum(joint[row][column] for row in range(4)) for column in range(4)]
    input_values = _scaled_representatives(
        [64, 320, 1280, 4096], input_weights, float(profile["input_mean_capped"]), 1, 8192
    )
    output_values = _scaled_representatives(
        [8, 24, 48, 96], output_weights, float(profile["output_mean_capped"]), 1, 128
    )
    scheduled: list[tuple[float, int]] = []
    for cell, amount in enumerate(counts):
        for occurrence in range(amount):
            scheduled.append(((occurrence + 0.5) / amount, cell))
    scheduled.sort()
    return [
        (input_values[cell // 4], output_values[cell % 4]) for _, cell in scheduled
    ]


def predictor_features(
    profile: dict[str, Any], feature_contract: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in feature_contract["profile_scalar_features"]:
        if name not in profile:
            raise RuntimeError(f"profile missing frozen scalar feature: {name}")
        output[f"feature_profile_{name}"] = profile[name]
    for source, spec in feature_contract["profile_array_features"].items():
        values = json.loads(profile[source])
        if len(values) != int(spec["length"]):
            raise RuntimeError(f"profile array feature length mismatch: {source}")
        for index, value in enumerate(values):
            output[f"{spec['output_prefix']}{index:02d}"] = value
    output.update(feature_contract["model_structure_features"])
    output.update(feature_contract["execution_strategy_features"])
    return output


def profile_example_rows(
    *,
    profile: dict[str, Any],
    requests: list[tuple[int, int]] | None,
    contract: dict[str, Any],
    feature_contract: dict[str, Any],
    kv_bytes_per_page: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    identifiers = {
        "profile_id": profile["profile_id"],
        "split_role": profile["split_role"],
        "source": profile["source"],
        "segment": profile["segment"],
        "source_split": profile["source_split"],
        "window_id": profile["window_id"],
        "cutoff_ms": profile["cutoff_ms"],
        "model": "qwen3-8b",
    }
    measurement = contract["measurement_contract"]
    h0_requests = pseudo_requests(profile)
    _, h0_events = teacher_for_requests(
        case=f"h0::{profile['profile_id']}",
        repeat=0,
        requests=h0_requests,
        wave_size=int(measurement["wave_size"]),
        chunk_tokens=int(measurement["chunked_prefill_tokens"]),
        page_size_tokens=int(measurement["page_size_tokens"]),
        kv_bytes_per_page=kv_bytes_per_page,
    )
    h0 = histogram_fields("h0", h0_events, len(h0_requests))
    feature_row = {**identifiers, **predictor_features(profile, feature_contract), **h0}
    if requests is None:
        return feature_row, None
    _, target_events = teacher_for_requests(
        case=f"hfull::{profile['profile_id']}",
        repeat=0,
        requests=requests,
        wave_size=int(measurement["wave_size"]),
        chunk_tokens=int(measurement["chunked_prefill_tokens"]),
        page_size_tokens=int(measurement["page_size_tokens"]),
        kv_bytes_per_page=kv_bytes_per_page,
    )
    target = histogram_fields("target", target_events, len(requests))
    residual: dict[str, float] = {}
    for kind in ("calls", "logical_bytes"):
        for index in range(12):
            residual[f"residual_{kind}_bin_{index:02d}"] = (
                target[f"target_{kind}_bin_{index:02d}"]
                - h0[f"h0_{kind}_bin_{index:02d}"]
            )
    example = {
        **feature_row,
        **target,
        **residual,
        "teacher_kind": "qwen3_pure_pd_bounded_wave_hfull_v1",
    }
    target_row = {
        **identifiers,
        "request_count": len(requests),
        **target,
        "teacher_kind": "qwen3_pure_pd_bounded_wave_hfull_v1",
    }
    return example, target_row


def validate_bundle(contract: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    expected = contract["dataset_contract"]
    development = bundle.get("development")
    blind = bundle.get("blind_features")
    if not isinstance(development, list) or not isinstance(blind, list):
        raise RuntimeError("bundle development/blind_features must be arrays")
    development_ids = []
    development_requests = 0
    roles: dict[str, int] = defaultdict(int)
    sources: dict[str, int] = defaultdict(int)
    for row in development:
        if set(row) != {"profile", "requests"}:
            raise RuntimeError("development bundle row must contain exactly profile and requests")
        profile = row["profile"]
        requests = row["requests"]
        if not isinstance(profile, dict) or not isinstance(requests, list) or not requests:
            raise RuntimeError("invalid development bundle row")
        if len(requests) != int(profile["request_count"]):
            raise RuntimeError(f"development count mismatch: {profile.get('profile_id')}")
        if any(
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(value, int) and value > 0 for value in pair)
            for pair in requests
        ):
            raise RuntimeError(f"invalid request pair: {profile.get('profile_id')}")
        development_ids.append(profile["profile_id"])
        development_requests += len(requests)
        roles[profile["split_role"]] += 1
        sources[profile["source"]] += 1
    blind_ids = []
    for profile in blind:
        if not isinstance(profile, dict) or profile.get("split_role") != "blind_confirmation":
            raise RuntimeError("invalid blind target-free profile")
        forbidden = {"requests", "input_lens", "output_lens", "full_request_list"}
        if forbidden.intersection(profile):
            raise RuntimeError(f"blind profile leaks full requests: {profile.get('profile_id')}")
        blind_ids.append(profile["profile_id"])
    checks = {
        "schema": bundle.get("schema_version") == "phase41-external-transfer-bundle-v1",
        "development_profiles": len(development) == int(expected["development_profiles"]),
        "development_requests": development_requests == int(expected["development_full_requests"]),
        "development_roles": dict(roles)
        == {
            "development_train": int(expected["development_train_profiles"]),
            "development_validation": int(expected["development_validation_profiles"]),
        },
        "development_sources": dict(sources) == expected["development_source_profiles"],
        "development_ids_unique": len(development_ids) == len(set(development_ids)),
        "blind_profiles": len(blind) == int(expected["blind_profiles"]),
        "blind_ids_unique": len(blind_ids) == len(set(blind_ids)),
        "development_blind_disjoint": not set(development_ids).intersection(blind_ids),
        "blind_targets_absent": bundle.get("blind_targets_generated") is False,
        "blind_requests_absent": all("requests" not in profile for profile in blind),
    }
    if not all(checks.values()):
        raise RuntimeError({"bundle_checks": checks})
    return {
        "checks": checks,
        "development_profiles": len(development),
        "development_requests": development_requests,
        "blind_profiles": len(blind),
        "roles": dict(roles),
        "sources": dict(sources),
    }
