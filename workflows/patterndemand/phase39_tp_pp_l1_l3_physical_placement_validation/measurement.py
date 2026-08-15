#!/usr/bin/env python3
"""Validate external raw shards and build conservative physical curves."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from contracts import file_sha, measurement_by_id, measurement_sha, validate_plan


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_raw(plan: dict, raw_dir: Path, spec: dict) -> tuple[list[dict], dict]:
    validate_plan(plan, spec)
    raw_dir = raw_dir.expanduser().resolve()
    files = sorted(raw_dir.rglob("*.jsonl"))
    records = []
    file_manifest = []
    errors = []
    for path in files:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        parsed = []
        for line_number, line in enumerate(lines, 1):
            try:
                record = json.loads(line)
                record["_raw_relative_path"] = str(path.relative_to(raw_dir))
                parsed.append(record)
            except json.JSONDecodeError as error:
                errors.append({"path": str(path), "line": line_number, "error": str(error)})
        records.extend(parsed)
        file_manifest.append({
            "relative_external_path": str(path.relative_to(raw_dir)),
            "sha256": file_sha(path),
            "bytes": path.stat().st_size,
            "records": len(parsed),
        })
    if errors:
        raise RuntimeError({"raw_json_errors": errors})
    return records, {
        "schema_version": "phase39-external-raw-manifest-v1",
        "external_raw_dir": str(raw_dir),
        "files": file_manifest,
        "file_count": len(file_manifest),
        "record_count": len(records),
        "raw_committed_to_git": False,
    }


def validate_raw(plan: dict, raw_dir: Path, spec: dict) -> dict:
    records, manifest = load_raw(plan, raw_dir, spec)
    measurement_contract = spec["measurement_contract"]
    payloads = [int(value) for value in measurement_contract["payload_bytes"]]
    iterations = int(measurement_contract["timed_iterations"])
    minimum = int(measurement_contract["minimum_independent_repeats"])
    extra = int(measurement_contract["extra_repeats_per_round"])
    maximum = int(measurement_contract["maximum_independent_repeats"])
    threshold = float(measurement_contract["repeat_median_cv_threshold"])
    expected_repeat_counts = set(range(minimum, maximum + 1, extra))
    known_ids = {row["measurement_id"] for row in plan["measurements"]}
    errors = []
    grouped = defaultdict(list)
    generated_at = _parse_time(plan["generated_at_utc"])
    for record in records:
        measurement_id = record.get("measurement_id")
        if measurement_id not in known_ids:
            errors.append({"unknown_measurement_id": measurement_id})
            continue
        measurement = measurement_by_id(plan, measurement_id)
        expected = {
            "schema_version": "phase39-distributed-raw-v1",
            "workflow_commit": plan["workflow_commit"],
            "plan_sha256": plan["plan_sha256"],
            "measurement_sha256": measurement_sha(measurement),
            "case_key": measurement["case_key"],
            "replica_id": measurement["replica_id"],
            "placement_id": measurement["placement_id"],
            "parallelism": measurement["parallelism"],
            "topology_level": measurement["topology_level"],
            "warmup_iterations": int(measurement_contract["warmup_iterations"]),
            "timed_iterations": iterations,
        }
        mismatches = {key: {"actual": record.get(key), "expected": value} for key, value in expected.items() if record.get(key) != value}
        if mismatches:
            errors.append({"measurement_id": measurement_id, "contract_mismatch": mismatches})
        try:
            if _parse_time(record["timestamp_utc"]) < generated_at or _parse_time(record["shard_started_at_utc"]) < generated_at:
                errors.append({"measurement_id": measurement_id, "error": "record_predates_frozen_plan"})
        except Exception:
            errors.append({"measurement_id": measurement_id, "error": "invalid_timestamp"})
        payload = record.get("payload_bytes")
        repeat = record.get("repeat_id")
        direction = record.get("direction")
        if payload not in payloads or not isinstance(repeat, int) or repeat < 0 or repeat >= maximum:
            errors.append({"measurement_id": measurement_id, "payload": payload, "repeat": repeat})
        allowed_directions = {"rank0_to_rank1", "rank1_to_rank0"} if measurement["parallelism"] == "pp" else {"collective"}
        if direction not in allowed_directions:
            errors.append({"measurement_id": measurement_id, "direction": direction})
        if isinstance(repeat, int):
            expected_raw_path = f"{measurement_id}/repeat_{repeat:02d}.jsonl"
            if record.get("_raw_relative_path") != expected_raw_path:
                errors.append({
                    "measurement_id": measurement_id,
                    "raw_relative_path": record.get("_raw_relative_path"),
                    "expected_raw_relative_path": expected_raw_path,
                })
        expected_op = "p2p_send_tensor" if measurement["parallelism"] == "pp" else "sglang_tp_all_reduce"
        expected_backend = measurement_contract["pp_backend" if measurement["parallelism"] == "pp" else "tp_backend"]
        if record.get("op") != expected_op or record.get("backend") != expected_backend:
            errors.append({
                "measurement_id": measurement_id,
                "actual_op": record.get("op"),
                "expected_op": expected_op,
                "actual_backend": record.get("backend"),
                "expected_backend": expected_backend,
            })
        if measurement["parallelism"] == "tp":
            dispatch = record.get("sglang_dispatch_components")
            expected_multi_node = measurement["topology_level"] != "L1"
            if not isinstance(dispatch, dict) or dispatch.get("multi_node") is not expected_multi_node:
                errors.append({
                    "measurement_id": measurement_id,
                    "sglang_dispatch_components": dispatch,
                    "expected_multi_node": expected_multi_node,
                })
        latency = record.get("latency_us", {}).get("median")
        samples = record.get("completion_cuda_samples_us")
        rank_samples = record.get("rank_cuda_samples_us")
        if not isinstance(latency, (int, float)) or not math.isfinite(float(latency)) or float(latency) <= 0:
            errors.append({"measurement_id": measurement_id, "invalid_latency": latency})
        if not isinstance(samples, list) or len(samples) != iterations or not all(math.isfinite(float(value)) and float(value) > 0 for value in samples):
            errors.append({"measurement_id": measurement_id, "invalid_completion_samples": True})
        if not isinstance(rank_samples, list) or len(rank_samples) != int(measurement["world_size"]) or any(len(values) != iterations for values in rank_samples):
            errors.append({"measurement_id": measurement_id, "invalid_rank_samples": True})
        if record.get("data_validation_pass") is not True:
            errors.append({"measurement_id": measurement_id, "data_validation_pass": record.get("data_validation_pass")})
        grouped[(measurement_id, repeat, payload)].append(record)

    expected_records_per_payload = {"pp": 2, "tp": 1}
    repeat_values = defaultdict(lambda: defaultdict(dict))
    repeats_by_measurement = defaultdict(set)
    for (measurement_id, repeat, payload), values in grouped.items():
        measurement = measurement_by_id(plan, measurement_id)
        expected_count = expected_records_per_payload[measurement["parallelism"]]
        if len(values) != expected_count:
            errors.append({"measurement_id": measurement_id, "repeat": repeat, "payload": payload, "records": len(values), "expected": expected_count})
            continue
        directions = {row["direction"] for row in values}
        expected_directions = {"rank0_to_rank1", "rank1_to_rank0"} if measurement["parallelism"] == "pp" else {"collective"}
        if directions != expected_directions:
            errors.append({"measurement_id": measurement_id, "repeat": repeat, "payload": payload, "directions": sorted(directions)})
            continue
        repeat_values[measurement_id][payload][repeat] = max(float(row["latency_us"]["median"]) for row in values)
        repeats_by_measurement[measurement_id].add(repeat)

    missing = []
    quality = []
    needs_extra = []
    final_high_variance = []
    for measurement in plan["measurements"]:
        measurement_id = measurement["measurement_id"]
        repeats = sorted(repeats_by_measurement.get(measurement_id, set()))
        if repeats and repeats != list(range(max(repeats) + 1)):
            errors.append({"measurement_id": measurement_id, "noncontiguous_repeats": repeats})
        count = len(repeats)
        if count < minimum:
            missing.append({"measurement_id": measurement_id, "present_repeats": repeats, "required_next": list(range(count, minimum))})
            continue
        if count not in expected_repeat_counts:
            errors.append({"measurement_id": measurement_id, "invalid_repeat_count": count, "allowed": sorted(expected_repeat_counts)})
            continue
        measurement_high = []
        for payload in payloads:
            values = repeat_values[measurement_id][payload]
            if sorted(values) != repeats:
                missing.append({"measurement_id": measurement_id, "payload": payload, "present_repeats": sorted(values), "expected": repeats})
                continue
            ordered = [values[repeat] for repeat in repeats]
            mean = statistics.fmean(ordered)
            cv = statistics.pstdev(ordered) / max(mean, 1e-12)
            row = {
                "measurement_id": measurement_id,
                "case_key": measurement["case_key"],
                "replica_id": measurement["replica_id"],
                "payload_bytes": payload,
                "repeat_count": count,
                "repeat_median_latency_us": statistics.median(ordered),
                "repeat_median_cv": cv,
            }
            quality.append(row)
            if cv > threshold:
                measurement_high.append({"payload_bytes": payload, "cv": cv})
        if measurement_high:
            if count < maximum:
                needs_extra.append({
                    "measurement_id": measurement_id,
                    "current_repeat_count": count,
                    "next_repeat_ids": list(range(count, min(count + extra, maximum))),
                    "high_variance_payloads": measurement_high,
                })
            else:
                final_high_variance.append({"measurement_id": measurement_id, "payloads": measurement_high})
    if errors:
        raise RuntimeError({"invalid_phase39_raw": errors[:100], "error_count": len(errors)})
    for record in records:
        record.pop("_raw_relative_path", None)
    complete = not missing and not needs_extra
    return {
        "schema_version": "phase39-measurement-quality-v1",
        "complete": complete,
        "raw_manifest": manifest,
        "missing": missing,
        "needs_extra_repeats": needs_extra,
        "final_high_variance": final_high_variance,
        "quality_rows": quality,
        "repeat_counts": {measurement["measurement_id"]: len(repeats_by_measurement.get(measurement["measurement_id"], set())) for measurement in plan["measurements"]},
        "records": records,
        "repeat_values": repeat_values,
    }


def build_curves(plan: dict, raw_audit: dict, spec: dict) -> tuple[dict, dict]:
    if not raw_audit["complete"]:
        raise RuntimeError("raw coverage/variance追加合同尚未完成")
    payloads = [int(value) for value in spec["measurement_contract"]["payload_bytes"]]
    quality_by_key = {(row["measurement_id"], row["payload_bytes"]): row for row in raw_audit["quality_rows"]}
    by_case = defaultdict(list)
    for measurement in plan["measurements"]:
        by_case[measurement["case_key"]].append(measurement)
    curves = []
    high_spread = []
    spread_threshold = float(spec["measurement_contract"]["cross_replica_relative_spread_diagnostic_threshold"])
    for case in spec["required_measurement_matrix"]:
        replicas = sorted(by_case[case["case_key"]], key=lambda row: row["replica_id"])
        knots = []
        for payload in payloads:
            replica_rows = []
            for measurement in replicas:
                quality = quality_by_key[(measurement["measurement_id"], payload)]
                replica_rows.append({
                    "replica_id": measurement["replica_id"],
                    "measurement_id": measurement["measurement_id"],
                    "placement_id": measurement["placement_id"],
                    "median_latency_us": quality["repeat_median_latency_us"],
                    "repeat_count": quality["repeat_count"],
                    "repeat_median_cv": quality["repeat_median_cv"],
                })
            latencies = [row["median_latency_us"] for row in replica_rows]
            low, high = min(latencies), max(latencies)
            spread = (high - low) / max(low, 1e-12)
            knot = {
                "payload_bytes": payload,
                "official_latency_us": high,
                "lower_latency_us": low,
                "upper_latency_us": high,
                "cross_replica_relative_spread": spread,
                "replicas": replica_rows,
            }
            knots.append(knot)
            if spread > spread_threshold:
                high_spread.append({"case_key": case["case_key"], "payload_bytes": payload, "relative_spread": spread})
        curve_id = f"phase39_{case['case_key']}_physical"
        curves.append({
            "curve_id": curve_id,
            "parallelism": case["parallelism"],
            "topology_level": case["topology_level"],
            "op": case["op"],
            "group_size": case["world_size"],
            "curve_evidence": "physical_measurement",
            "placement_replica_policy": spec["measurement_contract"]["curve_replica_policy"],
            "interpolation": spec["measurement_contract"]["interpolation"],
            "measurement_ids": [row["measurement_id"] for row in replicas],
            "placements": [{
                "replica_id": row["replica_id"],
                "placement_id": row["placement_id"],
                "classification_evidence": row["classification_evidence"],
                "rank_mapping": row["ranks"],
            } for row in replicas],
            "knots": knots,
        })
    payload = {
        "schema_version": "phase39-tp-pp-l1-l3-physical-curves-v1",
        "curve_evidence": "physical_measurement",
        "topology_plan_sha256": plan["plan_sha256"],
        "curve_replica_policy": spec["measurement_contract"]["curve_replica_policy"],
        "interpolation": spec["measurement_contract"]["interpolation"],
        "curves": curves,
    }
    registry = {
        "schema_version": "phase39-physical-curve-registry-v1",
        "boundary": "all 12 registered curves are physical for the frozen placement replicas only; Phase35 proxy curves remain comparison-only",
        "topology_plan_sha256": plan["plan_sha256"],
        "curves": [{
            "curve_id": curve["curve_id"],
            "parallelism": curve["parallelism"],
            "topology_level": curve["topology_level"],
            "group_size": curve["group_size"],
            "op": curve["op"],
            "evidence": "physical_measurement",
            "measurement_ids": curve["measurement_ids"],
        } for curve in curves],
        "high_cross_replica_spread": high_spread,
    }
    return payload, registry
