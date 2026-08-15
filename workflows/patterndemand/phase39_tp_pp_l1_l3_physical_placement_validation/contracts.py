#!/usr/bin/env python3
"""Phase39 topology inventory/plan contracts shared by measurement and analysis."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract() -> dict:
    return load_json(HERE / "experiment.json")


def _placeholder(value: Any) -> bool:
    return isinstance(value, str) and (not value.strip() or value.startswith("REPLACE_"))


def validate_inventory(inventory: dict, spec: dict | None = None) -> dict:
    spec = spec or contract()
    errors = []
    if inventory.get("schema_version") != "phase39-topology-inventory-v1":
        errors.append("schema_version")
    for field in ("created_at_utc", "created_by", "classification_source", "fabric_notes"):
        if _placeholder(inventory.get(field)):
            errors.append(f"missing_or_placeholder:{field}")
    if inventory.get("classification_frozen_before_measurement") is not True:
        errors.append("classification_not_frozen")
    if inventory.get("classification_not_inferred_from_benchmark") is not True:
        errors.append("classification_may_be_posthoc")
    classification_source = str(inventory.get("classification_source", "")).lower()
    if any(word in classification_source for word in ("benchmark", "latency", "bandwidth", "speed test")):
        errors.append("classification_source_uses_measured_performance")

    placements = inventory.get("placements")
    if not isinstance(placements, list):
        placements = []
        errors.append("placements_not_list")
    minimum_replicas = int(spec["minimum_placement_replicas_per_case"])
    expected_keys = {(level, replica) for level in ("L1", "L2", "L3") for replica in range(minimum_replicas)}
    actual_keys = Counter((row.get("topology_level"), row.get("replica_id")) for row in placements)
    if set(actual_keys) != expected_keys or any(count != 1 for count in actual_keys.values()):
        errors.append({"placement_replica_matrix": dict(actual_keys), "expected": sorted(expected_keys)})
    placement_ids = [row.get("placement_id") for row in placements]
    if any(_placeholder(value) for value in placement_ids) or len(set(placement_ids)) != len(placement_ids):
        errors.append("placement_ids_missing_or_not_unique")

    normalized = []
    for placement in placements:
        level = placement.get("topology_level")
        replica = placement.get("replica_id")
        nodes = placement.get("nodes") if isinstance(placement.get("nodes"), list) else []
        if _placeholder(placement.get("evidence")):
            errors.append(f"missing_evidence:{level}:{replica}")
        expected_nodes = 1 if level == "L1" else 2
        if len(nodes) != expected_nodes:
            errors.append(f"node_count:{level}:{replica}:{len(nodes)}")
        hosts = [node.get("host") for node in nodes]
        racks = [node.get("rack_id") for node in nodes]
        if any(_placeholder(value) for value in hosts + racks) or len(set(hosts)) != len(hosts):
            errors.append(f"host_or_rack_identity:{level}:{replica}")
        if level == "L2" and len(set(racks)) != 1:
            errors.append(f"l2_not_same_rack:{replica}")
        if level == "L3" and len(set(racks)) != 2:
            errors.append(f"l3_not_cross_rack:{replica}")
        normalized_nodes = []
        minimum_gpus = 8 if level == "L1" else 4
        for node in nodes:
            aliases = node.get("host_aliases") if isinstance(node.get("host_aliases"), list) else []
            gpu_ids = node.get("gpu_ids") if isinstance(node.get("gpu_ids"), list) else []
            nic_ids = node.get("nic_ids") if isinstance(node.get("nic_ids"), list) else []
            if node.get("host") not in aliases:
                errors.append(f"canonical_host_missing_from_aliases:{node.get('host')}")
            if len(gpu_ids) < minimum_gpus or len(set(gpu_ids)) != len(gpu_ids) or any(not isinstance(value, int) or value < 0 for value in gpu_ids):
                errors.append(f"gpu_inventory:{node.get('host')}")
            if not nic_ids or any(_placeholder(value) for value in nic_ids):
                errors.append(f"nic_inventory:{node.get('host')}")
            if _placeholder(node.get("network_domain")):
                errors.append(f"network_domain:{node.get('host')}")
            normalized_nodes.append({
                "host": node.get("host"),
                "host_aliases": sorted(set(aliases)),
                "rack_id": node.get("rack_id"),
                "network_domain": node.get("network_domain"),
                "nic_ids": list(nic_ids),
                "gpu_ids": list(gpu_ids),
            })
        normalized.append({
            "topology_level": level,
            "replica_id": replica,
            "placement_id": placement.get("placement_id"),
            "evidence": placement.get("evidence"),
            "nodes": normalized_nodes,
        })
    if errors:
        raise RuntimeError({"invalid_phase39_topology_inventory": errors})
    normalized.sort(key=lambda row: (row["topology_level"], row["replica_id"]))
    return {
        "schema_version": inventory["schema_version"],
        "created_at_utc": inventory["created_at_utc"],
        "created_by": inventory["created_by"],
        "classification_source": inventory["classification_source"],
        "classification_frozen_before_measurement": True,
        "classification_not_inferred_from_benchmark": True,
        "fabric_notes": inventory["fabric_notes"],
        "placements": normalized,
    }


def expand_plan(
    inventory: dict,
    inventory_sha256: str,
    generated_at_utc: str,
    workflow_commit: str,
    spec: dict | None = None,
) -> dict:
    spec = spec or contract()
    normalized = validate_inventory(inventory, spec)
    by_key = {(row["topology_level"], row["replica_id"]): row for row in normalized["placements"]}
    measurements = []
    for case in spec["required_measurement_matrix"]:
        world_size = int(case["world_size"])
        for replica in range(int(spec["minimum_placement_replicas_per_case"])):
            placement = by_key[(case["topology_level"], replica)]
            nodes = placement["nodes"]
            ranks = []
            if case["topology_level"] == "L1":
                selected = [(nodes[0], nodes[0]["gpu_ids"][:world_size])]
            else:
                per_node = world_size // 2
                selected = [(node, node["gpu_ids"][:per_node]) for node in nodes]
            rank = 0
            for node, gpu_ids in selected:
                for local_rank, physical_gpu in enumerate(gpu_ids):
                    ranks.append({
                        "rank": rank,
                        "local_rank": local_rank,
                        "host": node["host"],
                        "host_aliases": node["host_aliases"],
                        "rack_id": node["rack_id"],
                        "network_domain": node["network_domain"],
                        "nic_ids": node["nic_ids"],
                        "physical_gpu": physical_gpu,
                    })
                    rank += 1
            measurements.append({
                **case,
                "measurement_id": f"{case['case_key']}_r{replica}",
                "replica_id": replica,
                "placement_id": placement["placement_id"],
                "classification_evidence": placement["evidence"],
                "ranks": ranks,
            })
    plan = {
        "schema_version": "phase39-topology-plan-v1",
        "generated_at_utc": generated_at_utc,
        "workflow_commit": workflow_commit,
        "inventory_sha256": inventory_sha256,
        "classification_source": normalized["classification_source"],
        "classification_frozen_before_measurement": True,
        "classification_not_inferred_from_benchmark": True,
        "fabric_notes": normalized["fabric_notes"],
        "workflow_contract_schema_version": spec["schema_version"],
        "measurements": measurements,
    }
    plan["plan_sha256"] = canonical_sha({key: value for key, value in plan.items() if key != "plan_sha256"})
    validate_plan(plan, spec)
    return plan


def validate_plan(plan: dict, spec: dict | None = None) -> dict:
    spec = spec or contract()
    errors = []
    if plan.get("schema_version") != "phase39-topology-plan-v1":
        errors.append("schema_version")
    if plan.get("workflow_contract_schema_version") != spec["schema_version"]:
        errors.append("workflow_contract_schema_version")
    if not isinstance(plan.get("workflow_commit"), str) or len(plan["workflow_commit"]) != 40:
        errors.append("workflow_commit")
    if plan.get("classification_frozen_before_measurement") is not True or plan.get("classification_not_inferred_from_benchmark") is not True:
        errors.append("classification_freeze")
    expected_plan_sha = canonical_sha({key: value for key, value in plan.items() if key != "plan_sha256"})
    if plan.get("plan_sha256") != expected_plan_sha:
        errors.append("plan_sha256")
    measurements = plan.get("measurements") if isinstance(plan.get("measurements"), list) else []
    expected_matrix = {row["case_key"]: row for row in spec["required_measurement_matrix"]}
    required_replicas = int(spec["minimum_placement_replicas_per_case"])
    counts = Counter(row.get("case_key") for row in measurements)
    if set(counts) != set(expected_matrix) or any(count != required_replicas for count in counts.values()):
        errors.append({"measurement_matrix": dict(counts), "expected_replicas": required_replicas})
    measurement_ids = [row.get("measurement_id") for row in measurements]
    if len(set(measurement_ids)) != len(measurement_ids) or any(_placeholder(value) for value in measurement_ids):
        errors.append("measurement_ids")
    for measurement in measurements:
        case = expected_matrix.get(measurement.get("case_key"))
        if not case:
            continue
        for field in ("parallelism", "topology_level", "op", "world_size"):
            if measurement.get(field) != case[field]:
                errors.append(f"case_contract:{measurement.get('measurement_id')}:{field}")
        replica = measurement.get("replica_id")
        if replica not in range(required_replicas):
            errors.append(f"replica:{measurement.get('measurement_id')}")
        ranks = measurement.get("ranks") if isinstance(measurement.get("ranks"), list) else []
        world_size = int(case["world_size"])
        if [row.get("rank") for row in ranks] != list(range(world_size)):
            errors.append(f"global_ranks:{measurement.get('measurement_id')}")
        by_host = defaultdict(list)
        for row in ranks:
            by_host[row.get("host")].append(row)
            if row.get("host") not in row.get("host_aliases", []):
                errors.append(f"host_alias:{measurement.get('measurement_id')}:{row.get('rank')}")
        expected_hosts = 1 if case["topology_level"] == "L1" else 2
        if len(by_host) != expected_hosts:
            errors.append(f"host_count:{measurement.get('measurement_id')}")
        if case["topology_level"] != "L1" and any(len(rows) != world_size // 2 for rows in by_host.values()):
            errors.append(f"rank_split:{measurement.get('measurement_id')}")
        for host, host_ranks in by_host.items():
            if [row.get("local_rank") for row in host_ranks] != list(range(len(host_ranks))):
                errors.append(f"local_ranks:{measurement.get('measurement_id')}:{host}")
            gpus = [row.get("physical_gpu") for row in host_ranks]
            if len(set(gpus)) != len(gpus):
                errors.append(f"gpu_reuse:{measurement.get('measurement_id')}:{host}")
        racks = {row.get("rack_id") for row in ranks}
        if case["topology_level"] == "L2" and len(racks) != 1:
            errors.append(f"l2_racks:{measurement.get('measurement_id')}")
        if case["topology_level"] == "L3" and len(racks) != 2:
            errors.append(f"l3_racks:{measurement.get('measurement_id')}")
        if _placeholder(measurement.get("classification_evidence")):
            errors.append(f"classification_evidence:{measurement.get('measurement_id')}")
        measurement_without_sha = {key: value for key, value in measurement.items() if key != "measurement_sha256"}
        expected_measurement_sha = canonical_sha(measurement_without_sha)
        if "measurement_sha256" in measurement and measurement["measurement_sha256"] != expected_measurement_sha:
            errors.append(f"measurement_sha256:{measurement.get('measurement_id')}")
    if errors:
        raise RuntimeError({"invalid_phase39_topology_plan": errors})
    return {
        "ok": True,
        "plan_sha256": plan["plan_sha256"],
        "measurements": len(measurements),
        "matrix_cases": len(counts),
        "placement_replicas_per_case": required_replicas,
    }


def measurement_by_id(plan: dict, measurement_id: str) -> dict:
    matches = [row for row in plan["measurements"] if row["measurement_id"] == measurement_id]
    if len(matches) != 1:
        raise RuntimeError(f"measurement_id不存在或不唯一：{measurement_id}")
    return matches[0]


def measurement_sha(measurement: dict) -> str:
    return canonical_sha({key: value for key, value in measurement.items() if key != "measurement_sha256"})
