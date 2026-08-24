#!/usr/bin/env python3
"""Phase62 reserved-payload and fresh-placement contracts."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P60 = HERE.parent / "phase60_pd_multi_endpoint_composability"
PHASE60_RESULT = ROOT / "experiment-results/phase60_pd_multi_endpoint_composability"
PHASE61_RESULT = ROOT / "experiment-results/phase61_pd_contention_correction"


def _load_p60_contracts():
    spec = importlib.util.spec_from_file_location("phase62_pinned_phase60_contracts", P60 / "contracts.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned Phase60 contracts")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P60_CONTRACTS = _load_p60_contracts()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def contract() -> dict[str, Any]:
    return load_json(HERE / "experiment.json")


def selected_layouts(spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    spec = spec or contract()
    source = load_json(PHASE60_RESULT / "contracts/selected_model_transfer_layouts.json")["layouts"]
    selected = [row for row in source if row["model_id"] in spec["selected_models"]]
    if [row["model_id"] for row in selected] != spec["selected_models"]:
        raise RuntimeError({"selected_layouts": [row["model_id"] for row in selected]})
    return selected


def layout_by_id(model_id: str) -> dict[str, Any]:
    rows = [row for row in selected_layouts() if row["model_id"] == model_id]
    if len(rows) != 1:
        raise RuntimeError(f"unknown Phase62 model: {model_id}")
    return rows[0]


def _pair_grid() -> dict[str, Any]:
    return load_json(PHASE60_RESULT / "contracts/payload_pair_grid.json")


def payload_pairs(model_id: str) -> list[dict[str, Any]]:
    rows = _pair_grid()["reserved_future_blind"][model_id]
    if len(rows) != 10:
        raise RuntimeError({"reserved_pair_count": model_id, "actual": len(rows)})
    return copy.deepcopy(rows)


def development_pair_ids() -> set[str]:
    return {
        row["pair_id"]
        for values in _pair_grid()["development"].values()
        for row in values
    }


def validate_pair_contract(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = spec or contract()
    blind = {model: payload_pairs(model) for model in spec["selected_models"]}
    blind_ids = {row["pair_id"] for values in blind.values() for row in values}
    dev_ids = development_pair_ids()
    if len(blind_ids) != 20 or blind_ids & dev_ids:
        raise RuntimeError({"blind_pair_count": len(blind_ids), "development_overlap": sorted(blind_ids & dev_ids)})
    return {
        "ok": True,
        "counts": {model: len(values) for model, values in blind.items()},
        "reserved_sha256": canonical_sha(blind),
        "development_pair_ids_sha256": canonical_sha(sorted(dev_ids)),
    }


def iteration_counts(total_payload_bytes: int, spec: dict[str, Any] | None = None) -> tuple[int, int]:
    measurement = (spec or contract())["measurement_contract"]
    timed = int(measurement["target_bytes_per_mode_block"]) // max(int(total_payload_bytes), 1)
    timed = max(int(measurement["timed_iterations_min"]), min(int(measurement["timed_iterations_max"]), timed))
    warmup = max(int(measurement["warmup_iterations_min"]), min(int(measurement["warmup_iterations_max"]), timed // 10))
    return warmup, timed


def _phase60_plan() -> dict[str, Any]:
    return load_json(PHASE60_RESULT / "contracts/topology_plan.json")


def _endpoint_key(endpoint: dict[str, Any]) -> tuple[str, int, str]:
    return (str(endpoint["host"]), int(endpoint["physical_gpu"]), str(endpoint["ib_device"]))


def phase60_endpoint_keys() -> set[tuple[str, int, str]]:
    return {
        _endpoint_key(endpoint)
        for measurement in _phase60_plan()["measurements"]
        for endpoint in measurement["ranks"]
    }


def phase60_host_signatures() -> dict[str, set[tuple[str, ...]]]:
    output: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for measurement in _phase60_plan()["measurements"]:
        output[measurement["topology_level"]].add(tuple(sorted({row["host"] for row in measurement["ranks"]})))
    return output


def validate_inventory(inventory: dict[str, Any], spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = spec or contract()
    errors: list[Any] = []
    if inventory.get("schema_version") != "phase62-topology-inventory-v1":
        errors.append("schema_version")
    if inventory.get("blind_inventory_frozen_before_raw") is not True:
        errors.append("blind_inventory_not_frozen")
    if inventory.get("selection_not_influenced_by_phase61_predictions_or_errors") is not True:
        errors.append("placement_selection_may_use_model_or_error")
    freshness = inventory.get("freshness_contract") if isinstance(inventory.get("freshness_contract"), dict) else {}
    expected_freshness = {
        "all_endpoint_tuples_absent_from_phase60": True,
        "minimum_new_host_signatures_per_topology": int(spec["fresh_placement_contract"]["minimum_new_host_signatures_per_topology"]),
        "phase60_plan_sha256": file_sha(PHASE60_RESULT / "contracts/topology_plan.json"),
    }
    if freshness != expected_freshness:
        errors.append({"freshness_contract": freshness, "expected": expected_freshness})
    structural = copy.deepcopy(inventory)
    structural["schema_version"] = "phase60-topology-inventory-v1"
    try:
        base_audit = P60_CONTRACTS.validate_inventory(structural)
    except RuntimeError as exc:
        errors.append({"phase60_structural_contract": str(exc)})
        base_audit = {"normalized_placements": []}
    old_endpoints = phase60_endpoint_keys()
    old_hosts = phase60_host_signatures()
    normalized = []
    new_host_counts = Counter()
    for placement in base_audit.get("normalized_placements", []):
        endpoints = [
            endpoint
            for side in ("A", "B")
            for endpoint in placement["sides"][side]
        ]
        overlap = sorted(set(_endpoint_key(endpoint) for endpoint in endpoints) & old_endpoints)
        if overlap:
            errors.append({"phase60_endpoint_reuse": placement["placement_id"], "endpoints": overlap})
        host_signature = tuple(sorted({endpoint["host"] for endpoint in endpoints}))
        host_fresh = host_signature not in old_hosts[placement["topology_level"]]
        if host_fresh:
            new_host_counts[placement["topology_level"]] += 1
        normalized.append({
            **placement,
            "freshness": {
                "phase60_endpoint_overlap_count": len(overlap),
                "all_four_endpoint_tuples_fresh": len(overlap) == 0,
                "host_signature": list(host_signature),
                "host_signature_fresh_for_topology": host_fresh,
            },
        })
    minimum = int(spec["fresh_placement_contract"]["minimum_new_host_signatures_per_topology"])
    for level in ("L1", "L2", "L3"):
        if new_host_counts[level] < minimum:
            errors.append({"insufficient_new_host_signatures": level, "actual": new_host_counts[level], "minimum": minimum})
    if errors:
        raise RuntimeError({"invalid_phase62_inventory": errors})
    return {
        "ok": True,
        "placements": len(normalized),
        "fresh_endpoint_slots": sum(
            4 for placement in normalized if placement["freshness"]["all_four_endpoint_tuples_fresh"]
        ),
        "new_host_signatures": dict(new_host_counts),
        "max_simultaneous_nodes_per_shard": 2,
        "simultaneous_gpu_processes_per_shard": 3,
        "normalized_placements": sorted(normalized, key=lambda row: (row["topology_level"], row["replica_id"])),
    }


def _measurement_ranks(placement: dict[str, Any], configuration: str) -> list[dict[str, Any]]:
    sides = placement["sides"]
    slots = [sides["A"][0], sides["B"][0], sides["B"][1]] if configuration == "P1D2" else [sides["A"][0], sides["A"][1], sides["B"][0]]
    roles = ["P0", "D0", "D1"] if configuration == "P1D2" else ["P0", "P1", "D0"]
    return [{**endpoint, "rank": rank, "role": roles[rank]} for rank, endpoint in enumerate(slots)]


def expand_plan(
    inventory: dict[str, Any],
    inventory_sha256: str,
    generated_at_utc: str,
    workflow_commit: str,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = spec or contract()
    inventory_audit = validate_inventory(inventory, spec)
    pairs = validate_pair_contract(spec)
    measurements = []
    for model_id in spec["selected_models"]:
        layout = layout_by_id(model_id)
        for configuration in spec["research_scope"]["fixed_configurations"]:
            for placement in inventory_audit["normalized_placements"]:
                ranks = _measurement_ranks(placement, configuration)
                base = {
                    "measurement_id": f"{model_id}__{configuration.lower()}__{placement['topology_level'].lower()}__r{placement['replica_id']}",
                    "model_id": model_id,
                    "configuration": configuration,
                    "topology_level": placement["topology_level"],
                    "replica_id": placement["replica_id"],
                    "placement_id": placement["placement_id"],
                    "classification_evidence": placement["classification_evidence"],
                    "freshness": placement["freshness"],
                    "world_size": 3,
                    "op": "sglang_mooncake_two_flow_batch_transfer_sync",
                    "ranks": ranks,
                    "model_layout_sha256": canonical_sha(layout),
                    "reserved_pairs_sha256": canonical_sha(payload_pairs(model_id)),
                }
                measurements.append({**base, "measurement_sha256": canonical_sha(base)})
    base = {
        "schema_version": "phase62-topology-plan-v1",
        "workflow_commit": workflow_commit,
        "generated_at_utc": generated_at_utc,
        "inventory_sha256": inventory_sha256,
        "inventory_schema_version": inventory["schema_version"],
        "inventory_metadata": {
            key: inventory[key]
            for key in (
                "created_at_utc",
                "created_by",
                "classification_source",
                "classification_frozen_before_measurement",
                "classification_not_inferred_from_benchmark",
                "blind_inventory_frozen_before_raw",
                "selection_not_influenced_by_phase61_predictions_or_errors",
                "fabric_notes",
                "resource_allocation_contract",
                "freshness_contract",
            )
        },
        "source_phase60_plan_file_sha256": file_sha(PHASE60_RESULT / "contracts/topology_plan.json"),
        "frozen_contention_model_file_sha256": file_sha(PHASE61_RESULT / "model/contention_correction.json"),
        "selected_layouts_sha256": canonical_sha(selected_layouts(spec)),
        "reserved_pairs_sha256": pairs["reserved_sha256"],
        "development_pair_ids_sha256": pairs["development_pair_ids_sha256"],
        "freshness_summary": {
            "fresh_endpoint_slots": inventory_audit["fresh_endpoint_slots"],
            "new_host_signatures": inventory_audit["new_host_signatures"],
        },
        "measurements": measurements,
    }
    return {**base, "plan_sha256": canonical_sha(base)}


def validate_plan(plan: dict[str, Any], spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = spec or contract()
    errors: list[Any] = []
    pairs = validate_pair_contract(spec)
    if plan.get("schema_version") != "phase62-topology-plan-v1":
        errors.append("schema_version")
    base = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != canonical_sha(base):
        errors.append("plan_sha256")
    metadata = plan.get("inventory_metadata") if isinstance(plan.get("inventory_metadata"), dict) else {}
    resources = metadata.get("resource_allocation_contract") if isinstance(metadata.get("resource_allocation_contract"), dict) else {}
    if metadata.get("blind_inventory_frozen_before_raw") is not True or metadata.get("selection_not_influenced_by_phase61_predictions_or_errors") is not True:
        errors.append("blind_inventory_freeze")
    if resources.get("four_node_allocation_required") is not False or resources.get("simultaneous_nodes_per_shard") != {"L1": 1, "L2": 2, "L3": 2} or resources.get("simultaneous_gpu_processes_per_shard") != 3:
        errors.append("resource_allocation_contract")
    if plan.get("source_phase60_plan_file_sha256") != file_sha(PHASE60_RESULT / "contracts/topology_plan.json"):
        errors.append("phase60_plan_sha")
    if plan.get("frozen_contention_model_file_sha256") != file_sha(PHASE61_RESULT / "model/contention_correction.json"):
        errors.append("phase61_model_sha")
    if plan.get("selected_layouts_sha256") != canonical_sha(selected_layouts(spec)) or plan.get("reserved_pairs_sha256") != pairs["reserved_sha256"] or plan.get("development_pair_ids_sha256") != pairs["development_pair_ids_sha256"]:
        errors.append("layout_or_pair_sha")
    freshness = plan.get("freshness_summary") if isinstance(plan.get("freshness_summary"), dict) else {}
    minimum = int(spec["fresh_placement_contract"]["minimum_new_host_signatures_per_topology"])
    if freshness.get("fresh_endpoint_slots") != 24 or any(int(freshness.get("new_host_signatures", {}).get(level, 0)) < minimum for level in ("L1", "L2", "L3")):
        errors.append("freshness_summary")
    measurements = plan.get("measurements") if isinstance(plan.get("measurements"), list) else []
    expected = {
        (model, configuration, level, replica)
        for model in spec["selected_models"]
        for configuration in spec["research_scope"]["fixed_configurations"]
        for level in ("L1", "L2", "L3")
        for replica in (0, 1)
    }
    actual = Counter((row.get("model_id"), row.get("configuration"), row.get("topology_level"), row.get("replica_id")) for row in measurements)
    if set(actual) != expected or any(value != 1 for value in actual.values()):
        errors.append({"measurement_matrix": dict(actual)})
    old_endpoints = phase60_endpoint_keys()
    for row in measurements:
        expected_id = f"{row.get('model_id')}__{str(row.get('configuration', '')).lower()}__{str(row.get('topology_level', '')).lower()}__r{row.get('replica_id')}"
        if row.get("measurement_id") != expected_id or row.get("world_size") != 3 or row.get("op") != "sglang_mooncake_two_flow_batch_transfer_sync":
            errors.append(f"identity:{row.get('measurement_id')}")
        no_sha = {key: value for key, value in row.items() if key != "measurement_sha256"}
        if row.get("measurement_sha256") != canonical_sha(no_sha):
            errors.append(f"measurement_sha:{row.get('measurement_id')}")
        if row.get("model_layout_sha256") != canonical_sha(layout_by_id(row.get("model_id"))) or row.get("reserved_pairs_sha256") != canonical_sha(payload_pairs(row.get("model_id"))):
            errors.append(f"input_sha:{row.get('measurement_id')}")
        ranks = row.get("ranks") if isinstance(row.get("ranks"), list) else []
        expected_roles = ["P0", "D0", "D1"] if row.get("configuration") == "P1D2" else ["P0", "P1", "D0"]
        if len(ranks) != 3 or [rank.get("rank") for rank in ranks] != [0, 1, 2] or [rank.get("role") for rank in ranks] != expected_roles:
            errors.append(f"ranks:{row.get('measurement_id')}")
        if any(_endpoint_key(rank) in old_endpoints for rank in ranks):
            errors.append(f"phase60_endpoint_reuse:{row.get('measurement_id')}")
        expected_nodes = 1 if row.get("topology_level") == "L1" else 2
        if len({rank.get("host") for rank in ranks}) != expected_nodes:
            errors.append(f"node_count:{row.get('measurement_id')}")
        fresh = row.get("freshness") if isinstance(row.get("freshness"), dict) else {}
        if fresh.get("all_four_endpoint_tuples_fresh") is not True or int(fresh.get("phase60_endpoint_overlap_count", -1)) != 0:
            errors.append(f"freshness:{row.get('measurement_id')}")
    if errors:
        raise RuntimeError({"invalid_phase62_plan": errors})
    return {
        "ok": True,
        "plan_sha256": plan["plan_sha256"],
        "measurements": len(measurements),
        "world_size_per_shard": 3,
        "max_simultaneous_nodes_per_shard": 2,
        "official_points": int(spec["expected_official_points"]),
        "replica_points": int(spec["expected_replica_points"]),
        "fresh_endpoint_slots": 24,
    }


def measurement_by_id(plan: dict[str, Any], measurement_id: str) -> dict[str, Any]:
    rows = [row for row in plan["measurements"] if row["measurement_id"] == measurement_id]
    if len(rows) != 1:
        raise RuntimeError(f"unknown Phase62 measurement: {measurement_id}")
    return rows[0]
