#!/usr/bin/env python3
"""Phase63 four-model external-validation contracts."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P60 = HERE.parent / "phase60_pd_multi_endpoint_composability"
PHASE51_RESULT = ROOT / "experiment-results/phase51_pd_l1_l3_physical_curve_library"
PHASE62_RESULT = ROOT / "experiment-results/phase62_pd_contention_fresh_blind"


def _load_p60_contracts():
    spec = importlib.util.spec_from_file_location("phase63_pinned_phase60_contracts", P60 / "contracts.py")
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
    source = load_json(PHASE51_RESULT / "contracts/model_transfer_layouts.json")["layouts"]
    by_model = {row["model_id"]: row for row in source}
    try:
        return [copy.deepcopy(by_model[model]) for model in spec["selected_models"]]
    except KeyError as exc:
        raise RuntimeError({"missing_phase51_layout": str(exc)}) from exc


def layout_by_id(model_id: str) -> dict[str, Any]:
    rows = [row for row in selected_layouts() if row["model_id"] == model_id]
    if len(rows) != 1:
        raise RuntimeError(f"unknown Phase63 model: {model_id}")
    return rows[0]


def _pair_grid() -> dict[str, Any]:
    return load_json(HERE / "payload_pair_grid.json")


def payload_pairs(model_id: str) -> list[dict[str, Any]]:
    rows = _pair_grid()["models"].get(model_id)
    if not isinstance(rows, list) or len(rows) != 10:
        raise RuntimeError({"external_pair_count": model_id, "actual": None if rows is None else len(rows)})
    return copy.deepcopy(rows)


def validate_pair_contract(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = spec or contract()
    grid = _pair_grid()
    errors: list[Any] = []
    base = {key: value for key, value in grid.items() if key != "grid_sha256"}
    if grid.get("schema_version") != "phase63-four-model-external-payload-grid-v1":
        errors.append("schema_version")
    if grid.get("grid_sha256") != canonical_sha(base):
        errors.append("grid_sha256")
    if grid.get("selection_frozen_before_phase63_raw") is not True or grid.get("selection_uses_phase63_concurrent_targets") is not False:
        errors.append("selection_freeze")
    if set(grid.get("models", {})) != set(spec["selected_models"]):
        errors.append({"models": sorted(grid.get("models", {})), "expected": spec["selected_models"]})
    all_ids: list[str] = []
    model_hashes: dict[str, str] = {}
    for model_id in spec["selected_models"]:
        rows = payload_pairs(model_id)
        layout = layout_by_id(model_id)
        knots = {int(row["page_count"]): row for row in layout["knots"]}
        expected_shapes = Counter({"symmetric": 6, "asymmetric": 4})
        if Counter(row.get("pair_shape") for row in rows) != expected_shapes:
            errors.append(f"pair_shapes:{model_id}")
        for row in rows:
            all_ids.append(str(row.get("pair_id")))
            for suffix in ("0", "1"):
                page = int(row[f"page_count{suffix}"])
                knot = knots.get(page)
                if knot is None or int(row[f"payload_bytes{suffix}"]) != int(knot["payload_bytes"]) or int(row[f"descriptor_bytes{suffix}"]) != int(knot["descriptor_bytes"]):
                    errors.append({"pair_not_phase51_knot": row.get("pair_id"), "side": suffix})
        model_hashes[model_id] = canonical_sha(rows)
    if len(all_ids) != 40 or len(set(all_ids)) != 40:
        errors.append({"pair_cardinality": len(all_ids), "unique": len(set(all_ids))})
    if errors:
        raise RuntimeError({"invalid_phase63_pair_grid": errors})
    return {
        "ok": True,
        "models": len(spec["selected_models"]),
        "counts": {model: 10 for model in spec["selected_models"]},
        "pairs": 40,
        "grid_sha256": grid["grid_sha256"],
        "model_pair_sha256": model_hashes,
    }


def iteration_counts(total_payload_bytes: int, spec: dict[str, Any] | None = None) -> tuple[int, int]:
    measurement = (spec or contract())["measurement_contract"]
    timed = int(measurement["target_bytes_per_mode_block"]) // max(int(total_payload_bytes), 1)
    timed = max(int(measurement["timed_iterations_min"]), min(int(measurement["timed_iterations_max"]), timed))
    warmup = max(int(measurement["warmup_iterations_min"]), min(int(measurement["warmup_iterations_max"]), timed // 10))
    return warmup, timed


def _endpoint_key(endpoint: dict[str, Any]) -> tuple[str, int, str]:
    return (str(endpoint["host"]), int(endpoint["physical_gpu"]), str(endpoint["ib_device"]))


def _phase62_placement_keys() -> dict[tuple[str, int], set[tuple[str, int, str]]]:
    source = load_json(PHASE62_RESULT / "contracts/topology_plan.json")
    output: dict[tuple[str, int], set[tuple[str, int, str]]] = {}
    for level in ("L1", "L2", "L3"):
        for replica in (0, 1):
            rows = [
                row for row in source["measurements"]
                if row["topology_level"] == level and int(row["replica_id"]) == replica
            ]
            output[(level, replica)] = {_endpoint_key(endpoint) for row in rows for endpoint in row["ranks"]}
    return output


def validate_inventory(inventory: dict[str, Any], spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = spec or contract()
    errors: list[Any] = []
    if inventory.get("schema_version") != "phase63-topology-inventory-v1":
        errors.append("schema_version")
    if inventory.get("inventory_frozen_before_phase63_raw") is not True:
        errors.append("inventory_not_frozen")
    if inventory.get("selection_uses_phase63_latency_or_error") is not False:
        errors.append("placement_selection_may_use_target")
    if inventory.get("phase62_endpoint_reuse_preferred_but_not_required") is not True:
        errors.append("phase62_reuse_policy")
    structural = copy.deepcopy(inventory)
    structural["schema_version"] = "phase60-topology-inventory-v1"
    try:
        base_audit = P60_CONTRACTS.validate_inventory(structural)
    except RuntimeError as exc:
        errors.append({"phase60_structural_contract": str(exc)})
        base_audit = {"normalized_placements": []}
    phase62 = _phase62_placement_keys()
    normalized = []
    exact_reuse = 0
    reused_slots = 0
    all_phase62_slots = set().union(*phase62.values())
    for placement in base_audit.get("normalized_placements", []):
        endpoint_keys = {
            _endpoint_key(endpoint)
            for side in ("A", "B")
            for endpoint in placement["sides"][side]
        }
        key = (placement["topology_level"], int(placement["replica_id"]))
        exact = endpoint_keys == phase62[key]
        exact_reuse += int(exact)
        reused = len(endpoint_keys & all_phase62_slots)
        reused_slots += reused
        normalized.append({
            **placement,
            "phase62_comparability": {
                "exact_same_topology_replica_endpoint_set": exact,
                "phase62_endpoint_slots_reused": reused,
                "placement_source": "exact_phase62_reuse" if exact else "metadata_frozen_replacement_or_mixed",
            },
        })
    if errors:
        raise RuntimeError({"invalid_phase63_inventory": errors})
    return {
        "ok": True,
        "placements": len(normalized),
        "endpoint_slots": len(normalized) * 4,
        "exact_phase62_placements": exact_reuse,
        "phase62_endpoint_slots_reused": reused_slots,
        "maximum_simultaneous_nodes_per_shard": 2,
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
                    "phase62_comparability": placement["phase62_comparability"],
                    "world_size": 3,
                    "op": "sglang_mooncake_two_flow_batch_transfer_sync",
                    "ranks": ranks,
                    "model_layout_sha256": canonical_sha(layout),
                    "external_pairs_sha256": pairs["model_pair_sha256"][model_id],
                }
                measurements.append({**base, "measurement_sha256": canonical_sha(base)})
    base = {
        "schema_version": "phase63-topology-plan-v1",
        "workflow_commit": workflow_commit,
        "generated_at_utc": generated_at_utc,
        "inventory_sha256": inventory_sha256,
        "inventory_schema_version": inventory["schema_version"],
        "inventory_metadata": {
            key: inventory[key]
            for key in (
                "created_at_utc", "created_by", "classification_source",
                "classification_frozen_before_measurement", "classification_not_inferred_from_benchmark",
                "inventory_frozen_before_phase63_raw", "selection_uses_phase63_latency_or_error",
                "phase62_endpoint_reuse_preferred_but_not_required", "fabric_notes",
                "resource_allocation_contract",
            )
        },
        "source_phase62_result_commit": spec["source_two_model_validation_result_commit"],
        "source_phase62_summary_file_sha256": file_sha(PHASE62_RESULT / "summary.json"),
        "frozen_contention_model_file_sha256": file_sha(PHASE62_RESULT / "contracts/frozen_contention_correction.json"),
        "phase51_curves_file_sha256": file_sha(PHASE51_RESULT / "curves/pd_mooncake_physical_curves.json"),
        "selected_layouts_sha256": canonical_sha(selected_layouts(spec)),
        "external_pair_grid_sha256": pairs["grid_sha256"],
        "placement_summary": {
            "endpoint_slots": inventory_audit["endpoint_slots"],
            "exact_phase62_placements": inventory_audit["exact_phase62_placements"],
            "phase62_endpoint_slots_reused": inventory_audit["phase62_endpoint_slots_reused"],
        },
        "measurements": measurements,
    }
    return {**base, "plan_sha256": canonical_sha(base)}


def validate_plan(plan: dict[str, Any], spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = spec or contract()
    errors: list[Any] = []
    pairs = validate_pair_contract(spec)
    if plan.get("schema_version") != "phase63-topology-plan-v1":
        errors.append("schema_version")
    base = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != canonical_sha(base):
        errors.append("plan_sha256")
    metadata = plan.get("inventory_metadata") if isinstance(plan.get("inventory_metadata"), dict) else {}
    resources = metadata.get("resource_allocation_contract") if isinstance(metadata.get("resource_allocation_contract"), dict) else {}
    if metadata.get("inventory_frozen_before_phase63_raw") is not True or metadata.get("selection_uses_phase63_latency_or_error") is not False:
        errors.append("inventory_freeze")
    if resources.get("four_node_allocation_required") is not False or resources.get("simultaneous_nodes_per_shard") != {"L1": 1, "L2": 2, "L3": 2} or resources.get("simultaneous_gpu_processes_per_shard") != 3:
        errors.append("resource_allocation_contract")
    if plan.get("source_phase62_result_commit") != spec["source_two_model_validation_result_commit"] or plan.get("source_phase62_summary_file_sha256") != file_sha(PHASE62_RESULT / "summary.json"):
        errors.append("phase62_source")
    if plan.get("frozen_contention_model_file_sha256") != file_sha(PHASE62_RESULT / "contracts/frozen_contention_correction.json"):
        errors.append("frozen_model_sha")
    if plan.get("phase51_curves_file_sha256") != file_sha(PHASE51_RESULT / "curves/pd_mooncake_physical_curves.json") or plan.get("selected_layouts_sha256") != canonical_sha(selected_layouts(spec)):
        errors.append("phase51_sources")
    if plan.get("external_pair_grid_sha256") != pairs["grid_sha256"]:
        errors.append("pair_grid_sha")
    placement = plan.get("placement_summary") if isinstance(plan.get("placement_summary"), dict) else {}
    if int(placement.get("endpoint_slots", -1)) != 24 or int(placement.get("exact_phase62_placements", -1)) not in range(7) or int(placement.get("phase62_endpoint_slots_reused", -1)) not in range(25):
        errors.append("placement_summary")
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
    for row in measurements:
        expected_id = f"{row.get('model_id')}__{str(row.get('configuration', '')).lower()}__{str(row.get('topology_level', '')).lower()}__r{row.get('replica_id')}"
        if row.get("measurement_id") != expected_id or row.get("world_size") != 3 or row.get("op") != "sglang_mooncake_two_flow_batch_transfer_sync":
            errors.append(f"identity:{row.get('measurement_id')}")
        no_sha = {key: value for key, value in row.items() if key != "measurement_sha256"}
        if row.get("measurement_sha256") != canonical_sha(no_sha):
            errors.append(f"measurement_sha:{row.get('measurement_id')}")
        if row.get("model_layout_sha256") != canonical_sha(layout_by_id(row.get("model_id"))) or row.get("external_pairs_sha256") != pairs["model_pair_sha256"].get(row.get("model_id")):
            errors.append(f"input_sha:{row.get('measurement_id')}")
        ranks = row.get("ranks") if isinstance(row.get("ranks"), list) else []
        expected_roles = ["P0", "D0", "D1"] if row.get("configuration") == "P1D2" else ["P0", "P1", "D0"]
        if len(ranks) != 3 or [rank.get("rank") for rank in ranks] != [0, 1, 2] or [rank.get("role") for rank in ranks] != expected_roles:
            errors.append(f"ranks:{row.get('measurement_id')}")
        if len({_endpoint_key(rank) for rank in ranks}) != 3:
            errors.append(f"endpoint_uniqueness:{row.get('measurement_id')}")
        expected_nodes = 1 if row.get("topology_level") == "L1" else 2
        if len({rank.get("host") for rank in ranks}) != expected_nodes:
            errors.append(f"node_count:{row.get('measurement_id')}")
    if errors:
        raise RuntimeError({"invalid_phase63_plan": errors})
    return {
        "ok": True,
        "plan_sha256": plan["plan_sha256"],
        "measurements": len(measurements),
        "world_size_per_shard": 3,
        "maximum_simultaneous_nodes_per_shard": 2,
        "official_points": int(spec["expected_official_points"]),
        "replica_points": int(spec["expected_replica_points"]),
        "endpoint_slots": 24,
        "exact_phase62_placements": int(placement["exact_phase62_placements"]),
    }


def measurement_by_id(plan: dict[str, Any], measurement_id: str) -> dict[str, Any]:
    rows = [row for row in plan["measurements"] if row["measurement_id"] == measurement_id]
    if len(rows) != 1:
        raise RuntimeError(f"unknown Phase63 measurement: {measurement_id}")
    return rows[0]
