#!/usr/bin/env python3
"""Phase51 topology, model-descriptor and measurement-plan contracts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def contract() -> dict:
    return load_json(HERE / "experiment.json")


def _placeholder(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip() or value.startswith("REPLACE_")


def source_models() -> list[dict[str, Any]]:
    return load_json(HERE.parent / "phase48_pd_six_model_expanded_training/models.json")["models"]


def model_layouts(spec: dict | None = None) -> list[dict[str, Any]]:
    spec = spec or contract(); base_grid = [int(value) for value in spec["model_layout_contract"]["page_count_grid"]]
    maximum_tokens = int(spec["model_layout_contract"]["maximum_chunk_tokens"]); output=[]
    for model in source_models():
        page_size = int(model["page_size_tokens"]); max_pages = maximum_tokens // page_size
        pages = [value for value in base_grid if value <= max_pages]
        layers = int(model["num_hidden_layers"]); is_mla = bool(model["is_mla"])
        descriptor_count = layers if is_mla else 2 * layers
        bytes_per_page = int(model["kv_bytes_per_page"])
        if bytes_per_page % descriptor_count:
            raise RuntimeError({"model": model["model_id"], "kv_bytes_per_page": bytes_per_page, "descriptor_count": descriptor_count})
        descriptor_bytes_per_page = bytes_per_page // descriptor_count
        knots = [{"page_count": count, "payload_bytes": count * bytes_per_page, "descriptor_bytes": count * descriptor_bytes_per_page} for count in pages]
        output.append({
            "model_id": model["model_id"], "model_type": model["model_type"], "is_mla": is_mla,
            "num_hidden_layers": layers, "page_size_tokens": page_size,
            "kv_bytes_per_page": bytes_per_page, "kv_bytes_per_token": int(model["kv_bytes_per_token"]),
            "descriptor_layout": "one_per_layer" if is_mla else "K_and_V_per_layer",
            "descriptor_count": descriptor_count, "descriptor_bytes_per_page": descriptor_bytes_per_page,
            "maximum_page_count": max_pages, "knots": knots,
        })
    ids = [row["model_id"] for row in output]
    checks = {
        "six_models": len(ids) == 6 and len(set(ids)) == 6,
        "deepseek_12_knots": len(next(row for row in output if row["model_id"] == "deepseek-v2-lite")["knots"]) == 12,
        "five_non_mla_24_knots": all(len(row["knots"]) == 24 for row in output if row["model_id"] != "deepseek-v2-lite"),
        "total_per_topology_132": sum(len(row["knots"]) for row in output) == 132,
        "payload_exact": all(knot["payload_bytes"] == knot["descriptor_bytes"] * row["descriptor_count"] for row in output for knot in row["knots"]),
    }
    if not all(checks.values()): raise RuntimeError({"model_layout_checks": checks})
    return output


def layout_by_id(model_id: str, spec: dict | None = None) -> dict:
    rows = [row for row in model_layouts(spec) if row["model_id"] == model_id]
    if len(rows) != 1: raise RuntimeError(f"unknown model layout: {model_id}")
    return rows[0]


def iteration_counts(payload_bytes: int, spec: dict | None = None) -> tuple[int, int]:
    spec = spec or contract(); measurement = spec["measurement_contract"]
    timed = int(measurement["target_bytes_per_timed_block"]) // int(payload_bytes)
    timed = max(int(measurement["timed_iterations_min"]), min(int(measurement["timed_iterations_max"]), timed))
    warmup = timed // 10
    warmup = max(int(measurement["warmup_iterations_min"]), min(int(measurement["warmup_iterations_max"]), warmup))
    return warmup, timed


def validate_inventory(inventory: dict, spec: dict | None = None) -> dict:
    spec = spec or contract(); errors=[]
    if inventory.get("schema_version") != "phase51-topology-inventory-v1": errors.append("schema_version")
    for field in ("created_at_utc", "created_by", "classification_source", "fabric_notes"):
        if _placeholder(inventory.get(field)): errors.append(f"placeholder:{field}")
    if inventory.get("classification_frozen_before_measurement") is not True: errors.append("classification_not_frozen")
    if inventory.get("classification_not_inferred_from_benchmark") is not True: errors.append("classification_may_be_posthoc")
    source = str(inventory.get("classification_source", "")).lower()
    if any(word in source for word in ("benchmark", "latency", "bandwidth", "speed test")): errors.append("classification_source_uses_speed")
    placements = inventory.get("placements") if isinstance(inventory.get("placements"), list) else []
    placement_ids=[row.get("placement_id") for row in placements]
    if len(set(placement_ids))!=len(placement_ids): errors.append("placement_ids_not_unique")
    expected = {(level, replica) for level in ("L1","L2","L3") for replica in range(int(spec["minimum_placement_replicas_per_model_topology"]))}
    actual = Counter((row.get("topology_level"), row.get("replica_id")) for row in placements)
    if set(actual) != expected or any(count != 1 for count in actual.values()): errors.append({"placement_matrix": dict(actual), "expected": sorted(expected)})
    signatures = defaultdict(list); normalized=[]
    for placement in placements:
        level = placement.get("topology_level"); replica = placement.get("replica_id"); endpoints = placement.get("endpoints") if isinstance(placement.get("endpoints"), list) else []
        if _placeholder(placement.get("placement_id")) or _placeholder(placement.get("evidence")): errors.append(f"placement_identity:{level}:{replica}")
        if len(endpoints) != 2 or [row.get("rank") for row in endpoints] != [0,1]: errors.append(f"endpoint_ranks:{level}:{replica}")
        hosts = [row.get("host") for row in endpoints]; racks=[row.get("rack_id") for row in endpoints]; domains=[row.get("network_domain") for row in endpoints]
        if level == "L1" and (len(set(hosts)) != 1 or len(set(racks)) != 1): errors.append(f"l1_identity:{replica}")
        if level == "L2" and (len(set(hosts)) != 2 or len(set(racks)) != 1): errors.append(f"l2_identity:{replica}")
        if level == "L3" and (len(set(hosts)) != 2 or len(set(racks)) != 2): errors.append(f"l3_identity:{replica}")
        if len(set(domains)) != 1: errors.append(f"network_domain:{level}:{replica}")
        endpoint_keys=[]; normalized_endpoints=[]
        for endpoint in endpoints:
            aliases=endpoint.get("host_aliases") if isinstance(endpoint.get("host_aliases"),list) else []
            for field in ("host","transfer_hostname","rack_id","network_domain","ib_device"):
                if _placeholder(endpoint.get(field)): errors.append(f"endpoint_field:{level}:{replica}:{endpoint.get('rank')}:{field}")
            if endpoint.get("host") not in aliases: errors.append(f"host_alias:{level}:{replica}:{endpoint.get('rank')}")
            gpu=endpoint.get("physical_gpu")
            if not isinstance(gpu,int) or gpu<0: errors.append(f"gpu:{level}:{replica}:{endpoint.get('rank')}")
            endpoint_keys.append((endpoint.get("host"),gpu,endpoint.get("ib_device")))
            normalized_endpoints.append({"rank":endpoint.get("rank"),"host":endpoint.get("host"),"host_aliases":sorted(set(aliases)),"transfer_hostname":endpoint.get("transfer_hostname"),"rack_id":endpoint.get("rack_id"),"network_domain":endpoint.get("network_domain"),"physical_gpu":gpu,"ib_device":endpoint.get("ib_device")})
        if len(set(endpoint_keys)) != 2: errors.append(f"endpoint_reuse_inside_placement:{level}:{replica}")
        signatures[level].append(tuple(endpoint_keys)); normalized.append({"topology_level":level,"replica_id":replica,"placement_id":placement.get("placement_id"),"classification_evidence":placement.get("evidence"),"endpoints":normalized_endpoints})
    if any(len(set(values)) != len(values) for values in signatures.values()): errors.append("replica_endpoint_signatures_not_distinct")
    if errors: raise RuntimeError({"invalid_phase51_inventory": errors})
    return {"ok":True,"placements":len(normalized),"normalized_placements":sorted(normalized,key=lambda row:(row["topology_level"],row["replica_id"]))}


def expand_plan(inventory: dict, inventory_sha256: str, generated_at_utc: str, workflow_commit: str, spec: dict | None = None) -> dict:
    spec = spec or contract(); audit=validate_inventory(inventory,spec); layouts=model_layouts(spec); measurements=[]
    for layout in layouts:
        for placement in audit["normalized_placements"]:
            base={"measurement_id":f"{layout['model_id']}__{placement['topology_level'].lower()}__r{placement['replica_id']}","model_id":layout["model_id"],"topology_level":placement["topology_level"],"replica_id":placement["replica_id"],"placement_id":placement["placement_id"],"classification_evidence":placement["classification_evidence"],"world_size":2,"op":"sglang_mooncake_batch_transfer_sync","ranks":placement["endpoints"],"model_layout_sha256":canonical_sha(layout)}
            measurements.append({**base,"measurement_sha256":canonical_sha(base)})
    base={"schema_version":"phase51-topology-plan-v1","workflow_commit":workflow_commit,"generated_at_utc":generated_at_utc,"inventory_sha256":inventory_sha256,"inventory_schema_version":inventory["schema_version"],"inventory_metadata":{"created_at_utc":inventory["created_at_utc"],"created_by":inventory["created_by"],"classification_source":inventory["classification_source"],"classification_frozen_before_measurement":inventory["classification_frozen_before_measurement"],"classification_not_inferred_from_benchmark":inventory["classification_not_inferred_from_benchmark"],"fabric_notes":inventory["fabric_notes"]},"model_layouts_sha256":canonical_sha(layouts),"measurements":measurements}
    return {**base,"plan_sha256":canonical_sha(base)}


def validate_plan(plan: dict, spec: dict | None = None) -> dict:
    spec = spec or contract(); errors=[]
    if plan.get("schema_version") != "phase51-topology-plan-v1": errors.append("schema_version")
    metadata=plan.get("inventory_metadata") if isinstance(plan.get("inventory_metadata"),dict) else {}
    if metadata.get("classification_frozen_before_measurement") is not True or metadata.get("classification_not_inferred_from_benchmark") is not True:errors.append("inventory_metadata_freeze")
    if any(_placeholder(metadata.get(field)) for field in ("created_at_utc","created_by","classification_source","fabric_notes")):errors.append("inventory_metadata_fields")
    base={key:value for key,value in plan.items() if key!="plan_sha256"}
    if plan.get("plan_sha256") != canonical_sha(base): errors.append("plan_sha256")
    layouts=model_layouts(spec); layout_map={row["model_id"]:row for row in layouts}
    if plan.get("model_layouts_sha256") != canonical_sha(layouts): errors.append("model_layouts_sha256")
    measurements=plan.get("measurements") if isinstance(plan.get("measurements"),list) else []
    expected={(model,"L1",replica) for model in layout_map for replica in (0,1)}|{(model,"L2",replica) for model in layout_map for replica in (0,1)}|{(model,"L3",replica) for model in layout_map for replica in (0,1)}
    actual=Counter((row.get("model_id"),row.get("topology_level"),row.get("replica_id")) for row in measurements)
    if set(actual)!=expected or any(count!=1 for count in actual.values()): errors.append({"measurement_matrix":dict(actual),"expected_count":36})
    for measurement in measurements:
        expected_id=f"{measurement.get('model_id')}__{str(measurement.get('topology_level','')).lower()}__r{measurement.get('replica_id')}"
        if measurement.get("measurement_id")!=expected_id:errors.append(f"measurement_id:{measurement.get('measurement_id')}")
        if measurement.get("world_size")!=2 or measurement.get("op")!="sglang_mooncake_batch_transfer_sync":errors.append(f"operation:{measurement.get('measurement_id')}")
        no_sha={key:value for key,value in measurement.items() if key!="measurement_sha256"}
        if measurement.get("measurement_sha256")!=canonical_sha(no_sha): errors.append(f"measurement_sha:{measurement.get('measurement_id')}")
        layout=layout_map.get(measurement.get("model_id"))
        if layout and measurement.get("model_layout_sha256")!=canonical_sha(layout): errors.append(f"layout_sha:{measurement.get('measurement_id')}")
        ranks=measurement.get("ranks") if isinstance(measurement.get("ranks"),list) else []
        if len(ranks)!=2 or [row.get("rank") for row in ranks]!=[0,1]: errors.append(f"ranks:{measurement.get('measurement_id')}")
        hosts={row.get("host") for row in ranks}; racks={row.get("rack_id") for row in ranks}
        level=measurement.get("topology_level")
        if level=="L1" and (len(hosts)!=1 or len(racks)!=1): errors.append(f"l1:{measurement.get('measurement_id')}")
        if level=="L2" and (len(hosts)!=2 or len(racks)!=1): errors.append(f"l2:{measurement.get('measurement_id')}")
        if level=="L3" and (len(hosts)!=2 or len(racks)!=2): errors.append(f"l3:{measurement.get('measurement_id')}")
    if errors: raise RuntimeError({"invalid_phase51_plan":errors})
    return {"ok":True,"plan_sha256":plan["plan_sha256"],"measurements":len(measurements),"models":len(layouts),"curves":len(layouts)*3,"curve_knots":sum(len(row["knots"]) for row in layouts)*3}


def measurement_by_id(plan: dict, measurement_id: str) -> dict:
    rows=[row for row in plan["measurements"] if row["measurement_id"]==measurement_id]
    if len(rows)!=1: raise RuntimeError(f"unknown measurement_id: {measurement_id}")
    return rows[0]
