#!/usr/bin/env python3
"""Phase60 topology, payload-pair and measurement-plan contracts."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LAYOUTS_PATH = ROOT / "experiment-results/phase51_pd_l1_l3_physical_curve_library/contracts/model_transfer_layouts.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def contract() -> dict:
    return load_json(HERE / "experiment.json")


def _placeholder(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip() or value.startswith("REPLACE_")


def selected_layouts(spec: dict | None = None) -> list[dict]:
    spec = spec or contract()
    source = load_json(LAYOUTS_PATH)["layouts"]
    selected = [row for row in source if row["model_id"] in spec["selected_models"]]
    if [row["model_id"] for row in selected] != spec["selected_models"]:
        raise RuntimeError({"selected_layouts": [row["model_id"] for row in selected]})
    return selected


def layout_by_id(model_id: str) -> dict:
    rows = [row for row in selected_layouts() if row["model_id"] == model_id]
    if len(rows) != 1:
        raise RuntimeError(f"unknown Phase60 model: {model_id}")
    return rows[0]


def payload_pairs(model_id: str, role: str = "development_pairs", spec: dict | None = None) -> list[dict]:
    spec = spec or contract(); layout = layout_by_id(model_id)
    page_map = {int(row["page_count"]): row for row in layout["knots"]}
    raw = spec["pair_grid_contract"][role][model_id]; output=[]
    for index, pair in enumerate(raw):
        if not isinstance(pair, list) or len(pair) != 2 or any(int(value) not in page_map for value in pair):
            raise RuntimeError({"invalid_pair": model_id, "role": role, "pair": pair})
        a, b = (page_map[int(value)] for value in pair)
        output.append({
            "pair_id": f"{model_id}__{role[:3]}_{index:02d}__p{a['page_count']}_p{b['page_count']}",
            "page_count0": int(a["page_count"]), "page_count1": int(b["page_count"]),
            "payload_bytes0": int(a["payload_bytes"]), "payload_bytes1": int(b["payload_bytes"]),
            "descriptor_bytes0": int(a["descriptor_bytes"]), "descriptor_bytes1": int(b["descriptor_bytes"]),
        })
    if len({row["pair_id"] for row in output}) != len(output):
        raise RuntimeError("duplicate payload pair ids")
    return output


def validate_pair_contract(spec: dict | None = None) -> dict:
    spec = spec or contract(); counts={}
    for model_id in spec["selected_models"]:
        dev = payload_pairs(model_id, "development_pairs", spec)
        blind = payload_pairs(model_id, "reserved_future_blind_pairs", spec)
        dev_pairs={(r["page_count0"],r["page_count1"]) for r in dev}; blind_pairs={(r["page_count0"],r["page_count1"]) for r in blind}
        if dev_pairs & blind_pairs:
            raise RuntimeError({"development_blind_overlap": model_id, "pairs": sorted(dev_pairs & blind_pairs)})
        counts[model_id] = {"development": len(dev), "reserved_future_blind": len(blind)}
    if sum(row["development"] for row in counts.values()) != 20:
        raise RuntimeError({"development_pair_count": counts})
    return {"ok": True, "counts": counts, "development_sha256": canonical_sha({m: payload_pairs(m) for m in spec["selected_models"]}), "reserved_sha256": canonical_sha({m: payload_pairs(m,"reserved_future_blind_pairs") for m in spec["selected_models"]})}


def iteration_counts(total_payload_bytes: int, spec: dict | None = None) -> tuple[int, int]:
    measurement = (spec or contract())["measurement_contract"]
    timed = int(measurement["target_bytes_per_mode_block"]) // max(int(total_payload_bytes), 1)
    timed = max(int(measurement["timed_iterations_min"]), min(int(measurement["timed_iterations_max"]), timed))
    warmup = max(int(measurement["warmup_iterations_min"]), min(int(measurement["warmup_iterations_max"]), timed // 10))
    return warmup, timed


def validate_inventory(inventory: dict, spec: dict | None = None) -> dict:
    spec = spec or contract(); errors=[]
    if inventory.get("schema_version") != "phase60-topology-inventory-v1": errors.append("schema_version")
    for field in ("created_at_utc","created_by","classification_source","fabric_notes"):
        if _placeholder(inventory.get(field)): errors.append(f"placeholder:{field}")
    if inventory.get("classification_frozen_before_measurement") is not True: errors.append("classification_not_frozen")
    if inventory.get("classification_not_inferred_from_benchmark") is not True: errors.append("classification_may_be_posthoc")
    if any(word in str(inventory.get("classification_source","")).lower() for word in ("benchmark","latency","bandwidth","speed test")): errors.append("classification_source_uses_speed")
    resources=inventory.get("resource_allocation_contract") if isinstance(inventory.get("resource_allocation_contract"),dict) else {}
    expected_resources={"endpoint_slots_are_gpu_slots_not_nodes":True,"simultaneous_world_size_per_shard":3,"simultaneous_gpu_processes_per_shard":3,"simultaneous_nodes_per_shard":{"L1":1,"L2":2,"L3":2},"p1d2_uses_slots":["A0","B0","B1"],"p2d1_uses_slots":["A0","A1","B0"],"fourth_slot_is_not_launched_in_same_shard":True,"all_placements_and_replicas_may_run_sequentially":True,"replicas_may_reuse_same_node_pair_with_distinct_gpu_tuples":True,"four_node_allocation_required":False}
    if resources!=expected_resources:errors.append({"resource_allocation_contract":resources,"expected":expected_resources})
    placements=inventory.get("placements") if isinstance(inventory.get("placements"),list) else []
    expected={(level,replica) for level in ("L1","L2","L3") for replica in (0,1)}
    actual=Counter((row.get("topology_level"),row.get("replica_id")) for row in placements)
    if set(actual)!=expected or any(value!=1 for value in actual.values()): errors.append({"placement_matrix":dict(actual)})
    ids=[row.get("placement_id") for row in placements]
    if len(set(ids))!=len(ids): errors.append("placement_ids_not_unique")
    signatures=defaultdict(list); normalized=[]
    for placement in placements:
        level=placement.get("topology_level");replica=placement.get("replica_id")
        if _placeholder(placement.get("placement_id")) or _placeholder(placement.get("evidence")): errors.append(f"placement_identity:{level}:{replica}")
        sides=placement.get("sides") if isinstance(placement.get("sides"),dict) else {}
        if set(sides)!={"A","B"}: errors.append(f"sides:{level}:{replica}")
        endpoints=[]; normalized_sides={}
        for side in ("A","B"):
            rows=sides.get(side) if isinstance(sides.get(side),list) else []
            if len(rows)!=2: errors.append(f"side_endpoint_count:{level}:{replica}:{side}")
            normalized_rows=[]
            for slot,endpoint in enumerate(rows):
                aliases=endpoint.get("host_aliases") if isinstance(endpoint.get("host_aliases"),list) else []
                for field in ("host","transfer_hostname","rack_id","network_domain","ib_device"):
                    if _placeholder(endpoint.get(field)): errors.append(f"endpoint_field:{level}:{replica}:{side}{slot}:{field}")
                if endpoint.get("host") not in aliases: errors.append(f"host_alias:{level}:{replica}:{side}{slot}")
                gpu=endpoint.get("physical_gpu")
                if not isinstance(gpu,int) or gpu<0: errors.append(f"gpu:{level}:{replica}:{side}{slot}")
                normalized_endpoint={"slot":f"{side}{slot}","side":side,"host":endpoint.get("host"),"host_aliases":sorted(set(aliases)),"transfer_hostname":endpoint.get("transfer_hostname"),"rack_id":endpoint.get("rack_id"),"network_domain":endpoint.get("network_domain"),"physical_gpu":gpu,"ib_device":endpoint.get("ib_device")}
                normalized_rows.append(normalized_endpoint);endpoints.append(normalized_endpoint)
            normalized_sides[side]=normalized_rows
        keys=[(r["host"],r["physical_gpu"],r["ib_device"]) for r in endpoints]
        if len(set(keys))!=4: errors.append(f"endpoint_slots_not_distinct:{level}:{replica}")
        hosts_a={r["host"] for r in normalized_sides.get("A",[])};hosts_b={r["host"] for r in normalized_sides.get("B",[])}
        racks_a={r["rack_id"] for r in normalized_sides.get("A",[])};racks_b={r["rack_id"] for r in normalized_sides.get("B",[])}
        domains={r["network_domain"] for r in endpoints}
        if level=="L1" and (len(hosts_a|hosts_b)!=1 or len(racks_a|racks_b)!=1): errors.append(f"l1_identity:{replica}")
        if level=="L2" and (len(hosts_a)!=1 or len(hosts_b)!=1 or hosts_a==hosts_b or len(racks_a|racks_b)!=1): errors.append(f"l2_identity:{replica}")
        if level=="L3" and (len(hosts_a)!=1 or len(hosts_b)!=1 or hosts_a==hosts_b or len(racks_a)!=1 or len(racks_b)!=1 or racks_a==racks_b): errors.append(f"l3_identity:{replica}")
        if len(domains)!=1: errors.append(f"network_domain:{level}:{replica}")
        signatures[level].append(tuple(keys));normalized.append({"topology_level":level,"replica_id":replica,"placement_id":placement.get("placement_id"),"classification_evidence":placement.get("evidence"),"sides":normalized_sides})
    if any(len(set(values))!=len(values) for values in signatures.values()): errors.append("replica_endpoint_signatures_not_distinct")
    if errors: raise RuntimeError({"invalid_phase60_inventory":errors})
    return {"ok":True,"placements":len(normalized),"max_simultaneous_nodes_per_shard":2,"simultaneous_gpu_processes_per_shard":3,"normalized_placements":sorted(normalized,key=lambda row:(row["topology_level"],row["replica_id"]))}


def _measurement_ranks(placement: dict, configuration: str) -> list[dict]:
    sides=placement["sides"]
    slots=[sides["A"][0],sides["B"][0],sides["B"][1]] if configuration=="P1D2" else [sides["A"][0],sides["A"][1],sides["B"][0]]
    roles=["P0","D0","D1"] if configuration=="P1D2" else ["P0","P1","D0"]
    return [{**endpoint,"rank":rank,"role":roles[rank]} for rank,endpoint in enumerate(slots)]


def expand_plan(inventory: dict, inventory_sha256: str, generated_at_utc: str, workflow_commit: str, spec: dict | None = None) -> dict:
    spec=spec or contract();audit=validate_inventory(inventory,spec);pairs=validate_pair_contract(spec);measurements=[]
    for model_id in spec["selected_models"]:
        layout=layout_by_id(model_id)
        for configuration in spec["research_scope"]["fixed_configurations"]:
            for placement in audit["normalized_placements"]:
                ranks=_measurement_ranks(placement,configuration)
                base={"measurement_id":f"{model_id}__{configuration.lower()}__{placement['topology_level'].lower()}__r{placement['replica_id']}","model_id":model_id,"configuration":configuration,"topology_level":placement["topology_level"],"replica_id":placement["replica_id"],"placement_id":placement["placement_id"],"classification_evidence":placement["classification_evidence"],"world_size":3,"op":"sglang_mooncake_two_flow_batch_transfer_sync","ranks":ranks,"model_layout_sha256":canonical_sha(layout),"development_pairs_sha256":canonical_sha(payload_pairs(model_id))}
                measurements.append({**base,"measurement_sha256":canonical_sha(base)})
    base={"schema_version":"phase60-topology-plan-v1","workflow_commit":workflow_commit,"generated_at_utc":generated_at_utc,"inventory_sha256":inventory_sha256,"inventory_schema_version":inventory["schema_version"],"inventory_metadata":{key:inventory[key] for key in ("created_at_utc","created_by","classification_source","classification_frozen_before_measurement","classification_not_inferred_from_benchmark","fabric_notes","resource_allocation_contract")},"selected_layouts_sha256":canonical_sha(selected_layouts(spec)),"development_pairs_sha256":pairs["development_sha256"],"reserved_future_blind_pairs_sha256":pairs["reserved_sha256"],"measurements":measurements}
    return {**base,"plan_sha256":canonical_sha(base)}


def validate_plan(plan: dict, spec: dict | None = None) -> dict:
    spec=spec or contract();errors=[];pairs=validate_pair_contract(spec)
    if plan.get("schema_version")!="phase60-topology-plan-v1":errors.append("schema_version")
    base={key:value for key,value in plan.items() if key!="plan_sha256"}
    if plan.get("plan_sha256")!=canonical_sha(base):errors.append("plan_sha256")
    metadata=plan.get("inventory_metadata") if isinstance(plan.get("inventory_metadata"),dict) else {}
    if metadata.get("classification_frozen_before_measurement") is not True or metadata.get("classification_not_inferred_from_benchmark") is not True:errors.append("inventory_freeze")
    resources=metadata.get("resource_allocation_contract") if isinstance(metadata.get("resource_allocation_contract"),dict) else {}
    if resources.get("four_node_allocation_required") is not False or resources.get("simultaneous_nodes_per_shard")!={"L1":1,"L2":2,"L3":2} or resources.get("simultaneous_gpu_processes_per_shard")!=3:errors.append("resource_allocation_contract")
    if plan.get("selected_layouts_sha256")!=canonical_sha(selected_layouts(spec)) or plan.get("development_pairs_sha256")!=pairs["development_sha256"] or plan.get("reserved_future_blind_pairs_sha256")!=pairs["reserved_sha256"]:errors.append("layout_or_pair_sha")
    measurements=plan.get("measurements") if isinstance(plan.get("measurements"),list) else []
    expected={(model,config,level,replica) for model in spec["selected_models"] for config in spec["research_scope"]["fixed_configurations"] for level in ("L1","L2","L3") for replica in (0,1)}
    actual=Counter((r.get("model_id"),r.get("configuration"),r.get("topology_level"),r.get("replica_id")) for r in measurements)
    if set(actual)!=expected or any(v!=1 for v in actual.values()):errors.append({"measurement_matrix":dict(actual)})
    for row in measurements:
        expected_id=f"{row.get('model_id')}__{str(row.get('configuration','')).lower()}__{str(row.get('topology_level','')).lower()}__r{row.get('replica_id')}"
        if row.get("measurement_id")!=expected_id or row.get("world_size")!=3 or row.get("op")!="sglang_mooncake_two_flow_batch_transfer_sync":errors.append(f"identity:{row.get('measurement_id')}")
        no_sha={key:value for key,value in row.items() if key!="measurement_sha256"}
        if row.get("measurement_sha256")!=canonical_sha(no_sha):errors.append(f"measurement_sha:{row.get('measurement_id')}")
        if row.get("model_layout_sha256")!=canonical_sha(layout_by_id(row.get("model_id"))) or row.get("development_pairs_sha256")!=canonical_sha(payload_pairs(row.get("model_id"))):errors.append(f"input_sha:{row.get('measurement_id')}")
        ranks=row.get("ranks") if isinstance(row.get("ranks"),list) else []
        expected_roles=["P0","D0","D1"] if row.get("configuration")=="P1D2" else ["P0","P1","D0"]
        if len(ranks)!=3 or [r.get("rank") for r in ranks]!=[0,1,2] or [r.get("role") for r in ranks]!=expected_roles or len({(r.get("host"),r.get("physical_gpu")) for r in ranks})!=3:errors.append(f"ranks:{row.get('measurement_id')}")
        a=[r for r in ranks if r.get("side")=="A"];b=[r for r in ranks if r.get("side")=="B"]
        if row.get("configuration")=="P1D2" and (len(a)!=1 or len(b)!=2):errors.append(f"p1d2_sides:{row.get('measurement_id')}")
        if row.get("configuration")=="P2D1" and (len(a)!=2 or len(b)!=1):errors.append(f"p2d1_sides:{row.get('measurement_id')}")
        expected_nodes=1 if row.get("topology_level")=="L1" else 2
        if len({r.get("host") for r in ranks})!=expected_nodes:errors.append(f"simultaneous_node_count:{row.get('measurement_id')}")
    if errors:raise RuntimeError({"invalid_phase60_plan":errors})
    return {"ok":True,"plan_sha256":plan["plan_sha256"],"measurements":len(measurements),"world_size_per_shard":3,"max_simultaneous_nodes_per_shard":2,"development_points":int(spec["expected_development_points"]),"replica_points":int(spec["expected_replica_points"])}


def measurement_by_id(plan: dict, measurement_id: str) -> dict:
    rows=[row for row in plan["measurements"] if row["measurement_id"]==measurement_id]
    if len(rows)!=1:raise RuntimeError(f"unknown measurement_id: {measurement_id}")
    return rows[0]
