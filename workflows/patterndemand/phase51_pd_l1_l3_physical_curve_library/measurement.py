#!/usr/bin/env python3
"""Validate Git-external Phase51 raw records and build compact physical curves."""
from __future__ import annotations

import json, math, statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from contracts import file_sha, iteration_counts, layout_by_id, measurement_by_id, validate_plan

DIRECTIONS = ("rank0_to_rank1", "rank1_to_rank0")


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def _cv(values: list[float]) -> float:
    return statistics.stdev(values) / statistics.mean(values) if len(values) > 1 and statistics.mean(values) else 0.0


def _raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(path for path in raw_dir.rglob("*") if path.is_file())
    unexpected = [str(path.relative_to(raw_dir)) for path in files if path.suffix != ".jsonl"]
    if unexpected:
        raise RuntimeError({"unexpected_raw_files": unexpected})
    return files


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise RuntimeError({"blank_raw_line": str(path), "line": number})
        try:row=json.loads(line)
        except json.JSONDecodeError as exc:raise RuntimeError({"invalid_jsonl":str(path),"line":number}) from exc
        if not isinstance(row,dict):raise RuntimeError({"raw_row_not_object":str(path),"line":number})
        rows.append(row)
    return rows


def validate_raw(plan: dict, raw_dir: Path, *, require_complete: bool) -> dict[str, Any]:
    audit=validate_plan(plan);raw_dir=raw_dir.expanduser().resolve();spec=__import__("contracts").contract();mc=spec["measurement_contract"]
    if not raw_dir.is_dir():raise RuntimeError(f"raw directory missing: {raw_dir}")
    by_measurement:dict[str,dict[int,list[dict[str,Any]]]]=defaultdict(dict);manifest=[]
    for path in _raw_files(raw_dir):
        relative=path.relative_to(raw_dir);parts=relative.parts
        if len(parts)!=2 or not parts[1].startswith("repeat_") or not parts[1].endswith(".jsonl"):
            raise RuntimeError({"invalid_raw_path":str(relative)})
        measurement_id=parts[0]
        try:repeat_id=int(parts[1][7:-6])
        except ValueError as exc:raise RuntimeError({"invalid_repeat_path":str(relative)}) from exc
        if parts[1]!=f"repeat_{repeat_id:02d}.jsonl":raise RuntimeError({"noncanonical_repeat_path":str(relative)})
        if repeat_id in by_measurement[measurement_id]:raise RuntimeError({"duplicate_repeat":str(relative)})
        rows=_read_jsonl(path);by_measurement[measurement_id][repeat_id]=rows
        manifest.append({"path":str(relative),"sha256":file_sha(path),"bytes":path.stat().st_size,"records":len(rows)})
    expected_ids={row["measurement_id"] for row in plan["measurements"]};unknown=sorted(set(by_measurement)-expected_ids)
    if unknown:raise RuntimeError({"unknown_measurements":unknown})
    missing=[];needs_extra=[];final_runtime_variance=[];measurements=[];all_records=0
    for measurement_id in sorted(expected_ids):
        measurement=measurement_by_id(plan,measurement_id);layout=layout_by_id(measurement["model_id"]);repeats=by_measurement.get(measurement_id,{})
        repeat_ids=sorted(repeats);allowed=[int(mc["minimum_independent_repeats"]),int(mc["minimum_independent_repeats"])+int(mc["extra_repeats_per_round"]),int(mc["maximum_independent_repeats"])]
        if repeat_ids and repeat_ids!=list(range(len(repeat_ids))):raise RuntimeError({"noncontiguous_repeats":measurement_id,"actual":repeat_ids})
        if len(repeat_ids)>allowed[-1]:raise RuntimeError({"too_many_repeats":measurement_id,"count":len(repeat_ids),"maximum":allowed[-1]})
        if len(repeat_ids)<allowed[0]:missing.append({"measurement_id":measurement_id,"have":len(repeat_ids),"need":allowed[0]-len(repeat_ids),"target_total":allowed[0]})
        elif len(repeat_ids) not in allowed:
            target=next(value for value in allowed if value>len(repeat_ids));missing.append({"measurement_id":measurement_id,"have":len(repeat_ids),"need":target-len(repeat_ids),"target_total":target})
        medians:dict[tuple[int,str],list[float]]=defaultdict(list)
        expected_keys={(int(k["page_count"]),d) for k in layout["knots"] for d in DIRECTIONS}
        knot_map={int(k["page_count"]):k for k in layout["knots"]}
        for repeat_id,rows in sorted(repeats.items()):
            all_records+=len(rows);actual_keys=[]
            for row in rows:
                page=int(row.get("page_count",-1));direction=row.get("direction");actual_keys.append((page,direction));knot=knot_map.get(page)
                samples=row.get("latency_samples_us") if isinstance(row.get("latency_samples_us"),list) else []
                warmup,timed=iteration_counts(int(knot["payload_bytes"])) if knot else (-1,-1)
                endpoints=row.get("runtime_endpoints") if isinstance(row.get("runtime_endpoints"),list) else []
                expected_endpoints=measurement["ranks"]
                endpoint_ok=len(endpoints)==2 and all(
                    int(endpoints[i].get("rank",-1))==i
                    and endpoints[i].get("expected_host")==expected_endpoints[i]["host"]
                    and int(endpoints[i].get("physical_gpu",-1))==int(expected_endpoints[i]["physical_gpu"])
                    and endpoints[i].get("ib_device")==expected_endpoints[i]["ib_device"]
                    and endpoints[i].get("mooncake_protocol")=="rdma"
                    and endpoints[i].get("with_nvidia_peermem")=="0"
                    and isinstance(endpoints[i].get("gpu_name"),str) and bool(endpoints[i].get("gpu_name"))
                    and isinstance(endpoints[i].get("torch"),str) and bool(endpoints[i].get("torch"))
                    and isinstance(endpoints[i].get("cuda"),str) and bool(endpoints[i].get("cuda"))
                    for i in range(2)
                )
                checks={
                    "schema":row.get("schema_version")=="phase51-mooncake-raw-v1","workflow":row.get("workflow_commit")==plan["workflow_commit"],
                    "plan":row.get("plan_sha256")==audit["plan_sha256"],"measurement":row.get("measurement_id")==measurement_id and row.get("measurement_sha256")==measurement["measurement_sha256"],
                    "identity":row.get("model_id")==measurement["model_id"] and row.get("topology_level")==measurement["topology_level"] and row.get("replica_id")==measurement["replica_id"] and row.get("placement_id")==measurement["placement_id"] and row.get("repeat_id")==repeat_id,
                    "direction":direction in DIRECTIONS and row.get("sender_rank")==DIRECTIONS.index(direction) and row.get("receiver_rank")==1-DIRECTIONS.index(direction),
                    "layout":bool(knot) and int(row.get("page_size_tokens",-1))==layout["page_size_tokens"] and int(row.get("payload_bytes",-1))==knot["payload_bytes"] and row.get("descriptor_layout")==layout["descriptor_layout"] and int(row.get("descriptor_count",-1))==layout["descriptor_count"] and int(row.get("descriptor_bytes",-1))==knot["descriptor_bytes"],
                    "operation":row.get("op")=="MooncakeTransferEngine.batch_transfer_sync" and row.get("transport")=="rdma",
                    "iterations":row.get("warmup_iterations")==warmup and row.get("timed_iterations")==timed and len(samples)==timed,
                    "samples":bool(samples) and all(isinstance(v,(int,float)) and math.isfinite(float(v)) and float(v)>0 for v in samples),
                    "median":bool(samples) and _close(float(row.get("latency_us",{}).get("median",-1)),statistics.median(float(v) for v in samples)),
                    "transfer":row.get("data_validation_pass") is True and row.get("return_codes_all_zero") is True,
                    "endpoints":endpoint_ok,
                    "timestamp":isinstance(row.get("timestamp_utc"),str) and row["timestamp_utc"]>=plan["generated_at_utc"],
                }
                if not all(checks.values()):raise RuntimeError({"invalid_raw_record":measurement_id,"repeat":repeat_id,"page":page,"direction":direction,"checks":checks})
                medians[(page,direction)].append(statistics.median(float(v) for v in samples))
            if len(actual_keys)!=len(expected_keys) or set(actual_keys)!=expected_keys or len(set(actual_keys))!=len(actual_keys):
                raise RuntimeError({"raw_coverage":measurement_id,"repeat":repeat_id,"records":len(actual_keys),"expected":len(expected_keys)})
        cvs={f"pages_{page}__{direction}":_cv(values) for (page,direction),values in sorted(medians.items())}
        max_cv=max(cvs.values(),default=0.0);threshold=float(mc["repeat_median_cv_threshold"])
        if len(repeat_ids) in allowed and max_cv>threshold:
            if len(repeat_ids)<allowed[-1]:needs_extra.append({"measurement_id":measurement_id,"have":len(repeat_ids),"add":int(mc["extra_repeats_per_round"]),"max_cv":max_cv})
            else:final_runtime_variance.append({"measurement_id":measurement_id,"repeats":len(repeat_ids),"max_cv":max_cv})
        measurements.append({"measurement_id":measurement_id,"model_id":measurement["model_id"],"topology_level":measurement["topology_level"],"replica_id":measurement["replica_id"],"repeat_count":len(repeat_ids),"record_count":sum(len(v) for v in repeats.values()),"max_repeat_median_cv":max_cv,"repeat_median_cv":cvs})
    complete=not missing and not needs_extra
    if require_complete and not complete:raise RuntimeError({"raw_not_complete":True,"missing":missing,"needs_extra":needs_extra})
    return {"schema_version":"phase51-raw-audit-v1","raw_dir":str(raw_dir),"plan_sha256":audit["plan_sha256"],"complete":complete,"counts":{"files":len(manifest),"records":all_records,"measurements_with_data":len(by_measurement),"expected_measurements":36},"missing":missing,"needs_extra":needs_extra,"final_runtime_variance":final_runtime_variance,"files":manifest,"measurements":measurements,"records":by_measurement}


def build_curves(plan: dict, raw_audit: dict[str,Any]) -> dict[str,Any]:
    if not raw_audit["complete"]:raise RuntimeError("cannot build curves from incomplete raw")
    spec=__import__("contracts").contract();threshold=float(spec["measurement_contract"]["cross_replica_relative_spread_diagnostic_threshold"]);curves=[];spreads=[]
    for model_id in [row["model_id"] for row in __import__("contracts").model_layouts()]:
        layout=layout_by_id(model_id)
        for level in ("L1","L2","L3"):
            knots=[]
            for knot in layout["knots"]:
                replicas=[]
                for replica in (0,1):
                    mid=f"{model_id}__{level.lower()}__r{replica}";rows=raw_audit["records"][mid];directions={}
                    for direction in DIRECTIONS:
                        values=[]
                        for repeat_rows in rows.values():
                            row=next(r for r in repeat_rows if int(r["page_count"])==knot["page_count"] and r["direction"]==direction);values.append(float(row["latency_us"]["median"]))
                        directions[direction]={"repeat_medians_us":values,"median_across_repeats_us":statistics.median(values),"repeat_median_cv":_cv(values)}
                    official=max(value["median_across_repeats_us"] for value in directions.values());replicas.append({"replica_id":replica,"measurement_id":mid,"directions":directions,"slower_direction_latency_us":official})
                values=[r["slower_direction_latency_us"] for r in replicas];official=max(values);relative=(max(values)-min(values))/max(values) if max(values)>0 else 0.0
                knots.append({"page_count":knot["page_count"],"payload_bytes":knot["payload_bytes"],"descriptor_count":layout["descriptor_count"],"descriptor_bytes":knot["descriptor_bytes"],"replicas":replicas,"official_latency_us":official,"official_effective_gib_per_s":knot["payload_bytes"]/(official/1e6)/(1024**3),"cross_replica_relative_spread":relative})
                spreads.append({"model_id":model_id,"topology_level":level,"page_count":knot["page_count"],"payload_bytes":knot["payload_bytes"],"replica_0_us":values[0],"replica_1_us":values[1],"official_latency_us":official,"relative_spread":relative,"above_threshold":relative>threshold})
            curves.append({"curve_id":f"{model_id}__{level.lower()}","model_id":model_id,"topology_level":level,"page_size_tokens":layout["page_size_tokens"],"descriptor_layout":layout["descriptor_layout"],"descriptor_count":layout["descriptor_count"],"interpolation":"piecewise_linear_log2_payload_no_extrapolation","payload_support_bytes":[knots[0]["payload_bytes"],knots[-1]["payload_bytes"]],"knots":knots})
    if len(curves)!=18 or sum(len(row["knots"]) for row in curves)!=396:raise RuntimeError("curve cardinality contract failed")
    return {"schema_version":"phase51-pd-physical-curve-library-v1","workflow_commit":plan["workflow_commit"],"plan_sha256":plan["plan_sha256"],"curve_policy":"median repeats, slower direction, slower frozen placement replica","curves":curves,"spreads":spreads}
