#!/usr/bin/env python3
"""Validate external Phase60 raw and build compact composability evidence."""
from __future__ import annotations
import bisect,json,math,statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from contracts import contract,file_sha,iteration_counts,layout_by_id,load_json,measurement_by_id,payload_pairs,validate_plan

def _close(a:float,b:float)->bool:return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8)
def _cv(values:list[float])->float:return statistics.stdev(values)/statistics.mean(values) if len(values)>1 and statistics.mean(values) else 0.0
def _raw_files(raw_dir:Path)->list[Path]:
    files=sorted(path for path in raw_dir.rglob("*") if path.is_file());unexpected=[str(path.relative_to(raw_dir)) for path in files if path.suffix!=".jsonl"]
    if unexpected:raise RuntimeError({"unexpected_raw_files":unexpected})
    return files
def _read_jsonl(path:Path)->list[dict]:
    rows=[]
    for number,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():raise RuntimeError({"blank_raw_line":str(path),"line":number})
        try:row=json.loads(line)
        except json.JSONDecodeError as exc:raise RuntimeError({"invalid_jsonl":str(path),"line":number}) from exc
        if not isinstance(row,dict):raise RuntimeError({"raw_row_not_object":str(path),"line":number})
        rows.append(row)
    return rows
def _validate_mode(mode:dict,flow_ids:list[int],timed:int)->bool:
    samples=mode.get("flow_latency_samples_us") if isinstance(mode.get("flow_latency_samples_us"),dict) else {};summaries=mode.get("flow_latency_us") if isinstance(mode.get("flow_latency_us"),dict) else {};wave=mode.get("wave_latency_samples_us") if isinstance(mode.get("wave_latency_samples_us"),list) else [];skew=mode.get("sender_start_skew_samples_us") if isinstance(mode.get("sender_start_skew_samples_us"),list) else []
    if set(samples)!={str(v) for v in flow_ids} or set(summaries)!=set(samples) or len(wave)!=timed or len(skew)!=timed:return False
    if any(len(samples[str(v)])!=timed for v in flow_ids):return False
    all_values=[float(value) for values in samples.values() for value in values]+[float(v) for v in wave]+[float(v) for v in skew]
    if any(not math.isfinite(v) or v<0 for v in all_values) or any(v<=0 for values in samples.values() for v in values) or any(v<=0 for v in wave):return False
    for key,values in samples.items():
        if not _close(float(summaries[key].get("median",-1)),statistics.median(float(v) for v in values)):return False
    for i in range(timed):
        if not _close(float(wave[i]),max(float(samples[str(v)][i]) for v in flow_ids)):return False
    return _close(float(mode.get("wave_latency_us",{}).get("median",-1)),statistics.median(float(v) for v in wave)) and mode.get("return_codes_all_zero") is True and mode.get("data_validation_pass") is True
def validate_raw(plan:dict,raw_dir:Path,*,require_complete:bool)->dict[str,Any]:
    audit=validate_plan(plan);raw_dir=raw_dir.expanduser().resolve();spec=contract();mc=spec["measurement_contract"]
    if not raw_dir.is_dir():raise RuntimeError(f"raw directory missing: {raw_dir}")
    by_measurement:dict[str,dict[int,list[dict]]]=defaultdict(dict);manifest=[]
    for path in _raw_files(raw_dir):
        relative=path.relative_to(raw_dir);parts=relative.parts
        if len(parts)!=2 or not parts[1].startswith("repeat_") or not parts[1].endswith(".jsonl"):raise RuntimeError({"invalid_raw_path":str(relative)})
        try:repeat_id=int(parts[1][7:-6])
        except ValueError as exc:raise RuntimeError({"invalid_repeat_path":str(relative)}) from exc
        if parts[1]!=f"repeat_{repeat_id:02d}.jsonl" or repeat_id in by_measurement[parts[0]]:raise RuntimeError({"duplicate_or_noncanonical_repeat":str(relative)})
        rows=_read_jsonl(path);by_measurement[parts[0]][repeat_id]=rows;manifest.append({"path":str(relative),"sha256":file_sha(path),"bytes":path.stat().st_size,"records":len(rows)})
    expected_ids={row["measurement_id"] for row in plan["measurements"]};unknown=sorted(set(by_measurement)-expected_ids)
    if unknown:raise RuntimeError({"unknown_measurements":unknown})
    missing=[];needs_extra=[];final_runtime_variance=[];measurements=[];all_records=0
    for measurement_id in sorted(expected_ids):
        measurement=measurement_by_id(plan,measurement_id);layout=layout_by_id(measurement["model_id"]);expected_pairs=payload_pairs(measurement["model_id"]);pair_map={row["pair_id"]:row for row in expected_pairs};repeats=by_measurement.get(measurement_id,{});repeat_ids=sorted(repeats);allowed=[int(mc["minimum_independent_repeats"]),int(mc["minimum_independent_repeats"])+int(mc["extra_repeats_per_round"]),int(mc["maximum_independent_repeats"])]
        if repeat_ids and repeat_ids!=list(range(len(repeat_ids))):raise RuntimeError({"noncontiguous_repeats":measurement_id,"actual":repeat_ids})
        if len(repeat_ids)>allowed[-1]:raise RuntimeError({"too_many_repeats":measurement_id})
        if len(repeat_ids)<allowed[0]:missing.append({"measurement_id":measurement_id,"have":len(repeat_ids),"need":allowed[0]-len(repeat_ids),"target_total":allowed[0]})
        elif len(repeat_ids) not in allowed:target=next(value for value in allowed if value>len(repeat_ids));missing.append({"measurement_id":measurement_id,"have":len(repeat_ids),"need":target-len(repeat_ids),"target_total":target})
        medians:dict[str,list[float]]=defaultdict(list)
        for repeat_id,rows in sorted(repeats.items()):
            all_records+=len(rows);ids=[row.get("pair_id") for row in rows]
            if len(ids)!=len(expected_pairs) or set(ids)!=set(pair_map) or len(set(ids))!=len(ids):raise RuntimeError({"raw_pair_coverage":measurement_id,"repeat":repeat_id,"ids":ids})
            for row in rows:
                pair=pair_map[row["pair_id"]];warmup,timed=iteration_counts(pair["payload_bytes0"]+pair["payload_bytes1"]);endpoints=row.get("runtime_endpoints") if isinstance(row.get("runtime_endpoints"),list) else [];expected_endpoints=measurement["ranks"]
                endpoint_ok=len(endpoints)==3 and all(int(endpoints[i].get("rank",-1))==i and endpoints[i].get("role")==expected_endpoints[i]["role"] and endpoints[i].get("expected_host")==expected_endpoints[i]["host"] and int(endpoints[i].get("physical_gpu",-1))==int(expected_endpoints[i]["physical_gpu"]) and endpoints[i].get("ib_device")==expected_endpoints[i]["ib_device"] and endpoints[i].get("mooncake_protocol")=="rdma" and endpoints[i].get("with_nvidia_peermem")=="0" for i in range(3))
                checks={
                  "schema":row.get("schema_version")=="phase60-mooncake-multiflow-raw-v1","workflow":row.get("workflow_commit")==plan["workflow_commit"],"plan":row.get("plan_sha256")==audit["plan_sha256"],
                  "measurement":row.get("measurement_id")==measurement_id and row.get("measurement_sha256")==measurement["measurement_sha256"],
                  "identity":row.get("model_id")==measurement["model_id"] and row.get("configuration")==measurement["configuration"] and row.get("topology_level")==measurement["topology_level"] and row.get("replica_id")==measurement["replica_id"] and row.get("placement_id")==measurement["placement_id"] and row.get("repeat_id")==repeat_id,
                  "pair":all(int(row.get(key,-1))==int(pair[key]) for key in ("page_count0","page_count1","payload_bytes0","payload_bytes1","descriptor_bytes0","descriptor_bytes1")),
                  "layout":row.get("descriptor_layout")==layout["descriptor_layout"] and int(row.get("descriptor_count",-1))==int(layout["descriptor_count"]),
                  "operation":row.get("op")=="MooncakeTransferEngine.batch_transfer_sync" and row.get("transport")=="rdma" and row.get("wave_admission")=="gloo_barrier_then_two_synchronous_production_calls",
                  "mechanism":row.get("concurrency_mechanism")==(("one_shared_engine_two_threads") if measurement["configuration"]=="P1D2" else "two_sender_rank_engines"),
                  "iterations":row.get("warmup_iterations")==warmup and row.get("timed_iterations")==timed,
                  "solo0":_validate_mode(row.get("solo_flow0",{}),[0],timed),"solo1":_validate_mode(row.get("solo_flow1",{}),[1],timed),"concurrent":_validate_mode(row.get("concurrent_wave",{}),[0,1],timed),"endpoints":endpoint_ok,
                  "timestamp":isinstance(row.get("timestamp_utc"),str) and row["timestamp_utc"]>=plan["generated_at_utc"],
                }
                if not all(checks.values()):raise RuntimeError({"invalid_raw_record":measurement_id,"repeat":repeat_id,"pair":row.get("pair_id"),"checks":checks})
                medians[row["pair_id"]].append(float(row["concurrent_wave"]["wave_latency_us"]["median"]))
        cvs={pair_id:_cv(values) for pair_id,values in sorted(medians.items())};max_cv=max(cvs.values(),default=0.0);threshold=float(mc["repeat_median_cv_threshold"])
        if len(repeat_ids) in allowed and max_cv>threshold:
            if len(repeat_ids)<allowed[-1]:needs_extra.append({"measurement_id":measurement_id,"have":len(repeat_ids),"add":int(mc["extra_repeats_per_round"]),"max_cv":max_cv})
            else:final_runtime_variance.append({"measurement_id":measurement_id,"repeats":len(repeat_ids),"max_cv":max_cv})
        measurements.append({"measurement_id":measurement_id,"model_id":measurement["model_id"],"configuration":measurement["configuration"],"topology_level":measurement["topology_level"],"replica_id":measurement["replica_id"],"repeat_count":len(repeat_ids),"record_count":sum(len(v) for v in repeats.values()),"max_repeat_median_cv":max_cv,"repeat_median_cv":cvs})
    complete=not missing and not needs_extra
    if require_complete and not complete:raise RuntimeError({"raw_not_complete":True,"missing":missing,"needs_extra":needs_extra})
    return {"schema_version":"phase60-raw-audit-v1","raw_dir":str(raw_dir),"plan_sha256":audit["plan_sha256"],"complete":complete,"counts":{"files":len(manifest),"records":all_records,"measurements_with_data":len(by_measurement),"expected_measurements":24},"missing":missing,"needs_extra":needs_extra,"final_runtime_variance":final_runtime_variance,"files":manifest,"measurements":measurements,"records":by_measurement}

def _phase51_curve_map()->dict:
    path=Path(__file__).resolve().parents[3]/"experiment-results/phase51_pd_l1_l3_physical_curve_library/curves/pd_mooncake_physical_curves.json"
    rows=load_json(path)["curves"];return {(row["model_id"],row["topology_level"]):row for row in rows}
def _interpolate(curve:dict,payload:int)->float:
    knots=curve["knots"];xs=[int(row["payload_bytes"]) for row in knots];ys=[float(row["official_latency_us"]) for row in knots]
    if payload<xs[0] or payload>xs[-1]:raise RuntimeError({"payload_outside_phase51":payload,"support":[xs[0],xs[-1]]})
    if payload in xs:return ys[xs.index(payload)]
    right=bisect.bisect_right(xs,payload);left=right-1;fraction=(math.log2(payload)-math.log2(xs[left]))/(math.log2(xs[right])-math.log2(xs[left]));return ys[left]+fraction*(ys[right]-ys[left])
def build_analysis(plan:dict,raw:dict)->dict:
    if not raw["complete"]:raise RuntimeError("cannot aggregate incomplete raw")
    curves=_phase51_curve_map();points=[];replica_points=[];spreads=[]
    for model_id in contract()["selected_models"]:
      for configuration in contract()["research_scope"]["fixed_configurations"]:
       for level in ("L1","L2","L3"):
        curve=curves[(model_id,level)]
        for pair in payload_pairs(model_id):
            replicas=[]
            for replica in (0,1):
                mid=f"{model_id}__{configuration.lower()}__{level.lower()}__r{replica}";repeat_rows=raw["records"][mid];selected=[next(row for row in rows if row["pair_id"]==pair["pair_id"]) for rows in repeat_rows.values()]
                concurrent=[float(row["concurrent_wave"]["wave_latency_us"]["median"]) for row in selected];solo0=[float(row["solo_flow0"]["wave_latency_us"]["median"]) for row in selected];solo1=[float(row["solo_flow1"]["wave_latency_us"]["median"]) for row in selected]
                item={"model_id":model_id,"configuration":configuration,"topology_level":level,"pair_id":pair["pair_id"],"replica_id":replica,"measurement_id":mid,"payload_bytes0":pair["payload_bytes0"],"payload_bytes1":pair["payload_bytes1"],"repeat_count":len(selected),"concurrent_wave_us":statistics.median(concurrent),"solo0_us":statistics.median(solo0),"solo1_us":statistics.median(solo1),"matched_solo_ideal_us":max(statistics.median(solo0),statistics.median(solo1)),"concurrent_repeat_cv":_cv(concurrent)}
                replicas.append(item);replica_points.append(item)
            actual=max(row["concurrent_wave_us"] for row in replicas);matched=max(row["matched_solo_ideal_us"] for row in replicas);phase0=_interpolate(curve,pair["payload_bytes0"]);phase1=_interpolate(curve,pair["payload_bytes1"]);phase_ideal=max(phase0,phase1);spread=(max(row["concurrent_wave_us"] for row in replicas)-min(row["concurrent_wave_us"] for row in replicas))/actual if actual else 0.0
            points.append({"model_id":model_id,"configuration":configuration,"topology_level":level,"pair_id":pair["pair_id"],"page_count0":pair["page_count0"],"page_count1":pair["page_count1"],"payload_bytes0":pair["payload_bytes0"],"payload_bytes1":pair["payload_bytes1"],"phase51_flow0_us":phase0,"phase51_flow1_us":phase1,"phase51_ideal_us":phase_ideal,"phase51_serial_us":phase0+phase1,"matched_solo_ideal_us":matched,"actual_concurrent_wave_us":actual,"phase51_absolute_error_us":abs(phase_ideal-actual),"matched_solo_absolute_error_us":abs(matched-actual),"phase51_contention_ratio":actual/phase_ideal,"matched_solo_contention_ratio":actual/matched,"cross_replica_relative_spread":spread})
            spreads.append({"model_id":model_id,"configuration":configuration,"topology_level":level,"pair_id":pair["pair_id"],"replica0_us":replicas[0]["concurrent_wave_us"],"replica1_us":replicas[1]["concurrent_wave_us"],"official_us":actual,"relative_spread":spread,"above_threshold":spread>float(contract()["measurement_contract"]["cross_replica_relative_spread_diagnostic_threshold"])})
    metrics=[];groups=defaultdict(list)
    for row in points:
        keys=[("overall","all"),("configuration",row["configuration"]),("topology",row["topology_level"]),("config_topology",f"{row['configuration']}/{row['topology_level']}"),("model",row["model_id"])]
        for key in keys:groups[key].append(row)
    for (kind,value),rows in sorted(groups.items()):
        actual=sum(r["actual_concurrent_wave_us"] for r in rows);phase=sum(r["phase51_ideal_us"] for r in rows);matched=sum(r["matched_solo_ideal_us"] for r in rows)
        metrics.append({"slice_type":kind,"slice_value":value,"points":len(rows),"phase51_wape":sum(r["phase51_absolute_error_us"] for r in rows)/actual,"matched_solo_wape":sum(r["matched_solo_absolute_error_us"] for r in rows)/actual,"phase51_signed_bias":(phase-actual)/actual,"matched_solo_signed_bias":(matched-actual)/actual,"mean_phase51_contention_ratio":statistics.fmean(r["phase51_contention_ratio"] for r in rows),"mean_matched_solo_contention_ratio":statistics.fmean(r["matched_solo_contention_ratio"] for r in rows)})
    dc=contract()["development_decision_contract"];overall=next(row for row in metrics if row["slice_type"]=="overall");slices=[row for row in metrics if row["slice_type"]=="config_topology"]
    phase_pass=overall["phase51_wape"]<=float(dc["overall_wape_threshold"]) and all(row["phase51_wape"]<=float(dc["config_topology_wape_threshold"]) for row in slices);matched_pass=overall["matched_solo_wape"]<=float(dc["overall_wape_threshold"]) and all(row["matched_solo_wape"]<=float(dc["config_topology_wape_threshold"]) for row in slices)
    outcome="P1D1_DIRECTLY_COMPOSABLE_DEVELOPMENT" if phase_pass else "P1D1_CURVE_TRANSFER_DRIFT_REQUIRES_REVIEW" if matched_pass else "CONTENTION_CORRECTION_CANDIDATE"
    return {"points":points,"replica_points":replica_points,"spreads":spreads,"metrics":metrics,"decision":{"scientific_outcome":outcome,"phase51_baseline_pass":phase_pass,"matched_solo_baseline_pass":matched_pass,"overall_wape_threshold":dc["overall_wape_threshold"],"config_topology_wape_threshold":dc["config_topology_wape_threshold"],"future_blind_opened":False,"contention_model_fitted":False}}
