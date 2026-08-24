#!/usr/bin/env python3
"""Three-rank production Mooncake benchmark for P1D2/P2D1 two-flow waves."""
from __future__ import annotations
import argparse,concurrent.futures,json,math,os,platform,socket,statistics,subprocess,sys,threading,time
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import torch
import torch.distributed as dist
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(HERE))
from contracts import iteration_counts,layout_by_id,load_json,measurement_by_id,payload_pairs,validate_plan  # noqa:E402
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import MooncakeTransferEngine  # noqa:E402

FORBIDDEN_ENV=("MC_FORCE_TCP","MC_FORCE_MNNVL","MC_INTRANODE_NVLINK","SGLANG_MOONCAKE_CUSTOM_MEM_POOL")
def utc_now()->str:return datetime.now(timezone.utc).isoformat()
def percentile(values:list[float],q:float)->float:
    ordered=sorted(values);position=(len(ordered)-1)*q;lo=math.floor(position);hi=math.ceil(position);return ordered[lo] if lo==hi else ordered[lo]*(hi-position)+ordered[hi]*(position-lo)
def summary(values:list[float])->dict:return {"min":min(values),"median":statistics.median(values),"p95":percentile(values,.95),"max":max(values)}
def hostname_matches(expected:dict[str,Any])->bool:
    actual={platform.node(),socket.gethostname(),socket.getfqdn(),platform.node().split(".")[0],socket.gethostname().split(".")[0]};aliases=set(expected["host_aliases"]+[expected["host"]]);aliases|={value.split(".")[0] for value in aliases};return bool(actual&aliases)
def make_flows(configuration:str,pair:dict,layout:dict,span:int)->list[dict]:
    common=[
      {"flow_id":0,"page_count":pair["page_count0"],"payload_bytes":pair["payload_bytes0"],"descriptor_bytes":pair["descriptor_bytes0"],"pattern":17},
      {"flow_id":1,"page_count":pair["page_count1"],"payload_bytes":pair["payload_bytes1"],"descriptor_bytes":pair["descriptor_bytes1"],"pattern":29},
    ]
    if configuration=="P1D2":
        endpoints=[(0,1,0,0),(0,2,span,0)]
    else:endpoints=[(0,2,0,0),(1,2,0,span)]
    return [{**row,"sender_rank":src,"receiver_rank":dst,"source_offset":src_off,"destination_offset":dst_off,"descriptor_count":int(layout["descriptor_count"])} for row,(src,dst,src_off,dst_off) in zip(common,endpoints)]
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--expected-workflow-commit",required=True);p.add_argument("--topology-plan",type=Path,required=True);p.add_argument("--measurement-id",required=True);p.add_argument("--repeat-id",type=int,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();plan=load_json(a.topology_plan.expanduser().resolve());audit=validate_plan(plan);m=measurement_by_id(plan,a.measurement_id)
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if head!=a.expected_workflow_commit or plan["workflow_commit"]!=head:raise RuntimeError({"head":head,"expected":a.expected_workflow_commit,"plan":plan["workflow_commit"]})
    if os.environ.get("MOONCAKE_PROTOCOL")!="rdma" or os.environ.get("WITH_NVIDIA_PEERMEM")!="0" or os.environ.get("SGLANG_DISAGG_STAGING_BUFFER")!="0" or any(os.environ.get(name) is not None for name in FORBIDDEN_ENV):raise RuntimeError("transport environment differs from frozen RDMA/no-fallback contract")
    pythonpath=os.environ.get("PYTHONPATH","").split(os.pathsep)[0]
    if Path(pythonpath).resolve()!=(ROOT/"python").resolve():raise RuntimeError("repo python is not first PYTHONPATH entry")
    dist.init_process_group("gloo");rank=dist.get_rank();world=dist.get_world_size()
    if world!=3:raise RuntimeError(f"world_size must be 3, got {world}")
    expected=m["ranks"][rank]
    if not hostname_matches(expected):raise RuntimeError({"actual_host":platform.node(),"expected":expected})
    torch.cuda.set_device(0);device=torch.device("cuda",0);layout=layout_by_id(m["model_id"]);pairs=payload_pairs(m["model_id"]);span=max(max(row["payload_bytes0"],row["payload_bytes1"]) for row in pairs);buffer=torch.empty(2*span,dtype=torch.uint8,device=device);torch.cuda.synchronize()
    engine=MooncakeTransferEngine(hostname=expected["transfer_hostname"],gpu_id=0,ib_device=expected["ib_device"])
    if engine.batch_register([buffer.data_ptr()],[2*span])!=0:raise RuntimeError("Mooncake GPU buffer registration failed")
    gpu_uuid=getattr(torch.cuda.get_device_properties(0),"uuid",None)
    endpoint={"rank":rank,"role":expected["role"],"session_id":engine.get_session_id(),"buffer_ptr":buffer.data_ptr(),"buffer_bytes":2*span,"hostname":platform.node(),"expected_host":expected["host"],"physical_gpu":expected["physical_gpu"],"visible_gpu":0,"gpu_name":torch.cuda.get_device_name(0),"gpu_uuid":str(gpu_uuid) if gpu_uuid is not None else None,"ib_device":engine.get_ib_device(),"mooncake_protocol":os.environ["MOONCAKE_PROTOCOL"],"with_nvidia_peermem":os.environ["WITH_NVIDIA_PEERMEM"],"torch":torch.__version__,"cuda":torch.version.cuda,"python":sys.version}
    endpoints=[None]*world;dist.all_gather_object(endpoints,endpoint)
    for flow in make_flows(m["configuration"],pairs[0],layout,span):
        if rank==flow["sender_rank"] and engine.send_probe(endpoints[flow["receiver_rank"]]["session_id"])!=0:raise RuntimeError({"peer_probe_failed":flow["flow_id"]})
    dist.barrier();records=[];started=utc_now();executor=concurrent.futures.ThreadPoolExecutor(max_workers=2) if m["configuration"]=="P1D2" and rank==0 else None
    def invoke(flow:dict,gate:threading.Barrier|None=None)->dict:
        if gate is not None:gate.wait()
        src_base=buffer.data_ptr()+flow["source_offset"];peer=endpoints[flow["receiver_rank"]];dst_base=int(peer["buffer_ptr"])+flow["destination_offset"];count=flow["descriptor_count"];size=flow["descriptor_bytes"]
        src=[src_base+i*size for i in range(count)];dst=[dst_base+i*size for i in range(count)];lengths=[size]*count
        wall_start=time.time_ns();begin=time.perf_counter_ns();ret=engine.batch_transfer_sync(peer["session_id"],src,dst,lengths);end=time.perf_counter_ns();wall_end=time.time_ns()
        return {"flow_id":flow["flow_id"],"rank":rank,"ret":ret,"latency_us":(end-begin)/1000.0,"wall_start_ns":wall_start,"wall_end_ns":wall_end}
    def prepare(flows:list[dict])->None:
        for flow in flows:
            if rank==flow["sender_rank"]:buffer[flow["source_offset"]:flow["source_offset"]+flow["payload_bytes"]].fill_(flow["pattern"])
            if rank==flow["receiver_rank"]:buffer[flow["destination_offset"]:flow["destination_offset"]+flow["payload_bytes"]].zero_()
        torch.cuda.synchronize();dist.barrier()
    def validate(flows:list[dict])->None:
        valid=True
        for flow in flows:
            if rank==flow["receiver_rank"]:valid=valid and bool(torch.all(buffer[flow["destination_offset"]:flow["destination_offset"]+flow["payload_bytes"]]==flow["pattern"]).item())
        flag=torch.tensor([1 if valid else 0],dtype=torch.int32);dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        if int(flag.item())!=1:raise RuntimeError({"data_validation_failed":True,"flows":[row["flow_id"] for row in flows]})
    def execute(flows:list[dict],concurrent_mode:bool)->dict:
        dist.barrier();local=[]
        if concurrent_mode and m["configuration"]=="P1D2" and rank==0:
            gate=threading.Barrier(2);futures=[executor.submit(invoke,flow,gate) for flow in flows];local=[future.result() for future in futures]
        elif concurrent_mode:
            local=[invoke(flow) for flow in flows if rank==flow["sender_rank"]]
        else:
            flow=flows[0];local=[invoke(flow)] if rank==flow["sender_rank"] else []
        gathered=[None]*world;dist.all_gather_object(gathered,local);calls=[row for part in gathered for row in part]
        if len(calls)!=len(flows) or {row["flow_id"] for row in calls}!={row["flow_id"] for row in flows} or any(row["ret"]!=0 for row in calls):raise RuntimeError({"transfer_calls":calls,"expected_flows":[row["flow_id"] for row in flows]})
        calls=sorted(calls,key=lambda row:row["flow_id"]);starts=[row["wall_start_ns"] for row in calls]
        return {"calls":calls,"wave_latency_us":max(row["latency_us"] for row in calls),"sender_start_skew_us":(max(starts)-min(starts))/1000.0 if len(starts)>1 else 0.0}
    def run_mode(flows:list[dict],warmup:int,timed:int,concurrent_mode:bool)->dict:
        prepare(flows)
        for _ in range(warmup):execute(flows,concurrent_mode)
        results=[execute(flows,concurrent_mode) for _ in range(timed)];validate(flows)
        per_flow={str(flow["flow_id"]):[next(call for call in result["calls"] if call["flow_id"]==flow["flow_id"])["latency_us"] for result in results] for flow in flows}
        return {"flow_latency_samples_us":per_flow,"flow_latency_us":{key:summary(values) for key,values in per_flow.items()},"wave_latency_samples_us":[row["wave_latency_us"] for row in results],"wave_latency_us":summary([row["wave_latency_us"] for row in results]),"sender_start_skew_samples_us":[row["sender_start_skew_us"] for row in results],"sender_start_skew_us":summary([row["sender_start_skew_us"] for row in results]),"return_codes_all_zero":True,"data_validation_pass":True}
    try:
        for pair in pairs:
            flows=make_flows(m["configuration"],pair,layout,span);warmup,timed=iteration_counts(pair["payload_bytes0"]+pair["payload_bytes1"])
            solo0=run_mode([flows[0]],warmup,timed,False);solo1=run_mode([flows[1]],warmup,timed,False);concurrent_result=run_mode(flows,warmup,timed,True)
            if rank==0:records.append({"schema_version":"phase60-mooncake-multiflow-raw-v1","workflow_commit":head,"plan_sha256":audit["plan_sha256"],"measurement_sha256":m["measurement_sha256"],"measurement_id":m["measurement_id"],"model_id":m["model_id"],"configuration":m["configuration"],"topology_level":m["topology_level"],"replica_id":m["replica_id"],"placement_id":m["placement_id"],"repeat_id":a.repeat_id,"pair_id":pair["pair_id"],"page_count0":pair["page_count0"],"page_count1":pair["page_count1"],"payload_bytes0":pair["payload_bytes0"],"payload_bytes1":pair["payload_bytes1"],"descriptor_layout":layout["descriptor_layout"],"descriptor_count":layout["descriptor_count"],"descriptor_bytes0":pair["descriptor_bytes0"],"descriptor_bytes1":pair["descriptor_bytes1"],"op":"MooncakeTransferEngine.batch_transfer_sync","transport":"rdma","wave_admission":"gloo_barrier_then_two_synchronous_production_calls","concurrency_mechanism":"one_shared_engine_two_threads" if m["configuration"]=="P1D2" else "two_sender_rank_engines","warmup_iterations":warmup,"timed_iterations":timed,"solo_flow0":solo0,"solo_flow1":solo1,"concurrent_wave":concurrent_result,"runtime_endpoints":endpoints,"shard_started_at_utc":started,"timestamp_utc":utc_now()})
        if rank==0:
            if len(records)!=len(pairs):raise RuntimeError({"records":len(records),"expected":len(pairs)})
            output=a.output.expanduser().resolve()
            if output.exists():raise RuntimeError(f"refuse overwrite: {output}")
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.open("x",encoding="utf-8") as target:
                for row in records:target.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
            print(json.dumps({"status":"PASS","output":str(output),"records":len(records),"measurement_id":m["measurement_id"],"repeat_id":a.repeat_id},ensure_ascii=False))
    finally:
        if executor is not None:executor.shutdown(wait=True,cancel_futures=False)
        try:engine.batch_deregister([buffer.data_ptr()])
        finally:dist.destroy_process_group()
if __name__=="__main__":main()
