#!/usr/bin/env python3
"""Two-rank GPU benchmark of SGLang's production Mooncake batch-transfer wrapper."""
from __future__ import annotations
import argparse,json,math,os,platform,socket,statistics,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import torch
import torch.distributed as dist
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(HERE))
from contracts import iteration_counts,layout_by_id,load_json,measurement_by_id,validate_plan  # noqa:E402
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import MooncakeTransferEngine  # noqa:E402

FORBIDDEN_ENV=("MC_FORCE_TCP","MC_FORCE_MNNVL","MC_INTRANODE_NVLINK","SGLANG_MOONCAKE_CUSTOM_MEM_POOL")
def utc_now()->str:return datetime.now(timezone.utc).isoformat()
def percentile(values:list[float],q:float)->float:
    ordered=sorted(values);position=(len(ordered)-1)*q;lo=math.floor(position);hi=math.ceil(position);return ordered[lo] if lo==hi else ordered[lo]*(hi-position)+ordered[hi]*(position-lo)
def hostname_matches(expected:dict[str,Any])->bool:
    actual={platform.node(),socket.gethostname(),socket.getfqdn(),platform.node().split(".")[0],socket.gethostname().split(".")[0]};aliases=set(expected["host_aliases"]+[expected["host"]]);aliases|={value.split(".")[0] for value in aliases};return bool(actual&aliases)
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--expected-workflow-commit",required=True);p.add_argument("--topology-plan",type=Path,required=True);p.add_argument("--measurement-id",required=True);p.add_argument("--repeat-id",type=int,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();plan=load_json(a.topology_plan.expanduser().resolve());audit=validate_plan(plan);m=measurement_by_id(plan,a.measurement_id)
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if head!=a.expected_workflow_commit or plan["workflow_commit"]!=head:raise RuntimeError({"head":head,"expected":a.expected_workflow_commit,"plan":plan["workflow_commit"]})
    if os.environ.get("MOONCAKE_PROTOCOL")!="rdma" or os.environ.get("WITH_NVIDIA_PEERMEM")!="0" or os.environ.get("SGLANG_DISAGG_STAGING_BUFFER")!="0" or any(os.environ.get(name) is not None for name in FORBIDDEN_ENV):raise RuntimeError("transport environment differs from frozen RDMA/no-fallback contract")
    pythonpath=os.environ.get("PYTHONPATH","").split(os.pathsep)[0]
    if Path(pythonpath).resolve()!= (ROOT/"python").resolve():raise RuntimeError("repo python is not first PYTHONPATH entry")
    dist.init_process_group("gloo");rank=dist.get_rank();world=dist.get_world_size()
    if world!=2:raise RuntimeError(f"world_size must be 2, got {world}")
    expected_rank=m["ranks"][rank]
    if not hostname_matches(expected_rank):raise RuntimeError({"actual_host":platform.node(),"expected":expected_rank})
    local_rank=int(os.environ.get("LOCAL_RANK","0"));torch.cuda.set_device(local_rank);device=torch.device("cuda",local_rank);layout=layout_by_id(m["model_id"]);max_payload=max(row["payload_bytes"] for row in layout["knots"]);buffer=torch.empty(max_payload,dtype=torch.uint8,device=device);torch.cuda.synchronize()
    engine=MooncakeTransferEngine(hostname=expected_rank["transfer_hostname"],gpu_id=local_rank,ib_device=expected_rank["ib_device"])
    if engine.batch_register([buffer.data_ptr()],[max_payload])!=0:raise RuntimeError("Mooncake GPU buffer registration failed")
    gpu_uuid=getattr(torch.cuda.get_device_properties(local_rank),"uuid",None)
    endpoint={"rank":rank,"session_id":engine.get_session_id(),"buffer_ptr":buffer.data_ptr(),"max_payload_bytes":max_payload,"hostname":platform.node(),"expected_host":expected_rank["host"],"physical_gpu":expected_rank["physical_gpu"],"visible_gpu":local_rank,"gpu_name":torch.cuda.get_device_name(local_rank),"gpu_uuid":str(gpu_uuid) if gpu_uuid is not None else None,"ib_device":engine.get_ib_device(),"mooncake_protocol":os.environ["MOONCAKE_PROTOCOL"],"with_nvidia_peermem":os.environ["WITH_NVIDIA_PEERMEM"],"torch":torch.__version__,"cuda":torch.version.cuda,"python":sys.version}
    endpoints=[None]*world;dist.all_gather_object(endpoints,endpoint)
    peer=endpoints[1-rank]
    if engine.send_probe(peer["session_id"])!=0:raise RuntimeError("Mooncake peer probe failed")
    dist.barrier();local_records=[];started=utc_now()
    try:
        for knot in layout["knots"]:
            pages=int(knot["page_count"]);payload=int(knot["payload_bytes"]);descriptor_bytes=int(knot["descriptor_bytes"]);count=int(layout["descriptor_count"]);warmup,timed=iteration_counts(payload);lengths=[descriptor_bytes]*count
            for sender,receiver,direction,pattern in ((0,1,"rank0_to_rank1",17),(1,0,"rank1_to_rank0",29)):
                if rank==sender:buffer[:payload].fill_(pattern)
                else:buffer[:payload].zero_()
                torch.cuda.synchronize();dist.barrier();samples=[];status=[]
                if rank==sender:
                    src=[buffer.data_ptr()+i*descriptor_bytes for i in range(count)];dst=[int(peer["buffer_ptr"])+i*descriptor_bytes for i in range(count)]
                    for _ in range(warmup):
                        ret=engine.batch_transfer_sync(peer["session_id"],src,dst,lengths)
                        if ret!=0:raise RuntimeError({"warmup_transfer_failed":ret,"payload":payload,"direction":direction})
                    for _ in range(timed):
                        torch.cuda.synchronize();begin=time.perf_counter_ns();ret=engine.batch_transfer_sync(peer["session_id"],src,dst,lengths);torch.cuda.synchronize();end=time.perf_counter_ns()
                        if ret!=0:raise RuntimeError({"timed_transfer_failed":ret,"payload":payload,"direction":direction})
                        samples.append((end-begin)/1000.0);status.append(ret)
                dist.barrier();valid=True
                if rank==receiver:valid=bool(torch.all(buffer[:payload]==pattern).item())
                valid_tensor=torch.tensor([1 if valid else 0],dtype=torch.int32);dist.all_reduce(valid_tensor,op=dist.ReduceOp.MIN)
                if int(valid_tensor.item())!=1:raise RuntimeError({"data_validation_failed":True,"payload":payload,"direction":direction})
                if rank==sender:
                    local_records.append({"schema_version":"phase51-mooncake-raw-v1","workflow_commit":head,"plan_sha256":audit["plan_sha256"],"measurement_sha256":m["measurement_sha256"],"measurement_id":m["measurement_id"],"model_id":m["model_id"],"topology_level":m["topology_level"],"replica_id":m["replica_id"],"placement_id":m["placement_id"],"repeat_id":a.repeat_id,"direction":direction,"sender_rank":sender,"receiver_rank":receiver,"page_count":pages,"page_size_tokens":layout["page_size_tokens"],"payload_bytes":payload,"descriptor_layout":layout["descriptor_layout"],"descriptor_count":count,"descriptor_bytes":descriptor_bytes,"op":"MooncakeTransferEngine.batch_transfer_sync","transport":"rdma","warmup_iterations":warmup,"timed_iterations":timed,"latency_us":{"min":min(samples),"median":statistics.median(samples),"p95":percentile(samples,.95),"max":max(samples)},"latency_samples_us":samples,"data_validation_pass":True,"return_codes_all_zero":all(value==0 for value in status),"runtime_endpoints":endpoints,"shard_started_at_utc":started,"timestamp_utc":utc_now()})
                dist.barrier()
        gathered=[None]*world;dist.gather_object(local_records,gathered if rank==0 else None,dst=0)
        if rank==0:
            records=[row for part in gathered for row in part];expected=len(layout["knots"])*2
            if len(records)!=expected:raise RuntimeError({"records":len(records),"expected":expected})
            output=a.output.expanduser().resolve()
            if output.exists():raise RuntimeError(f"refuse overwrite: {output}")
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.open("x",encoding="utf-8") as target:
                for row in sorted(records,key=lambda value:(value["page_count"],value["direction"])):target.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
            print(json.dumps({"status":"PASS","output":str(output),"records":len(records),"measurement_id":m["measurement_id"],"repeat_id":a.repeat_id},ensure_ascii=False))
    finally:
        # A barrier here can hide the original error forever when only one rank fails.
        try:engine.batch_deregister([buffer.data_ptr()])
        finally:dist.destroy_process_group()
if __name__=="__main__":main()
