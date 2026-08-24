#!/usr/bin/env python3
"""Phase60 read-only source/runtime/plan audit; creates an empty external raw root."""
from __future__ import annotations
import argparse,importlib.util,inspect,json,os,re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from common import load_json,repo_root,require_clean_before_run,require_expected_head,utc_now,verify_pinned_inputs,write_json  # noqa:E402
from contracts import file_sha,validate_plan  # noqa:E402

FORBIDDEN_ENV=("MC_FORCE_TCP","MC_FORCE_MNNVL","MC_INTRANODE_NVLINK","SGLANG_MOONCAKE_CUSTOM_MEM_POOL")
def external(path:Path,label:str)->Path:
    value=path.expanduser().resolve();root=repo_root().resolve()
    if value==root or root in value.parents:raise RuntimeError(f"{label} must remain outside Git: {value}")
    return value
def run_checks(expected:str,plan_path:Path,raw_dir:Path,audit_output:Path,container_image:str)->dict:
    spec=load_json(HERE/"experiment.json");head=require_expected_head(expected);require_clean_before_run();pins=verify_pinned_inputs(spec)
    if container_image!=spec["container_contract"]["image"]:raise RuntimeError({"container_image":container_image,"expected":spec["container_contract"]["image"]})
    plan_path=external(plan_path,"topology plan");raw_dir=external(raw_dir,"raw directory");audit_output=external(audit_output,"preflight audit")
    if not plan_path.is_file():raise RuntimeError(f"topology plan missing: {plan_path}")
    plan=load_json(plan_path);plan_audit=validate_plan(plan)
    if plan.get("workflow_commit")!=head:raise RuntimeError({"plan_workflow_commit":plan.get("workflow_commit"),"HEAD":head})
    if raw_dir.exists():
        if not raw_dir.is_dir() or any(raw_dir.iterdir()):raise RuntimeError(f"raw directory must be absent or empty: {raw_dir}")
    else:raw_dir.mkdir(parents=True,exist_ok=False)
    if audit_output.exists():raise RuntimeError(f"refuse overwrite: {audit_output}")
    if raw_dir==audit_output or raw_dir in audit_output.parents:raise RuntimeError("preflight audit must not be inside raw directory")
    expected_python=(repo_root()/"python").resolve();pythonpath=[Path(v).resolve() for v in os.environ.get("PYTHONPATH","").split(os.pathsep) if v]
    env_checks={"repo_python_first":bool(pythonpath) and pythonpath[0]==expected_python,"rdma":os.environ.get("MOONCAKE_PROTOCOL")=="rdma","dmabuf":os.environ.get("WITH_NVIDIA_PEERMEM")=="0","no_staging":os.environ.get("SGLANG_DISAGG_STAGING_BUFFER")=="0","offline_hf":os.environ.get("HF_HUB_OFFLINE")=="1","offline_transformers":os.environ.get("TRANSFORMERS_OFFLINE")=="1","no_forced_fallback":all(os.environ.get(name) is None for name in FORBIDDEN_ENV)}
    modules={name:importlib.util.find_spec(name) is not None for name in spec["container_contract"]["required_runtime_modules"]}
    if not all(env_checks.values()) or not all(modules.values()):raise RuntimeError({"environment":env_checks,"modules":modules})
    import mooncake,torch
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import MooncakeTransferEngine
    origin=Path(inspect.getsourcefile(MooncakeTransferEngine) or "").resolve();conn=(repo_root()/"python/sglang/srt/disaggregation/mooncake/conn.py").read_text(encoding="utf-8")
    runtime_checks={"cuda_available":torch.cuda.is_available(),"at_least_one_visible_gpu_on_coordinator":torch.cuda.device_count()>=1,"production_wrapper_from_repo":origin==expected_python/"sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py","batch_transfer_sync":callable(getattr(MooncakeTransferEngine,"batch_transfer_sync",None)),"batch_register":callable(getattr(MooncakeTransferEngine,"batch_register",None)),"production_conn_has_thread_pool":"ThreadPoolExecutor" in conn and "batch_transfer_sync" in conn}
    if not all(runtime_checks.values()):raise RuntimeError({"runtime_checks":runtime_checks})
    result={"schema_version":"phase60-preflight-v1","status":"PASS","captured_at_utc":utc_now(),"workflow_commit":head,"plan_path":str(plan_path),"plan_file_sha256":file_sha(plan_path),"plan_audit":plan_audit,"raw_dir":str(raw_dir),"pinned_inputs":pins,"environment_checks":env_checks,"module_checks":modules,"runtime_checks":runtime_checks,"environment":{"declared_container_image":container_image,"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_device_count":torch.cuda.device_count(),"gpu_names":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"mooncake_version":getattr(mooncake,"__version__",None),"wrapper_origin":str(origin),"container_image_env":{k:v for k,v in os.environ.items() if re.search(r"(IMAGE|CONTAINER|PYTORCH_VERSION)",k)}}}
    write_json(audit_output,result);return result
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--expected-workflow-commit",required=True);p.add_argument("--topology-plan",type=Path,required=True);p.add_argument("--raw-dir",type=Path,required=True);p.add_argument("--audit-output",type=Path,required=True);p.add_argument("--container-image",required=True);a=p.parse_args();print(json.dumps(run_checks(a.expected_workflow_commit,a.topology_plan,a.raw_dir,a.audit_output,a.container_image),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
