"""
Benchmark the latency of running a single static batch without a server.

This script does not launch a server and uses the low-level APIs.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

# Usage (latency test)
## with dummy weights:
python -m sglang.benchmark.one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --load-format dummy
## sweep through multiple data points and store (append) the results in a jsonl file:
python -m sglang.benchmark.one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch-size 1 12 14 --input-len 256 512 --output-len 32 256 --run-name test_run
## run with profiling:
python -m sglang.benchmark.one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch-size 1 12 14 --input-len 256 512 --profile
## run with profiling to custom directory:
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log
python -m sglang.benchmark.one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch-size 1 --input-len 256 --profile
## run with CUDA profiler (nsys):
nsys profile --force-overwrite=true -o bench_one_batch python -m sglang.benchmark.one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch-size 1 --input-len 256 --profile --profile-activities CUDA_PROFILER
# Usage (correctness test):
python -m sglang.benchmark.one_batch --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --correct

## Reference output (of the correctness test above, can be gpu dependent):
input_ids=[[1, 450, 7483, 310, 3444, 338], [1, 450, 7483, 310, 278, 3303, 13187, 290, 338], [1, 20628, 338, 263, 6575, 1460, 2462, 322, 306, 763]]

prefill logits (first half): tensor([[-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [ -9.1875, -10.2500,   2.7129,  ...,  -4.3359,  -4.0664,  -4.1328]],
       device='cuda:0')

prefill logits (final): tensor([[-8.3125, -7.1172,  3.3457,  ..., -4.9570, -4.1328, -3.4141],
        [-8.9141, -9.0156,  4.1445,  ..., -4.9922, -4.4961, -4.0781],
        [-9.6328, -9.0547,  4.0195,  ..., -5.3047, -4.7148, -4.4570]],
       device='cuda:0')

========== Prompt 0 ==========
<s> The capital of France is Paris.
The capital of the United States is Washington, D.C.


========== Prompt 1 ==========
<s> The capital of the United Kindom is London.
The capital of the United Kingdom is London.
The capital of the

========== Prompt 2 ==========
<s> Today is a sunny day and I like to go for a walk in the park.
I'm going to the park
"""

import argparse
import copy
import dataclasses
import hashlib
import itertools
import json
import logging
import multiprocessing
import os
import time
from array import array
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import comm_profile
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.dp_attn import prepare_mlp_sync_batch_raw
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.model_executor.cuda_graph_config import Phase
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    kill_process_tree,
    maybe_reindex_device_id,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.srt.utils.tensor_bridge import use_mlx


def start_profile(
    profile_activities,
    profile_record_shapes=False,
    rank_print=print,
    trace_filename=None,
):
    """
    Abstracted function to start profiling based on profile_activities.
    Returns profiler object (or None).
    """
    if use_mlx():
        import mlx.core as mx

        if trace_filename:
            mlx_trace_filename = trace_filename.replace(".trace.json.gz", ".gputrace")
            mx.metal.start_capture(mlx_trace_filename)
            rank_print(f"MLX Metal capture started directly to {mlx_trace_filename}")
        return "mlx"

    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStart()
            rank_print("CUDA Profiler started (nsys will begin capturing)")
        except Exception as e:
            rank_print(f"Failed to start CUDA profiler: {e}")
        return None
    else:
        activities = []
        if "CPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "GPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if "XPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.XPU)
        if activities:
            profiler = torch.profiler.profile(
                activities=activities,
                with_stack=True,
                record_shapes=profile_record_shapes,
            )
            profiler.start()
            return profiler
        return None


def stop_profile(
    profiler,
    profile_activities,
    rank_print=print,
    save_trace=False,
    trace_filename=None,
    stage=None,
):
    """
    Abstracted function to stop profiling based on profile_activities.
    Optionally saves trace results and prints completion messages.
    """
    if profiler == "mlx":
        import mlx.core as mx

        mx.metal.stop_capture()

        if save_trace and trace_filename:
            # Change SGLang's default torch extension to Apple's .gputrace extension
            mlx_trace_filename = trace_filename.replace(".trace.json.gz", ".gputrace")

            stage_desc = f"for {stage}" if stage else ""
            rank_print(f"MLX Metal gputrace {stage_desc} saved to {mlx_trace_filename}")
        return

    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStop()
            rank_print("CUDA Profiler stopped (nsys should dump traces)")
        except Exception as e:
            rank_print(f"Failed to stop CUDA profiler: {e}")
    elif profiler is not None:
        profiler.stop()

    if save_trace:
        if profiler is not None:
            if trace_filename:
                _save_profile_trace_results(
                    profiler, profile_activities, trace_filename
                )
                stage_desc = f"for {stage}" if stage else ""
                rank_print(
                    f"torch profiler chrome trace {stage_desc} saved to {trace_filename}"
                )
        if "CUDA_PROFILER" in profile_activities:
            rank_print(f"CUDA profiler trace for {stage} completed")


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    prompt_filename: str = ""
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    log_decode_step: int = 0
    profile: bool = False
    profile_record_shapes: bool = False
    profile_activities: Tuple[str] = ("CPU", "GPU")
    profile_stage: str = "all"
    profile_prefix: str = "profile"
    profile_start_step: Optional[int] = None
    profile_steps: Optional[int] = None
    profile_all_ranks: bool = False
    warmup_each_workload: bool = False
    comm_profile: bool = False
    comm_profile_mode: str = "full-trace"
    input_lens_per_request: Tuple[int] = ()
    output_lens_per_request: Tuple[int] = ()
    prefill_chunk_size: int = 0
    trace_replay_plan: str = ""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--prompt-filename", type=str, default=BenchArgs.prompt_filename
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=BenchArgs.log_decode_step,
            help="Log decode latency by step, default is set to zero to disable.",
        )
        parser.add_argument("--profile", action="store_true", help="Enable profiling.")
        parser.add_argument(
            "--profile-record-shapes",
            action="store_true",
            help="Record tensor shapes in profiling results.",
        )
        parser.add_argument(
            "--profile-activities",
            type=str,
            nargs="+",
            default=["CPU", "GPU"],
            choices=["CPU", "GPU", "CUDA_PROFILER", "XPU"],
            help="Profiler activities: CPU, GPU, XPU, CUDA_PROFILER. If CPU/GPU/XPU, use torch profiler. If CUDA_PROFILER, use CUDA profiler.",
        )
        parser.add_argument(
            "--profile-stage",
            type=str,
            default=BenchArgs.profile_stage,
            choices=["all", "prefill", "decode"],
            help="Which stage to profile: all, prefill, or decode only.",
        )
        parser.add_argument(
            "--profile-prefix",
            "--profile-filename-prefix",  # deprecated alias, kept for back-compat
            dest="profile_prefix",
            type=str,
            default=BenchArgs.profile_prefix,
            help="Prefix of the profiling file names. The full profiling result file(s) be "
            '"[profile_prefix]_batch[batch_size]_input[input_len]_output[output_len].trace.json.gz"',
        )
        parser.add_argument(
            "--profile-start-step",
            type=int,
            default=None,
            help="Decode step at which to start profiling (0-indexed). If not specified, defaults to output_len // 2.",
        )
        parser.add_argument(
            "--profile-steps",
            type=int,
            default=None,
            help="Number of decode steps to profile starting from profile-start-step. If not specified, profiles only one step.",
        )
        parser.add_argument(
            "--profile-all-ranks",
            action="store_true",
            help=(
                "Enable the requested profiler on every TP rank and add a rank "
                "component to each trace prefix. By default only rank 0 is profiled."
            ),
        )
        parser.add_argument(
            "--warmup-each-workload",
            action="store_true",
            help=(
                "Before measuring every workload point, run an unprofiled warmup "
                "with the same batch size, input length, and execution knobs. "
                "Decode warmup lengths are capped at 32 tokens."
            ),
        )
        parser.add_argument(
            "--comm-profile",
            action="store_true",
            help="Record per-phase collective communication calls and tensor bytes.",
        )
        parser.add_argument(
            "--comm-profile-mode",
            type=str,
            default=BenchArgs.comm_profile_mode,
            choices=["full-trace", "histogram-only"],
            help="Store raw collective events or only the lossless aggregate histogram.",
        )
        parser.add_argument(
            "--input-lens-per-request",
            type=int,
            nargs="+",
            default=BenchArgs.input_lens_per_request,
            help="Per-request input lengths for a heterogeneous draining batch.",
        )
        parser.add_argument(
            "--output-lens-per-request",
            type=int,
            nargs="+",
            default=BenchArgs.output_lens_per_request,
            help="Per-request exact output lengths for a draining mixed-length batch.",
        )
        parser.add_argument(
            "--prefill-chunk-size",
            type=int,
            default=BenchArgs.prefill_chunk_size,
            help="Controlled prefill chunk size for boundary experiments; 0 disables chunking.",
        )
        parser.add_argument(
            "--trace-replay-plan",
            type=str,
            default=BenchArgs.trace_replay_plan,
            help=(
                "JSONL workload plan containing workload_id, input_lens_per_request, "
                "and output_lens_per_request. The model is loaded once and all plans "
                "are replayed as heterogeneous draining batches."
            ),
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        result = {}
        for attr, attr_type in attrs:
            value = getattr(args, attr)
            # Handle None values - don't try to cast them
            if value is None or attr_type == type(None):
                result[attr] = value
            else:
                result[attr] = attr_type(value)
        return cls(**result)


def load_model(server_args, port_args, gpu_id, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

    model_config = ModelConfig.from_server_args(server_args)
    runner_kwargs = dict(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )

    _use_mlx = use_mlx()
    if _use_mlx:
        from sglang.srt.hardware_backend.mlx.model_runner_stub import (
            MlxModelRunnerStub,
        )

        model_runner = MlxModelRunnerStub(**runner_kwargs)
    else:
        model_runner = ModelRunner(**runner_kwargs)
        model_runner.alloc_memory_pool()
        model_runner.init_attention_backends()
        model_runner.init_cuda_graphs()
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()

    if _use_mlx:
        model_runner = _MlxBenchRunner(model_runner, server_args)
    else:
        model_runner = _TorchBenchRunner(model_runner)

    return model_runner, tokenizer


def prepare_inputs_for_correctness_test(bench_args, tokenizer, custom_prompts):
    if custom_prompts:
        custom_input_len = len(custom_prompts)
        bs = bench_args.batch_size[0]
        if custom_input_len > bs:
            logging.warning(
                f"Custom input size ({custom_input_len}) is larger than batch_size ({bs}). "
                f"Using the first {bs} prompts."
            )
            custom_prompts = custom_prompts[:bs]

    prompts = (
        custom_prompts
        if custom_prompts
        else [
            "The capital of France is",
            "The capital of the United Kindom is",
            "Today is a sunny day and I like",
        ]
    )
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=array("q", tmp_input_ids),
            sampling_params=sampling_params,
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_range(len(req.prefix_indices), len(req.origin_input_ids))
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req: Req = reqs[i]
        req.full_untruncated_fill_ids.extend(input_ids[i][bench_args.cut_len :])
        if model_runner is not None:
            # Use req.req_pool_idx instead of i to handle slot 0 padding correctly
            req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
                req.req_pool_idx, : bench_args.cut_len
            ].to(req.prefix_indices.dtype)
            req.logprob_start_len = -1
        req.set_extend_range(
            len(req.prefix_indices), len(req.full_untruncated_fill_ids)
        )
    return reqs


def prepare_synthetic_inputs_for_latency_test(
    batch_size, input_len, custom_inputs=None
):
    input_ids = (
        custom_inputs
        if custom_inputs
        else np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=array("q", input_ids[i]),
            sampling_params=sampling_params,
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_range(len(req.prefix_indices), len(req.origin_input_ids))
        reqs.append(req)

    return reqs


class TreeCacheNamespace(SimpleNamespace):
    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()

    def evict(self, params: EvictParams):
        pass


@torch.no_grad
def extend(reqs, model_runner):
    # Create dummy tree_cache for benchmarks (no prefix caching, just allocation)
    dummy_tree_cache = TreeCacheNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )

    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    if (
        batch.input_ids is None
        and getattr(batch, "prefill_input_ids_cpu", None) is not None
    ):
        batch.input_ids = batch.prefill_input_ids_cpu.to(
            batch.device, non_blocking=True
        )
        batch.prefill_input_ids_cpu = None

    forward_batch = ForwardBatch.init_new(batch, model_runner)
    logits_output = model_runner.forward(forward_batch).logits_output
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def decode(input_token_ids, batch, model_runner):
    batch.input_ids = input_token_ids.to(torch.int64)
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    forward_batch = ForwardBatch.init_new(batch, model_runner)
    logits_output = model_runner.forward(forward_batch).logits_output
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits


def _maybe_prepare_mlp_sync_batch(batch: ScheduleBatch, model_runner):
    if require_mlp_sync(model_runner.server_args):
        prepare_mlp_sync_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=get_attention_tp_size(),
            attn_cp_size=model_runner.attn_cp_size,
            tp_group=model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
            offload_tags=set(),
        )


class _TorchBenchRunner:
    """Wraps ModelRunner for the standard PyTorch benchmark path."""

    def __init__(self, model_runner):
        self.torch_runner = model_runner

    def clear(self):
        self.torch_runner.req_to_token_pool.clear()
        self.torch_runner.token_to_kv_pool_allocator.clear()

    def extend(self, reqs):
        return extend(reqs, self.torch_runner)

    def decode(self, next_token_ids, batch):
        return decode(next_token_ids, batch, self.torch_runner)

    def cleanup(self, batch):
        pass

    def synchronize(self):
        synchronize(self.torch_runner.device)

    def max_batch_size(self, input_len, output_len):
        return self.torch_runner.max_total_num_tokens // (input_len + output_len)


class _MlxBenchRunner:
    """Wraps MlxModelRunner for the MLX benchmark path."""

    def __init__(self, model_runner, server_args):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        # Radix cache requires the scheduler's allocator/trie; disable in
        # standalone bench mode where no scheduler is present.
        init_kwargs = dict(
            model_path=server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            disable_radix_cache=True,
            mem_fraction_static=server_args.mem_fraction_static,
            quantization=server_args.quantization,
        )
        if server_args.max_total_tokens is not None:
            init_kwargs["pool_size"] = server_args.max_total_tokens
        self.mlx_runner = MlxModelRunner(**init_kwargs)
        self.mlx_runner.init_cache_pools(req_to_token_pool=None)
        self.fake_torch_runner = model_runner

    def clear(self):
        self.mlx_runner.clear()

    def extend(self, reqs):
        req_ids = [str(req.rid) for req in reqs]
        results = []
        for rid, req in zip(req_ids, reqs):
            token_ids = [int(t) for t in req.get_fill_ids()]
            next_token = self.mlx_runner.prefill(
                req_id=rid,
                new_token_ids=token_ids,
                full_token_ids=token_ids,
                prefix_slot_ids=[],
                new_slot_ids=[],
                req_pool_idx=0,
            )
            results.append(next_token)
        return torch.tensor(results), None, req_ids

    def decode(self, next_token_ids, req_ids):
        next_token_ids = self.mlx_runner.decode_batch(req_ids)
        return torch.tensor(next_token_ids), None

    def cleanup(self, batch):
        if isinstance(batch, list):
            for req_id in batch:
                self.mlx_runner.remove_request(req_id)

    def synchronize(self):
        pass

    def max_batch_size(self, input_len, output_len):
        return self.fake_torch_runner.max_total_num_tokens // (input_len + output_len)


def _read_prompts_from_file(prompt_file, rank_print):
    """Read custom prompts from the file specified by `--prompt-filename`."""
    if not prompt_file:
        return []
    if not os.path.exists(prompt_file):
        rank_print(
            f"Custom prompt file {prompt_file} not found. Using default inputs..."
        )
        return []
    with open(prompt_file, "r") as pf:
        return pf.readlines()


def _get_torch_profiler_output_dir():
    return os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp")


def _create_torch_profiler_filename(
    profile_prefix, batch_size, input_len, output_len, stage
):
    output_dir = _get_torch_profiler_output_dir()
    filename = f"{profile_prefix}_batch{batch_size}_input{input_len}_output{output_len}_{stage}.trace.json.gz"
    return os.path.join(output_dir, filename)


def _save_profile_trace_results(profiler, profile_activities, filename):
    parent_dir = os.path.dirname(os.path.abspath(filename))
    os.makedirs(parent_dir, exist_ok=True)
    profiler.export_chrome_trace(filename)
    if "GPU" in profile_activities:
        sort_by = "self_cuda_time_total"
    elif "XPU" in profile_activities:
        sort_by = "self_xpu_time_total"
    else:
        sort_by = "self_cpu_time_total"
    print(profiler.key_averages(group_by_input_shape=True).table(sort_by=sort_by))


def correctness_test(
    server_args,
    port_args,
    bench_args,
    gpu_id,
    tp_rank,
):
    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)

    # Prepare inputs
    custom_prompts = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
    input_ids, reqs = prepare_inputs_for_correctness_test(
        bench_args, tokenizer, custom_prompts
    )
    rank_print(f"\n{input_ids=}\n")

    if bench_args.cut_len > 0:
        # Prefill
        next_token_ids, next_token_logits, batch = model_runner.extend(reqs)
        rank_print(f"prefill logits (first half): {next_token_logits} \n")

        # Prepare extend inputs
        torch_runner = getattr(model_runner, "torch_runner", None)
        reqs = prepare_extend_inputs_for_correctness_test(
            bench_args, input_ids, reqs, torch_runner
        )

    # Extend (prefill w/ KV cache)
    next_token_ids, next_token_logits, batch = model_runner.extend(reqs)
    rank_print(f"prefill logits (final): {next_token_logits} \n")

    # Decode
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    for _ in range(bench_args.output_len[0] - 1):
        next_token_ids, _ = model_runner.decode(next_token_ids, batch)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])

    # Clean up
    model_runner.cleanup(batch)

    # Print output texts
    for i in range(len(reqs)):
        rank_print(f"========== Prompt {i} ==========")
        rank_print(tokenizer.decode(output_ids[i]), "\n")


def synchronize(device):
    torch.get_device_module(device).synchronize()


def latency_test_run_once(
    run_name,
    model_runner,
    rank_print,
    reqs,
    batch_size,
    input_len,
    output_len,
    log_decode_step,
    profile,
    profile_record_shapes,
    profile_activities,
    profile_prefix,
    profile_stage,
    tp_rank,
    profile_start_step=None,
    profile_steps=None,
    enable_comm_profile=False,
    comm_profile_mode="full-trace",
    output_lens_per_request=None,
    prefill_chunk_size=0,
    full_input_ids=None,
    input_lens_per_request=None,
):
    effective_input_lens = (
        list(input_lens_per_request)
        if input_lens_per_request
        else [input_len] * batch_size
    )
    effective_output_lens = (
        list(output_lens_per_request)
        if output_lens_per_request
        else [output_len] * batch_size
    )
    if len(effective_output_lens) != batch_size:
        raise ValueError(
            "output_lens_per_request must contain exactly one value per request"
        )
    if len(effective_input_lens) != batch_size:
        raise ValueError(
            "input_lens_per_request must contain exactly one value per request"
        )
    if min(effective_input_lens) < 1 or max(effective_input_lens) != input_len:
        raise ValueError(
            "Per-request input lengths must be positive and max must equal input_len"
        )
    if min(effective_output_lens) < 1 or max(effective_output_lens) != output_len:
        raise ValueError(
            "Per-request output lengths must be positive and max must equal output_len"
        )

    max_batch_size = model_runner.max_batch_size(input_len, output_len)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping ({batch_size}, {input_len}, {output_len}) due to max batch size limit"
        )
        return

    model_runner.clear()

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "input_lens_per_request": effective_input_lens,
        "output_len": output_len,
        "output_lens_per_request": effective_output_lens,
    }

    comm_profile.enable(bool(enable_comm_profile))
    comm_profile.set_capture_mode(comm_profile_mode)
    comm_profile.reset()
    comm_profile.set_phase(None)
    comm_profile.set_decode_step(None)
    comm_profile.set_active_batch_size(None)

    tot_latency = 0

    profiler = None
    enable_profile_prefill = profile and profile_stage in ["all", "prefill"]
    trace_filename_prefill = None
    if enable_profile_prefill:
        trace_filename_prefill = _create_torch_profiler_filename(
            profile_prefix, batch_size, input_len, output_len, "prefill"
        )
        profiler = start_profile(
            profile_activities,
            profile_record_shapes=profile_record_shapes,
            rank_print=rank_print,
            trace_filename=trace_filename_prefill,  # pass it in here for the MLX path only
        )

    if prefill_chunk_size < 0:
        raise ValueError("prefill_chunk_size must be non-negative")
    if prefill_chunk_size > 0 and len(set(effective_input_lens)) != 1:
        raise ValueError(
            "Chunked prefill does not yet support heterogeneous per-request input lengths"
        )
    if prefill_chunk_size > 0 and full_input_ids is None:
        raise ValueError("full_input_ids are required for chunked prefill")

    chunk_size = prefill_chunk_size or input_len
    prefill_chunk_records = []
    next_token_ids = None
    batch = None
    chunk_start = 0
    chunk_index = 0
    while chunk_start < input_len:
        chunk_end = min(chunk_start + chunk_size, input_len)
        if chunk_index > 0:
            torch_runner = getattr(model_runner, "torch_runner", None)
            if torch_runner is None:
                raise RuntimeError(
                    "Chunked prefill currently requires the PyTorch runner"
                )
            for request_index, req in enumerate(reqs):
                req.full_untruncated_fill_ids.extend(
                    int(token)
                    for token in full_input_ids[request_index][chunk_start:chunk_end]
                )
                req.prefix_indices = torch_runner.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :chunk_start
                ].to(req.prefix_indices.dtype)
                req.logprob_start_len = -1
                req.set_extend_range(chunk_start, chunk_end)

        model_runner.synchronize()
        comm_profile.set_phase("prefill")
        comm_profile.set_decode_step(None)
        comm_profile.set_active_batch_size(batch_size)
        comm_profile.set_prefill_chunk(chunk_index, chunk_end - chunk_start)
        tic = time.perf_counter()
        next_token_ids, _, batch = model_runner.extend(reqs)
        model_runner.synchronize()
        chunk_latency = time.perf_counter() - tic
        prefill_chunk_records.append(
            {
                "chunk_index": chunk_index,
                "start_token": chunk_start,
                "end_token": chunk_end,
                "chunk_tokens_per_request": chunk_end - chunk_start,
                "latency": chunk_latency,
            }
        )
        comm_profile.set_phase(None)
        comm_profile.set_active_batch_size(None)
        comm_profile.set_prefill_chunk(None, None)
        chunk_start = chunk_end
        chunk_index += 1

    prefill_latency = sum(item["latency"] for item in prefill_chunk_records)

    if enable_profile_prefill:
        stop_profile(
            profiler,
            profile_activities,
            rank_print=rank_print,
            save_trace=True,
            trace_filename=trace_filename_prefill,
            stage="prefill",
        )

    tot_latency += prefill_latency
    throughput = sum(effective_input_lens) / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput
    measurement_results["prefill_chunk_size"] = prefill_chunk_size
    measurement_results["prefill_chunk_records"] = prefill_chunk_records

    decode_latencies = []
    decode_step_records = []
    active_req_indices = list(range(batch_size))
    # Determine profiling start step and end step
    profile_start = (
        profile_start_step if profile_start_step is not None else (output_len // 2)
    )
    profile_end = profile_start + (profile_steps if profile_steps is not None else 1)
    enable_profile_decode = profile and profile_stage in ["all", "decode"]
    trace_filename_decode = None
    profiler = None
    for i in range(output_len - 1):
        keep_positions = [
            pos
            for pos, request_index in enumerate(active_req_indices)
            if i < effective_output_lens[request_index] - 1
        ]
        if not keep_positions:
            break
        if len(keep_positions) != len(active_req_indices):
            if not hasattr(batch, "filter_batch"):
                raise RuntimeError(
                    "Mixed output lengths require a runner with ScheduleBatch.filter_batch"
                )
            batch.filter_batch(keep_indices=keep_positions)
            next_token_ids = next_token_ids[keep_positions]
            active_req_indices = [active_req_indices[pos] for pos in keep_positions]
        active_batch_size = len(active_req_indices)

        model_runner.synchronize()
        # Start profiler at the specified step
        if enable_profile_decode and i == profile_start:
            trace_filename_decode = _create_torch_profiler_filename(
                profile_prefix, batch_size, input_len, output_len, "decode"
            )
            profiler = start_profile(
                profile_activities,
                profile_record_shapes=profile_record_shapes,
                rank_print=rank_print,
                trace_filename=trace_filename_decode,
            )

        comm_profile.set_phase("decode")
        comm_profile.set_decode_step(i)
        comm_profile.set_active_batch_size(active_batch_size)
        tic = time.perf_counter()
        next_token_ids, _ = model_runner.decode(next_token_ids, batch)
        model_runner.synchronize()
        latency = time.perf_counter() - tic
        comm_profile.set_phase(None)
        comm_profile.set_decode_step(None)
        comm_profile.set_active_batch_size(None)

        # Stop profiler after the specified number of steps
        if enable_profile_decode and profiler is not None and i >= profile_end - 1:
            stop_profile(
                profiler,
                profile_activities,
                rank_print=rank_print,
                save_trace=True,
                trace_filename=trace_filename_decode,
                stage="decode",
            )
            profiler = None

        tot_latency += latency
        throughput = active_batch_size / latency
        decode_latencies.append(latency)
        decode_step_records.append(
            {
                "decode_step": i,
                "active_batch_size": active_batch_size,
                "latency": latency,
            }
        )
        if i < 5 or (log_decode_step > 0 and i % log_decode_step == 0):
            rank_print(
                f"Decode {i}. Batch size: {active_batch_size}, latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
            )

    # Record decode timing from 2nd output
    if output_len > 1:
        med_decode_latency = np.median(decode_latencies)
        med_decode_throughput = np.median(
            [
                item["active_batch_size"] / item["latency"]
                for item in decode_step_records
            ]
        )
        rank_print(
            f"Decode.  median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    measurement_results["decode_step_records"] = decode_step_records
    total_tokens = sum(effective_input_lens) + sum(effective_output_lens)
    throughput = total_tokens / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput

    if enable_comm_profile:
        local_comm_profile = {
            "tp_rank": tp_rank,
            "capture_mode": comm_profile.capture_mode(),
            "raw_events_saved": comm_profile.raw_events_saved(),
            "stats": comm_profile.snapshot(),
            "events": comm_profile.snapshot_events(),
            "event_histograms": comm_profile.snapshot_event_histograms(),
            "events_total": comm_profile.total_events(),
            "events_truncated": (
                comm_profile.raw_events_saved()
                and comm_profile.total_events() > len(comm_profile.snapshot_events())
            ),
        }
        if dist.is_available() and dist.is_initialized():
            gathered_comm_profiles = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_comm_profiles, local_comm_profile)
        else:
            gathered_comm_profiles = [local_comm_profile]
        if tp_rank == 0:
            measurement_results["comm_profile"] = gathered_comm_profiles
            measurement_results["generated_output_tokens"] = (
                effective_output_lens if output_lens_per_request else output_len
            )
            measurement_results["generated_output_tokens_per_request"] = (
                effective_output_lens
            )
            measurement_results["actual_decode_steps"] = len(decode_step_records)

    comm_profile.set_phase(None)
    comm_profile.set_active_batch_size(None)
    comm_profile.set_prefill_chunk(None, None)
    model_runner.cleanup(batch)
    return measurement_results


def latency_test(
    server_args,
    port_args,
    bench_args,
    gpu_id,
    tp_rank,
):
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)

    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, tp_rank
        )

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)

    if bench_args.warmup_each_workload:
        rank_print("Warmup is enabled separately for every workload point.")
    else:
        # Preserve the legacy single warmup when the per-workload protocol is off.
        reqs = prepare_synthetic_inputs_for_latency_test(
            bench_args.batch_size[0], bench_args.input_len[0]
        )
        rank_print("Warmup ...")
        latency_test_run_once(
            bench_args.run_name,
            model_runner,
            rank_print,
            reqs,
            bench_args.batch_size[0],
            bench_args.input_len[0],
            min(32, bench_args.output_len[0]),
            log_decode_step=0,
            profile=False,
            profile_record_shapes=False,
            profile_activities=("CPU", "GPU"),
            profile_prefix="",
            profile_stage="all",
            tp_rank=tp_rank,
            profile_start_step=None,
            profile_steps=None,
            enable_comm_profile=False,
        )

    rank_print("Benchmark ...")

    custom_inputs = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
    custom_inputs = [tokenizer.encode(p.strip()) for p in custom_inputs]
    custom_input_len = len(custom_inputs)

    def prepare_workload_inputs(bs, il, bs_aligned_inputs):
        if bench_args.prefill_chunk_size > 0:
            if bs_aligned_inputs:
                full_input_ids = [
                    np.asarray(ids, dtype=np.int32) for ids in bs_aligned_inputs
                ]
            else:
                full_input_ids = np.random.randint(
                    0, 10000, (bs, il), dtype=np.int32
                )
            first_chunk_end = min(bench_args.prefill_chunk_size, il)
            first_chunk_inputs = [ids[:first_chunk_end] for ids in full_input_ids]
            reqs = prepare_synthetic_inputs_for_latency_test(
                bs, first_chunk_end, first_chunk_inputs
            )
        else:
            full_input_ids = None
            reqs = prepare_synthetic_inputs_for_latency_test(bs, il, bs_aligned_inputs)
        return reqs, full_input_ids

    trace_plans = []
    if bench_args.trace_replay_plan:
        plan_path = os.path.abspath(bench_args.trace_replay_plan)
        with open(plan_path) as source:
            trace_plans = [json.loads(line) for line in source if line.strip()]
        if not trace_plans:
            raise ValueError(f"trace replay plan is empty: {plan_path}")
        rank_print(f"Loaded {len(trace_plans)} trace replay workloads from {plan_path}")

    if trace_plans:
        workload_specs = []
        for plan in trace_plans:
            input_lens = tuple(int(value) for value in plan["input_lens_per_request"])
            output_lens = tuple(int(value) for value in plan["output_lens_per_request"])
            if not input_lens or len(input_lens) != len(output_lens):
                raise ValueError(
                    f"invalid trace plan lengths for {plan.get('workload_id')}"
                )
            workload_specs.append(
                {
                    "batch_size": len(input_lens),
                    "input_len": max(input_lens),
                    "output_len": max(output_lens),
                    "input_lens": input_lens,
                    "output_lens": output_lens,
                    "plan": plan,
                }
            )
    else:
        workload_specs = [
            {
                "batch_size": bs,
                "input_len": il,
                "output_len": ol,
                "input_lens": (
                    tuple(bench_args.input_lens_per_request)
                    if bench_args.input_lens_per_request
                    else tuple([il] * bs)
                ),
                "output_lens": (
                    tuple(bench_args.output_lens_per_request)
                    if bench_args.output_lens_per_request
                    else tuple([ol] * bs)
                ),
                "plan": None,
            }
            for bs, il, ol in itertools.product(
                bench_args.batch_size, bench_args.input_len, bench_args.output_len
            )
        ]

    # Run the sweep or the fixed trace-derived workload plan.
    result_list = []
    for spec in workload_specs:
        bs = spec["batch_size"]
        il = spec["input_len"]
        ol = spec["output_len"]
        effective_input_lens = spec["input_lens"]
        effective_output_lens = spec["output_lens"]
        plan = spec["plan"]
        bs_aligned_inputs = []
        if plan is not None:
            seed = int(
                hashlib.sha256(plan["workload_id"].encode()).hexdigest()[:8], 16
            )
            rng = np.random.default_rng(seed)
            bs_aligned_inputs = [
                rng.integers(0, 10000, length, dtype=np.int32)
                for length in effective_input_lens
            ]
        elif bench_args.input_lens_per_request:
            rng = np.random.default_rng(0)
            bs_aligned_inputs = [
                rng.integers(0, 10000, length, dtype=np.int32)
                for length in effective_input_lens
            ]
        elif custom_inputs:
            if custom_input_len == bs:
                bs_aligned_inputs = custom_inputs
            elif custom_input_len > bs:
                rank_print(
                    f"Custom input size ({custom_input_len}) is larger than batch_size ({bs}). "
                    f"Using the first {bs} prompts."
                )
                bs_aligned_inputs = copy.deepcopy(custom_inputs[:bs])
            else:
                rank_print(
                    f"Custom input size ({custom_input_len}) is smaller than batch_size ({bs}). "
                    f"Pad to the desired batch_size with the last prompt."
                )
                bs_aligned_inputs = copy.deepcopy(custom_inputs)
                bs_aligned_inputs.extend(
                    [bs_aligned_inputs[-1]] * (bs - custom_input_len)
                )

        reqs, full_input_ids = prepare_workload_inputs(bs, il, bs_aligned_inputs)

        if bench_args.warmup_each_workload:
            warmup_output_lens = tuple(
                min(32, length) for length in effective_output_lens
            )
            rank_print(
                f"Warmup workload: batch_size={bs}, input_len={il}, "
                f"output_len={min(32, ol)}"
            )
            latency_test_run_once(
                run_name=(
                    f"{bench_args.run_name}:{plan['workload_id']}"
                    if plan is not None
                    else bench_args.run_name
                ),
                model_runner=model_runner,
                rank_print=rank_print,
                reqs=reqs,
                batch_size=bs,
                input_len=il,
                output_len=min(32, ol),
                log_decode_step=0,
                profile=False,
                profile_record_shapes=False,
                profile_activities=("CPU", "GPU"),
                profile_prefix="",
                profile_stage="all",
                tp_rank=tp_rank,
                profile_start_step=None,
                profile_steps=None,
                enable_comm_profile=False,
                output_lens_per_request=warmup_output_lens,
                prefill_chunk_size=bench_args.prefill_chunk_size,
                full_input_ids=full_input_ids,
                input_lens_per_request=effective_input_lens,
            )
            reqs, full_input_ids = prepare_workload_inputs(
                bs, il, bs_aligned_inputs
            )

        profile_this_rank = bench_args.profile and (
            tp_rank == 0 or bench_args.profile_all_ranks
        )
        profile_prefix = (
            f"{bench_args.profile_prefix}_rank{tp_rank}"
            if bench_args.profile_all_ranks
            else bench_args.profile_prefix
        )
        ret = latency_test_run_once(
            (
                f"{bench_args.run_name}:{plan['workload_id']}"
                if plan is not None
                else bench_args.run_name
            ),
            model_runner,
            rank_print,
            reqs,
            bs,
            il,
            ol,
            bench_args.log_decode_step,
            profile_this_rank,
            bench_args.profile_record_shapes if profile_this_rank else False,
            bench_args.profile_activities,
            profile_prefix,
            bench_args.profile_stage,
            tp_rank,
            bench_args.profile_start_step,
            bench_args.profile_steps,
            bench_args.comm_profile,
            bench_args.comm_profile_mode,
            effective_output_lens,
            bench_args.prefill_chunk_size,
            full_input_ids,
            effective_input_lens,
        )
        if ret is not None:
            ret["same_shape_workload_warmup"] = bench_args.warmup_each_workload
            ret["workload_warmup_output_len"] = (
                min(32, ol) if bench_args.warmup_each_workload else None
            )
            if plan is not None:
                ret["trace_replay_plan"] = plan
            result_list.append(ret)

    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")

    if server_args.tp_size > 1:
        destroy_model_parallel()
        destroy_distributed_environment()


def main(server_args, bench_args):
    # Post-init write to the legacy cuda_graph_max_bs_decode field would
    # not propagate to cuda_graph_config; update the decode phase directly.
    if server_args.cuda_graph_config is not None:
        server_args.cuda_graph_config[Phase.DECODE].max_bs = max(bench_args.batch_size)

    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    # Calculate local ranks for multi-node setup
    nranks_per_node = server_args.tp_size // server_args.nnodes
    local_rank_start = server_args.node_rank * nranks_per_node
    local_rank_end = local_rank_start + nranks_per_node

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0, 0)
    else:
        workers = []
        for tp_rank in range(local_rank_start, local_rank_end):
            with maybe_reindex_device_id(tp_rank - local_rank_start) as gpu_id:
                proc = multiprocessing.Process(
                    target=work_func,
                    args=(
                        server_args,
                        port_args,
                        bench_args,
                        gpu_id,
                        tp_rank,
                    ),
                )
                proc.start()
                workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


def cli_main():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    cli_main()
