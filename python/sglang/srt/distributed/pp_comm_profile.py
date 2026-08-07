"""Histogram-only sender-side pipeline-parallel communication profiling.

The profiler is intentionally separate from the one-batch TP profiler.  A PP
server is long lived, so saving every event would create a large raw trace and
would make abrupt server shutdowns difficult to recover from.  Instead, each
PP rank periodically replaces one compact JSON snapshot containing only exact
payload histograms.

Enable it by setting ``SGLANG_PP_COMM_PROFILE_DIR`` to an output directory.
Only tensor sends are counted; the receive side must not record the same P2P
transfer again.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_OUTPUT_DIR_ENV = "SGLANG_PP_COMM_PROFILE_DIR"
_RUN_ID_ENV = "SGLANG_PP_COMM_PROFILE_RUN_ID"
_FLUSH_INTERVAL_ENV = "SGLANG_PP_COMM_PROFILE_FLUSH_INTERVAL"
_SCHEMA_VERSION = 1

_lock = threading.Lock()
_histograms: Dict[tuple, Dict[str, Any]] = {}
_events_total = 0
_events_since_flush = 0
_registered_atexit = False
_identity: Optional[tuple[int, int]] = None


def is_enabled() -> bool:
    return bool(os.environ.get(_OUTPUT_DIR_ENV))


def _safe_run_id() -> str:
    value = os.environ.get(_RUN_ID_ENV, "default")
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def _flush_interval() -> int:
    try:
        return max(1, int(os.environ.get(_FLUSH_INTERVAL_ENV, "32")))
    except ValueError:
        return 32


def _phase_and_batch_shape(
    batch: Any,
) -> tuple[str, Optional[int], Optional[int], Optional[int], Optional[str]]:
    if batch is None:
        return "unknown", None, None, None, None

    mode = getattr(batch, "forward_mode", None)
    mode_name = getattr(mode, "name", "unknown").lower()
    if mode_name == "decode":
        phase = "decode"
    elif mode_name == "mixed":
        phase = "mixed"
    elif mode_name in {
        "extend",
        "target_verify",
        "draft_extend_v2",
        "split_prefill",
        "dllm_extend",
    }:
        phase = "prefill"
    else:
        phase = mode_name

    reqs = getattr(batch, "reqs", None)
    active_batch_size = len(reqs) if reqs is not None else None
    if phase == "decode":
        active_tokens = active_batch_size
    else:
        input_ids = getattr(batch, "input_ids", None)
        if isinstance(input_ids, torch.Tensor):
            active_tokens = int(input_ids.numel())
        else:
            active_tokens = getattr(batch, "extend_num_tokens", None)
            if active_tokens is not None:
                active_tokens = int(active_tokens)

    forward_iter = getattr(batch, "forward_iter", None)
    if forward_iter is not None:
        forward_iter = int(forward_iter)

    # A benchmark can submit batched request IDs as
    # ``<workload_id>::req<N>``.  Keeping only their common workload prefix
    # separates many configurations in one long-lived PP server without
    # saving individual request IDs or raw events.
    workload_ids = {
        str(req.rid).rsplit("::req", 1)[0]
        for req in (reqs or [])
        if getattr(req, "rid", None) is not None and "::req" in str(req.rid)
    }
    workload_id = next(iter(workload_ids)) if len(workload_ids) == 1 else None
    return phase, active_batch_size, active_tokens, forward_iter, workload_id


def record_send(
    tensor_dict: Dict[str, Any],
    *,
    msg_type: str,
    pp_rank: int,
    pp_size: int,
    batch: Any = None,
) -> None:
    """Record actual tensor sends from one PP rank to the next rank.

    A tensor dictionary may contain multiple tensors and ``send_tensor_dict``
    issues one P2P send for each non-empty tensor.  Therefore each tensor is a
    distinct logical message and increments the histogram by one.
    """
    if not is_enabled():
        return

    (
        phase,
        active_batch_size,
        active_tokens,
        forward_iter,
        workload_id,
    ) = _phase_and_batch_shape(batch)
    dst_rank = (int(pp_rank) + 1) % int(pp_size)

    global _events_total, _events_since_flush, _identity, _registered_atexit
    should_flush = False
    with _lock:
        _identity = (int(pp_rank), int(pp_size))
        if not _registered_atexit:
            atexit.register(flush)
            _registered_atexit = True

        for tensor_name, tensor in tensor_dict.items():
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                continue
            payload_bytes = int(tensor.numel() * tensor.element_size())
            dtype = str(tensor.dtype).removeprefix("torch.")
            shape = tuple(int(dim) for dim in tensor.shape)
            transport = "cpu" if tensor.is_cpu else "gpu"
            key = (
                phase,
                msg_type,
                int(pp_rank),
                dst_rank,
                int(pp_size),
                tensor_name,
                payload_bytes,
                dtype,
                shape,
                transport,
                active_batch_size,
                active_tokens,
                workload_id,
            )
            row = _histograms.get(key)
            if row is None:
                row = {
                    "phase": phase,
                    "raw_op": "p2p_send_tensor",
                    "msg_type": msg_type,
                    "boundary": f"pp{pp_rank}->pp{dst_rank}",
                    "src_pp_rank": int(pp_rank),
                    "dst_pp_rank": dst_rank,
                    "pp_size": int(pp_size),
                    "tensor_name": tensor_name,
                    "payload_bytes": payload_bytes,
                    "logical_bytes": 0,
                    "dtype": dtype,
                    "tensor_shape": list(shape),
                    "transport": transport,
                    "active_batch_size": active_batch_size,
                    "active_tokens": active_tokens,
                    "workload_id": workload_id,
                    "count": 0,
                    "first_forward_iter": forward_iter,
                    "last_forward_iter": forward_iter,
                }
                _histograms[key] = row
            row["count"] += 1
            row["logical_bytes"] += payload_bytes
            if forward_iter is not None:
                first = row["first_forward_iter"]
                row["first_forward_iter"] = (
                    forward_iter if first is None else min(first, forward_iter)
                )
                last = row["last_forward_iter"]
                row["last_forward_iter"] = (
                    forward_iter if last is None else max(last, forward_iter)
                )
            _events_total += 1
            _events_since_flush += 1

        should_flush = _events_since_flush >= _flush_interval()

    if should_flush:
        flush()


def flush() -> None:
    """Atomically persist the current histogram snapshot for this process."""
    output_dir = os.environ.get(_OUTPUT_DIR_ENV)
    if not output_dir:
        return

    global _events_since_flush
    with _lock:
        if _identity is None:
            return
        pp_rank, pp_size = _identity
        rows = sorted(
            (dict(row) for row in _histograms.values()),
            key=lambda row: (
                row["phase"],
                row["msg_type"],
                row["workload_id"] or "",
                row["src_pp_rank"],
                row["payload_bytes"],
                row["tensor_name"],
            ),
        )
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "capture_mode": "histogram-only",
            "raw_events_saved": False,
            "run_id": _safe_run_id(),
            "pid": os.getpid(),
            "pp_rank": pp_rank,
            "pp_size": pp_size,
            "events_total": _events_total,
            "generated_at_ns": time.time_ns(),
            "histograms": rows,
        }
        _events_since_flush = 0

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{_safe_run_id()}_pp{pp_rank}_pid{os.getpid()}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)
