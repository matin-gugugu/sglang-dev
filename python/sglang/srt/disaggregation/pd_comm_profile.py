"""Sender-side logical KV-transfer profiling for PD disaggregation.

The profiler is disabled unless ``SGLANG_PD_COMM_PROFILE_DIR`` is set.  When
enabled it writes one compact JSONL record for each call made by a
``CommonKVSender`` after TP/CP filtering.  Records contain indices and byte
counts only; tensor contents, token IDs and transport packets are never saved.

This is deliberately a logical data-plane trace.  A request chunk is counted
once even if Mooncake/NIXL expands it into many per-layer descriptors.  The
decode receiver is not instrumented, so the same transfer is not double
counted.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

_OUTPUT_DIR_ENV = "SGLANG_PD_COMM_PROFILE_DIR"
_RUN_ID_ENV = "SGLANG_PD_COMM_PROFILE_RUN_ID"
_SCHEMA_VERSION = 1

_lock = threading.Lock()
_sequence = 0


def is_enabled() -> bool:
    return bool(os.environ.get(_OUTPUT_DIR_ENV))


def _safe(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_.:" else "_" for ch in text)


def _component_counts(state_indices: Optional[Iterable[Any]]) -> list[int]:
    counts = []
    for indices in state_indices or []:
        counts.append(0 if indices is None else len(indices))
    return counts


def record_send(
    *,
    rid: Optional[str],
    bootstrap_room: int,
    backend: str,
    page_start: int,
    page_end: int,
    kv_page_count: int,
    kv_bytes_per_page: int,
    state_indices: Optional[Iterable[Any]],
    state_bytes_per_index: int,
    page_size_tokens: int,
) -> None:
    """Append one sender-counted logical KV-transfer record when enabled."""
    if not is_enabled():
        return

    global _sequence
    component_counts = _component_counts(state_indices)
    kv_logical_bytes = int(kv_page_count) * int(kv_bytes_per_page)
    state_index_count = sum(component_counts)
    state_logical_bytes = state_index_count * int(state_bytes_per_index)
    with _lock:
        sequence = _sequence
        _sequence += 1
        output_dir = Path(os.environ[_OUTPUT_DIR_ENV])
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / (
            f"{_safe(os.environ.get(_RUN_ID_ENV, 'default'))}_"
            f"prefill_pid{os.getpid()}.jsonl"
        )
        row = {
            "schema_version": _SCHEMA_VERSION,
            "capture_mode": "sender_logical_kv_chunk",
            "raw_tensor_contents_saved": False,
            "run_id": os.environ.get(_RUN_ID_ENV, "default"),
            "pid": os.getpid(),
            "sequence": sequence,
            "timestamp_ns": time.time_ns(),
            "rid": rid,
            "bootstrap_room": int(bootstrap_room),
            "backend": backend,
            "page_start": int(page_start),
            "page_end": int(page_end),
            "kv_page_count": int(kv_page_count),
            "page_size_tokens": int(page_size_tokens),
            "kv_bytes_per_page": int(kv_bytes_per_page),
            "kv_logical_bytes": kv_logical_bytes,
            "state_component_index_counts": component_counts,
            "state_index_count": state_index_count,
            "state_bytes_per_index_sum": int(state_bytes_per_index),
            "state_logical_bytes": state_logical_bytes,
            "logical_bytes": kv_logical_bytes + state_logical_bytes,
        }
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()


def reset_for_test() -> None:
    """Reset process-local state for an isolated unit test."""
    global _sequence
    with _lock:
        _sequence = 0
