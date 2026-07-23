"""Lightweight collective communication counters for benchmark experiments."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import torch

_enabled = False
_current_phase: Optional[str] = None
_current_decode_step: Optional[int] = None
_current_active_batch_size: Optional[int] = None
_current_prefill_chunk_index: Optional[int] = None
_current_prefill_chunk_tokens: Optional[int] = None
_counters: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
    lambda: defaultdict(lambda: {"calls": 0, "bytes": 0})
)
_events: List[Dict[str, Any]] = []
_event_histograms: Dict[tuple, Dict[str, Any]] = {}
_total_events = 0
_MAX_RAW_EVENTS = 20000
_event_sequences: Dict[tuple, int] = defaultdict(int)
_CAPTURE_MODES = {"full-trace", "histogram-only"}
_capture_mode = "full-trace"


def enable(value: bool = True) -> None:
    global _enabled
    _enabled = value


def set_capture_mode(mode: str) -> None:
    global _capture_mode
    if mode not in _CAPTURE_MODES:
        raise ValueError(
            f"Unsupported communication profile mode: {mode}. "
            f"Expected one of {sorted(_CAPTURE_MODES)}."
        )
    _capture_mode = mode


def capture_mode() -> str:
    return _capture_mode


def raw_events_saved() -> bool:
    return _capture_mode == "full-trace"


def is_enabled() -> bool:
    return _enabled


def set_phase(phase: Optional[str]) -> None:
    global _current_phase
    _current_phase = phase


def set_decode_step(step: Optional[int]) -> None:
    global _current_decode_step
    _current_decode_step = step


def set_active_batch_size(size: Optional[int]) -> None:
    global _current_active_batch_size
    _current_active_batch_size = size


def set_prefill_chunk(index: Optional[int], tokens: Optional[int]) -> None:
    global _current_prefill_chunk_index, _current_prefill_chunk_tokens
    _current_prefill_chunk_index = index
    _current_prefill_chunk_tokens = tokens


def reset() -> None:
    global _total_events
    _counters.clear()
    _events.clear()
    _event_histograms.clear()
    _event_sequences.clear()
    _total_events = 0
    set_decode_step(None)
    set_active_batch_size(None)
    set_prefill_chunk(None, None)


def tensor_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, (list, tuple)):
        return sum(tensor_nbytes(item) for item in value)
    return 0


def record(
    op: str,
    *values: Any,
    bytes_override: Optional[int] = None,
    output_value: Any = None,
    group_id: Optional[str] = None,
    group_size: Optional[int] = None,
    group_ranks: Optional[Sequence[int]] = None,
    rank: Optional[int] = None,
) -> None:
    if not _enabled or _current_phase is None:
        return
    total_bytes = (
        int(bytes_override)
        if bytes_override is not None
        else sum(tensor_nbytes(value) for value in values)
    )
    slot = _counters[_current_phase][op]
    slot["calls"] += 1
    slot["bytes"] += total_bytes

    sequence_key = (_current_phase, _current_decode_step, group_id or "unknown")
    event_seq = _event_sequences[sequence_key]
    _event_sequences[sequence_key] += 1
    first_tensor = next(
        (value for value in values if isinstance(value, torch.Tensor)), None
    )
    output_bytes = tensor_nbytes(output_value)
    dtype = (
        str(first_tensor.dtype).removeprefix("torch.")
        if first_tensor is not None
        else None
    )
    tensor_shape = list(first_tensor.shape) if first_tensor is not None else None
    event = {
        "phase": _current_phase,
        "decode_step": _current_decode_step,
        "active_batch_size": _current_active_batch_size,
        "prefill_chunk_index": _current_prefill_chunk_index,
        "prefill_chunk_tokens": _current_prefill_chunk_tokens,
        "event_seq_in_step": event_seq,
        "op": op,
        "group_id": group_id or "unknown",
        "group_size": group_size,
        "group_ranks": list(group_ranks) if group_ranks is not None else None,
        "rank": rank,
        "input_payload_bytes": total_bytes,
        "output_payload_bytes": output_bytes,
        "dtype": dtype,
        "tensor_shape": tensor_shape,
    }

    global _total_events
    _total_events += 1
    if raw_events_saved() and len(_events) < _MAX_RAW_EVENTS:
        _events.append(event)

    histogram_key = (
        _current_phase,
        op,
        group_id or "unknown",
        group_size,
        _current_active_batch_size,
        _current_prefill_chunk_index,
        _current_prefill_chunk_tokens,
        total_bytes,
        output_bytes,
        dtype,
        tuple(tensor_shape or ()),
    )
    slot = _event_histograms.get(histogram_key)
    if slot is None:
        slot = {
            "phase": _current_phase,
            "op": op,
            "group_id": group_id or "unknown",
            "group_size": group_size,
            "active_batch_size": _current_active_batch_size,
            "prefill_chunk_index": _current_prefill_chunk_index,
            "prefill_chunk_tokens": _current_prefill_chunk_tokens,
            "input_payload_bytes": total_bytes,
            "output_payload_bytes": output_bytes,
            "dtype": dtype,
            "tensor_shape": tensor_shape,
            "count": 0,
            "first_decode_step": _current_decode_step,
            "last_decode_step": _current_decode_step,
        }
        _event_histograms[histogram_key] = slot
    slot["count"] += 1
    if _current_decode_step is not None:
        first = slot["first_decode_step"]
        slot["first_decode_step"] = (
            _current_decode_step if first is None else min(first, _current_decode_step)
        )
        last = slot["last_decode_step"]
        slot["last_decode_step"] = (
            _current_decode_step if last is None else max(last, _current_decode_step)
        )


def snapshot() -> Dict[str, Dict[str, Dict[str, int]]]:
    return copy.deepcopy({phase: dict(ops) for phase, ops in _counters.items()})


def snapshot_events() -> List[Dict[str, Any]]:
    return copy.deepcopy(_events)


def snapshot_event_histograms() -> List[Dict[str, Any]]:
    return copy.deepcopy(list(_event_histograms.values()))


def total_events() -> int:
    return _total_events
