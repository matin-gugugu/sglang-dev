"""Lightweight collective communication counters for benchmark experiments."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, Optional

import torch

_enabled = False
_current_phase: Optional[str] = None
_counters: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
    lambda: defaultdict(lambda: {"calls": 0, "bytes": 0})
)


def enable(value: bool = True) -> None:
    global _enabled
    _enabled = value


def is_enabled() -> bool:
    return _enabled


def set_phase(phase: Optional[str]) -> None:
    global _current_phase
    _current_phase = phase


def reset() -> None:
    _counters.clear()


def tensor_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, (list, tuple)):
        return sum(tensor_nbytes(item) for item in value)
    return 0


def record(op: str, *values: Any, bytes_override: Optional[int] = None) -> None:
    if not _enabled or _current_phase is None:
        return
    total_bytes = int(bytes_override) if bytes_override is not None else sum(
        tensor_nbytes(value) for value in values
    )
    slot = _counters[_current_phase][op]
    slot["calls"] += 1
    slot["bytes"] += total_bytes


def snapshot() -> Dict[str, Dict[str, Dict[str, int]]]:
    return copy.deepcopy({phase: dict(ops) for phase, ops in _counters.items()})
