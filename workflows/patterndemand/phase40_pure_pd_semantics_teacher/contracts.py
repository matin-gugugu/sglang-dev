#!/usr/bin/env python3
"""Pure-PD workload, teacher and 12-bin aggregation contracts."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

BIN_EDGES_BYTES = [
    4096.0,
    13777.246867516858,
    46340.95001184158,
    155871.75497763665,
    524288.0,
    1763487.5990421579,
    5931641.601515722,
    19951584.63713749,
    67108864.0,
    225726412.6773962,
    759250124.9940125,
    2553802833.553599,
    8589934592.0,
]


def workload_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    repeats = int(contract["measurement_contract"]["independent_repeats"])
    max_new_tokens = int(contract["measurement_contract"]["max_new_tokens"])
    token_id = int(contract["measurement_contract"]["input_token_id"])
    for scenario in contract["workload_scenarios"]:
        for repeat in range(repeats):
            for request_index, prompt_tokens in enumerate(scenario["prompt_tokens"]):
                rows.append(
                    {
                        "scenario": scenario["name"],
                        "repeat": repeat,
                        "request_index": request_index,
                        "rid": f"p40::{scenario['name']}::rep{repeat}::req{request_index}",
                        "prompt_tokens": int(prompt_tokens),
                        "max_new_tokens": max_new_tokens,
                        "input_token_id": token_id,
                    }
                )
    return rows


def teacher_chunks_for_wave(
    requests: list[dict[str, Any]],
    *,
    chunk_tokens: int,
    page_size_tokens: int,
    kv_bytes_per_page: int,
) -> list[dict[str, Any]]:
    """Replay the frozen FCFS pure-prefill token budget.

    With TP=PP=1, no radix cache, no overlap and a fixed chunk budget, SGLang
    consumes requests in order.  A request that exhausts the remaining batch
    budget becomes the globally continued chunked request for the next pass.
    """
    if chunk_tokens <= 0 or page_size_tokens <= 0:
        raise ValueError("chunk and page size must be positive")
    if chunk_tokens % page_size_tokens:
        raise ValueError("chunk size must be page aligned")
    rows = []
    request_index = 0
    token_offset = 0
    batch_index = 0
    while request_index < len(requests):
        budget = chunk_tokens
        while budget > 0 and request_index < len(requests):
            request = requests[request_index]
            total = int(request["prompt_tokens"])
            remaining = total - token_offset
            send_tokens = min(remaining, budget)
            is_last = send_tokens == remaining
            if not is_last:
                send_tokens -= send_tokens % page_size_tokens
            if send_tokens <= 0:
                break
            page_start = token_offset // page_size_tokens
            page_count = math.ceil(send_tokens / page_size_tokens)
            rows.append(
                {
                    "scenario": request["scenario"],
                    "repeat": int(request["repeat"]),
                    "rid": request["rid"],
                    "batch_index": batch_index,
                    "chunk_index": sum(1 for row in rows if row["rid"] == request["rid"]),
                    "page_start": page_start,
                    "page_end": page_start + page_count,
                    "kv_page_count": page_count,
                    "logical_bytes": page_count * kv_bytes_per_page,
                }
            )
            token_offset += send_tokens
            budget -= send_tokens
            if token_offset == total:
                request_index += 1
                token_offset = 0
            else:
                break
        batch_index += 1
    return rows


def build_teacher(
    contract: dict[str, Any], model_contract: dict[str, Any]
) -> list[dict[str, Any]]:
    all_requests = workload_rows(contract)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in all_requests:
        grouped[(row["scenario"], row["repeat"])].append(row)
    result = []
    for key in sorted(grouped):
        requests = sorted(grouped[key], key=lambda row: row["request_index"])
        result.extend(
            teacher_chunks_for_wave(
                requests,
                chunk_tokens=int(contract["measurement_contract"]["chunked_prefill_tokens"]),
                page_size_tokens=int(contract["measurement_contract"]["page_size_tokens"]),
                kv_bytes_per_page=int(model_contract["derived"]["kv_bytes_per_page"]),
            )
        )
    return result


def bin_index(payload_bytes: int) -> int:
    for index in range(len(BIN_EDGES_BYTES) - 1):
        if payload_bytes < BIN_EDGES_BYTES[index + 1]:
            return index
    return len(BIN_EDGES_BYTES) - 2


def histogram_rows(
    events: Iterable[dict[str, Any]],
    requests: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    request_counts = defaultdict(int)
    for row in requests:
        request_counts[row["scenario"]] += 1
        request_counts["overall"] += 1
    accum: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {"calls": 0.0, "logical_bytes": 0.0}
    )
    for event in events:
        for scenario in (event["scenario"], "overall"):
            key = (scenario, bin_index(int(event["logical_bytes"])))
            accum[key]["calls"] += 1
            accum[key]["logical_bytes"] += int(event["logical_bytes"])
    result = []
    for scenario in sorted(request_counts):
        denominator = request_counts[scenario]
        for index in range(12):
            values = accum[(scenario, index)]
            result.append(
                {
                    "source": source,
                    "scenario": scenario,
                    "bin_index": index,
                    "lower_bytes": BIN_EDGES_BYTES[index],
                    "upper_bytes": BIN_EDGES_BYTES[index + 1],
                    "calls_per_1000_requests": values["calls"] * 1000.0 / denominator,
                    "logical_bytes_per_1000_requests": values["logical_bytes"] * 1000.0 / denominator,
                }
            )
    return result


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL: {path}:{line_number}") from error
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refuse empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))
