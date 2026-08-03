#!/usr/bin/env python3
"""Validate histogram-only mixed Decode or chunked Prefill results."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


MODEL_METADATA = {
    "deepseek-v2-lite": {
        "hidden_size": 2048,
        "calls_per_forward": 55,
    },
    "qwen3-8b": {
        "hidden_size": 4096,
        "calls_per_forward": 73,
    },
    "qwen3-30b-a3b": {
        "hidden_size": 2048,
        # TP-only path: two reductions per decoder layer plus the final
        # reduction. The admission smoke must confirm this before Phase 13.
        "calls_per_forward": 97,
    },
}
DTYPE_BYTES = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("mixed-decode", "chunked-prefill"), required=True
    )
    parser.add_argument("--model", choices=tuple(MODEL_METADATA), required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output-lens", type=int, nargs="*")
    parser.add_argument("--chunk-size", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def representative_histograms(record, tp):
    profiles = sorted(record["comm_profile"], key=lambda item: item["tp_rank"])
    assert [profile["tp_rank"] for profile in profiles] == list(range(tp))
    reference = profiles[0]
    assert reference["capture_mode"] == "histogram-only"
    assert reference["raw_events_saved"] is False
    assert reference["events"] == []
    assert reference["events_truncated"] is False
    for profile in profiles[1:]:
        assert profile["stats"] == reference["stats"]
        assert profile["event_histograms"] == reference["event_histograms"]
    return reference["event_histograms"]


def validate_common(record, tp):
    assert record["same_shape_workload_warmup"] is True
    histograms = representative_histograms(record, tp)
    for item in histograms:
        assert item["op"] == "all_reduce"
        assert item["group_size"] == tp
        assert item["count"] > 0
        assert item["input_payload_bytes"] > 0
    return histograms


def expected_decode_histogram(output_lens, metadata):
    expected = defaultdict(int)
    for step in range(max(output_lens) - 1):
        active_batch = sum(step < output_len - 1 for output_len in output_lens)
        if active_batch:
            payload = (
                active_batch * metadata["hidden_size"] * DTYPE_BYTES
            )
            expected[(active_batch, payload)] += metadata["calls_per_forward"]
    return dict(expected)


def validate_mixed(rows, args, metadata):
    assert args.output_lens
    expected = expected_decode_histogram(args.output_lens, metadata)
    support_counts = set()
    decode_calls = set()
    decode_payloads = set()
    for record in rows:
        assert record["batch_size"] == len(args.output_lens)
        assert record["output_len"] == max(args.output_lens)
        assert record["output_lens_per_request"] == args.output_lens
        assert record["generated_output_tokens_per_request"] == args.output_lens
        assert record["actual_decode_steps"] == max(args.output_lens) - 1
        histograms = validate_common(record, args.tp)
        observed = defaultdict(int)
        for item in histograms:
            if item["phase"] != "decode":
                continue
            active_batch = int(item["active_batch_size"])
            payload = int(item["input_payload_bytes"])
            assert item["tensor_shape"] == [
                active_batch,
                metadata["hidden_size"],
            ]
            observed[(active_batch, payload)] += int(item["count"])
        assert dict(observed) == expected, (dict(observed), expected)
        support_counts.add(len(observed))
        decode_calls.add(sum(observed.values()))
        decode_payloads.add(
            sum(
                payload * count
                for (_, payload), count in observed.items()
            )
        )
    assert len(support_counts) == len(decode_calls) == len(decode_payloads) == 1
    return {
        "rows": len(rows),
        "decode_supports": support_counts.pop(),
        "decode_calls": decode_calls.pop(),
        "decode_total_payload_bytes": decode_payloads.pop(),
    }


def chunk_lengths(input_len, chunk_size):
    return [
        min(chunk_size, input_len - start)
        for start in range(0, input_len, chunk_size)
    ]


def validate_chunked(rows, args, metadata):
    assert args.chunk_size > 0
    summaries = []
    for record in rows:
        assert record["prefill_chunk_size"] == args.chunk_size
        assert record["generated_output_tokens"] == record["output_len"]
        assert record["generated_output_tokens_per_request"] == [
            record["output_len"]
        ] * record["batch_size"]
        histograms = validate_common(record, args.tp)
        observed = {}
        for item in histograms:
            if item["phase"] != "prefill":
                continue
            index = int(item["prefill_chunk_index"])
            tokens = int(item["prefill_chunk_tokens"])
            payload = int(item["input_payload_bytes"])
            active_batch = int(item["active_batch_size"])
            assert active_batch == record["batch_size"]
            assert item["tensor_shape"] == [
                record["batch_size"] * tokens,
                metadata["hidden_size"],
            ]
            observed[index] = {
                "tokens": tokens,
                "payload": payload,
                "count": int(item["count"]),
            }
        expected_tokens = chunk_lengths(
            record["input_len"], args.chunk_size
        )
        assert sorted(observed) == list(range(len(expected_tokens)))
        for index, tokens in enumerate(expected_tokens):
            expected_payload = (
                record["batch_size"]
                * tokens
                * metadata["hidden_size"]
                * DTYPE_BYTES
            )
            assert observed[index] == {
                "tokens": tokens,
                "payload": expected_payload,
                "count": metadata["calls_per_forward"],
            }
        assert len(record["prefill_chunk_records"]) == math.ceil(
            record["input_len"] / args.chunk_size
        )
        summaries.append(
            {
                "batch_size": record["batch_size"],
                "input_len": record["input_len"],
                "chunks": len(expected_tokens),
            }
        )
    return {
        "rows": len(rows),
        "min_chunks": min(item["chunks"] for item in summaries),
        "max_chunks": max(item["chunks"] for item in summaries),
        "unique_workloads": len(
            {(item["batch_size"], item["input_len"]) for item in summaries}
        ),
    }


def main():
    args = parse_args()
    rows = read_jsonl(args.result)
    assert len(rows) == args.expected_rows, (
        args.result,
        len(rows),
        args.expected_rows,
    )
    metadata = MODEL_METADATA[args.model]
    if args.mode == "mixed-decode":
        summary = validate_mixed(rows, args, metadata)
    else:
        summary = validate_chunked(rows, args, metadata)
    print(
        json.dumps(
            {
                "validated": str(args.result),
                "mode": args.mode,
                "model": args.model,
                "tp": args.tp,
                **summary,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
