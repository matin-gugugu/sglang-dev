#!/usr/bin/env python3
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

def canonical_hist(profile):
    histogram = profile.get("event_histograms")
    if histogram is None:
        rebuilt = {}
        for event in profile.get("events", []):
            key = (
                event["phase"], event["op"], event["group_id"],
                event.get("group_size"), event["input_payload_bytes"],
                event.get("output_payload_bytes", 0), event.get("dtype"),
                tuple(event.get("tensor_shape") or ()),
            )
            slot = rebuilt.get(key)
            if slot is None:
                slot = {
                    "phase": event["phase"],
                    "op": event["op"],
                    "group_id": event["group_id"],
                    "group_size": event.get("group_size"),
                    "input_payload_bytes": event["input_payload_bytes"],
                    "output_payload_bytes": event.get("output_payload_bytes", 0),
                    "dtype": event.get("dtype"),
                    "tensor_shape": event.get("tensor_shape"),
                    "count": 0,
                    "first_decode_step": event.get("decode_step"),
                    "last_decode_step": event.get("decode_step"),
                }
                rebuilt[key] = slot
            slot["count"] += 1
            step = event.get("decode_step")
            if step is not None:
                first = slot["first_decode_step"]
                last = slot["last_decode_step"]
                slot["first_decode_step"] = step if first is None else min(first, step)
                slot["last_decode_step"] = step if last is None else max(last, step)
        histogram = list(rebuilt.values())
    return sorted(
        histogram,
        key=lambda x: (
            x["phase"], x["op"], x["group_id"],
            x["input_payload_bytes"], x["output_payload_bytes"],
            tuple(x.get("tensor_shape") or ()),
        ),
    )

def aggregate_row(row, model, parallel_form, parallel_size, source):
    profiles = sorted(row["comm_profile"], key=lambda x: x["tp_rank"])
    reference = canonical_hist(profiles[0])
    rank_hist_consistent = all(canonical_hist(p) == reference for p in profiles[1:])

    summary = defaultdict(lambda: {"count": 0, "payload_bytes": 0})
    records = []
    for item in reference:
        count = int(item["count"])
        payload = int(item["input_payload_bytes"])
        key = (item["phase"], item["op"])
        summary[key]["count"] += count
        summary[key]["payload_bytes"] += count * payload
        records.append(item)

    stats_conservation = True
    for profile in profiles:
        hist = canonical_hist(profile)
        rebuilt = defaultdict(lambda: {"calls": 0, "bytes": 0})
        for item in hist:
            slot = rebuilt[(item["phase"], item["op"])]
            slot["calls"] += int(item["count"])
            slot["bytes"] += int(item["count"]) * int(item["input_payload_bytes"])
        for phase, ops in profile["stats"].items():
            for op, values in ops.items():
                if rebuilt[(phase, op)] != values:
                    stats_conservation = False

    decode_steps = int(row.get("actual_decode_steps", max(row["output_len"] - 1, 0)))
    derived = {}
    for (phase, op), values in sorted(summary.items()):
        denominator = row["input_len"] if phase == "prefill" else max(decode_steps, 1)
        derived.setdefault(phase, {})[op] = {
            **values,
            "calls_per_token": values["count"] / denominator,
            "payload_bytes_per_token": values["payload_bytes"] / denominator,
        }

    return {
        "schema_version": "pattern-demand-v1",
        "features": {
            "model": model,
            "parallel_form": parallel_form,
            "parallel_size": parallel_size,
            "batch_size": row["batch_size"],
            "input_len": row["input_len"],
            "output_len": row["output_len"],
        },
        "execution": {
            "run_name": row["run_name"],
            "source": source,
            "generated_output_tokens": row.get("generated_output_tokens"),
            "actual_decode_steps": decode_steps,
            "prefill_latency_ms": row["prefill_latency"] * 1000,
            "median_decode_latency_ms": row.get("median_decode_latency", 0) * 1000,
        },
        "group_level_message_histogram": records,
        "derived": derived,
        "validation": {
            "rank_count": len(profiles),
            "rank_histogram_consistent": rank_hist_consistent,
            "stats_conservation_passed": stats_conservation,
            "output_length_consistent": (
                row.get("generated_output_tokens", row["output_len"]) == row["output_len"]
            ),
            "raw_events_truncated": any(p.get("events_truncated", False) for p in profiles),
        },
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--parallel-form", default="TP")
    ap.add_argument("--parallel-size", type=int, required=True)
    args = ap.parse_args()

    paths = []
    for pattern in args.input:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("no input files")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fout:
        for path in paths:
            with open(path) as fin:
                for line in fin:
                    row = json.loads(line)
                    result = aggregate_row(
                        row, args.model, args.parallel_form, args.parallel_size, path
                    )
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"wrote {output} from {len(paths)} files")

if __name__ == "__main__":
    main()
