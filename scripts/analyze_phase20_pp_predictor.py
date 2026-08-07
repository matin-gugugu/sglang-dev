#!/usr/bin/env python3
"""Build and evaluate the Phase-20 pure-PP PatternDemand predictor.

The label is the sender-side, per-forward-boundary logical demand histogram.
It deliberately excludes the trivial ``pp_size - 1`` boundary multiplier.  The
expanded pipeline demand is retained as a derived column for the downstream
topology cost stage.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


BIN_COUNT = 12
MIN_PAYLOAD = 4 * 1024
MAX_PAYLOAD = 8 * 1024 * 1024 * 1024
BIN_EDGES = np.geomspace(MIN_PAYLOAD, MAX_PAYLOAD, BIN_COUNT + 1)
BIN_CENTERS = np.sqrt(BIN_EDGES[:-1] * BIN_EDGES[1:])
HIDDEN_SIZE = 4096
DTYPE_BYTES = 2
PROXY_TENSOR_COUNT = 2
PAYLOAD_PER_TOKEN = HIDDEN_SIZE * DTYPE_BYTES
SEED = 20260808


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiment-results/phase19_pp_pattern/qwen3-8b-formal-v3"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment-results/phase20_pp_predictor/qwen3-8b-v1"),
    )
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    return p.parse_args()


def payload_bin(payload):
    return int(np.clip(np.searchsorted(BIN_EDGES, payload, side="right") - 1, 0, BIN_COUNT - 1))


def vector_from_payload_hist(hist):
    calls = np.zeros(BIN_COUNT, dtype=np.float64)
    byte = np.zeros(BIN_COUNT, dtype=np.float64)
    for payload, count in hist.items():
        idx = payload_bin(int(payload))
        calls[idx] += float(count)
        byte[idx] += float(payload) * float(count)
    return calls, byte


def strip_repeat(workload_id):
    return re.sub(r"/r\d+$", "", workload_id)


def load_cell(cell):
    cfg = json.loads((cell / "run_config.json").read_text())
    clients = [json.loads(line) for line in (cell / "client_results.jsonl").read_text().splitlines() if line]
    profiles = []
    for path in sorted((cell / "profile").glob("*.json")):
        profiles.append(json.loads(path.read_text()))
    return cfg, clients, profiles


def normalized_sender_rows(profile):
    rows = defaultdict(lambda: defaultdict(int))
    for row in profile["histograms"]:
        if row.get("msg_type") != "proxy" or not row.get("workload_id"):
            continue
        key = (row["phase"], row["raw_op"], int(row["payload_bytes"]), row["tensor_name"])
        rows[row["workload_id"]][key] += int(row["count"])
    return rows


def validate_boundaries(profiles):
    senders = sorted((p for p in profiles if int(p["pp_rank"]) < int(p["pp_size"]) - 1), key=lambda p: p["pp_rank"])
    reference = normalized_sender_rows(senders[0])
    for sender in senders[1:]:
        current = normalized_sender_rows(sender)
        if current != reference:
            raise ValueError(
                f"forward-boundary mismatch: pp0 and pp{sender['pp_rank']} do not have identical logical demand"
            )
    return senders[0], len(senders)


def extract_truth(profile):
    truth = defaultdict(lambda: defaultdict(int))
    for row in profile["histograms"]:
        if row.get("msg_type") != "proxy" or not row.get("workload_id"):
            continue
        truth[(row["workload_id"], row["phase"])][int(row["payload_bytes"])] += int(row["count"])
    return truth


def prefill_h0(input_lens, max_microbatch, chunk_tokens):
    """Greedy token-budget/request-cap proxy for PP prefill forward batches."""
    remaining = list(map(int, input_lens))
    batches = []
    while any(x > 0 for x in remaining):
        budget = chunk_tokens
        used = 0
        tokens = 0
        for idx in range(len(remaining)):
            if remaining[idx] <= 0 or used >= max_microbatch or budget <= 0:
                continue
            take = min(remaining[idx], budget)
            remaining[idx] -= take
            budget -= take
            tokens += take
            used += 1
        if tokens <= 0:
            raise RuntimeError("H0 prefill simulator made no progress")
        batches.append(tokens)
    return batches


def decode_h0(input_lens, output_lens, max_microbatch, chunk_tokens):
    """Static grouping H0; the residual learns PP scheduler merge/split effects."""
    groups = []
    current = []
    current_tokens = 0
    for idx, length in enumerate(input_lens):
        if current and (len(current) >= max_microbatch or current_tokens + length > chunk_tokens):
            groups.append(current)
            current = []
            current_tokens = 0
        if length > chunk_tokens:
            if current:
                groups.append(current)
                current = []
                current_tokens = 0
            groups.append([idx])
        else:
            current.append(idx)
            current_tokens += length
    if current:
        groups.append(current)

    payloads = defaultdict(int)
    for group in groups:
        max_m = max(output_lens[i] for i in group)
        # Prefill emits the first token; Decode contains the remaining M_i - 1 forwards.
        for step in range(1, max_m):
            active = sum(int(output_lens[i] > step) for i in group)
            if active:
                payloads[active * PAYLOAD_PER_TOKEN] += PROXY_TENSOR_COUNT
    return payloads


def h0_hist(input_lens, output_lens, phase, max_microbatch, chunk_tokens):
    if phase == "prefill":
        out = defaultdict(int)
        for tokens in prefill_h0(input_lens, max_microbatch, chunk_tokens):
            out[tokens * PAYLOAD_PER_TOKEN] += PROXY_TENSOR_COUNT
        return out
    return decode_h0(input_lens, output_lens, max_microbatch, chunk_tokens)


def q(values, quantile):
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def feature_vector(client, phase, chunk_tokens):
    ins = list(map(int, client["input_lens"]))
    outs = list(map(int, client["actual_output_lens"]))
    feats = [
        math.log1p(len(ins)),
        math.log1p(sum(ins)),
        math.log1p(np.mean(ins)),
        math.log1p(max(ins)),
        math.log1p(np.std(ins)),
        math.log1p(q(ins, 0.50)),
        math.log1p(q(ins, 0.90)),
        math.log1p(sum(outs)),
        math.log1p(np.mean(outs)),
        math.log1p(max(outs)),
        math.log1p(np.std(outs)),
        math.log1p(q(outs, 0.50)),
        math.log1p(q(outs, 0.90)),
    ]
    for step in (2, 8, 16, 32, 64, 96, 127):
        feats.append(sum(v > step for v in outs) / len(outs))
    feats.extend(
        [
            math.log2(int(client["pp_size"])),
            math.log2(int(client["pp_max_micro_batch_size"])),
            math.log2(chunk_tokens),
            float(phase == "prefill"),
            float(phase == "decode"),
            math.log2(HIDDEN_SIZE),
            math.log2(DTYPE_BYTES),
            math.log2(PROXY_TENSOR_COUNT),
        ]
    )
    return np.asarray(feats, dtype=np.float64)


def encode_target(calls, byte):
    call_total = max(float(calls.sum()), 1e-9)
    byte_total = max(float(byte.sum()), 1e-9)
    return np.concatenate(
        [
            [math.log1p(call_total), math.log1p(byte_total)],
            np.log1p(calls / call_total * 100.0),
            np.log1p(byte / byte_total * 100.0),
        ]
    )


def decode_target(z):
    call_total = max(math.expm1(float(np.clip(z[0], 0, 30))), 0.0)
    byte_total = max(math.expm1(float(np.clip(z[1], 0, 40))), 0.0)
    call_share = np.maximum(np.expm1(np.clip(z[2 : 2 + BIN_COUNT], 0, 8)), 0)
    byte_share = np.maximum(np.expm1(np.clip(z[2 + BIN_COUNT :], 0, 8)), 0)
    if call_share.sum() == 0:
        call_share[0] = 1
    if byte_share.sum() == 0:
        byte_share[0] = 1
    return call_total * call_share / call_share.sum(), byte_total * byte_share / byte_share.sum()


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 48),
            nn.SiLU(),
            nn.Linear(48, 48),
            nn.SiLU(),
            nn.Linear(48, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_predict(x, y, train_idx, valid_idx, test_idx, epochs, patience):
    torch.manual_seed(SEED)
    mean = x[train_idx].mean(axis=0)
    std = x[train_idx].std(axis=0)
    std[std < 1e-8] = 1.0
    xx = (x - mean) / std
    model = MLP(x.shape[1], y.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    xt = torch.tensor(xx, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    best_state = None
    best_loss = float("inf")
    stale = 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xt[train_idx]), yt[train_idx])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(xt[valid_idx]), yt[valid_idx]))
        if val < best_loss - 1e-6:
            best_loss = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(xt[test_idx]).numpy()
    return pred


def oracle_collapsed(calls, byte, groups):
    pc = np.zeros(BIN_COUNT)
    pb = np.zeros(BIN_COUNT)
    for inds in groups:
        c = calls[inds].sum()
        b = byte[inds].sum()
        if c <= 0:
            continue
        avg = b / c
        target = inds[int(np.argmin(np.abs(BIN_CENTERS[inds] - avg)))]
        pc[target] += c
        pb[target] += b
    return pc, pb


def safe_pct(num, den):
    return abs(num - den) / max(abs(den), 1e-9) * 100.0


def sample_metrics(tc, tb, pc, pb):
    tc_total, tb_total = tc.sum(), tb.sum()
    pc_total, pb_total = pc.sum(), pb.sum()
    ts = tc / max(tc_total, 1e-9)
    ps = pc / max(pc_total, 1e-9)
    tb_share = tb / max(tb_total, 1e-9)
    pb_share = pb / max(pb_total, 1e-9)
    return {
        "calls_ape": safe_pct(pc_total, tc_total),
        "bytes_ape": safe_pct(pb_total, tb_total),
        "calls_l1": float(np.abs(ps - ts).sum()),
        "bytes_l1": float(np.abs(pb_share - tb_share).sum()),
        "calls_emd": float(np.abs(np.cumsum(ps) - np.cumsum(ts)).sum() / (BIN_COUNT - 1)),
    }


def select_validation(samples, train_idx):
    names = sorted({samples[i]["workload"] for i in train_idx})
    valid_names = {name for name in names if int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 5 == 0}
    if not valid_names:
        valid_names = {names[0]}
    valid = np.asarray([i for i in train_idx if samples[i]["workload"] in valid_names], dtype=int)
    train = np.asarray([i for i in train_idx if samples[i]["workload"] not in valid_names], dtype=int)
    if not len(train) or not len(valid):
        cut = max(1, len(train_idx) // 5)
        valid, train = np.asarray(train_idx[:cut]), np.asarray(train_idx[cut:])
    return train, valid


def build_folds(samples):
    folds = []
    dimensions = {
        "workload": sorted({s["workload"] for s in samples}),
        "strategy": sorted({s["strategy"] for s in samples}),
        "pp_size": sorted({str(s["pp_size"]) for s in samples}),
    }
    for dimension, values in dimensions.items():
        for value in values:
            test = []
            train = []
            for idx, sample in enumerate(samples):
                actual = str(sample[dimension])
                (test if actual == str(value) else train).append(idx)
            tr, va = select_validation(samples, train)
            folds.append((dimension, str(value), tr, va, np.asarray(test, dtype=int)))
    return folds


def write_gzip_csv(path, rows):
    if not rows:
        return
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repeats = defaultdict(list)
    boundary_checks = []
    for cell in sorted(args.input_dir.glob("pp*/mb*")):
        if not (cell / "DONE").exists():
            continue
        cfg, clients, profiles = load_cell(cell)
        representative, boundary_count = validate_boundaries(profiles)
        truth = extract_truth(representative)
        boundary_checks.append(
            {"cell": str(cell.relative_to(args.input_dir)), "boundaries": boundary_count, "status": "PASS"}
        )
        for client in clients:
            for phase in ("prefill", "decode"):
                key = (client["workload_id"], phase)
                if key not in truth:
                    raise ValueError(f"missing truth: {key} in {cell}")
                base = strip_repeat(client["workload_id"])
                repeats[(base, phase)].append((client, truth[key], cfg))

    samples = []
    repeat_checks = []
    for (base, phase), rows in sorted(repeats.items()):
        if len(rows) != 3:
            raise ValueError(f"expected 3 repeats for {base}/{phase}, got {len(rows)}")
        signatures = [{int(k): int(v) for k, v in hist.items()} for _, hist, _ in rows]
        repeat_equal = signatures[0] == signatures[1] == signatures[2]
        if not repeat_equal:
            raise ValueError(f"repeat histogram mismatch for {base}/{phase}")
        client, exact_hist, cfg = rows[0]
        h0 = h0_hist(
            client["input_lens"],
            client["actual_output_lens"],
            phase,
            int(client["pp_max_micro_batch_size"]),
            int(cfg["chunked_prefill_size"]),
        )
        calls, byte = vector_from_payload_hist(exact_hist)
        h0_calls, h0_byte = vector_from_payload_hist(h0)
        sample = {
            "sample_id": f"{base}/{phase}",
            "model": client["model"],
            "workload": client["workload"],
            "phase": phase,
            "strategy": client["strategy"],
            "pp_size": int(client["pp_size"]),
            "boundary_count": int(client["pp_size"]) - 1,
            "pp_max_micro_batch_size": int(client["pp_max_micro_batch_size"]),
            "chunked_prefill_size": int(cfg["chunked_prefill_size"]),
            "batch_size": len(client["input_lens"]),
            "input_lens": list(map(int, client["input_lens"])),
            "output_lens": list(map(int, client["actual_output_lens"])),
            "calls": calls,
            "bytes": byte,
            "h0_calls": h0_calls,
            "h0_bytes": h0_byte,
            "features": feature_vector(client, phase, int(cfg["chunked_prefill_size"])),
        }
        samples.append(sample)
        repeat_checks.append({"sample_id": sample["sample_id"], "repeat_count": 3, "exact": True})

    x = np.stack([s["features"] for s in samples])
    y = np.stack([encode_target(s["calls"], s["bytes"]) for s in samples])
    h0_y = np.stack([encode_target(s["h0_calls"], s["h0_bytes"]) for s in samples])
    residual = np.clip(y - h0_y, -4.0, 4.0)

    predictions = []
    fold_summaries = []
    for dimension, held_out, train_idx, valid_idx, test_idx in build_folds(samples):
        direct = train_predict(x, y, train_idx, valid_idx, test_idx, args.epochs, args.patience)
        residual_pred = train_predict(x, residual, train_idx, valid_idx, test_idx, args.epochs, args.patience)
        for local_idx, sample_idx in enumerate(test_idx):
            sample = samples[sample_idx]
            tc, tb = sample["calls"], sample["bytes"]
            methods = {
                "total_bytes_oracle": oracle_collapsed(tc, tb, [np.arange(BIN_COUNT)]),
                "three_bin_oracle": oracle_collapsed(
                    tc, tb, [np.arange(0, 4), np.arange(4, 8), np.arange(8, 12)]
                ),
                "structured_h0": (sample["h0_calls"], sample["h0_bytes"]),
                "direct_dnn": decode_target(direct[local_idx]),
                "h0_dnn_residual": decode_target(h0_y[sample_idx] + residual_pred[local_idx]),
            }
            for method, (pc, pb) in methods.items():
                metric = sample_metrics(tc, tb, pc, pb)
                predictions.append(
                    {
                        "fold_dimension": dimension,
                        "held_out": held_out,
                        "sample_id": sample["sample_id"],
                        "phase": sample["phase"],
                        "pp_size": sample["pp_size"],
                        "strategy": sample["strategy"],
                        "method": method,
                        **metric,
                        "true_calls": float(tc.sum()),
                        "pred_calls": float(pc.sum()),
                        "true_bytes": float(tb.sum()),
                        "pred_bytes": float(pb.sum()),
                    }
                )
        fold_summaries.append(
            {
                "fold_dimension": dimension,
                "held_out": held_out,
                "train": len(train_idx),
                "validation": len(valid_idx),
                "test": len(test_idx),
            }
        )

    metric_rows = []
    for dimension in sorted({p["fold_dimension"] for p in predictions}):
        for method in sorted({p["method"] for p in predictions}):
            for phase in ("all", "prefill", "decode"):
                chosen = [
                    p
                    for p in predictions
                    if p["fold_dimension"] == dimension
                    and p["method"] == method
                    and (phase == "all" or p["phase"] == phase)
                ]
                row = {"fold_dimension": dimension, "method": method, "phase": phase, "samples": len(chosen)}
                for metric in ("calls_ape", "bytes_ape", "calls_l1", "bytes_l1", "calls_emd"):
                    vals = np.asarray([p[metric] for p in chosen])
                    row[f"mean_{metric}"] = float(vals.mean())
                    row[f"p95_{metric}"] = float(np.quantile(vals, 0.95))
                metric_rows.append(row)

    dataset_rows = []
    for s in samples:
        row = {
            "sample_id": s["sample_id"],
            "model": s["model"],
            "workload": s["workload"],
            "phase": s["phase"],
            "strategy": s["strategy"],
            "pp_size": s["pp_size"],
            "boundary_count": s["boundary_count"],
            "pp_max_micro_batch_size": s["pp_max_micro_batch_size"],
            "batch_size": s["batch_size"],
            "input_lens": json.dumps(s["input_lens"]),
            "output_lens": json.dumps(s["output_lens"]),
            "per_boundary_calls": float(s["calls"].sum()),
            "per_boundary_bytes": float(s["bytes"].sum()),
            "pipeline_calls": float(s["calls"].sum() * s["boundary_count"]),
            "pipeline_bytes": float(s["bytes"].sum() * s["boundary_count"]),
            "h0_calls": float(s["h0_calls"].sum()),
            "h0_bytes": float(s["h0_bytes"].sum()),
        }
        for i in range(BIN_COUNT):
            row[f"calls_bin_{i}"] = float(s["calls"][i])
            row[f"bytes_bin_{i}"] = float(s["bytes"][i])
            row[f"h0_calls_bin_{i}"] = float(s["h0_calls"][i])
            row[f"h0_bytes_bin_{i}"] = float(s["h0_bytes"][i])
        dataset_rows.append(row)

    write_gzip_csv(args.output_dir / "dataset_12bin.csv.gz", dataset_rows)
    write_gzip_csv(args.output_dir / "holdout_predictions.csv.gz", predictions)
    with (args.output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    (args.output_dir / "folds.json").write_text(json.dumps(fold_summaries, indent=2) + "\n")
    (args.output_dir / "audit.json").write_text(
        json.dumps(
            {
                "schema_version": "phase20-pp-predictor-audit-v1",
                "status": "PASS",
                "samples": len(samples),
                "unique_request_configurations": len(samples) // 2,
                "raw_repetitions_per_sample": 3,
                "bins": BIN_COUNT,
                "bin_edges_bytes": BIN_EDGES.tolist(),
                "label_scope": "sender-side per-forward-boundary logical proxy demand",
                "boundary_expansion": "multiply calls and bytes by pp_size - 1",
                "boundary_checks": boundary_checks,
                "repeat_checks": repeat_checks,
            },
            indent=2,
        )
        + "\n"
    )

    def metric(dimension, method, phase, name):
        row = next(
            r
            for r in metric_rows
            if r["fold_dimension"] == dimension and r["method"] == method and r["phase"] == phase
        )
        return row[name]

    readme = f"""# Phase 20: pure-PP PatternDemand predictor (Qwen3-8B)

## Scope

This stage converts the archived Phase-19 profiler output into a prediction task:

`traffic/workload profile + PP execution policy + PP size + model structure -> per-boundary 12-bin message histogram`.

The truth label is the histogram emitted by SGLang's histogram-only PP profiler on a representative forward boundary. All `{len(boundary_checks)}` cells passed cross-boundary equality checks, and all `{len(samples)}` phase samples were identical across three repetitions. Pipeline-wide logical demand is derived by multiplying the representative-boundary calls and bytes by `pp_size - 1`.

## Data

- Model: Qwen3-8B (`hidden_size=4096`, BF16, two proxy tensors: `hidden_states` and `residual`).
- Configuration samples: `{len(samples) // 2}`; phase-separated samples: `{len(samples)}`.
- PP sizes: 2/4/8; microbatch caps: 1/4/16; workloads: 13; repeats: 3.
- Labels: exact payload histogram is preserved in the source; the predictor target uses 12 logarithmic payload bins.

## Models and controls

- `total_bytes_oracle`: knows the true calls/bytes but collapses them to one average-size bin. This is an intentionally strong representation-loss control, not a deployable predictor.
- `three_bin_oracle`: knows the true three coarse bins and maps them to 12 bins. This isolates information lost by three hard buckets.
- `structured_h0`: derives forward-message demand from token survival, the 4096-token prefill chunk limit, the microbatch cap, hidden size, dtype and the two PP proxy tensors.
- `direct_dnn`: predicts the 12-bin target directly.
- `h0_dnn_residual`: the DNN only learns the bounded residual in encoded histogram space on top of structured H0.

## Strict holdout headline

Mean errors below are computed only on held-out groups (not random rows):

| holdout | method | calls APE | bytes APE | calls distribution L1 | calls EMD |
|---|---|---:|---:|---:|---:|
| workload | structured H0 | {metric('workload','structured_h0','all','mean_calls_ape'):.2f}% | {metric('workload','structured_h0','all','mean_bytes_ape'):.2f}% | {metric('workload','structured_h0','all','mean_calls_l1'):.4f} | {metric('workload','structured_h0','all','mean_calls_emd'):.4f} |
| workload | direct DNN | {metric('workload','direct_dnn','all','mean_calls_ape'):.2f}% | {metric('workload','direct_dnn','all','mean_bytes_ape'):.2f}% | {metric('workload','direct_dnn','all','mean_calls_l1'):.4f} | {metric('workload','direct_dnn','all','mean_calls_emd'):.4f} |
| workload | H0 + DNN residual | {metric('workload','h0_dnn_residual','all','mean_calls_ape'):.2f}% | {metric('workload','h0_dnn_residual','all','mean_bytes_ape'):.2f}% | {metric('workload','h0_dnn_residual','all','mean_calls_l1'):.4f} | {metric('workload','h0_dnn_residual','all','mean_calls_emd'):.4f} |
| strategy | H0 + DNN residual | {metric('strategy','h0_dnn_residual','all','mean_calls_ape'):.2f}% | {metric('strategy','h0_dnn_residual','all','mean_bytes_ape'):.2f}% | {metric('strategy','h0_dnn_residual','all','mean_calls_l1'):.4f} | {metric('strategy','h0_dnn_residual','all','mean_calls_emd'):.4f} |
| PP size | H0 + DNN residual | {metric('pp_size','h0_dnn_residual','all','mean_calls_ape'):.2f}% | {metric('pp_size','h0_dnn_residual','all','mean_bytes_ape'):.2f}% | {metric('pp_size','h0_dnn_residual','all','mean_calls_l1'):.4f} | {metric('pp_size','h0_dnn_residual','all','mean_calls_emd'):.4f} |

See `metrics.csv` for phase-specific and P95 values. `holdout_predictions.csv.gz` retains every held-out prediction for auditing.

## Interpretation boundary

This closes the first pure-PP predictor on one model under controlled simultaneous arrivals/draining batches. It does **not** yet establish cross-model PP generalization or real online arrival/burst prediction. Qwen3-8B is the only model here, so the structural model fields are constant. A second PP-capable model and trace-derived arrival windows are required before those claims.

This stage predicts logical demand, not latency. PP P2P `op x payload x topology/backend -> latency` curves remain the next independent input to the communication-time equation.
"""
    (args.output_dir / "README.md").write_text(readme)

    hashes = {}
    for path in sorted(args.output_dir.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output_dir / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in hashes.items()))
    (args.output_dir / "DONE").write_text("PASS\n")
    print(json.dumps({"status": "PASS", "samples": len(samples), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
