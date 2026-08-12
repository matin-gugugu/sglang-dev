#!/usr/bin/env python3
"""Finite CV search for Phase32 TP/PP split-head gated H0+DNN residuals."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_phase27c_pp_scheduler_feature_predictors import (
    BIN_COUNT,
    ENCODED_SIZE,
    feature_matrix,
    is_log_feature,
    parse_histograms,
    prepare_development,
    residual_bounds,
    seed_all,
    target_decode,
)
from train_phase31c_known_model_residuals import (
    BIN_EDGES,
    aggregate,
    encoded_from_vectors,
    feature_sets,
    fixed_prediction_rows,
    headline,
    records_for_validation,
    vectors_from_encoded,
)


ALPHAS = (0.25, 0.5, 0.75, 1.0)
FOLDS = 5
MODELS = ("deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b")
POLICIES = {"tp": ("latency", "balanced", "throughput"), "pp": ("mb1", "mb4", "mb16")}
FORMAL = {
    "tp": {"calls_wape": 0.10, "bytes_wape": 0.02, "mean_histogram_tv": 0.20, "mean_normalized_log_payload_emd": 0.025, "common_reference_cost_wape": 0.05},
    "pp": {"calls_wape": 0.15, "bytes_wape": 0.03, "mean_histogram_tv": 0.22, "mean_normalized_log_payload_emd": 0.04, "common_reference_cost_wape": 0.05},
}
PRIOR_COUNTS = {"tp": 18, "pp": 12}
REGULAR_LIMITS = {"tp": 42, "pp": 30}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=root / "experiment-results/phase31b_known_model_hfull_dataset")
    parser.add_argument("--phase32a-dir", type=Path, default=root / "experiment-results/phase32a_expanded_search_contract")
    parser.add_argument("--phase31c-dir", type=Path, default=root / "experiment-results/phase31c_known_model_residual_training")
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase32b_expanded_residual_search")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode())


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def fold_map(rows: list[dict[str, str]]) -> dict[str, int]:
    segments: dict[str, list[str]] = {}
    for row in rows:
        segments.setdefault(row["segment"], []).append(row["profile_id"])
    output = {}
    for segment, values in sorted(segments.items()):
        unique = sorted(set(values), key=lambda value: hashlib.sha256(f"phase32-fold:{segment}:{value}".encode()).hexdigest())
        for index, profile in enumerate(unique):
            output[profile] = index % FOLDS
    return output


def derived_features(rows: list[dict[str, str]]) -> np.ndarray:
    result = []
    for row in rows:
        joint = np.asarray([float(row[f"feature_profile_joint_lm_{index}"]) for index in range(16)], dtype=np.float64)
        joint = np.maximum(joint, 0); joint /= max(joint.sum(), 1e-12)
        chunk = np.asarray([float(row[f"feature_profile_chunk_output_bucket_joint_{index}"]) for index in range(10)], dtype=np.float64)
        chunk = np.maximum(chunk, 0); chunk /= max(chunk.sum(), 1e-12)
        transition = np.asarray([float(row[f"feature_profile_chunk_class_transition_{index}"]) for index in range(4)], dtype=np.float64)
        transition = np.maximum(transition, 0); transition /= max(transition.sum(), 1e-12)
        survival = np.asarray([float(row[f"feature_profile_survival_m_gt_{value}"]) for value in (1, 8, 16, 32, 64)], dtype=np.float64)
        rolling = np.asarray([float(row[f"feature_profile_rolling_multichunk_fraction_max_{value}"]) for value in (4, 16, 32)], dtype=np.float64)
        entropy = lambda value: float(-(value[value > 0] * np.log(value[value > 0])).sum())
        grid = joint.reshape(4, 4)
        result.append([
            entropy(joint), float(np.square(joint).sum()), float(joint.max()),
            *grid.sum(axis=1).tolist(), *grid.sum(axis=0).tolist(),
            float(np.trace(grid)), entropy(chunk), float(np.square(chunk).sum()), float(chunk.max()),
            entropy(transition), float(transition[1] + transition[2]), float(transition[0] + transition[3]),
            float(survival.mean()), *np.diff(survival).tolist(),
            float(rolling.max() - rolling.min()), float(rolling.mean()),
            math.log1p(max(float(row["feature_profile_request_count"]), 0.0)) * float(row.get("feature_profile_input_multichunk_fraction", 0.0)),
        ])
    return np.asarray(result, dtype=np.float32)


def matrices(rows: list[dict[str, str]], feature_names: list[str], h0_encoded: np.ndarray, engineered: bool) -> tuple[np.ndarray, np.ndarray]:
    base = feature_matrix(rows, feature_names)
    derived = derived_features(rows) if engineered else np.empty((len(rows), 0), dtype=np.float32)
    shape = np.concatenate([base, derived, h0_encoded.astype(np.float32)], axis=1)
    total_names = [name for name in feature_names if not name.startswith("feature_model_") and name not in {"feature_parallel_size_log2"}]
    total = np.concatenate([feature_matrix(rows, total_names), derived, h0_encoded[:, [0, BIN_COUNT + 1]].astype(np.float32)], axis=1)
    return total, shape


def head_keys(rows: list[dict[str, str]], mode: str) -> tuple[list[str], np.ndarray]:
    def value(row: dict[str, str]) -> str:
        if mode == "shared": return "shared"
        if mode == "policy": return row["policy"]
        if mode == "model": return row["model"]
        if mode == "model_policy": return row["model"] + "::" + row["policy"]
        raise ValueError(mode)
    keys = sorted({value(row) for row in rows})
    mapping = {key: index for index, key in enumerate(keys)}
    return keys, np.asarray([mapping[value(row)] for row in rows], dtype=np.int64)


class SplitResidualNet(nn.Module):
    def __init__(self, total_size: int, shape_size: int, heads: int, config: dict):
        super().__init__()
        width = int(config["width"])
        self.total_trunk = nn.Sequential(nn.Linear(total_size, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.shape_trunk = nn.Sequential(nn.Linear(shape_size, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.total_base = nn.Linear(width, 2)
        self.shape_base = nn.Linear(width, 24)
        self.total_delta = nn.ModuleList([nn.Linear(width, 2, bias=False) for _ in range(heads)])
        self.shape_delta = nn.ModuleList([nn.Linear(width, 24, bias=False) for _ in range(heads)])
        for layer in [*self.total_delta, *self.shape_delta]:
            nn.init.zeros_(layer.weight)
        self.gate_mode = config["gate_mode"]
        if self.gate_mode == "sample":
            self.total_gate = nn.Linear(width, 2)
            self.shape_gate = nn.Linear(width, 24)
        elif self.gate_mode == "policy":
            self.policy_gate = nn.Embedding(heads, ENCODED_SIZE)
            nn.init.constant_(self.policy_gate.weight, float(config.get("gate_init", 0.0)))
        self.calls_only = bool(config.get("calls_only", False))

    def forward(self, total_x: torch.Tensor, shape_x: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        total_hidden = self.total_trunk(total_x)
        shape_hidden = self.shape_trunk(shape_x)
        total = self.total_base(total_hidden)
        shape = self.shape_base(shape_hidden)
        if len(self.total_delta) > 1:
            total_extra = torch.zeros_like(total)
            shape_extra = torch.zeros_like(shape)
            for index, (total_layer, shape_layer) in enumerate(zip(self.total_delta, self.shape_delta)):
                mask = head == index
                if torch.any(mask):
                    total_extra[mask] = total_layer(total_hidden[mask])
                    shape_extra[mask] = shape_layer(shape_hidden[mask])
            total = total + total_extra
            shape = shape + shape_extra
        output = torch.cat([total[:, :1], shape[:, :12], total[:, 1:], shape[:, 12:]], dim=1)
        output = torch.tanh(output)
        if self.gate_mode == "sample":
            gate = torch.cat([torch.sigmoid(self.total_gate(total_hidden)[:, :1]), torch.sigmoid(self.shape_gate(shape_hidden)[:, :12]), torch.sigmoid(self.total_gate(total_hidden)[:, 1:]), torch.sigmoid(self.shape_gate(shape_hidden)[:, 12:])], dim=1)
            output = output * gate
        elif self.gate_mode == "policy":
            output = output * torch.sigmoid(self.policy_gate(head))
        if self.calls_only:
            output = torch.cat([output[:, :13], torch.zeros_like(output[:, 13:])], dim=1)
        return output


def scale_fit(values: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[indices].mean(axis=0)
    std = values[indices].std(axis=0); std[std < 1e-6] = 1.0
    return np.clip((values - mean) / std, -6.0, 6.0).astype(np.float32), mean, std


def loss_value(predicted_residual: torch.Tensor, h0: torch.Tensor, target: torch.Tensor, bounds: torch.Tensor, config: dict) -> torch.Tensor:
    predicted = h0 + predicted_residual * bounds
    weights = torch.ones(ENCODED_SIZE, device=predicted.device)
    weights[0] = float(config["calls_weight"])
    weights[BIN_COUNT + 1] = float(config["bytes_weight"])
    weights[1:13] *= float(config["shape_weight"])
    weights[14:26] *= float(config["bytes_shape_weight"])
    base = (nn.functional.smooth_l1_loss(predicted, target, reduction="none") * weights).mean()
    pred_calls = torch.expm1(torch.clamp(predicted[:, 0], 0, 30))
    pred_bytes = torch.expm1(torch.clamp(predicted[:, 13], 0, 40))
    target_calls = torch.expm1(torch.clamp(target[:, 0], 0, 30))
    target_bytes = torch.expm1(torch.clamp(target[:, 13], 0, 40))
    pred_cost = 5.0 * pred_calls + pred_bytes / 1e5
    target_cost = 5.0 * target_calls + target_bytes / 1e5
    cost = nn.functional.smooth_l1_loss(torch.log1p(pred_cost), torch.log1p(target_cost))
    tv = torch.abs(torch.softmax(predicted[:, 1:13], dim=1) - torch.softmax(target[:, 1:13], dim=1)).mean()
    return base + float(config["cost_weight"]) * cost + float(config["tv_weight"]) * tv


def fit_fold(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], feature_names: list[str], config: dict, folds: dict[str, int], fold: int, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[dict, np.ndarray]:
    train = np.asarray([index for index, row in enumerate(rows) if folds[row["profile_id"]] != fold], dtype=int)
    validation = np.asarray([index for index, row in enumerate(rows) if folds[row["profile_id"]] == fold], dtype=int)
    total_raw, shape_raw = matrices(rows, feature_names, arrays["h0_encoded"], bool(config["engineered"]))
    total, total_mean, total_std = scale_fit(total_raw, train)
    shape, shape_mean, shape_std = scale_fit(shape_raw, train)
    keys, heads = head_keys(rows, config["head_mode"])
    bounds = residual_bounds().astype(np.float32)
    targets = arrays["target_encoded"].astype(np.float32)
    seed_all(seed)
    model = SplitResidualNet(total.shape[1], shape.shape[1], len(keys), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(total[train]), torch.from_numpy(shape[train]), torch.from_numpy(heads[train]), torch.from_numpy(arrays["h0_encoded"][train].astype(np.float32)), torch.from_numpy(targets[train])), batch_size=args.batch_size, shuffle=True, generator=generator)
    validation_tensors = tuple(torch.from_numpy(value[validation]).to(device) for value in (total, shape, heads, arrays["h0_encoded"].astype(np.float32), targets))
    bounds_t = torch.from_numpy(bounds).to(device)
    best_state, best_loss, best_epoch, stale = None, math.inf, -1, 0
    history = []
    for epoch in range(args.epochs):
        model.train(); losses = []
        for total_x, shape_x, head_x, h0_x, target_x in loader:
            total_x, shape_x, head_x, h0_x, target_x = (value.to(device) for value in (total_x, shape_x, head_x, h0_x, target_x))
            loss = loss_value(model(total_x, shape_x, head_x), h0_x, target_x, bounds_t, config)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_value(model(*validation_tensors[:3]), validation_tensors[3], validation_tensors[4], bounds_t, config).cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), validation_loss, epoch, 0
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        residual = model(torch.from_numpy(total[validation]).to(device), torch.from_numpy(shape[validation]).to(device), torch.from_numpy(heads[validation]).to(device)).cpu().numpy() * bounds
    checkpoint = {"schema_version": "phase32b-split-residual-fold-v1", "config": config, "feature_names": feature_names, "engineered": bool(config["engineered"]), "head_keys": keys, "total_mean": torch.from_numpy(total_mean), "total_std": torch.from_numpy(total_std), "shape_mean": torch.from_numpy(shape_mean), "shape_std": torch.from_numpy(shape_std), "model_state": {key: value.detach().cpu() for key, value in best_state.items()}, "best_epoch": best_epoch, "best_validation_loss": best_loss, "seed": seed, "fold": fold}
    return checkpoint, residual


def predict_checkpoint(rows: list[dict[str, str]], h0_encoded: np.ndarray, checkpoint: dict, device: torch.device) -> np.ndarray:
    config = checkpoint["config"]
    total_raw, shape_raw = matrices(rows, checkpoint["feature_names"], h0_encoded, bool(checkpoint["engineered"]))
    total = np.clip((total_raw - checkpoint["total_mean"].numpy()) / checkpoint["total_std"].numpy(), -6, 6).astype(np.float32)
    shape = np.clip((shape_raw - checkpoint["shape_mean"].numpy()) / checkpoint["shape_std"].numpy(), -6, 6).astype(np.float32)
    keys, raw_heads = head_keys(rows, config["head_mode"])
    if keys != checkpoint["head_keys"]: raise RuntimeError("head key mismatch")
    model = SplitResidualNet(total.shape[1], shape.shape[1], len(keys), config).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    with torch.no_grad(): raw = model(torch.from_numpy(total).to(device), torch.from_numpy(shape).to(device), torch.from_numpy(raw_heads).to(device)).cpu().numpy()
    return raw * residual_bounds()


def all_records(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], prediction: tuple[np.ndarray, np.ndarray], parallelism: str, method: str) -> list[dict]:
    compatible = [{**row, "split_role": "development_validation"} for row in rows]
    return records_for_validation(compatible, arrays, prediction, parallelism, method)


def evaluate_residual(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], residual: np.ndarray, parallelism: str, method: str) -> dict:
    best = None
    for alpha in ALPHAS:
        prediction = vectors_from_encoded(arrays["h0_encoded"] + alpha * residual)
        records = all_records(rows, arrays, prediction, parallelism, method)
        metrics = aggregate(records); head = headline(metrics)
        h0 = headline(aggregate(all_records(rows, arrays, (arrays["h0_calls"], arrays["h0_bytes"]), parallelism, "h0")))
        score = sum(float(head[key]) / FORMAL[parallelism][key] for key in FORMAL[parallelism])
        if float(head["calls_wape"]) >= float(h0["calls_wape"]): score += 3
        if float(head["common_reference_cost_wape"]) >= float(h0["common_reference_cost_wape"]): score += 3
        per_model = [row for row in metrics if row["phase"] == "total" and row["policy"] == "all" and row["model"] != "all"]
        if parallelism == "tp":
            score += sum(max(0.0, float(row["calls_wape"]) / 0.15 - 1) + max(0.0, float(row["common_reference_cost_wape"]) / 0.08 - 1) for row in per_model)
        else:
            score += sum(max(0.0, float(row["calls_wape"]) / 0.20 - 1) + max(0.0, float(row["common_reference_cost_wape"]) / 0.08 - 1) for row in per_model)
        value = {"alpha": alpha, "score": score, "prediction": prediction, "records": records, "metrics": metrics, "headline": head, "per_model": per_model}
        if best is None or value["score"] < best["score"]: best = value
    return best


def config_grid(parallelism: str) -> list[dict]:
    configs = []
    if parallelism == "tp":
        # 8 split-head, 8 gated, 8 low-dimensional shape/calls variants = 24 new groups.
        for head in ("shared", "policy", "model", "model_policy"):
            for lr in (1e-3, 3e-3):
                configs.append({"family": "tp_split_total_shape", "head_mode": head, "gate_mode": "none", "engineered": False, "calls_only": False, "learning_rate": lr, "width": 64, "weight_decay": 1e-4, "calls_weight": 5, "bytes_weight": 5, "shape_weight": 1, "bytes_shape_weight": 1, "cost_weight": 1, "tv_weight": 1})
        for head in ("shared", "policy", "model", "model_policy"):
            for lr in (1e-3, 3e-3):
                configs.append({"family": "tp_residual_gate", "head_mode": head, "gate_mode": "sample", "engineered": True, "calls_only": False, "learning_rate": lr, "width": 64, "weight_decay": 3e-4, "calls_weight": 8, "bytes_weight": 6, "shape_weight": 1, "bytes_shape_weight": 1, "cost_weight": 2, "tv_weight": 2})
        for head in ("shared", "policy"):
            for calls_only in (False, True):
                for width in (32, 64):
                    configs.append({"family": "tp_lowdim_sequence_shape", "head_mode": head, "gate_mode": "sample", "engineered": True, "calls_only": calls_only, "learning_rate": 1e-3, "width": width, "weight_decay": 1e-3, "calls_weight": 10, "bytes_weight": 8, "shape_weight": 2, "bytes_shape_weight": 2, "cost_weight": 3, "tv_weight": 3})
    else:
        # 12 bytes/cost-protected split heads and 6 MB-independent gates = 18 new groups.
        for head in ("policy", "model_policy"):
            for bytes_weight in (6, 10, 16):
                for lr in (1e-3, 3e-3):
                    configs.append({"family": "pp_bytes_cost_protection", "head_mode": head, "gate_mode": "none", "engineered": True, "calls_only": False, "learning_rate": lr, "width": 64, "weight_decay": 3e-4, "calls_weight": 8, "bytes_weight": bytes_weight, "shape_weight": 2, "bytes_shape_weight": 2, "cost_weight": 4, "tv_weight": 3})
        for bytes_weight in (8, 12, 16):
            for lr in (1e-3, 3e-3):
                configs.append({"family": "pp_mb_independent_gate", "head_mode": "policy", "gate_mode": "policy", "gate_init": 0.0, "engineered": True, "calls_only": False, "learning_rate": lr, "width": 64, "weight_decay": 5e-4, "calls_weight": 10, "bytes_weight": bytes_weight, "shape_weight": 2, "bytes_shape_weight": 2, "cost_weight": 5, "tv_weight": 3})
    expected = REGULAR_LIMITS[parallelism] - PRIOR_COUNTS[parallelism]
    if len(configs) != expected: raise RuntimeError((parallelism, len(configs), expected))
    return configs


def run_seed(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], feature_names: list[str], config: dict, fold_assignments: dict[str, int], seed: int, args: argparse.Namespace, device: torch.device) -> tuple[list[dict], np.ndarray]:
    checkpoints = []
    oof = np.zeros((len(rows), ENCODED_SIZE), dtype=np.float32)
    for fold in range(FOLDS):
        checkpoint, residual = fit_fold(rows, arrays, feature_names, config, fold_assignments, fold, seed + fold * 1009, args, device)
        indices = [index for index, row in enumerate(rows) if fold_assignments[row["profile_id"]] == fold]
        oof[indices] = residual
        checkpoints.append(checkpoint)
    return checkpoints, oof


def main() -> None:
    args = parse_args()
    for name in ("checkpoints", "analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda": raise RuntimeError("CUDA unavailable")
    dataset_summary = json.loads((args.dataset_dir / "summary.json").read_text())
    contract = json.loads((args.phase32a_dir / "summary.json").read_text())
    if dataset_summary["fixed_target_state"] != "not_generated" or contract["new_confirmation"]["target_state"] != "not_generated": raise RuntimeError("target isolation failed")

    grid_rows, inventory, frozen_rows, validation_rows, selected = [], [], [], [], {}
    for parallelism in ("tp", "pp"):
        rows = read_csv_gz(args.dataset_dir / f"dataset/{parallelism}_development_examples.csv.gz")
        original_fixed = read_csv_gz(args.dataset_dir / f"dataset/{parallelism}_fixed_prediction_features.csv.gz")
        new_confirm = read_csv_gz(args.phase32a_dir / f"dataset/{parallelism}_new_confirmation_features.csv.gz")
        if any(name.startswith("target_") for name in set(original_fixed[0]) | set(new_confirm[0])): raise RuntimeError("target exposed in prediction features")
        arrays = prepare_development(rows)
        feature_names = feature_sets(rows)["full"]
        folds = fold_map(rows)
        h0_metrics = aggregate(all_records(rows, arrays, (arrays["h0_calls"], arrays["h0_bytes"]), parallelism, "h0"))
        h0_headline = headline(h0_metrics)
        configs = config_grid(parallelism)
        screened = []
        for index, config in enumerate(configs):
            candidate_id = f"{parallelism}32_c{PRIOR_COUNTS[parallelism] + index + 1:02d}_{config['family']}_{config['head_mode']}_lr{config['learning_rate']:g}_w{config['width']}"
            checkpoints, residual = run_seed(rows, arrays, feature_names, config, folds, args.seed + index * 37, args, device)
            result = evaluate_residual(rows, arrays, residual, parallelism, candidate_id)
            grid_rows.append({"parallelism": parallelism, "candidate_id": candidate_id, **config, "screen_seed": args.seed + index * 37, "alpha": result["alpha"], "score": result["score"], **{key: result["headline"][key] for key in FORMAL[parallelism]}})
            screened.append({"candidate_id": candidate_id, "config": config, "screen_checkpoints": checkpoints, "screen_residual": residual, **result})
        screened.sort(key=lambda value: value["score"])

        confirmed = []
        for rank, candidate in enumerate(screened[:3], 1):
            seed_oof, seed_models = [], []
            for seed_offset in (0, 101, 202):
                seed = args.seed + seed_offset
                checkpoints, oof = run_seed(rows, arrays, feature_names, candidate["config"], folds, seed, args, device)
                seed_oof.append(oof); seed_models.append(checkpoints)
                path = args.output_dir / "checkpoints" / f"{parallelism}_top{rank}_seed{seed}.pt"
                torch.save({"parallelism": parallelism, "candidate_id": candidate["candidate_id"], "rank": rank, "seed": seed, "folds": checkpoints}, path)
                inventory.append({"parallelism": parallelism, "candidate_rank": rank, "candidate_id": candidate["candidate_id"], "seed": seed, "path": str(path.relative_to(args.output_dir)), "sha256": sha256(path), "bytes": path.stat().st_size})
            mean_oof = np.mean(seed_oof, axis=0)
            result = evaluate_residual(rows, arrays, mean_oof, parallelism, candidate["candidate_id"] + "_3seed")
            confirmed.append({"candidate_id": candidate["candidate_id"], "config": candidate["config"], "seed_models": seed_models, "mean_oof": mean_oof, **result})
        confirmed.sort(key=lambda value: value["score"])
        best = confirmed[0]
        selected_id = best["candidate_id"] + f"_5fold_3seed_alpha{best['alpha']}"
        selected[parallelism] = {"candidate_id": selected_id, "config": best["config"], "cv_headline": best["headline"], "cv_score": best["score"], "h0_cv_headline": h0_headline, "top3": [{"candidate_id": value["candidate_id"], "score": value["score"], "alpha": value["alpha"], "headline": value["headline"]} for value in confirmed]}
        validation_rows.extend(all_records(rows, arrays, (arrays["h0_calls"], arrays["h0_bytes"]), parallelism, f"{parallelism}_h0"))
        validation_rows.extend(best["records"])
        for prediction_set, prediction_rows in (("original_fixed", original_fixed), ("new_confirmation", new_confirm)):
            h0_calls, h0_bytes = parse_histograms(prediction_rows, "h0")
            h0_encoded = encoded_from_vectors(h0_calls, h0_bytes)
            residuals = [predict_checkpoint(prediction_rows, h0_encoded, checkpoint, device) for seed_models in best["seed_models"] for checkpoint in seed_models]
            prediction = vectors_from_encoded(h0_encoded + best["alpha"] * np.mean(residuals, axis=0))
            rows_out = fixed_prediction_rows(prediction_rows, {"h0": (h0_calls, h0_bytes), "h0_plus_dnn_residual": prediction}, parallelism, selected_id)
            for row in rows_out: row["prediction_set"] = prediction_set
            frozen_rows.extend(rows_out)

    write_csv(args.output_dir / "analysis/candidate_grid.csv", grid_rows)
    write_csv(args.output_dir / "analysis/checkpoint_inventory.csv", inventory)
    write_csv_gz(args.output_dir / "analysis/grouped_cv_predictions.csv.gz", validation_rows)
    write_csv_gz(args.output_dir / "analysis/frozen_predictions.csv.gz", frozen_rows)
    frozen_sha = sha256(args.output_dir / "analysis/frozen_predictions.csv.gz")
    family_best = {}
    for parallelism in ("tp", "pp"):
        family_best[parallelism] = {}
        for row in [value for value in grid_rows if value["parallelism"] == parallelism]:
            family = row["family"]
            if family not in family_best[parallelism] or float(row["score"]) < float(family_best[parallelism][family]["score"]): family_best[parallelism][family] = row
    # Dependency-free, deterministic candidate-score figure.
    width, height, margin = 1100, 520, 55
    values = [float(row["score"]) for row in grid_rows]
    maximum = max(values) * 1.05
    bar_width = (width - 2 * margin) / len(values)
    colors = {"tp": "#2563eb", "pp": "#ea580c"}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="55" y="30" font-family="sans-serif" font-size="20">Phase32 grouped-CV candidate scores（越低越好）</text>']
    for index, row in enumerate(grid_rows):
        value = float(row["score"]); x = margin + index * bar_width; bar_height = value / maximum * (height - 120); y = height - 65 - bar_height
        svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 2, 1):.2f}" height="{bar_height:.2f}" fill="{colors[row["parallelism"]]}"/>')
    svg.extend([f'<line x1="{margin}" y1="{height-65}" x2="{width-margin}" y2="{height-65}" stroke="#111"/>', '<rect x="780" y="18" width="14" height="14" fill="#2563eb"/><text x="800" y="31" font-family="sans-serif" font-size="14">TP新增24组</text>', '<rect x="900" y="18" width="14" height="14" fill="#ea580c"/><text x="920" y="31" font-family="sans-serif" font-size="14">PP新增18组</text>', '</svg>'])
    (args.output_dir / "figures/candidate_scores.svg").write_text("\n".join(svg) + "\n")
    checks = {
        "phase31_fixed_targets_not_read": dataset_summary["fixed_target_state"] == "not_generated",
        "new_confirmation_targets_not_generated_or_read": contract["new_confirmation"]["target_state"] == "not_generated",
        "tp_new_24_cumulative_42": Counter(row["parallelism"] for row in grid_rows)["tp"] == 24,
        "pp_new_18_cumulative_30": Counter(row["parallelism"] for row in grid_rows)["pp"] == 18,
        "top3_three_seed_each_direction": Counter(row["parallelism"] for row in inventory) == Counter({"tp": 9, "pp": 9}),
        "fivefold_profile_group_isolation": all(len({folds for folds in fold_map(read_csv_gz(args.dataset_dir / f"dataset/{p}_development_examples.csv.gz")).values()}) == 5 for p in ("tp", "pp")),
        "frozen_two_prediction_sets": {row["prediction_set"] for row in frozen_rows} == {"original_fixed", "new_confirmation"},
        "frozen_methods_h0_and_dnn": {row["method"] for row in frozen_rows} == {"h0", "h0_plus_dnn_residual"},
        "selected_nonzero_residual": all(any(abs(float(row["predicted_total_calls_per_1000"]) - float(next(old for old in frozen_rows if old["prediction_set"] == row["prediction_set"] and old["parallelism"] == row["parallelism"] and old["example_id"] == row["example_id"] and old["method"] == "h0")["predicted_total_calls_per_1000"])) > 1e-6 for row in frozen_rows if row["parallelism"] == p and row["method"] == "h0_plus_dnn_residual") for p in ("tp", "pp")),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {"schema_version": "phase32b-expanded-residual-search-v1", "status": status, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "device": str(device), "search": {"prior_counts": PRIOR_COUNTS, "new_counts": {"tp": 24, "pp": 18}, "cumulative_counts": REGULAR_LIMITS, "folds": FOLDS, "screen_seeds": 1, "top_candidates": 3, "confirmation_seeds": 3}, "selected": selected, "family_best": family_best, "frozen_prediction_sha256": frozen_sha, "fixed_targets_read": False, "new_confirmation_targets_read": False, "counts": {"grid_rows": len(grid_rows), "checkpoints": len(inventory), "frozen_rows": len(frozen_rows)}, "checks": checks}
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase32b-audit-v1", "status": status, "checks": checks, "frozen_prediction_sha256": frozen_sha})
    write_json(args.output_dir / "logs/training.log", {"event": "phase32b_expanded_search_complete", "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "repository_head_at_training": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(), "device": str(device), "fixed_targets_read": False, "new_confirmation_targets_read": False})
    (args.output_dir / "README.md").write_text(f"""# Phase 32B：TP/PP扩容有限搜索与预测冻结\n\n本阶段只使用Phase31的39训练、10验证及五折profile-grouped CV；原10固定target和Phase32新确认target均未读取。TP新增24组、累计42组；PP新增18组、累计30组。每个新组初筛1个seed，开发侧前三名做5-fold × 3-seed确认。\n\nTP探索总量/形状分头、共享主干加model/policy小头、sample residual gate和低维顺序/形状摘要；PP探索bytes/cost保护loss和MB独立gate。两个方向均保持`H0 + DNN residual`且residual非零。\n\n选中模型已经同时对不变的原10固定窗口和9个新BurstGPT确认窗口冻结预测，SHA-256为`{frozen_sha}`。下一阶段必须先归档本目录，之后才能生成新确认Hfull target；原固定集后续结果只能称为重复工程复评。\n""")
    (args.output_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS": raise RuntimeError(checks)
    print(json.dumps({"status": status, "selected": selected, "frozen_prediction_sha256": frozen_sha}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
