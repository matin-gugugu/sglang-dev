#!/usr/bin/env python3
"""Train one Phase34 six-model TP or PP target-free H0+DNN residual direction."""

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

from train_phase27c_pp_scheduler_feature_predictors import BIN_COUNT, ENCODED_SIZE, parse_histograms, prepare_development, residual_bounds, seed_all
from train_phase31c_known_model_residuals import aggregate, encoded_from_vectors, feature_sets, fixed_prediction_rows, headline, records_for_validation, vectors_from_encoded
from train_phase32b_expanded_residual_search import SplitResidualNet, matrices, scale_fit
from train_phase33c_target_free_selection import bytes_anchor


FOLDS = 5
ALPHAS = (0.25, 0.5, 0.75, 1.0)
MODELS = (
    "deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b",
    "llama-3.2-3b-instruct", "qwen2.5-14b-instruct", "mixtral-8x7b-instruct-v0.1",
)
FORMAL = {
    "tp": {"calls_wape": 0.10, "bytes_wape": 0.02, "mean_histogram_tv": 0.20, "mean_normalized_log_payload_emd": 0.025, "common_reference_cost_wape": 0.05},
    "pp": {"calls_wape": 0.15, "bytes_wape": 0.03, "mean_histogram_tv": 0.22, "mean_normalized_log_payload_emd": 0.04, "common_reference_cost_wape": 0.05},
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallelism", choices=("tp", "pp"), required=True)
    parser.add_argument("--dataset-dir", type=Path, default=root / "experiment-results/phase34b_six_model_hfull_dataset")
    parser.add_argument("--phase34a-dir", type=Path, default=root / "experiment-results/phase34a_six_model_contract")
    parser.add_argument("--phase33a-dir", type=Path, default=root / "experiment-results/phase33a_fresh_data_contract")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda:0")
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
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    def convert(item: object) -> object:
        if isinstance(item, np.generic): return item.item()
        raise TypeError(item.__class__.__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=convert) + "\n")


def target_free(rows: list[dict[str, str]], name: str) -> None:
    forbidden = [key for key in rows[0] if key.startswith("target_")]
    if forbidden: raise RuntimeError(f"{name} exposes target: {forbidden}")


def fold_map(rows: list[dict[str, str]]) -> dict[str, int]:
    segments: dict[str, list[str]] = {}
    for row in rows: segments.setdefault(row["segment"], []).append(row["profile_id"])
    output = {}
    for segment, values in sorted(segments.items()):
        profiles = sorted(set(values), key=lambda value: hashlib.sha256(f"phase34-fold:{segment}:{value}".encode()).hexdigest())
        for index, profile in enumerate(profiles): output[profile] = index % FOLDS
    return output


def raw_head_key(row: dict[str, str], mode: str) -> str:
    if mode == "shared": return "shared"
    if mode == "policy": return row["policy"]
    if mode == "model": return row["model"]
    if mode == "model_policy": return row["model"] + "::" + row["policy"]
    raise ValueError(mode)


def training_heads(rows: list[dict[str, str]], mode: str) -> tuple[list[str], np.ndarray]:
    keys = sorted({raw_head_key(row, mode) for row in rows}); mapping = {key: index for index, key in enumerate(keys)}
    return keys, np.asarray([mapping[raw_head_key(row, mode)] for row in rows], dtype=np.int64)


def inference_heads(rows: list[dict[str, str]], mode: str, keys: list[str]) -> np.ndarray:
    mapping = {key: index for index, key in enumerate(keys)}
    missing = sorted({raw_head_key(row, mode) for row in rows} - set(mapping))
    if missing: raise RuntimeError(f"inference heads absent from checkpoint: {missing}")
    return np.asarray([mapping[raw_head_key(row, mode)] for row in rows], dtype=np.int64)


def config_grid(parallelism: str) -> list[dict]:
    configs = []
    if parallelism == "tp":
        for head in ("shared", "policy", "model"):
            for lr in (1e-3, 3e-3):
                configs.append({"family": "split_total_shape", "head_mode": head, "gate_mode": "none", "engineered": True, "calls_only": True, "learning_rate": lr, "width": 64, "weight_decay": 3e-4, "calls_weight": 7.0, "shape_weight": 2.0, "wape_weight": 3.0, "mape_weight": 0.25, "tv_weight": 2.0, "emd_weight": 1.0, "cost_weight": 3.0, "mb16_weight": 1.0})
        for head in ("shared", "policy", "model_policy"):
            for lr in (1e-3, 3e-3):
                configs.append({"family": "shared_trunk_small_heads", "head_mode": head, "gate_mode": "sample", "engineered": True, "calls_only": True, "learning_rate": lr, "width": 64, "weight_decay": 7e-4, "calls_weight": 9.0, "shape_weight": 2.0, "wape_weight": 4.0, "mape_weight": 0.5, "tv_weight": 2.0, "emd_weight": 1.0, "cost_weight": 4.0, "mb16_weight": 1.0})
        for head, width in (("shared", 32), ("shared", 64), ("policy", 32), ("policy", 64), ("model_policy", 32), ("model_policy", 64)):
            configs.append({"family": "lowdim_cost_protected_gate", "head_mode": head, "gate_mode": "sample", "engineered": True, "calls_only": True, "learning_rate": 1e-3, "width": width, "weight_decay": 1e-3, "calls_weight": 10.0, "shape_weight": 3.0, "wape_weight": 5.0, "mape_weight": 0.75, "tv_weight": 3.0, "emd_weight": 2.0, "cost_weight": 5.0, "mb16_weight": 1.0})
    else:
        for head in ("shared", "policy", "model"):
            for lr in (1e-3, 3e-3):
                configs.append({"family": "pp_split_retrain", "head_mode": head, "gate_mode": "none", "engineered": True, "calls_only": True, "learning_rate": lr, "width": 64, "weight_decay": 3e-4, "calls_weight": 8.0, "shape_weight": 2.0, "wape_weight": 4.0, "mape_weight": 0.5, "tv_weight": 3.0, "emd_weight": 2.0, "cost_weight": 4.0, "mb16_weight": 1.25})
        for head in ("policy", "model", "model_policy"):
            for lr in (1e-3, 3e-3):
                configs.append({"family": "pp_calls_shape_gate", "head_mode": head, "gate_mode": "sample", "engineered": True, "calls_only": True, "learning_rate": lr, "width": 64, "weight_decay": 7e-4, "calls_weight": 10.0, "shape_weight": 3.0, "wape_weight": 5.0, "mape_weight": 0.75, "tv_weight": 3.0, "emd_weight": 2.0, "cost_weight": 5.0, "mb16_weight": 1.5})
        for head in ("shared", "policy", "model_policy"):
            for width in (32, 64):
                configs.append({"family": "pp_mb16_cost_protected", "head_mode": head, "gate_mode": "sample", "engineered": True, "calls_only": True, "learning_rate": 1e-3, "width": width, "weight_decay": 1e-3, "calls_weight": 12.0, "shape_weight": 3.0, "wape_weight": 6.0, "mape_weight": 1.0, "tv_weight": 4.0, "emd_weight": 3.0, "cost_weight": 6.0, "mb16_weight": 2.0})
    if len(configs) != 18: raise RuntimeError(len(configs))
    return configs


def direction_loss(raw_residual: torch.Tensor, h0: torch.Tensor, target: torch.Tensor, bounds: torch.Tensor, anchor_bytes_total: torch.Tensor, sample_weight: torch.Tensor, config: dict) -> torch.Tensor:
    predicted = h0 + raw_residual * bounds
    encoded = nn.functional.smooth_l1_loss(predicted[:, :13], target[:, :13], reduction="none")
    encoded_weight = torch.ones(13, device=predicted.device); encoded_weight[0] = float(config["calls_weight"]); encoded_weight[1:] *= float(config["shape_weight"])
    per_sample = (encoded * encoded_weight).mean(dim=1)
    base = (per_sample * sample_weight).sum() / sample_weight.sum()
    calls = torch.expm1(torch.clamp(predicted[:, 0], 0, 30)); target_calls = torch.expm1(torch.clamp(target[:, 0], 0, 30))
    absolute = torch.abs(calls - target_calls)
    wape = (absolute * sample_weight).sum() / torch.clamp((target_calls * sample_weight).sum(), min=1e-8)
    mape = ((absolute / torch.clamp(target_calls, min=1e-8)) * sample_weight).sum() / sample_weight.sum()
    shape = torch.softmax(predicted[:, 1:13], dim=1); target_shape = torch.softmax(target[:, 1:13], dim=1)
    tv_per = 0.5 * torch.abs(shape - target_shape).sum(dim=1)
    emd_per = torch.abs(torch.cumsum(shape, dim=1) - torch.cumsum(target_shape, dim=1)).mean(dim=1)
    tv = (tv_per * sample_weight).sum() / sample_weight.sum(); emd = (emd_per * sample_weight).sum() / sample_weight.sum()
    target_bytes = torch.expm1(torch.clamp(target[:, BIN_COUNT + 1], 0, 40))
    cost = 5.0 * calls + anchor_bytes_total / 1e5; target_cost = 5.0 * target_calls + target_bytes / 1e5
    cost_wape = (torch.abs(cost - target_cost) * sample_weight).sum() / torch.clamp((target_cost * sample_weight).sum(), min=1e-8)
    return base + float(config["wape_weight"]) * wape + float(config["mape_weight"]) * mape + float(config["tv_weight"]) * tv + float(config["emd_weight"]) * emd + float(config["cost_weight"]) * cost_wape


def fit_fold(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], features: list[str], config: dict, folds: dict[str, int], fold: int, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[dict, np.ndarray]:
    train = np.asarray([i for i, row in enumerate(rows) if folds[row["profile_id"]] != fold], dtype=int)
    validation = np.asarray([i for i, row in enumerate(rows) if folds[row["profile_id"]] == fold], dtype=int)
    total_raw, shape_raw = matrices(rows, features, arrays["h0_encoded"], bool(config["engineered"]))
    total, total_mean, total_std = scale_fit(total_raw, train); shape, shape_mean, shape_std = scale_fit(shape_raw, train)
    keys, heads = training_heads(rows, config["head_mode"])
    bounds = residual_bounds().astype(np.float32); targets = arrays["target_encoded"].astype(np.float32)
    anchored = bytes_anchor(rows, arrays["h0_bytes"], args.parallelism).sum(axis=1).astype(np.float32)
    weights = np.asarray([float(config["mb16_weight"]) if args.parallelism == "pp" and row["policy"] == "mb16" else 1.0 for row in rows], dtype=np.float32)
    seed_all(seed); model = SplitResidualNet(total.shape[1], shape.shape[1], len(keys), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(*(torch.from_numpy(value[train]) for value in (total, shape, heads, arrays["h0_encoded"].astype(np.float32), targets, anchored, weights))), batch_size=args.batch_size, shuffle=True, generator=generator)
    validation_tensors = tuple(torch.from_numpy(value[validation]).to(device) for value in (total, shape, heads, arrays["h0_encoded"].astype(np.float32), targets, anchored, weights))
    bounds_t = torch.from_numpy(bounds).to(device)
    best_state, best_loss, best_epoch, stale = None, math.inf, -1, 0
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            total_x, shape_x, head_x, h0_x, target_x, anchor_x, weight_x = (value.to(device) for value in batch)
            loss = direction_loss(model(total_x, shape_x, head_x), h0_x, target_x, bounds_t, anchor_x, weight_x, config)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(direction_loss(model(*validation_tensors[:3]), validation_tensors[3], validation_tensors[4], bounds_t, validation_tensors[5], validation_tensors[6], config).cpu())
        if validation_loss < best_loss - 1e-6:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), validation_loss, epoch, 0
        else:
            stale += 1
            if stale >= args.patience: break
    if best_state is None: raise RuntimeError("no checkpoint")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        residual = model(torch.from_numpy(total[validation]).to(device), torch.from_numpy(shape[validation]).to(device), torch.from_numpy(heads[validation]).to(device)).cpu().numpy() * bounds
    checkpoint = {
        "schema_version": f"phase34c-{args.parallelism}-six-model-split-calls-shape-fold-v1", "parallelism": args.parallelism,
        "config": config, "feature_names": features, "engineered": bool(config["engineered"]), "head_keys": keys,
        "total_mean": torch.from_numpy(total_mean), "total_std": torch.from_numpy(total_std), "shape_mean": torch.from_numpy(shape_mean), "shape_std": torch.from_numpy(shape_std),
        "model_state": {key: value.detach().cpu() for key, value in best_state.items()}, "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "seed": seed, "fold": fold, "bytes_rule": "lowdim_mean_structural_anchor_preserving_h0_bin_shape",
    }
    return checkpoint, residual


def run_seed(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], features: list[str], config: dict, folds: dict[str, int], seed: int, args: argparse.Namespace, device: torch.device) -> tuple[list[dict], np.ndarray]:
    checkpoints, oof = [], np.zeros((len(rows), ENCODED_SIZE), dtype=np.float32)
    for fold in range(FOLDS):
        checkpoint, residual = fit_fold(rows, arrays, features, config, folds, fold, seed + fold * 1009, args, device)
        indices = [i for i, row in enumerate(rows) if folds[row["profile_id"]] == fold]
        oof[indices] = residual; checkpoints.append(checkpoint)
    return checkpoints, oof


def all_records(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], prediction: tuple[np.ndarray, np.ndarray], parallelism: str, method: str) -> list[dict]:
    compatible = [{**row, "split_role": "development_validation"} for row in rows]
    return records_for_validation(compatible, arrays, prediction, parallelism, method)


def metric_bundle(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], prediction: tuple[np.ndarray, np.ndarray], parallelism: str, method: str) -> dict:
    records = all_records(rows, arrays, prediction, parallelism, method); metrics = aggregate(records)
    return {"records": records, "metrics": metrics, "headline": headline(metrics)}


def evaluate(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], residual: np.ndarray, parallelism: str, method: str) -> dict:
    anchor = bytes_anchor(rows, arrays["h0_bytes"], parallelism)
    h0_bundle = metric_bundle(rows, arrays, (arrays["h0_calls"], arrays["h0_bytes"]), parallelism, "h0")
    h0 = h0_bundle["headline"]; best = None
    h0_mb16 = next((row for row in h0_bundle["metrics"] if row["phase"] == "total" and row["policy"] == "mb16" and row["model"] == "all"), None)
    for alpha in ALPHAS:
        calls, _ = vectors_from_encoded(arrays["h0_encoded"] + alpha * residual)
        value = metric_bundle(rows, arrays, (calls, anchor), parallelism, method); head = value["headline"]
        score = sum(float(head[key]) / threshold for key, threshold in FORMAL[parallelism].items())
        if float(head["calls_wape"]) >= float(h0["calls_wape"]): score += 5.0
        if float(head["common_reference_cost_wape"]) >= float(h0["common_reference_cost_wape"]): score += 5.0
        per_model = [row for row in value["metrics"] if row["phase"] == "total" and row["policy"] == "all" and row["model"] != "all"]
        model_calls_limit = 0.15 if parallelism == "tp" else 0.20
        score += sum(max(0.0, float(row["calls_wape"]) / model_calls_limit - 1.0) + max(0.0, float(row["common_reference_cost_wape"]) / 0.08 - 1.0) for row in per_model)
        mb16 = next((row for row in value["metrics"] if row["phase"] == "total" and row["policy"] == "mb16" and row["model"] == "all"), None)
        if parallelism == "pp" and (mb16 is None or float(mb16["calls_mape"]) >= float(h0_mb16["calls_mape"])): score += 5.0
        candidate = {**value, "alpha": alpha, "score": score, "per_model": per_model, "mb16": mb16, "prediction": (calls, anchor), "h0": h0, "h0_mb16": h0_mb16}
        if best is None or candidate["score"] < best["score"]: best = candidate
    return best


def predict_checkpoint(rows: list[dict[str, str]], h0_encoded: np.ndarray, checkpoint: dict, device: torch.device) -> np.ndarray:
    config = checkpoint["config"]
    total_raw, shape_raw = matrices(rows, checkpoint["feature_names"], h0_encoded, bool(checkpoint["engineered"]))
    total = np.clip((total_raw - checkpoint["total_mean"].numpy()) / checkpoint["total_std"].numpy(), -6, 6).astype(np.float32)
    shape = np.clip((shape_raw - checkpoint["shape_mean"].numpy()) / checkpoint["shape_std"].numpy(), -6, 6).astype(np.float32)
    heads = inference_heads(rows, config["head_mode"], checkpoint["head_keys"])
    model = SplitResidualNet(total.shape[1], shape.shape[1], len(checkpoint["head_keys"]), config).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    with torch.no_grad(): raw = model(torch.from_numpy(total).to(device), torch.from_numpy(shape).to(device), torch.from_numpy(heads).to(device)).cpu().numpy()
    return raw * residual_bounds()


def infer(rows: list[dict[str, str]], bundles: list[list[dict]], alpha: float, parallelism: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    h0_calls, h0_bytes = parse_histograms(rows, "h0"); h0_encoded = encoded_from_vectors(h0_calls, h0_bytes)
    residuals = [predict_checkpoint(rows, h0_encoded, checkpoint, device) for bundle in bundles for checkpoint in bundle]
    calls, _ = vectors_from_encoded(h0_encoded + alpha * np.mean(residuals, axis=0))
    return calls, bytes_anchor(rows, h0_bytes, parallelism)


def main() -> None:
    args = parse_args(); parallelism = args.parallelism
    for name in ("checkpoints", "analysis", "figures", "logs"): (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA unavailable")
    dataset_summary = json.loads((args.dataset_dir / "summary.json").read_text())
    phase34a_summary = json.loads((args.phase34a_dir / "summary.json").read_text())
    if dataset_summary["blind_confirmation_target_state"] != "not_generated" or phase34a_summary["blind_confirmation"]["target_state"] != "not_generated": raise RuntimeError("blind target isolation failed")

    rows = read_csv_gz(args.dataset_dir / f"dataset/{parallelism}_six_model_development_examples.csv.gz")
    new_blind = read_csv_gz(args.phase34a_dir / f"dataset/{parallelism}_blind_confirmation_features.csv.gz")
    old_blind = read_csv_gz(args.phase33a_dir / f"dataset/{parallelism}_blind_confirmation_features.csv.gz")
    target_free(new_blind, "phase34_blind"); target_free(old_blind, "phase33_repeated")
    if set(row["model"] for row in rows) != set(MODELS): raise RuntimeError("six-model dataset mismatch")
    arrays = prepare_development(rows); features = feature_sets(rows)["full"]; folds = fold_map(rows)
    anchor = bytes_anchor(rows, arrays["h0_bytes"], parallelism)
    anchor_rel = np.max(np.abs(anchor.sum(axis=1) - arrays["target_bytes"].sum(axis=1)) / np.maximum(arrays["target_bytes"].sum(axis=1), 1e-12))

    grid_rows, screened = [], []
    for index, config in enumerate(config_grid(parallelism), 1):
        candidate_id = f"{parallelism}34_c{index:02d}_{config['family']}_{config['head_mode']}_lr{config['learning_rate']:g}_w{config['width']}"
        checkpoints, oof = run_seed(rows, arrays, features, config, folds, args.seed + index * 37, args, device)
        result = evaluate(rows, arrays, oof, parallelism, candidate_id)
        grid_rows.append({"candidate_id": candidate_id, **config, "screen_seed": args.seed + index * 37, "alpha": result["alpha"], "score": result["score"], **{key: result["headline"][key] for key in FORMAL[parallelism]}, "mb16_calls_mape": result["mb16"]["calls_mape"] if result["mb16"] else ""})
        screened.append({"candidate_id": candidate_id, "config": config, **result})
        print(json.dumps({"parallelism": parallelism, "screen": index, "candidate_id": candidate_id, "score": result["score"], "headline": result["headline"]}, ensure_ascii=False), flush=True)
    screened.sort(key=lambda value: value["score"])

    inventory, confirmed = [], []
    for rank, candidate in enumerate(screened[:3], 1):
        seed_oof, bundles = [], []
        for seed_offset in (0, 101, 202):
            seed = args.seed + seed_offset
            checkpoints, oof = run_seed(rows, arrays, features, candidate["config"], folds, seed, args, device)
            seed_oof.append(oof); bundles.append(checkpoints)
            path = args.output_dir / "checkpoints" / f"{parallelism}_top{rank}_seed{seed}.pt"
            torch.save({"parallelism": parallelism, "candidate_id": candidate["candidate_id"], "rank": rank, "seed": seed, "folds": checkpoints}, path)
            inventory.append({"parallelism": parallelism, "rank": rank, "candidate_id": candidate["candidate_id"], "seed": seed, "path": str(path.relative_to(args.output_dir)), "sha256": sha256(path), "bytes": path.stat().st_size})
        result = evaluate(rows, arrays, np.mean(seed_oof, axis=0), parallelism, candidate["candidate_id"] + "_3seed")
        confirmed.append({"candidate_id": candidate["candidate_id"], "config": candidate["config"], "bundles": bundles, **result})
        print(json.dumps({"parallelism": parallelism, "confirmed_rank": rank, "candidate_id": candidate["candidate_id"], "score": result["score"], "headline": result["headline"]}, ensure_ascii=False), flush=True)
    confirmed.sort(key=lambda value: value["score"]); best = confirmed[0]
    selected_id = best["candidate_id"] + f"_5fold_3seed_alpha{best['alpha']}"
    h0_bundle = metric_bundle(rows, arrays, (arrays["h0_calls"], arrays["h0_bytes"]), parallelism, f"{parallelism}_h0")

    frozen = []
    for prediction_set, prediction_rows in (("phase34_blind_new", new_blind), ("phase33_blind_repeated", old_blind)):
        h0_calls, h0_bytes = parse_histograms(prediction_rows, "h0")
        prediction = infer(prediction_rows, best["bundles"], float(best["alpha"]), parallelism, device)
        output = fixed_prediction_rows(prediction_rows, {"h0": (h0_calls, h0_bytes), "h0_plus_dnn_residual": prediction}, parallelism, selected_id)
        for row in output: row["prediction_set"] = prediction_set
        frozen.extend(output)
    write_csv(args.output_dir / "analysis/candidate_grid.csv", grid_rows)
    write_csv(args.output_dir / "analysis/checkpoint_inventory.csv", inventory)
    write_csv_gz(args.output_dir / "analysis/grouped_cv_predictions_and_metrics.csv.gz", h0_bundle["records"] + best["records"])
    write_csv_gz(args.output_dir / "analysis/frozen_predictions.csv.gz", frozen)
    frozen_sha = sha256(args.output_dir / "analysis/frozen_predictions.csv.gz")

    per_model = {row["model"]: row for row in best["metrics"] if row["phase"] == "total" and row["policy"] == "all" and row["model"] != "all"}
    per_policy = {row["policy"]: row for row in best["metrics"] if row["phase"] == "total" and row["model"] == "all" and row["policy"] != "all"}
    formal_overall = all(float(best["headline"][key]) <= threshold for key, threshold in FORMAL[parallelism].items())
    positive = float(best["headline"]["calls_wape"]) < float(h0_bundle["headline"]["calls_wape"]) and float(best["headline"]["common_reference_cost_wape"]) < float(h0_bundle["headline"]["common_reference_cost_wape"])
    model_limit = 0.15 if parallelism == "tp" else 0.20
    no_model_severe = all(float(row["calls_wape"]) <= model_limit and float(row["common_reference_cost_wape"]) <= 0.08 for row in per_model.values())
    mb16_guard = True
    if parallelism == "pp":
        predicted_mb16 = per_policy["mb16"]; h0_mb16 = best["h0_mb16"]
        improvement = 1.0 - float(predicted_mb16["calls_mape"]) / max(float(h0_mb16["calls_mape"]), 1e-12)
        worse = [float(predicted_mb16[key]) > 1.1 * float(h0_mb16[key]) for key in ("bytes_wape", "mean_histogram_tv", "common_reference_cost_wape")]
        mb16_guard = improvement > 0 and not all(worse)
    checks = {
        "phase34_blind_target_absent_and_unread": phase34a_summary["blind_confirmation"]["target_state"] == "not_generated",
        "phase33_repeated_target_not_read": True,
        "six_models_all_train_and_validation": set(row["model"] for row in rows) == set(MODELS),
        "exactly_18_screen_candidates": len(grid_rows) == 18,
        "top3_three_seed_fivefold": len(inventory) == 9 and len({row["rank"] for row in inventory}) == 3,
        "profile_grouped_fivefold_no_derived_row_leakage": len(set(folds.values())) == 5 and all(len({folds[row["profile_id"]] for row in rows if row["profile_id"] == profile}) == 1 for profile in folds),
        "bytes_anchor_matches_teacher_all_six": anchor_rel < 1e-10,
        "selected_nonzero_dnn_calls_residual": any(abs(float(row["predicted_total_calls_per_1000"]) - float(next(old for old in frozen if old["prediction_set"] == row["prediction_set"] and old["example_id"] == row["example_id"] and old["method"] == "h0")["predicted_total_calls_per_1000"])) > 1e-6 for row in frozen if row["method"] == "h0_plus_dnn_residual"),
        "frozen_new_and_repeated_sets_before_new_target": {row["prediction_set"] for row in frozen} == {"phase34_blind_new", "phase33_blind_repeated"},
        "pp_mb16_guard_if_applicable": mb16_guard,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": f"phase34c-{parallelism}-six-model-target-free-training-v1", "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "parallelism": parallelism, "device": str(device),
        "data": {"profiles": len(folds), "development_train": 75, "development_validation": 19, "unique_teacher_requests": 35524, "phase_rows": len(rows), "models": list(MODELS)},
        "search": {"regular_candidates": 18, "absolute_limit": 24, "screen_seeds": 1, "top_candidates": 3, "confirmation_seeds": 3, "folds": 5, "stop_reason": "regular search complete; selected top confirmed candidate"},
        "selected": {"candidate_id": selected_id, "config": best["config"], "alpha": best["alpha"], "development_cv_headline": best["headline"], "h0_development_cv_headline": h0_bundle["headline"], "formal_development_pass": formal_overall and positive and no_model_severe and mb16_guard, "positive_vs_h0_calls_and_cost": positive, "no_model_severe_regression": no_model_severe, "pp_mb16_guard_if_applicable": mb16_guard, "per_model": per_model, "per_policy": per_policy, "top3": [{"candidate_id": value["candidate_id"], "score": value["score"], "alpha": value["alpha"], "headline": value["headline"]} for value in confirmed]},
        "bytes_anchor": {"maximum_development_target_relative_error": float(anchor_rel), "definition": "allowed low-dimensional mean × model bytes/token × communication multiplier; H0 bin shape retained"},
        "target_isolation": {"phase34_blind_targets_read": False, "phase33_blind_targets_read": False},
        "frozen_prediction_sha256": frozen_sha, "counts": {"checkpoints": len(inventory), "frozen_rows": len(frozen)}, "checks": checks,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": f"phase34c-{parallelism}-audit-v1", "status": status, "checks": checks, "frozen_prediction_sha256": frozen_sha})
    write_json(args.output_dir / "logs/training.log", {"event": f"phase34c_{parallelism}_six_model_target_free_training_complete", "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "repository_head_at_training": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(), "device": str(device), "phase34_blind_targets_read": False})

    width, height, margin = 1000, 480, 55; maximum = max(float(row["score"]) for row in grid_rows) * 1.05; bar_width = (width - 2 * margin) / len(grid_rows)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="55" y="30" font-family="sans-serif" font-size="20">Phase34 {parallelism.upper()}六模型开发侧候选分数（越低越好）</text>']
    for index, row in enumerate(grid_rows):
        value = float(row["score"]); bar_height = value / maximum * (height - 110); x = margin + index * bar_width; y = height - 55 - bar_height
        svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 3, 1):.2f}" height="{bar_height:.2f}" fill="#2563eb"/>')
    svg.append('</svg>'); (args.output_dir / "figures/candidate_scores.svg").write_text("\n".join(svg) + "\n")
    head = best["headline"]
    (args.output_dir / "README.md").write_text(f"""# Phase 34C-{parallelism.upper()}：六模型H0+DNN residual训练与预测冻结

本方向使用固定94个开发画像、35,524个唯一完整teacher请求和六个模型，共{len(rows):,}条phase样本。全部候选都做profile-grouped五折：同一画像派生的六模型、并行配置、policy和phase始终在同一折。Phase34新确认target与Phase33重复集target均未读取。

常规有限搜索18组，每组1个seed初筛，前三名做3-seed × 5-fold确认。选中`{selected_id}`，保留非零DNN residual；开发侧calls/bytes/TV/EMD/cost WAPE为`{head['calls_wape']:.2%}`、`{head['bytes_wape']:.2%}`、`{head['mean_histogram_tv']:.4f}`、`{head['mean_normalized_log_payload_emd']:.4f}`、`{head['common_reference_cost_wape']:.2%}`。六模型bytes结构锚点最大相对误差为`{anchor_rel:.3e}`。

已对Phase34的12个全新确认画像和Phase33的9个已打开重复画像冻结H0及H0+DNN预测。冻结文件SHA-256为`{frozen_sha}`；只有归档本结果后才能生成Phase34新确认Hfull target。
""")
    (args.output_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS": raise RuntimeError(checks)
    print(json.dumps({"status": status, "parallelism": parallelism, "selected": summary["selected"], "frozen_prediction_sha256": frozen_sha}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
