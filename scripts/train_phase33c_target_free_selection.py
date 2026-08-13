#!/usr/bin/env python3
"""Phase33 target-free TP search and conservative PP bytes calibration."""

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
    parse_histograms,
    prepare_development,
    residual_bounds,
    seed_all,
)
from train_phase31c_known_model_residuals import (
    aggregate,
    encoded_from_vectors,
    feature_sets,
    fixed_prediction_rows,
    headline,
    records_for_validation,
    vectors_from_encoded,
)
from train_phase32b_expanded_residual_search import (
    SplitResidualNet,
    head_keys,
    matrices,
    predict_checkpoint,
    scale_fit,
)


FOLDS = 5
ALPHAS = (0.25, 0.5, 0.75, 1.0)
TP_FORMAL = {
    "calls_wape": 0.10,
    "bytes_wape": 0.02,
    "mean_histogram_tv": 0.20,
    "mean_normalized_log_payload_emd": 0.025,
    "common_reference_cost_wape": 0.05,
}
PP_FORMAL = {
    "calls_wape": 0.15,
    "bytes_wape": 0.03,
    "mean_histogram_tv": 0.22,
    "mean_normalized_log_payload_emd": 0.04,
    "common_reference_cost_wape": 0.05,
}
MODELS = ("deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase33a-dir", type=Path, default=root / "experiment-results/phase33a_fresh_data_contract")
    parser.add_argument("--phase33b-dir", type=Path, default=root / "experiment-results/phase33b_expanded_development_dataset")
    parser.add_argument("--phase31b-dir", type=Path, default=root / "experiment-results/phase31b_known_model_hfull_dataset")
    parser.add_argument("--phase32a-dir", type=Path, default=root / "experiment-results/phase32a_expanded_search_contract")
    parser.add_argument("--phase32b-dir", type=Path, default=root / "experiment-results/phase32b_expanded_residual_search")
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase33c_target_free_model_selection")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=35)
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
    if not rows:
        raise ValueError(path)
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(path)
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    def numpy_value(item: object) -> object:
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Object of type {item.__class__.__name__} is not JSON serializable")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=numpy_value) + "\n")


def target_free(rows: list[dict[str, str]], name: str) -> None:
    forbidden = [key for key in rows[0] if key.startswith("target_")]
    if forbidden:
        raise RuntimeError(f"{name} unexpectedly exposes targets: {forbidden}")


def fold_map(rows: list[dict[str, str]]) -> dict[str, int]:
    segments: dict[str, list[str]] = {}
    for row in rows:
        segments.setdefault(row["segment"], []).append(row["profile_id"])
    result = {}
    for segment, profiles in sorted(segments.items()):
        unique = sorted(set(profiles), key=lambda value: hashlib.sha256(f"phase33-fold:{segment}:{value}".encode()).hexdigest())
        for index, profile in enumerate(unique):
            result[profile] = index % FOLDS
    return result


def bytes_anchor(rows: list[dict[str, str]], h0_bytes: np.ndarray, parallelism: str) -> np.ndarray:
    """Allowed low-dimensional mean plus model prior gives exact structural Hfull bytes total."""
    output = []
    for row, vector in zip(rows, h0_bytes):
        tokens = float(row["feature_profile_input_mean_capped"]) if row["phase"] == "prefill" else max(float(row["feature_profile_output_mean_capped"]) - 1.0, 0.0)
        per_token = float(row["feature_model_payload_bytes_per_active_token_prior"])
        collective_count = float(row["feature_model_logical_collectives_per_forward_prior"]) if parallelism == "tp" else float(row["feature_pp_proxy_tensor_count"])
        total = tokens * per_token * collective_count * 1000.0
        mass = float(vector.sum())
        if mass <= 0:
            raise RuntimeError(f"zero H0 bytes mass: {row['example_id']}")
        output.append(vector * (total / mass))
    return np.stack(output)


def all_records(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], prediction: tuple[np.ndarray, np.ndarray], parallelism: str, method: str) -> list[dict]:
    compatible = [{**row, "split_role": "development_validation"} for row in rows]
    return records_for_validation(compatible, arrays, prediction, parallelism, method)


def metric_bundle(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], prediction: tuple[np.ndarray, np.ndarray], parallelism: str, method: str) -> dict:
    records = all_records(rows, arrays, prediction, parallelism, method)
    metrics = aggregate(records)
    return {"records": records, "metrics": metrics, "headline": headline(metrics)}


def tp_grid() -> list[dict]:
    configs = []
    for head in ("shared", "policy", "model"):
        for lr in (1e-3, 3e-3):
            configs.append({"family": "split_total_shape", "head_mode": head, "gate_mode": "none", "engineered": True, "calls_only": True, "learning_rate": lr, "width": 64, "weight_decay": 3e-4, "calls_weight": 7.0, "shape_weight": 2.0, "wape_weight": 3.0, "mape_weight": 0.25, "tv_weight": 2.0, "emd_weight": 1.0, "cost_weight": 3.0})
    for head in ("shared", "policy", "model_policy"):
        for lr in (1e-3, 3e-3):
            configs.append({"family": "shared_trunk_small_heads", "head_mode": head, "gate_mode": "sample", "engineered": True, "calls_only": True, "learning_rate": lr, "width": 64, "weight_decay": 7e-4, "calls_weight": 9.0, "shape_weight": 2.0, "wape_weight": 4.0, "mape_weight": 0.5, "tv_weight": 2.0, "emd_weight": 1.0, "cost_weight": 4.0})
    for head, width in (("shared", 32), ("shared", 64), ("policy", 32), ("policy", 64), ("model_policy", 32), ("model_policy", 64)):
        configs.append({"family": "lowdim_cost_protected_gate", "head_mode": head, "gate_mode": "sample", "engineered": True, "calls_only": True, "learning_rate": 1e-3, "width": width, "weight_decay": 1e-3, "calls_weight": 10.0, "shape_weight": 3.0, "wape_weight": 5.0, "mape_weight": 0.75, "tv_weight": 3.0, "emd_weight": 2.0, "cost_weight": 5.0})
    if len(configs) != 18:
        raise RuntimeError(len(configs))
    return configs


def tp_loss(raw_residual: torch.Tensor, h0: torch.Tensor, target: torch.Tensor, bounds: torch.Tensor, anchor_bytes_total: torch.Tensor, config: dict) -> torch.Tensor:
    predicted = h0 + raw_residual * bounds
    encoded = nn.functional.smooth_l1_loss(predicted[:, :13], target[:, :13], reduction="none")
    encoded_weight = torch.ones(13, device=predicted.device)
    encoded_weight[0] = float(config["calls_weight"])
    encoded_weight[1:] *= float(config["shape_weight"])
    base = (encoded * encoded_weight).mean()
    calls = torch.expm1(torch.clamp(predicted[:, 0], 0, 30))
    target_calls = torch.expm1(torch.clamp(target[:, 0], 0, 30))
    absolute = torch.abs(calls - target_calls)
    wape = absolute.sum() / torch.clamp(target_calls.sum(), min=1e-8)
    mape = torch.mean(absolute / torch.clamp(target_calls, min=1e-8))
    shape = torch.softmax(predicted[:, 1:13], dim=1)
    target_shape = torch.softmax(target[:, 1:13], dim=1)
    tv = 0.5 * torch.abs(shape - target_shape).sum(dim=1).mean()
    emd = torch.abs(torch.cumsum(shape, dim=1) - torch.cumsum(target_shape, dim=1)).mean()
    target_bytes = torch.expm1(torch.clamp(target[:, BIN_COUNT + 1], 0, 40))
    cost = 5.0 * calls + anchor_bytes_total / 1e5
    target_cost = 5.0 * target_calls + target_bytes / 1e5
    cost_wape = torch.abs(cost - target_cost).sum() / torch.clamp(target_cost.sum(), min=1e-8)
    return base + float(config["wape_weight"]) * wape + float(config["mape_weight"]) * mape + float(config["tv_weight"]) * tv + float(config["emd_weight"]) * emd + float(config["cost_weight"]) * cost_wape


def fit_tp_fold(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], features: list[str], config: dict, folds: dict[str, int], fold: int, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[dict, np.ndarray]:
    train = np.asarray([i for i, row in enumerate(rows) if folds[row["profile_id"]] != fold], dtype=int)
    validation = np.asarray([i for i, row in enumerate(rows) if folds[row["profile_id"]] == fold], dtype=int)
    total_raw, shape_raw = matrices(rows, features, arrays["h0_encoded"], bool(config["engineered"]))
    total, total_mean, total_std = scale_fit(total_raw, train)
    shape, shape_mean, shape_std = scale_fit(shape_raw, train)
    keys, heads = head_keys(rows, config["head_mode"])
    bounds = residual_bounds().astype(np.float32)
    targets = arrays["target_encoded"].astype(np.float32)
    anchored = bytes_anchor(rows, arrays["h0_bytes"], "tp").sum(axis=1).astype(np.float32)
    seed_all(seed)
    model = SplitResidualNet(total.shape[1], shape.shape[1], len(keys), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(*(torch.from_numpy(value[train]) for value in (total, shape, heads, arrays["h0_encoded"].astype(np.float32), targets, anchored))), batch_size=args.batch_size, shuffle=True, generator=generator)
    validation_tensors = tuple(torch.from_numpy(value[validation]).to(device) for value in (total, shape, heads, arrays["h0_encoded"].astype(np.float32), targets, anchored))
    bounds_t = torch.from_numpy(bounds).to(device)
    best_state, best_loss, best_epoch, stale = None, math.inf, -1, 0
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            total_x, shape_x, head_x, h0_x, target_x, anchor_x = (value.to(device) for value in batch)
            loss = tp_loss(model(total_x, shape_x, head_x), h0_x, target_x, bounds_t, anchor_x, config)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(tp_loss(model(*validation_tensors[:3]), validation_tensors[3], validation_tensors[4], bounds_t, validation_tensors[5], config).cpu())
        if validation_loss < best_loss - 1e-6:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), validation_loss, epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        residual = model(torch.from_numpy(total[validation]).to(device), torch.from_numpy(shape[validation]).to(device), torch.from_numpy(heads[validation]).to(device)).cpu().numpy() * bounds
    checkpoint = {
        "schema_version": "phase33c-tp-split-calls-residual-fold-v1", "config": config, "feature_names": features,
        "engineered": bool(config["engineered"]), "head_keys": keys,
        "total_mean": torch.from_numpy(total_mean), "total_std": torch.from_numpy(total_std),
        "shape_mean": torch.from_numpy(shape_mean), "shape_std": torch.from_numpy(shape_std),
        "model_state": {key: value.detach().cpu() for key, value in best_state.items()},
        "best_epoch": best_epoch, "best_validation_loss": best_loss, "seed": seed, "fold": fold,
        "bytes_rule": "lowdim_mean_structural_anchor_preserving_h0_bin_shape",
    }
    return checkpoint, residual


def run_tp_seed(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], features: list[str], config: dict, folds: dict[str, int], seed: int, args: argparse.Namespace, device: torch.device) -> tuple[list[dict], np.ndarray]:
    checkpoints, oof = [], np.zeros((len(rows), ENCODED_SIZE), dtype=np.float32)
    for fold in range(FOLDS):
        checkpoint, residual = fit_tp_fold(rows, arrays, features, config, folds, fold, seed + fold * 1009, args, device)
        indices = [i for i, row in enumerate(rows) if folds[row["profile_id"]] == fold]
        oof[indices] = residual
        checkpoints.append(checkpoint)
    return checkpoints, oof


def evaluate_tp(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], residual: np.ndarray, method: str) -> dict:
    anchor = bytes_anchor(rows, arrays["h0_bytes"], "tp")
    h0 = metric_bundle(rows, arrays, (arrays["h0_calls"], arrays["h0_bytes"]), "tp", "h0")["headline"]
    best = None
    for alpha in ALPHAS:
        calls, _ = vectors_from_encoded(arrays["h0_encoded"] + alpha * residual)
        value = metric_bundle(rows, arrays, (calls, anchor), "tp", method)
        head = value["headline"]
        score = sum(float(head[key]) / threshold for key, threshold in TP_FORMAL.items())
        if float(head["calls_wape"]) >= float(h0["calls_wape"]): score += 4.0
        if float(head["common_reference_cost_wape"]) >= float(h0["common_reference_cost_wape"]): score += 4.0
        per_model = [row for row in value["metrics"] if row["phase"] == "total" and row["policy"] == "all" and row["model"] != "all"]
        score += sum(max(0.0, float(row["calls_wape"]) / 0.15 - 1.0) + max(0.0, float(row["common_reference_cost_wape"]) / 0.08 - 1.0) for row in per_model)
        candidate = {**value, "alpha": alpha, "score": score, "per_model": per_model, "prediction": (calls, anchor)}
        if best is None or candidate["score"] < best["score"]:
            best = candidate
    return best


def weighted_median_ratio(predicted: np.ndarray, target: np.ndarray) -> float:
    ratios = target / np.maximum(predicted, 1e-12)
    order = np.argsort(ratios)
    weights = predicted[order]
    return float(ratios[order][np.searchsorted(np.cumsum(weights), weights.sum() / 2.0)])


def pp_group_key(row: dict[str, str], mode: str) -> str:
    if mode == "global": return "global"
    if mode == "model": return row["model"]
    if mode == "policy": return row["policy"]
    if mode == "model_policy": return row["model"] + "::" + row["policy"]
    raise ValueError(mode)


def fit_pp_scales(rows: list[dict[str, str]], raw_bytes: np.ndarray, target_bytes: np.ndarray, mode: str) -> dict[str, float]:
    indices = [i for i, row in enumerate(rows) if row["split_role"] == "development_train"]
    groups = sorted({pp_group_key(rows[i], mode) for i in indices})
    result = {}
    for group in groups:
        selected = [i for i in indices if pp_group_key(rows[i], mode) == group]
        result[group] = float(np.clip(weighted_median_ratio(raw_bytes[selected].sum(axis=1), target_bytes[selected].sum(axis=1)), 0.8, 1.2))
    return result


def apply_pp_candidate(rows: list[dict[str, str]], raw_bytes: np.ndarray, anchor: np.ndarray, config: dict) -> np.ndarray:
    if config["kind"] == "anchor_blend":
        return (1.0 - float(config["strength"])) * raw_bytes + float(config["strength"]) * anchor
    scales = config["scales"]
    return raw_bytes * np.asarray([scales[pp_group_key(row, config["mode"])] for row in rows])[:, None]


def pp_incumbent_prediction(rows: list[dict[str, str]], checkpoint_paths: list[Path], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    h0_calls, h0_bytes = parse_histograms(rows, "h0")
    h0_encoded = encoded_from_vectors(h0_calls, h0_bytes)
    residuals = []
    for path in checkpoint_paths:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        residuals.extend(predict_checkpoint(rows, h0_encoded, checkpoint, device) for checkpoint in bundle["folds"])
    return vectors_from_encoded(h0_encoded + 0.75 * np.mean(residuals, axis=0))


def subset(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], indices: list[int]) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    return [rows[i] for i in indices], {key: value[indices] for key, value in arrays.items()}


def infer_tp(rows: list[dict[str, str]], bundles: list[list[dict]], alpha: float, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    h0_calls, h0_bytes = parse_histograms(rows, "h0")
    h0_encoded = encoded_from_vectors(h0_calls, h0_bytes)
    residuals = [predict_checkpoint(rows, h0_encoded, checkpoint, device) for bundle in bundles for checkpoint in bundle]
    calls, _ = vectors_from_encoded(h0_encoded + alpha * np.mean(residuals, axis=0))
    return calls, bytes_anchor(rows, h0_bytes, "tp")


def main() -> None:
    args = parse_args()
    for name in ("checkpoints", "analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA unavailable")
    summary33a = json.loads((args.phase33a_dir / "summary.json").read_text())
    summary33b = json.loads((args.phase33b_dir / "summary.json").read_text())
    if summary33a["target_state"]["blind_confirmation"] != "not_generated" or summary33b["blind_confirmation_target_state"] != "not_generated":
        raise RuntimeError("Phase33 blind target isolation failed")

    feature_sets_by_name = {
        "phase33_blind": {
            p: read_csv_gz(args.phase33a_dir / f"dataset/{p}_blind_confirmation_features.csv.gz") for p in ("tp", "pp")
        },
        "phase31_fixed_repeated": {
            p: read_csv_gz(args.phase31b_dir / f"dataset/{p}_fixed_prediction_features.csv.gz") for p in ("tp", "pp")
        },
        "phase32_confirmation_repeated": {
            p: read_csv_gz(args.phase32a_dir / f"dataset/{p}_new_confirmation_features.csv.gz") for p in ("tp", "pp")
        },
    }
    for set_name, directions in feature_sets_by_name.items():
        for parallelism, rows in directions.items():
            target_free(rows, f"{set_name}/{parallelism}")

    # TP: exactly 18 one-seed groups, top three repeated with three seeds and five profile-grouped folds.
    tp_rows = read_csv_gz(args.phase33b_dir / "dataset/tp_combined_development_examples.csv.gz")
    tp_arrays = prepare_development(tp_rows)
    features = feature_sets(tp_rows)["full"]
    folds = fold_map(tp_rows)
    anchor_rel = np.max(np.abs(bytes_anchor(tp_rows, tp_arrays["h0_bytes"], "tp").sum(axis=1) - tp_arrays["target_bytes"].sum(axis=1)) / np.maximum(tp_arrays["target_bytes"].sum(axis=1), 1e-12))
    grid_rows, screened = [], []
    for index, config in enumerate(tp_grid(), 1):
        candidate_id = f"tp33_c{index:02d}_{config['family']}_{config['head_mode']}_lr{config['learning_rate']:g}_w{config['width']}"
        checkpoints, oof = run_tp_seed(tp_rows, tp_arrays, features, config, folds, args.seed + index * 37, args, device)
        result = evaluate_tp(tp_rows, tp_arrays, oof, candidate_id)
        row = {"candidate_id": candidate_id, **config, "screen_seed": args.seed + index * 37, "alpha": result["alpha"], "score": result["score"], **{key: result["headline"][key] for key in TP_FORMAL}}
        grid_rows.append(row); screened.append({"candidate_id": candidate_id, "config": config, **result})
    screened.sort(key=lambda value: value["score"])

    inventory, confirmed = [], []
    for rank, candidate in enumerate(screened[:3], 1):
        seed_oof, bundles = [], []
        for seed_offset in (0, 101, 202):
            seed = args.seed + seed_offset
            checkpoints, oof = run_tp_seed(tp_rows, tp_arrays, features, candidate["config"], folds, seed, args, device)
            seed_oof.append(oof); bundles.append(checkpoints)
            path = args.output_dir / "checkpoints" / f"tp_top{rank}_seed{seed}.pt"
            torch.save({"parallelism": "tp", "candidate_id": candidate["candidate_id"], "rank": rank, "seed": seed, "folds": checkpoints}, path)
            inventory.append({"parallelism": "tp", "rank": rank, "candidate_id": candidate["candidate_id"], "seed": seed, "path": str(path.relative_to(args.output_dir)), "sha256": sha256(path), "bytes": path.stat().st_size})
        result = evaluate_tp(tp_rows, tp_arrays, np.mean(seed_oof, axis=0), candidate["candidate_id"] + "_3seed")
        confirmed.append({"candidate_id": candidate["candidate_id"], "config": candidate["config"], "bundles": bundles, **result})
    confirmed.sort(key=lambda value: value["score"])
    tp_best = confirmed[0]
    tp_id = tp_best["candidate_id"] + f"_5fold_3seed_alpha{tp_best['alpha']}"
    tp_h0 = metric_bundle(tp_rows, tp_arrays, (tp_arrays["h0_calls"], tp_arrays["h0_bytes"]), "tp", "tp_h0")

    # PP: retain Phase32 calls/shape DNN and compare only eight bounded bytes calibrations on fresh development.
    pp_rows = read_csv_gz(args.phase33b_dir / "dataset/pp_new_development_examples.csv.gz")
    pp_arrays = prepare_development(pp_rows)
    pp_paths = sorted((args.phase32b_dir / "checkpoints").glob("pp_top1_seed*.pt"))
    if len(pp_paths) != 3:
        raise RuntimeError("Phase32 PP incumbent checkpoints != 3")
    pp_calls, pp_raw_bytes = pp_incumbent_prediction(pp_rows, pp_paths, device)
    pp_anchor = bytes_anchor(pp_rows, pp_arrays["h0_bytes"], "pp")
    pp_anchor_rel = np.max(np.abs(pp_anchor.sum(axis=1) - pp_arrays["target_bytes"].sum(axis=1)) / np.maximum(pp_arrays["target_bytes"].sum(axis=1), 1e-12))
    pp_configs = [{"candidate_id": f"pp33_anchor_blend_{strength:g}", "kind": "anchor_blend", "strength": strength} for strength in (0.25, 0.5, 0.75, 1.0)]
    for mode in ("global", "model", "policy", "model_policy"):
        pp_configs.append({"candidate_id": f"pp33_train_scale_{mode}", "kind": "train_scale", "mode": mode, "scales": fit_pp_scales(pp_rows, pp_raw_bytes, pp_arrays["target_bytes"], mode)})
    validation_indices = [i for i, row in enumerate(pp_rows) if row["split_role"] == "development_validation"]
    pp_val_rows, pp_val_arrays = subset(pp_rows, pp_arrays, validation_indices)
    pp_incumbent = metric_bundle(pp_val_rows, pp_val_arrays, (pp_calls[validation_indices], pp_raw_bytes[validation_indices]), "pp", "pp32_incumbent")
    pp_h0 = metric_bundle(pp_val_rows, pp_val_arrays, (pp_arrays["h0_calls"][validation_indices], pp_arrays["h0_bytes"][validation_indices]), "pp", "pp_h0")
    pp_candidates = []
    for config in pp_configs:
        calibrated = apply_pp_candidate(pp_rows, pp_raw_bytes, pp_anchor, config)
        value = metric_bundle(pp_val_rows, pp_val_arrays, (pp_calls[validation_indices], calibrated[validation_indices]), "pp", config["candidate_id"])
        head = value["headline"]
        score = sum(float(head[key]) / threshold for key, threshold in PP_FORMAL.items())
        if float(head["calls_wape"]) > float(pp_incumbent["headline"]["calls_wape"]) + 1e-12: score += 10
        if float(head["mean_histogram_tv"]) > float(pp_incumbent["headline"]["mean_histogram_tv"]) + 1e-12: score += 10
        if float(head["mean_normalized_log_payload_emd"]) > float(pp_incumbent["headline"]["mean_normalized_log_payload_emd"]) + 1e-12: score += 10
        pp_candidates.append({"config": config, **value, "score": score, "calibrated": calibrated})
    pp_candidates.sort(key=lambda value: value["score"])
    pp_best = pp_candidates[0]
    pp_id = pp_best["config"]["candidate_id"] + "_phase32_incumbent_calls_shape"
    for value in pp_candidates:
        config = value["config"]
        grid_rows.append({"candidate_id": config["candidate_id"], "parallelism": "pp", "family": config["kind"], "mode": config.get("mode", "anchor"), "parameter_json": json.dumps({key: val for key, val in config.items() if key not in {"candidate_id", "kind"}}, sort_keys=True, separators=(",", ":")), "score": value["score"], **{key: value["headline"][key] for key in PP_FORMAL}})

    pp_checkpoint = {
        "schema_version": "phase33c-pp-independent-bytes-calibration-v1", "selected_candidate_id": pp_id,
        "configuration": pp_best["config"], "selection_source": "phase33_fresh_train_and_validation_only",
        "source_incumbent_alpha": 0.75,
        "source_checkpoints": [{"path": str(path.relative_to(args.phase32b_dir)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in pp_paths],
        "blind_targets_read": False, "phase31_fixed_targets_read": False, "phase32_confirmation_targets_read": False,
    }
    write_json(args.output_dir / "checkpoints/pp_bytes_calibration.json", pp_checkpoint)

    # Freeze all predictions before any Phase33 blind target exists.
    frozen = []
    for prediction_set, direction_rows in feature_sets_by_name.items():
        rows = direction_rows["tp"]
        h0_calls, h0_bytes = parse_histograms(rows, "h0")
        tp_prediction = infer_tp(rows, tp_best["bundles"], float(tp_best["alpha"]), device)
        output = fixed_prediction_rows(rows, {"h0": (h0_calls, h0_bytes), "h0_plus_dnn_residual": tp_prediction}, "tp", tp_id)
        for row in output: row["prediction_set"] = prediction_set
        frozen.extend(output)

        rows = direction_rows["pp"]
        h0_calls, h0_bytes = parse_histograms(rows, "h0")
        calls, raw_bytes = pp_incumbent_prediction(rows, pp_paths, device)
        calibrated = apply_pp_candidate(rows, raw_bytes, bytes_anchor(rows, h0_bytes, "pp"), pp_best["config"])
        output = fixed_prediction_rows(rows, {"h0": (h0_calls, h0_bytes), "h0_plus_dnn_residual": (calls, calibrated)}, "pp", pp_id)
        for row in output: row["prediction_set"] = prediction_set
        frozen.extend(output)

    write_csv(args.output_dir / "analysis/candidate_grid.csv", grid_rows)
    write_csv(args.output_dir / "analysis/checkpoint_inventory.csv", inventory)
    development_records = tp_h0["records"] + tp_best["records"] + pp_h0["records"] + pp_incumbent["records"] + pp_best["records"]
    write_csv_gz(args.output_dir / "analysis/development_predictions_and_metrics.csv.gz", development_records)
    write_csv_gz(args.output_dir / "analysis/frozen_predictions.csv.gz", frozen)
    frozen_sha = sha256(args.output_dir / "analysis/frozen_predictions.csv.gz")

    tp_per_model = {row["model"]: row for row in tp_best["metrics"] if row["phase"] == "total" and row["policy"] == "all" and row["model"] != "all"}
    tp_formal_dev = all(float(tp_best["headline"][key]) <= threshold for key, threshold in TP_FORMAL.items())
    tp_positive = float(tp_best["headline"]["calls_wape"]) < float(tp_h0["headline"]["calls_wape"]) and float(tp_best["headline"]["common_reference_cost_wape"]) < float(tp_h0["headline"]["common_reference_cost_wape"])
    tp_no_model_severe = all(float(row["calls_wape"]) <= 0.15 and float(row["common_reference_cost_wape"]) <= 0.08 for row in tp_per_model.values())
    pp_formal_dev = all(float(pp_best["headline"][key]) <= threshold for key, threshold in PP_FORMAL.items())
    pp_protected = all(abs(float(pp_best["headline"][key]) - float(pp_incumbent["headline"][key])) <= 1e-12 for key in ("calls_wape", "mean_histogram_tv", "mean_normalized_log_payload_emd")) and float(pp_best["headline"]["common_reference_cost_wape"]) <= float(pp_incumbent["headline"]["common_reference_cost_wape"])
    checks = {
        "phase33_blind_target_absent_and_unread": summary33a["target_state"]["blind_confirmation"] == "not_generated",
        "all_three_prediction_sets_target_free": all(not any(key.startswith("target_") for key in rows[0]) for directions in feature_sets_by_name.values() for rows in directions.values()),
        "tp_18_regular_candidates": sum(1 for row in grid_rows if row["candidate_id"].startswith("tp33_")) == 18,
        "tp_top3_three_seed_fivefold": len(inventory) == 9 and len({row["rank"] for row in inventory}) == 3,
        "tp_profile_grouped_fivefold": len(set(folds.values())) == 5,
        "pp_8_conservative_candidates": sum(1 for row in grid_rows if row["candidate_id"].startswith("pp33_")) == 8,
        "pp_incumbent_calls_shape_protected": pp_protected,
        "bytes_anchor_matches_development_teacher": anchor_rel < 1e-10 and pp_anchor_rel < 1e-10,
        "frozen_methods_h0_and_dnn": {row["method"] for row in frozen} == {"h0", "h0_plus_dnn_residual"},
        "frozen_three_prediction_sets": {row["prediction_set"] for row in frozen} == set(feature_sets_by_name),
        "dnn_calls_residual_nonzero": all(any(abs(float(row["predicted_total_calls_per_1000"]) - float(next(old for old in frozen if old["prediction_set"] == row["prediction_set"] and old["parallelism"] == row["parallelism"] and old["example_id"] == row["example_id"] and old["method"] == "h0")["predicted_total_calls_per_1000"])) > 1e-6 for row in frozen if row["parallelism"] == direction and row["method"] == "h0_plus_dnn_residual") for direction in ("tp", "pp")),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase33c-target-free-model-selection-v1", "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
        "search": {"tp_regular_candidates": 18, "tp_absolute_limit": 24, "tp_top_candidates": 3, "tp_confirmation_seeds": 3, "folds": 5, "pp_regular_candidates": 8, "pp_absolute_limit": 12},
        "tp": {"selected_candidate_id": tp_id, "config": tp_best["config"], "alpha": tp_best["alpha"], "development_cv_headline": tp_best["headline"], "h0_development_cv_headline": tp_h0["headline"], "per_model": tp_per_model, "formal_development_pass": tp_formal_dev and tp_positive and tp_no_model_severe, "positive_vs_h0": tp_positive, "no_model_severe_regression": tp_no_model_severe, "top3": [{"candidate_id": value["candidate_id"], "score": value["score"], "alpha": value["alpha"], "headline": value["headline"]} for value in confirmed]},
        "pp": {"selected_candidate_id": pp_id, "development_validation_headline": pp_best["headline"], "phase32_incumbent_on_same_validation": pp_incumbent["headline"], "h0_on_same_validation": pp_h0["headline"], "formal_development_pass": pp_formal_dev and pp_protected, "incumbent_calls_shape_protected": pp_protected, "candidates": [{"candidate_id": value["config"]["candidate_id"], "score": value["score"], "headline": value["headline"]} for value in pp_candidates]},
        "bytes_anchor": {"definition": "allowed low-dimensional capped mean × model bytes/token prior × structural communication multiplier × 1000; H0 bin shape retained", "tp_max_development_target_relative_error": float(anchor_rel), "pp_max_development_target_relative_error": float(pp_anchor_rel)},
        "target_isolation": {"phase33_blind_targets_read": False, "phase31_fixed_targets_read": False, "phase32_confirmation_targets_read": False},
        "frozen_prediction_sha256": frozen_sha, "counts": {"tp_development_profiles": len({row["profile_id"] for row in tp_rows}), "pp_fresh_development_profiles": len({row["profile_id"] for row in pp_rows}), "frozen_rows": len(frozen), "checkpoints": len(inventory)},
        "checks": checks,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase33c-audit-v1", "status": status, "checks": checks, "frozen_prediction_sha256": frozen_sha})
    write_json(args.output_dir / "logs/training.log", {"event": "phase33c_target_free_selection_complete", "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "repository_head_at_training": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(), "device": str(device), "phase33_blind_targets_read": False})

    values = [(row["candidate_id"], float(row["score"]), "#2563eb" if row["candidate_id"].startswith("tp33") else "#ea580c") for row in grid_rows]
    width, height, margin = 1200, 520, 55
    maximum = max(value for _, value, _ in values) * 1.05
    bar_width = (width - 2 * margin) / len(values)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="55" y="30" font-family="sans-serif" font-size="20">Phase33开发侧有限候选综合分数（越低越好）</text>']
    for index, (_, value, color) in enumerate(values):
        bar_height = value / maximum * (height - 120); x = margin + index * bar_width; y = height - 65 - bar_height
        svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 2, 1):.2f}" height="{bar_height:.2f}" fill="{color}"/>')
    svg.append('</svg>')
    (args.output_dir / "figures/candidate_scores.svg").write_text("\n".join(svg) + "\n")
    (args.output_dir / "README.md").write_text(f"""# Phase 33C：TP继续收敛与PP保守改进（打开新确认真值前）

本阶段只在开发数据上选模型。TP使用Phase31与Phase33合并后的94个开发画像、35524个完整teacher请求，比较18组`H0 + DNN residual`；每组先1个seed，开发前三名再做3-seed、5折profile分组确认。PP不重训calls/形状网络，保留Phase32 incumbent，只在45个全新开发画像上比较8种独立bytes校准。

bytes总量锚点不是读取完整请求列表或target，而是用部署时允许的低维`input/output mean capped`、模型bytes/token先验和已验证结构通信倍数计算；开发集审计与Hfull teacher相对误差分别不超过TP `{anchor_rel:.3e}`、PP `{pp_anchor_rel:.3e}`。bytes的12-bin形状仍保留H0分配。

TP开发五折结果：calls/bytes/TV/EMD/cost WAPE分别为`{tp_best['headline']['calls_wape']:.2%}`、`{tp_best['headline']['bytes_wape']:.2%}`、`{tp_best['headline']['mean_histogram_tv']:.4f}`、`{tp_best['headline']['mean_normalized_log_payload_emd']:.4f}`、`{tp_best['headline']['common_reference_cost_wape']:.2%}`。PP新验证结果分别为`{pp_best['headline']['calls_wape']:.2%}`、`{pp_best['headline']['bytes_wape']:.2%}`、`{pp_best['headline']['mean_histogram_tv']:.4f}`、`{pp_best['headline']['mean_normalized_log_payload_emd']:.4f}`、`{pp_best['headline']['common_reference_cost_wape']:.2%}`。

9个Phase33全新确认窗口的Hfull target仍不存在。TP/PP对它们的预测已经冻结，SHA-256为`{frozen_sha}`。Phase31固定集和Phase32确认集也一并冻结，但后两者后续只能作为重复工程证据。只有归档本目录后，才可一次性生成Phase33新确认target。
""")
    (args.output_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "tp": summary["tp"], "pp": summary["pp"], "frozen_prediction_sha256": frozen_sha}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
