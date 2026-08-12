#!/usr/bin/env python3
"""Final bounded TP round: weighted totals and finite model/policy heads."""

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
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_phase27c_pp_scheduler_feature_predictors import (
    ENCODED_SIZE,
    MLP,
    feature_matrix,
    is_log_feature,
    parse_histograms,
    predict,
    prepare_development,
    residual_bounds,
    seed_all,
)
from train_phase31c_known_model_residuals import (
    ALPHAS,
    FIT_ROLE,
    VALIDATION_ROLE,
    aggregate,
    calibrated,
    candidate_score,
    encoded_from_vectors,
    feature_sets,
    fixed_prediction_rows,
    headline,
    records_for_validation,
    vectors_from_encoded,
)


MODELS = ("deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b")
POLICIES = ("latency", "balanced", "throughput")
CONFIGS = (
    {"feature_set": "full", "head_mode": "shared", "learning_rate": 3e-3, "calls_total_weight": 4.0, "bytes_total_weight": 4.0},
    {"feature_set": "no_arrival", "head_mode": "shared", "learning_rate": 3e-3, "calls_total_weight": 8.0, "bytes_total_weight": 4.0},
    {"feature_set": "full", "head_mode": "policy", "learning_rate": 1e-3, "calls_total_weight": 8.0, "bytes_total_weight": 4.0},
    {"feature_set": "full", "head_mode": "model", "learning_rate": 1e-3, "calls_total_weight": 8.0, "bytes_total_weight": 4.0},
    {"feature_set": "full", "head_mode": "model_policy", "learning_rate": 1e-3, "calls_total_weight": 8.0, "bytes_total_weight": 4.0},
    {"feature_set": "no_arrival", "head_mode": "model_policy", "learning_rate": 1e-3, "calls_total_weight": 8.0, "bytes_total_weight": 4.0},
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=root / "experiment-results/phase31b_known_model_hfull_dataset")
    parser.add_argument("--phase31c-dir", type=Path, default=root / "experiment-results/phase31c_known_model_residual_training")
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase31e_tp_weighted_residual_round2")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode())


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


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


def head_key(row: dict[str, str], mode: str) -> str:
    if mode == "shared":
        return "shared"
    if mode == "policy":
        return row["policy"]
    if mode == "model":
        return row["model"]
    if mode == "model_policy":
        return f"{row['model']}::{row['policy']}"
    raise ValueError(mode)


def fit_weighted_head(
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    feature_names: list[str],
    config: dict,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict, list[dict]]:
    train_indices = np.asarray([index for index, row in enumerate(rows) if row["split_role"] == FIT_ROLE], dtype=int)
    validation_indices = np.asarray([index for index, row in enumerate(rows) if row["split_role"] == VALIDATION_ROLE], dtype=int)
    if not len(train_indices) or not len(validation_indices):
        raise RuntimeError("head lacks train or validation rows")
    features = feature_matrix(rows, feature_names)
    feature_mean = features[train_indices].mean(axis=0)
    feature_std = features[train_indices].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    scaled = np.clip((features - feature_mean) / feature_std, -6.0, 6.0).astype(np.float32)
    bounds = residual_bounds()
    targets = (arrays["bounded_residual"] / bounds).astype(np.float32)
    element_weights = np.ones(ENCODED_SIZE, dtype=np.float32)
    element_weights[0] = config["calls_total_weight"]
    element_weights[13] = config["bytes_total_weight"]

    seed_all(seed)
    model = MLP(len(feature_names), ENCODED_SIZE, bounded=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    weights_tensor = torch.from_numpy(element_weights).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(scaled[train_indices]), torch.from_numpy(targets[train_indices])),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_x = torch.from_numpy(scaled[validation_indices]).to(device)
    validation_y = torch.from_numpy(targets[validation_indices]).to(device)
    best_state = None
    best_loss = math.inf
    stale = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = (nn.functional.smooth_l1_loss(model(batch_x), batch_y, reduction="none") * weights_tensor).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = float((nn.functional.smooth_l1_loss(model(validation_x), validation_y, reduction="none") * weights_tensor).mean().cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("no weighted checkpoint")
    checkpoint = {
        "schema_version": "phase31e-tp-weighted-residual-head-v1",
        "method": "enhanced_bounded_residual",
        "feature_names": feature_names,
        "log_feature_names": [name for name in feature_names if is_log_feature(name)],
        "feature_mean": torch.from_numpy(feature_mean),
        "feature_std": torch.from_numpy(feature_std),
        "target_mean": torch.zeros(ENCODED_SIZE),
        "target_std_or_residual_bounds": torch.from_numpy(bounds),
        "model_state": {name: value.detach().cpu() for name, value in best_state.items()},
        "architecture": {"hidden_sizes": [64, 64], "activation": "relu", "bounded_tanh": True},
        "best_epoch": int(np.argmin([row["validation_loss"] for row in history])),
        "best_validation_loss": best_loss,
        "loss_element_weights": element_weights.tolist(),
        "seed": seed,
    }
    return checkpoint, history


def subset_arrays(arrays: dict[str, np.ndarray], indices: list[int]) -> dict[str, np.ndarray]:
    return {name: value[indices] for name, value in arrays.items()}


def fit_configuration(
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    feature_names: list[str],
    config: dict,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict, tuple[np.ndarray, np.ndarray], list[dict]]:
    keys = sorted({head_key(row, config["head_mode"]) for row in rows})
    checkpoints = {}
    histories = []
    calls = np.zeros_like(arrays["h0_calls"])
    logical_bytes = np.zeros_like(arrays["h0_bytes"])
    for key_index, key in enumerate(keys):
        indices = [index for index, row in enumerate(rows) if head_key(row, config["head_mode"]) == key]
        head_rows = [rows[index] for index in indices]
        head_arrays = subset_arrays(arrays, indices)
        checkpoint, history = fit_weighted_head(head_rows, head_arrays, feature_names, config, args, device, seed + key_index)
        part_calls, part_bytes = predict(head_rows, checkpoint, head_arrays["h0_encoded"], device)
        calls[indices] = part_calls
        logical_bytes[indices] = part_bytes
        checkpoints[key] = checkpoint
        histories.extend({"head": key, **row} for row in history)
    return checkpoints, (calls, logical_bytes), histories


def predict_configuration(
    rows: list[dict[str, str]],
    h0_encoded: np.ndarray,
    checkpoints: dict,
    mode: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    calls = np.zeros((len(rows), 12), dtype=np.float64)
    logical_bytes = np.zeros((len(rows), 12), dtype=np.float64)
    for key, checkpoint in checkpoints.items():
        indices = [index for index, row in enumerate(rows) if head_key(row, mode) == key]
        part_calls, part_bytes = predict([rows[index] for index in indices], checkpoint, h0_encoded[indices], device)
        calls[indices] = part_calls
        logical_bytes[indices] = part_bytes
    return calls, logical_bytes


def main() -> None:
    args = parse_args()
    for name in ("checkpoints", "analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset_summary = json.loads((args.dataset_dir / "summary.json").read_text())
    phase31c_summary = json.loads((args.phase31c_dir / "summary.json").read_text())
    if dataset_summary["status"] != "PASS" or dataset_summary["fixed_target_state"] != "not_generated":
        raise RuntimeError("Phase31B target isolation failed")
    if phase31c_summary["status"] != "PASS" or phase31c_summary["fixed_targets_read"] is not False:
        raise RuntimeError("Phase31C contract failed")
    development = read_csv_gz(args.dataset_dir / "dataset/tp_development_examples.csv.gz")
    fixed = read_csv_gz(args.dataset_dir / "dataset/tp_fixed_prediction_features.csv.gz")
    if any(name.startswith("target_") for name in fixed[0]):
        raise RuntimeError("fixed target exposed")
    arrays = prepare_development(development)
    fixed_h0_calls, fixed_h0_bytes = parse_histograms(fixed, "h0")
    fixed_h0_encoded = encoded_from_vectors(fixed_h0_calls, fixed_h0_bytes)
    sets = feature_sets(development)
    h0_records = records_for_validation(development, arrays, (arrays["h0_calls"], arrays["h0_bytes"]), "tp", "h0")
    h0_headline = headline(aggregate(h0_records))
    incumbent = phase31c_summary["selected"]["tp"]
    incumbent_score = candidate_score(incumbent["validation_headline"], h0_headline, "tp")

    grid_rows = []
    candidates = []
    for config_index, base_config in enumerate(CONFIGS):
        config = dict(base_config)
        candidate_id = f"tp31e_c{config_index + 1:02d}_{config['feature_set']}_{config['head_mode']}_lr{config['learning_rate']:g}_cw{config['calls_total_weight']:g}_bw{config['bytes_total_weight']:g}"
        checkpoints, raw_prediction, _ = fit_configuration(development, arrays, sets[config["feature_set"]], config, args, device, args.seed + config_index * 31)
        best = None
        for alpha in ALPHAS:
            prediction = calibrated(arrays["h0_encoded"], *raw_prediction, alpha)
            records = records_for_validation(development, arrays, prediction, "tp", candidate_id)
            head = headline(aggregate(records))
            score = candidate_score(head, h0_headline, "tp")
            if best is None or score < best["score"]:
                best = {"score": score, "alpha": alpha, "prediction": prediction, "records": records, "headline": head}
        row = {"candidate_id": candidate_id, **config, "alpha": best["alpha"], "score": best["score"], **{name: best["headline"][name] for name in ("calls_wape", "bytes_wape", "mean_histogram_tv", "mean_normalized_log_payload_emd", "common_reference_cost_wape")}}
        grid_rows.append(row)
        candidates.append({"config": config, "candidate_id": candidate_id, **best})
    candidates.sort(key=lambda value: value["score"])

    ensembles = []
    checkpoint_inventory = []
    for rank, initial in enumerate(candidates[:2], 1):
        config = initial["config"]
        development_residuals = []
        fixed_residuals = []
        saved = []
        for seed_offset in (0, 101, 202):
            seed = args.seed + seed_offset
            checkpoints, development_raw, _ = fit_configuration(development, arrays, sets[config["feature_set"]], config, args, device, seed)
            fixed_raw = predict_configuration(fixed, fixed_h0_encoded, checkpoints, config["head_mode"], device)
            development_residuals.append(encoded_from_vectors(*development_raw) - arrays["h0_encoded"])
            fixed_residuals.append(encoded_from_vectors(*fixed_raw) - fixed_h0_encoded)
            checkpoint_path = args.output_dir / "checkpoints" / f"tp31e_top{rank}_seed{seed}.pt"
            torch.save({"candidate_id": initial["candidate_id"], "config": config, "seed": seed, "heads": checkpoints}, checkpoint_path)
            item = {"candidate_rank": rank, "candidate_id": initial["candidate_id"], "seed": seed, "path": str(checkpoint_path.relative_to(args.output_dir)), "sha256": sha256(checkpoint_path), "bytes": checkpoint_path.stat().st_size}
            checkpoint_inventory.append(item)
            saved.append(item)
        mean_development_residual = np.mean(development_residuals, axis=0)
        mean_fixed_residual = np.mean(fixed_residuals, axis=0)
        best = None
        for alpha in ALPHAS:
            prediction = vectors_from_encoded(arrays["h0_encoded"] + alpha * mean_development_residual)
            records = records_for_validation(development, arrays, prediction, "tp", initial["candidate_id"] + "_ensemble")
            head = headline(aggregate(records))
            score = candidate_score(head, h0_headline, "tp")
            if best is None or score < best["score"]:
                best = {"score": score, "alpha": alpha, "prediction": prediction, "records": records, "headline": head}
        fixed_prediction = vectors_from_encoded(fixed_h0_encoded + best["alpha"] * mean_fixed_residual)
        ensembles.append({"candidate_id": initial["candidate_id"], "config": config, "checkpoints": saved, "fixed_prediction": fixed_prediction, **best})
    ensembles.sort(key=lambda value: value["score"])
    best_new = ensembles[0]

    if best_new["score"] < incumbent_score:
        selected_source = "phase31e_new"
        selected_id = f"{best_new['candidate_id']}_3seed_alpha{best_new['alpha']}"
        selected_headline = best_new["headline"]
        selected_score = best_new["score"]
        validation_records = h0_records + best_new["records"]
        frozen_rows = fixed_prediction_rows(fixed, {"h0": (fixed_h0_calls, fixed_h0_bytes), "h0_plus_dnn_residual": best_new["fixed_prediction"]}, "tp", selected_id)
        selected_config = {**best_new["config"], "alpha": best_new["alpha"], "checkpoints": best_new["checkpoints"]}
    else:
        selected_source = "phase31c_incumbent"
        selected_id = incumbent["candidate_id"]
        selected_headline = incumbent["validation_headline"]
        selected_score = incumbent_score
        old_validation = read_csv_gz(args.phase31c_dir / "analysis/selected_validation_predictions.csv.gz")
        old_method = old_validation[-1]["method"]
        validation_records = [row for row in old_validation if row["method"] in {"h0", old_method} and row.get("model")]
        old_frozen = read_csv_gz(args.phase31c_dir / "analysis/frozen_fixed_prediction.csv.gz")
        frozen_rows = [row for row in old_frozen if row["parallelism"] == "tp"]
        selected_config = {"source": "phase31c", "candidate_id": incumbent["candidate_id"], "checkpoints": incumbent["checkpoints"]}

    write_csv(args.output_dir / "analysis/candidate_grid.csv", grid_rows)
    write_csv(args.output_dir / "analysis/checkpoint_inventory.csv", checkpoint_inventory)
    write_csv_gz(args.output_dir / "analysis/selected_validation_predictions.csv.gz", validation_records)
    write_csv_gz(args.output_dir / "analysis/frozen_fixed_prediction.csv.gz", frozen_rows)
    prediction_sha = sha256(args.output_dir / "analysis/frozen_fixed_prediction.csv.gz")
    try:
        import matplotlib.pyplot as plt
        labels = ("calls WAPE", "bytes WAPE", "TV", "cost WAPE")
        keys = ("calls_wape", "bytes_wape", "mean_histogram_tv", "common_reference_cost_wape")
        x = np.arange(len(keys))
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(x - 0.18, [h0_headline[key] for key in keys], 0.36, label="H0")
        axis.bar(x + 0.18, [selected_headline[key] for key in keys], 0.36, label="selected H0+DNN")
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / "figures/validation_comparison.png", dpi=180)
        plt.close(figure)
    except Exception as error:
        write_json(args.output_dir / "figures/plot_failure.json", {"error": repr(error)})

    checks = {
        "phase31b_fixed_targets_absent": dataset_summary["fixed_target_state"] == "not_generated",
        "phase31c_fixed_targets_not_read": phase31c_summary["fixed_targets_read"] is False,
        "fixed_features_contain_no_target": not any(name.startswith("target_") for name in fixed[0]),
        "new_grid_exact_six_total_tp_search_eighteen": len(grid_rows) == 6,
        "top2_three_seeds": len(checkpoint_inventory) == 6,
        "frozen_prediction_rows_1080": len(frozen_rows) == 1080,
        "methods_h0_and_dnn": {row["method"] for row in frozen_rows} == {"h0", "h0_plus_dnn_residual"},
        "selected_residual_nonzero": any(abs(float(row["predicted_total_calls_per_1000"]) - float(next(value for value in frozen_rows if value["example_id"] == row["example_id"] and value["method"] == "h0")["predicted_total_calls_per_1000"])) > 1e-6 for row in frozen_rows if row["method"] == "h0_plus_dnn_residual"),
        "fixed_targets_not_a_script_input": not any("target" in name for name in vars(args) if name not in {"dataset_dir"}),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase31e-tp-weighted-residual-round2-v1",
        "status": status,
        "round": "TP finite final round; six additional configurations, 18 cumulative",
        "selected_source": selected_source,
        "selected_candidate_id": selected_id,
        "selected_config": selected_config,
        "selected_validation_headline": selected_headline,
        "selected_validation_score": selected_score,
        "incumbent_validation_score": incumbent_score,
        "h0_validation_headline": h0_headline,
        "fixed_targets_read": False,
        "frozen_prediction_sha256": prediction_sha,
        "counts": {"new_candidates": len(grid_rows), "checkpoints": len(checkpoint_inventory), "frozen_prediction_rows": len(frozen_rows)},
        "checks": checks,
        "evidence_note": "Phase31D target had already been opened before this predeclared final route; training/selection is development-only, but any re-evaluation on the same fixed set is repeated-test rather than fresh independent confirmation.",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase31e-training-audit-v1", "status": status, "checks": checks, "fixed_targets_read": False})
    (args.output_dir / "README.md").write_text(f"""# Phase 31E：TP最后一轮加权H0+DNN residual

本阶段执行今晚参考文档允许的最后6个TP配置，使TP累计搜索量达到18组上限。新增配置只改变总calls/bytes损失权重与共享、policy、model、model×policy小头；网络仍为64×64有界DNN residual，最终形式仍是`H0 + DNN residual`。

训练和选型只读取Phase31B的39个训练画像、10个验证画像及不含target的固定预测特征，没有读取Phase31D Hfull target。选中来源为`{selected_source}`，固定预测SHA为`{prediction_sha}`。

需要公开的证据限制：Phase31D第一轮固定评测已经完成，因此后续在同一固定集上的结果属于重复评测，不是全新的独立确认；这不构成target进入训练或模型选择，但结论必须带此限制，且不得更换固定窗口或降低阈值。
""")
    write_json(args.output_dir / "logs/training.log", {"event": "phase31e_tp_weighted_training_complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "status": status, "repository_head_at_training": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(), "device": str(device), "fixed_targets_read": False})
    (args.output_dir / "DONE").write_text(f"{status}\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "selected_source": selected_source, "selected_candidate_id": selected_id, "validation": selected_headline, "prediction_sha256": prediction_sha}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
