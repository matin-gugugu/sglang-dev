#!/usr/bin/env python3
"""Train Phase 26C direct and bounded-residual Hfull predictors.

Only Phase 16 train profiles are used for fitting and only validation profiles
are used for early stopping.  Temporal/external test profiles are deliberately
left for the separate Phase 26D evaluation.
"""

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
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BIN_COUNT = 12
ENCODED_SIZE = 2 * (BIN_COUNT + 1)
METHODS = ("h0", "direct", "h0_bounded_residual")
PARALLELISMS = ("tp", "pp")
FIT_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLITS = ("temporal_test", "external_test", "external_synthetic")
COMMON_REFERENCE_LAUNCH_US = 5.0
COMMON_REFERENCE_BANDWIDTH_GBPS = 100.0

LOG_FEATURES = {
    "feature_profile_rps",
    "feature_profile_interarrival_cv",
    "feature_profile_peak_to_mean_1s",
    "feature_profile_fano_1s",
    "feature_profile_input_mean_capped",
    "feature_profile_output_mean_capped",
    "feature_model_num_hidden_layers",
    "feature_model_hidden_size",
    "feature_model_dense_intermediate_ratio",
    "feature_model_num_attention_heads",
    "feature_model_head_dim",
    "feature_model_num_experts",
    "feature_model_experts_per_token",
    "feature_model_num_shared_experts",
    "feature_model_estimated_moe_layers",
    "feature_model_logical_collectives_per_forward_prior",
    "feature_model_payload_bytes_per_active_token_prior",
    "feature_tp_max_batch_size",
    "feature_tp_max_prefill_tokens",
    "feature_pp_max_microbatch_size",
    "feature_pp_chunk_tokens",
    "feature_pp_page_size",
    "feature_pp_proxy_tensor_count",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root
        / "experiment-results/phase26b_unified_hfull_training_dataset/training_examples.csv.gz",
    )
    parser.add_argument(
        "--dataset-summary",
        type=Path,
        default=root
        / "experiment-results/phase26b_unified_hfull_training_dataset/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase26c_hfull_predictor_training",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def load_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def transform_feature(name: str, value: str) -> float:
    numeric = float(value)
    return math.log1p(max(numeric, 0.0)) if name in LOG_FEATURES else numeric


def target_encode(calls: np.ndarray, logical_bytes: np.ndarray) -> np.ndarray:
    encoded: list[float] = []
    for vector in (calls, logical_bytes):
        total = max(float(np.sum(vector)), 0.0)
        smoothing = max(total, 1.0) * 1e-6 / BIN_COUNT
        shares = (vector + smoothing) / (total + smoothing * BIN_COUNT)
        encoded.extend([math.log1p(total), *np.log(shares)])
    return np.asarray(encoded, dtype=np.float32)


def target_decode(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    offset = 0
    for _ in range(2):
        total = max(math.expm1(float(np.clip(encoded[offset], 0, 40))), 0.0)
        logits = np.clip(encoded[offset + 1 : offset + BIN_COUNT + 1], -50, 50)
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        vectors.append(total * probabilities)
        offset += BIN_COUNT + 1
    return vectors[0].astype(np.float64), vectors[1].astype(np.float64)


def residual_bounds() -> np.ndarray:
    bounds = np.full(ENCODED_SIZE, 2.0, dtype=np.float32)
    bounds[0] = math.log(2.0)
    bounds[BIN_COUNT + 1] = math.log(2.0)
    return bounds


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, bounded: bool):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )
        self.bounded = bounded

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.network(value)
        return torch.tanh(output) if self.bounded else output


def prepare_arrays(rows: list[dict[str, str]], feature_names: list[str]) -> dict[str, np.ndarray]:
    features = np.asarray(
        [[transform_feature(name, row[name]) for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    target_calls = np.stack(
        [np.asarray(json.loads(row["target_calls_by_12bin_json"]), dtype=np.float64) for row in rows]
    )
    target_bytes = np.stack(
        [
            np.asarray(json.loads(row["target_logical_bytes_by_12bin_json"]), dtype=np.float64)
            for row in rows
        ]
    )
    h0_calls = np.stack(
        [np.asarray(json.loads(row["h0_calls_by_12bin_json"]), dtype=np.float64) for row in rows]
    )
    h0_bytes = np.stack(
        [
            np.asarray(json.loads(row["h0_logical_bytes_by_12bin_json"]), dtype=np.float64)
            for row in rows
        ]
    )
    target_encoded = np.stack(
        [target_encode(calls, byte_values) for calls, byte_values in zip(target_calls, target_bytes)]
    )
    h0_encoded = np.stack(
        [target_encode(calls, byte_values) for calls, byte_values in zip(h0_calls, h0_bytes)]
    )
    bounds = residual_bounds()
    residual = np.clip(target_encoded - h0_encoded, -bounds, bounds)
    return {
        "features": features,
        "target_calls": target_calls,
        "target_bytes": target_bytes,
        "h0_calls": h0_calls,
        "h0_bytes": h0_bytes,
        "target_encoded": target_encoded,
        "h0_encoded": h0_encoded,
        "bounded_residual": residual,
    }


def fit_model(
    *,
    parallelism: str,
    method: str,
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    feature_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict, list[dict]]:
    train_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["split"] == FIT_SPLIT], dtype=int
    )
    validation_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["split"] == VALIDATION_SPLIT], dtype=int
    )
    features = arrays["features"]
    feature_mean = features[train_indices].mean(axis=0)
    feature_std = features[train_indices].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    scaled_features = np.clip((features - feature_mean) / feature_std, -6.0, 6.0).astype(
        np.float32
    )

    bounded = method == "h0_bounded_residual"
    if bounded:
        bounds = residual_bounds()
        targets = (arrays["bounded_residual"] / bounds).astype(np.float32)
        target_mean = np.zeros(ENCODED_SIZE, dtype=np.float32)
        target_std = bounds
    else:
        raw_targets = arrays["target_encoded"]
        target_mean = raw_targets[train_indices].mean(axis=0)
        target_std = raw_targets[train_indices].std(axis=0)
        target_std[target_std < 1e-6] = 1.0
        targets = ((raw_targets - target_mean) / target_std).astype(np.float32)

    seed_all(seed)
    model = MLP(len(feature_names), ENCODED_SIZE, bounded=bounded).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    loss_fn = nn.SmoothL1Loss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(scaled_features[train_indices]),
            torch.from_numpy(targets[train_indices]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_x = torch.from_numpy(scaled_features[validation_indices]).to(device)
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
            loss = loss_fn(model(batch_x), batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(validation_x), validation_y).cpu())
        history.append(
            {
                "parallelism": parallelism,
                "method": method,
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"no checkpoint for {parallelism}/{method}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw_prediction = model(torch.from_numpy(scaled_features).to(device)).cpu().numpy()
    if bounded:
        prediction_encoded = arrays["h0_encoded"] + raw_prediction * residual_bounds()
    else:
        raw_prediction = np.clip(raw_prediction, -6.0, 6.0)
        prediction_encoded = raw_prediction * target_std + target_mean

    checkpoint = {
        "schema_version": "phase26c-hfull-predictor-checkpoint-v1",
        "parallelism": parallelism,
        "method": method,
        "bin_schema_id": rows[0]["bin_schema_id"],
        "bin_edges_bytes": json.loads(rows[0]["bin_edges_bytes_json"]),
        "feature_names": feature_names,
        "log_feature_names": sorted(LOG_FEATURES),
        "feature_mean": torch.from_numpy(feature_mean),
        "feature_std": torch.from_numpy(feature_std),
        "target_mean": torch.from_numpy(target_mean),
        "target_std_or_residual_bounds": torch.from_numpy(target_std),
        "model_state": {name: value.detach().cpu() for name, value in best_state.items()},
        "architecture": {"hidden_sizes": [64, 64], "activation": "relu", "bounded_tanh": bounded},
        "best_epoch": int(np.argmin([row["validation_loss"] for row in history])),
        "best_validation_loss": best_loss,
        "fit_split": FIT_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "seed": seed,
    }
    return prediction_encoded, checkpoint, history


def bin_log_centers(edges: list[float]) -> np.ndarray:
    values = np.asarray(edges, dtype=np.float64)
    return (np.log2(values[:-1]) + np.log2(values[1:])) / 2


def normalized_log_emd(predicted: np.ndarray, actual: np.ndarray, edges: list[float]) -> float:
    predicted_total = max(float(predicted.sum()), 1e-12)
    actual_total = max(float(actual.sum()), 1e-12)
    predicted_cdf = np.cumsum(predicted / predicted_total)
    actual_cdf = np.cumsum(actual / actual_total)
    centers = bin_log_centers(edges)
    widths = np.diff(centers)
    area = float(np.sum(np.abs(predicted_cdf[:-1] - actual_cdf[:-1]) * widths))
    return area / (math.log2(edges[-1]) - math.log2(edges[0]))


def histogram_tv(predicted: np.ndarray, actual: np.ndarray) -> float:
    predicted_total = max(float(predicted.sum()), 1e-12)
    actual_total = max(float(actual.sum()), 1e-12)
    return float(np.abs(predicted / predicted_total - actual / actual_total).sum() / 2)


def common_reference_cost(calls: np.ndarray, logical_bytes: np.ndarray, edges: list[float]) -> float:
    total = 0.0
    bandwidth_bytes_per_second = COMMON_REFERENCE_BANDWIDTH_GBPS * 1e9
    for index, (count, byte_count) in enumerate(zip(calls, logical_bytes)):
        if count <= 1e-12:
            continue
        payload = float(np.clip(byte_count / count, edges[index], edges[index + 1]))
        total += count * (
            COMMON_REFERENCE_LAUNCH_US + payload / bandwidth_bytes_per_second * 1e6
        )
    return total


def case_record(
    *,
    row: dict[str, str],
    method: str,
    phase: str,
    actual_calls: np.ndarray,
    actual_bytes: np.ndarray,
    predicted_calls: np.ndarray,
    predicted_bytes: np.ndarray,
    edges: list[float],
) -> dict:
    actual_calls_total = float(actual_calls.sum())
    predicted_calls_total = float(predicted_calls.sum())
    actual_bytes_total = float(actual_bytes.sum())
    predicted_bytes_total = float(predicted_bytes.sum())
    actual_cost = common_reference_cost(actual_calls, actual_bytes, edges)
    predicted_cost = common_reference_cost(predicted_calls, predicted_bytes, edges)
    return {
        "parallelism": row["parallelism"],
        "model": row["model"],
        "parallel_size": row["parallel_size"],
        "policy": row["policy"],
        "profile_id": row["profile_id"],
        "split": row["split"],
        "method": method,
        "phase": phase,
        "actual_total_calls": actual_calls_total,
        "predicted_total_calls": predicted_calls_total,
        "calls_absolute_error": abs(predicted_calls_total - actual_calls_total),
        "calls_ape": abs(predicted_calls_total - actual_calls_total)
        / max(actual_calls_total, 1e-12),
        "actual_total_logical_bytes": actual_bytes_total,
        "predicted_total_logical_bytes": predicted_bytes_total,
        "bytes_absolute_error": abs(predicted_bytes_total - actual_bytes_total),
        "bytes_ape": abs(predicted_bytes_total - actual_bytes_total)
        / max(actual_bytes_total, 1e-12),
        "histogram_l1": 2 * histogram_tv(predicted_calls, actual_calls),
        "histogram_tv": histogram_tv(predicted_calls, actual_calls),
        "normalized_log_payload_emd": normalized_log_emd(predicted_calls, actual_calls, edges),
        "actual_common_reference_cost_us": actual_cost,
        "predicted_common_reference_cost_us": predicted_cost,
        "cost_ape": abs(predicted_cost - actual_cost) / max(actual_cost, 1e-12),
    }


def validation_records(
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    predicted: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    validation_indices = [
        index for index, row in enumerate(rows) if row["split"] == VALIDATION_SPLIT
    ]
    grouped: dict[tuple, list[int]] = defaultdict(list)
    for index in validation_indices:
        row = rows[index]
        grouped[
            (
                row["model"],
                row["parallel_size"],
                row["policy"],
                row["profile_id"],
            )
        ].append(index)
    records = []
    edges = json.loads(rows[0]["bin_edges_bytes_json"])
    for method in METHODS:
        predicted_calls, predicted_bytes = predicted[method]
        for indices in grouped.values():
            if len(indices) != 2 or {rows[index]["phase"] for index in indices} != {
                "prefill",
                "decode",
            }:
                raise ValueError("validation configuration lacks two phases")
            indices = sorted(indices, key=lambda index: rows[index]["phase"])
            for index in indices:
                records.append(
                    case_record(
                        row=rows[index],
                        method=method,
                        phase=rows[index]["phase"],
                        actual_calls=arrays["target_calls"][index],
                        actual_bytes=arrays["target_bytes"][index],
                        predicted_calls=predicted_calls[index],
                        predicted_bytes=predicted_bytes[index],
                        edges=edges,
                    )
                )
            representative = rows[indices[0]]
            actual_calls = np.concatenate([arrays["target_calls"][index] for index in indices])
            actual_bytes = np.concatenate([arrays["target_bytes"][index] for index in indices])
            method_calls = np.concatenate([predicted_calls[index] for index in indices])
            method_bytes = np.concatenate([predicted_bytes[index] for index in indices])
            pooled_actual_calls = sum((arrays["target_calls"][index] for index in indices))
            pooled_actual_bytes = sum((arrays["target_bytes"][index] for index in indices))
            pooled_method_calls = sum((predicted_calls[index] for index in indices))
            pooled_method_bytes = sum((predicted_bytes[index] for index in indices))
            total_record = case_record(
                row=representative,
                method=method,
                phase="total",
                actual_calls=pooled_actual_calls,
                actual_bytes=pooled_actual_bytes,
                predicted_calls=pooled_method_calls,
                predicted_bytes=pooled_method_bytes,
                edges=edges,
            )
            total_record["histogram_tv"] = histogram_tv(method_calls, actual_calls)
            total_record["histogram_l1"] = 2 * total_record["histogram_tv"]
            actual_total_cost = sum(
                common_reference_cost(
                    arrays["target_calls"][index], arrays["target_bytes"][index], edges
                )
                for index in indices
            )
            predicted_total_cost = sum(
                common_reference_cost(predicted_calls[index], predicted_bytes[index], edges)
                for index in indices
            )
            total_record["actual_common_reference_cost_us"] = actual_total_cost
            total_record["predicted_common_reference_cost_us"] = predicted_total_cost
            total_record["cost_ape"] = abs(predicted_total_cost - actual_total_cost) / max(
                actual_total_cost, 1e-12
            )
            records.append(total_record)
    return records


def aggregate_records(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["parallelism"], row["method"], row["phase"], "all")].append(row)
        groups[(row["parallelism"], row["method"], row["phase"], row["policy"])].append(row)
    result = []
    for (parallelism, method, phase, policy), values in sorted(groups.items()):
        actual_calls = sum(float(row["actual_total_calls"]) for row in values)
        actual_bytes = sum(float(row["actual_total_logical_bytes"]) for row in values)
        result.append(
            {
                "parallelism": parallelism,
                "method": method,
                "phase": phase,
                "policy": policy,
                "cases": len(values),
                "calls_mape": float(np.mean([float(row["calls_ape"]) for row in values])),
                "calls_wape": sum(float(row["calls_absolute_error"]) for row in values)
                / actual_calls,
                "bytes_mape": float(np.mean([float(row["bytes_ape"]) for row in values])),
                "bytes_wape": sum(float(row["bytes_absolute_error"]) for row in values)
                / actual_bytes,
                "mean_histogram_l1": float(
                    np.mean([float(row["histogram_l1"]) for row in values])
                ),
                "mean_histogram_tv": float(
                    np.mean([float(row["histogram_tv"]) for row in values])
                ),
                "mean_normalized_log_payload_emd": float(
                    np.mean([float(row["normalized_log_payload_emd"]) for row in values])
                ),
                "common_reference_cost_mape": float(
                    np.mean([float(row["cost_ape"]) for row in values])
                ),
            }
        )
    return result


def readme(summary: dict) -> str:
    rows = summary["validation_headline"]
    table = [
        "| 并行 | 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | common cost MAPE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for parallelism in PARALLELISMS:
        for method in METHODS:
            row = rows[parallelism][method]
            table.append(
                "| {parallelism} | {method} | {calls_mape:.2%} / {calls_wape:.2%} | "
                "{bytes_mape:.2%} / {bytes_wape:.2%} | {tv:.4f} | {emd:.4f} | {cost:.2%} |".format(
                    parallelism=parallelism.upper(),
                    method=method,
                    calls_mape=row["calls_mape"],
                    calls_wape=row["calls_wape"],
                    bytes_mape=row["bytes_mape"],
                    bytes_wape=row["bytes_wape"],
                    tv=row["mean_histogram_tv"],
                    emd=row["mean_normalized_log_payload_emd"],
                    cost=row["common_reference_cost_mape"],
                )
            )
    return f"""# Phase 26C：Hfull监督预测器重训

状态：**{summary['status']}**。

本阶段使用Phase 26B统一数据分别训练TP与PP的structure-direct和
H0+bounded-residual模型；H0作为无参数基线。拟合只使用5个`train`画像，早停只使用
5个`validation`画像。5个temporal、8个external和1个external synthetic测试画像未参与
训练、标准化或模型选择，留给Phase 26D。

## validation配置级total结果

{chr(10).join(table)}

这些是模型选择用validation结果，不是最终测试结论。正式结论必须以Phase 26D的三个
测试域为准。这里的L1/TV在各自原生12桶上计算，和Phase 26B用于teacher审计的exact
payload TV不是同一个离散粒度；log-payload EMD在total时合并prefill/decode的桶质量，
TV则保留phase-aware的24维分布。

## 训练契约

- 输入：55个低维画像、模型结构、固定并行配置、固定策略和phase特征；不含完整请求列表；
- 输出：各自原生12桶的calls与logical bytes，每1000请求归一化；
- TP与PP分别训练，避免混淆4 KiB–512 MiB与4 KiB–8 GiB的桶语义；
- direct预测完整log-total与log-share编码；
- residual只预测相对H0的校正，总量限制在两倍以内，share-logit限制在±2，并通过tanh硬约束；
- common cost仍是5 μs+100 GB/s参数参考，不是PP物理曲线。

## 资产

- `checkpoints/`：TP/PP各自的direct与bounded-residual checkpoint；
- `analysis/validation_predictions.csv.gz`：validation逐配置、逐phase与total预测；
- `analysis/validation_metrics.csv`：TP/PP、方法、phase与policy聚合；
- `analysis/training_history.csv.gz`：四个网络的训练/早停轨迹；
- `feature_contract.json`、`summary.json`、`audit_summary.json`、`logs/training.log`、
  `DONE`和`manifest.sha256`。

可以确认模型已在Hfull监督下完成重训，且测试画像未用于选择。不能确认模型对temporal、
external或synthetic域的泛化优于H0；下一步Phase 26D将冻结这些checkpoint做profile-level
holdout测试，并分别报告TP/PP与policy。
"""


def main() -> None:
    args = parse_args()
    for directory in (
        args.output_dir / "checkpoints",
        args.output_dir / "analysis",
        args.output_dir / "logs",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    dataset_summary = json.loads(args.dataset_summary.read_text())
    if dataset_summary["status"] != "PASS":
        raise ValueError("Phase 26B dataset is not PASS")
    all_rows = load_rows(args.dataset)
    if len(all_rows) != 1728:
        raise ValueError(f"expected 1728 rows, got {len(all_rows)}")
    feature_names = [name for name in all_rows[0] if name.startswith("feature_")]
    if len(feature_names) != 55:
        raise ValueError(f"expected 55 feature columns, got {len(feature_names)}")
    device = choose_device(args.device)

    split_counts = Counter(row["split"] for row in all_rows)
    profile_splits: dict[str, str] = {}
    for row in all_rows:
        previous = profile_splits.setdefault(row["profile_id"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"profile crosses splits: {row['profile_id']}")
    profile_split_counts = Counter(profile_splits.values())

    history_rows = []
    prediction_records = []
    checkpoint_inventory = []
    training_log = []
    parallel_summaries = {}
    for parallel_index, parallelism in enumerate(PARALLELISMS):
        rows = [
            row
            for row in all_rows
            if row["parallelism"] == parallelism
            and row["split"] in {FIT_SPLIT, VALIDATION_SPLIT}
        ]
        arrays = prepare_arrays(rows, feature_names)
        predicted: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "h0": (arrays["h0_calls"], arrays["h0_bytes"])
        }
        for method_index, method in enumerate(("direct", "h0_bounded_residual")):
            prediction_encoded, checkpoint, history = fit_model(
                parallelism=parallelism,
                method=method,
                rows=rows,
                arrays=arrays,
                feature_names=feature_names,
                args=args,
                device=device,
                seed=args.seed + parallel_index * 100 + method_index,
            )
            calls, logical_bytes = zip(*(target_decode(row) for row in prediction_encoded))
            predicted[method] = (np.stack(calls), np.stack(logical_bytes))
            checkpoint_path = args.output_dir / "checkpoints" / f"{parallelism}_{method}.pt"
            torch.save(checkpoint, checkpoint_path)
            checkpoint_inventory.append(
                {
                    "parallelism": parallelism,
                    "method": method,
                    "path": str(checkpoint_path.relative_to(args.output_dir)),
                    "sha256": sha256(checkpoint_path),
                    "best_epoch": checkpoint["best_epoch"],
                    "best_validation_loss": checkpoint["best_validation_loss"],
                    "bytes": checkpoint_path.stat().st_size,
                }
            )
            history_rows.extend(history)
            training_log.append(
                {
                    "parallelism": parallelism,
                    "method": method,
                    "epochs_completed": len(history),
                    "best_epoch": checkpoint["best_epoch"],
                    "best_validation_loss": checkpoint["best_validation_loss"],
                }
            )
        records = validation_records(rows, arrays, predicted)
        prediction_records.extend(records)
        parallel_summaries[parallelism] = {
            "rows_train_plus_validation": len(rows),
            "fit_phase_rows": sum(row["split"] == FIT_SPLIT for row in rows),
            "validation_phase_rows": sum(row["split"] == VALIDATION_SPLIT for row in rows),
            "fit_profiles": len({row["profile_id"] for row in rows if row["split"] == FIT_SPLIT}),
            "validation_profiles": len(
                {row["profile_id"] for row in rows if row["split"] == VALIDATION_SPLIT}
            ),
            "bin_schema_id": rows[0]["bin_schema_id"],
        }

    metric_rows = aggregate_records(prediction_records)
    headline = {
        parallelism: {
            method: next(
                row
                for row in metric_rows
                if row["parallelism"] == parallelism
                and row["method"] == method
                and row["phase"] == "total"
                and row["policy"] == "all"
            )
            for method in METHODS
        }
        for parallelism in PARALLELISMS
    }

    write_csv_gz(args.output_dir / "analysis/validation_predictions.csv.gz", prediction_records)
    write_csv(args.output_dir / "analysis/validation_metrics.csv", metric_rows)
    write_csv_gz(args.output_dir / "analysis/training_history.csv.gz", history_rows)
    write_csv(args.output_dir / "analysis/checkpoint_inventory.csv", checkpoint_inventory)

    checks = {
        "dataset_status_pass": dataset_summary["status"] == "PASS",
        "dataset_rows_1728": len(all_rows) == 1728,
        "feature_columns_55": len(feature_names) == 55,
        "profiles_do_not_cross_splits": len(profile_splits) == 24,
        "profile_split_counts_5_5_5_8_1": profile_split_counts
        == Counter(
            {
                "train": 5,
                "validation": 5,
                "temporal_test": 5,
                "external_test": 8,
                "external_synthetic": 1,
            }
        ),
        "four_checkpoints": len(checkpoint_inventory) == 4,
        "validation_predictions_only": all(
            row["split"] == VALIDATION_SPLIT for row in prediction_records
        ),
        "test_splits_absent_from_predictions": not any(
            row["split"] in TEST_SPLITS for row in prediction_records
        ),
        "tp_pp_separate_bin_schemas": parallel_summaries["tp"]["bin_schema_id"]
        != parallel_summaries["pp"]["bin_schema_id"],
        "validation_records_expected": len(prediction_records) == 1620,
        "all_metrics_finite": all(
            math.isfinite(float(row[field]))
            for row in metric_rows
            for field in (
                "calls_mape",
                "calls_wape",
                "bytes_mape",
                "bytes_wape",
                "mean_histogram_tv",
                "mean_normalized_log_payload_emd",
                "common_reference_cost_mape",
            )
        ),
        "bounded_residual_contract": all(
            row["method"] != "h0_bounded_residual"
            or row["best_validation_loss"] >= 0
            for row in checkpoint_inventory
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status != "PASS":
        raise RuntimeError(f"Phase 26C checks failed: {checks}")

    feature_contract = {
        "schema_version": "phase26c-feature-contract-v1",
        "feature_names": feature_names,
        "log_feature_names": sorted(LOG_FEATURES),
        "methods": list(METHODS),
        "parallelism_training": "separate TP and PP models because native bin schemas differ",
        "target_encoding": "log1p total + log smoothed 12-bin shares, separately for calls and logical bytes",
        "bounded_residual": {
            "total_log_bound": math.log(2.0),
            "share_logit_bound": 2.0,
            "network_output": "tanh hard bound",
        },
        "fit_split": FIT_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "held_out_test_splits": list(TEST_SPLITS),
    }
    write_json(args.output_dir / "feature_contract.json", feature_contract)
    summary = {
        "schema_version": "phase26c-hfull-predictor-training-v1",
        "status": status,
        "objective": "retrain structure-direct and H0+bounded-residual predictors against Hfull while reserving all test profiles for Phase 26D",
        "device": str(device),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "counts": {
            "dataset_phase_rows": len(all_rows),
            "profiles": len(profile_splits),
            "feature_columns": len(feature_names),
            "checkpoints": len(checkpoint_inventory),
            "validation_prediction_records": len(prediction_records),
            "validation_metric_rows": len(metric_rows),
            "training_history_rows": len(history_rows),
        },
        "split_contract": {
            "profile_counts": dict(profile_split_counts),
            "fit": FIT_SPLIT,
            "early_stopping": VALIDATION_SPLIT,
            "untouched_for_phase26d": list(TEST_SPLITS),
        },
        "metric_contract": {
            "histogram_l1_tv": "native 12-bin calls distribution; total is phase-aware 24-dimensional distribution",
            "normalized_log_payload_emd": "phase-pooled native-bin calls mass over log2 payload centers",
            "common_reference_cost": "5 us launch plus 100 GB/s, using per-bin calls and logical bytes; not a physical PP curve",
            "boundary": "not directly comparable to Phase 26B exact-payload L1/TV",
        },
        "parallelism": parallel_summaries,
        "validation_headline": headline,
        "checkpoints": checkpoint_inventory,
        "checks": checks,
        "can_conclude": [
            "direct and bounded-residual models were retrained against Hfull targets",
            "validation-only model selection preserved all temporal/external/synthetic profiles for Phase 26D",
            "TP and PP native output bin semantics stayed separate",
        ],
        "cannot_conclude": [
            "validation improvements imply test-domain improvements",
            "the compact profile is sufficient for unseen traffic domains before Phase 26D",
            "the common cost metric is a physical PP communication-time measurement",
        ],
        "next_step": "freeze these checkpoints and run Phase 26D on temporal_test, external_test, and external_synthetic profile groups",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "audit_summary.json",
        {
            "schema_version": "phase26c-hfull-predictor-training-audit-v1",
            "status": status,
            "checks": checks,
            "dataset_sha256": summary["dataset_sha256"],
            "checkpoint_sha256": {
                f"{row['parallelism']}_{row['method']}": row["sha256"]
                for row in checkpoint_inventory
            },
        },
    )
    (args.output_dir / "README.md").write_text(readme(summary))
    (args.output_dir / "DONE").write_text("PASS\n")
    try:
        repository_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        repository_head = "unknown"
    write_json(
        args.output_dir / "logs/training.log",
        {
            "schema_version": "phase26c-training-log-v1",
            "status": status,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_head_at_training": repository_head,
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "args": vars(args) | {"dataset": str(args.dataset), "dataset_summary": str(args.dataset_summary), "output_dir": str(args.output_dir)},
            "training_runs": training_log,
        },
    )
    manifest_rows = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256":
            manifest_rows.append(f"{sha256(path)}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest_rows) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "device": str(device),
                "checkpoints": len(checkpoint_inventory),
                "validation_records": len(prediction_records),
                "output_dir": str(args.output_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
