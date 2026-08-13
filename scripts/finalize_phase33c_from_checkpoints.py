#!/usr/bin/env python3
"""Finalize Phase33C after training completed but JSON serialization failed."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from train_phase27c_pp_scheduler_feature_predictors import ENCODED_SIZE, prepare_development
from train_phase31c_known_model_residuals import aggregate, feature_sets, headline
from train_phase32b_expanded_residual_search import predict_checkpoint
from train_phase33c_target_free_selection import (
    FOLDS,
    PP_FORMAL,
    TP_FORMAL,
    all_records,
    bytes_anchor,
    evaluate_tp,
    fold_map,
    metric_bundle,
    pp_incumbent_prediction,
    apply_pp_candidate,
    read_csv_gz,
    sha256,
    target_free,
    write_json,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase33a-dir", type=Path, default=root / "experiment-results/phase33a_fresh_data_contract")
    parser.add_argument("--phase33b-dir", type=Path, default=root / "experiment-results/phase33b_expanded_development_dataset")
    parser.add_argument("--phase31b-dir", type=Path, default=root / "experiment-results/phase31b_known_model_hfull_dataset")
    parser.add_argument("--phase32a-dir", type=Path, default=root / "experiment-results/phase32a_expanded_search_contract")
    parser.add_argument("--phase32b-dir", type=Path, default=root / "experiment-results/phase32b_expanded_residual_search")
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase33c_target_free_model_selection")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def oof_from_bundle(rows: list[dict[str, str]], folds: dict[str, int], path: Path, device: torch.device) -> np.ndarray:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    result = np.zeros((len(rows), ENCODED_SIZE), dtype=np.float32)
    for checkpoint in bundle["folds"]:
        fold = int(checkpoint["fold"])
        indices = [index for index, row in enumerate(rows) if folds[row["profile_id"]] == fold]
        part = [rows[index] for index in indices]
        arrays = prepare_development(part)
        result[indices] = predict_checkpoint(part, arrays["h0_encoded"], checkpoint, device)
    return result


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA unavailable")
    phase33a = json.loads((args.phase33a_dir / "summary.json").read_text())
    phase33b = json.loads((args.phase33b_dir / "summary.json").read_text())
    if phase33a["target_state"]["blind_confirmation"] != "not_generated" or phase33b["blind_confirmation_target_state"] != "not_generated":
        raise RuntimeError("blind target is no longer sealed")

    output = args.output_dir
    required = [output / "analysis/candidate_grid.csv", output / "analysis/checkpoint_inventory.csv", output / "analysis/frozen_predictions.csv.gz", output / "analysis/development_predictions_and_metrics.csv.gz", output / "checkpoints/pp_bytes_calibration.json"]
    required.extend(output / f"checkpoints/tp_top{rank}_seed{seed}.pt" for rank in (1, 2, 3) for seed in (20260813, 20260914, 20261015))
    if not all(path.is_file() for path in required):
        raise RuntimeError([str(path) for path in required if not path.is_file()])

    tp_rows = read_csv_gz(args.phase33b_dir / "dataset/tp_combined_development_examples.csv.gz")
    tp_arrays = prepare_development(tp_rows)
    folds = fold_map(tp_rows)
    confirmed = []
    for rank in (1, 2, 3):
        paths = sorted((output / "checkpoints").glob(f"tp_top{rank}_seed*.pt"))
        seed_oof = [oof_from_bundle(tp_rows, folds, path, device) for path in paths]
        bundle = torch.load(paths[0], map_location="cpu", weights_only=False)
        result = evaluate_tp(tp_rows, tp_arrays, np.mean(seed_oof, axis=0), bundle["candidate_id"] + "_3seed")
        confirmed.append({"rank_at_screen": rank, "candidate_id": bundle["candidate_id"], "config": bundle["folds"][0]["config"], **result})
    confirmed.sort(key=lambda value: value["score"])
    tp_best = confirmed[0]
    tp_id = tp_best["candidate_id"] + f"_5fold_3seed_alpha{tp_best['alpha']}"
    tp_h0 = metric_bundle(tp_rows, tp_arrays, (tp_arrays["h0_calls"], tp_arrays["h0_bytes"]), "tp", "tp_h0")
    tp_anchor_rel = np.max(np.abs(bytes_anchor(tp_rows, tp_arrays["h0_bytes"], "tp").sum(axis=1) - tp_arrays["target_bytes"].sum(axis=1)) / np.maximum(tp_arrays["target_bytes"].sum(axis=1), 1e-12))

    pp_rows = read_csv_gz(args.phase33b_dir / "dataset/pp_new_development_examples.csv.gz")
    pp_arrays = prepare_development(pp_rows)
    pp_paths = sorted((args.phase32b_dir / "checkpoints").glob("pp_top1_seed*.pt"))
    pp_calls, pp_raw_bytes = pp_incumbent_prediction(pp_rows, pp_paths, device)
    pp_anchor = bytes_anchor(pp_rows, pp_arrays["h0_bytes"], "pp")
    pp_anchor_rel = np.max(np.abs(pp_anchor.sum(axis=1) - pp_arrays["target_bytes"].sum(axis=1)) / np.maximum(pp_arrays["target_bytes"].sum(axis=1), 1e-12))
    validation = [index for index, row in enumerate(pp_rows) if row["split_role"] == "development_validation"]
    pp_val_rows = [pp_rows[index] for index in validation]
    pp_val_arrays = {key: value[validation] for key, value in pp_arrays.items()}
    calibration = json.loads((output / "checkpoints/pp_bytes_calibration.json").read_text())
    pp_config = calibration["configuration"]
    pp_calibrated = apply_pp_candidate(pp_rows, pp_raw_bytes, pp_anchor, pp_config)
    pp_best = metric_bundle(pp_val_rows, pp_val_arrays, (pp_calls[validation], pp_calibrated[validation]), "pp", calibration["selected_candidate_id"])
    pp_incumbent = metric_bundle(pp_val_rows, pp_val_arrays, (pp_calls[validation], pp_raw_bytes[validation]), "pp", "pp32_incumbent")
    pp_h0 = metric_bundle(pp_val_rows, pp_val_arrays, (pp_arrays["h0_calls"][validation], pp_arrays["h0_bytes"][validation]), "pp", "pp_h0")

    frozen_path = output / "analysis/frozen_predictions.csv.gz"
    frozen = read_csv_gz(frozen_path)
    feature_sets = {
        "phase33_blind": {p: read_csv_gz(args.phase33a_dir / f"dataset/{p}_blind_confirmation_features.csv.gz") for p in ("tp", "pp")},
        "phase31_fixed_repeated": {p: read_csv_gz(args.phase31b_dir / f"dataset/{p}_fixed_prediction_features.csv.gz") for p in ("tp", "pp")},
        "phase32_confirmation_repeated": {p: read_csv_gz(args.phase32a_dir / f"dataset/{p}_new_confirmation_features.csv.gz") for p in ("tp", "pp")},
    }
    for name, directions in feature_sets.items():
        for parallelism, rows in directions.items():
            target_free(rows, f"{name}/{parallelism}")

    tp_per_model = {row["model"]: row for row in tp_best["metrics"] if row["phase"] == "total" and row["policy"] == "all" and row["model"] != "all"}
    tp_formal = all(float(tp_best["headline"][key]) <= threshold for key, threshold in TP_FORMAL.items())
    tp_positive = float(tp_best["headline"]["calls_wape"]) < float(tp_h0["headline"]["calls_wape"]) and float(tp_best["headline"]["common_reference_cost_wape"]) < float(tp_h0["headline"]["common_reference_cost_wape"])
    tp_models_ok = all(float(row["calls_wape"]) <= 0.15 and float(row["common_reference_cost_wape"]) <= 0.08 for row in tp_per_model.values())
    pp_formal = all(float(pp_best["headline"][key]) <= threshold for key, threshold in PP_FORMAL.items())
    pp_protected = all(abs(float(pp_best["headline"][key]) - float(pp_incumbent["headline"][key])) <= 1e-12 for key in ("calls_wape", "mean_histogram_tv", "mean_normalized_log_payload_emd")) and float(pp_best["headline"]["common_reference_cost_wape"]) <= float(pp_incumbent["headline"]["common_reference_cost_wape"])

    inventory = read_csv(output / "analysis/checkpoint_inventory.csv")
    grid = read_csv(output / "analysis/candidate_grid.csv")
    checks = {
        "phase33_blind_target_absent_and_unread": phase33a["target_state"]["blind_confirmation"] == "not_generated",
        "all_three_prediction_sets_target_free": all(not any(key.startswith("target_") for key in rows[0]) for directions in feature_sets.values() for rows in directions.values()),
        "tp_18_regular_candidates": sum(row["candidate_id"].startswith("tp33_") for row in grid) == 18,
        "tp_top3_three_seed_fivefold": len(inventory) == 9 and len({row["rank"] for row in inventory}) == 3,
        "tp_profile_grouped_fivefold": len(set(folds.values())) == FOLDS,
        "pp_8_conservative_candidates": sum(row["candidate_id"].startswith("pp33_") for row in grid) == 8,
        "pp_incumbent_calls_shape_protected": bool(pp_protected),
        "bytes_anchor_matches_development_teacher": bool(tp_anchor_rel < 1e-10 and pp_anchor_rel < 1e-10),
        "frozen_methods_h0_and_dnn": {row["method"] for row in frozen} == {"h0", "h0_plus_dnn_residual"},
        "frozen_three_prediction_sets": {row["prediction_set"] for row in frozen} == set(feature_sets),
        "frozen_rows_6048": len(frozen) == 6048,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase33c-target-free-model-selection-v1", "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
        "recovery_note": "Training and prediction freezing completed in the original run; final metadata was rebuilt from all nine saved checkpoints after a NumPy boolean JSON serialization error.",
        "search": {"tp_regular_candidates": 18, "tp_absolute_limit": 24, "tp_top_candidates": 3, "tp_confirmation_seeds": 3, "folds": 5, "pp_regular_candidates": 8, "pp_absolute_limit": 12},
        "tp": {"selected_candidate_id": tp_id, "config": tp_best["config"], "alpha": tp_best["alpha"], "development_cv_headline": tp_best["headline"], "h0_development_cv_headline": tp_h0["headline"], "per_model": tp_per_model, "formal_development_pass": bool(tp_formal and tp_positive and tp_models_ok), "positive_vs_h0": bool(tp_positive), "no_model_severe_regression": bool(tp_models_ok), "top3": [{"rank_at_screen": value["rank_at_screen"], "candidate_id": value["candidate_id"], "score": value["score"], "alpha": value["alpha"], "headline": value["headline"]} for value in confirmed]},
        "pp": {"selected_candidate_id": calibration["selected_candidate_id"], "configuration": pp_config, "development_validation_headline": pp_best["headline"], "phase32_incumbent_on_same_validation": pp_incumbent["headline"], "h0_on_same_validation": pp_h0["headline"], "formal_development_pass": bool(pp_formal and pp_protected), "incumbent_calls_shape_protected": bool(pp_protected)},
        "bytes_anchor": {"definition": "allowed low-dimensional capped mean × model bytes/token prior × structural communication multiplier × 1000; H0 bin shape retained", "tp_max_development_target_relative_error": float(tp_anchor_rel), "pp_max_development_target_relative_error": float(pp_anchor_rel)},
        "target_isolation": {"phase33_blind_targets_read": False, "phase31_fixed_targets_read": False, "phase32_confirmation_targets_read": False},
        "frozen_prediction_sha256": sha256(frozen_path),
        "counts": {"tp_development_profiles": len({row["profile_id"] for row in tp_rows}), "pp_fresh_development_profiles": len({row["profile_id"] for row in pp_rows}), "frozen_rows": len(frozen), "checkpoints": len(inventory)},
        "checks": checks,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "audit_summary.json", {"schema_version": "phase33c-audit-v1", "status": status, "checks": checks, "frozen_prediction_sha256": sha256(frozen_path)})
    write_json(output / "logs/training.log", {"event": "phase33c_target_free_selection_finalized_from_checkpoints", "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "repository_head_at_finalization": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(), "phase33_blind_targets_read": False})
    (output / "figures").mkdir(exist_ok=True)
    values = [(row["candidate_id"], float(row["score"]), "#2563eb" if row["candidate_id"].startswith("tp33") else "#ea580c") for row in grid]
    width, height, margin = 1200, 520, 55; maximum = max(value for _, value, _ in values) * 1.05; bar_width = (width - 2 * margin) / len(values)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="55" y="30" font-family="sans-serif" font-size="20">Phase33开发侧有限候选综合分数（越低越好）</text>']
    for index, (_, value, color) in enumerate(values):
        bar_height = value / maximum * (height - 120); x = margin + index * bar_width; y = height - 65 - bar_height
        svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 2, 1):.2f}" height="{bar_height:.2f}" fill="{color}"/>')
    svg.append('</svg>'); (output / "figures/candidate_scores.svg").write_text("\n".join(svg) + "\n")
    (output / "README.md").write_text(f"""# Phase 33C：TP继续收敛与PP保守改进（打开新确认真值前）

本阶段只在开发数据上选模型。TP使用Phase31与Phase33合并后的94个开发画像、35524个完整teacher请求，比较18组`H0 + DNN residual`；每组先1个seed，开发前三名再做3-seed、5折profile分组确认。PP不重训calls/形状网络，保留Phase32 incumbent，只在45个全新开发画像上比较8种独立bytes校准。

bytes总量锚点来自部署时允许的低维均值、模型bytes/token先验和已验证结构通信倍数，不读取完整请求列表或确认target；开发集审计与Hfull teacher最大相对误差为TP `{tp_anchor_rel:.3e}`、PP `{pp_anchor_rel:.3e}`。bytes的12-bin形状仍保留H0分配。

TP开发五折calls/bytes/TV/EMD/cost WAPE为`{tp_best['headline']['calls_wape']:.2%}`、`{tp_best['headline']['bytes_wape']:.2%}`、`{tp_best['headline']['mean_histogram_tv']:.4f}`、`{tp_best['headline']['mean_normalized_log_payload_emd']:.4f}`、`{tp_best['headline']['common_reference_cost_wape']:.2%}`。PP新验证结果为`{pp_best['headline']['calls_wape']:.2%}`、`{pp_best['headline']['bytes_wape']:.2%}`、`{pp_best['headline']['mean_histogram_tv']:.4f}`、`{pp_best['headline']['mean_normalized_log_payload_emd']:.4f}`、`{pp_best['headline']['common_reference_cost_wape']:.2%}`。

9个Phase33全新确认窗口的Hfull target仍不存在。三套预测冻结SHA-256为`{sha256(frozen_path)}`。Phase31固定集和Phase32确认集后续只能作为重复工程证据。原训练运行在写summary时遇到NumPy布尔序列化错误；模型、九个checkpoint和冻结预测均已完成，本元数据由九个checkpoint重新推断验证后恢复，没有重训或改变候选。
""")
    (output / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(output)}" for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (output / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "tp": summary["tp"], "pp": summary["pp"], "frozen_prediction_sha256": summary["frozen_prediction_sha256"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
