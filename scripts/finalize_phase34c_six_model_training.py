#!/usr/bin/env python3
"""Finalize Phase34C after independent TP/PP target-free training completes."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from train_phase27c_pp_scheduler_feature_predictors import parse_histograms
from train_phase31c_known_model_residuals import aggregate, fixed_prediction_rows, headline
from train_phase33c_target_free_selection import bytes_anchor, pp_incumbent_prediction


EXISTING_MODELS = {"deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase34c-dir", type=Path, default=root / "experiment-results/phase34c_six_model_target_free_training")
    parser.add_argument("--phase34a-dir", type=Path, default=root / "experiment-results/phase34a_six_model_contract")
    parser.add_argument("--phase33c-dir", type=Path, default=root / "experiment-results/phase33c_target_free_model_selection")
    parser.add_argument("--phase32b-dir", type=Path, default=root / "experiment-results/phase32b_expanded_residual_search")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


def deterministic_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            output.write(text.encode())


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    for row in rows[1:]: fields.extend(name for name in row if name not in fields)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    for row in rows[1:]: fields.extend(name for name in row if name not in fields)
    buffer = io.StringIO(newline=""); writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def phase33_tp_prediction(rows: list[dict[str, str]], args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, np.ndarray, str]:
    from train_phase33c_target_free_selection import infer_tp
    summary = json.loads((args.phase33c_dir / "summary.json").read_text())
    selected = summary["tp"]["selected_candidate_id"]; base = selected.split("_5fold_")[0]
    inventory = [row for row in read_csv(args.phase33c_dir / "analysis/checkpoint_inventory.csv") if row["parallelism"] == "tp" and row["candidate_id"] == base]
    if len(inventory) != 3: raise RuntimeError(f"Phase33 selected TP checkpoint count: {len(inventory)}")
    bundles = [torch.load(args.phase33c_dir / row["path"], map_location="cpu", weights_only=False)["folds"] for row in inventory]
    prediction = infer_tp(rows, bundles, float(summary["tp"]["alpha"]), device)
    return prediction[0], prediction[1], selected


def phase33_pp_prediction(rows: list[dict[str, str]], args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, np.ndarray, str]:
    summary = json.loads((args.phase33c_dir / "summary.json").read_text())
    paths = sorted((args.phase32b_dir / "checkpoints").glob("pp_top1_seed*.pt"))
    if len(paths) != 3: raise RuntimeError("Phase32 PP source checkpoints != 3")
    calls, _ = pp_incumbent_prediction(rows, paths, device)
    _, h0_bytes = parse_histograms(rows, "h0")
    return calls, bytes_anchor(rows, h0_bytes, "pp"), summary["pp"]["selected_candidate_id"]


def subset_metrics(path: Path) -> dict:
    records = [row for row in read_csv(path) if row["model"] in EXISTING_MODELS]
    return {method: headline(aggregate([row for row in records if row["method"] == method])) for method in sorted({row["method"] for row in records})}


def main() -> None:
    args = parse_args(); (args.phase34c_dir / "analysis").mkdir(parents=True, exist_ok=True); (args.phase34c_dir / "logs").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA unavailable")
    summaries = {p: json.loads((args.phase34c_dir / p / "summary.json").read_text()) for p in ("tp", "pp")}
    if any(summaries[p]["status"] != "PASS" for p in summaries): raise RuntimeError("direction training incomplete")
    phase34a = json.loads((args.phase34a_dir / "summary.json").read_text())
    if phase34a["blind_confirmation"]["target_state"] != "not_generated" or (args.phase34a_dir / "labels").exists(): raise RuntimeError("Phase34 blind target already exists")

    combined_frozen = []
    for parallelism in ("tp", "pp"):
        for row in read_csv(args.phase34c_dir / parallelism / "analysis/frozen_predictions.csv.gz"):
            row["model_version"] = "phase34_six_model"; combined_frozen.append(row)

    # Freeze the Phase33 three-model incumbents on exactly the new Phase34 blind windows before opening targets.
    baseline_inventory = []
    for parallelism in ("tp", "pp"):
        rows = [row for row in read_csv(args.phase34a_dir / f"dataset/{parallelism}_blind_confirmation_features.csv.gz") if row["model"] in EXISTING_MODELS]
        if any(name.startswith("target_") for name in rows[0]): raise RuntimeError("baseline feature target exposure")
        h0_calls, h0_bytes = parse_histograms(rows, "h0")
        if parallelism == "tp": calls, logical_bytes, selected = phase33_tp_prediction(rows, args, device)
        else: calls, logical_bytes, selected = phase33_pp_prediction(rows, args, device)
        output = fixed_prediction_rows(rows, {"h0": (h0_calls, h0_bytes), "h0_plus_dnn_residual": (calls, logical_bytes)}, parallelism, selected)
        for row in output:
            row["prediction_set"] = "phase34_blind_new"; row["model_version"] = "phase33_three_model_baseline"; combined_frozen.append(row)
        baseline_inventory.append({"parallelism": parallelism, "selected_candidate_id": selected, "models": 3, "profiles": 12, "prediction_rows": len(output)})

    write_csv_gz(args.phase34c_dir / "analysis/frozen_predictions_all_versions.csv.gz", combined_frozen)
    frozen_sha = sha256(args.phase34c_dir / "analysis/frozen_predictions_all_versions.csv.gz")
    write_csv(args.phase34c_dir / "analysis/phase33_baseline_inventory.csv", baseline_inventory)
    checkpoint_inventory = []
    for parallelism in ("tp", "pp"):
        for row in read_csv(args.phase34c_dir / parallelism / "analysis/checkpoint_inventory.csv"):
            row["direction_path"] = f"{parallelism}/{row['path']}"; checkpoint_inventory.append(row)
    write_csv(args.phase34c_dir / "analysis/checkpoint_inventory.csv", checkpoint_inventory)

    phase33 = json.loads((args.phase33c_dir / "summary.json").read_text())
    old_three_metrics = {p: subset_metrics(args.phase34c_dir / p / "analysis/grouped_cv_predictions_and_metrics.csv.gz") for p in ("tp", "pp")}
    comparison = {
        "tp": {"phase33_three_model_development_cv": phase33["tp"]["development_cv_headline"], "phase34_six_model_predictor_on_original_three_development_oof": old_three_metrics["tp"], "comparability": "same 94 profiles and original three models; Phase34 retrained with six-model rows"},
        "pp": {"phase33_three_model_development_validation": phase33["pp"]["development_validation_headline"], "phase34_six_model_predictor_on_original_three_development_oof": old_three_metrics["pp"], "comparability": "indicative only: Phase33 value used fresh-role validation with retained incumbent; Phase34 is full 94-profile OOF retraining"},
    }
    write_json(args.phase34c_dir / "analysis/phase33_development_comparison.json", comparison)

    checks = {
        "tp_and_pp_training_pass": all(summaries[p]["status"] == "PASS" for p in summaries),
        "new_blind_target_still_absent": phase34a["blind_confirmation"]["target_state"] == "not_generated" and not (args.phase34a_dir / "labels").exists(),
        "both_phase34_six_model_predictions_frozen": all(any(row["parallelism"] == p and row["prediction_set"] == "phase34_blind_new" and row["model_version"] == "phase34_six_model" for row in combined_frozen) for p in ("tp", "pp")),
        "phase33_three_model_baselines_frozen_on_same_new_windows": all(any(row["parallelism"] == p and row["prediction_set"] == "phase34_blind_new" and row["model_version"] == "phase33_three_model_baseline" for row in combined_frozen) for p in ("tp", "pp")),
        "nine_checkpoints_each_direction": len(checkpoint_inventory) == 18 and all(sum(row["parallelism"] == p for row in checkpoint_inventory) == 9 for p in ("tp", "pp")),
        "nonzero_dnn_residual_both_directions": all(summaries[p]["checks"]["selected_nonzero_dnn_calls_residual"] for p in summaries),
        "profile_grouped_fivefold_both_directions": all(summaries[p]["checks"]["profile_grouped_fivefold_no_derived_row_leakage"] for p in summaries),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase34c-six-model-target-free-training-freeze-v1", "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "models": phase34a["models"],
        "data": {"development_profiles": 94, "development_train": 75, "development_validation": 19, "unique_teacher_requests": 35524, "new_blind_profiles": 12, "new_blind_future_teacher_requests": phase34a["blind_confirmation"]["requests"]},
        "tp": summaries["tp"]["selected"], "pp": summaries["pp"]["selected"],
        "search": {"tp_regular": 18, "pp_regular": 18, "absolute_limit_each": 24, "top3_each": True, "seeds": 3, "folds": 5},
        "phase33_development_comparison": comparison,
        "target_isolation": {"phase34_blind_target_state": "not_generated", "phase34_blind_targets_read": False, "phase33_open_targets_read_for_training_or_selection": False},
        "frozen_prediction_sha256": frozen_sha, "counts": {"combined_frozen_rows": len(combined_frozen), "phase34_checkpoints": len(checkpoint_inventory)}, "checks": checks,
    }
    write_json(args.phase34c_dir / "summary.json", summary)
    write_json(args.phase34c_dir / "audit_summary.json", {"schema_version": "phase34c-finalize-audit-v1", "status": status, "checks": checks, "frozen_prediction_sha256": frozen_sha})
    write_json(args.phase34c_dir / "logs/finalize.log", {"event": "phase34c_target_free_predictions_and_checkpoints_frozen", "status": status, "frozen_prediction_sha256": frozen_sha, "phase34_blind_targets_read": False})
    tp, pp = summaries["tp"]["selected"]["development_cv_headline"], summaries["pp"]["selected"]["development_cv_headline"]
    (args.phase34c_dir / "README.md").write_text(f"""# Phase 34C：六模型TP/PP训练与新确认预测冻结

本阶段在同一94个开发画像、35,524个唯一完整teacher请求上，分别重新训练六模型TP与PP `H0 + DNN residual`。两个方向均完成18组常规初筛和前三名3-seed × 5-fold确认；同一画像派生的六模型、并行配置、policy和phase从未跨折。

TP开发OOF的calls/bytes/TV/EMD/cost WAPE为`{tp['calls_wape']:.2%}`、`{tp['bytes_wape']:.2%}`、`{tp['mean_histogram_tv']:.4f}`、`{tp['mean_normalized_log_payload_emd']:.4f}`、`{tp['common_reference_cost_wape']:.2%}`。PP对应为`{pp['calls_wape']:.2%}`、`{pp['bytes_wape']:.2%}`、`{pp['mean_histogram_tv']:.4f}`、`{pp['mean_normalized_log_payload_emd']:.4f}`、`{pp['common_reference_cost_wape']:.2%}`。两个选中模型都保留非零DNN residual。

在Phase34新确认target不存在时，已冻结六模型TP/PP预测；同时把Phase33三模型incumbent对同一批新窗口的预测作为可比基线冻结。合并冻结文件SHA-256为`{frozen_sha}`。只有本目录完成Git归档后，才允许一次性打开12个新确认窗口的Hfull target。
""")
    (args.phase34c_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.phase34c_dir)}" for path in sorted(args.phase34c_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.phase34c_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS": raise RuntimeError(checks)
    print(json.dumps({"status": status, "frozen_prediction_sha256": frozen_sha, "tp": tp, "pp": pp}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
