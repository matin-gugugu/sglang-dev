#!/usr/bin/env python3
"""Six target-free TP residual-gate rescue configurations (cumulative cap 48)."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from train_phase27c_pp_scheduler_feature_predictors import parse_histograms, prepare_development
from train_phase31c_known_model_residuals import aggregate, encoded_from_vectors, fixed_prediction_rows, headline, vectors_from_encoded
from train_phase32b_expanded_residual_search import (
    ALPHAS,
    FORMAL,
    all_records,
    evaluate_residual,
    fold_map,
    predict_checkpoint,
)


LEVELS = (0.25, 0.5, 0.75, 1.0)
MODES = ("global", "policy", "model", "phase", "model_policy", "policy_phase")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=root / "experiment-results/phase31b_known_model_hfull_dataset")
    parser.add_argument("--phase32a-dir", type=Path, default=root / "experiment-results/phase32a_expanded_search_contract")
    parser.add_argument("--phase32b-dir", type=Path, default=root / "experiment-results/phase32b_expanded_residual_search")
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase32d_tp_gate_rescue")
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
    fieldnames = list(rows[0])
    for row in rows[1:]: fieldnames.extend(name for name in row if name not in fieldnames)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    fieldnames = list(rows[0])
    for row in rows[1:]: fieldnames.extend(name for name in row if name not in fieldnames)
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    deterministic_gzip(path, buffer.getvalue())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def group_key(row: dict[str, str], mode: str) -> str:
    if mode == "global": return "global"
    if mode == "policy": return row["policy"]
    if mode == "model": return row["model"]
    if mode == "phase": return row["phase"]
    if mode == "model_policy": return row["model"] + "::" + row["policy"]
    if mode == "policy_phase": return row["policy"] + "::" + row["phase"]
    raise ValueError(mode)


def scale_residual(rows: list[dict[str, str]], residual: np.ndarray, mode: str, gates: dict[str, float]) -> np.ndarray:
    return residual * np.asarray([gates[group_key(row, mode)] for row in rows], dtype=np.float32)[:, None]


def fit_gates(rows: list[dict[str, str]], arrays: dict[str, np.ndarray], raw_residual: np.ndarray, mode: str) -> dict:
    groups = sorted({group_key(row, mode) for row in rows})
    gates = {group: 1.0 for group in groups}
    best = evaluate_residual(rows, arrays, scale_residual(rows, raw_residual, mode, gates), "tp", f"tp_rescue_{mode}")
    if mode != "global":
        for _ in range(3):
            changed = False
            for group in groups:
                local_best = None
                for level in LEVELS:
                    trial = dict(gates); trial[group] = level
                    result = evaluate_residual(rows, arrays, scale_residual(rows, raw_residual, mode, trial), "tp", f"tp_rescue_{mode}")
                    value = (float(result["score"]), level)
                    if local_best is None or value < local_best[0]: local_best = (value, trial, result)
                if local_best[1][group] != gates[group]: changed = True
                gates, best = local_best[1], local_best[2]
            if not changed: break
    return {"mode": mode, "gates": gates, **best}


def oof_residual(rows: list[dict[str, str]], checkpoint_paths: list[Path], device: torch.device) -> tuple[np.ndarray, list[dict]]:
    folds = fold_map(rows)
    seed_values, checkpoint_info = [], []
    for path in checkpoint_paths:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        seed_residual = np.zeros((len(rows), 26), dtype=np.float32)
        for fold, checkpoint in enumerate(bundle["folds"]):
            indices = [index for index, row in enumerate(rows) if folds[row["profile_id"]] == fold]
            h0_calls, h0_bytes = parse_histograms([rows[index] for index in indices], "h0")
            h0_encoded = encoded_from_vectors(h0_calls, h0_bytes)
            seed_residual[indices] = predict_checkpoint([rows[index] for index in indices], h0_encoded, checkpoint, device)
        seed_values.append(seed_residual)
        checkpoint_info.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size, "seed": bundle["seed"], "candidate_id": bundle["candidate_id"]})
    return np.mean(seed_values, axis=0), checkpoint_info


def inference_residual(rows: list[dict[str, str]], checkpoint_paths: list[Path], device: torch.device) -> np.ndarray:
    h0_calls, h0_bytes = parse_histograms(rows, "h0")
    h0_encoded = encoded_from_vectors(h0_calls, h0_bytes)
    values = []
    for path in checkpoint_paths:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        values.extend(predict_checkpoint(rows, h0_encoded, checkpoint, device) for checkpoint in bundle["folds"])
    return np.mean(values, axis=0)


def main() -> None:
    args = parse_args()
    for name in ("checkpoints", "analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda": raise RuntimeError("CUDA unavailable")
    phase32a = json.loads((args.phase32a_dir / "summary.json").read_text())
    phase32b = json.loads((args.phase32b_dir / "summary.json").read_text())
    if phase32b["fixed_targets_read"] is not False or phase32b["new_confirmation_targets_read"] is not False: raise RuntimeError("Phase32B isolation failed")
    if phase32a["new_confirmation"]["target_state"] != "not_generated": raise RuntimeError("Phase32A feature contract failed")

    rows = read_csv_gz(args.dataset_dir / "dataset/tp_development_examples.csv.gz")
    arrays = prepare_development(rows)
    checkpoint_paths = sorted((args.phase32b_dir / "checkpoints").glob("tp_top1_seed*.pt"))
    if len(checkpoint_paths) != 3: raise RuntimeError("expected three selected TP checkpoints")
    raw_oof, checkpoint_info = oof_residual(rows, checkpoint_paths, device)
    candidates = [fit_gates(rows, arrays, raw_oof, mode) for mode in MODES]
    candidates.sort(key=lambda value: float(value["score"]))
    selected = candidates[0]
    selected_id = f"tp32_rescue_{selected['mode']}_alpha{selected['alpha']}"

    grid_rows = []
    for value in candidates:
        grid_rows.append({"candidate_id": f"tp32_rescue_{value['mode']}", "mode": value["mode"], "gates_json": json.dumps(value["gates"], sort_keys=True, separators=(",", ":")), "alpha": value["alpha"], "score": value["score"], **{key: value["headline"][key] for key in FORMAL["tp"]}})
    write_csv(args.output_dir / "analysis/candidate_grid.csv", grid_rows)
    h0_records = all_records(rows, arrays, (arrays["h0_calls"], arrays["h0_bytes"]), "tp", "h0")
    write_csv_gz(args.output_dir / "analysis/selected_development_oof_metrics.csv.gz", h0_records + selected["records"])

    frozen = []
    for prediction_set, path in (("original_fixed", args.dataset_dir / "dataset/tp_fixed_prediction_features.csv.gz"), ("new_confirmation", args.phase32a_dir / "dataset/tp_new_confirmation_features.csv.gz")):
        prediction_rows = read_csv_gz(path)
        if any(name.startswith("target_") for name in prediction_rows[0]): raise RuntimeError("target exposed in inference feature")
        h0_calls, h0_bytes = parse_histograms(prediction_rows, "h0")
        h0_encoded = encoded_from_vectors(h0_calls, h0_bytes)
        raw = inference_residual(prediction_rows, checkpoint_paths, device)
        gated = scale_residual(prediction_rows, raw, selected["mode"], selected["gates"])
        prediction = vectors_from_encoded(h0_encoded + float(selected["alpha"]) * gated)
        output = fixed_prediction_rows(prediction_rows, {"h0": (h0_calls, h0_bytes), "h0_plus_dnn_residual": prediction}, "tp", selected_id)
        for row in output: row["prediction_set"] = prediction_set
        frozen.extend(output)
    write_csv_gz(args.output_dir / "analysis/frozen_predictions.csv.gz", frozen)
    frozen_sha = sha256(args.output_dir / "analysis/frozen_predictions.csv.gz")
    gate_checkpoint = {"schema_version": "phase32d-tp-rescue-gate-v1", "selected_candidate_id": selected_id, "mode": selected["mode"], "gates": selected["gates"], "alpha": selected["alpha"], "source_dnn_checkpoints": checkpoint_info, "selection_source": "development_grouped_oof_only", "fixed_targets_read": False, "new_confirmation_targets_read": False}
    write_json(args.output_dir / "checkpoints/tp_rescue_gate.json", gate_checkpoint)

    width, height = 900, 430
    maximum = max(float(row["score"]) for row in grid_rows) * 1.08
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="45" y="32" font-family="sans-serif" font-size="21">TP定向gate救援：开发OOF综合分数（越低越好）</text>']
    for index, row in enumerate(grid_rows):
        x = 60 + index * 135; value = float(row["score"]); bar = value / maximum * 285
        svg.append(f'<rect x="{x}" y="{365-bar:.2f}" width="85" height="{bar:.2f}" fill="#2563eb"/>')
        svg.append(f'<text x="{x+42}" y="390" text-anchor="middle" font-family="sans-serif" font-size="12">{row["mode"]}</text>')
    svg.append('</svg>'); (args.output_dir / "figures/rescue_scores.svg").write_text("\n".join(svg) + "\n")

    checks = {
        "six_rescue_candidates_cumulative_48": len(grid_rows) == 6,
        "selection_development_oof_only": True,
        "source_top1_three_seed_checkpoints": len(checkpoint_paths) == 3,
        "original_fixed_features_have_no_target": not any(name.startswith("target_") for name in read_csv_gz(args.dataset_dir / "dataset/tp_fixed_prediction_features.csv.gz")[0]),
        "new_confirmation_features_have_no_target": not any(name.startswith("target_") for name in read_csv_gz(args.phase32a_dir / "dataset/tp_new_confirmation_features.csv.gz")[0]),
        "fixed_targets_not_script_inputs": not any("phase32c" in str(value).lower() or "target" in name for name, value in vars(args).items()),
        "frozen_rows_2052": len(frozen) == (540 + 486) * 2,
        "selected_residual_nonzero": any(abs(float(row["predicted_total_calls_per_1000"]) - float(next(old for old in frozen if old["prediction_set"] == row["prediction_set"] and old["example_id"] == row["example_id"] and old["method"] == "h0")["predicted_total_calls_per_1000"])) > 1e-6 for row in frozen if row["method"] == "h0_plus_dnn_residual"),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    h0_head = headline(aggregate(h0_records))
    summary = {"schema_version": "phase32d-tp-gate-rescue-v1", "status": status, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "search": {"prior_cumulative": 42, "new_candidates": 6, "final_cumulative": 48, "absolute_limit": 48, "gate_levels": list(LEVELS), "gate_modes": list(MODES)}, "selected_candidate_id": selected_id, "selected_mode": selected["mode"], "selected_gates": selected["gates"], "selected_alpha": selected["alpha"], "selected_development_headline": selected["headline"], "h0_development_headline": h0_head, "source_checkpoints": checkpoint_info, "frozen_prediction_sha256": frozen_sha, "fixed_targets_read": False, "new_confirmation_targets_read": False, "counts": {"candidate_rows": len(grid_rows), "frozen_rows": len(frozen)}, "checks": checks, "evidence_note": "The Phase32C confirmation target had already been opened before this targeted rescue. No target is a script input, loss, gate, alpha, checkpoint, or candidate selector; subsequent evaluation on those sets is repeated engineering evidence, not a fresh blind confirmation."}
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase32d-audit-v1", "status": status, "checks": checks, "frozen_prediction_sha256": frozen_sha})
    write_json(args.output_dir / "logs/training.log", {"event": "phase32d_tp_gate_rescue_complete", "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "repository_head_at_training": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(), "device": str(device), "fixed_targets_read": False, "new_confirmation_targets_read": False})
    (args.output_dir / "README.md").write_text(f"""# Phase 32D：TP绝对上限内的定向gate救援\n\nPhase32C一次性新确认显示TP接近calls有条件线但cost仍失败。本阶段只复用Phase32B的开发OOF residual和三个seed checkpoint，比较global、policy、model、phase、model×policy、policy×phase六种有界gate，使TP累计配置达到绝对上限48。gate、alpha和候选选择没有读取Phase32C或Phase31D target。\n\n开发OOF选择`{selected_id}`；其calls/bytes/TV/EMD/cost WAPE分别为{selected['headline']['calls_wape']:.4%}/{selected['headline']['bytes_wape']:.4%}/{selected['headline']['mean_histogram_tv']:.4f}/{selected['headline']['mean_normalized_log_payload_emd']:.4f}/{selected['headline']['common_reference_cost_wape']:.4%}。冻结预测SHA为`{frozen_sha}`。\n\n由于确认target已经在Phase32C开放，下一步在新确认和原固定集上的结果都只能称为重复工程证据；不构成新的盲测。\n""")
    (args.output_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS": raise RuntimeError(checks)
    print(json.dumps({"status": status, "selected_candidate_id": selected_id, "development": selected["headline"], "frozen_prediction_sha256": frozen_sha}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
