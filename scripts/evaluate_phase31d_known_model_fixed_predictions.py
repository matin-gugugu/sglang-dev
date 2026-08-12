#!/usr/bin/env python3
"""Generate fixed Hfull targets after prediction freeze and evaluate TP/PP closure."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from build_phase25_full_window_teacher import PP_BIN_EDGES, TP_BIN_EDGES, normalize, tp_histograms
from build_phase31b_known_model_hfull_dataset import (
    MICROBATCHES,
    MODELS,
    PHASES,
    PP_SIZES,
    STRATEGIES,
    TP_SIZES,
    all_model_features,
    histogram_fields,
    pp_histograms,
    summarize_profile,
)
from prepare_phase15_trace_windows import BURST_FILES, MOONCAKE_FILES, load_segment
from train_phase27c_pp_scheduler_feature_predictors import case_record


FIXED_ROLE = "fixed_prediction"
HISTORY_SECONDS = 300
METHODS = ("h0", "h0_plus_dnn_residual")
OFFICIAL = {
    "tp": {
        "calls_wape": 0.10,
        "bytes_wape": 0.02,
        "mean_histogram_tv": 0.20,
        "mean_normalized_log_payload_emd": 0.025,
        "common_reference_cost_wape": 0.05,
    },
    "pp": {
        "calls_wape": 0.15,
        "bytes_wape": 0.03,
        "mean_histogram_tv": 0.22,
        "mean_normalized_log_payload_emd": 0.04,
        "common_reference_cost_wape": 0.05,
    },
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=root / "experiment-results/phase31a_known_model_convergence_contract/selection/selected_windows.csv",
    )
    parser.add_argument(
        "--phase31b-dir",
        type=Path,
        default=root / "experiment-results/phase31b_known_model_hfull_dataset",
    )
    parser.add_argument(
        "--phase31c-dir",
        type=Path,
        default=root / "experiment-results/phase31c_known_model_residual_training",
    )
    parser.add_argument(
        "--model-features",
        type=Path,
        default=root / "experiment-results/phase16_model_features/model_features.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiment-results/phase31d_known_model_fixed_evaluation",
    )
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


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as source:
        return list(csv.DictReader(source))


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


def load_fixed_requests(args: argparse.Namespace) -> tuple[list[dict], dict[str, list[tuple[int, int]]], dict]:
    selection = [row for row in read_csv(args.selection) if row["role"] == FIXED_ROLE]
    if len(selection) != 10:
        raise ValueError(f"expected 10 fixed profiles, got {len(selection)}")
    raw_manifest = json.loads((args.raw_dir / "source_manifest.json").read_text())
    raw_checks = {}
    for row in raw_manifest["sources"]:
        path = args.raw_dir / row["name"]
        raw_checks[row["name"]] = path.stat().st_size == int(row["actual_size"]) and sha256(path) == row["sha256"]
    if len(raw_checks) != 6 or not all(raw_checks.values()):
        raise RuntimeError({"raw_source_checks": raw_checks})
    file_by_segment = {
        segment: args.raw_dir / name
        for name, (segment, _) in {**BURST_FILES, **MOONCAKE_FILES}.items()
    }
    arrays = {segment: load_segment(path) for segment, path in file_by_segment.items()}
    profiles = []
    request_windows = {}
    for selected in selection:
        timestamps, inputs, outputs = arrays[selected["segment"]]
        cutoff = int(selected["cutoff_ms"])
        left = int(np.searchsorted(timestamps, cutoff - HISTORY_SECONDS * 1000, side="left"))
        right = int(np.searchsorted(timestamps, cutoff, side="left"))
        compatible = {**selected, "phase27_profile_id": selected["profile_id"], "phase27_role": FIXED_ROLE}
        profile, requests = summarize_profile(compatible, timestamps[left:right], inputs[left:right], outputs[left:right])
        if len(requests) != int(selected["history_count"]):
            raise RuntimeError(f"history count mismatch: {selected['profile_id']}")
        profile["split_role"] = profile.pop("phase27_role")
        profiles.append(profile)
        request_windows[profile["profile_id"]] = requests
    return profiles, request_windows, raw_checks


def generate_targets(
    profiles: list[dict],
    request_windows: dict[str, list[tuple[int, int]]],
    model_map: dict,
) -> list[dict]:
    targets = []
    for profile in profiles:
        requests = request_windows[profile["profile_id"]]
        for model_name in MODELS:
            model_meta, _ = model_map[model_name]
            for tp_size in TP_SIZES:
                for policy, strategy in STRATEGIES.items():
                    histograms = {
                        phase: normalize(histogram, len(requests))
                        for phase, histogram in tp_histograms(requests, strategy, model_meta).items()
                    }
                    for phase in PHASES:
                        fields = histogram_fields(histograms[phase], TP_BIN_EDGES)
                        targets.append(
                            {
                                "example_id": f"tp/{model_name}/p{tp_size}/{policy}/{profile['profile_id']}/{phase}",
                                "profile_id": profile["profile_id"],
                                "source": profile["source"],
                                "segment": profile["segment"],
                                "model": model_name,
                                "parallelism": "tp",
                                "parallel_size": tp_size,
                                "policy": policy,
                                "phase": phase,
                                **{f"target_{key}": value for key, value in fields.items() if not key.startswith("exact_")},
                                "teacher_kind": "tp_full_window_fixed_draining_structural_teacher",
                            }
                        )
            bytes_per_token = int(model_meta["payload_bytes_per_active_token_prior"])
            for pp_size in PP_SIZES:
                for microbatch in MICROBATCHES:
                    histograms, audit = pp_histograms(requests, pp_size, microbatch, bytes_per_token)
                    if not audit["all_requests_complete"]:
                        raise RuntimeError(f"PP completion failure: {profile['profile_id']}")
                    for phase in PHASES:
                        policy = f"mb{microbatch}"
                        fields = histogram_fields(histograms[phase], PP_BIN_EDGES)
                        targets.append(
                            {
                                "example_id": f"pp/{model_name}/p{pp_size}/{policy}/{profile['profile_id']}/{phase}",
                                "profile_id": profile["profile_id"],
                                "source": profile["source"],
                                "segment": profile["segment"],
                                "model": model_name,
                                "parallelism": "pp",
                                "parallel_size": pp_size,
                                "policy": policy,
                                "phase": phase,
                                **{f"target_{key}": value for key, value in fields.items() if not key.startswith("exact_")},
                                "teacher_kind": "pp_scheduler_faithful_full_window_teacher_adapted_by_model_hidden_size",
                            }
                        )
    return targets


def vector(row: dict[str, str], name: str) -> np.ndarray:
    return np.asarray(json.loads(row[name]), dtype=np.float64)


def evaluation_records(predictions: list[dict[str, str]], targets: list[dict]) -> list[dict]:
    target_by_id = {row["example_id"]: row for row in targets}
    if len(target_by_id) != len(targets):
        raise RuntimeError("duplicate target ids")
    records = []
    grouped = defaultdict(list)
    for prediction in predictions:
        target = target_by_id[prediction["example_id"]]
        actual_calls = vector(target, "target_calls_by_12bin_json")
        actual_bytes = vector(target, "target_logical_bytes_by_12bin_json")
        predicted_calls = vector(prediction, "predicted_calls_by_12bin_json")
        predicted_bytes = vector(prediction, "predicted_logical_bytes_by_12bin_json")
        edges = TP_BIN_EDGES.tolist() if prediction["parallelism"] == "tp" else PP_BIN_EDGES.tolist()
        record = case_record(prediction, prediction["method"], prediction["phase"], actual_calls, actual_bytes, predicted_calls, predicted_bytes, edges)
        record.update(
            {
                "example_id": prediction["example_id"],
                "source": prediction["source"],
                "model": prediction["model"],
                "parallelism": prediction["parallelism"],
            }
        )
        records.append(record)
        key = (
            prediction["profile_id"], prediction["model"], prediction["parallelism"],
            prediction["parallel_size"], prediction["policy"], prediction["method"],
        )
        grouped[key].append((prediction, actual_calls, actual_bytes, predicted_calls, predicted_bytes))
    for values in grouped.values():
        if len(values) != 2 or {value[0]["phase"] for value in values} != set(PHASES):
            raise RuntimeError("configuration does not contain exactly two phases")
        values.sort(key=lambda value: value[0]["phase"])
        representative = values[0][0]
        actual_calls = sum((value[1] for value in values))
        actual_bytes = sum((value[2] for value in values))
        predicted_calls = sum((value[3] for value in values))
        predicted_bytes = sum((value[4] for value in values))
        edges = TP_BIN_EDGES.tolist() if representative["parallelism"] == "tp" else PP_BIN_EDGES.tolist()
        total = case_record(representative, representative["method"], "total", actual_calls, actual_bytes, predicted_calls, predicted_bytes, edges)
        actual_phase_calls = np.concatenate([value[1] for value in values])
        predicted_phase_calls = np.concatenate([value[3] for value in values])
        total["histogram_l1"] = float(np.abs(actual_phase_calls / max(actual_phase_calls.sum(), 1e-12) - predicted_phase_calls / max(predicted_phase_calls.sum(), 1e-12)).sum())
        total["histogram_tv"] = total["histogram_l1"] / 2
        total.update(
            {
                "example_id": representative["example_id"].rsplit("/", 1)[0] + "/total",
                "source": representative["source"],
                "model": representative["model"],
                "parallelism": representative["parallelism"],
            }
        )
        records.append(total)
    return records


def metric_row(values: list[dict], *, parallelism: str, method: str, phase: str, slice_type: str, slice_value: str) -> dict:
    actual_calls = sum(float(row["actual_total_calls"]) for row in values)
    actual_bytes = sum(float(row["actual_total_logical_bytes"]) for row in values)
    actual_cost = sum(float(row["actual_common_reference_cost_us"]) for row in values)
    return {
        "parallelism": parallelism,
        "method": method,
        "phase": phase,
        "slice_type": slice_type,
        "slice_value": slice_value,
        "cases": len(values),
        "calls_mape": statistics.fmean(float(row["calls_ape"]) for row in values),
        "calls_wape": sum(float(row["calls_absolute_error"]) for row in values) / max(actual_calls, 1e-12),
        "bytes_mape": statistics.fmean(float(row["bytes_ape"]) for row in values),
        "bytes_wape": sum(float(row["bytes_absolute_error"]) for row in values) / max(actual_bytes, 1e-12),
        "mean_histogram_l1": statistics.fmean(float(row["histogram_l1"]) for row in values),
        "mean_histogram_tv": statistics.fmean(float(row["histogram_tv"]) for row in values),
        "mean_normalized_log_payload_emd": statistics.fmean(float(row["normalized_log_payload_emd"]) for row in values),
        "common_reference_cost_mape": statistics.fmean(float(row["cost_ape"]) for row in values),
        "common_reference_cost_wape": sum(float(row["cost_absolute_error"]) for row in values) / max(actual_cost, 1e-12),
    }


def aggregate(records: list[dict]) -> list[dict]:
    output = []
    for parallelism in ("tp", "pp"):
        for method in METHODS:
            for phase in (*PHASES, "total"):
                base = [row for row in records if row["parallelism"] == parallelism and row["method"] == method and row["phase"] == phase]
                output.append(metric_row(base, parallelism=parallelism, method=method, phase=phase, slice_type="overall", slice_value="all"))
                for field in ("model", "policy", "parallel_size", "source"):
                    for value in sorted({str(row[field]) for row in base}):
                        subset = [row for row in base if str(row[field]) == value]
                        output.append(metric_row(subset, parallelism=parallelism, method=method, phase=phase, slice_type=field, slice_value=value))
    return output


def find_metric(metrics: list[dict], parallelism: str, method: str, slice_type: str = "overall", slice_value: str = "all") -> dict:
    return next(
        row for row in metrics
        if row["parallelism"] == parallelism and row["method"] == method and row["phase"] == "total"
        and row["slice_type"] == slice_type and str(row["slice_value"]) == str(slice_value)
    )


def relative_improvement(dnn: dict, h0: dict, key: str) -> float:
    return 1.0 - float(dnn[key]) / max(float(h0[key]), 1e-12)


def closure(metrics: list[dict], parallelism: str) -> dict:
    h0 = find_metric(metrics, parallelism, "h0")
    dnn = find_metric(metrics, parallelism, "h0_plus_dnn_residual")
    models = {}
    for model in MODELS:
        model_h0 = find_metric(metrics, parallelism, "h0", "model", model)
        model_dnn = find_metric(metrics, parallelism, "h0_plus_dnn_residual", "model", model)
        models[model] = {
            "h0": model_h0,
            "h0_plus_dnn_residual": model_dnn,
            "calls_relative_improvement": relative_improvement(model_dnn, model_h0, "calls_wape"),
            "cost_relative_improvement": relative_improvement(model_dnn, model_h0, "common_reference_cost_wape"),
        }
    common_official = all(float(dnn[key]) <= threshold for key, threshold in OFFICIAL[parallelism].items())
    calls_gain = relative_improvement(dnn, h0, "calls_wape")
    cost_gain = relative_improvement(dnn, h0, "common_reference_cost_wape")
    if parallelism == "tp":
        per_model_official = all(
            value["h0_plus_dnn_residual"]["calls_wape"] <= 0.15
            and value["h0_plus_dnn_residual"]["bytes_wape"] <= 0.04
            and value["h0_plus_dnn_residual"]["common_reference_cost_wape"] <= 0.08
            for value in models.values()
        )
        official = common_official and calls_gain > 0 and cost_gain > 0 and per_model_official
        conditional = (
            dnn["calls_wape"] <= 0.12 and dnn["common_reference_cost_wape"] <= 0.06
            and calls_gain > 0 and cost_gain > 0
            and all(value["calls_relative_improvement"] > 0 and value["cost_relative_improvement"] > 0 for value in models.values())
            and all(value["h0_plus_dnn_residual"]["calls_wape"] <= 0.18 and value["h0_plus_dnn_residual"]["common_reference_cost_wape"] <= 0.10 for value in models.values())
        )
        extra = {"single_model_official_guard": per_model_official}
    else:
        per_model_official = all(
            value["h0_plus_dnn_residual"]["calls_wape"] <= 0.20
            and value["h0_plus_dnn_residual"]["common_reference_cost_wape"] <= 0.08
            for value in models.values()
        )
        mb16_h0 = find_metric(metrics, parallelism, "h0", "policy", "mb16")
        mb16_dnn = find_metric(metrics, parallelism, "h0_plus_dnn_residual", "policy", "mb16")
        mb16_calls_gain = relative_improvement(mb16_dnn, mb16_h0, "calls_mape")
        mb16_protection = not all(
            mb16_dnn[key] > 1.10 * mb16_h0[key]
            for key in ("bytes_wape", "mean_histogram_tv", "common_reference_cost_wape")
        )
        mb16_official = mb16_dnn["calls_mape"] <= 0.60 and mb16_calls_gain >= 0.50 and mb16_protection
        official = common_official and calls_gain >= 0.20 and cost_gain >= 0.20 and per_model_official and mb16_official
        conditional = (
            dnn["calls_wape"] <= 0.18 and dnn["common_reference_cost_wape"] <= 0.07
            and calls_gain > 0 and cost_gain > 0
            and all(value["calls_relative_improvement"] > 0 and value["cost_relative_improvement"] > 0 for value in models.values())
            and all(value["h0_plus_dnn_residual"]["calls_wape"] <= 0.25 and value["h0_plus_dnn_residual"]["common_reference_cost_wape"] <= 0.10 for value in models.values())
            and mb16_calls_gain >= 0.20 and mb16_protection
        )
        extra = {
            "single_model_official_guard": per_model_official,
            "mb16": {
                "h0": mb16_h0,
                "h0_plus_dnn_residual": mb16_dnn,
                "calls_mape_relative_improvement": mb16_calls_gain,
                "bytes_tv_cost_not_simultaneously_worse_by_over_10pct": mb16_protection,
                "official_guard": mb16_official,
            },
        }
    decision = "formal_pass" if official else "conditional_pass" if conditional else "fail"
    return {
        "decision": decision,
        "h0": h0,
        "h0_plus_dnn_residual": dnn,
        "calls_relative_improvement": calls_gain,
        "cost_relative_improvement": cost_gain,
        "models": models,
        "official_common_metrics": common_official,
        **extra,
    }


def main() -> None:
    args = parse_args()
    for name in ("labels", "analysis", "figures", "logs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    phase31b = json.loads((args.phase31b_dir / "summary.json").read_text())
    phase31c = json.loads((args.phase31c_dir / "summary.json").read_text())
    prediction_path = args.phase31c_dir / "analysis/frozen_fixed_prediction.csv.gz"
    if phase31b["status"] != "PASS" or phase31b["fixed_target_state"] != "not_generated":
        raise RuntimeError("Phase31B target isolation contract failed")
    if phase31c["status"] != "PASS" or phase31c["fixed_targets_read"] is not False:
        raise RuntimeError("Phase31C training contract failed")
    if sha256(prediction_path) != phase31c["frozen_prediction_sha256"]:
        raise RuntimeError("frozen prediction SHA mismatch")
    predictions = read_csv(prediction_path)
    if len(predictions) != 2160 or set(row["method"] for row in predictions) != set(METHODS):
        raise RuntimeError("unexpected frozen predictions")
    profiles, request_windows, raw_checks = load_fixed_requests(args)
    model_map = all_model_features(args.model_features)
    if set(model_map) != set(MODELS):
        raise RuntimeError("unexpected models")
    targets = generate_targets(profiles, request_windows, model_map)
    records = evaluation_records(predictions, targets)
    metrics = aggregate(records)
    decisions = {parallelism: closure(metrics, parallelism) for parallelism in ("tp", "pp")}

    write_csv_gz(args.output_dir / "labels/fixed_prediction_hfull_targets.csv.gz", targets)
    write_csv_gz(args.output_dir / "analysis/per_case_metrics.csv.gz", records)
    write_csv(args.output_dir / "analysis/aggregate_metrics.csv", metrics)

    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        keys = ("calls_wape", "bytes_wape", "mean_histogram_tv", "common_reference_cost_wape")
        labels = ("calls WAPE", "bytes WAPE", "TV", "cost WAPE")
        for column, parallelism in enumerate(("tp", "pp")):
            h0 = decisions[parallelism]["h0"]
            dnn = decisions[parallelism]["h0_plus_dnn_residual"]
            x = np.arange(len(keys))
            axes[0, column].bar(x - 0.18, [h0[key] for key in keys], 0.36, label="H0")
            axes[0, column].bar(x + 0.18, [dnn[key] for key in keys], 0.36, label="H0+DNN residual")
            axes[0, column].set_xticks(x, labels, rotation=25, ha="right")
            axes[0, column].set_title(f"{parallelism.upper()} overall")
            axes[0, column].legend()
            model_x = np.arange(len(MODELS))
            axes[1, column].bar(model_x - 0.18, [decisions[parallelism]["models"][model]["h0_plus_dnn_residual"]["calls_wape"] for model in MODELS], 0.36, label="calls")
            axes[1, column].bar(model_x + 0.18, [decisions[parallelism]["models"][model]["h0_plus_dnn_residual"]["common_reference_cost_wape"] for model in MODELS], 0.36, label="cost")
            axes[1, column].set_xticks(model_x, MODELS, rotation=20, ha="right")
            axes[1, column].set_title(f"{parallelism.upper()} per-model DNN WAPE")
            axes[1, column].legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / "figures/fixed_prediction_comparison.png", dpi=180)
        plt.close(figure)
    except Exception as error:
        write_json(args.output_dir / "figures/plot_failure.json", {"error": repr(error)})

    checks = {
        "phase31b_development_fixed_target_absent": phase31b["fixed_target_state"] == "not_generated",
        "phase31c_fixed_targets_not_read": phase31c["fixed_targets_read"] is False,
        "frozen_prediction_sha_matches": sha256(prediction_path) == phase31c["frozen_prediction_sha256"],
        "fixed_profiles_exact_10": len(profiles) == 10,
        "models_exact_three": set(model_map) == set(MODELS),
        "target_phase_rows_1080": len(targets) == 1080,
        "prediction_phase_rows_2160": len(predictions) == 2160,
        "evaluation_rows_3240": len(records) == 3240,
        "raw_sources_6_of_6": len(raw_checks) == 6 and all(raw_checks.values()),
        "all_metrics_finite": all(math.isfinite(float(row[key])) for row in metrics for key in ("calls_mape", "calls_wape", "bytes_mape", "bytes_wape", "mean_histogram_tv", "mean_normalized_log_payload_emd", "common_reference_cost_mape", "common_reference_cost_wape")),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase31d-known-model-fixed-evaluation-v1",
        "status": status,
        "models": list(MODELS),
        "fixed_profiles": len(profiles),
        "fixed_requests": sum(len(value) for value in request_windows.values()),
        "target_phase_rows": len(targets),
        "prediction_phase_rows": len(predictions),
        "evaluation_rows_including_totals": len(records),
        "decisions": decisions,
        "fixed_prediction_sha256": phase31c["frozen_prediction_sha256"],
        "inputs": {
            "selection_sha256": sha256(args.selection),
            "phase31b_summary_sha256": sha256(args.phase31b_dir / "summary.json"),
            "phase31c_summary_sha256": sha256(args.phase31c_dir / "summary.json"),
            "model_features_sha256": sha256(args.model_features),
        },
        "checks": checks,
        "scope": "known three-model in-distribution normal-history first-stage evaluation; not unseen-model or extreme-traffic generalization",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase31d-fixed-evaluation-audit-v1", "status": status, "checks": checks, "raw_source_checks": raw_checks})
    (args.output_dir / "README.md").write_text(f"""# Phase 31D：三模型固定预测集最终评测

本阶段在Phase31C预测文件和SHA冻结之后，才读取10个固定预测窗口的完整请求并生成Hfull target。训练、候选选择、alpha和checkpoint均未读取这些target，固定预测窗口也未因结果而更换。

## 范围与规模

- 模型：DeepSeek-V2-Lite、Qwen3-8B、Qwen3-30B-A3B；
- 固定画像：10个，与训练/验证窗口请求级不重叠；
- TP：TP2/4/8 × latency/balanced/throughput；
- PP：PP2/4/8 × MB1/4/16；
- Hfull target：{len(targets):,}条phase rows；冻结预测：{len(predictions):,}条phase rows（H0与H0+DNN residual）；
- 逐case评测：{len(records):,}条，包含prefill、decode和total。

## 裁定

- TP：`{decisions['tp']['decision']}`；
- PP：`{decisions['pp']['decision']}`。

完整整体、单模型、policy、并行规模和来源指标见`analysis/aggregate_metrics.csv`；核心裁定见`summary.json`。结论仅限当前三个已知模型与正常历史流量范围，不代表未见模型、极端流量或所有生产环境的零样本泛化。
""")
    write_json(args.output_dir / "logs/evaluation.log", {"event": "phase31d_fixed_evaluation_complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "status": status, "repository_head_at_evaluation": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "fixed_prediction_sha256": phase31c["frozen_prediction_sha256"], "decisions": {key: value["decision"] for key, value in decisions.items()}})
    (args.output_dir / "DONE").write_text(f"{status}\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "decisions": decisions}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
