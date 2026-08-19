#!/usr/bin/env python3
"""Generate the PatternDemand figures from frozen experiment artifacts.

Inputs are copied verbatim from commit
ffb413ffec69fd2f87bc958ed73f618696457baa on branch
experiment/pattern-demand-v0.5.15-clean.  The script intentionally does not
train, refit, smooth, or cherry-pick examples by prediction error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont


COMMIT = "ffb413ffec69fd2f87bc958ed73f618696457baa"
BRANCH = "experiment/pattern-demand-v0.5.15-clean"
METHOD_H0 = "h0"
METHOD_DNN = "h0_plus_dnn_residual"
METHOD_LABELS = {METHOD_H0: "H0", METHOD_DNN: "H0 + DNN residual"}
BIN_EDGES = np.array(
    [
        4096.0,
        13777.246867516858,
        46340.95001184158,
        155871.75497763665,
        524288.0,
        1763487.5990421579,
        5931641.601515722,
        19951584.63713749,
        67108864.0,
        225726412.6773962,
        759250124.9940125,
        2553802833.553599,
        8589934592.0,
    ],
    dtype=float,
)
BIN_LABELS = [
    "4–14K",
    "14–45K",
    "45–152K",
    "152–512K",
    "512K–1.7M",
    "1.7–5.7M",
    "5.7–19M",
    "19–64M",
    "64–215M",
    "215–724M",
    "724M–2.4G",
    "2.4–8G",
]
MODEL_ORDER = [
    "deepseek-v2-lite",
    "llama-3.2-3b-instruct",
    "mixtral-8x7b-instruct-v0.1",
    "qwen2.5-14b-instruct",
    "qwen3-30b-a3b",
    "qwen3-8b",
]
MODEL_SHORT = {
    "deepseek-v2-lite": "DeepSeek-V2-Lite",
    "llama-3.2-3b-instruct": "Llama-3.2-3B",
    "mixtral-8x7b-instruct-v0.1": "Mixtral-8×7B",
    "qwen2.5-14b-instruct": "Qwen2.5-14B",
    "qwen3-30b-a3b": "Qwen3-30B-A3B",
    "qwen3-8b": "Qwen3-8B",
}

COLORS = {
    "blue": "#246BCE",
    "blue_dark": "#174A8B",
    "orange": "#D97706",
    "gray": "#A7AFB9",
    "gray_light": "#E7EBF0",
    "gray_dark": "#525B66",
    "green": "#238A5A",
    "red": "#C43D3D",
    "purple": "#7C5CC4",
}
TOPOLOGY_COLORS = {"L1": COLORS["green"], "L2": COLORS["orange"], "L3": COLORS["purple"]}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "PingFang SC",
                "Hiragino Sans GB",
                "Arial Unicode MS",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#AAB2BD",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E2E7ED",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "xtick.color": "#364152",
            "ytick.color": "#364152",
            "text.color": "#172033",
            "axes.labelcolor": "#172033",
            "axes.titlecolor": "#172033",
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: mpl.figure.Figure, stem: str, output_dir: Path) -> None:
    for ext in ("png", "svg"):
        path = output_dir / ext / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        if ext == "svg":
            # Matplotlib writes path-data lines with trailing spaces.  They are
            # valid SVG, but normalizing them keeps git diff --check clean.
            normalized = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(normalized + "\n", encoding="utf-8")
    plt.close(fig)


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    total = vector.sum()
    return vector / total if total > 0 else np.zeros_like(vector)


def parse_vector(value: str) -> np.ndarray:
    vector = np.asarray(json.loads(value), dtype=float)
    if vector.shape != (12,):
        raise ValueError(f"expected 12-bin vector, got {vector.shape}")
    return vector


def profile_shape_stats(calls: np.ndarray) -> tuple[float, float, float, int]:
    share = normalize(calls)
    idx = np.arange(12, dtype=float)
    center = float(np.sum(idx * share))
    width = float(np.sqrt(np.sum(((idx - center) ** 2) * share)))
    positive = share[share > 0]
    entropy = float(-np.sum(positive * np.log(positive)) / np.log(12)) if positive.size else 0.0
    return center, width, entropy, int(np.count_nonzero(calls))


def select_nearest_unique(
    profiles: list[dict], field: str, quantiles: list[float]
) -> list[dict]:
    values = np.array([p[field] for p in profiles], dtype=float)
    targets = np.quantile(values, quantiles)
    selected: list[dict] = []
    used: set[str] = set()
    for target in targets:
        candidates = sorted(
            profiles,
            key=lambda p: (abs(p[field] - target), p["profile_id"]),
        )
        chosen = next(p for p in candidates if p["profile_id"] not in used)
        selected.append(chosen)
        used.add(chosen["profile_id"])
    return selected


def load_tp_pp_profiles(data_dir: Path, parallelism: str) -> list[dict]:
    if parallelism == "tp":
        prediction_file = "phase34c_tp_predictions.csv.gz"
        policy = "balanced"
    elif parallelism == "pp":
        prediction_file = "phase34c_pp_predictions.csv.gz"
        policy = "mb4"
    else:
        raise ValueError(parallelism)

    predictions = pd.read_csv(data_dir / prediction_file)
    targets = pd.read_csv(data_dir / "phase34d_targets.csv.gz")
    pred_mask = (
        (predictions["model"] == "qwen3-8b")
        & (predictions["parallelism"] == parallelism)
        & (predictions["parallel_size"] == 4)
        & (predictions["policy"] == policy)
        & (predictions["prediction_set"] == "phase34_blind_new")
    )
    target_mask = (
        (targets["model"] == "qwen3-8b")
        & (targets["parallelism"] == parallelism)
        & (targets["parallel_size"] == 4)
        & (targets["policy"] == policy)
    )
    predictions = predictions.loc[pred_mask].copy()
    targets = targets.loc[target_mask].copy()
    if len(predictions) != 48 or len(targets) != 24:
        raise AssertionError(
            f"unexpected {parallelism} example count: predictions={len(predictions)}, targets={len(targets)}"
        )

    profiles: list[dict] = []
    for profile_id, target_group in targets.groupby("profile_id", sort=True):
        if len(target_group) != 2 or set(target_group["phase"]) != {"prefill", "decode"}:
            raise AssertionError(f"invalid phase coverage for {profile_id}")
        target_calls = sum(
            (parse_vector(x) for x in target_group["target_calls_by_12bin_json"]),
            start=np.zeros(12),
        )
        target_bytes = sum(
            (parse_vector(x) for x in target_group["target_logical_bytes_by_12bin_json"]),
            start=np.zeros(12),
        )
        record = {
            "profile_id": profile_id,
            "segment": target_group["segment"].iloc[0],
            "target_calls": target_calls,
            "target_bytes": target_bytes,
            "configuration": f"{parallelism.upper()}4 / {policy}",
        }
        pred_profile = predictions[predictions["profile_id"] == profile_id]
        for method in (METHOD_H0, METHOD_DNN):
            method_group = pred_profile[pred_profile["method"] == method]
            if len(method_group) != 2 or set(method_group["phase"]) != {"prefill", "decode"}:
                raise AssertionError(f"invalid prediction coverage for {profile_id}, {method}")
            record[f"{method}_calls"] = sum(
                (parse_vector(x) for x in method_group["predicted_calls_by_12bin_json"]),
                start=np.zeros(12),
            )
            record[f"{method}_bytes"] = sum(
                (parse_vector(x) for x in method_group["predicted_logical_bytes_by_12bin_json"]),
                start=np.zeros(12),
            )
        center, width, entropy, nonzero = profile_shape_stats(target_calls)
        record.update(center=center, width=width, entropy=entropy, nonzero_bins=nonzero)
        profiles.append(record)
    if len(profiles) != 12:
        raise AssertionError(f"expected 12 {parallelism} profiles, got {len(profiles)}")
    return profiles


def load_pd_profiles(data_dir: Path) -> list[dict]:
    predictions = pd.read_csv(data_dir / "phase49_predictions.csv.gz")
    targets = pd.read_csv(data_dir / "phase50_targets.csv.gz")
    predictions = predictions[predictions["model"] == "qwen3-8b"].copy()
    targets = targets[targets["model"] == "qwen3-8b"].copy()
    if len(predictions) != 600 or len(targets) != 300:
        raise AssertionError("unexpected PD profile count")

    profiles: list[dict] = []
    for _, target in targets.sort_values("profile_id").iterrows():
        profile_id = target["profile_id"]
        target_calls = target[[f"target_calls_bin_{i:02d}" for i in range(12)]].to_numpy(float)
        target_bytes = target[
            [f"target_logical_bytes_bin_{i:02d}" for i in range(12)]
        ].to_numpy(float)
        pred_profile = predictions[predictions["profile_id"] == profile_id]
        if len(pred_profile) != 2:
            raise AssertionError(f"invalid PD prediction coverage for {profile_id}")
        record = {
            "profile_id": profile_id,
            "segment": target["segment"],
            "target_calls": target_calls,
            "target_bytes": target_bytes,
            "configuration": "P1→D1",
        }
        for method in (METHOD_H0, METHOD_DNN):
            row = pred_profile[pred_profile["method"] == method]
            if len(row) != 1:
                raise AssertionError(f"invalid PD method coverage for {profile_id}, {method}")
            row = row.iloc[0]
            record[f"{method}_calls"] = row[
                [f"predicted_calls_bin_{i:02d}" for i in range(12)]
            ].to_numpy(float)
            record[f"{method}_bytes"] = row[
                [f"predicted_logical_bytes_bin_{i:02d}" for i in range(12)]
            ].to_numpy(float)
        center, width, entropy, nonzero = profile_shape_stats(target_calls)
        record.update(center=center, width=width, entropy=entropy, nonzero_bins=nonzero)
        profiles.append(record)
    return profiles


def overall_accuracy_table(data_dir: Path) -> pd.DataFrame:
    phase34 = pd.read_csv(data_dir / "phase34d_aggregate_metrics.csv")
    phase34 = phase34[
        (phase34["evidence_set"] == "phase34_blind_six_model")
        & (phase34["phase"] == "total")
        & (phase34["slice_type"] == "overall")
        & (phase34["slice_value"] == "all")
    ].copy()
    phase50 = pd.read_csv(data_dir / "phase50_aggregate_metrics.csv")
    records: list[dict] = []
    phase34_metrics = {
        "Total calls WAPE": "calls_wape",
        "Total bytes WAPE": "bytes_wape",
        "Histogram TV": "mean_histogram_tv",
        "Payload EMD": "mean_normalized_log_payload_emd",
    }
    for parallelism in ("tp", "pp"):
        subset = phase34[phase34["parallelism"] == parallelism].set_index("method")
        for label, column in phase34_metrics.items():
            h0 = float(subset.loc[METHOD_H0, column])
            dnn = float(subset.loc[METHOD_DNN, column])
            records.append(
                {
                    "parallelism": parallelism.upper(),
                    "metric": label,
                    "h0": h0,
                    "dnn": dnn,
                    "ratio": dnn / h0,
                    "evidence": "Phase34D six-model fresh-blind",
                }
            )
    phase50 = phase50.set_index("method")
    phase50_metrics = {
        "Total calls WAPE": "calls_total_wape",
        "Total bytes WAPE": "bytes_total_wape",
        "Histogram TV": "mean_calls_histogram_tv",
        "Payload EMD": "mean_normalized_log_payload_emd",
    }
    for label, column in phase50_metrics.items():
        h0 = float(phase50.loc[METHOD_H0, column])
        dnn = float(phase50.loc[METHOD_DNN, column])
        records.append(
            {
                "parallelism": "PD",
                "metric": label,
                "h0": h0,
                "dnn": dnn,
                "ratio": dnn / h0,
                "evidence": "Phase50 six-model fresh-blind",
            }
        )
    return pd.DataFrame(records)


def plot_accuracy_overview(table: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["Total calls WAPE", "Total bytes WAPE", "Histogram TV", "Payload EMD"]
    metric_labels = ["总 calls\nWAPE", "总 bytes\nWAPE", "直方图\nTV", "payload\nEMD"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.25), sharey=True)
    for ax, parallelism in zip(axes, ["TP", "PP", "PD"]):
        subset = table[table["parallelism"] == parallelism].set_index("metric").loc[metrics]
        values = subset["ratio"].to_numpy()
        bars = ax.bar(
            np.arange(4),
            values,
            color=[COLORS["blue"] if value <= 1 else COLORS["red"] for value in values],
            width=0.66,
        )
        ax.axhline(1.0, color=COLORS["gray_dark"], linestyle="--", linewidth=1.1)
        ax.set_xticks(np.arange(4), metric_labels)
        ax.set_ylim(0, 1.12)
        ax.set_title(parallelism, fontweight="bold")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.2f}×",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        if parallelism in {"TP", "PP"}:
            ax.annotate(
                "结构锚点使 bytes 总量误差≈0",
                xy=(1, values[1]),
                xytext=(1.4, 0.2),
                arrowprops={"arrowstyle": "->", "color": COLORS["gray_dark"], "lw": 0.8},
                ha="center",
                fontsize=8.5,
                color=COLORS["gray_dark"],
            )
    axes[0].set_ylabel("H0 + DNN residual / H0（越低越好）")
    fig.suptitle("TP / PP / PD fresh-blind 预测精度：DNN residual 相对 H0 的误差比例", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.915,
        "TP/PP：Phase34D；PD：Phase50。虚线 1.0 表示与 H0 持平。",
        ha="center",
        color=COLORS["gray_dark"],
        fontsize=9.5,
    )
    fig.subplots_adjust(top=0.77, bottom=0.18, wspace=0.16)
    save_figure(fig, "01_blind_accuracy_overview", output_dir)


def model_composite_table(data_dir: Path) -> pd.DataFrame:
    phase34 = pd.read_csv(data_dir / "phase34d_aggregate_metrics.csv")
    phase34 = phase34[
        (phase34["evidence_set"] == "phase34_blind_six_model")
        & (phase34["phase"] == "total")
        & (phase34["slice_type"] == "model")
    ].copy()
    metrics = ["calls_wape", "bytes_wape", "mean_histogram_tv", "mean_normalized_log_payload_emd"]
    records: list[dict] = []
    for parallelism in ("tp", "pp"):
        for model in MODEL_ORDER:
            subset = phase34[
                (phase34["parallelism"] == parallelism) & (phase34["slice_value"] == model)
            ].set_index("method")
            ratios = [float(subset.loc[METHOD_DNN, m] / subset.loc[METHOD_H0, m]) for m in metrics]
            records.append(
                {
                    "model": model,
                    "parallelism": parallelism.upper(),
                    "ratio": float(np.mean(ratios)),
                    "definition": "mean(total-calls WAPE, total-bytes WAPE, TV, EMD ratios)",
                }
            )
    with open(data_dir / "phase50_model_metrics.json", encoding="utf-8") as handle:
        pd_metrics = json.load(handle)
    for model in MODEL_ORDER:
        records.append(
            {
                "model": model,
                "parallelism": "PD",
                "ratio": float(pd_metrics[model]["composite_ratio"]),
                "definition": "Phase50 official mean(histogram calls/bytes WAPE, TV, EMD ratios)",
            }
        )
    return pd.DataFrame(records)


def plot_model_heatmap(table: pd.DataFrame, output_dir: Path) -> None:
    matrix = (
        table.pivot(index="model", columns="parallelism", values="ratio")
        .loc[MODEL_ORDER, ["TP", "PP", "PD"]]
        .to_numpy()
    )
    fig, ax = plt.subplots(figsize=(7.7, 5.3))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "improvement", ["#123B6D", "#2F7FD3", "#EEF3F8", "#C43D3D"]
    )
    image = ax.imshow(matrix, vmin=0.0, vmax=1.08, cmap=cmap, aspect="auto")
    ax.set_xticks(range(3), ["TP", "PP", "PD"])
    ax.set_yticks(range(6), [MODEL_SHORT[m] for m in MODEL_ORDER])
    ax.tick_params(length=0)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if value < 0.63 else "#172033"
            ax.text(col, row, f"{value:.2f}×", ha="center", va="center", color=color, fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("四指标 composite ratio（越低越好）")
    ax.set_title("六模型 robustness：每个模型上的 H0 + DNN / H0", fontsize=14, fontweight="bold", pad=32)
    ax.text(
        0.5,
        1.035,
        "TP/PP 为 Phase34D 派生 composite；PD 为 Phase50 官方 composite",
        transform=ax.transAxes,
        ha="center",
        color=COLORS["gray_dark"],
        fontsize=9,
    )
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.grid(which="major", visible=False)
    fig.subplots_adjust(left=0.28, right=0.88, top=0.82, bottom=0.08)
    save_figure(fig, "02_six_model_robustness_heatmap", output_dir)


def ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(list(values), dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def plot_error_ecdf(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    phase34 = pd.read_csv(data_dir / "phase34d_per_case_metrics.csv.gz")
    phase34 = phase34[
        (phase34["evidence_set"] == "phase34_blind_six_model") & (phase34["phase"] == "total")
    ].copy()
    phase50 = pd.read_csv(data_dir / "phase50_per_unit_metrics.csv.gz")
    records: list[dict] = []
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1), sharex=True, sharey=True)
    for ax, parallelism in zip(axes, ["TP", "PP", "PD"]):
        for method, color, linestyle in [
            (METHOD_H0, COLORS["orange"], "--"),
            (METHOD_DNN, COLORS["blue"], "-"),
        ]:
            if parallelism == "PD":
                values = phase50.loc[phase50["method"] == method, "mean_calls_histogram_tv"].to_numpy()
            else:
                values = phase34.loc[
                    (phase34["parallelism"] == parallelism.lower()) & (phase34["method"] == method),
                    "histogram_tv",
                ].to_numpy()
            x, y = ecdf(values)
            ax.step(x, y, where="post", color=color, linestyle=linestyle, linewidth=2, label=METHOD_LABELS[method])
            records.append(
                {
                    "parallelism": parallelism,
                    "method": method,
                    "cases": len(values),
                    "median_tv": float(np.median(values)),
                    "p90_tv": float(np.quantile(values, 0.9)),
                    "p95_tv": float(np.quantile(values, 0.95)),
                }
            )
        ax.set_title(parallelism, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("单画像 calls histogram TV")
    axes[0].set_ylabel("累计画像比例")
    axes[2].legend(loc="lower right")
    fig.suptitle("误差分布：DNN residual 是否改善大多数画像", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.91,
        "曲线越靠左越好；TP/PP 每个配置画像为一个 case，PD 每个 model-profile 为一个 unit。",
        ha="center",
        color=COLORS["gray_dark"],
        fontsize=9.5,
    )
    fig.subplots_adjust(top=0.77, bottom=0.16, wspace=0.12)
    save_figure(fig, "03_error_distribution_ecdf", output_dir)
    return pd.DataFrame(records)


def format_total(value: float, kind: str) -> str:
    if kind == "calls":
        if value >= 1e6:
            return f"{value / 1e6:.2f}M calls/1k req"
        if value >= 1e3:
            return f"{value / 1e3:.1f}K calls/1k req"
        return f"{value:.0f} calls/1k req"
    gib = value / (1024**3)
    if gib >= 100:
        return f"{gib:.0f} GiB/1k req"
    if gib >= 10:
        return f"{gib:.1f} GiB/1k req"
    return f"{gib:.2f} GiB/1k req"


def plot_histogram_group(
    parallelism: str,
    selected: list[dict],
    group_kind: str,
    labels: list[str],
    stem: str,
    output_dir: Path,
) -> None:
    target_presence = np.zeros(12, dtype=bool)
    for profile in selected:
        target_presence |= (profile["target_calls"] > 0) | (profile["target_bytes"] > 0)
    shown = np.flatnonzero(target_presence)
    lo, hi = int(shown.min()), int(shown.max())
    bins = np.arange(lo, hi + 1)
    x = np.arange(len(bins), dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.0), sharex=True, sharey="row")
    for col, (profile, label) in enumerate(zip(selected, labels)):
        for row, kind in enumerate(["calls", "bytes"]):
            ax = axes[row, col]
            target = normalize(profile[f"target_{kind}"])[bins] * 100
            h0 = normalize(profile[f"{METHOD_H0}_{kind}"])[bins] * 100
            dnn = normalize(profile[f"{METHOD_DNN}_{kind}"])[bins] * 100
            ax.bar(x, target, width=0.78, color=COLORS["gray_light"], edgecolor=COLORS["gray"], linewidth=0.7)
            ax.plot(x, h0, color=COLORS["orange"], linestyle="--", marker="o", markersize=4, linewidth=1.6)
            ax.plot(x, dnn, color=COLORS["blue"], linestyle="-", marker="o", markersize=4, linewidth=1.8)
            ax.set_ylim(bottom=0)
            if col == 0:
                ax.set_ylabel("calls 占比（%）" if kind == "calls" else "logical bytes 占比（%）")
            if row == 0:
                detail = f"中心={profile['center']:.2f}" if group_kind == "center" else f"宽度={profile['width']:.2f} bins"
                ax.set_title(f"{label}\n{profile['segment']} · {detail}", fontweight="bold")
            total = float(profile[f"target_{kind}"].sum())
            ax.text(
                0.98,
                0.94,
                "Hfull: " + format_total(total, kind),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.2,
                color=COLORS["gray_dark"],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
            )
            if row == 1:
                ax.set_xticks(x, [BIN_LABELS[i] for i in bins], rotation=42, ha="right")
                ax.set_xlabel("payload bytes / message")

    # Shares are percentages.  A fixed range prevents a later shared-axis panel
    # from clipping a 100% peak after an earlier panel has established limits.
    for ax in axes.flat:
        ax.set_ylim(0, 105)

    title_kind = "消息尺度位置（小 / 中 / 大）" if group_kind == "center" else "消息分布宽度（分散 / 中等 / 集中）"
    config = selected[0]["configuration"]
    phase_note = "total = prefill + decode" if parallelism in {"TP", "PP"} else "pure-PD profile total"
    fig.suptitle(
        f"{parallelism} 12-bin 直方图样例：{title_kind}",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        f"Qwen3-8B · {config} · {phase_note}；样例仅按 Hfull 分布选择，不读取预测误差。",
        ha="center",
        color=COLORS["gray_dark"],
        fontsize=9.4,
    )
    handles = [
        Patch(facecolor=COLORS["gray_light"], edgecolor=COLORS["gray"], label="Hfull teacher"),
        Line2D([0], [0], color=COLORS["orange"], linestyle="--", marker="o", label="H0"),
        Line2D([0], [0], color=COLORS["blue"], marker="o", label="H0 + DNN residual"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=3)
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.07, right=0.99, hspace=0.26, wspace=0.12)
    save_figure(fig, stem, output_dir)


def select_and_plot_histograms(
    data_dir: Path, output_dir: Path
) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    audit_records: list[dict] = []
    selection_map: dict[str, list[dict]] = {}
    specs = [
        ("TP", load_tp_pp_profiles(data_dir, "tp"), "04_tp_message_scale", "05_tp_distribution_width"),
        ("PP", load_tp_pp_profiles(data_dir, "pp"), "06_pp_message_scale", "07_pp_distribution_width"),
        ("PD", load_pd_profiles(data_dir), "08_pd_message_scale", "09_pd_distribution_width"),
    ]
    for parallelism, profiles, center_stem, width_stem in specs:
        # The scale examples should show position changes without degenerating
        # into a single-bin spike.  The floor is fixed per communication mode
        # from Hfull only; PP naturally has a more concentrated calls shape.
        entropy_floor = {"TP": 0.25, "PP": 0.16, "PD": 0.25}[parallelism]
        center_candidates = [p for p in profiles if p["entropy"] >= entropy_floor]
        center_selected = select_nearest_unique(center_candidates, "center", [0.00, 0.50, 1.00])
        width_selected = select_nearest_unique(profiles, "width", [0.90, 0.50, 0.10])
        plot_histogram_group(
            parallelism,
            center_selected,
            "center",
            ["偏小消息画像", "中等消息画像", "偏大消息画像"],
            center_stem,
            output_dir,
        )
        plot_histogram_group(
            parallelism,
            width_selected,
            "width",
            ["非常分散", "较为分散", "较为集中"],
            width_stem,
            output_dir,
        )
        selection_map[f"{parallelism}_center"] = center_selected
        selection_map[f"{parallelism}_width"] = width_selected
        for group, chosen, q_values, labels in [
            ("message_scale", center_selected, [0.00, 0.50, 1.00], ["small", "medium", "large"]),
            ("distribution_width", width_selected, [0.90, 0.50, 0.10], ["dispersed", "moderate", "concentrated"]),
        ]:
            for position, (profile, quantile, label) in enumerate(zip(chosen, q_values, labels), start=1):
                audit_records.append(
                    {
                        "parallelism": parallelism,
                        "group": group,
                        "position": position,
                        "label": label,
                        "selection_field": "center" if group == "message_scale" else "width",
                        "eligibility_rule": (
                            f"Hfull normalized entropy >= {entropy_floor:.2f}"
                            if group == "message_scale"
                            else "all Hfull profiles"
                        ),
                        "target_quantile": quantile,
                        "profile_id": profile["profile_id"],
                        "segment": profile["segment"],
                        "configuration": profile["configuration"],
                        "hfull_center_bin": profile["center"],
                        "hfull_width_bins": profile["width"],
                        "hfull_normalized_entropy": profile["entropy"],
                        "hfull_nonzero_call_bins": profile["nonzero_bins"],
                        "hfull_total_calls_per_1000": float(profile["target_calls"].sum()),
                        "hfull_total_logical_bytes_per_1000": float(profile["target_bytes"].sum()),
                        "selection_uses_prediction_error": False,
                    }
                )
    return pd.DataFrame(audit_records), selection_map


def quality_flags_phase39(knot: dict) -> tuple[bool, bool]:
    spread = float(knot.get("cross_replica_relative_spread", 0.0)) > 0.25
    repeat = any(float(replica.get("repeat_median_cv", 0.0)) > 0.15 for replica in knot.get("replicas", []))
    return spread, repeat


def quality_flags_phase51(knot: dict) -> tuple[bool, bool]:
    spread = float(knot.get("cross_replica_relative_spread", 0.0)) > 0.25
    repeat = False
    for replica in knot.get("replicas", []):
        for direction in replica.get("directions", {}).values():
            repeat |= float(direction.get("repeat_median_cv", 0.0)) > 0.15
    return spread, repeat


def plot_curve(ax: mpl.axes.Axes, curve: dict, topology: str, phase: str) -> tuple[int, int]:
    knots = curve["knots"]
    x = np.array([k["payload_bytes"] for k in knots], dtype=float)
    y = np.array([k["official_latency_us"] for k in knots], dtype=float)
    if phase == "phase39":
        lower = np.array([k["lower_latency_us"] for k in knots], dtype=float)
        upper = np.array([k["upper_latency_us"] for k in knots], dtype=float)
        flagger = quality_flags_phase39
    else:
        lower = np.array(
            [min(r["slower_direction_latency_us"] for r in k["replicas"]) for k in knots], dtype=float
        )
        upper = np.array(
            [max(r["slower_direction_latency_us"] for r in k["replicas"]) for k in knots], dtype=float
        )
        flagger = quality_flags_phase51
    color = TOPOLOGY_COLORS[topology]
    ax.fill_between(x, lower, upper, color=color, alpha=0.10, linewidth=0)
    ax.plot(x, y, color=color, marker="o", markersize=2.7, linewidth=1.65, label=topology)
    spread_points: list[int] = []
    repeat_points: list[int] = []
    for i, knot in enumerate(knots):
        spread, repeat = flagger(knot)
        if spread:
            spread_points.append(i)
        if repeat:
            repeat_points.append(i)
    if spread_points:
        ax.scatter(x[spread_points], y[spread_points], marker="^", s=42, color=COLORS["red"], zorder=4)
    if repeat_points:
        ax.scatter(
            x[repeat_points],
            y[repeat_points],
            marker="o",
            s=46,
            facecolors="none",
            edgecolors=COLORS["red"],
            linewidths=1.2,
            zorder=4,
        )
    return len(spread_points), len(repeat_points)


def plot_tp_pp_curves(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    with open(data_dir / "phase39_physical_curves.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    curves = payload["curves"]
    panels = [("TP2", "tp", 2), ("TP4", "tp", 4), ("TP8", "tp", 8), ("PP", "pp", 2)]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.1), sharex=True, sharey=True)
    audit: list[dict] = []
    for ax, (title, parallelism, group_size) in zip(axes.flat, panels):
        for topology in ("L1", "L2", "L3"):
            matches = [
                curve
                for curve in curves
                if curve["parallelism"] == parallelism
                and int(curve["group_size"]) == group_size
                and curve["topology_level"] == topology
            ]
            if len(matches) != 1:
                raise AssertionError(f"curve lookup failed: {title}, {topology}")
            curve = matches[0]
            spread, repeat = plot_curve(ax, curve, topology, "phase39")
            audit.append(
                {
                    "phase": "Phase39",
                    "curve_id": curve["curve_id"],
                    "knots": len(curve["knots"]),
                    "placement_variance_knots": spread,
                    "runtime_variance_knots": repeat,
                }
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title, fontweight="bold")
    axes[1, 0].set_xlabel("payload bytes")
    axes[1, 1].set_xlabel("payload bytes")
    axes[0, 0].set_ylabel("实测通信时间（μs）")
    axes[1, 0].set_ylabel("实测通信时间（μs）")
    handles = [Line2D([0], [0], color=TOPOLOGY_COLORS[x], marker="o", label=x) for x in ("L1", "L2", "L3")]
    handles += [
        Patch(facecolor=COLORS["gray"], alpha=0.18, label="replica min–max"),
        Line2D([0], [0], marker="^", linestyle="none", color=COLORS["red"], label="placement spread >25%"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=COLORS["red"], label="repeat CV >15%"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=6, fontsize=8.5)
    fig.suptitle("TP / PP：L1 / L2 / L3 实测物理通信曲线", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.925,
        "Phase39；原始 knot 直接连线，阴影为 replica 包络，不施加单调化或平滑。",
        ha="center",
        color=COLORS["gray_dark"],
        fontsize=9.5,
    )
    fig.subplots_adjust(top=0.80, bottom=0.09, hspace=0.20, wspace=0.10)
    save_figure(fig, "10_tp_pp_physical_curves", output_dir)
    return pd.DataFrame(audit)


def plot_pd_curves(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    with open(data_dir / "phase51_pd_physical_curves.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    curves = payload["curves"]
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.8), sharey=True)
    audit: list[dict] = []
    for ax, model in zip(axes.flat, MODEL_ORDER):
        for topology in ("L1", "L2", "L3"):
            matches = [
                curve
                for curve in curves
                if curve["model_id"] == model and curve["topology_level"] == topology
            ]
            if len(matches) != 1:
                raise AssertionError(f"PD curve lookup failed: {model}, {topology}")
            curve = matches[0]
            spread, repeat = plot_curve(ax, curve, topology, "phase51")
            audit.append(
                {
                    "phase": "Phase51",
                    "curve_id": curve["curve_id"],
                    "knots": len(curve["knots"]),
                    "placement_variance_knots": spread,
                    "runtime_variance_knots": repeat,
                }
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(MODEL_SHORT[model], fontweight="bold")
    for ax in axes[1, :]:
        ax.set_xlabel("payload bytes")
    axes[0, 0].set_ylabel("实测通信时间（μs）")
    axes[1, 0].set_ylabel("实测通信时间（μs）")
    handles = [Line2D([0], [0], color=TOPOLOGY_COLORS[x], marker="o", label=x) for x in ("L1", "L2", "L3")]
    handles += [
        Patch(facecolor=COLORS["gray"], alpha=0.18, label="replica min–max"),
        Line2D([0], [0], marker="^", linestyle="none", color=COLORS["red"], label="placement spread >25%"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=COLORS["red"], label="repeat CV >15%"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.888), ncol=6, fontsize=8.5)
    fig.suptitle("纯 PD：六模型 L1 / L2 / L3 实测物理通信曲线", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.925,
        "Phase51 · Mooncake/RDMA · P1→D1；原始 knot + replica 包络，异常方差点显式标注。",
        ha="center",
        color=COLORS["gray_dark"],
        fontsize=9.5,
    )
    fig.subplots_adjust(top=0.79, bottom=0.09, hspace=0.24, wspace=0.14)
    save_figure(fig, "11_pd_physical_curves", output_dir)
    return pd.DataFrame(audit)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_source_manifest(data_dir: Path, audit_dir: Path) -> None:
    records = []
    for path in sorted(data_dir.iterdir()):
        if path.is_file():
            records.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "branch": BRANCH,
                    "commit": COMMIT,
                }
            )
    pd.DataFrame(records).to_csv(audit_dir / "source_manifest.csv", index=False)


def write_readme(output_dir: Path, selection: pd.DataFrame) -> None:
    readme = f"""# PatternDemand 图集

本目录由 `generate_patterndemand_figures.py` 从冻结实验产物生成。数据对应分支
`{BRANCH}`、提交 `{COMMIT}`。

## 图表

1. `01_blind_accuracy_overview`：TP/PP/PD fresh-blind 四个对齐指标的 DNN/H0 误差比例。
2. `02_six_model_robustness_heatmap`：六模型 composite ratio；TP/PP 为 Phase34D 派生口径，PD 为 Phase50 官方口径。
3. `03_error_distribution_ecdf`：单画像 calls histogram TV 的 ECDF。
4. `04`–`09`：TP/PP/PD 各两组 12-bin 样例；每组包含三个画像，并同时展示 calls 与 logical bytes。
5. `10_tp_pp_physical_curves`：Phase39 TP2/TP4/TP8/PP 的 L1/L2/L3 曲线。
6. `11_pd_physical_curves`：Phase51 六模型纯 PD 的 L1/L2/L3 曲线。

每张图同时提供 PNG 和 SVG。`contact_sheet.png` 用于快速浏览。

## 样例选择规则

- 固定 Qwen3-8B；TP 固定 TP4/balanced，PP 固定 PP4/mb4，PD 固定 P1→D1。
- `消息尺度`组：先用固定的 Hfull calls 归一化熵下限排除单桶尖峰（TP/PD 0.25，PP 0.16），再按 12-bin 加权中心取最小/中位/最大画像。
- `分布宽度`组：按 Hfull calls 的加权 bin 标准差，取 90%/50%/10% 分位附近的唯一画像。
- 选择过程只读取 Hfull，不读取 H0 或 DNN 的预测误差。完整选择记录见 `audit/sample_selection.csv`。
- 横轴仅保留该组三个 Hfull 画像在 calls 或 logical bytes 中实际出现过的连续桶范围。

## 口径说明

- 图 01 使用可对齐的 total calls WAPE、total bytes WAPE、calls histogram TV 和 normalized log-payload EMD。
- 图 02 的 composite 仅表示“相对各自 H0 的改善比例”。TP/PP 与 PD 的官方评估阶段对 histogram WAPE 的可用字段不同，因此不应把不同列当成绝对误差横向比较；具体定义保存在 `audit/model_composite_metrics.csv`。
- 物理曲线不做平滑和单调化。实线为 official knot，阴影为 replica min–max；三角形表示跨 replica spread >25%，空心圆表示 repeat median CV >15%。
- Phase54–57 在该提交中只有 workflow，没有完成的 `experiment-results`，本图集不将它们当作已完成精度结果。

## 复现

```bash
python generate_patterndemand_figures.py --data-dir /path/to/copied/frozen/artifacts
```

需要 Python 3.11+、NumPy、pandas、Matplotlib 和 Pillow。脚本不会训练模型或修改原始数据。

## 验证

生成时执行了 schema、行数、profile/method/phase 覆盖、12-bin 长度、曲线唯一性及输出完整性检查。结果见 `audit/validation_report.md`。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def make_contact_sheet(output_dir: Path) -> None:
    pngs = sorted((output_dir / "png").glob("*.png"))
    thumbs: list[tuple[str, Image.Image]] = []
    thumb_w, thumb_h = 620, 360
    for path in pngs:
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h - 35), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        x = (thumb_w - image.width) // 2
        y = 28 + (thumb_h - 28 - image.height) // 2
        canvas.paste(image, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), path.stem, fill="#172033")
        thumbs.append((path.stem, canvas))
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "#DDE3EA")
    for i, (_, image) in enumerate(thumbs):
        sheet.paste(image, ((i % cols) * thumb_w, (i // cols) * thumb_h))
    sheet.save(output_dir / "contact_sheet.png", optimize=True)


def validate_outputs(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    pngs = sorted((output_dir / "png").glob("*.png"))
    svgs = sorted((output_dir / "svg").glob("*.svg"))
    expected = 11
    checks = {
        "PNG count is 11": len(pngs) == expected,
        "SVG count is 11": len(svgs) == expected,
        "all PNG files are nonempty": all(p.stat().st_size > 20_000 for p in pngs),
        "all SVG files are nonempty": all(p.stat().st_size > 5_000 for p in svgs),
        "accuracy table has 12 aligned cells": len(tables["accuracy"]) == 12,
        "model composite table has 18 cells": len(tables["composite"]) == 18,
        "sample selection has 18 rows": len(tables["selection"]) == 18,
        "no sample selected using prediction error": not tables["selection"][
            "selection_uses_prediction_error"
        ].any(),
        "Phase39 has 12 unique curves": tables["phase39_curves"]["curve_id"].nunique() == 12,
        "Phase51 has 18 unique curves": tables["phase51_curves"]["curve_id"].nunique() == 18,
    }
    failures = [name for name, passed in checks.items() if not passed]
    lines = [
        "# Validation report",
        "",
        f"- Branch: `{BRANCH}`",
        f"- Commit: `{COMMIT}`",
        f"- Result: **{'PASS' if not failures else 'FAIL'}**",
        "",
        "## Checks",
        "",
    ]
    lines.extend([f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in checks.items()])
    lines += [
        "",
        "## Confidence and caveats",
        "",
        "- High confidence that figures reproduce the copied frozen tables and curve JSON: all joins and coverage checks passed.",
        "- Histogram examples are illustrative, not best-case claims; they are selected from Hfull shape quantiles without prediction-error access.",
        "- The TP/PP composite in figure 02 is derived from Phase34D fields, while the PD composite is the Phase50 official value. Interpret ratios within each cell, not as a shared absolute metric.",
        "- Physical-curve replica envelopes show measured variability and should not be read as statistical confidence intervals.",
        "",
    ]
    (output_dir / "audit" / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    if failures:
        raise AssertionError("validation failed: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    default_data = Path(__file__).resolve().parents[2] / "work" / "patterndemand_figures" / "data"
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    for subdir in ("png", "svg", "audit"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    setup_style()

    accuracy = overall_accuracy_table(data_dir)
    composite = model_composite_table(data_dir)
    plot_accuracy_overview(accuracy, output_dir)
    plot_model_heatmap(composite, output_dir)
    ecdf_table = plot_error_ecdf(data_dir, output_dir)
    selection, _ = select_and_plot_histograms(data_dir, output_dir)
    phase39_curves = plot_tp_pp_curves(data_dir, output_dir)
    phase51_curves = plot_pd_curves(data_dir, output_dir)

    accuracy.to_csv(output_dir / "audit" / "overall_accuracy_metrics.csv", index=False)
    composite.to_csv(output_dir / "audit" / "model_composite_metrics.csv", index=False)
    ecdf_table.to_csv(output_dir / "audit" / "ecdf_summary.csv", index=False)
    selection.to_csv(output_dir / "audit" / "sample_selection.csv", index=False)
    pd.concat([phase39_curves, phase51_curves], ignore_index=True).to_csv(
        output_dir / "audit" / "physical_curve_quality_flags.csv", index=False
    )
    write_source_manifest(data_dir, output_dir / "audit")
    write_readme(output_dir, selection)
    make_contact_sheet(output_dir)
    validate_outputs(
        output_dir,
        {
            "accuracy": accuracy,
            "composite": composite,
            "selection": selection,
            "phase39_curves": phase39_curves,
            "phase51_curves": phase51_curves,
        },
    )
    print(f"Generated 11 figures in {output_dir}")


if __name__ == "__main__":
    main()
