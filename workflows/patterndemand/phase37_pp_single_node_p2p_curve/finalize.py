#!/usr/bin/env python3
"""聚合Phase37外置raw样本为可提交的紧凑连续曲线。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_json, refresh_manifest, repo_root, sha256, utc_now, validate_result_tree, write_json


def read_records(raw_dir: Path) -> list[dict]:
    records = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                records.append(json.loads(line))
    return records


def curve_id(gpu_name: str, category: str) -> str:
    safe = "_".join("".join(character.lower() if character.isalnum() else " " for character in gpu_name).split())
    return f"pp_single_node_{safe}_{category.lower()}_measured"


def make_figure(path: Path, rows: list[dict]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[row["topology_category"]].append(row)
    width, height, margin = 1000, 520, 75
    all_x = [math.log2(int(row["payload_bytes"])) for row in rows]
    all_y = [math.log10(max(float(row["median_latency_us"]), 1e-9)) for row in rows]
    xmin, xmax, ymin, ymax = min(all_x), max(all_x), min(all_y), max(all_y)
    colors = ["#2563eb", "#d97706", "#059669", "#7c3aed", "#dc2626", "#0891b2"]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="75" y="32" font-family="sans-serif" font-size="20">Phase37 PP单机P2P实测连续曲线</text>']
    for index, (category, values) in enumerate(sorted(groups.items())):
        values.sort(key=lambda row: int(row["payload_bytes"]))
        points = []
        for row in values:
            x = margin + (math.log2(int(row["payload_bytes"])) - xmin) / max(xmax - xmin, 1e-9) * (width - 2 * margin)
            y = height - margin - (math.log10(max(float(row["median_latency_us"]), 1e-9)) - ymin) / max(ymax - ymin, 1e-9) * (height - 2 * margin)
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[index % len(colors)]
        svg.extend([f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>', f'<text x="{width - 210}" y="{55 + index * 20}" font-family="sans-serif" font-size="13" fill="{color}">{category}</text>'])
    svg.extend(['<text x="440" y="505" font-family="sans-serif" font-size="13">payload bytes（log2）</text>', '<text x="18" y="270" font-family="sans-serif" font-size="13" transform="rotate(-90 18 270)">latency us（log10）</text>', '</svg>'])
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def finalize(output_dir: Path, raw_dir: Path) -> dict:
    contract = load_json(HERE / "experiment.json")
    state = load_json(output_dir / "audit/runtime_state.json")
    records = read_records(raw_dir)
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["topology_category"], record["direction"], int(record["payload_bytes"]))].append(record)
    expected_sizes = set(int(value) for value in contract["payload_bytes"])
    selected_categories = set(state["selected_pairs"])
    point_rows = []
    directional_rows = []
    curves = []
    high_variance = []
    complete = True
    gpu_names = {int(row["index"]): row["name"] for row in state["gpu_inventory"]}
    for category in sorted(selected_categories):
        knots = []
        pair = state["selected_pairs"][category]
        pair_models = [gpu_names.get(int(index), "unknown_gpu") for index in pair["gpus"]]
        gpu_name = pair_models[0] if len(set(pair_models)) == 1 else "_to_".join(pair_models)
        directions = [f"gpu{pair['gpus'][0]}_to_gpu{pair['gpus'][1]}", f"gpu{pair['gpus'][1]}_to_gpu{pair['gpus'][0]}"]
        for payload in sorted(expected_sizes):
            by_direction_repeat = {}
            for direction in directions:
                values = grouped.get((category, direction, payload), [])
                medians_by_repeat = {int(value["repeat_id"]): float(value["latency_us"]["median"]) for value in values}
                if len(medians_by_repeat) != len(values) or len(medians_by_repeat) < int(contract["minimum_independent_repeats"]):
                    complete = False
                by_direction_repeat[direction] = medians_by_repeat
                direction_medians = list(medians_by_repeat.values())
                directional_rows.append({
                    "topology_category": category,
                    "raw_link": pair["raw_link"],
                    "direction": direction,
                    "payload_bytes": payload,
                    "repeat_count": len(direction_medians),
                    "median_latency_us": statistics.median(direction_medians) if direction_medians else float("nan"),
                    "repeat_median_cv": statistics.pstdev(direction_medians) / statistics.fmean(direction_medians) if len(direction_medians) > 1 and statistics.fmean(direction_medians) else 0.0,
                })
            repeat_ids = sorted(set.intersection(*(set(values) for values in by_direction_repeat.values())))
            if any(set(values) != set(repeat_ids) for values in by_direction_repeat.values()):
                complete = False
            medians = [max(by_direction_repeat[direction][repeat_id] for direction in directions) for repeat_id in repeat_ids]
            median = statistics.median(medians) if medians else float("nan")
            cv = statistics.pstdev(medians) / statistics.fmean(medians) if len(medians) > 1 and statistics.fmean(medians) else 0.0
            if cv > float(contract["repeat_median_cv_threshold"]):
                high_variance.append({"topology_category": category, "payload_bytes": payload, "repeat_median_cv": cv})
            row = {
                "curve_id": curve_id(gpu_name, category),
                "topology_category": category,
                "raw_link": pair["raw_link"],
                "physical_gpu_pair": ",".join(str(value) for value in pair["gpus"]),
                "direction_aggregation": "per_repeat_max_of_two_directions_then_median_across_repeats",
                "payload_bytes": payload,
                "repeat_count": len(repeat_ids),
                "median_latency_us": median,
                "repeat_median_min_us": min(medians) if medians else float("nan"),
                "repeat_median_max_us": max(medians) if medians else float("nan"),
                "repeat_median_cv": cv,
                "algorithmic_bandwidth_GBps": payload / (median / 1e6) / 1e9 if median > 0 else 0.0,
            }
            point_rows.append(row)
            knots.append({key: row[key] for key in ("payload_bytes", "median_latency_us", "repeat_count", "repeat_median_cv", "algorithmic_bandwidth_GBps")})
        curves.append({
            "curve_id": curve_id(gpu_name, category),
            "topology_scope": "single_node",
            "topology_category": category,
            "raw_link": pair["raw_link"],
            "physical_gpu_pair": pair["gpus"],
            "gpu_models": pair_models,
            "backend": contract["backend"],
            "measurement_scope": contract["measurement_scope"],
            "direction_policy": contract["direction_policy"],
            "interpolation": contract["interpolation"],
            "knots": knots,
        })
    data_valid = all(record.get("data_validation_pass") for record in records)
    peer_access_recorded = all("cuda_peer_access" in record for record in records)
    expected_grid = set()
    for category in selected_categories:
        pair = state["selected_pairs"][category]
        directions = [f"gpu{pair['gpus'][0]}_to_gpu{pair['gpus'][1]}", f"gpu{pair['gpus'][1]}_to_gpu{pair['gpus'][0]}"]
        expected_grid.update((category, direction, payload) for direction in directions for payload in expected_sizes)
    exact_grid = set(grouped) == expected_grid
    checks = {
        "source_and_backend_contract_pass": all(state["source_semantics"].values()),
        "all_selected_topology_categories_measured": exact_grid,
        "minimum_repeats_complete": complete,
        "all_data_validation_pass": data_valid,
        "cuda_peer_access_status_recorded": peer_access_recorded,
        "raw_samples_external_to_git": state["raw_samples_external_to_git"],
        "no_training_or_model_weights": True,
    }
    if not all(checks.values()):
        status = "FAIL"
    elif high_variance and len(selected_categories) == 1:
        status = "PASS_WITH_RUNTIME_VARIANCE_AND_LIMITED_TOPOLOGY"
    elif high_variance:
        status = "PASS_WITH_RUNTIME_VARIANCE"
    elif len(selected_categories) == 1:
        status = "PASS_WITH_LIMITED_TOPOLOGY"
    else:
        status = "PASS"

    (output_dir / "analysis").mkdir(exist_ok=True)
    (output_dir / "curves").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    with (output_dir / "analysis/curve_points.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(point_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(point_rows)
    with (output_dir / "analysis/directional_curve_points.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(directional_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(directional_rows)
    write_json(output_dir / "curves/pp_single_node_p2p_knots.json", {
        "schema_version": "phase37-pp-single-node-p2p-curves-v1",
        "created_at_utc": utc_now(),
        "curve_evidence": "physical_measurement",
        "metadata_overhead_included": False,
        "direction_policy": contract["direction_policy"],
        "curves": curves,
    })
    raw_files = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        raw_files.append({"bundle_relative_name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size, "records": sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)})
    write_json(output_dir / "audit/RAW_ASSET_MANIFEST.json", {"schema_version": "phase37-external-raw-assets-v1", "raw_committed_to_git": False, "bundle_id": state["raw_bundle_id"], "files": raw_files})
    write_json(output_dir / "analysis/variance_audit.json", {"threshold": contract["repeat_median_cv_threshold"], "high_variance_points": high_variance})
    make_figure(output_dir / "figures/pp_single_node_p2p_curves.svg", point_rows)
    summary = {
        "schema_version": "phase37-pp-single-node-p2p-result-v1",
        "status": status,
        "completed_at_utc": utc_now(),
        "workflow_commit": state["workflow_commit"],
        "objective": contract["objective"],
        "measurement_scope": contract["measurement_scope"],
        "backend": contract["backend"],
        "topology_categories_measured": sorted(selected_categories),
        "selected_pairs": state["selected_pairs"],
        "payload_points_per_curve": len(expected_sizes),
        "directions_measured_per_pair": 2,
        "records": len(records),
        "repeat_count": state["repeat_count"],
        "high_variance_points": len(high_variance),
        "checks": checks,
        "claim_boundary": "本阶段只形成单机物理P2P tensor-only曲线；尚未代入冻结消息直方图，也不评价PatternDemand cost精度",
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(
        "# Phase37：PP单机P2P真实连续曲线\n\n"
        f"最终状态：`{status}`。本阶段在{len(selected_categories)}类实际单机GPU拓扑上测量了"
        f"`payload → latency`曲线，每条曲线{len(expected_sizes)}个payload点，至少{contract['minimum_independent_repeats']}次独立重复。\n\n"
        "正式曲线测量的是SGLang `send_tensor_dict(async_send=True)`中GPU tensor对应的NCCL `isend/irecv`原语。"
        "每个GPU对双向分别实测，正式拓扑类别曲线按每次repeat的双向较慢值聚合，不能挑选较快方向。"
        "它与消息直方图的sender-only logical message口径一致，不包含CPU metadata、tensor分配、scheduler和通信计算重叠。\n\n"
        "逐次raw样本保存在Git仓库外；Git只保存紧凑曲线、repeat方差、环境、拓扑、日志、外置raw SHA清单和manifest。"
        "曲线保留真实非单调点，不做平滑。Phase38才会将Phase34冻结直方图确定性代入这些曲线。\n",
        encoding="utf-8",
    )
    tree = validate_result_tree(output_dir)
    if not tree["ok"]:
        raise RuntimeError(f"结果树含禁止资产：{tree['violations']}")
    (output_dir / "DONE").write_text(status + "\n", encoding="utf-8")
    refresh_manifest(output_dir)
    if status == "FAIL":
        raise RuntimeError(f"Phase37失败：{checks}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "experiment-results/phase37_pp_single_node_p2p_curve")
    args = parser.parse_args()
    print(json.dumps(finalize(args.output_dir.resolve(), args.raw_dir.expanduser().resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
