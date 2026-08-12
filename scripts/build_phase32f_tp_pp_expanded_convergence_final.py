#!/usr/bin/env python3
"""Build the Phase32 final archive and Chinese handoff documents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


METRICS = (
    "calls_mape", "calls_wape", "bytes_mape", "bytes_wape",
    "mean_histogram_tv", "mean_normalized_log_payload_emd",
    "common_reference_cost_mape", "common_reference_cost_wape",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase32a-dir", type=Path, default=root / "experiment-results/phase32a_expanded_search_contract")
    parser.add_argument("--phase32b-dir", type=Path, default=root / "experiment-results/phase32b_expanded_residual_search")
    parser.add_argument("--phase32c-dir", type=Path, default=root / "experiment-results/phase32c_frozen_prediction_evaluation")
    parser.add_argument("--phase32d-dir", type=Path, default=root / "experiment-results/phase32d_tp_gate_rescue")
    parser.add_argument("--phase32e-dir", type=Path, default=root / "experiment-results/phase32e_tp_rescue_repeated_evaluation")
    parser.add_argument("--output-dir", type=Path, default=root / "experiment-results/phase32f_tp_pp_expanded_convergence_final")
    parser.add_argument("--deliverable-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def verify_manifest(directory: Path) -> bool:
    for line in (directory / "manifest.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if sha256(directory / relative) != expected:
            return False
    return True


def select_rows(rows: list[dict[str, str]], evidence: str, parallelism: str, slice_type: str) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row["evidence_set"] == evidence and row["parallelism"] == parallelism
        and row["phase"] == "total" and row["slice_type"] == slice_type
    ]


def compact_rows(rows: list[dict[str, str]], evidence_label: str) -> list[dict]:
    by_slice = {(row["slice_value"], row["method"]): row for row in rows}
    output = []
    for slice_value in sorted({row["slice_value"] for row in rows}):
        h0 = by_slice[(slice_value, "h0")]
        for method in ("h0", "h0_plus_dnn_residual"):
            row = by_slice[(slice_value, method)]
            item = {
                "evidence_set": evidence_label,
                "parallelism": row["parallelism"],
                "slice_type": row["slice_type"],
                "slice_value": slice_value,
                "method": method,
                "cases": row["cases"],
                **{key: row[key] for key in METRICS},
                "calls_wape_relative_improvement_vs_h0": "" if method == "h0" else 1 - float(row["calls_wape"]) / float(h0["calls_wape"]),
                "cost_wape_relative_improvement_vs_h0": "" if method == "h0" else 1 - float(row["common_reference_cost_wape"]) / float(h0["common_reference_cost_wape"]),
            }
            output.append(item)
    return output


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def metric_table(decision: dict) -> str:
    h0, dnn = decision["h0"], decision["h0_plus_dnn_residual"]
    return "\n".join((
        "| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | EMD | cost MAPE/WAPE |",
        "|---|---:|---:|---:|---:|---:|",
        f"| H0 | {pct(h0['calls_mape'])}/{pct(h0['calls_wape'])} | {pct(h0['bytes_mape'])}/{pct(h0['bytes_wape'])} | {h0['mean_histogram_tv']:.4f} | {h0['mean_normalized_log_payload_emd']:.4f} | {pct(h0['common_reference_cost_mape'])}/{pct(h0['common_reference_cost_wape'])} |",
        f"| H0+DNN residual | {pct(dnn['calls_mape'])}/{pct(dnn['calls_wape'])} | {pct(dnn['bytes_mape'])}/{pct(dnn['bytes_wape'])} | {dnn['mean_histogram_tv']:.4f} | {dnn['mean_normalized_log_payload_emd']:.4f} | {pct(dnn['common_reference_cost_mape'])}/{pct(dnn['common_reference_cost_wape'])} |",
    ))


def model_table(tp: dict, pp: dict) -> str:
    lines = ["| 方向 | 模型 | calls WAPE | bytes WAPE | TV | EMD | cost WAPE |", "|---|---|---:|---:|---:|---:|---:|"]
    for parallelism, value in (("TP", tp), ("PP", pp)):
        for model, block in sorted(value["models"].items()):
            row = block["h0_plus_dnn_residual"]
            lines.append(f"| {parallelism} | {model} | {pct(row['calls_wape'])} | {pct(row['bytes_wape'])} | {row['mean_histogram_tv']:.4f} | {row['mean_normalized_log_payload_emd']:.4f} | {pct(row['common_reference_cost_wape'])} |")
    return "\n".join(lines)


def svg(path: Path, tp: dict, pp: dict) -> None:
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="590" viewBox="0 0 1000 590">', '<rect width="100%" height="100%" fill="white"/>', '<text x="500" y="38" text-anchor="middle" font-family="sans-serif" font-size="24">Phase32 新确认：H0 与 H0+DNN residual</text>']
    keys = (("calls_wape", "calls WAPE"), ("bytes_wape", "bytes WAPE"), ("mean_histogram_tv", "TV"), ("mean_normalized_log_payload_emd", "EMD"), ("common_reference_cost_wape", "cost WAPE"))
    for panel, (name, decision) in enumerate((("TP（救援后重复工程）", tp), ("PP（一次性确认）", pp))):
        left = 45 + panel * 490
        pieces.append(f'<text x="{left+220}" y="78" text-anchor="middle" font-family="sans-serif" font-size="18">{name} / {decision["decision"]}</text>')
        for index, (key, label) in enumerate(keys):
            y = 112 + index * 86
            pieces.append(f'<text x="{left}" y="{y-7}" font-family="sans-serif" font-size="13">{label}</text>')
            for offset, (method, color) in enumerate((("h0", "#94a3b8"), ("h0_plus_dnn_residual", "#2563eb"))):
                value = float(decision[method][key]); width = min(value / 0.25 * 320, 320)
                yy = y + offset * 27
                pieces.append(f'<rect x="{left+88}" y="{yy}" width="{width:.2f}" height="20" fill="{color}" rx="3"/>')
                pieces.append(f'<text x="{left+94+width:.2f}" y="{yy+15}" font-family="sans-serif" font-size="12">{100*value:.2f}%</text>')
    pieces.extend(['<rect x="375" y="555" width="18" height="12" fill="#94a3b8"/><text x="400" y="566" font-family="sans-serif" font-size="13">H0</text>', '<rect x="505" y="555" width="18" height="12" fill="#2563eb"/><text x="530" y="566" font-family="sans-serif" font-size="13">H0+DNN residual</text>', '</svg>'])
    path.write_text("\n".join(pieces) + "\n")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    for name in ("analysis", "checkpoints", "predictions", "figures", "logs", "docs"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    args.deliverable_dir.mkdir(parents=True, exist_ok=True)
    dirs = {"phase32a": args.phase32a_dir, "phase32b": args.phase32b_dir, "phase32c": args.phase32c_dir, "phase32d": args.phase32d_dir, "phase32e": args.phase32e_dir}
    summaries = {name: read_json(path / "summary.json") for name, path in dirs.items()}

    tp = summaries["phase32e"]["decisions"]["new_confirmation_repeated"]
    pp = summaries["phase32c"]["decisions"]["new_confirmation"]["pp"]
    rows_e = read_csv(args.phase32e_dir / "analysis/aggregate_metrics.csv")
    rows_c = read_csv(args.phase32c_dir / "analysis/aggregate_metrics.csv")
    tp_overall = compact_rows(select_rows(rows_e, "new_confirmation_repeated", "tp", "overall"), "new_confirmation_repeated_after_rescue")
    pp_overall = compact_rows(select_rows(rows_c, "new_confirmation", "pp", "overall"), "new_confirmation_once")
    write_csv(args.output_dir / "analysis/headline_metrics.csv", tp_overall + pp_overall)
    per_model = compact_rows(select_rows(rows_e, "new_confirmation_repeated", "tp", "model"), "new_confirmation_repeated_after_rescue") + compact_rows(select_rows(rows_c, "new_confirmation", "pp", "model"), "new_confirmation_once")
    per_policy = compact_rows(select_rows(rows_e, "new_confirmation_repeated", "tp", "policy"), "new_confirmation_repeated_after_rescue") + compact_rows(select_rows(rows_c, "new_confirmation", "pp", "policy"), "new_confirmation_once")
    per_size = compact_rows(select_rows(rows_e, "new_confirmation_repeated", "tp", "parallel_size"), "new_confirmation_repeated_after_rescue") + compact_rows(select_rows(rows_c, "new_confirmation", "pp", "parallel_size"), "new_confirmation_once")
    write_csv(args.output_dir / "analysis/per_model_metrics.csv", per_model)
    write_csv(args.output_dir / "analysis/per_policy_metrics.csv", per_policy)
    write_csv(args.output_dir / "analysis/per_parallel_size_metrics.csv", per_size)

    search_rows = [
        {"parallelism": "tp", "stage": "phase31_prior", "candidate_id": "aggregate_prior_candidates", "candidate_count": 18},
        {"parallelism": "pp", "stage": "phase31_prior", "candidate_id": "aggregate_prior_candidates", "candidate_count": 12},
    ]
    for row in read_csv(args.phase32b_dir / "analysis/candidate_grid.csv"):
        search_rows.append({"parallelism": row["parallelism"], "stage": "phase32b_regular", "candidate_id": row["candidate_id"], "candidate_count": 1, "family": row["family"], "score": row["score"]})
    for row in read_csv(args.phase32d_dir / "analysis/candidate_grid.csv"):
        search_rows.append({"parallelism": "tp", "stage": "phase32d_targeted_rescue", "candidate_id": row["candidate_id"], "candidate_count": 1, "family": "development_oof_residual_gate", "score": row["score"]})
    write_csv(args.output_dir / "analysis/search_inventory.csv", search_rows)

    tp_gate_path = args.phase32d_dir / "checkpoints/tp_rescue_gate.json"
    pp_checkpoints = sorted((args.phase32b_dir / "checkpoints").glob("pp_top1_seed*.pt"))
    tp_source_checkpoints = []
    for value in read_json(tp_gate_path)["source_dnn_checkpoints"]:
        tp_source_checkpoints.append({
            **value,
            "path": str(Path("experiment-results/phase32b_expanded_residual_search/checkpoints") / Path(value["path"]).name),
        })
    checkpoint_registry = {
        "schema_version": "phase32f-best-checkpoint-registry-v1",
        "tp": {"model": summaries["phase32d"]["selected_candidate_id"], "gate": {"path": str(tp_gate_path.relative_to(repo_root)), "sha256": sha256(tp_gate_path)}, "source_dnn_checkpoints": tp_source_checkpoints},
        "pp": {"model": summaries["phase32b"]["selected"]["pp"]["candidate_id"], "source_dnn_checkpoints": [{"path": str(path.relative_to(repo_root)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in pp_checkpoints]},
        "note": "实际.pt checkpoint保存在Phase32B；TP附加gate保存在Phase32D。本目录登记路径与SHA，避免重复提交大文件。",
    }
    write_json(args.output_dir / "checkpoints/best_model_registry.json", checkpoint_registry)
    prediction_registry = {
        "schema_version": "phase32f-frozen-prediction-registry-v1",
        "tp": {"path": str((args.phase32d_dir / "analysis/frozen_predictions.csv.gz").relative_to(repo_root)), "sha256": summaries["phase32d"]["frozen_prediction_sha256"], "evidence": "new confirmation and original fixed; repeated engineering evaluation"},
        "pp": {"path": str((args.phase32b_dir / "analysis/frozen_predictions.csv.gz").relative_to(repo_root)), "sha256": summaries["phase32b"]["frozen_prediction_sha256"], "evidence": "new confirmation prediction frozen before one-time target opening"},
        "new_confirmation_teacher_labels": {"path": str((args.phase32c_dir / "labels/new_confirmation_hfull_targets.csv.gz").relative_to(repo_root)), "sha256": sha256(args.phase32c_dir / "labels/new_confirmation_hfull_targets.csv.gz")},
    }
    write_json(args.output_dir / "predictions/frozen_prediction_registry.json", prediction_registry)
    svg(args.output_dir / "figures/final_h0_vs_dnn.svg", tp, pp)

    report = f"""# Phase 32A-F：TP/PP扩容收敛实验最终报告

## 最终裁定

- **TP：未收口（fail）**。最优仍是`H0+DNN residual`，累计搜索达到绝对上限48组。救援后的新确认重复工程结果为calls MAPE/WAPE={pct(tp['h0_plus_dnn_residual']['calls_mape'])}/{pct(tp['h0_plus_dnn_residual']['calls_wape'])}、bytes MAPE/WAPE={pct(tp['h0_plus_dnn_residual']['bytes_mape'])}/{pct(tp['h0_plus_dnn_residual']['bytes_wape'])}、TV={tp['h0_plus_dnn_residual']['mean_histogram_tv']:.4f}、EMD={tp['h0_plus_dnn_residual']['mean_normalized_log_payload_emd']:.4f}、cost MAPE/WAPE={pct(tp['h0_plus_dnn_residual']['common_reference_cost_mape'])}/{pct(tp['h0_plus_dnn_residual']['common_reference_cost_wape'])}。calls WAPE仅比12%有条件线高0.19个百分点，但cost WAPE仍比6%线高2.57个百分点，不能判为有条件通过。
- **PP：有条件收口（conditional pass）**。最优模型是`{summaries['phase32b']['selected']['pp']['candidate_id']}`。一次性新确认上calls/TV/EMD/cost与各模型保护均通过；bytes WAPE={pct(pp['h0_plus_dnn_residual']['bytes_wape'])}高于正式3%线，因此不是正式通过。
- **第一阶段整体：部分收口**。PP已经满足今晚定义的有条件收口，TP方向一致改善但到达绝对搜索上限后仍未达到阈值。

## TP主结果（救援后重复工程证据）

{metric_table(tp)}

calls/cost WAPE相对H0改善{pct(tp['calls_relative_improvement'])}/{pct(tp['cost_relative_improvement'])}。需要强调：Phase32D选模完全只用开发侧分组OOF，但Phase32C已经先打开确认target，所以该数值是重复工程证据，不是新盲测。Phase32B在target开放前的一次性TP结果为calls WAPE 12.58%、cost WAPE 8.82%，同样裁定fail。

## PP主结果（一次性新确认）

{metric_table(pp)}

calls/cost WAPE相对H0改善{pct(pp['calls_relative_improvement'])}/{pct(pp['cost_relative_improvement'])}。PP MB16 calls MAPE相对H0改善{pct(pp['mb16']['calls_mape_relative_improvement'])}，且bytes/TV/cost没有同时恶化超过10%，满足MB16保护。

## 每模型结果

{model_table(tp, pp)}

## 数据、输入与隔离

- 既有59个请求级互斥正常画像保持不变：39训练、10验证、10原固定预测；新增9个BurstGPT正常确认窗口，与Phase27/28/30/31所有角色保持300秒请求区间隔离，且彼此互斥；原10个固定窗口没有更换。
- 开发teacher使用21,058个完整请求；原固定集2,786个请求；新增确认集2,976个请求。新增teacher为972条phase labels，完整请求列表没有保存或提交。
- 三个已知模型全部覆盖：deepseek-v2-lite、qwen3-8b、qwen3-30b-a3b。TP覆盖TP2/4/8与latency/balanced/throughput；PP覆盖PP2/4/8与MB1/4/16。
- 完整请求只用于离线Hfull teacher。最终预测输入仍是低维历史画像、模型结构、固定TP/PP配置与固定策略；输出仍是fixed-draining拓扑无关消息直方图，再代入同一连续通信代价曲线。
- 训练、alpha、gate、checkpoint与候选选择只使用训练/验证/5折profile分组OOF；固定target不进入上述任何环节。

## 搜索与停止原因

- TP：Phase31累计18组；Phase32B新增24组到常规上限42；Phase32D新增global/policy/model/phase/model×policy/policy×phase六组开发OOF gate，到绝对上限48后停止。
- PP：Phase31累计12组；Phase32B新增18组到常规上限30，覆盖bytes/cost保护loss与MB独立gate。一次性新确认达到有条件通过，按停止规则不再使用6组救援额度。
- 初筛每组1个seed；每方向前三组进行3-seed、5折确认。没有无边界搜索，也没有改变固定预测集或阈值。

## 可以得出的结论

在当前三个已知模型、正常历史流量、既定TP/PP配置与fixed-draining策略范围内，PP的H0+DNN residual已经给出可自洽的有条件收口证据；TP residual在全部三个模型上均改善calls和cost，但绝对cost误差仍阻止收口。结构H0仍是必要基线，DNN没有被取消。

## 不可以得出的结论

不能声称TP已经通过；不能把TP救援复评包装成新盲测；不能声称PP正式通过；不能外推到未见模型、极端流量或生产全域。新增确认只有BurstGPT，因为累计300秒隔离后Mooncake没有剩余完整窗口。

## 保存位置

- node55：`/sgl-workspace/sglang-src/experiment-results/phase32f_tp_pp_expanded_convergence_final`
- 本地：`{args.output_dir}`
- 整体、逐模型、逐policy、逐并行规模、搜索清单分别见`analysis/`；checkpoint与冻结预测的实际路径和SHA见`checkpoints/`、`predictions/`。

## 下一步

今晚搜索已经按绝对上限停止。若后续继续TP，必须先冻结新的请求级互斥确认集，再从开发侧增加正常窗口或改进专门针对总量/代价的target-free residual；不能继续用已打开的Phase32确认target挑模型。PP保持当前incumbent，新增到6个模型时统一重训并重新确认。
"""

    milestone = f"""# Phase 32A-F里程碑归档

1. Phase32A冻结扩容合同与9个新请求级互斥BurstGPT窗口：PASS，target未生成。
2. Phase32B完成TP 42组、PP 30组常规上限搜索并先冻结预测：PASS。
3. Phase32C在冻结SHA后一次性生成新Hfull teacher并评测：TP fail，PP conditional_pass。
4. Phase32D用开发OOF完成TP最后6组gate救援：PASS，累计48组绝对上限。
5. Phase32E只评估已冻结救援预测：TP仍fail；明确标为重复工程证据。
6. Phase32F停止训练并完成总归档：TP未收口，PP有条件收口。

数据量：新确认9个画像、2,976个完整teacher请求、972条target phase rows；Phase32B冻结4,104条预测phase rows，Phase32D冻结2,052条TP预测phase rows。目录体积以最终manifest和`du`复核为准。

提交链：Phase32A=`b476e093`，Phase32B=`89b91d9b`，Phase32C=`0a8b5d74`，Phase32D脚本/结果=`067e890c`/`842479a7`，Phase32E结果=`c1f73c6f`；Phase32F提交以最终Git HEAD为准。
"""

    handoff = f"""# 新会话完整交接（截至Phase32F）

## 当前结论

- 基础定义不变：低维历史画像、模型结构、固定执行策略和既定TP/PP配置作为输入；Hfull只作离线teacher；目标是fixed-draining拓扑无关消息直方图与同一连续通信代价曲线。
- TP最佳：`{summaries['phase32d']['selected_candidate_id']}`，仍为H0+DNN residual。新确认重复工程calls/bytes/TV/EMD/cost WAPE={pct(tp['h0_plus_dnn_residual']['calls_wape'])}/{pct(tp['h0_plus_dnn_residual']['bytes_wape'])}/{tp['h0_plus_dnn_residual']['mean_histogram_tv']:.4f}/{tp['h0_plus_dnn_residual']['mean_normalized_log_payload_emd']:.4f}/{pct(tp['h0_plus_dnn_residual']['common_reference_cost_wape'])}，累计48组绝对上限，裁定fail。
- PP最佳：`{summaries['phase32b']['selected']['pp']['candidate_id']}`，仍为H0+DNN residual。一次性新确认calls/bytes/TV/EMD/cost WAPE={pct(pp['h0_plus_dnn_residual']['calls_wape'])}/{pct(pp['h0_plus_dnn_residual']['bytes_wape'])}/{pp['h0_plus_dnn_residual']['mean_histogram_tv']:.4f}/{pp['h0_plus_dnn_residual']['mean_normalized_log_payload_emd']:.4f}/{pct(pp['h0_plus_dnn_residual']['common_reference_cost_wape'])}，裁定conditional_pass。

## 数据与证据边界

- 原59个正常互斥画像不变；新增9个BurstGPT请求级互斥确认窗口，2,976个完整teacher请求。新增确认不含Mooncake。
- Phase32C是PP主证据和TP常规模型的一次性确认；Phase32D之后TP评测只能称重复工程证据。所有选模仍只用开发侧数据。

## 仓库

- 分支：`experiment/pattern-demand-v0.5.15-clean`
- node55：`/sgl-workspace/sglang-src`
- 本地：`{Path(__file__).resolve().parents[1]}`
- 最终目录：`experiment-results/phase32f_tp_pp_expanded_convergence_final`

## 必须保护

继续保护本地`data/`、远端Phase16 GPU目录、Phase19 formal-v1/v2/smoke与PID、Phase23历史PID/tmp、raw trace、缓存和所有PID；不得使用`git add .`。

## 下一步

不要继续在已打开的Phase32 target上调TP。若扩到6个模型，应先增加模型结构特征与开发teacher、统一重训TP/PP，再冻结新的互斥确认集。PP可作为当前incumbent。
"""

    asset_append = f"""\n\n# Phase32A-F资产补充

- `experiment-results/phase32a_expanded_search_contract/`：扩容合同、新互斥窗口选择、无target特征与清单；84 KiB级。
- `experiment-results/phase32b_expanded_residual_search/`：42个新增候选初筛、18个正式checkpoint、4,104条冻结预测；约13 MiB。
- `experiment-results/phase32c_frozen_prediction_evaluation/`：新确认Hfull标签、一次性评测、逐切片指标；约652 KiB。
- `experiment-results/phase32d_tp_gate_rescue/`：TP六组定向gate、gate checkpoint、2,052条冻结预测；约724 KiB。
- `experiment-results/phase32e_tp_rescue_repeated_evaluation/`：TP救援后重复工程评测；约292 KiB。
- `experiment-results/phase32f_tp_pp_expanded_convergence_final/`：最终报告、摘要、逐模型/逐policy/逐并行规模指标、搜索清单、checkpoint/预测索引、图表、日志、DONE与manifest。

正式提交链：`b476e093`（32A）、`89b91d9b`（32B）、`0a8b5d74`（32C）、`842479a7`（32D结果）、`c1f73c6f`（32E结果）；32F以最终HEAD为准。所有大体积raw、完整请求列表、缓存和PID均未提交。
"""
    guide_append = f"""\n\n# Phase32F状态补充：扩容有限收敛结果

基础研究定义、Hfull teacher、fixed-draining语义、指标与阈值均未改变。Phase32将TP累计搜索从18组扩到48组绝对上限，将PP从12组扩到30组常规上限，并新增9个与既有角色请求区间互斥的BurstGPT正常确认窗口。

最终结论：PP H0+DNN residual在一次性新确认上达到有条件通过，calls/bytes/TV/EMD/cost WAPE分别为{pct(pp['h0_plus_dnn_residual']['calls_wape'])}/{pct(pp['h0_plus_dnn_residual']['bytes_wape'])}/{pp['h0_plus_dnn_residual']['mean_histogram_tv']:.4f}/{pp['h0_plus_dnn_residual']['mean_normalized_log_payload_emd']:.4f}/{pct(pp['h0_plus_dnn_residual']['common_reference_cost_wape'])}；TP虽相对H0持续改善，但在48组上限处仍fail，救援后重复工程calls/cost WAPE为{pct(tp['h0_plus_dnn_residual']['calls_wape'])}/{pct(tp['h0_plus_dnn_residual']['common_reference_cost_wape'])}。因此第一阶段当前是“PP有条件收口、TP未收口”。

证据边界：新增确认仅BurstGPT；TP救援发生在Phase32C target开放之后，尽管训练与gate选择未读取target，其复评仍不是新盲测。
"""

    prior_guide = "\n".join(line.rstrip() for line in (args.deliverable_dir / "截至目前实验结构总导引_含Phase31状态补充.md").read_text().splitlines()) + "\n"
    prior_assets = "\n".join(line.rstrip() for line in (args.deliverable_dir / "实验资产与保存流程全量索引_截至Phase31G.md").read_text().splitlines()) + "\n"
    guide = prior_guide + guide_append
    assets = prior_assets + asset_append
    documents = {
        "Phase32A-F_TP_PP扩容收敛实验最终报告.md": report,
        "Phase32A-F_TP_PP扩容收敛实验里程碑报告.md": milestone,
        "新会话完整交接_截至Phase32F.md": handoff,
        "实验资产与保存流程全量索引_截至Phase32F.md": assets,
        "截至目前实验结构总导引_含Phase32F状态补充.md": guide,
    }
    for name, content in documents.items():
        (args.deliverable_dir / name).write_text(content)
        (args.output_dir / "docs" / name).write_text(content)
    (args.output_dir / "FINAL_REPORT.md").write_text(report)
    (args.output_dir / "README.md").write_text("# Phase 32F：TP/PP扩容收敛最终归档\n\n归档状态PASS；科学裁定为TP=`fail`、PP=`conditional_pass`。本目录汇总报告、指标、搜索清单、checkpoint与预测SHA索引、交接文档、图表和审计清单。\n")

    checks = {
        "phase32a_to_e_status_pass": all(summary["status"] == "PASS" for summary in summaries.values()),
        "phase32a_to_e_manifests_pass": all(verify_manifest(path) for path in dirs.values()),
        "tp_cumulative_absolute_48": summaries["phase32d"]["search"]["final_cumulative"] == 48,
        "pp_cumulative_regular_30": summaries["phase32b"]["search"]["cumulative_counts"]["pp"] == 30,
        "tp_scientific_decision_fail": tp["decision"] == "fail",
        "pp_scientific_decision_conditional": pp["decision"] == "conditional_pass",
        "tp_and_pp_calls_cost_improve_h0": tp["calls_relative_improvement"] > 0 and tp["cost_relative_improvement"] > 0 and pp["calls_relative_improvement"] > 0 and pp["cost_relative_improvement"] > 0,
        "new_confirmation_profiles_9": summaries["phase32c"]["counts"]["new_profiles"] == 9,
        "new_confirmation_requests_2976": summaries["phase32c"]["counts"]["new_full_requests"] == 2976,
        "target_isolation_contract_pass": summaries["phase32b"]["fixed_targets_read"] is False and summaries["phase32b"]["new_confirmation_targets_read"] is False and summaries["phase32d"]["fixed_targets_read"] is False and summaries["phase32d"]["new_confirmation_targets_read"] is False,
        "checkpoint_registry_complete": len(pp_checkpoints) == 3 and tp_gate_path.is_file(),
        "documents_written": all((args.output_dir / "docs" / name).is_file() and (args.deliverable_dir / name).is_file() for name in documents),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": "phase32f-tp-pp-expanded-convergence-final-v1",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_closure": {"tp": "fail", "pp": "conditional_pass", "overall": "partial_closure"},
        "best_models": {"tp": summaries["phase32d"]["selected_candidate_id"], "pp": summaries["phase32b"]["selected"]["pp"]["candidate_id"]},
        "primary_metrics": {"tp_repeated_engineering": tp, "pp_new_confirmation_once": pp},
        "search_counts": {"tp": {"regular": 42, "final": 48, "absolute": 48}, "pp": {"regular": 30, "final": 30, "absolute": 36}},
        "evidence": {"development_profiles": 49, "original_fixed_profiles": 10, "new_confirmation_profiles": 9, "new_confirmation_scope": "BurstGPT-only request-disjoint normal windows", "new_confirmation_full_requests": 2976, "new_confirmation_target_phase_rows": 972},
        "commits_before_final_archive": {"phase32a": "b476e093", "phase32b": "89b91d9b", "phase32c": "0a8b5d74", "phase32d": "842479a7", "phase32e": "c1f73c6f"},
        "checks": checks,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit_summary.json", {"schema_version": "phase32f-audit-v1", "status": status, "checks": checks})
    write_json(args.output_dir / "logs/finalization.log", {"event": "phase32f_final_archive_complete", "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "repository_head_before_final_archive": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.version, "platform": platform.platform(), "scientific_closure": summary["scientific_closure"]})
    (args.output_dir / "DONE").write_text(status + "\n")
    manifest = [f"{sha256(path)}  {path.relative_to(args.output_dir)}" for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if status != "PASS":
        raise RuntimeError(checks)
    print(json.dumps({"status": status, "scientific_closure": summary["scientific_closure"], "best_models": summary["best_models"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
