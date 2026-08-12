#!/usr/bin/env python3
"""Build the final Chinese TP/PP convergence report from frozen Phase31 assets."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiment-results"
OUTPUT = RESULTS / "phase31g_tp_pp_convergence_final"
PHASES = {
    "phase31a": RESULTS / "phase31a_known_model_convergence_contract",
    "phase31b": RESULTS / "phase31b_known_model_hfull_dataset",
    "phase31c": RESULTS / "phase31c_known_model_residual_training",
    "phase31d": RESULTS / "phase31d_known_model_fixed_evaluation",
    "phase31e": RESULTS / "phase31e_tp_weighted_residual_round2",
    "phase31f": RESULTS / "phase31f_tp_round2_fixed_evaluation",
}
MODELS = ("deepseek-v2-lite", "qwen3-8b", "qwen3-30b-a3b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def commit_for(path: Path) -> str:
    return subprocess.check_output(["git", "log", "-n", "1", "--format=%H", "--", str(path.relative_to(ROOT))], cwd=ROOT, text=True).strip()


def pct(value: float) -> str:
    return f"{100 * float(value):.2f}%"


def render_svg(path: Path, decisions: dict) -> None:
    keys = ("calls_wape", "bytes_wape", "mean_histogram_tv", "common_reference_cost_wape")
    labels = ("calls WAPE", "bytes WAPE", "TV", "cost WAPE")
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="620" viewBox="0 0 1100 620">',
        '<rect width="1100" height="620" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:24px;font-weight:700}.label{font-size:14px}.value{font-size:12px}</style>',
        '<text x="550" y="38" text-anchor="middle" class="title">Phase 31 TP/PP fixed prediction convergence</text>',
    ]
    for panel, parallelism in enumerate(("tp", "pp")):
        left = 55 + panel * 535
        decision = decisions[parallelism]
        pieces.append(f'<text x="{left + 235}" y="80" text-anchor="middle" class="title">{parallelism.upper()} — {decision["decision"]}</text>')
        for index, (key, label) in enumerate(zip(keys, labels)):
            y = 120 + index * 112
            pieces.append(f'<text x="{left}" y="{y - 11}" class="label">{label}</text>')
            for row_index, (name, color) in enumerate((("h0", "#8b95a5"), ("h0_plus_dnn_residual", "#276ef1"))):
                value = float(decision[name][key])
                width = min(value / 0.30 * 380, 380)
                bar_y = y + row_index * 31
                pieces.append(f'<text x="{left}" y="{bar_y + 16}" class="value">{"H0" if name == "h0" else "H0+DNN"}</text>')
                pieces.append(f'<rect x="{left + 82}" y="{bar_y}" width="{width:.2f}" height="21" rx="3" fill="{color}"/>')
                pieces.append(f'<text x="{left + 88 + width:.2f}" y="{bar_y + 16}" class="value">{100 * value:.2f}%</text>')
    pieces.extend(['<text x="550" y="596" text-anchor="middle" class="label">Fixed set: 10 request-disjoint normal-history profiles, three known models</text>', '</svg>'])
    path.write_text("\n".join(pieces) + "\n")


def main() -> None:
    for name in ("analysis", "figures", "logs", "docs"):
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)
    summaries = {name: read_json(path / "summary.json") for name, path in PHASES.items()}
    if not all(summary["status"] == "PASS" for summary in summaries.values()):
        raise RuntimeError("an input Phase31 asset is not PASS")
    decisions = summaries["phase31d"]["decisions"]
    phase31c = summaries["phase31c"]
    phase31e = summaries["phase31e"]
    phase31f = summaries["phase31f"]
    if phase31f["decision"]["decision"] != decisions["tp"]["decision"]:
        raise RuntimeError("TP repeated evaluation drift")

    headline_rows = []
    model_rows = []
    for parallelism in ("tp", "pp"):
        decision = decisions[parallelism]
        for method in ("h0", "h0_plus_dnn_residual"):
            row = decision[method]
            headline_rows.append({"parallelism": parallelism, "decision": decision["decision"], "method": method, **{key: row[key] for key in ("calls_mape", "calls_wape", "bytes_mape", "bytes_wape", "mean_histogram_tv", "mean_normalized_log_payload_emd", "common_reference_cost_mape", "common_reference_cost_wape")}})
        for model in MODELS:
            value = decision["models"][model]
            for method in ("h0", "h0_plus_dnn_residual"):
                row = value[method]
                model_rows.append({"parallelism": parallelism, "model": model, "method": method, **{key: row[key] for key in ("calls_mape", "calls_wape", "bytes_mape", "bytes_wape", "mean_histogram_tv", "mean_normalized_log_payload_emd", "common_reference_cost_mape", "common_reference_cost_wape")}})
    with (PHASES["phase31d"] / "analysis/aggregate_metrics.csv").open(newline="") as source:
        aggregate_rows = list(csv.DictReader(source))
    policy_rows = [row for row in aggregate_rows if row["phase"] == "total" and row["slice_type"] == "policy"]
    write_csv(OUTPUT / "analysis/headline_metrics.csv", headline_rows)
    write_csv(OUTPUT / "analysis/per_model_metrics.csv", model_rows)
    write_csv(OUTPUT / "analysis/per_policy_metrics.csv", policy_rows)
    render_svg(OUTPUT / "figures/final_headline.svg", decisions)

    commits = {name: commit_for(path) for name, path in PHASES.items()}
    tp = decisions["tp"]
    pp = decisions["pp"]
    pp_mb16 = pp["mb16"]
    summary = {
        "schema_version": "phase31g-tp-pp-convergence-final-v1",
        "status": "PASS",
        "combined_closure": "not_closed_tp_failed_pp_conditional_pass",
        "tp": {
            "decision": tp["decision"],
            "best_model": phase31c["selected"]["tp"],
            "final_round_selected_source": phase31e["selected_source"],
            "search_configurations_total": 18,
            "fixed_metrics": tp["h0_plus_dnn_residual"],
            "h0_metrics": tp["h0"],
            "calls_relative_improvement": tp["calls_relative_improvement"],
            "cost_relative_improvement": tp["cost_relative_improvement"],
            "per_model": tp["models"],
        },
        "pp": {
            "decision": pp["decision"],
            "best_model": phase31c["selected"]["pp"],
            "search_configurations_total": 12,
            "fixed_metrics": pp["h0_plus_dnn_residual"],
            "h0_metrics": pp["h0"],
            "calls_relative_improvement": pp["calls_relative_improvement"],
            "cost_relative_improvement": pp["cost_relative_improvement"],
            "per_model": pp["models"],
            "mb16": pp_mb16,
        },
        "data": {
            "profiles": summaries["phase31a"]["profiles"],
            "roles": summaries["phase31a"]["role_counts"],
            "development_full_requests": summaries["phase31b"]["full_window_requests_development"],
            "fixed_requests": summaries["phase31d"]["fixed_requests"],
            "development_target_phase_rows": summaries["phase31b"]["target_phase_rows"],
            "fixed_target_phase_rows": summaries["phase31d"]["target_phase_rows"],
            "models": summaries["phase31a"]["models"],
        },
        "commits": commits,
        "input_summary_sha256": {name: sha256(path / "summary.json") for name, path in PHASES.items()},
        "conclusion_scope": "known three-model normal-history in-distribution fixed-draining first-stage evidence only",
        "cannot_claim": ["TP reached formal or conditional threshold", "unseen-model zero-shot generalization", "extreme-traffic or production-wide generalization", "fresh independent evidence from Phase31F repeated fixed evaluation"],
    }
    write_json(OUTPUT / "summary.json", summary)
    checks = {
        "all_phase31_inputs_pass": all(value["status"] == "PASS" for value in summaries.values()),
        "tp_decision_fail": tp["decision"] == "fail",
        "pp_decision_conditional_pass": pp["decision"] == "conditional_pass",
        "tp_search_cap_18": phase31e["counts"]["new_candidates"] + phase31c["search_limits"]["candidates_per_parallelism"] == 18,
        "pp_three_models": set(pp["models"]) == set(MODELS),
        "tp_three_models": set(tp["models"]) == set(MODELS),
        "phase31f_repeated_metrics_identical": phase31f["checks"]["same_fixed_h0_calls_wape"] and phase31f["decision"]["h0_plus_dnn_residual"] == tp["h0_plus_dnn_residual"],
    }
    write_json(OUTPUT / "audit_summary.json", {"schema_version": "phase31g-final-audit-v1", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks})

    model_lines = []
    for parallelism in ("tp", "pp"):
        for model in MODELS:
            value = decisions[parallelism]["models"][model]["h0_plus_dnn_residual"]
            model_lines.append(f"| {parallelism.upper()} | {model} | {pct(value['calls_wape'])} | {pct(value['bytes_wape'])} | {value['mean_histogram_tv']:.4f} | {value['mean_normalized_log_payload_emd']:.4f} | {pct(value['common_reference_cost_wape'])} |")
    report = f"""# Phase 31G：今晚TP/PP收敛最终汇总

## 最终裁定

- TP：**未收口（fail）**。最佳模型仍为Phase31C共享`H0+DNN residual`，固定集calls WAPE为{pct(tp['h0_plus_dnn_residual']['calls_wape'])}、cost WAPE为{pct(tp['h0_plus_dnn_residual']['common_reference_cost_wape'])}；相对H0分别改善{pct(tp['calls_relative_improvement'])}和{pct(tp['cost_relative_improvement'])}，但仍超过有条件阈值12%/6%。Phase31E已将TP搜索补足至18组上限，新模型没有在验证集超过incumbent，因此停止搜索。
- PP：**有条件收口（conditional pass）**。最佳模型为按MB小头的`H0+DNN residual`；固定集calls WAPE {pct(pp['h0_plus_dnn_residual']['calls_wape'])}、bytes WAPE {pct(pp['h0_plus_dnn_residual']['bytes_wape'])}、TV {pp['h0_plus_dnn_residual']['mean_histogram_tv']:.4f}、EMD {pp['h0_plus_dnn_residual']['mean_normalized_log_payload_emd']:.4f}、cost WAPE {pct(pp['h0_plus_dnn_residual']['common_reference_cost_wape'])}。calls/cost相对H0改善{pct(pp['calls_relative_improvement'])}/{pct(pp['cost_relative_improvement'])}。
- 整体第一阶段：**尚未完全收口**，原因只在TP；PP已达到今晚定义的有条件通过。

## 数据与模型

- 59个请求级互斥正常历史画像：39训练、10验证、10固定预测；三个已知模型均覆盖训练、验证和固定预测；
- 开发Hfull teacher使用21,058个完整窗口请求，固定评测使用2,786个请求；开发/固定target分别5,292/1,080条phase rows；
- TP覆盖TP2/4/8与latency/balanced/throughput；PP覆盖PP2/4/8与MB1/4/16；
- 完整请求只用于离线teacher，不是预测输入；最终输入仍是低维画像、模型结构、固定并行配置和策略。

## 每模型固定集指标

| 方向 | 模型 | calls WAPE | bytes WAPE | TV | EMD | cost WAPE |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(model_lines)}

PP MB16 calls MAPE由{pct(pp_mb16['h0']['calls_mape'])}降至{pct(pp_mb16['h0_plus_dnn_residual']['calls_mape'])}，相对改善{pct(pp_mb16['calls_mape_relative_improvement'])}，达到单独保护条件。

## 可以得出的结论

在当前三个已知模型和正常历史流量范围内，PP的`H0+DNN residual`对fixed-draining消息直方图及统一参考通信代价具有稳定价值，并达到有条件收口；TP residual也在三个模型上方向一致地改善calls和cost，但绝对误差还不足以宣告收口。

## 不可以得出的结论

不能声称TP已经通过；不能声称对未见模型、极端流量或所有生产环境具备零样本泛化；Phase31F是同一固定集重复一致性评测，不是新盲测。

## 保存位置

- 本地：`{OUTPUT}`；
- node55：`/sgl-workspace/sglang-src/experiment-results/{OUTPUT.name}`；
- 逐模型、逐policy及整体指标分别在`analysis/per_model_metrics.csv`、`analysis/per_policy_metrics.csv`和`analysis/headline_metrics.csv`。

## 下一步

不要继续在已打开的固定集上调TP。下一轮应先冻结新的请求级互斥确认集，再只用开发侧扩充正常窗口或改进低维顺序特征/形状residual；若仍保持今晚边界，则当前诚实结论就是“PP有条件收口、TP未收口”。
"""
    (OUTPUT / "README.md").write_text(report)
    (OUTPUT / "FINAL_REPORT.md").write_text(report)
    milestone_lines = []
    milestones = (
        ("31A", "冻结59个请求级互斥画像与三模型/TP/PP合同", "phase31a"),
        ("31B", "生成开发Hfull teacher与target-free固定特征", "phase31b"),
        ("31C", "TP/PP各12组有限H0+DNN residual训练并冻结预测", "phase31c"),
        ("31D", "首次打开固定Hfull target；TP fail、PP conditional pass", "phase31d"),
        ("31E", "TP追加6组加权/多头方案至18组上限，保留incumbent", "phase31e"),
        ("31F", "同一固定集一致性复评，TP指标逐值不变", "phase31f"),
    )
    for phase, description, key in milestones:
        directory = PHASES[key]
        files = sum(1 for path in directory.rglob("*") if path.is_file())
        size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        milestone_lines.append(f"| Phase {phase} | {description} | `{commits[key][:12]}` | `{directory.name}` | {files} | {size / 1024 / 1024:.2f} MiB |")
    milestone_report = f"""# Phase31A-G TP/PP收敛实验里程碑报告

本报告补充今晚执行参考，不替代实验总导引。

| 里程碑 | 实际状态与证据 | Git提交 | 结果目录 | 文件数 | 数据量 |
|---|---|---|---|---:|---:|
{chr(10).join(milestone_lines)}
| Phase 31G | 汇总最终裁定、逐模型/逐policy指标、结论边界与下一步 | 本目录归档提交 | `{OUTPUT.name}` | 见manifest | 见manifest |

最终裁定：TP未收口，PP有条件收口，整体第一阶段尚未完全收口。TP固定集calls/cost WAPE为{pct(tp['h0_plus_dnn_residual']['calls_wape'])}/{pct(tp['h0_plus_dnn_residual']['common_reference_cost_wape'])}，相对H0改善{pct(tp['calls_relative_improvement'])}/{pct(tp['cost_relative_improvement'])}；PP固定集calls/bytes/cost WAPE为{pct(pp['h0_plus_dnn_residual']['calls_wape'])}/{pct(pp['h0_plus_dnn_residual']['bytes_wape'])}/{pct(pp['h0_plus_dnn_residual']['common_reference_cost_wape'])}。

所有里程碑均包含中文README、summary、logs、DONE与manifest；正式训练阶段保存checkpoint和冻结预测，评测阶段保存逐case/聚合指标与图表。raw trace、完整请求列表、缓存和PID未提交。
"""
    (OUTPUT / "docs/Phase31A_G_里程碑报告.md").write_text(milestone_report)
    handoff = f"""# 新会话完整交接（截至Phase31G）

## 当前结论

- 研究基础定义不变：低维历史画像、模型结构、固定执行策略及既定TP/PP配置作为输入；Hfull只作离线teacher；预测fixed-draining拓扑无关消息直方图，再代入连续通信代价曲线。
- 数据合同：59个请求级互斥正常画像，39训练、10验证、10固定预测；三个已知模型均覆盖三种角色。
- TP最佳：Phase31C `tp_c03_full_shared_lr0.003_3seed_alpha0.75`，固定calls/bytes/TV/EMD/cost为{pct(tp['h0_plus_dnn_residual']['calls_wape'])}/{pct(tp['h0_plus_dnn_residual']['bytes_wape'])}/{tp['h0_plus_dnn_residual']['mean_histogram_tv']:.4f}/{tp['h0_plus_dnn_residual']['mean_normalized_log_payload_emd']:.4f}/{pct(tp['h0_plus_dnn_residual']['common_reference_cost_wape'])}。TP搜索已到18组上限，裁定fail。
- PP最佳：Phase31C `pp_c05_full_policy_heads_lr0.001_3seed_alpha1.0`，固定calls/bytes/TV/EMD/cost为{pct(pp['h0_plus_dnn_residual']['calls_wape'])}/{pct(pp['h0_plus_dnn_residual']['bytes_wape'])}/{pp['h0_plus_dnn_residual']['mean_histogram_tv']:.4f}/{pp['h0_plus_dnn_residual']['mean_normalized_log_payload_emd']:.4f}/{pct(pp['h0_plus_dnn_residual']['common_reference_cost_wape'])}，裁定conditional_pass。

## 仓库与保存

- 分支：`experiment/pattern-demand-v0.5.15-clean`；
- node55仓库：`/sgl-workspace/sglang-src`；
- 本地仓库：`{ROOT}`；
- 最终总报告：`experiment-results/{OUTPUT.name}`；
- Phase31A-F提交：{', '.join(f'{key}={value[:12]}' for key, value in commits.items())}。

## 必须继续保护

- 本地`data/`；
- 远端`experiment-results/phase16_profiledemand_gpu/`；
- 远端Phase19 formal-v1/v2/smoke与PID；
- 远端Phase23历史PID、server PID及tmp；
- 不使用`git add .`，不提交raw trace、完整请求列表、缓存或PID。

## 下一步

不要再用已打开的10个固定窗口调TP。若继续TP，应先target-blind冻结新的请求级互斥确认集，再扩充开发侧正常窗口或实现低维顺序/形状residual，并在新确认集上只评一次。PP已条件收口，不应继续搜索；后续可随模型从3个扩展到6个时统一重训。

## 结论边界

可以说PP在当前三个已知模型和正常流量范围内有用且条件收口，TP residual方向一致改善但未收口。不能说TP通过，也不能声称未见模型、极端流量或生产全域零样本泛化；Phase31F不是新盲测。
"""
    (OUTPUT / "docs/新会话完整交接_截至Phase31G.md").write_text(handoff)
    asset_index = f"""# 实验资产与保存流程全量索引（截至Phase31G）

## Phase31正式资产

| 阶段 | 远端/本地相对目录 | 提交 | manifest状态 |
|---|---|---|---|
{chr(10).join(f'| {key.upper()} | `experiment-results/{PHASES[key].name}` | `{commits[key]}` | PASS |' for key in PHASES)}
| PHASE31G | `experiment-results/{OUTPUT.name}` | 本目录归档提交 | PASS |

远端根目录为`/sgl-workspace/sglang-src`，本地根目录为`{ROOT}`。每个目录均以自身`manifest.sha256`为校验入口。

## 数据量

- 画像：59（39/10/10）；开发完整请求21,058，固定完整请求2,786；
- 开发Hfull标签5,292条phase rows；固定Hfull标签1,080条phase rows；
- Phase31C checkpoint 12个，Phase31E候选checkpoint 6个；
- 冻结预测：Phase31C共2,160条phase-method rows；Phase31E TP共1,080条。

## 保存流程

1. 先冻结选择与SHA；2. target-free训练并冻结预测；3. 选择性`git add`/`git add -f`正式目录；4. commit/push；5. 另一端`git pull --ff-only`；6. 在目录内校验manifest；7. 只在预测归档后打开固定target。

## 排除资产

不提交本地`data/`、raw profiler trace、完整请求列表、模型权重缓存、旧Phase16 GPU目录、Phase19 formal/smoke、Phase23 PID/tmp及任何PID文件。Phase31中的`.pt`是本次正式小型DNN checkpoint，不是模型权重缓存。
"""
    (OUTPUT / "docs/实验资产与保存流程全量索引_截至Phase31G.md").write_text(asset_index)
    write_json(OUTPUT / "logs/build.log", {"event": "phase31g_final_report_built", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "repository_head_at_build": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version, "platform": platform.platform(), "combined_closure": summary["combined_closure"]})
    (OUTPUT / "DONE").write_text("PASS\n")
    manifest = [f"{sha256(path)}  {path.relative_to(OUTPUT)}" for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path.name != "manifest.sha256"]
    (OUTPUT / "manifest.sha256").write_text("\n".join(manifest) + "\n")
    if not all(checks.values()):
        raise RuntimeError(checks)
    print(json.dumps({"status": "PASS", "combined_closure": summary["combined_closure"], "tp": tp["decision"], "pp": pp["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
