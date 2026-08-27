#!/usr/bin/env python3
"""Deterministic Phase72 evidence tables, claim freeze, guide and SVG figures."""
from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("refuse empty CSV")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def load_sources(root: Path, spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["phase"]: json.loads((root / "experiment-results" / row["directory"] / "summary.json").read_text(encoding="utf-8"))
        for row in spec["source_results"]
    }


def build_claim_scope(old: dict[str, Any]) -> dict[str, Any]:
    frozen = list(old["frozen_claims"]) + [
        {"id": "F13", "claim": "Phase58/59在受保护development上继续改善H0，但严格逐模型、逐segment直方图精度门没有通过；这些结果不是fresh-blind正结论。", "evidence": "Phase58/59"},
        {"id": "F14", "claim": "冻结R61两流拥塞修正在reserved payload和新placement上通过fresh-blind，并由Phase63扩展为六模型两流物理证据。", "evidence": "Phase60-63"},
        {"id": "F15", "claim": "四流图公式的Phase64零样本失败、Phase66第一次盲测失败和Phase68第二次盲测失败均被保留，未用盲测标签回调。", "evidence": "Phase64/66/68"},
        {"id": "F16", "claim": "R69高page残差在两个代表模型、四种已测图、L1-L3和最多四条流的第三次fresh-blind上通过预注册精度门。", "evidence": "Phase69/70"},
        {"id": "F17", "claim": "在bin_aligned边际wave合同下，Phase71的21/21配置×拓扑cost比较和7/7配置placement比较均显示H0+DNN弱或严格优于H0。", "evidence": "Phase71"},
        {"id": "F18", "claim": "12-bin边际直方图不能唯一恢复并发消息配对；Phase71敏感性分析显示P2D2 all-to-all的代价和placement可明显随wave配对改变。", "evidence": "Phase71"},
    ]
    prohibited = list(old["prohibited_claims"]) + [
        {"id": "N11", "claim": "不能宣称PD calls/bytes histogram WAPE已经达到统一10%或15%的逐模型、逐segment绝对门。"},
        {"id": "N12", "claim": "不能把R69四流修正外推到Phase70没有测量的另外四个模型。"},
        {"id": "N13", "claim": "不能外推到超过四条并发流、P2D4/P4D2/P4D4或未测通信图。"},
        {"id": "N14", "claim": "不能宣称边际12-bin直方图恢复了真实wave配对，或placement对任意并发配对都稳定。"},
        {"id": "N15", "claim": "Phase71仍是communication-only确定性重算，不是完整scheduler或端到端线上收益证明。"},
    ]
    return {
        "schema_version": "phase72-claim-scope-v1",
        "frozen_claims": frozen,
        "prohibited_claims": prohibited,
        "future_scheduler_dimensions": old["future_scheduler_dimensions"],
        "governance": {
            "predictor_status": "freeze TP/PP/PD accepted predictor evidence; Phase59 strict histogram target remains open",
            "curve_status": "freeze Phase39 TP/PP, Phase51 P1D1 and Phase60-70 measured PD multi-flow physical evidence",
            "gpu_status": "no mandatory B200 rerun remains inside the frozen scope",
            "next_research_boundary": "either add a genuinely different low-dimensional workload reconstruction baseline on CPU, or open a separately contracted scheduler study",
        },
    }


def _outcome(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_evidence(spec: dict[str, Any], sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "phase": row["phase"], "chain": row["chain"], "evidence_class": row["evidence_class"],
        "status": sources[row["phase"]]["status"],
        "scientific_outcome": _outcome(sources[row["phase"]].get("scientific_outcome", sources[row["phase"]].get("outcome"))),
        "result_commit": row["result_commit"], "role": row["role"],
    } for row in spec["source_results"]]


def build_histogram_rows(sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for phase in ("Phase58", "Phase59"):
        block = sources[phase]["development_validation"]
        for metric in ("calls_histogram_wape", "bytes_histogram_wape"):
            rows.append({
                "phase": phase, "metric": metric, "h0_wape": block["h0"][metric],
                "refined_wape": block["h0_plus_dnn_refined"][metric],
                "relative_ratio": block["h0_plus_dnn_refined"][metric] / block["h0"][metric],
                "target_met": sources[phase]["gates"]["target_met"],
            })
    return rows


def build_ladder_rows(sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    values = [
        ("Phase62", "two-flow / 2 models", sources["Phase62"]["metrics"]["frozen_corrected_overall_wape"], True),
        ("Phase63", "two-flow / 6 models", sources["Phase63"]["metrics"]["combined_six_model_frozen_corrected_overall_wape"], True),
        ("Phase64", "four-flow zero-shot", sources["Phase64"]["metrics"]["graph_formula_overall_wape"], False),
        ("Phase66", "four-flow correction v1", sources["Phase66"]["metrics"]["phase65_overall_wape"], False),
        ("Phase68", "four-flow correction v2", sources["Phase68"]["metrics"]["phase67_overall_wape"], False),
        ("Phase70", "four-flow correction v3", sources["Phase70"]["metrics"]["phase69_overall_wape"], True),
    ]
    return [{"phase": p, "scope": scope, "overall_wape": wape, "fresh_gate_pass": passed} for p, scope, wape, passed in values]


def build_phase71_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base = root / "experiment-results/phase71_pd_multiflow_cost_placement_integration/analysis"
    cost_raw = [r for r in read_csv(base / "cost_method_comparison.csv") if r["slice_type"] == "overall" and r["slice_value"] == "all"]
    cost = [{
        "configuration": r["configuration"], "topology": r["topology_level"], "cases": int(r["cases"]),
        "h0_cost_wape": float(r["h0_cost_wape"]), "dnn_cost_wape": float(r["dnn_cost_wape"]),
        "strict_improvement": r["strict_mape_and_wape_improvement"] == "True",
    } for r in cost_raw]
    placement_raw = [r for r in read_csv(base / "placement_method_comparison.csv") if r["slice_type"] == "overall" and r["slice_value"] == "all"]
    placement = [{
        "configuration": r["configuration"], "cases": int(r["cases"]),
        "h0_agreement_rate": float(r["h0_agreement_rate"]), "dnn_agreement_rate": float(r["dnn_agreement_rate"]),
        "h0_mean_teacher_regret": float(r["h0_mean_teacher_regret"]), "dnn_mean_teacher_regret": float(r["dnn_mean_teacher_regret"]),
        "weak_improvement": r["weak_agreement_and_regret_improvement"] == "True",
    } for r in placement_raw]
    wave_raw = [r for r in read_csv(base / "wave_sensitivity.csv") if r["role_method"] == "teacher"]
    wave = [{
        "configuration": r["configuration"], "mean_relative_cost_range": float(r["mean_relative_cost_range"]),
        "max_relative_cost_range": float(r["max_relative_cost_range"]), "placement_stability_rate": float(r["placement_stability_rate"]),
        "official_policy": r["official_policy"],
    } for r in wave_raw]
    return cost, placement, wave


def _svg_start(title: str, width: int = 1100, height: int = 560) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:24px;font-weight:700}.axis{font-size:13px}.label{font-size:12px}.note{font-size:12px;fill:#5f6b7a}.grid{stroke:#dfe5ec;stroke-width:1}.axisline{stroke:#344054;stroke-width:1.4}</style>',
        f'<text x="550" y="36" text-anchor="middle" class="title">{html.escape(title)}</text>',
    ]


def grouped_bar_svg(title: str, categories: list[str], series: list[tuple[str, str, list[float]]], ymax: float, threshold: float | None = None, note: str = "") -> str:
    width, height = 1100, 560
    left, right, top, bottom = 90, 30, 75, 110
    plot_w, plot_h = width-left-right, height-top-bottom
    out = _svg_start(title, width, height)
    for i in range(6):
        value = ymax * i / 5
        y = top + plot_h * (1-i/5)
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>', f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{value*100:.0f}%</text>']
    if threshold is not None:
        y = top + plot_h * (1-threshold/ymax)
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#d92d20" stroke-width="2" stroke-dasharray="7 5"/>', f'<text x="{left+plot_w-3}" y="{y-6:.1f}" text-anchor="end" class="note">{threshold*100:.0f}% reference</text>']
    group_w = plot_w / len(categories)
    gap, bar_w = 5, min(42, (group_w-20)/max(1,len(series)))
    for ci, category in enumerate(categories):
        center = left + group_w*(ci+0.5)
        total = len(series)*bar_w + (len(series)-1)*gap
        for si, (_name, color, values) in enumerate(series):
            value = values[ci]
            h = min(plot_h, plot_h*value/ymax)
            x = center-total/2+si*(bar_w+gap)
            y = top+plot_h-h
            out += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="2" fill="{color}"/>', f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" class="label">{value*100:.1f}</text>']
        out.append(f'<text x="{center:.1f}" y="{top+plot_h+23}" text-anchor="middle" class="axis">{html.escape(category)}</text>')
    out += [f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axisline"/>']
    lx = left
    for name, color, _values in series:
        out += [f'<rect x="{lx}" y="{height-52}" width="15" height="15" fill="{color}"/>', f'<text x="{lx+22}" y="{height-40}" class="axis">{html.escape(name)}</text>']
        lx += 190
    if note:
        out.append(f'<text x="{width-right}" y="{height-18}" text-anchor="end" class="note">{html.escape(note)}</text>')
    out.append('</svg>\n')
    return "\n".join(out)


def ladder_svg(rows: list[dict[str, Any]]) -> str:
    categories = [r["phase"] for r in rows]
    values = [r["overall_wape"] for r in rows]
    colors = ["#12b76a" if r["fresh_gate_pass"] else "#f79009" for r in rows]
    out = _svg_start("PD multi-flow evidence ladder (overall WAPE; gate status separate)")
    left, top, plot_w, plot_h = 90, 75, 980, 375
    ymax = 0.25
    for i in range(6):
        v=ymax*i/5; y=top+plot_h*(1-i/5)
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>', f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{v*100:.0f}%</text>']
    ty=top+plot_h*(1-0.10/ymax)
    out.append(f'<line x1="{left}" y1="{ty:.1f}" x2="{left+plot_w}" y2="{ty:.1f}" stroke="#d92d20" stroke-width="2" stroke-dasharray="7 5"/>')
    group=plot_w/len(rows)
    for i,(cat,val,color) in enumerate(zip(categories,values,colors)):
        x=left+group*(i+0.5)-30; h=min(plot_h,plot_h*val/ymax); y=top+plot_h-h
        out += [f'<rect x="{x:.1f}" y="{y:.1f}" width="60" height="{h:.1f}" rx="3" fill="{color}"/>', f'<text x="{x+30:.1f}" y="{y-6:.1f}" text-anchor="middle" class="label">{val*100:.2f}%</text>', f'<text x="{x+30:.1f}" y="{top+plot_h+23}" text-anchor="middle" class="axis">{cat}</text>']
    out += ['<rect x="90" y="505" width="15" height="15" fill="#12b76a"/><text x="112" y="517" class="axis">fresh/external gate pass</text>', '<rect x="310" y="505" width="15" height="15" fill="#f79009"/><text x="332" y="517" class="axis">gate fail retained</text>', '<text x="1070" y="540" text-anchor="end" class="note">Overall WAPE alone does not determine the full preregistered gate.</text>', '</svg>\n']
    return "\n".join(out)


def combined_phase71_svg(cost: list[dict[str, Any]], placement: list[dict[str, Any]]) -> str:
    configs=[]
    for row in cost:
        if row["configuration"] not in configs: configs.append(row["configuration"])
    avg_h0=[sum(r["h0_cost_wape"] for r in cost if r["configuration"]==c)/3 for c in configs]
    avg_dnn=[sum(r["dnn_cost_wape"] for r in cost if r["configuration"]==c)/3 for c in configs]
    agreement={r["configuration"]:(r["h0_agreement_rate"],r["dnn_agreement_rate"]) for r in placement}
    svg=grouped_bar_svg("Phase71 communication-only integration", configs, [("H0 cost WAPE", "#98a2b3", avg_h0),("H0+DNN cost WAPE", "#2e90fa", avg_dnn)], 0.035, None, "Top labels are cost WAPE; placement agreement is listed below.")
    lines=svg.splitlines()
    insert=[]
    for i,c in enumerate(configs):
        h0,dnn=agreement[c]; x=90+(980/len(configs))*(i+0.5)
        insert.append(f'<text x="{x:.1f}" y="493" text-anchor="middle" class="note">agree {h0*100:.1f}% → {dnn*100:.1f}%</text>')
    return "\n".join(lines[:-1]+insert+[lines[-1]]) + "\n"


def build_artifacts(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    sources=load_sources(root,spec)
    old_claim=json.loads((root/spec["pinned_inputs"][1]["path"]).read_text(encoding="utf-8"))
    claims=build_claim_scope(old_claim)
    evidence=build_evidence(spec,sources)
    gaps=[{"phase":p,"formal_result_in_current_git_tree":False,"policy":"excluded_from_evidence"} for p in spec["formal_result_gap_policy"]["phases"]]
    histogram=build_histogram_rows(sources)
    ladder=build_ladder_rows(sources)
    cost,placement,wave=build_phase71_rows(root)
    p53=(root/spec["pinned_inputs"][0]["path"]).read_text(encoding="utf-8").rstrip()
    guide = render_guide(p53,sources,claims)
    report = render_report(sources,claims)
    figures={
        "figures/pd_histogram_development_accuracy.svg": grouped_bar_svg("PD histogram development accuracy (not fresh blind)", ["P58 calls","P58 bytes","P59 calls","P59 bytes"], [("H0", "#98a2b3", [r["h0_wape"] for r in histogram]),("refined", "#7f56d9", [r["refined_wape"] for r in histogram])], 0.30, 0.15, "Both Phase58 and Phase59 target_met=false."),
        "figures/pd_multiflow_evidence_ladder.svg": ladder_svg(ladder),
        "figures/phase71_cost_and_placement.svg": combined_phase71_svg(cost,placement),
        "figures/phase71_wave_sensitivity.svg": grouped_bar_svg("Phase71 teacher wave sensitivity", [r["configuration"] for r in wave], [("max relative cost range", "#f79009", [r["max_relative_cost_range"] for r in wave]),("placement stability", "#12b76a", [r["placement_stability_rate"] for r in wave])], 1.8, None, "Wave policies are diagnostics; official policy remains bin_aligned."),
    }
    tables={
        "tables/evidence_index.csv":csv_text(evidence), "tables/formal_result_gaps.csv":csv_text(gaps),
        "tables/histogram_development_accuracy.csv":csv_text(histogram), "tables/multiflow_evidence_ladder.csv":csv_text(ladder),
        "tables/phase71_cost_summary.csv":csv_text(cost), "tables/phase71_placement_summary.csv":csv_text(placement), "tables/phase71_wave_summary.csv":csv_text(wave),
    }
    return {"sources":sources,"claims":claims,"evidence":evidence,"gaps":gaps,"tables":tables,"figures":figures,"guide":guide,"report":report}


def render_guide(previous: str, s: dict[str, dict[str, Any]], claims: dict[str, Any]) -> str:
    p59=s["Phase59"]["development_validation"]; p70=s["Phase70"]["metrics"]; p71=s["Phase71"]["headline"]
    supplement=f'''# PatternDemand实验结构总导引（截至Phase71）

> 本文件由Phase72生成。第一部分保留Phase53截至Phase52的正式总导引；第二部分只追加当前正式Git树中可审计的Phase58–71结果。Phase54–57没有在当前正式Git树中形成结果目录，因此不引用本地未跟踪资产补证据。

## 一页结论

- TP与PP：沿用Phase53冻结结论；六模型直方图预测、L1–L3物理曲线和communication-only placement链完整。
- 纯PD P1D1：沿用Phase50/51/52冻结结论；六模型H0+DNN相对H0改进、18条物理曲线和第一版placement链完整。
- PD直方图绝对精度：仍未达到严格目标。Phase59 development calls/bytes histogram WAPE分别为{p59['h0_plus_dnn_refined']['calls_histogram_wape']*100:.2f}%/{p59['h0_plus_dnn_refined']['bytes_histogram_wape']*100:.2f}%，`target_met=false`。
- PD多流：R61两流修正经Phase62 fresh-blind和Phase63六模型外部验证；四流经历两次保留失败后，R69在Phase70两个代表模型、四图、L1–L3上fresh-blind通过，overall WAPE={p70['phase69_overall_wape']*100:.3f}%。
- 代价与placement：Phase71在固定`bin_aligned`边际wave合同下，21/21 cost与7/7 placement比较均通过；DNN最大cost WAPE={p71['maximum_dnn_cost_wape']*100:.3f}%，最低placement agreement={p71['minimum_dnn_placement_agreement']*100:.1f}%。
- 关键限制：边际12-bin直方图不包含真实并发配对。敏感性中最大相对代价范围={p71['maximum_wave_policy_relative_cost_range']*100:.1f}%，最低placement稳定率={p71['minimum_wave_policy_placement_stability']*100:.1f}%。

## Phase58–71新增证据

1. **Phase58–59：精度探索，目标未达。** 两阶段均在development上改善H0，但没有通过逐模型、逐segment合同；不能写成fresh-blind达标。
2. **Phase60–63：两流链。** P1D2/P2D1实测揭示不能直接叠加单链路；R61轻量修正经reserved payload、新GPU/主机placement fresh-blind通过，并扩展到六模型物理证据。
3. **Phase64–70：四流链。** 零样本、第一版、第二版失败均保留；R69只对page>32增加轻量残差，第三次fresh-blind通过。有效范围严格限于两个代表模型、四种已测图、L1–L3和最多四流。
4. **Phase71：确定性集成。** 冻结直方图、曲线、R61/R69，在预注册边际wave下计算communication-only cost和placement。诊断wave不参与选优。

## 当前实验是否还需要GPU

在当前冻结范围内，没有必须重跑的B200 teacher或物理曲线实验。若未来扩到新模型、新transport、超过四流的新图，或要验证真实请求并发顺序，必须另立GPU合同；不能从Phase70/71自动外推。

## 下一研究边界

若继续PatternDemand预测器，优先在CPU上加入与H0+DNN真正不同的“低维画像→代表性完整工作负载→teacher直方图”baseline，并使用未打开标签的新盲测。若转向调度器，则另立合同加入计算时间、显存、资源空闲、排队拥塞、通信计算重叠和L1不可用时的受约束L2/L3选择。

## 结论边界索引

- 冻结可用结论：F01–F18。
- 禁止越界结论：N01–N15。
- 完整机器可读定义：`audit/claim_scope.json`。
'''
    return supplement+"\n---\n\n# 附录：Phase53原始总导引（截至Phase52，原文冻结）\n\n"+previous+"\n"


def render_report(s: dict[str, dict[str, Any]], claims: dict[str, Any]) -> str:
    p58=s["Phase58"]["development_validation"]; p59=s["Phase59"]["development_validation"]; p71=s["Phase71"]["headline"]
    return f'''# Phase72结论冻结报告

## 做了什么

只读核验Phase53与Phase58–71的正式result commit、status、manifest和固定摘要；生成新版导引、15行证据索引、18项可用结论、15项禁止越界声明及四张确定性SVG。没有GPU、网络、训练、预测、teacher、物理测量或scheduler仿真。

## 关键数字

- Phase58 refined calls/bytes histogram WAPE：{p58['h0_plus_dnn_refined']['calls_histogram_wape']*100:.2f}% / {p58['h0_plus_dnn_refined']['bytes_histogram_wape']*100:.2f}%，目标未达。
- Phase59 refined calls/bytes histogram WAPE：{p59['h0_plus_dnn_refined']['calls_histogram_wape']*100:.2f}% / {p59['h0_plus_dnn_refined']['bytes_histogram_wape']*100:.2f}%，目标仍未达。
- Phase70 R69 overall WAPE：{s['Phase70']['metrics']['phase69_overall_wape']*100:.3f}%，第三次四流fresh-blind通过；范围只含两个代表模型。
- Phase71：cost 21/21、placement 7/7；DNN最大cost WAPE {p71['maximum_dnn_cost_wape']*100:.3f}%，最低agreement {p71['minimum_dnn_placement_agreement']*100:.1f}%。

## 论文写法

可以写：预测直方图虽未达到统一严格绝对门，但H0+DNN相对H0稳定改善；在冻结曲线和预注册wave合同下，这种改善传递到communication-only cost与placement。必须同时写出：真实并发配对不可由边际直方图识别，四流修正只在Phase70测量范围内成立，完整scheduler尚未研究。

## 当前结论

TP/PP/P1D1 PD主链与当前多流物理扩展已完成一次可审计冻结。下一步不应重复既有GPU测量；应选择“CPU替代baseline”或“新scheduler合同”之一。
'''
