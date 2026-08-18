#!/usr/bin/env python3
"""Build the Phase53 evidence index, frozen claims and canonical Chinese reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def load_source_summaries(root: Path, spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    import json

    return {
        item["phase"]: json.loads(
            (root / "experiment-results" / item["directory"] / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        for item in spec["source_results"]
    }


def _phase_facts(s: dict[str, dict[str, Any]]) -> dict[str, str]:
    p34 = s["Phase34D"]
    p35 = s["Phase35"]
    p36 = s["Phase36"]
    p38 = s["Phase38"]
    p39 = s["Phase39"]
    p40 = s["Phase40"]
    p41 = s["Phase41"]
    p42 = s["Phase42"]
    p43 = s["Phase43"]
    p44 = s["Phase44"]
    p45 = s["Phase45"]
    p46 = s["Phase46"]
    p47 = s["Phase47"]
    p48 = s["Phase48"]
    p49 = s["Phase49"]
    p50 = s["Phase50"]
    p51 = s["Phase51"]
    p52 = s["Phase52"]
    p38_head = p38["physical_cost_headline"][0]
    return {
        "Phase34D": (
            f"12个fresh blind画像、{p34['counts']['blind_full_teacher_requests']}个完整请求；"
            "六模型TP/PP的H0+DNN residual均正式通过"
        ),
        "Phase35": (
            f"统一复播{p35['counts']['prediction_phase_rows']}条phase预测且与冻结预测零差异；"
            "当时仅TP L1为物理曲线，其余为proxy"
        ),
        "Phase36": (
            f"跨环境复播{p36['counts']['prediction_rows']}条预测，teacher/target未读取，差异为零"
        ),
        "Phase37": "在NVLINK_NV18单机类别上完成首条PP tensor-only物理P2P曲线，拓扑覆盖有限",
        "Phase38": (
            f"PP单机物理曲线卷积：cost WAPE={pct(p38_head['cost_wape'])}；"
            "是已打开Phase34 target上的重复工程"
        ),
        "Phase39": (
            f"{p39['counts']['physical_curves']}条TP/PP L1-L3物理曲线；"
            f"{p39['counts']['placement_decision_rows']}个communication-only决策，top1={pct(p39['overall_top1_agreement'])}"
        ),
        "Phase40": (
            f"Qwen3-8B的{p40['counts']['exact_requests']}个请求、"
            f"{p40['counts']['gpu_logical_chunks']}个逻辑chunk与scheduler-faithful teacher精确一致"
        ),
        "Phase41": (
            f"{p41['counts']['gpu_sentinel_requests']}请求/{p41['counts']['gpu_sentinel_waves']} waves GPU sentinel精确；"
            f"生成{p41['counts']['development_profiles']}个开发画像"
        ),
        "Phase42": (
            f"以{p42['counts']['development_train_profiles']}/{p42['counts']['development_validation_profiles']}个画像完成首轮训练并先冻结12个blind画像预测"
        ),
        "Phase43": (
            f"12画像pilot fresh blind：composite ratio={p43['blind_metrics']['composite_ratio']:.4f}>1，"
            "DNN未优于H0，作为正式负结果保留"
        ),
        "Phase44": (
            f"扩为{p44['counts']['profiles']}个互斥开发画像，"
            f"{p44['counts']['train_profiles']}/{p44['counts']['validation_profiles']}训练验证并加入H0保护门"
        ),
        "Phase45": (
            f"在target打开前冻结{p45['counts']['blind_profiles']}个Qwen3 fresh-blind画像的"
            f"{p45['counts']['frozen_prediction_rows']}行预测"
        ),
        "Phase46": (
            f"Qwen3 300画像fresh blind：composite ratio={p46['blind_metrics']['composite_ratio']:.4f}<1，"
            "四项直方图指标严格优于H0"
        ),
        "Phase47": (
            f"补齐{p47['counts']['models']}个模型、{p47['counts']['exact_requests']}个请求的GPU teacher精确验证；"
            "与Phase40的Qwen3-8B合计六模型"
        ),
        "Phase48": (
            f"{p48['counts']['profiles']}画像×{p48['counts']['models']}模型，"
            f"{p48['counts']['example_rows']}个共享残差训练样本"
        ),
        "Phase49": (
            f"在target打开前冻结{p49['counts']['blind_profiles']}画像×{p49['counts']['models']}模型的"
            f"{p49['counts']['frozen_prediction_rows']}行预测"
        ),
        "Phase50": (
            f"{p50['counts']['blind_units']}个六模型fresh-blind单元；"
            f"overall composite ratio={p50['blind_metrics']['composite_ratio']:.4f}，六模型均过四指标保护门"
        ),
        "Phase51": (
            f"{p51['counts']['models']}模型×{p51['counts']['topology_levels']}拓扑，"
            f"{p51['counts']['physical_curves']}条Mooncake/RDMA物理曲线、{p51['counts']['curve_knots']}个knots"
        ),
        "Phase52": (
            f"{p52['counts']['unit_topology_cost_rows']}条物理cost与{p52['counts']['placement_decision_rows']}个决策；"
            "L1/L2/L3 cost误差和placement均确认H0+DNN改善"
        ),
    }


def build_evidence_rows(
    spec: dict[str, Any], summaries: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    facts = _phase_facts(summaries)
    current_use = {
        "Phase34D": "TP/PP六模型预测器的正式fresh-blind结论",
        "Phase35": "保留统一接口、零差异复播和proxy到物理曲线的演进记录",
        "Phase36": "证明冻结预测可跨环境复现并按W/R合同回传",
        "Phase37": "PP单机物理测量先导；最终L1-L3库由Phase39承接",
        "Phase38": "PP物理卷积先导；最终TP/PP L1-L3结论由Phase39承接",
        "Phase39": "TP/PP固定配置下L1-L3物理cost和communication-only placement的当前结论",
        "Phase40": "纯PD Qwen3 teacher语义锚点",
        "Phase41": "纯PD完整窗口构造、wave边界和首轮开发数据",
        "Phase42": "首轮小数据残差与预测先冻结流程",
        "Phase43": "不可删除的负结果，直接推动Phase44保护扩容",
        "Phase44": "Qwen3保护训练的扩大开发协议",
        "Phase45": "Qwen3 300画像blind的预测先验",
        "Phase46": "Qwen3保护残差的正式fresh-blind确认",
        "Phase47": "其余五模型teacher语义锚点",
        "Phase48": "六模型共享保护残差训练",
        "Phase49": "六模型300画像blind的预测先验",
        "Phase50": "纯PD六模型预测器的正式fresh-blind结论",
        "Phase51": "纯PD六模型L1-L3物理通信曲线库",
        "Phase52": "纯PD物理cost和communication-only placement的当前结论",
    }
    boundaries = {
        "Phase34D": "六个已知模型和BurstGPT fresh windows；不证明未见第七模型",
        "Phase35": "除TP L1外的曲线为proxy，不能当作物理实测",
        "Phase36": "只证明复播，不是新的精度盲测",
        "Phase37": "只覆盖NVLINK_NV18单机tensor-only类别",
        "Phase38": "不含L2/L3、metadata、计算、显存或scheduler",
        "Phase39": "target已打开；只在冻结TP/PP配置和冻结placement上做communication-only判断",
        "Phase40": "代表请求的GPU语义核验，不是大规模训练或物理延迟测量",
        "Phase41": "Qwen3开发数据；完整请求和raw保持Git外",
        "Phase42": "执行PASS不等于DNN科学结论为正",
        "Phase43": "样本仅12画像，但负结论有效且没有被删除",
        "Phase44": "只用开发/验证目标选模型，不得读取后续blind target",
        "Phase45": "无Hfull；必须等结果合入后才能打开target",
        "Phase46": "只证明Qwen3，尚不证明其他模型和物理时间",
        "Phase47": "证明teacher语义，不证明预测精度或物理代价",
        "Phase48": "六个已知模型共享训练，不证明unseen-model generalization",
        "Phase49": "无Hfull；必须等结果合入后才能打开target",
        "Phase50": "六个已知模型、300个BurstGPT画像；不证明线上arrival-aware",
        "Phase51": "冻结端点/布局/payload support内的物理传输，不是端到端服务延迟",
        "Phase52": "bin-mean确定性卷积且target已打开；不是新盲测或完整scheduler",
    }
    rows = []
    for order, item in enumerate(spec["source_results"], start=1):
        rows.append(
            {
                "order": order,
                "phase": item["phase"],
                "chain": item["chain"],
                "result_commit": item["result_commit"],
                "status": summaries[item["phase"]]["status"],
                "evidence_class": item["evidence_class"],
                "role": item["role"],
                "key_fact": facts[item["phase"]],
                "current_use": current_use[item["phase"]],
                "evidence_boundary": boundaries[item["phase"]],
            }
        )
    return rows


def build_chain_rows(s: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    p39 = s["Phase39"]
    p50 = s["Phase50"]
    p51 = s["Phase51"]
    p52 = s["Phase52"]
    place = p52["placement_headline"]
    return [
        {
            "chain": "TP",
            "predictor": "Phase34D六模型H0+DNN residual fresh-blind PASS",
            "teacher": "scheduler-faithful Hfull；完整请求仅离线生成标签",
            "physical_curves": "Phase39 TP2/4/8 × L1/L2/L3，共9条物理曲线",
            "cost": "Phase39 WAPE：L1 7.57%，L2 7.52%，L3 7.85%",
            "placement": f"Phase39 communication-only top1={pct(p39['overall_top1_agreement'])}，regret=0",
            "status": "FROZEN_COMPLETE_WITHIN_SCOPE",
            "boundary": "固定TP/PP配置；target已打开；不含计算、显存、资源和重叠",
        },
        {
            "chain": "PP",
            "predictor": "Phase34D六模型H0+DNN residual fresh-blind PASS",
            "teacher": "scheduler-faithful Hfull；完整请求仅离线生成标签",
            "physical_curves": "Phase39 PP × L1/L2/L3，共3条物理曲线",
            "cost": "Phase39 WAPE：L1 4.41%，L2 3.99%，L3 4.22%",
            "placement": f"Phase39 communication-only top1={pct(p39['overall_top1_agreement'])}，regret=0",
            "status": "FROZEN_COMPLETE_WITHIN_SCOPE",
            "boundary": "固定TP/PP配置；target已打开；不含计算、显存、资源和重叠",
        },
        {
            "chain": "PD",
            "predictor": (
                f"Phase50六模型×{p50['counts']['blind_profiles']}画像H0+DNN residual fresh-blind PASS，"
                f"composite ratio={p50['blind_metrics']['composite_ratio']:.4f}"
            ),
            "teacher": "Phase40/47 GPU精确验证的纯P1-D1 fixed-draining Hfull teacher",
            "physical_curves": (
                f"Phase51 {p51['counts']['physical_curves']}条六模型L1/L2/L3 Mooncake/RDMA物理曲线"
            ),
            "cost": "Phase52 H0+DNN cost WAPE：L1 2.15%，L2 2.16%，L3 2.15%，三层均优于H0",
            "placement": (
                f"Phase52 agreement {pct(place['h0']['agreement_rate'])}→"
                f"{pct(place['h0_plus_dnn_residual']['agreement_rate'])}；mean regret "
                f"{pct(place['h0']['mean_teacher_regret'])}→"
                f"{pct(place['h0_plus_dnn_residual']['mean_teacher_regret'])}"
            ),
            "status": "FROZEN_COMPLETE_WITHIN_SCOPE",
            "boundary": "纯P1-D1、fixed-draining；bin-mean卷积；不含在线到达、实例数或完整scheduler",
        },
    ]


def build_claim_scope(spec: dict[str, Any]) -> dict[str, Any]:
    frozen_claims = [
        {"id": "F01", "claim": "最终预测输入是常态历史流量的低维画像、模型结构、固定执行策略和固定并行配置，不含完整请求列表。", "evidence": "Phase26-35与Phase40-50合同"},
        {"id": "F02", "claim": "预测目标是在fixed-draining语义下的拓扑无关12-bin消息调用数和逻辑字节直方图。", "evidence": "Phase34D、Phase40/47、Phase50"},
        {"id": "F03", "claim": "TP/PP size与纯PD的P1/D1均是预测器输入；当前placement模块只选冻结的L1/L2/L3，不选并行度。", "evidence": "Phase35、Phase39、Phase52"},
        {"id": "F04", "claim": "Hfull是经过代表性GPU实验验证的scheduler-faithful离线teacher；完整请求只用于离线标签。", "evidence": "TP/PP teacher链与Phase40/41/47"},
        {"id": "F05", "claim": "TP在六个已知模型的fresh-blind集合上保留H0+DNN residual并正式优于H0。", "evidence": "Phase34D"},
        {"id": "F06", "claim": "PP在六个已知模型的fresh-blind集合上保留H0+DNN residual并正式优于H0。", "evidence": "Phase34D"},
        {"id": "F07", "claim": "纯PD六模型teacher的请求级、聚合级和12-bin语义已与GPU sender-side事件精确对齐。", "evidence": "Phase40、Phase47"},
        {"id": "F08", "claim": "纯PD在300画像×六模型fresh-blind集合上，H0+DNN residual的四项主直方图指标均严格优于H0。", "evidence": "Phase49/50"},
        {"id": "F09", "claim": "TP2/4/8和PP的L1/L2/L3物理通信曲线已在冻结环境中补全，可用于Phase34直方图的确定性代价卷积。", "evidence": "Phase39"},
        {"id": "F10", "claim": "纯PD六模型L1/L2/L3 Mooncake/RDMA物理曲线库已完成，共18条曲线和396个knots。", "evidence": "Phase51"},
        {"id": "F11", "claim": "在固定并行配置与冻结候选placement内，TP/PP和PD均完成了communication-only cost与placement验证。", "evidence": "Phase39、Phase52"},
        {"id": "F12", "claim": "Phase43的小样本负结果没有被删除；Phase44-50通过扩大互斥开发/盲测和H0保护门后才形成最终正结论。", "evidence": "Phase43-50"},
    ]
    prohibited_claims = [
        {"id": "N01", "claim": "不能宣称对未见第七模型或任意新模型泛化。"},
        {"id": "N02", "claim": "不能宣称覆盖所有流量分布、所有policy或任意线上工作负载。"},
        {"id": "N03", "claim": "12-bin卷积不能恢复bin内每条消息的精确物理代价。"},
        {"id": "N04", "claim": "不能把通信代价写成端到端请求延迟或吞吐收益。"},
        {"id": "N05", "claim": "当前placement没有处理计算时间和显存可行性。"},
        {"id": "N06", "claim": "当前placement没有处理资源空闲、排队、拥塞或通信计算重叠。"},
        {"id": "N07", "claim": "fixed-draining结果不证明online arrival-aware调度。"},
        {"id": "N08", "claim": "Phase39/52不是完整scheduler验证，也不是线上收益实验。"},
        {"id": "N09", "claim": "纯PD结果不包含P或D内部TP/PP，也不证明混合并行PD。"},
        {"id": "N10", "claim": "调度器尚不能选择TP/PP size、P/D实例数或扩缩容策略。"},
    ]
    return {
        "schema_version": "phase53-claim-scope-v1",
        "frozen_claims": frozen_claims,
        "prohibited_claims": prohibited_claims,
        "future_scheduler_dimensions": list(spec["future_scheduler_dimensions"]),
        "governance": {
            "predictor_status": "freeze current TP/PP/PD predictors; no retuning on opened targets",
            "curve_status": "freeze Phase39 TP/PP and Phase51 PD physical curves as environment-specific evidence",
            "next_research_boundary": "a new scheduler protocol must add explicit system constraints without rewriting predictor evidence",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    safe = lambda value: str(value).replace("|", "／").replace("\n", " ")
    lines = ["|" + "|".join(map(safe, headers)) + "|", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("|" + "|".join(safe(value) for value in row) + "|" for row in rows)
    return "\n".join(lines)


def render_guide(
    summaries: dict[str, dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    chain_rows: list[dict[str, Any]],
    claims: dict[str, Any],
) -> str:
    p34 = summaries["Phase34D"]
    p39 = summaries["Phase39"]
    p43 = summaries["Phase43"]
    p46 = summaries["Phase46"]
    p50 = summaries["Phase50"]
    p51 = summaries["Phase51"]
    p52 = summaries["Phase52"]
    p52_cost = {row["topology_level"]: row for row in p52["cost_headline"]}
    p52_place = p52["placement_headline"]
    phase_table = _markdown_table(
        ["链", "预测器", "物理曲线", "placement", "冻结状态"],
        [[r["chain"], r["predictor"], r["physical_curves"], r["placement"], r["status"]] for r in chain_rows],
    )
    physical_rows = []
    for row in p39["cost_headline"]:
        physical_rows.append(
            [row["parallelism"].upper(), row["topology_level"], pct(row["cost_wape"]), pct(row["cost_mape"]), "Phase39物理实测"]
        )
    for level in ("L1", "L2", "L3"):
        row = p52_cost[level]
        physical_rows.append(["PD", level, pct(row["dnn_cost_wape"]), pct(row["dnn_cost_mape"]), "Phase51曲线×Phase50冻结直方图"])
    physical_table = _markdown_table(["链", "拓扑", "H0+DNN cost WAPE", "H0+DNN cost MAPE", "证据"], physical_rows)
    evidence_table = _markdown_table(
        ["阶段", "链", "类别", "当前作用", "边界"],
        [[r["phase"], r["chain"], r["evidence_class"], r["current_use"], r["evidence_boundary"]] for r in evidence_rows],
    )
    frozen_list = "\n".join(f"- `{row['id']}` {row['claim']}（{row['evidence']}）" for row in claims["frozen_claims"])
    prohibited_list = "\n".join(f"- `{row['id']}` {row['claim']}" for row in claims["prohibited_claims"])
    return f"""# PatternDemand实验结构总导引：截至Phase52

## 1. 当前正式结论

截至Phase52，TP、PP与纯PD三条链均已在当前研究边界内闭环：低维历史画像、模型结构、固定执行策略和固定并行配置进入预测器，输出fixed-draining语义下拓扑无关的12-bin消息调用数与逻辑字节直方图；随后将直方图代入冻结的L1/L2/L3物理通信曲线，得到communication-only通信代价和placement判断。

{phase_table}

这里的“完成”只指当前PatternDemand通信预测问题。它不等于完整调度器已经完成。

## 2. 研究对象和三个容易混淆的量

1. `H0`：只使用结构和低维画像构成的可解释基线。
2. `H0+DNN residual`：DNN只学习H0剩余的误差，并受H0保护；TP、PP、PD最终都保留这个形式。
3. `Hfull`：把完整请求窗口交给经过GPU验证的scheduler-faithful teacher离线生成的标签。完整请求列表不进入最终预测器，也不进入Git结果。

预测器预测的是消息需求，不是延迟。物理曲线负责把每个消息bin映射成通信时间；scheduler层才需要把通信和计算、显存、资源、排队、拥塞与重叠联合起来。

## 3. TP链

- 早期阶段建立TP scheduler-faithful teacher、结构公式和12-bin标签；Phase33完成原三模型的新盲测裁定。
- Phase34扩为六个已知模型和12个fresh request-disjoint BurstGPT画像，共{p34['counts']['blind_full_teacher_requests']}个完整teacher请求。预测在target打开前冻结；TP `H0+DNN residual`正式通过。
- Phase35统一推理复播零差异，但当时只有TP单机B200 L1为物理曲线，TP L2/L3仍是proxy。这些proxy不再作为最终物理证据。
- Phase36证明冻结预测能在另一GPU环境零差异复播。
- Phase39补齐TP2/4/8×L1/L2/L3九条物理曲线，并在648个固定TP配置case上进行代价与placement重算。

Phase39 TP total cost WAPE为：L1 7.57%、L2 7.52%、L3 7.85%。这些是Phase34 target已打开后的repeated-engineering物理代价，不是新的盲测。

## 4. PP链

- Phase33完成原三模型裁定；Phase34扩到相同六个已知模型和12个fresh blind画像，PP `H0+DNN residual`正式通过。
- Phase35的PP L1/L2/L3均为参数化proxy，只验证接口和敏感性。
- Phase37在可用机器的NVLINK_NV18类别上得到首条单机PP P2P物理曲线；这是有限拓扑先导。
- Phase38将Phase34冻结PP直方图代入该曲线，total cost WAPE为4.48%。
- Phase39最终补齐PP L1/L2/L3三条冻结物理曲线。PP total cost WAPE分别为4.41%、3.99%、4.22%。

Phase39中TP/PP合计{p39['counts']['placement_decision_rows']}个communication-only决策，top1 agreement={pct(p39['overall_top1_agreement'])}，mean regret=0。这个结果只说明在冻结候选和通信代价占优关系下预测与teacher选择一致，不包含真实调度约束。

## 5. 纯PD链

- 研究配置是纯P1→D1，P和D内部不包含TP/PP。Phase40以Qwen3-8B验证sender-side Mooncake语义、fixed-draining、原子wave放行、page/chunk预算和teacher精确一致。
- Phase41先以4853请求、82 waves的真实完整窗口GPU sentinel验证全窗口teacher，再生成94个开发画像。
- Phase42冻结首轮小数据DNN预测；Phase43随后才打开12个blind画像target。结果composite ratio={p43['blind_metrics']['composite_ratio']:.4f}，DNN不如H0。这是正式负结果，不能删除。
- Phase44将互斥开发集扩到1200画像并加入四指标H0保护；Phase45先冻结300个新blind画像预测，Phase46再打开target。Qwen3 composite ratio={p46['blind_metrics']['composite_ratio']:.4f}，四项指标均严格改善。
- Phase47对DeepSeek、Qwen3-30B、Llama、Qwen2.5和Mixtral补做GPU teacher精确验证；与Qwen3-8B组成六模型。
- Phase48在1200画像×六模型上训练共享保护残差；Phase49先冻结300画像×六模型预测；Phase50再一次性打开1800个画像-模型target单元。overall composite ratio={p50['blind_metrics']['composite_ratio']:.4f}，六模型与三segment均过保护门。
- Phase51以SGLang生产Mooncake/RDMA路径完成{p51['counts']['physical_curves']}条L1/L2/L3模型相关物理曲线、{p51['counts']['curve_knots']}个knots。
- Phase52冻结Phase49/50/51，做12-bin平均payload卷积。H0+DNN的cost WAPE为L1 {pct(p52_cost['L1']['dnn_cost_wape'])}、L2 {pct(p52_cost['L2']['dnn_cost_wape'])}、L3 {pct(p52_cost['L3']['dnn_cost_wape'])}，三层均严格优于H0。

Phase52 placement agreement从{pct(p52_place['h0']['agreement_rate'])}提升到{pct(p52_place['h0_plus_dnn_residual']['agreement_rate'])}；mean regret从{pct(p52_place['h0']['mean_teacher_regret'])}降到{pct(p52_place['h0_plus_dnn_residual']['mean_teacher_regret'])}。这是bin-mean、communication-only重复工程结果。

## 6. 当前物理代价总表

{physical_table}

TP/PP数值来自Phase39冻结物理曲线；PD数值来自Phase51曲线与Phase50六模型blind直方图在Phase52的确定性卷积。不同链的primitive、布局和目标不同，不能把数值横向解释为谁的端到端系统更快。

## 7. 正式证据演进索引

{evidence_table}

## 8. 冻结的可宣称结论

{frozen_list}

## 9. 明确禁止的越界结论

{prohibited_list}

## 10. 下一研究边界

当前预测器和物理曲线先冻结。下一研究主题属于scheduler层，至少要增加：计算时间、显存可行性、资源是否空闲、排队和拥塞、通信计算重叠、以及L1不可用时真正受约束的L2/L3选择。新阶段不能回到已经打开的Phase34或Phase50 target上继续调参并称作新盲测，也不能修改本导引冻结的teacher和直方图语义。
"""


def render_freeze_report(
    workflow_commit: str,
    evidence_rows: list[dict[str, Any]],
    chain_rows: list[dict[str, Any]],
    claims: dict[str, Any],
) -> str:
    chain_table = _markdown_table(
        ["链", "预测器", "teacher", "曲线", "代价", "placement", "边界"],
        [[r["chain"], r["predictor"], r["teacher"], r["physical_curves"], r["cost"], r["placement"], r["boundary"]] for r in chain_rows],
    )
    claim_table = _markdown_table(
        ["ID", "冻结结论", "正式证据"],
        [[r["id"], r["claim"], r["evidence"]] for r in claims["frozen_claims"]],
    )
    prohibited = "\n".join(f"- `{row['id']}` {row['claim']}" for row in claims["prohibited_claims"])
    phase_commits = "\n".join(
        f"- {row['phase']}：`{row['result_commit']}`，状态 `{row['status']}`，{row['role']}。"
        for row in evidence_rows
    )
    return f"""# Phase53：TP、PP、PD实验链与当前结论冻结报告

## 1. Phase53做了什么

Phase53没有训练、推理、teacher重算、GPU通信测量或scheduler仿真。它在workflow commit `{workflow_commit}` 上核验Phase34D至Phase52共{len(evidence_rows)}个正式结果目录的manifest、result commit和状态，并把截至Phase52的三条实验链写成一个新的规范入口。

它解决的是“以后引用哪一个结果、能说到哪里”的问题，不产生新的科学样本。

## 2. 三条链的冻结状态

{chain_table}

## 3. 冻结结论

{claim_table}

## 4. 证据层级与替代关系

1. 新盲测预测结论：TP/PP以Phase34D为准；纯PD六模型以Phase50为准。
2. 物理曲线：TP/PP以Phase39为准；纯PD以Phase51为准。
3. communication-only cost/placement：TP/PP以Phase39为准；纯PD以Phase52为准。
4. Phase35的TP L2/L3与PP L1/L2/L3 proxy只保留为接口演进；不得覆盖Phase39物理结果。
5. Phase37/38是PP单机先导，Phase39给出最终冻结的TP/PP L1-L3矩阵。
6. Phase43是有效负结果；Phase46和Phase50是采用新开发/新blind协议后的后续正结果，不是删除或重算Phase43。

## 5. 禁止越界

{prohibited}

## 6. 正式来源提交

{phase_commits}

## 7. 后续治理

- 当前TP、PP、PD预测器停止在已打开target上继续调参。
- 当前物理曲线保留环境、primitive、布局、payload和placement限定。
- 新scheduler研究必须把计算、显存、资源、排队/拥塞、重叠和受约束L2/L3决策写入新合同。
- 新研究可以消费冻结直方图与曲线，但不能把scheduler收益倒写成预测器的新盲测结论。
"""
