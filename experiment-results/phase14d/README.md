# Phase 14D：无 backend 的 TP×phase 条件斜率验证

更新时间：2026-08-05

状态：smoke、正式分析、确定性复验和归档均已完成；模型未达到收敛门槛

## 1. 目标

Phase 14C 证明只给 TP、Prefill/Decode 增加固定截距不足以覆盖新的 Decode 形态。
Phase 14D 不新增 GPU 采集，复用 Phase 14C 的 162 个聚合配置，检验是否允许
PatternDemand 系数随 TP、phase 或 TP×phase 改变即可降低残差。

整个阶段不使用实际 backend、kernel name、模型身份或 Phase 2 cost curve。backend
signature 只保留在预测结果中做事后残差诊断。

## 2. 数据与验证设计

| 项目 | 设置 |
|---|---|
| 硬件来源 | 单节点 8×B200 |
| 模型 | Qwen3-8B、Qwen3-30B-A3B、DeepSeek-V2-Lite |
| TP | 2、4、8 |
| phase | Prefill、Decode |
| 聚合配置 | 162 |
| TP-linked workload | 54 |
| 外层验证 | 6 折完整 workload 留出 |
| alpha 选择 | 每个外层训练集内部 nested CV |
| 跨模型验证 | leave-one-model-out |

同一 workload 的 TP2/4/8 始终处于同一折。五组候选共享相同数据切分和 ridge alpha
集合，避免由切分差异造成虚假提升。

## 3. 候选模型

| 候选 | 特征数 | 含义 |
|---|---:|---|
| additive TP×phase | 34 | Phase 14C 基线，PatternDemand 斜率固定 |
| ring-equivalent PatternDemand | 30 | 每次 AllReduce 映射为 `2×(TP-1)` 轮、每轮 `payload/TP` |
| TP-conditioned slopes | 92 | PatternDemand 斜率可随 TP2/4/8 改变 |
| phase-conditioned slopes | 63 | PatternDemand 斜率可随 Prefill/Decode 改变 |
| TP×phase-conditioned slopes | 179 | 共享基线加完整 TP×phase 交互斜率 |

完整交互不是六个互不相关的独立拟合，而是共享基线加 ridge 正则化交互；但相对 162
个样本仍然自由度过高。

## 4. workload 留出结果

| 方法 | MAPE | P95 APE | R² |
|---|---:|---:|---:|
| additive TP×phase | 12.805% | 39.366% | 0.8961 |
| ring-equivalent PatternDemand | 25.154% | 77.843% | 0.2996 |
| TP-conditioned slopes | **12.150%** | **39.013%** | 0.8953 |
| phase-conditioned slopes | 22.317% | 59.070% | 0.6764 |
| TP×phase-conditioned slopes | 22.113% | 59.611% | 0.6556 |

TP-conditioned slopes 是共同外层 CV 结果中的描述性最优候选，MAPE 相对 additive
基线只下降约 5.1%，P95 只下降约 0.9%。候选族本身没有再嵌套选择，因此 12.150%
不能解释为无偏的生产模型得分。

完整 TP×phase 交互的 MAPE 增至 22.113%，Decode MAPE 为 35.065%，说明 179 维
斜率在当前数据量下明显过拟合。简单 ring-equivalent 假设也不能解释真实时间。

## 5. 描述性最优候选的分组结果

TP-conditioned slopes：

| 范围 | MAPE | P95 APE |
|---|---:|---:|
| Prefill | 9.433% | 33.299% |
| Decode | 17.583% | 69.643% |
| TP2 | 13.371% | 40.720% |
| TP4 | 11.108% | 30.469% |
| TP8 | 11.971% | 34.344% |

Decode profile 中，balanced 和 bimodal 的 MAPE 分别降至 2.820% 和 5.171%，但
`uniform_b16` 仍为 30.258%，`uniform_b4` 仍为 37.479%。新的斜率主要改善熟悉
形态，没有解决最困难的新形态。

leave-one-model-out MAPE：

| 留出模型 | MAPE | P95 APE |
|---|---:|---:|
| Qwen3-8B | 47.337% | 80.225% |
| Qwen3-30B-A3B | 19.353% | 41.032% |
| DeepSeek-V2-Lite | 17.921% | 46.128% |

## 6. 收敛判断

描述性最优候选只通过 Prefill MAPE <15% 门槛，以下门槛均未通过：

- 整体 MAPE <10%；
- 整体 P95 APE <25%；
- Decode MAPE <10%；
- 每个留出模型 MAPE <15%。

因此 Phase 14D 的实验执行已收敛，但“只靠 TP、Prefill/Decode 调整 PatternDemand
斜率即可得到稳定时间模型”的假设未收敛。继续增加同类 TP 截距或交互项不是优先方向。

## 7. 下一步

TP 数据覆盖可暂时冻结在三模型、TP2/4/8。下一步若继续，应优先补充运行前可推导、
但当前直方图丢失的时序结构，例如：

1. 每个 Decode step 的 active-batch 序列；
2. collective 消息大小的顺序、切换次数和 run length；
3. drain 速度、step 数和并发宽度统计；
4. 在外层训练集内部同时选择模型族与 ridge alpha。

这些仍不需要读取实际 backend。Kimi-K2.5 可在结构模型稳定后作为新的整模型外部验证，
不建议现在用来扩大同一个未收敛模型的训练集。

## 8. 正式产物

```text
experiment-results/phase14d/
├── README.md
├── audit_summary.json
├── manifest.sha256
├── smoke/
│   └── smoke_summary.json
├── tp_phase_interaction_analysis/
│   ├── README.md
│   ├── alpha_selection.csv
│   ├── fold_assignments.csv
│   ├── metrics.csv
│   ├── predictions.csv
│   ├── posthoc_backend_diagnostics.csv
│   ├── summary.json
│   └── tp_phase_interaction_analysis.png
├── smoke.log
├── runner.log
├── revalidate_smoke.log
└── revalidate_phase14d.log
```

首次后台正式分析耗时约 38 秒，只使用 CPU。正式分析和 smoke 的独立重算均逐字节
一致；根 manifest 覆盖除自身以外的全部 Phase 14D 文件。

## 9. 外推限制

结论只覆盖当前单节点 B200、三个模型、TP2/4/8 与六类 Decode/两类 chunked
Prefill 形态。不能外推到跨节点、PP、PD、expert-parallel All-to-All、其他 GPU，
也不能证明未知模型或未知 Decode 调度上的误差可控。
