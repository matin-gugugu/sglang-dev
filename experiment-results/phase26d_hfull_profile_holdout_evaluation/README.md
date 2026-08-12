# Phase 26D：Hfull画像级留出评测

状态：**PASS**。

本阶段冻结Phase 26C四个checkpoint，在此前未用于拟合、标准化或早停的14个画像上做
正式测试：5个temporal、8个external和1个external synthetic。评测H0、structure-direct
和H0+bounded residual，并分别报告TP/PP、policy和phase。

## 配置级total核心结果

| 测试域 | 并行 | 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | common cost MAPE |
|---|---|---|---:|---:|---:|---:|---:|
| 全部测试画像 | TP | h0 | 11.60% / 10.75% | 3.50% / 1.27% | 0.1379 | 0.0142 | 7.10% |
| 全部测试画像 | TP | direct | 54.85% / 55.98% | 79.09% / 88.65% | 0.3738 | 0.0456 | 53.64% |
| 全部测试画像 | TP | h0_bounded_residual | 14.70% / 16.33% | 9.43% / 10.32% | 0.1407 | 0.0144 | 12.29% |
| 全部测试画像 | PP | h0 | 70.44% / 27.42% | 3.50% / 1.27% | 0.2854 | 0.0502 | 9.81% |
| 全部测试画像 | PP | direct | 61.97% / 59.38% | 73.85% / 86.87% | 0.5306 | 0.0836 | 54.22% |
| 全部测试画像 | PP | h0_bounded_residual | 58.24% / 27.77% | 5.23% / 3.70% | 0.2716 | 0.0465 | 10.32% |
| Temporal | TP | h0 | 18.30% / 13.58% | 7.83% / 4.93% | 0.2225 | 0.0235 | 14.41% |
| Temporal | TP | direct | 51.59% / 51.03% | 57.75% / 34.73% | 0.2478 | 0.0305 | 40.91% |
| Temporal | TP | h0_bounded_residual | 18.25% / 12.62% | 8.72% / 6.82% | 0.2201 | 0.0228 | 13.80% |
| Temporal | PP | h0 | 22.99% / 8.51% | 7.83% / 4.93% | 0.1766 | 0.0199 | 13.01% |
| Temporal | PP | direct | 56.77% / 66.83% | 44.11% / 11.41% | 0.4579 | 0.0533 | 39.86% |
| Temporal | PP | h0_bounded_residual | 23.97% / 16.64% | 7.87% / 4.07% | 0.1583 | 0.0184 | 15.99% |
| External | TP | h0 | 7.28% / 9.84% | 0.97% / 0.97% | 0.0813 | 0.0079 | 2.79% |
| External | TP | direct | 56.00% / 55.53% | 91.77% / 92.05% | 0.4448 | 0.0538 | 58.37% |
| External | TP | h0_bounded_residual | 12.09% / 16.76% | 8.72% / 9.65% | 0.0905 | 0.0090 | 10.66% |
| External | PP | h0 | 98.79% / 39.98% | 0.97% / 0.97% | 0.3544 | 0.0694 | 7.82% |
| External | PP | direct | 63.84% / 50.17% | 92.22% / 92.24% | 0.5734 | 0.1022 | 60.41% |
| External | PP | h0_bounded_residual | 80.04% / 36.67% | 3.38% / 3.36% | 0.3430 | 0.0646 | 6.94% |
| Synthetic | TP | h0 | 12.69% / 13.43% | 2.15% / 2.15% | 0.1678 | 0.0176 | 4.95% |
| Synthetic | TP | direct | 62.05% / 72.09% | 84.40% / 84.81% | 0.4360 | 0.0546 | 79.49% |
| Synthetic | TP | h0_bounded_residual | 17.85% / 20.61% | 18.66% / 18.62% | 0.1455 | 0.0159 | 17.72% |
| Synthetic | PP | h0 | 80.91% / 29.43% | 2.15% / 2.15% | 0.2769 | 0.0477 | 9.72% |
| Synthetic | PP | direct | 72.97% / 85.77% | 75.57% / 75.57% | 0.5519 | 0.0869 | 76.46% |
| Synthetic | PP | h0_bounded_residual | 55.09% / 19.28% | 6.81% / 6.81% | 0.2673 | 0.0421 | 9.06% |

Synthetic只有1个画像，保留为外部极端哨兵，不能单独支撑统计泛化结论。方法判断应重点看
Temporal、External及全部测试画像，并同时看calls、bytes、TV、EMD和cost，而不是只挑一个
改善数字。

## 首版候选决策

- TP：latency/balanced/throughput全部保留H0；bounded residual在Temporal近似持平，但在
  External的calls、bytes与cost显著退化；
- PP MB1：保留H0；全部测试画像calls MAPE 8.77%、common cost MAPE 3.03%，residual反而退化；
- PP MB4：把bounded residual作为待复验候选；calls MAPE 42.09%降到33.19%、TV 0.2253
  降到0.2167，但cost MAPE从9.93%小幅升到10.26%；
- PP MB16：把bounded residual作为待复验候选；calls MAPE 160.46%降到125.36%、TV
  0.5298降到0.4797、cost MAPE 16.47%降到13.65%。绝对误差仍很高，不能视为已解决；
- Direct在TP/PP都拒绝。

上述TP/PP策略选择是在本轮holdout结果后形成的首版候选，不能再把同一批测试画像上的组合
成绩当作无偏测试分数；必须用新增画像复验。PP MB4/MB16的绝对calls误差表明下一阶段应增强
compact画像中的scheduler-sensitive离散统计，而不是更换Hfull teacher。

## 口径

- calls/bytes均按每1000请求归一化；
- L1/TV使用各自原生12桶，total保留prefill/decode的24维phase-aware分布；
- normalized log-payload EMD在total时合并phase后计算payload质量迁移；
- common cost使用5 μs启动项+100 GB/s参数曲线，不是PP物理链路测量；
- TP与PP的桶schema继续分离。

## 资产

- `analysis/test_predictions.csv.gz`：逐配置、逐phase/total预测；
- `analysis/test_metrics.csv`：按测试域、并行、方法、phase和policy聚合；
- `analysis/residual_vs_h0.csv`：bounded residual相对H0的逐指标变化；
- `analysis/first_version_candidate.csv`：透明规则生成的策略级首版候选；
- `figures/test_domain_calls_mape.png`：三测试域TP/PP calls MAPE；
- `contract.json`、`summary.json`、`audit_summary.json`、`logs/evaluation.log`、`DONE`
  和`manifest.sha256`。

可以据此判断冻结模型在这些画像域的实际泛化表现。不能外推到online arrival-aware、其他
scheduler契约或未观测模型结构，也不能把Synthetic单画像结果解释为总体分布。
