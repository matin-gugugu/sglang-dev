# Phase 29D3：TP第二独立确认与最终冻结结论

状态：**PASS**。本阶段没有训练、早停、调参或改写预测，只把Phase 29C已冻结
的3,888条第二确认预测，与Phase 29D2在预测/映射冻结后才生成的972条Hfull真值精确连接。

## 第二独立确认的四方法total结果

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| h0 | 12.44% / 10.96% | 2.70% / 1.25% | 0.1439 | 0.0145 | 7.05% / 4.71% |
| legacy_bounded_residual | 13.48% / 7.97% | 3.24% / 1.44% | 0.1687 | 0.0186 | 8.15% / 3.48% |
| enhanced_bounded_residual | 12.64% / 7.88% | 2.70% / 1.29% | 0.1627 | 0.0181 | 7.77% / 3.53% |
| enhanced_direct | 30.84% / 24.79% | 25.30% / 18.52% | 0.1624 | 0.0180 | 26.50% / 18.55% |

## 第一确认后冻结映射的第二确认结果

冻结映射整体：calls MAPE 13.13% / WAPE 8.09%，
bytes MAPE 2.99% / WAPE 1.38%，TV
0.1468，norm EMD 0.0153，
cost MAPE 7.80% / WAPE
3.49%。

- latency：第二确认冻结方法 `legacy_bounded_residual`，确认=`False`，最终 `h0`。
- balanced：第二确认冻结方法 `legacy_bounded_residual`，确认=`False`，最终 `h0`。
- throughput：第二确认冻结方法 `h0`，确认=`True`，最终 `h0`。

第二确认对上述映射是无偏的，因为映射、四方法预测及其hash都在第二真值生成前冻结。这里
可以决定每种固定TP策略首版采用残差DNN还是H0受保护回退；不能把同一第二确认结果继续用于
挑选新的特征、超参数或映射后再声称无偏。

最终架构仍是H0结构先验加有界残差DNN；若某策略未通过两轮确认，则该策略使用H0回退，
不等于删除DNN研究路线。cost仍是5 μs + 100 GB/s统一参考曲线，不是placement/topology
物理链路实测；结论仅覆盖fixed-draining和已冻结TP配置/策略。
