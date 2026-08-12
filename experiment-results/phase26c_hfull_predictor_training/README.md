# Phase 26C：Hfull监督预测器重训

状态：**PASS**。

本阶段使用Phase 26B统一数据分别训练TP与PP的structure-direct和
H0+bounded-residual模型；H0作为无参数基线。拟合只使用5个`train`画像，早停只使用
5个`validation`画像。5个temporal、8个external和1个external synthetic测试画像未参与
训练、标准化或模型选择，留给Phase 26D。

## validation配置级total结果

| 并行 | 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | common cost MAPE |
|---|---|---:|---:|---:|---:|---:|
| TP | h0 | 12.34% / 11.55% | 3.17% / 1.27% | 0.1089 | 0.0117 | 8.16% |
| TP | direct | 121.64% / 142.70% | 72.93% / 63.29% | 0.3536 | 0.0424 | 82.13% |
| TP | h0_bounded_residual | 12.67% / 11.37% | 4.97% / 4.10% | 0.1091 | 0.0117 | 7.68% |
| PP | h0 | 39.05% / 10.47% | 3.17% / 1.27% | 0.2334 | 0.0344 | 15.27% |
| PP | direct | 65.98% / 68.31% | 29.70% / 29.00% | 0.5373 | 0.0702 | 53.06% |
| PP | h0_bounded_residual | 32.03% / 15.59% | 9.22% / 5.59% | 0.2086 | 0.0303 | 14.95% |

这些是模型选择用validation结果，不是最终测试结论。正式结论必须以Phase 26D的三个
测试域为准。这里的L1/TV在各自原生12桶上计算，和Phase 26B用于teacher审计的exact
payload TV不是同一个离散粒度；log-payload EMD在total时合并prefill/decode的桶质量，
TV则保留phase-aware的24维分布。

## 训练契约

- 输入：55个低维画像、模型结构、固定并行配置、固定策略和phase特征；不含完整请求列表；
- 输出：各自原生12桶的calls与logical bytes，每1000请求归一化；
- TP与PP分别训练，避免混淆4 KiB–512 MiB与4 KiB–8 GiB的桶语义；
- direct预测完整log-total与log-share编码；
- residual只预测相对H0的校正，总量限制在两倍以内，share-logit限制在±2，并通过tanh硬约束；
- common cost仍是5 μs+100 GB/s参数参考，不是PP物理曲线。

## 资产

- `checkpoints/`：TP/PP各自的direct与bounded-residual checkpoint；
- `analysis/validation_predictions.csv.gz`：validation逐配置、逐phase与total预测；
- `analysis/validation_metrics.csv`：TP/PP、方法、phase与policy聚合；
- `analysis/training_history.csv.gz`：四个网络的训练/早停轨迹；
- `figures/validation_method_comparison.png`：TP/PP的calls、TV与common cost对比；
- `feature_contract.json`、`summary.json`、`audit_summary.json`、`logs/training.log`、
  `DONE`和`manifest.sha256`。

可以确认模型已在Hfull监督下完成重训，且测试画像未用于选择。不能确认模型对temporal、
external或synthetic域的泛化优于H0；下一步Phase 26D将冻结这些checkpoint做profile-level
holdout测试，并分别报告TP/PP与policy。
