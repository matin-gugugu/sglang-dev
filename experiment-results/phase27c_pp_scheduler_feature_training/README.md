# Phase 27C：PP 调度敏感低维特征训练

状态：**PASS**。本阶段只读取 Phase 27B 的 30 个开发训练画像、12 个开发
验证画像和不含 target 的确认集特征文件；独立确认真值文件没有作为脚本参数，也没有被读取。

## 开发验证集 total 结果

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | common cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| h0 | 59.40% / 18.50% | 2.05% / 1.71% | 0.2491 | 0.0418 | 8.92% / 6.83% |
| legacy_bounded_residual | 28.53% / 16.95% | 2.68% / 1.85% | 0.1854 | 0.0275 | 7.74% / 6.48% |
| enhanced_bounded_residual | 23.78% / 9.77% | 2.52% / 2.02% | 0.1711 | 0.0255 | 4.52% / 3.64% |
| enhanced_direct | 17.38% / 16.25% | 38.70% / 16.16% | 0.1309 | 0.0156 | 16.64% / 10.39% |

`legacy_bounded_residual`只使用Phase 26同口径的长度联合分布、均值、生存率、模型与固定
PP配置；`enhanced_bounded_residual`在相同样本与训练流程下额外使用4096-token chunk、
chunk×输出、顺序转移、连续段和局部拥塞摘要。因此两者差异主要回答“调度敏感画像是否
提供额外信息”。`enhanced_direct`仅作为控制组，不参与候选规则。

## 已冻结的确认集候选

- mb1：`enhanced_bounded_residual`
- mb4：`enhanced_bounded_residual`
- mb16：`enhanced_bounded_residual`

候选规则只看开发验证集：相对H0至少赢得calls MAPE、TV、cost MAPE中的两项，且cost
MAPE不超过H0的110%；多个residual合格时选择胜项更多、三项相对比值之和更小者。

`analysis/independent_confirmation_predictions.csv.gz`已经在不读确认真值时写出四种方法的
1,296行预测。下一步评测脚本只做hash核验和真值join，不能再训练、早停或改变候选映射。
当前可以确认训练隔离和候选冻结成立；不能把这里的validation结果当作独立泛化结论。
