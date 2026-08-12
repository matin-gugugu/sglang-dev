# Phase 29C：对齐三模型TP残差DNN训练

状态：**PASS**。本阶段在Phase 29B的30个开发训练画像上拟合，并只用12个
开发验证画像早停。训练输入是低维历史画像、模型结构、固定TP size、固定执行策略、phase
和compact32 H0；训练真值是完整历史窗口Hfull teacher消息直方图。

## 开发验证集结果

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| h0 | 12.05% / 11.08% | 2.05% / 1.71% | 0.1538 | 0.0144 | 6.54% / 5.05% |
| legacy_bounded_residual | 9.15% / 7.46% | 2.04% / 1.65% | 0.1547 | 0.0152 | 4.87% / 3.47% |
| enhanced_bounded_residual | 7.98% / 6.87% | 2.45% / 1.63% | 0.1677 | 0.0163 | 3.95% / 3.07% |
| enhanced_direct | 27.82% / 25.31% | 23.18% / 18.60% | 0.1632 | 0.0162 | 23.33% / 18.69% |

`legacy_bounded_residual`使用Phase 26同口径55列；`enhanced_bounded_residual`使用113列，
加入TP批处理敏感画像；`enhanced_direct`是控制组。最终设计仍是H0结构先验加残差DNN，H0
只作为基线与受保护回退，不代表取消DNN。

开发验证集冻结的候选为：

- latency：`legacy_bounded_residual`
- balanced：`legacy_bounded_residual`
- throughput：`enhanced_bounded_residual`

第一、第二独立确认集各972条feature均未包含target。本阶段已同时写出各四种方法的3,888条
预测，两个确认target文件都不是训练脚本参数。下一阶段只能按hash连接预先冻结的预测与真值，
不能重新训练、调参或改写这些预测。因此本阶段只能给出开发验证结论，不能替代独立泛化结论。
