# Phase 27D：PP 独立确认集评测

状态：**PASS**。本阶段没有训练、早停或重新选择方法，只把 Phase 27C 已写入
Git并通过hash冻结的1,296行预测，与Phase 27B的18个独立确认画像Hfull真值做精确join。

## 18个独立确认画像的total结果

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | common cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| h0 | 62.13% / 20.08% | 3.15% / 1.28% | 0.2733 | 0.0444 | 10.96% / 7.07% |
| legacy_bounded_residual | 30.56% / 15.56% | 3.92% / 1.87% | 0.2096 | 0.0317 | 8.03% / 5.46% |
| enhanced_bounded_residual | 25.84% / 11.35% | 3.46% / 1.97% | 0.1844 | 0.0280 | 6.52% / 4.32% |
| enhanced_direct | 31.83% / 19.58% | 34.12% / 14.21% | 0.1376 | 0.0174 | 23.30% / 10.74% |

## Phase 27C预先冻结候选

- mb1：`enhanced_bounded_residual`，calls MAPE 6.59%，TV 0.0439，cost MAPE 4.40%。
- mb4：`enhanced_bounded_residual`，calls MAPE 15.38%，TV 0.1347，cost MAPE 6.67%。
- mb16：`enhanced_bounded_residual`，calls MAPE 55.55%，TV 0.3747，cost MAPE 8.48%。

## 确认后的首版建议

- mb1：冻结候选确认=`False`；后续建议 `h0`。
- mb4：冻结候选确认=`True`；后续建议 `enhanced_bounded_residual`。
- mb16：冻结候选确认=`True`；后续建议 `enhanced_bounded_residual`。

MB1的冻结residual候选只改善TV，calls和cost均退化，因此回退H0；MB4/MB16在calls、TV、
cost三项都改善，保留增强residual候选。但这份5 μs + 100 GB/s确认集已经参与上述建议，
不能在同一数据上计算一个“新混合规则”的无偏总分，建议还需要下一批新窗口确认。

这里的主结论应同时看calls、bytes、TV/EMD和common cost。common cost仍是5 μs +
100 GB/s参考曲线，不是PP P2P物理实测。增强residual相对legacy residual的差异才是新增
chunk/顺序低维画像的主要证据；direct只是控制组。

本阶段可以判断新增调度敏感画像在全新窗口上是否重复改善，也可以判断冻结的分策略候选
是否优于H0。仍不能声称跨模型PP泛化，因为teacher和训练均只有Qwen3-8B；也不能把
fixed-draining结果外推到online arrival-aware调度。
