# Phase 29D1：TP第一独立确认

状态：**PASS**。本阶段没有训练、早停或重写预测，只把Phase 29C已写入Git并
通过hash冻结的3,888条预测，与Phase 29B物理隔离的972条第一确认Hfull真值精确连接。

## 18个独立确认画像的total结果

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| h0 | 12.45% / 10.97% | 3.15% / 1.28% | 0.1581 | 0.0156 | 6.78% / 5.04% |
| legacy_bounded_residual | 11.51% / 5.76% | 3.33% / 1.58% | 0.1959 | 0.0212 | 4.98% / 2.63% |
| enhanced_bounded_residual | 12.28% / 5.83% | 3.68% / 1.74% | 0.1988 | 0.0214 | 5.41% / 2.65% |
| enhanced_direct | 77.17% / 30.25% | 32.11% / 20.67% | 0.1619 | 0.0187 | 40.09% / 20.25% |

## 面向第二确认的冻结建议

- latency：验证集候选 `legacy_bounded_residual`，第一确认通过=`True`，第二确认冻结建议 `legacy_bounded_residual`。
- balanced：验证集候选 `legacy_bounded_residual`，第一确认通过=`True`，第二确认冻结建议 `legacy_bounded_residual`。
- throughput：验证集候选 `enhanced_bounded_residual`，第一确认通过=`False`，第二确认冻结建议 `h0`。

第一确认可以无偏检验开发验证阶段冻结的候选；根据第一确认产生的新建议不能再在第一确认
上声称无偏，因此只用于尚未生成真值的第二独立确认。第二批的四方法预测已在Phase 29C同期
冻结，下一阶段不得重新训练或改写预测。

这里的cost使用统一5 μs + 100 GB/s参考曲线，不是具体拓扑的物理实测；当前结论只适用于
fixed-draining与已冻结TP配置/策略，不能外推到online arrival-aware。
