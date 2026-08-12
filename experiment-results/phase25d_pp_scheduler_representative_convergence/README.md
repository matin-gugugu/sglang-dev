# Phase 25D：scheduler-faithful PP代表请求收敛分析

状态：**PASS**。这里的PASS表示实验数据生成、回归校验和证据链完整，
并不表示H32/H64/H128通过收敛门槛。

本阶段使用Phase 25B恢复的SGLang lane scheduler，重新计算H32/H64/H128/Hfull和compact32，
取代Phase 24使用的静态PP分组。Hfull与Phase 25B teacher的432/432条记录完全一致。

| 请求规模 | calls MAPE | calls WAPE | bytes MAPE | 直方图TV | norm EMD | cost MAPE | P95 calls APE | 全部门槛 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| H32 | 71.99% | 25.21% | 5.82% | 0.4209 | 0.0511 | 17.13% | 294.37% | 未通过 |
| H64 | 33.50% | 12.93% | 2.78% | 0.3318 | 0.0347 | 8.40% | 150.86% | 未通过 |
| H128 | 18.65% | 7.93% | 2.01% | 0.2624 | 0.0241 | 6.06% | 59.39% | 未通过 |

## 覆盖范围与保存资产

实验覆盖24个BurstGPT/Mooncake窗口、`PP2/4/8`、`MB1/4/16`、Prefill/Decode/total，
所有指标均按每1,000请求归一化。目录保存精确payload直方图、逐case与聚合指标、
compact32误差分解、新旧teacher比较、图表、日志、DONE和manifest。

## GPU证据与结论边界

Phase 25B已有42请求smoke上全部9种配置的GPU证据；Phase 25C又增加BurstGPT和Mooncake
长prompt尾部窗口上3个对角配置的完全一致证据。这些结果验证了已审计区域内的teacher契约，
但不意味着H32/H64/H128标签本身都是GPU直接测量，也不覆盖online arrival-aware调度。

正式结论是：H32、H64和H128都不能在当前PP配置范围内统一替代Hfull；
完整请求列表仍只用于离线生成标签。下一阶段预测器仍只读取紧凑历史画像、模型结构、
固定PP配置、固定策略和phase，并以Hfull作为监督目标。
