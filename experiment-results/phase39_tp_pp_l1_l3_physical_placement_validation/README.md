# Phase39：TP/PP L1–L3物理曲线库与placement验证

最终状态：`PASS_WITH_RUNTIME_AND_PLACEMENT_VARIANCE`。本阶段完成24个GPU通信measurement shard、12条物理曲线；没有加载模型、checkpoint或重新生成预测。raw逐次样本保存在Git外。

## 物理cost

- `pp/L1`：total WAPE `4.4118%`，MAPE `13.2098%`，bias `1.2088%`。
- `pp/L2`：total WAPE `3.9935%`，MAPE `9.5629%`，bias `0.9803%`。
- `pp/L3`：total WAPE `4.2173%`，MAPE `9.6559%`，bias `1.4055%`。
- `tp/L1`：total WAPE `7.5732%`，MAPE `8.8243%`，bias `-2.1950%`。
- `tp/L2`：total WAPE `7.5223%`，MAPE `8.3474%`，bias `-2.2327%`。
- `tp/L3`：total WAPE `7.8476%`，MAPE `8.5593%`，bias `-2.2614%`。

## communication-only placement

- `overall=all`：top1 `100.0000%`，mean regret `0.0000%`，P95 regret `0.0000%`。
- `parallelism=pp`：top1 `100.0000%`，mean regret `0.0000%`，P95 regret `0.0000%`。
- `parallelism=tp`：top1 `100.0000%`，mean regret `0.0000%`，P95 regret `0.0000%`。

TP/PP size与policy始终是输入，决策器只在L1/L2/L3之间选择。该排名不包含计算、显存、排队、资源可用性、metadata或通信计算重叠，不能直接声称为完整线上scheduler收益。

Phase34冻结直方图指标复现最大绝对差为`1.110e-16`。Phase34D target已打开，因此本阶段是重复工程证据，不是新盲测。
