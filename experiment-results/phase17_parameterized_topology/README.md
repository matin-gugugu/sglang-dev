# Phase 17：L2/L3 参数化连续代价与 ProfileDemand 传播

本实验严格分离需求与代价：Phase 16 的 1296 条标签/留出预测提供消息货物清单；L1
使用 B200 单节点实测曲线；L2/L3 各使用 optimistic/nominal/pessimistic 三组显式假设
参数生成连续曲线。L2/L3 没有物理时间真值，不能报告真实 MAPE，也不能把场景参数写成
实际集群规格。

参数化单次 AllReduce 采用：

`launch + 2(p-1)×round + [2(p-1)/p]×payload/BW_eff(payload)`，其中
`BW_eff=BW_max×(1-exp(-payload/saturation))`。L2/L3 rank mapping 简化为两节点均分
rank 的 network-level ring proxy，尚未重建 hierarchical NCCL。额外的
`protocol_transition_stress` 使用低时延/低带宽与高启动/高带宽两条子曲线的较小值，
只用于检验算法切换非线性何时使消息尺度分布不可省略。

## 成本表征误差：traffic-segment holdout，Prefill+Decode

| curve | representation | MAPE | P95 APE | bias |
|---|---|---:|---:|---:|
| l1_measured | total_bytes_data_only | 84.82% | 97.71% | -83.14% |
| l1_measured | onebin_calls_bytes | 1.50% | 6.10% | 0.89% |
| l1_measured | threebin_calls_bytes | 1.49% | 6.14% | 0.92% |
| l1_measured | twelvebin_exact | 0.30% | 1.09% | 0.27% |
| l1_measured | h0_predicted_12bin | 12.18% | 37.50% | -0.96% |
| l1_measured | residual_predicted_12bin | 14.60% | 35.11% | 2.40% |
| l2_nominal | total_bytes_data_only | 85.03% | 98.09% | -85.32% |
| l2_nominal | onebin_calls_bytes | 0.01% | 0.06% | -0.01% |
| l2_nominal | threebin_calls_bytes | 0.00% | 0.02% | -0.00% |
| l2_nominal | twelvebin_exact | 0.00% | 0.00% | -0.00% |
| l2_nominal | h0_predicted_12bin | 13.40% | 46.12% | -1.24% |
| l2_nominal | residual_predicted_12bin | 15.62% | 42.83% | 2.79% |
| l3_nominal | total_bytes_data_only | 92.72% | 99.20% | -93.38% |
| l3_nominal | onebin_calls_bytes | 0.02% | 0.10% | -0.02% |
| l3_nominal | threebin_calls_bytes | 0.00% | 0.02% | -0.00% |
| l3_nominal | twelvebin_exact | 0.00% | 0.01% | -0.00% |
| l3_nominal | h0_predicted_12bin | 14.23% | 46.35% | -1.38% |
| l3_nominal | residual_predicted_12bin | 16.42% | 43.01% | 3.18% |
| l2_protocol_transition_stress | total_bytes_data_only | 88.53% | 98.59% | -88.73% |
| l2_protocol_transition_stress | onebin_calls_bytes | 0.04% | 0.16% | -0.02% |
| l2_protocol_transition_stress | threebin_calls_bytes | 0.01% | 0.05% | -0.01% |
| l2_protocol_transition_stress | twelvebin_exact | 0.00% | 0.02% | -0.00% |
| l2_protocol_transition_stress | h0_predicted_12bin | 13.37% | 44.11% | -1.23% |
| l2_protocol_transition_stress | residual_predicted_12bin | 15.59% | 40.96% | 2.93% |
| l3_protocol_transition_stress | total_bytes_data_only | 93.30% | 99.27% | -93.89% |
| l3_protocol_transition_stress | onebin_calls_bytes | 0.03% | 0.12% | -0.01% |
| l3_protocol_transition_stress | threebin_calls_bytes | 0.01% | 0.05% | -0.01% |
| l3_protocol_transition_stress | twelvebin_exact | 0.00% | 0.01% | -0.00% |
| l3_protocol_transition_stress | h0_predicted_12bin | 14.04% | 45.37% | -1.33% |
| l3_protocol_transition_stress | residual_predicted_12bin | 16.22% | 42.10% | 3.20% |

`total_bytes_data_only` 是乐观的纯带宽基线，故意忽略 calls/RTT；`onebin` 保留总 calls
和总 bytes；三桶、12桶进一步保留尺度分布；`exact_payload_oracle` 为需求真值。

## latency配置相对throughput配置的中位比值

| curve | calls ratio L/T | bytes ratio L/T | cost ratio L/T |
|---|---:|---:|---:|
| l1_measured | 3.820 | 1.000 | 2.785 |
| l2_nominal | 3.820 | 1.000 | 2.989 |
| l3_nominal | 3.820 | 1.000 | 3.383 |
| l2_protocol_transition_stress | 3.820 | 1.000 | 3.074 |
| l3_protocol_transition_stress | 3.820 | 1.000 | 3.395 |

两种策略处理同一组请求。若 bytes ratio 接近 1 而 calls/cost ratio 显著大于 1，说明
高 RTT 拓扑会放大“小 batch、多次启动”的代价，这正是消息直方图对拓扑感知调度的
价值。

## 仅以通信成本选择 batching 策略

| curve | representation | accuracy | mean regret | P95 regret |
|---|---|---:|---:|---:|
| l1_measured | total_bytes_data_only | 0.00% | 186.38% | 350.54% |
| l1_measured | onebin_calls_bytes | 100.00% | 0.00% | 0.00% |
| l1_measured | threebin_calls_bytes | 100.00% | 0.00% | 0.00% |
| l1_measured | twelvebin_exact | 100.00% | 0.00% | 0.00% |
| l1_measured | h0_predicted_12bin | 99.54% | 0.00% | 0.00% |
| l1_measured | residual_predicted_12bin | 99.54% | 0.00% | 0.00% |
| l2_nominal | total_bytes_data_only | 0.00% | 208.63% | 404.99% |
| l2_nominal | onebin_calls_bytes | 100.00% | 0.00% | 0.00% |
| l2_nominal | threebin_calls_bytes | 100.00% | 0.00% | 0.00% |
| l2_nominal | twelvebin_exact | 100.00% | 0.00% | 0.00% |
| l2_nominal | h0_predicted_12bin | 100.00% | 0.00% | 0.00% |
| l2_nominal | residual_predicted_12bin | 100.00% | 0.00% | 0.00% |
| l3_nominal | total_bytes_data_only | 0.00% | 253.22% | 537.82% |
| l3_nominal | onebin_calls_bytes | 100.00% | 0.00% | 0.00% |
| l3_nominal | threebin_calls_bytes | 100.00% | 0.00% | 0.00% |
| l3_nominal | twelvebin_exact | 100.00% | 0.00% | 0.00% |
| l3_nominal | h0_predicted_12bin | 100.00% | 0.00% | 0.00% |
| l3_nominal | residual_predicted_12bin | 100.00% | 0.00% | 0.00% |
| l2_protocol_transition_stress | total_bytes_data_only | 0.00% | 222.35% | 462.27% |
| l2_protocol_transition_stress | onebin_calls_bytes | 100.00% | 0.00% | 0.00% |
| l2_protocol_transition_stress | threebin_calls_bytes | 100.00% | 0.00% | 0.00% |
| l2_protocol_transition_stress | twelvebin_exact | 100.00% | 0.00% | 0.00% |
| l2_protocol_transition_stress | h0_predicted_12bin | 100.00% | 0.00% | 0.00% |
| l2_protocol_transition_stress | residual_predicted_12bin | 100.00% | 0.00% | 0.00% |
| l3_protocol_transition_stress | total_bytes_data_only | 0.00% | 254.15% | 538.93% |
| l3_protocol_transition_stress | onebin_calls_bytes | 100.00% | 0.00% | 0.00% |
| l3_protocol_transition_stress | threebin_calls_bytes | 100.00% | 0.00% | 0.00% |
| l3_protocol_transition_stress | twelvebin_exact | 100.00% | 0.00% | 0.00% |
| l3_protocol_transition_stress | h0_predicted_12bin | 100.00% | 0.00% | 0.00% |
| l3_protocol_transition_stress | residual_predicted_12bin | 100.00% | 0.00% | 0.00% |

该表只在 latency/balanced/throughput 三种配置中选择通信成本最低者，并以精确 payload
直方图为 oracle。它用于检验通信表征是否会导致策略排序错误，不等同于完整在线调度；
完整目标还必须加入排队时延、计算时间、显存和吞吐约束。

## 证据边界

- 这里可以报告不同参数场景下的结构成本、表征误差和决策敏感性；
- 若 nominal alpha–beta 下单桶已足够而 transition stress 下直方图才有优势，应如实报告
  “直方图价值取决于代价曲线非线性”，不能只保留有利场景；
- 不可以报告真实 L2/L3 通信时间准确率；
- placement 前还需加入显存可行性、计算时间和资源可用性，否则只最小化通信可能得到
  平凡选择；
- 未来获得两节点资源后，只需用相同 `op×payload×group_size×rank_mapping` 微基准替换
  参数曲线，不需要重跑全部模型 PatternDemand 网格。
