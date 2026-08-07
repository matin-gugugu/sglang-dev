# Phase 18：TP离线候选决策pilot

本实验使用Phase16的4905条真实GPU墙钟记录和Phase17的通信代价传播，
在每个`model×profile×topology`内枚举TP2/4/8与三档batching策略。
L1使用实测曲线；L2/L3仍为参数化敏感性场景。

候选总服务工作量定义为：

`noncomm_proxy = measured_L1_wall - exact_L1_comm`

`candidate_total = noncomm_proxy + representation_comm(topology)`

它只隔离通信表征对候选排序的影响，不是online queueing或生产调度器。

## 核心结果

| topology | objective | representation | accuracy | mean regret | P95 regret | rank agreement |
|---|---|---|---:|---:|---:|---:|
| l1_measured | gpu_efficiency | exact_payload_oracle | 100.00% | 0.000% | 0.000% | 100.00% |
| l1_measured | gpu_efficiency | h0_predicted_12bin | 100.00% | 0.000% | 0.000% | 100.00% |
| l1_measured | gpu_efficiency | onebin_calls_bytes | 100.00% | 0.000% | 0.000% | 100.00% |
| l1_measured | gpu_efficiency | residual_predicted_12bin | 100.00% | 0.000% | 0.000% | 100.00% |
| l1_measured | gpu_efficiency | total_bytes_data_only | 100.00% | 0.000% | 0.000% | 99.96% |
| l1_measured | gpu_efficiency | twelvebin_exact | 100.00% | 0.000% | 0.000% | 100.00% |
| l1_measured | latency | exact_payload_oracle | 100.00% | 0.000% | 0.000% | 100.00% |
| l1_measured | latency | h0_predicted_12bin | 100.00% | 0.000% | 0.000% | 99.77% |
| l1_measured | latency | onebin_calls_bytes | 100.00% | 0.000% | 0.000% | 99.88% |
| l1_measured | latency | residual_predicted_12bin | 100.00% | 0.000% | 0.000% | 99.73% |
| l1_measured | latency | total_bytes_data_only | 91.67% | 0.013% | 0.074% | 99.11% |
| l1_measured | latency | twelvebin_exact | 100.00% | 0.000% | 0.000% | 99.96% |
| l2_nominal | gpu_efficiency | exact_payload_oracle | 100.00% | 0.000% | 0.000% | 100.00% |
| l2_nominal | gpu_efficiency | h0_predicted_12bin | 100.00% | 0.000% | 0.000% | 99.88% |
| l2_nominal | gpu_efficiency | onebin_calls_bytes | 100.00% | 0.000% | 0.000% | 100.00% |
| l2_nominal | gpu_efficiency | residual_predicted_12bin | 100.00% | 0.000% | 0.000% | 99.88% |
| l2_nominal | gpu_efficiency | total_bytes_data_only | 100.00% | 0.000% | 0.000% | 99.34% |
| l2_nominal | gpu_efficiency | twelvebin_exact | 100.00% | 0.000% | 0.000% | 100.00% |
| l2_nominal | latency | exact_payload_oracle | 100.00% | 0.000% | 0.000% | 100.00% |
| l2_nominal | latency | h0_predicted_12bin | 100.00% | 0.000% | 0.000% | 99.69% |
| l2_nominal | latency | onebin_calls_bytes | 100.00% | 0.000% | 0.000% | 100.00% |
| l2_nominal | latency | residual_predicted_12bin | 100.00% | 0.000% | 0.000% | 99.73% |
| l2_nominal | latency | total_bytes_data_only | 100.00% | 0.000% | 0.000% | 98.53% |
| l2_nominal | latency | twelvebin_exact | 100.00% | 0.000% | 0.000% | 100.00% |
| l3_nominal | gpu_efficiency | exact_payload_oracle | 100.00% | 0.000% | 0.000% | 100.00% |
| l3_nominal | gpu_efficiency | h0_predicted_12bin | 100.00% | 0.000% | 0.000% | 99.50% |
| l3_nominal | gpu_efficiency | onebin_calls_bytes | 100.00% | 0.000% | 0.000% | 99.96% |
| l3_nominal | gpu_efficiency | residual_predicted_12bin | 100.00% | 0.000% | 0.000% | 99.58% |
| l3_nominal | gpu_efficiency | total_bytes_data_only | 100.00% | 0.000% | 0.000% | 97.92% |
| l3_nominal | gpu_efficiency | twelvebin_exact | 100.00% | 0.000% | 0.000% | 100.00% |
| l3_nominal | latency | exact_payload_oracle | 100.00% | 0.000% | 0.000% | 100.00% |
| l3_nominal | latency | h0_predicted_12bin | 100.00% | 0.000% | 0.000% | 99.19% |
| l3_nominal | latency | onebin_calls_bytes | 100.00% | 0.000% | 0.000% | 100.00% |
| l3_nominal | latency | residual_predicted_12bin | 100.00% | 0.000% | 0.000% | 99.42% |
| l3_nominal | latency | total_bytes_data_only | 100.00% | 0.000% | 0.000% | 97.45% |
| l3_nominal | latency | twelvebin_exact | 100.00% | 0.000% | 0.000% | 100.00% |

## 自动解读

- L1 latency目标下，total-bytes选择准确率为91.67%，但平均regret仅0.013%；
- 同口径H0预测12桶的选择准确率为100.00%，平均regret为0.000%；
- 与Phase17的communication-only结果相比，加入真实非通信墙钟后，当前候选排序主要由计算/运行时部分主导；通信表征仍改善排序，但端到端收益不能由通信侧regret直接替代；
- 参数化L2/L3中选择差异依旧很小，说明若要形成更强的完整调度证据，需要真实高RTT链路、online queue/SLO约束或更接近决策边界的候选对照。

## 证据边界

- Phase16完整网格的时间标签只有一次正式回放，本结果属于pilot；
- arrival offset被保存在紧凑标签中，但当前不宣称online continuous batching；
- `balanced`和`gpu_efficiency`是显式资源加权目标，不是生产SLO；
- 后续真实调度器还需要队列、显存、并发副本和资源可用性。
