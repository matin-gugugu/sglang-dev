# Phase 16G：ProfileDemand v1 四方法留出评测

输入为低维常态流量画像、数值化 batching 策略、可泛化模型结构、候选 TP 和阶段；输出为
每 1000 请求的 12 桶 group-level calls 与代表 rank logical bytes。共使用 1296 条
GPU 聚合标签。对比 Model-ID 直接 DNN、结构特征直接 DNN、透明公式 H0，以及正式方法
H0+DNN residual。

外层测试分别留出完整流量 segment、模型、执行策略和 TP；内层早停再按 profile 分组，
避免同一画像及其跨 TP 重复标签泄漏。`metrics.csv` 报告直方图 calls/bytes WAPE、分布
L1/EMD，以及预测直方图乘 B200 L1 连续 AllReduce 曲线后的结构代价误差。

H0 只从 4×4 长度联合分布和均值合成 32 个伪请求，不读取 GPU replay 的 32 条真实长度
或顺序；residual 因而学习分桶内形态和 batching 边界，而不是重复精确公式。正式 checkpoint
为 `formal_h0_residual_model.pt`。总量 residual 被限制在两倍以内、分布 logit residual
被限制在 ±2；整模型留出时也对标准化输入和输出做训练域裁剪，保证 DNN 只能校正 H0，
不能在未见结构上产生无物理意义的指数外推。

边界：当前到达率/突发特征虽进入输入，但 GPU 标签仍是同时进入的 draining microbatch，
不能把结果表述为 online arrival-aware batching 已完成。L1 传播也是结构代价评估，不是新增
的端到端通信时间真值。

## 核心结果

| 外层留出 | 方法 | total calls MAPE | total bytes MAPE | log-payload EMD | L1 结构代价 MAPE | P95 APE |
|---|---|---:|---:|---:|---:|---:|
| traffic_segment_holdout | h0 | 9.18% | 5.36% | 0.017 | 9.20% | 30.40% |
| traffic_segment_holdout | h0_residual | 12.57% | 7.64% | 0.016 | 11.34% | 30.65% |
| model_holdout | h0 | 9.18% | 5.36% | 0.017 | 9.20% | 30.40% |
| model_holdout | h0_residual | 10.05% | 5.29% | 0.017 | 8.71% | 27.71% |
| strategy_holdout | h0 | 9.18% | 5.36% | 0.017 | 9.20% | 30.40% |
| strategy_holdout | h0_residual | 9.22% | 4.43% | 0.017 | 8.05% | 27.59% |
| tp_holdout | h0 | 9.18% | 5.36% | 0.017 | 9.20% | 30.40% |
| tp_holdout | h0_residual | 8.46% | 4.51% | 0.016 | 7.55% | 25.31% |

H0 在四类留出中固定为 9.20% L1 结构代价 MAPE。受约束 residual 在未见模型、策略和 TP
上分别降至 8.71%、
8.05% 和
7.55%；但在完整未见
流量 segment 上为 11.34%，
弱于 H0，因此未知流量域应回退 H0，不能宣称 DNN 全面优于结构公式。

residual 的 total calls MAPE 为 8.46%–12.57%，total bytes MAPE 为 4.43%–7.64%；虽然硬桶
vector WAPE 较高，但 log-payload EMD 只有 0.016–0.017，说明主要是相邻硬桶边界迁移而非
消息质量跨越多个尺度。未见流量 segment 时，structure-direct DNN 的 L1 代价 MAPE 为
53.63%，进一步支持“结构公式为主、DNN 只校正残差”。
