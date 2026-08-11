# Phase 22：纯 PP 服务画像 PatternDemand 预测器

## 完成范围

- Qwen3-8B，纯 `TP=1`、`PP=2/4/8`；
- `pp_max_micro_batch_size=1/4/16`；
- 24 个 BurstGPT/Mooncake 画像的 216 个 draining 配置、432 个阶段标签；
- 6 个分层画像的 108 个在线窗口、216 个阶段标签；
- 对比 Direct DNN、结构化 H0、H0 + bounded DNN residual；
- 所有 GPU 标签使用首个 sender 边界作为 group-level 真值，其余边界只做一致性检查。

## 在线严格留出结果

| 留出方式 | 方法 | calls MAPE | bytes MAPE | histogram L1 |
|---|---|---:|---:|---:|
| profile_holdout | direct_dnn | 125.88% | 142.73% | 0.6459 |
| profile_holdout | h0 | 74.45% | 3.10% | 0.8837 |
| profile_holdout | h0_residual | 44.87% | 2.78% | 0.9055 |
| strategy_holdout | direct_dnn | 61.61% | 44.02% | 0.7042 |
| strategy_holdout | h0 | 74.45% | 3.10% | 0.8837 |
| strategy_holdout | h0_residual | 44.62% | 1.42% | 0.8798 |
| pp_holdout | direct_dnn | 63.36% | 32.77% | 0.6146 |
| pp_holdout | h0 | 74.45% | 3.10% | 0.8837 |
| pp_holdout | h0_residual | 53.83% | 2.17% | 0.8176 |

## 重复稳定性

- 108 个 `画像×PP×策略×phase` 重复组；
- 精确直方图一致：34/108（31.48%）；
- 两次重复的 calls 平均相对差：7.08%；
- calls P95 相对差：30.80%；
- logical bytes 两次重复完全一致。

这说明模型结构和长度画像能够稳定确定总逻辑字节，但在线batch边界会改变消息调用次数，
相同输入画像并不对应唯一的精确calls直方图。因此正式目标应是条件期望直方图，而不是
一次调度实现的精确直方图。

## 结论

当前结构化 H0 + residual 在三种在线留出下的bytes MAPE为
2.78%、
1.42%和
2.17%，验证了结构公式对通信总量的价值。

但calls MAPE仍为44.87%、
44.62%和
53.83%，histogram L1也未收敛。因此本阶段数据和执行
闭环通过，但不能把当前checkpoint作为调度器默认PP预测器。

下一步应增加紧凑的输入/输出长度生存曲线和更细联合分布，并在分层小样本上增加重复，
直接学习期望calls/直方图残差；不应通过扩大黑盒DNN掩盖输入画像信息不足。
