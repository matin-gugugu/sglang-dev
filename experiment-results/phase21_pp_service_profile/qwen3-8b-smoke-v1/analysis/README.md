# Phase 21：到达感知的纯 PP PatternDemand smoke

## 目标

在 `TP=1` 的纯 PP 配置下，用完全相同的 32 个请求长度对比两种执行方式：

- `profiled`：根据常态画像的 RPS 与到达间隔 CV 构造确定性的 gamma-renewal 到达过程；
- `draining`：32 个请求同时提交。

该配对实验检验到达与突发特征是否会通过 SGLang batching 改变 PP 消息直方图，
不是对下一时间窗口请求序列的预测。

## 数据与有效性

- 模型：Qwen3-8B，纯 PP，`PP=2/4/8`；
- 策略：`pp_max_micro_batch_size=1/4/16`；
- 画像：3 个，重复 3 次，两种 arrival mode；
- 9/9 cell 通过审计，共 162 次画像回放、
  5184 次逻辑请求执行；
- sender 边界直方图在所有 PP 边界一致，统计使用首个 sender 作为 group-level 真值，
  pipeline-wide demand 再乘以 `PP-1`，不重复累计 send/recv。

## 主要结果

在 81 个配对重复中，
63 个的精确 payload 直方图发生变化，
占 77.8%。

| PP | max microbatch | changed pairs | phase changes | mean phase TVD |
|---:|---:|---:|---:|---:|
| 2 | 1 | 3/9 | 3/18 | 0.0110 |
| 2 | 4 | 9/9 | 18/18 | 0.4857 |
| 2 | 16 | 9/9 | 18/18 | 0.7480 |
| 4 | 1 | 3/9 | 3/18 | 0.0104 |
| 4 | 4 | 9/9 | 18/18 | 0.4650 |
| 4 | 16 | 9/9 | 18/18 | 0.5487 |
| 8 | 1 | 3/9 | 3/18 | 0.0086 |
| 8 | 4 | 9/9 | 18/18 | 0.5618 |
| 8 | 16 | 9/9 | 18/18 | 0.5599 |

重复稳定性：43/108
个 `PP×策略×画像×arrival×phase` 分组的三次精确直方图完全一致。稳定性不足的分组
必须在正式训练前单独报告，而不能用均值掩盖调度抖动。

## 文件

- `cell_summary.csv`：cell 级有效性与到达影响比例；
- `arrival_effect_pairs.csv`：Prefill/Decode 配对 calls、bytes 和分布 TVD；
- `repeat_stability.csv`：三次重复的精确一致性；
- `arrival_effect_rate.svg`：到达影响热力图；
- `summary.json`：机器可读结论。

## 结论边界

当前只覆盖 3/24 个画像和一个模型，属于机制验证。32 个请求是从画像窗口中分层选出的
长度样本；`profiled` 到达是依据画像 RPS/CV 重新构造的稳态实现，不是原始连续 trace
的逐请求时间戳回放。因此当前可以证明“到达过程会改变 PP PatternDemand”，但不能声称
已经完成跨画像、跨模型的在线 PP 预测器。
