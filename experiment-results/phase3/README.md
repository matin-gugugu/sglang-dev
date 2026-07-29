# Phase 3：PatternDemand × Backend-aware 连续代价曲线消融

## 1. 本轮要回答的问题

本轮将第一阶段采集的消息直方图与第二阶段单机 L1 代价曲线组合，验证：

```text
PatternDemand(phase, op, TP, payload, count)
    ×
C(topology, backend, algorithm, TP, payload)
    →
T_base
```

重点不是训练最终神经网络，而是先判断结构化基础项是否成立，以及
`total bytes only`、三个硬桶和连续 histogram 三种表征分别能解释多少通信代价。

## 2. 为什么先补测 CustomAllReduce 曲线

Qwen3-8B 的 Decode 和不超过 16 MiB 的 Prefill AllReduce 实际走
`SGLang CustomAllReduceV2`。真实推理 trace 中小消息 kernel 通常约 4–5 μs，
而上一轮 NCCL call-envelope 约 41–49 μs，二者不能直接卷积。

因此本轮先补齐：

- TP ∈ {2, 4, 8}；
- payload：8 KiB–16 MiB，共 13 个尺度；
- 每个点 100 次采样、5 次独立重复；
- 共 39 个聚合点、195 条重复记录、19,500 个 kernel 样本。

Profiler 启动和 CPU launch 会造成各 rank 进入 kernel 的时间偏斜。曲线同时保存：

- `intrinsic latency`：同一次 collective 各 rank kernel duration 的最小值，
  作为去除 launch skew 后的结构化基础代价；
- `completion latency`：各 rank kernel duration 的最大值；
- `rank skew`：`completion - intrinsic`。

`T_base` 使用 intrinsic curve，completion/rank-skew 进入残差和不确定性建模。

### 2.1 CustomAllReduce 主要结果

| TP | 8 KiB intrinsic | Algorithm 切换点 | 16 MiB intrinsic |
|---:|---:|---|---:|
| 2 | 4.160 μs | 8 MiB：`ONE_SHOT_PUSH → ONE_SHOT_PULL` | 37.536 μs |
| 4 | 4.225 μs | 4 MiB：`ONE_SHOT_PUSH → TWO_SHOT_PULL` | 47.423 μs |
| 8 | 4.927 μs | 1 MiB：`ONE_SHOT_PUSH → TWO_SHOT_PULL` | 55.712 μs |

该结果证明 algorithm 阈值随 TP 改变，预测输入必须保留 `group_size`、
`backend`、`algorithm` 和连续 payload。

## 3. 消融口径

### 3.1 Backend-aware 连续曲线

- payload ≤ 16 MiB：使用本轮 SGLang CustomAllReduce intrinsic curve；
- payload > 16 MiB：使用上一轮 NCCL curve；
- 在同一 backend 内按 `log2(payload)` 线性插值；
- workload 预测为 `Σ count(payload) × C(payload)`。

### 3.2 三硬桶

使用以下桶：

- small：`payload ≤ 64 KiB`；
- medium：`64 KiB < payload ≤ 4 MiB`；
- large：`payload > 4 MiB`。

每个 `TP × bucket` 都用微基准点拟合：

```text
cost_per_call = startup_us + transfer_us_per_byte × payload
```

聚合后等价于：

```text
T_bin = calls_bin × startup_us + bytes_bin × transfer_us_per_byte
```

因此该基线保留了原设计中的 `B_eq + R_eq`，不是只取桶中心的弱基线。

### 3.3 Total bytes only

该基线只保留总逻辑字节，不使用 calls 和尺寸分布。三种方案都只使用一个
校准 workload：

- Decode：每个 TP 的 uniform workload；
- Prefill：TP=2、L=128。

校准后在未参与校准的 mixed/longtail 和更长 Prefill 上评估。由于 Decode
对照的总 payload 完全相同，bytes-only 无论怎样调系数都无法区分三种消息形态。

### 3.4 Ground truth

结构化标签定义为每个 trace 的：

```text
calls × median_per_invocation_kernel_time
```

再对独立重复取中位数。Profiler 的 kernel total 与该结构化标签之差定义为
rank-skew/runtime residual。

因此以下 MAPE 衡量的是 `T_base`，不是已经包含等待、重叠和调度抖动的最终
通信总时间。最终预测仍应使用：

```text
T_comm = T_base + residual(workload, model, TP, overlap, rank_skew, runtime_state)
```

## 4. 消融结果

| 留出集 | Total bytes only | 三硬桶 | 连续 histogram |
|---|---:|---:|---:|
| Decode mixed/longtail，TP=2/4/8 | 63.27% | 0.86% | 0.86% |
| Prefill L=512/1024/2048/4096，TP=2 | 169.36% | 9.43% | 1.49% |

### 4.1 等总 payload Decode

uniform、mixed、longtail 的总逻辑 payload 都是 71,761,920 bytes，但 calls 分别为：

| 形态 | Calls | TP=2 structural target |
|---|---:|---:|
| uniform | 1,095 | 4.906 ms |
| mixed | 2,555 | 11.365 ms |
| longtail | 3,723 | 16.560 ms |

bytes-only 对三者给出相同预测；三桶和连续 histogram 都能恢复 calls 所导致的
2.3×/3.4× 代价增长。

在该数据上三桶与连续曲线几乎相同，不是连续方案无效，而是所有 Decode payload
都位于 16–64 KiB 的同一启动平台。该实验主要证明“不能丢掉 calls/消息形态”。

### 4.2 Prefill 连续尺寸

Prefill 的单次 payload 从 1 MiB 增长到 32 MiB，并经历：

- Custom `ONE_SHOT_PUSH`；
- Custom `ONE_SHOT_PULL`；
- 32 MiB NCCL fallback。

连续 histogram 的 MAPE 为 1.49%，三桶为 9.43%。这部分证明只保留三个硬区间会
产生量化误差，连续 payload 和 backend/algorithm 边界能够显著改善结构化预测。

### 4.3 为什么仍需要神经网络残差

Decode profiler total 中的 runtime residual 占比约为：

- TP=2：1.1%–33.2%；
- TP=4：31.3%–68.9%；
- TP=8：64.0%–78.7%。

它主要来自 rank skew、kernel 内同步等待和运行时调度状态。结构化曲线能准确预测
无偏斜基础项，但不能单独解释这些长尾。因此神经网络应校正 residual，而不是绕过
PatternDemand 直接预测总时间。

## 5. 结果文件

CustomAllReduce 微基准：

- `../phase2/b200_l1_custom_kernel_curve/`：原始 JSONL、每轮日志、环境与拓扑；
- `../phase2/summary_l1_custom_kernel_curve/custom_kernel_curve_summary.csv`；
- `../phase2/summary_l1_custom_kernel_curve/custom_kernel_cost_knots.json`；
- `../phase2/summary_l1_custom_kernel_curve/b200_l1_custom_kernel_curve.png`。

PatternDemand 消融：

- `pattern_cost_ablation/workload_predictions.csv`：14 个 workload 的真值、残差和预测；
- `pattern_cost_ablation/ablation_metrics.csv`：MAPE、P95 APE、MAE、R²；
- `pattern_cost_ablation/phase3_summary.json`：桶拟合参数、校准参数与完整指标；
- `pattern_cost_ablation/phase3_pattern_cost_ablation.png`：四面板主结果图。

## 6. 当前结论与边界

当前可以得出三个明确结论：

1. total bytes 无法描述等总字节、不同 calls/消息形态的工作负载；
2. 三硬桶足以解释当前小消息 Decode，但在跨尺度 Prefill 上存在明显量化误差；
3. backend-aware 连续 histogram 可以作为可信的结构化 `T_base`。

当前只有 Qwen3-8B、单机 L1 和 14 个唯一 workload，不能将 1.49% 解释为最终模型
的泛化误差。下一步需要系统采集更大的 `B × L × M × TP` profiler 标签集，按
workload 而不是 repeat 划分训练/测试集，再训练 residual 模型；之后补 L2/L3。

