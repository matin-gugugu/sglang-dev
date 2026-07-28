# Phase 2：B200 单机 L1 连续 Collective 代价曲线

## 1. 实验目的

本轮测量第二阶段的单机 NVLink 链路代价函数：

```text
C_L1(op, payload_bytes, group_size)
```

它接收第一阶段 `PatternDemand` 输出的原语、逻辑消息大小和通信组规模，为
`T_base = Σ count × C_L1(...)` 提供连续、可插值的代价。该实验不是最终预测精度评测，
而是为后续“消息直方图优于总字节/三个硬桶”的定量对照准备 L1 代价标签。

## 2. 实验配置与统计口径

- 设备：单机 8 × NVIDIA B200，NVLink；
- 软件：PyTorch 2.11.0+cu130、CUDA 13.0、NCCL 2.28.9；
- 原语：NCCL AllReduce、AllGather；
- group size：TP ∈ {2, 4, 8}；
- payload：1 KiB–128 MiB 的 2 次幂点，额外加入 48 KiB，共 19 个尺度；
- 每个点：预热 30 次、采样 100 次；
- 每条曲线：5 次独立进程级重复；
- 总量：6 条曲线、114 个聚合点、570 条重复记录、57,000 个调用样本；
- 时间口径：各 rank CUDA Event call-envelope 的最大值，表示一次 group-level
  collective 完成时间。

Payload 口径与第一阶段保持一致：

- AllReduce：`payload_bytes` 是代表 rank 的逻辑输入字节数，
  ring 等效字节因子为 `2(p-1)/p`；
- AllGather：`payload_bytes` 是聚合后的逻辑输出字节数，每 rank 输入为
  `payload_bytes/p`，ring 等效字节因子为 `(p-1)/p`。

## 3. 主要结果

下表给出 pooled median；带宽是按 ring 等效字节计算的 bus bandwidth。

| Op | TP | 1 KiB 延迟 | 8 MiB 延迟 | 128 MiB 延迟 | 128 MiB bus BW |
|---|---:|---:|---:|---:|---:|
| AllReduce | 2 | 42.50 μs | 47.68 μs | 284.48 μs | 471.80 GB/s |
| AllReduce | 4 | 45.44 μs | 52.80 μs | 357.79 μs | 562.69 GB/s |
| AllReduce | 8 | 46.56 μs | 77.34 μs | 399.94 μs | 587.30 GB/s |
| AllGather | 2 | 45.22 μs | 43.65 μs | 179.52 μs | 373.82 GB/s |
| AllGather | 4 | 45.02 μs | 57.97 μs | 207.33 μs | 485.53 GB/s |
| AllGather | 8 | 48.66 μs | 50.72 μs | 215.38 μs | 545.28 GB/s |

可直接支持论文论点的观察：

1. 小消息存在约 41–49 μs 的启动平台，不能用 `bytes/BW` 单独描述；
2. 有效带宽随 payload 连续变化，不存在覆盖所有尺度的固定带宽；
3. group size 会同时改变启动、算法轮次和饱和过程。例如 AllReduce 8 MiB
   从 TP=2 的 47.68 μs 增长到 TP=8 的 77.34 μs；
4. AllReduce 与 AllGather 的曲线形态不同，因此代价函数必须保留 `op`；
5. 基于单调带宽包络的启发式 25% 饱和点多在 8–16 MiB，而 75%/90% 饱和点
   随 `op × TP` 落在 32–128 MiB。三个固定消息桶只能做展示，预测应使用连续曲线。

TP=4 AllReduce 的 r2 在 1 KiB–8 MiB 区间出现一次约 20–25 μs 的整体上移，
r3/r4 均回到主簇。所有样本均保留，最大 repeat-median CV 为 0.225。该现象不应被
解释为稳定的 payload 阈值，而应作为运行时状态不确定性：中心预测使用 pooled
median 连续曲线，P95 与 repeat CV 用于误差带或残差特征。

## 4. 结果文件与用途

原始结果：

- `b200_l1_collective_curve/environment.json`：硬件与软件版本；
- `b200_l1_collective_curve/nvidia_topology.txt`：物理拓扑；
- `b200_l1_collective_curve/suite.log`：完整套件日志；
- `b200_l1_collective_curve/tp{2,4,8}/{all_reduce,all_gather}/r{0..4}/curve.jsonl`：
  每个 payload 的统计量及 100 个调用样本；
- 同目录 `run_attempt1.log`：单次独立运行日志。

汇总结果：

- `summary_l1_curve/collective_curve_repeat_records.csv`：570 条重复级记录；
- `summary_l1_curve/collective_curve_summary.csv`：114 个 pooled 聚合点；
- `summary_l1_curve/collective_cost_knots.json`：按 `log2(payload)` 线性插值的
  调度器/预测模型输入；
- `summary_l1_curve/b200_l1_collective_curve.png`：延迟、P95 与 bus bandwidth
  的四面板图。

## 5. 对预测模型的使用方式

第一阶段不直接输出总时间，而是输出：

```text
(phase, op, group_size, payload_bin, count)
```

第二阶段对每个 histogram bin 的代表 payload 查询或插值本轮曲线：

```text
T_base = Σ_phase,op,tp,bin count_bin × C_L1(op, payload_bin, tp)
```

最终神经网络只拟合 `T_measured - T_base`，用于吸收 backend/algorithm 切换、
rank skew、通信计算重叠和运行时状态，而不绕过 PatternDemand 变成纯黑盒。

## 6. 当前边界与下一步

本轮只覆盖单机 L1、NCCL、无计算重叠的 collective call-envelope；不能直接代表
SGLang 自定义 AllReduce，也不能外推 L2/L3。

下一步应进入 Phase 3：

1. 将 Qwen3-8B 现有 TP=2/4/8 推理 histogram 与本轮连续曲线离线卷积，生成
   `T_base`；
2. 与 profiler 的真实通信时间对齐，形成 workload 级预测数据表；
3. 在相同 train/test split 下比较 `total bytes only`、三个硬桶和连续 histogram；
4. 报告 MAPE、P95 APE、R²，并单独展示“等总 payload、不同消息形态”对照；
5. 完成 L1 消融后，再测 L2 同机架跨节点真实曲线，最后处理 L3。

