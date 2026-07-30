# Phase 5：Qwen3-8B 稳定性、all-rank 标签与预测模型复核

## 1. 本阶段回答的问题

本阶段基于 Qwen3-8B，继续检验第一阶段 PatternDemand 是否稳定、消息
直方图是否比 total bytes 更有信息，以及代表 rank 的 GPU kernel 时间能否
作为最终通信时延标签。实验于 2026-07-29 至 2026-07-30 在 8×B200 节点
上完成。

核心结论如下：

1. PatternDemand 是稳定的。25 个原高波动 workload 在 10 次重复中，
   calls、payload 与消息直方图签名全部一致。
2. 消息形态不能被 total bytes 替代。在总 payload 仅相差约 3% 的
   Decode 对照中，小消息高频方案的 all-rank 通信时间是大消息低频方案的
   10.07×、15.36×、25.26×，且差距随 TP=2、4、8 增大。
3. 固定 rank 0 不是可靠的最终时延标签。all-rank 关键路径标签在 29 个
   workload 中有 24 个更稳定，IQR/median 中位数由 34.15% 降至 17.16%。
4. 当前四模型的 5.43% MAPE 是对 representative-rank proxy 的拟合精度，
   不是最终调度通信时延精度。把这些模型直接用于 all-rank 关键路径时，
   MAPE 约为 88%，说明下一阶段必须以修正后的标签重新采集和训练。

## 2. 统计口径

第一阶段需求仍采用单份 group-level PatternDemand：

- `count`：一次 TP group collective 计一次，不按 rank 累加；
- `payload_bytes`：代表 rank 的 collective 逻辑输入大小，不是所有 rank
  求和，也不是估算链路流量；
- 每个 rank 都独立采集逻辑 calls/payload，但只用于验证各 rank 完全一致，
  不重复累计；
- ring equivalent bytes/rounds 是结构化建模量，不是实测 wire traffic。

all-rank 通信标签定义为：

\[
T_{\mathrm{critical}}=\sum_e\max_r t_{e,r},
\]

其中 \(e\) 是按调用顺序对齐的 group-level collective，\(r\) 是 TP rank。
只有当所有 rank 的 kernel 数量、原语序列和 PatternDemand 均一致时，才允许
按序号对齐。Decode 使用 8-step profile window，并按 full-phase calls 等比
扩展；Prefill 全阶段采集。

该定义避免把各 rank 时间求和，同时能够覆盖不同 collective 中“慢 rank”
发生切换的情况。

## 3. 实验 A：25 个高波动点扩展到 10 次重复

从 Phase 4 的 195 个 workload 中选择原始 3 次重复
`IQR / median > 20%` 的 25 个点，增加 r3–r9 共 7 次重复。由于批量运行中的
笛卡尔积顺带覆盖 4 个邻近点，最终有 29 个 workload 达到 10 次重复：

- 新增测量记录：203；
- 25/25 目标点的 PatternDemand 签名在 10 次中完全一致；
- 原 25 点 IQR/median 中位数：108.90%；
- 10 次重复后 IQR/median 中位数：51.35%；
- 17/25 点有所改善，但仍有 15/25 点高于 20%。

GPU 健康遥测共 27,116 条样本，其中 6,986 条活跃样本全部处于 P0；活跃时
SM clock 中位数和 P95 均为 1965 MHz，温度中位数 35°C、P95 39°C。这说明
长尾不能主要归因于明显降频。结合不同重复中 rank 0 交替快/慢的现象，
固定 rank 的 collective 等待角色是主要误差来源。

输出：

- `qwen3_8b_stability/`：新增 compact ground truth、运行日志与 GPU 遥测；
- `qwen3_8b_stability_summary/stability_comparison.csv`：25 点前后对照；
- `qwen3_8b_stability_summary/qwen3_8b_stability_comparison.png`：IQR 变化图；
- `qwen3_8b_stability_summary/summary.json`：稳定性和遥测摘要。

## 4. 实验 B：四种模型的混合 3/10 次重复重评

数据仍有 195 个唯一 workload，按完整 workload 分组后划分为
train/validation/test=135/30/30；166 个 workload 使用 3 次重复中位数，
29 个 workload 使用 10 次重复中位数。

| 模型 | Test MAPE | Stable-test MAPE | 等总 payload MAPE | Test R² |
|---|---:|---:|---:|---:|
| Total bytes only | 69.07% | 75.30% | 102.30% | 0.4435 |
| Three hard bins | 15.48% | 14.28% | 13.73% | 0.9754 |
| Continuous histogram | 6.00% | 5.50% | 4.75% | 0.9860 |
| Continuous histogram + DNN residual | 5.43% | 4.22% | 6.24% | 0.9917 |

该结果证明消息尺度与 calls 分布相对 total bytes 有显著增益，也说明连续代价
曲线优于三个硬桶。但本表目标仍是 representative-rank GPU kernel envelope，
因此只能作为“表征能力消融”，不能作为最终调度时延精度。

输出：

- `qwen3_8b_prediction_eval_stabilized/aggregated_workloads.csv`：195 个聚合点；
- `qwen3_8b_prediction_eval_stabilized/metrics.csv`：各 scope 指标；
- `qwen3_8b_prediction_eval_stabilized/predictions.csv`：逐点预测；
- `qwen3_8b_prediction_eval_stabilized/summary.json`：数据、划分与指标摘要；
- 两张 PNG：整体留出评测和等总 payload 评测。

## 5. 实验 C：TP=2/4/8 all-rank 关键路径

正式网格包含 29 个唯一 workload、3 次重复，共 87 条 all-rank 记录和
57 个运行组：

- 18 个机理点：每个 TP 下 3 个 Prefill 长度和 3 个近等总 payload Decode；
- 11 个代表性高波动点：验证修正标签能否降低 rank 角色噪声。

所有记录均满足：

- 每个 rank 的 logical calls/payload 完全一致；
- 每个 rank 的 matched kernel 数等于 group-level collective calls；
- 各 rank backend sequence 完全一致；
- all-rank critical time 不小于 rank 0 time。

最终结果：

- `critical / rank0` 中位数 1.26×，P95 30.05×，最大 60.09×；
- rank0 IQR/median 中位数 34.15%；
- all-rank critical IQR/median 中位数 17.16%；
- 24/29 个 workload 的 all-rank 标签更稳定。

近等总 payload 对照的 B1/M512 与 B16/M32 逻辑总量之比仅为 1.030：

| TP | 小消息高频 / 大消息低频的 all-rank 时间比 |
|---:|---:|
| 2 | 10.07× |
| 4 | 15.36× |
| 8 | 25.26× |

这组结果直接证明：相同 total bytes 不代表相同通信代价，calls、单消息大小与
group size/rounds 必须进入结构化需求和代价模型。

输出：

- `qwen3_8b_all_rank/`：逐组 compact all-rank ground truth 与日志；
- `qwen3_8b_all_rank_summary/all_rank_summary.csv`：29 个聚合 workload；
- `qwen3_8b_all_rank_summary/qwen3_8b_equal_payload_all_rank.png`：核心等总量图；
- `qwen3_8b_all_rank_summary/qwen3_8b_all_rank_critical.png`：rank0/critical 差距；
- `qwen3_8b_all_rank_summary/qwen3_8b_all_rank_stability.png`：标签稳定性；
- `qwen3_8b_all_rank_summary/summary.json`：all-rank 摘要。

## 6. 实验 D：旧目标与修正目标的差距

在 29 个共同 workload 上，all-rank critical target 相对 Phase 4/5
representative-rank target 的中位比值为 44.57×，P95 为 77.47×。四个旧模型
没有重新训练，直接对 corrected target 评测时 MAPE 均约 88%。

该诊断不是对直方图特征的最终公平评测，而是证明 ground-truth 定义发生了
实质变化。相关输出位于：

- `qwen3_8b_all_rank_target_gap/all_rank_target_gap.csv`；
- `qwen3_8b_all_rank_target_gap/qwen3_8b_all_rank_target_gap.png`；
- `qwen3_8b_all_rank_target_gap/summary.json`。

## 7. 对开题设计的判断

原两阶段设计仍然可行，但需修正“时间标签与第二阶段曲线”的采集口径：

1. 第一阶段继续输出 topology-independent PatternDemand，包括
   `op × payload histogram × calls × group size × ring-equivalent rounds`；
2. 第二阶段的 `op × payload × TP × topology → latency` 曲线必须使用
   all-rank critical latency，而不是固定 rank kernel time；
3. 结构化公式先计算通信基线，DNN 只拟合 corrected target 的残差；
4. 当前 5.43% 结果作为消息表征消融保留，但最终模型需在 corrected labels
   上重新训练和留出评测。

## 8. 下一步

1. 给每个 workload 增加同形状 warmup，重点复核仍有长尾的 Prefill 点；
2. 将 all-rank 采集扩到完整 195 点，至少 train/validation/test 各 TP、阶段
   均有覆盖；
3. 用 all-rank 口径重测连续 `op × payload × TP × topology` 代价曲线；
4. 在 corrected dataset 上重新比较 total bytes、三桶、连续直方图和
   连续直方图+DNN residual；
5. Qwen3-8B 口径闭环后，再选择结构不同的模型做跨模型验证。

原始 profiler trace 保留在远端
`/sgl-workspace/sglang-src/experiment-results/phase5/**/traces/`，不提交 Git；
compact JSONL、CSV、日志、遥测、图和模型文件提交到实验分支。
