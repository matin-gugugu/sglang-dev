# Phase 5：Qwen3-8B PatternDemand 稳定性与通信标签复核

## 1. 本阶段回答的问题

本阶段基于 Qwen3-8B 检验：

1. 第一阶段 PatternDemand 是否可重复；
2. 消息直方图是否比 total bytes 保留更多预测信息；
3. 固定 rank 的 GPU kernel 时间为何长尾；
4. all-rank traces 应如何构造与两阶段公式一致的通信代价标签。

实验于 2026-07-29 至 2026-07-30 在单节点 8×B200 上完成。

最终结论：

1. PatternDemand 稳定。25 个原高波动 workload 在 10 次重复中，
   calls、payload、消息直方图和 ring-equivalent demand 全部一致。
2. total bytes 不能替代消息形态。在逻辑总 payload 仅相差 3.02% 的
   Decode 对照中，小消息高频方案的 intrinsic 通信代价分别是大消息低频
   方案的 10.49×、12.12×、12.21×（TP=2、4、8）。
3. 固定 rank 的 kernel duration 会混入 rank 到达顺序和等待角色，因此
   不是稳定的结构化通信标签。
4. “每次 collective 取跨 rank 最大 duration 再求和”也不能作为标签。
   它会重复累计不同 rank 上相互重叠的等待区间，相对 intrinsic 口径的
   中位放大达到 57.82×。
5. Phase 5 因而引入 skew-free intrinsic 与 post-rendezvous 两个候选：
   前者是 duration-only 可移植下包络，后者表示最后一个 rank 到齐后的
   completion。Phase 6 完整 195 点比较后，将 post-rendezvous 选为同节点
   主标签，intrinsic 保留为下包络消融。

## 2. PatternDemand 与时间标签口径

第一阶段需求始终采用单份 group-level PatternDemand：

- `count`：一次 TP group collective 计一次，不按 rank 累加；
- `payload_bytes`：代表 rank 的逻辑输入大小；
- 各 rank 独立采集 calls/payload，只用于一致性校验，不重复累计；
- ring-equivalent bytes/rounds 是结构化建模量，不是实测 wire traffic。

设第 \(e\) 个 collective 在 rank \(r\) 上的 kernel duration 为
\(d_{e,r}\)。Phase 5 首先构造 skew-free intrinsic 候选：

\[
T_{\mathrm{intrinsic}}=\sum_e\min_r d_{e,r}.
\]

它去除某些 rank 提前进入 collective 所产生的 pre-entry wait，作为
duration-only 下包络。Decode 使用 8-step profile window，并按 full-phase
calls 等比扩展；Prefill 全阶段采集。

同时保留两个诊断量，但不用于训练：

- `post_rendezvous_completion`：
  \(\sum_e(\max_r end_{e,r}-\max_r start_{e,r})\)，表示最后一个 rank
  到齐后的完成时间；Phase 6 完整数据证明它是同节点最稳定、语义最直接的
  主标签；
- `synchronization_inclusive_max_duration_sum`：
  \(\sum_e\max_r d_{e,r}\)，用于量化 rank 到达偏斜，不是通信本体标签。

## 3. 实验 A：高波动点扩展到 10 次重复

从 Phase 4 的 195 个 workload 中选择原始三重复
`IQR / median > 20%` 的 25 个点，增加 r3–r9。批量笛卡尔积额外覆盖
4 个邻近点，最终 29 个 workload 达到 10 次重复：

- 新增测量记录：203；
- 25/25 目标点的 PatternDemand 签名完全一致；
- GPU 活跃遥测均为 P0，SM clock 中位数和 P95 均为 1965 MHz；
- 固定 rank 时间仍有明显长尾，说明问题主要来自 rank 等待角色，而不是
  PatternDemand 或明显降频。

输出：

- `qwen3_8b_stability/`：compact ground truth、日志和 GPU 遥测；
- `qwen3_8b_stability_summary/`：25 点稳定性对照 CSV、PNG 和 JSON。

## 4. 实验 B：旧 representative-rank 目标上的四模型消融

Phase 4 的 195 个唯一 workload 按完整 workload 分为
train/validation/test=135/30/30；166 点使用三重复中位数，29 点使用十重复
中位数。

| 模型 | Test MAPE | Stable-test MAPE | 等总 payload MAPE | Test R² |
|---|---:|---:|---:|---:|
| Total bytes only | 69.07% | 75.30% | 102.30% | 0.4435 |
| Three hard bins | 15.48% | 14.28% | 13.73% | 0.9754 |
| Continuous histogram | 6.00% | 5.50% | 4.75% | 0.9860 |
| Continuous histogram + DNN residual | 5.43% | 4.22% | 6.24% | 0.9917 |

该表仍使用旧 representative-rank 目标，只作为消息表征能力消融。它证明
连续消息直方图明显优于 total bytes 和三个硬桶，但不是最终 intrinsic
标签上的正式精度。

输出位于 `qwen3_8b_prediction_eval_stabilized/`。

## 5. 实验 C：TP=2/4/8 all-rank 标签复核

正式网格包含 29 个唯一 workload、三次重复，共 87 条记录。所有记录均满足：

- 每个 rank 的 logical calls/payload 完全一致；
- 每个 rank 的 matched kernel 数等于 group-level calls；
- backend sequence 完全一致；
- PatternDemand 未按 rank 重复累计。

标签稳定性：

- rank0 `IQR / median` 中位数：34.15%；
- all-rank intrinsic `IQR / median` 中位数：5.44%；
- 29 个 workload 中 27 个使用 intrinsic 后更稳定。

同步等待诊断：

- synchronization-inclusive / intrinsic 中位数：57.82×；
- P95：82.60×；
- 最大：103.92×。

这不是网络本体慢了几十倍，而是说明提前进入 collective 的 kernel 把等待
时间记入 duration；逐调用取最大值再求和会跨 rank 重复累计等待。

近等总 payload 的 intrinsic 对照：

| TP | 逻辑 payload 比 | 小消息高频 / 大消息低频 |
|---:|---:|---:|
| 2 | 1.030 | 10.49× |
| 4 | 1.030 | 12.12× |
| 8 | 1.030 | 12.21× |

这直接支持论文核心论点：总 bytes 相近不代表通信代价相近，calls、单消息
大小、group size 与 rounds 必须进入 PatternDemand。

输出：

- `qwen3_8b_all_rank/`：compact all-rank labels 与日志；
- `qwen3_8b_all_rank_summary/all_rank_summary.csv`：29 个聚合点；
- `qwen3_8b_all_rank_summary/qwen3_8b_equal_payload_all_rank.png`：核心对照图；
- `qwen3_8b_all_rank_summary/qwen3_8b_all_rank_stability.png`：标签稳定性；
- `qwen3_8b_all_rank_summary/summary.json`：统计摘要。

## 6. 旧目标与 intrinsic 标签的关系

在 29 个共同 workload 上，intrinsic / 旧 representative-rank 聚合目标的
中位比为 0.993，P95 为 1.036，最大为 1.116。说明三重复中位数在多数点上
能够近似 intrinsic，但固定 rank 的单次测量和尾部仍不可靠。

旧模型未重训，直接在这 29 个 intrinsic 点上诊断：

- total bytes MAPE：66.15%；
- three hard bins：19.89%；
- continuous histogram：3.80%；
- continuous + DNN residual：50.45%。

DNN 在小诊断子集上的大尾部不能作为最终结论；需要在完整 intrinsic 数据集
上重新分组划分、训练和留出评测。

输出位于 `qwen3_8b_all_rank_target_gap/`。

## 7. 对开题设计的修正

两阶段设计继续成立。结合 Phase 6 的完整标签对照，统计口径闭环为：

1. 第一阶段输出 topology-independent PatternDemand：
   `op × payload histogram × calls × group size × ring-equivalent rounds`；
2. 同节点主标签使用 post-rendezvous completion，并匹配 synchronized
   completion curve；intrinsic 保留为可移植下包络；
3. 结构化公式先预测与目标语义匹配的通信基线；
4. DNN 只拟合结构化公式残差，不学习或放大 rank 到达等待；
5. synchronization/overlap 若要进入调度器，应作为独立特征或独立残差项，
   不能混入链路本体标签。

## 8. 下一步

1. 用 v2 三标签口径采集完整 195 点、三重复 all-rank 数据；
2. 在 corrected dataset 上重训 total bytes、三桶、连续直方图和
   连续直方图+DNN residual；
3. 完成 grouped workload holdout、等总 payload holdout 和 TP 分层评测；
4. Qwen3-8B 闭环后，再选结构不同的模型做跨模型泛化；
5. 后续拓扑实验分别测 L1/L2/L3 intrinsic 代价曲线，不把 rank 启动偏斜
   当成网络 RTT。

原始 profiler traces 保留在远端
`/sgl-workspace/sglang-src/experiment-results/phase5/**/traces/`，不提交 Git；
compact JSONL、CSV、日志、遥测、图和模型文件提交实验分支。
