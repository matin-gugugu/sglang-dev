# Phase 13：三模型 PatternDemand 与多支撑点时间验证

更新时间：2026-08-04

状态：Phase 13A、Phase 13B 的采集、分析、复验和归档均已完成

## 1. 阶段范围

Phase 13 包含两个连续子阶段：

1. Phase 13A 联合 Qwen3-8B、DeepSeek-V2-Lite 和 Qwen3-30B-A3B 的
   TP2/4/8 规则网格，验证模型结构、workload、TP group size 和 raw op 如何决定
   PatternDemand；
2. Phase 13B 在 TP=2 上为 Qwen3-30B-A3B 补齐 mixed Decode 与 chunked Prefill
   多支撑点 all-rank 时间标签，再执行三模型时间预测消融和整模型留出验证。

Phase 13A 聚合 585 个 `model × workload` 配置，解析 PatternDemand 的逐
`(raw_op, payload)` 直方图重建失败数为 0。Phase 13B 的正式采集和分析结果如下。

## 2. Phase 13B 数据审计

| 项目 | 结果 |
|---|---:|
| 模型与并行配置 | Qwen3-30B-A3B，单节点 B200，TP=2 |
| smoke 单元 | 2/2 通过 |
| 正式实验单元 | 18/18 生成 `DONE` |
| Qwen3-30B-A3B 标签 | 117/117 |
| 聚合配置 | 39，每个配置恰好 3 个 repeat |
| 三模型原始标签 | 351 |
| 三模型聚合配置 | 117 |
| 数据切分 | train 75 / validation 21 / test 21 |
| all-rank 对齐 | 117/117 kernel count 与 backend sequence 完全一致 |
| profiled-to-full scale | 全部为 1.0 |
| 保留的 raw op | `all_reduce`、`fused_allreduce_residual_rmsnorm` |
| 正式目录残留 raw trace | 0 |

主目标沿用 Phase 11 的 all-rank post-rendezvous completion time：

```text
sum over aligned collectives of
(max rank kernel end - max rank kernel start)
```

三次重复先在完整配置内取中位数，再按完整 workload/profile/chunk 配置切分，
没有 repeat 泄漏。正式复验确认 18 个单元的 result、all-rank label、日志、telemetry、
validator 输出和 `TRACES_REMOVED` 标记均完整。

## 3. 测量稳定性

| 时间口径 | IQR/median 中位数 | P95 | IQR 超过 20% 的配置 |
|---|---:|---:|---:|
| post-rendezvous | 0.438% | 2.425% | 0/117 |
| intrinsic | 0.487% | 4.681% | 1/117 |
| sync-inclusive | 20.852% | 51.780% | 63/117 |

post-rendezvous 标签在当前三次重复下最稳定。sync-inclusive 指标大量包含跨 rank
等待，不适合作为当前结构预测的主目标。

## 4. 三模型时间预测结果

测试集共 21 个配置：

| 方法 | MAPE | P95 APE | R² |
|---|---:|---:|---:|
| total bytes only | 15.138% | 31.344% | 0.9080 |
| three hard bins | 9.226% | 26.045% | 0.9295 |
| continuous histogram | 3.771% | 13.432% | 0.9875 |
| continuous histogram + DNN residual | 3.272% | 19.341% | 0.9581 |

continuous histogram 是当前最稳妥的主模型：相较 total bytes，MAPE 从 15.138%
降到 3.771%，同时取得更好的 P95 和 R²。DNN residual 的平均 MAPE 略低，但尾部
误差和 R² 变差，因此不能仅凭平均误差判定其更优。

分 phase 结果进一步显示该风险：Prefill 测试集 18 个点上，DNN residual MAPE
为 0.624%，continuous histogram 为 2.991%；Decode 只有 3 个测试点，DNN residual
为 19.160%，反而显著差于 continuous histogram 的 8.456%。Decode 样本太少，
当前不能声称 DNN 对动态 mixed Decode 已稳定泛化。

在 11 个近等 total-payload 测试配置上，total bytes、three bins、continuous 和
DNN residual 的 MAPE 分别为 18.833%、11.370%、3.947% 和 5.543%。这直接支持：
总字节数不能替代消息尺度直方图。

## 5. 整模型留出与受控对照

每次用两个模型训练、完整留出第三个模型时，continuous / DNN residual 的 MAPE：

| 留出模型 | continuous histogram | + DNN residual |
|---|---:|---:|
| Qwen3-8B | 5.642% | 3.071% |
| DeepSeek-V2-Lite | 5.037% | 2.028% |
| Qwen3-30B-A3B | 4.791% | 6.143% |

DNN residual 在留出 Qwen3-30B-A3B 时退化，说明它尚未形成跨模型普适残差规律；
结构化 continuous histogram 应继续作为默认基线。

45 组近等总 payload、不同消息结构对照中，测量时间比的中位数为 1.136，最大值
为 1.332。36 组 payload 边界对照的中位数为 1.085，最大值为 1.530。这些结果说明
即使 total payload 接近，消息次数、尺度分布、raw op 和 backend lowering 仍会造成
可测量的时间差异。

## 6. 采集过程修正与审计链

首次 smoke 暴露了 FlashInfer MNNVL fused allreduce 的 kernel 识别缺口。修正后：

- one-shot fused kernel 能被识别；
- two-shot 路径只计 collective kernel，不把独立 RMSNorm kernel 误算成第二次通信；
- 边界探针 `L=1023/1025/2049` 的期望与实测 kernel 数分别为
  `97/97`、`194/194`、`291/291`；
- 最终 117 条标签全部通过 op-aware、all-rank validator。

失败尝试、修正后 driver、正式复验日志均随本阶段产物保留，便于复核问题和修正。
runner 与 extractor 的关键提交为 `3b1b63f` 和 `8c727bd`。

## 7. 正式产物

```text
experiment-results/phase13/
├── README.md
├── audit_summary.json
├── manifest.sha256
├── qwen3_30b_a3b_multiscale_timing_smoke/
├── multiscale_timing_ground_truth/qwen3-30b-a3b/
├── three_model_multiscale_timing_analysis/
├── phase13b_driver.log
├── phase13b_driver_attempt1.log
├── phase13b_driver_attempt2.log
├── phase13b_attempt1_extract_failure.log
├── phase13b_attempt2_chunk_extract.log
├── revalidate_phase13b_smoke.log
└── revalidate_phase13b.log
```

## 8. 结论边界

当前可以声称：在单节点 B200、TP=2 和本阶段 mixed Decode/chunked Prefill
workload 范围内，保留 raw op 和连续消息尺度直方图可显著改善 collective 时间预测，
post-rendezvous 标签的三重复稳定性良好。

当前不能外推到 TP4/8 时间、跨节点、L2/L3、PP、PD、expert-parallel All-to-All、
其他 GPU/backend 或未知 runtime lowering。DNN residual 也尚不能替代结构模型。
