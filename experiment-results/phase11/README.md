# Phase 11：双模型多支撑点 all-rank 通信时间与消融

## 1. 阶段目标

Phase 10 已证明 mixed Decode 与 chunked Prefill 会产生不能由总 calls 和总
payload 唯一确定的消息尺度直方图。本阶段在完全相同的 234 条 PatternDemand
配置上增加 all-rank GPU 时间标签，回答以下问题：

1. mixed Decode 在总 calls 和总 payload 完全相同时，消息直方图变化是否对应
   可测量的通信时间差异；
2. chunk size 跨越边界产生的 calls 跳变是否对应真实时间跳变；
3. total bytes、三硬桶、精确直方图乘连续代价曲线和结构曲线加 DNN residual
   四类预测器的误差如何；
4. Qwen3-8B 与 DeepSeek-V2-Lite 的残差和跨模型泛化是否一致。

## 2. 实验规模与时间口径

| 项目 | 设置 |
|---|---|
| 模型 | Qwen3-8B、DeepSeek-V2-Lite |
| 并行配置 | 单节点 B200，TP=2 |
| mixed Decode | 2 模型 × 3 profile × 3 次 = 18 条 |
| chunked Prefill | 2 模型 × 3 chunk size × 12 workload × 3 次 = 216 条 |
| 总标签 | 36 个实验单元，234 条 all-rank compact labels |
| 聚合配置 | 78 个；每个配置先聚合 3 次重复 |
| 划分 | train/validation/test = 50/14/14 |
| 划分单位 | 完整 workload/profile/chunk 配置；repeat 不跨集合 |
| profiler | GPU activities、all ranks、完整阶段窗口 |
| 原始 trace | 标签提取和验证后删除，不提交 |

主标签沿用 Phase 6 的同节点 L1 口径：

\[
T_{\mathrm{post}}=
\sum_e\left(\max_r f_{e,r}-\max_r s_{e,r}\right).
\]

即每个严格对齐的 group-level collective 从最后一个 rank 进入到所有 rank
完成的时间，再对完整阶段求和。三重复取中位数作为配置级目标。

`intrinsic = sum(min rank duration)` 作为可移植下包络；
`sync-inclusive = sum(max rank duration)` 仅作为到达偏斜诊断，不进入训练。

## 3. 完整性与审计

重新审计和逐单元验证结果：

- 36/36 `DONE`、`result.jsonl`、`all_rank_ground_truth.jsonl`；
- 36/36 run、extract、validate log 和 telemetry；
- 234/234 compact labels，schema 为 `all-rank-comm-labels-v2`；
- 234/234 固定输出长度与实际生成长度一致；
- 每条记录的所有 rank kernel count、backend sequence 和 PatternDemand 一致；
- 完整阶段缩放系数均为 1.0；
- run log 未发现 OOM、timeout、Traceback、NCCL error 或被跳过的正式点；
- raw trace 文件为 0，36/36 单元保存 `TRACES_REMOVED`；
- 重新执行的全目录验证输出保存为 `revalidate_qwen.log` 和
  `revalidate_deepseek.log`。

采集时 Qwen 使用 GPU 0–1，DeepSeek 使用 GPU 2–3，两套 Phase 11 runner
在同一节点并行执行。GPU 4–7 未出现正式负载，也未发现外部 GPU 作业。两套实验
使用不同 GPU，但共享节点、供电和互联资源，因此不能把本轮描述为“整机独占”；
潜在共享资源干扰作为本阶段限制保留。重新审计时 8 张 GPU 均为空闲。

## 4. 标签稳定性

| 标签 | IQR/median 中位数 | P95 | IQR > 20% 配置数 |
|---|---:|---:|---:|
| Post-rendezvous | **0.46%** | **2.26%** | **0/78** |
| Intrinsic | 0.51% | 3.40% | 1/78 |
| Sync-inclusive | 22.72% | 52.59% | 43/78 |

Post-rendezvous 在本轮多支撑点配置上仍然最稳定。Sync-inclusive 的高波动再次
说明 rank 提前到达等待不能作为结构通信代价直接训练。

## 5. mixed Decode：相同总量，不同真实时间

每个模型的三种 profile 都固定：

```text
B = 8
prompt_len = 512
max_output_len = 64
sum(actual output tokens) = 256
```

同一模型内三组 profile 的总 calls 和总 payload 完全相同，但消息直方图不同。

### Qwen3-8B

| profile | calls | total payload | post-rendezvous 中位数 |
|---|---:|---:|---:|
| balanced | 4,599 | 148,307,968 B | 25,425.71 μs |
| staircase | 4,599 | 148,307,968 B | 25,383.18 μs |
| bimodal | 4,599 | 148,307,968 B | 24,564.86 μs |

最大/最小时间比为 1.035，最大绝对差为 860.86 μs。

### DeepSeek-V2-Lite

| profile | calls | total payload | post-rendezvous 中位数 |
|---|---:|---:|---:|
| balanced | 3,465 | 55,869,440 B | 21,891.82 μs |
| staircase | 3,465 | 55,869,440 B | 21,084.54 μs |
| bimodal | 3,465 | 55,869,440 B | 22,109.65 μs |

最大/最小时间比为 1.049，最大绝对差为 1,025.10 μs。

六组 pairwise 对照的时间比中位数为 1.034、最大为 1.049。差异不大，但明显
高于 post-rendezvous 的总体重复 IQR 中位数，证明相同粗粒度总量不能唯一确定
真实通信时间。Total-bytes 和三硬桶在同一模型的三组 profile 上给出相同预测；
连续直方图曲线能产生非 1 的结构差异，但仍存在模型相关残差。

## 6. chunked Prefill：calls 边界对应真实时间跳变

共得到 24 组总 payload 完全相同、chunk size 不同、消息次数或直方图不同的
对照：

- 实测时间比中位数为 **1.140**；
- 最大时间比为 **1.332**；
- total-bytes baseline 对每一组都只能预测 1.0 的时间比。

还构造了 24 个 chunk 边界三元组 `(boundary-1, boundary, boundary+1)`。
边界上方 calls 相比边界点增加 1.5× 或 2×，24/24 的 post-rendezvous 时间
都出现正向跳变：

| 模型 | 边界数 | 时间比中位数 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| Qwen3-8B | 12 | 1.053 | 1.008 | 1.221 |
| DeepSeek-V2-Lite | 12 | 1.101 | 1.026 | 1.353 |

该结果把 Phase 10 的 PatternDemand 离散跳变连接到了真实 L1 通信时间，而不再
只是结构公式层面的 calls 对照。

## 7. 四模型 grouped holdout

四类模型为：

1. `total_bytes_only`：phase 条件下的总逻辑 payload 回归；
2. `three_hard_bins`：`<=64 KiB`、`64 KiB–4 MiB`、`>4 MiB` 的 calls 和 bytes；
3. `continuous_histogram`：精确 payload 直方图逐点查询 Phase 2 B200 L1 连续
   cost curve，再按 phase 使用训练集做一个乘性校准；
4. `continuous_histogram_dnn_residual`：小型 MLP 只预测
   `log(measured / calibrated structural)`，不绕过结构曲线。

DNN 不使用显式 model one-hot；输入为工作负载、精确直方图统计、三桶统计和
结构估计，便于执行双向模型留出。

### 7.1 主测试集

| 模型 | Test MAPE | P95 APE | R² |
|---|---:|---:|---:|
| Total bytes only | 8.34% | 26.29% | 0.9973 |
| Three hard bins | 6.80% | 11.23% | 0.9923 |
| Continuous histogram | 3.36% | **7.34%** | 0.9917 |
| Continuous + DNN residual | **3.17%** | 12.48% | 0.9770 |

DNN 把平均 MAPE 从 3.36% 降到 3.17%，但 P95、RMSE 和 R²变差。因此本阶段
不能声称残差网络全面优于连续结构曲线；连续直方图是更稳健的尾部误差基线。

分阶段结果进一步显示数据量限制：

| 场景 | 样本数 | Total bytes | Three bins | Continuous | Continuous + DNN |
|---|---:|---:|---:|---:|---:|
| Prefill test | 12 | 9.37% | 6.84% | 2.71% | **1.40%** |
| Decode test | 2 | **2.13%** | 6.52% | 7.23% | 13.82% |

Decode 只有每模型 3 个 profile，严格拆分后 test 仅 2 个点。它足以做等总量
机制对照，不足以支撑稳定的 DNN Decode 泛化结论。

按模型看，Qwen 测试集上 DNN MAPE 为 2.09%、连续曲线为 3.69%；DeepSeek
上分别为 4.25% 和 3.02%。方向不一致，说明两模型残差结构不同，也说明继续
增加模型和 Decode profile 比增加网络容量更重要。

### 7.2 双向 model holdout

| 留出模型 | Total bytes | Three bins | Continuous | Continuous + DNN |
|---|---:|---:|---:|---:|
| Qwen3-8B（39 配置） | 12.32% | 9.09% | 5.06% | **3.88%** |
| DeepSeek-V2-Lite（39 配置） | 12.70% | 22.57% | 6.61% | **3.46%** |

完整配置上的模型留出说明，精确结构曲线和 model-neutral residual 特征具有一定
跨模型能力。但 Decode-only 留出的连续曲线 MAPE 仍为 11.72%/12.54%，DNN 为
12.68%/12.09%，且每个方向只有 3 个 Decode 点；不能据此宣称已解决未见模型
的 mixed Decode 预测。

## 8. 结论边界

本阶段可以准确声称：

1. 在双模型 TP=2 单节点 L1 上，相同总 calls 和总 payload 的 mixed Decode
   直方图可以对应 3.4%–4.9% 的实测时间跨度；
2. 相同总 payload、不同 chunk 消息结构的 24 组对照产生最高 1.332× 时间差；
3. chunk calls 在边界上增加 1.5×/2× 时，24/24 配置出现正向真实时间跳变；
4. grouped holdout 上连续直方图把 total-bytes MAPE 从 8.34% 降至 3.36%，
   并把 P95 从 26.29% 降至 7.34%；
5. residual DNN 对 Prefill 和双向模型留出有帮助，但当前 Decode 数据过少，且
   主测试集尾部误差变差，不能写成无条件优于结构模型。

本阶段仍不能声称：

- 第三模型已经验证；
- TP=4/8 的多支撑点时间标签已经完成；
- L2/L3 物理实测已经完成；
- 并行采集不存在任何共享节点干扰；
- residual DNN 已在 mixed Decode 或未见模型上稳定收敛。

## 9. 正式产物

```text
experiment-results/phase11/
├── README.md
├── revalidate_qwen.log
├── revalidate_deepseek.log
├── multiscale_timing_ground_truth/
│   ├── qwen3-8b/...
│   ├── deepseek-v2-lite/...
│   └── *_driver.log
└── multiscale_timing_analysis/
    ├── summary.json
    ├── metrics.csv
    ├── predictions.csv
    ├── split_assignments.csv
    ├── equal_payload_comparison.csv
    ├── boundary_comparison.csv
    ├── model_holdout_metrics.csv
    ├── model_holdout_predictions.csv
    ├── dnn_residual_model.pt
    ├── multiscale_timing_analysis.png
    └── analyze.log
```

分析脚本：

```bash
MPLBACKEND=Agg PYTHONPATH=scripts \
  python scripts/analyze_multiscale_timing.py
```

逐单元重新验证：

```bash
bash scripts/run_cross_model_multiscale_timing_dataset.sh all qwen
bash scripts/run_cross_model_multiscale_timing_dataset.sh all deepseek
```

runner 遇到已验证的 `DONE` 单元时只重新执行 validator，不覆盖正式数据。

## 10. 下一步

1. 保留当前 split 与分析协议，用第三模型补充同一套多支撑点时间数据；
2. 对 mixed Decode 增加更多 output survival profile，并对高波动/尾部代表点
   增加到至少 10 次；
3. 第三模型加入后重新评估 model-holdout，不因 DNN 在小样本上的平均 MAPE
   改善而扩大结论；
4. PP/PD 分别冻结独立事件 schema、sender-only 计数和 timeline 口径后再采集；
5. 第二台授权机器到位后复制协议到 L2/L3，不把本轮 L1 结果外推为跨节点实测。
