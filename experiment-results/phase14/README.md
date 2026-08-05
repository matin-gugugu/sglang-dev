# Phase 14：TP4/8 代表点时间泛化验证

更新时间：2026-08-05

状态：smoke、正式采集、TP2/4/8 联合分析、复验和归档均已完成

## 1. 目标

Phase 13B 已在单节点 B200、TP=2 上验证三模型多支撑点时间预测。Phase 14 不重跑
完整规则网格，而是选择 Qwen3-8B 与 Qwen3-30B-A3B 的代表 workload，补齐 TP=4/8
all-rank post-rendezvous 时间标签，检验：

1. logical calls 和 payload 直方图跨 TP 是否保持一致；
2. group size、equivalent rounds 和 backend transition 如何改变真实时间；
3. 仅在 TP2 上校准的 continuous histogram 能否零样本预测 TP4/8；
4. TP4/8 是否仍需要额外的 per-TP calibration。

## 2. 正式网格

硬件与模型：

| 维度 | 设置 |
|---|---|
| hardware | 单节点 8×B200 |
| models | Qwen3-8B、Qwen3-30B-A3B |
| TP | 4、8；TP2 复用 Phase 11/13B |
| repeats | 0、1、2 |
| mixed Decode | balanced、staircase、bimodal |
| chunk sizes | 1024、4096 |
| chunk boundary input lengths | T-1、T、T+1 |
| chunk batch sizes | 1、4 |

新增正式规模：

- mixed Decode：2 models × 2 TP × 3 profiles × 3 repeats = 36 标签；
- chunked Prefill：2 models × 2 TP × 2 chunk sizes × 3 repeats × 6 workloads
  = 144 标签；
- 合计 180 条新增标签，聚合为 60 个 TP4/8 配置；
- 联合匹配 TP2 后共 270 条标签、90 个聚合配置、30 组三点 TP scaling 对照。

这组设计保留了 mixed batch drain、chunk 边界，以及 `B=4,T≈1024` 与
`B=1,T≈4096` 的近等 active-token/total-payload 对照，但避免重跑完整网格。

## 3. 时间与结构口径

主目标沿用 Phase 11/13B：

```text
sum over aligned collectives of
(max rank kernel end - max rank kernel start)
```

即 all-rank post-rendezvous completion time。每个完整配置的三次重复先取中位数。

PatternDemand 必须保存 `(raw_op, payload) -> calls`，并分别保存 group size、
ring-equivalent bytes/rounds 和实际 backend sequence。continuous curve 使用 Phase 2
已有的 TP2/4/8 group-size-aware 曲线；它按 payload 边缘化计价，因此不会被描述为
专门的 fused-op 微基准。raw op 和 backend transition 在分析中单独审计。

## 4. 执行

```bash
bash scripts/run_phase14_tp_group_size_timing.sh smoke

nohup bash scripts/run_phase14_tp_group_size_timing.sh formal-analyze \
  > experiment-results/phase14/phase14_driver.log 2>&1 &
```

TP8 使用全部 8 张 GPU，所有 model/TP 单元必须串行运行。通用 runner 会在每个单元
开始前检查目标 GPU 空闲状态。

## 5. 验收条件

- smoke 8/8 单元通过；
- 正式 60/60 单元生成 `DONE`；
- 新增 180/180 compact labels；
- 每个聚合配置恰好 3 个 repeat；
- 每条标签的所有 rank kernel count、backend sequence 和 PatternDemand 完全对齐；
- raw-op 直方图边缘化后与 payload-only 直方图一致；
- 曲线外推调用比例显式报告，不允许静默外推；
- TP2-calibrated 和 per-TP descriptive calibration 分开报告；
- raw profiler trace 在 validator 通过后删除；
- 日志、telemetry、README、summary、CSV、图和 manifest 提交并 push。

## 6. 数据与复验结果

| 项目 | 结果 |
|---|---:|
| smoke 单元 | 8/8 通过 |
| 正式实验单元 | 60/60 生成 `DONE` |
| TP4/8 新增 compact labels | 180/180 |
| TP2/4/8 联合原始标签 | 270 |
| 聚合配置 | 90，每个配置恰好 3 个 repeat |
| TP scaling 对照 | 30 组 |
| all-rank 对齐 | 180/180 完全一致 |
| raw-op/payload 边缘化复验 | 全部通过 |
| 正式目录残留 raw trace | 0 |

正式标签的 repeat 分布为 `r0/r1/r2 = 60/60/60`。每条标签的所有 rank
kernel count、backend sequence、profiled/full-phase PatternDemand 均完全对齐；
profiled-to-full scale 全部为 1.0。smoke 与 formal 的独立复验日志均为
`PASS`，最终 driver 日志未发现 Traceback、OOM、NCCL error 或 validator failure。

三次重复的 post-rendezvous IQR/median：

| TP | 中位数 | P95 | 超过 20% |
|---|---:|---:|---:|
| TP2 | 0.439% | 2.855% | 0/30 |
| TP4 | 0.590% | 2.241% | 0/30 |
| TP8 | 0.490% | 1.918% | 0/30 |

## 7. TP scaling 与 backend transition

30/30 组对照的 logical calls、logical bytes 和 `(raw_op,payload)` 直方图跨
TP 完全不变，但 group-size 时间并不保持不变：

| 指标 | 中位数 | P95 |
|---|---:|---:|
| TP4 / TP2 真实时间 | 1.329× | 1.394× |
| TP8 / TP2 真实时间 | 1.536× | 1.679× |

其中 12/30 组至少发生一次 backend sequence transition。主要变化是 TP2 上的
SGLang custom one-shot 在 TP4/8 上切换为 two-shot；大消息仍可能使用 NCCL，
Qwen3-30B-A3B 的 fused residual RMSNorm 路径由 FlashInfer MNNVL 执行。TP4 与
TP8 的匹配 workload backend sequence 相同，因此 TP8 相比 TP4 的额外时间不能
只归因于 backend 切换，group size 和同步/轮次成本本身也必须进入模型。

## 8. TP2 零样本泛化结果

TP4/8 共 60 个聚合配置：

| 方法 | MAPE | P95 APE | R² |
|---|---:|---:|---:|
| total bytes，TP2 拟合 | 25.500% | 42.551% | 0.8715 |
| continuous histogram，TP2 校准 | 60.523% | 191.862% | -3.4224 |
| continuous histogram，per-TP 描述性校准 | 19.761% | 99.955% | 0.9818 |

TP2 校准的 continuous histogram 没有零样本泛化到 TP4/8。该结果不是采集失败，
而是否定了“group-size-aware curve 加一个 TP2 calibration factor 即可跨 TP
泛化”的假设。曲线外推调用仅为 `5238/104550 = 5.010%`，因此失败不能只由
payload 超出微基准范围解释。

per-TP 描述性校准使用了被评估 TP 的真实标签，只能用于判断额外校准的潜在价值，
不能当作零样本结果。其 Decode MAPE 为 TP4 7.803%、TP8 1.041%，但 Prefill
仍为 TP4 20.262%、TP8 26.929%，且整体 P95 仍接近 100%。后续应优先验证
不读取运行后 backend 的 `PatternDemand + TP + phase` 条件模型，再决定是否需要
可在运行前推导的 backend proxy。

## 9. 无实际 backend 的 TP×phase 条件模型

在 Phase 14 归档后，进一步复用 90 个聚合配置执行 Phase 14B 分析。模型不读取
实际 backend、kernel name、模型身份或 Phase 2 backend curve，只使用
PatternDemand、TP、phase 和 TP×phase。采用完整 workload 的 6 折外层留出，
并在每个训练折内部选择 ridge alpha；同一 workload 的 TP2/4/8 始终位于同一折。

| 方法 | MAPE | P95 APE | R² |
|---|---:|---:|---:|
| PatternDemand | 18.495% | 41.554% | 0.9408 |
| PatternDemand + TP | 13.795% | 46.939% | 0.9100 |
| PatternDemand + phase | 19.048% | 42.820% | 0.9318 |
| PatternDemand + TP + phase + TP×phase | 11.819% | 38.261% | 0.9597 |

TP×phase 条件把平均误差显著降低，但没有通过整体 MAPE <10% 和 P95 <25%
门槛。Prefill/Decode MAPE 分别为 11.858%/11.664%，Decode P95 为 43.128%。
完整留出 Qwen3-8B 和 Qwen3-30B-A3B 的 MAPE 分别为 33.169% 和 29.258%，
说明当前模型还不能稳定跨模型泛化。

backend 只用于预测后的残差诊断：跨 TP backend 不切换的 54 个配置 MAPE 为
8.947%，发生切换的 36 个配置为 16.127%。因此该模型应作为不依赖实际 backend
的有效基线保留，但尚不能作为生产默认模型。

## 10. 正式产物

```text
experiment-results/phase14/
├── README.md
├── audit_summary.json
├── manifest.sha256
├── tp_group_size_timing_smoke/
├── tp_group_size_timing_ground_truth/
├── tp_group_size_timing_analysis/
├── tp_phase_no_backend_analysis/
├── phase14_smoke_driver.log
├── phase14_driver.log
├── revalidate_phase14_smoke.log
├── revalidate_phase14.log
└── revalidate_tp_phase_no_backend.log
```

runner 与初始分析实现提交为 `fb76599`。根 manifest 覆盖除自身以外的全部
Phase 14 归档文件，并通过逐项 SHA-256 复验。

## 11. 结论边界

Phase 14 只覆盖单节点 B200、Qwen3 同家族两模型、代表性 TP2/4/8 mixed/chunked
workload。不能外推到跨节点 L2/L3、PP、PD、expert-parallel All-to-All、其他 GPU
或未知 backend lowering。当前可以声称 logical PatternDemand 跨 TP 不变，但不能
声称 TP2 校准的通信时间模型可直接零样本迁移到 TP4/8。无实际 backend 的
TP×phase 条件模型在已见模型的 workload 留出上有效，但尚不能声称其尾部误差可控、
稳定跨模型泛化，或 backend effect 已经被完全消除。
