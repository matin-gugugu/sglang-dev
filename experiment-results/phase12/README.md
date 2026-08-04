# Phase 12：Qwen3-30B-A3B 多 TP PatternDemand 数据集

## 1. 阶段目标

Phase 11 在 Qwen3-8B 与 DeepSeek-V2-Lite、单节点 B200、TP=2 的范围内，
把精确消息直方图连接到了 all-rank post-rendezvous collective kernel time。
Phase 12 引入第三个模型 Qwen3-30B-A3B，采集 TP=2/4/8 下规则
Prefill/Decode 网格的通信结构，回答：

1. 同一 Qwen 家族的稠密模型与 MoE 模型是否具有不同 TP PatternDemand；
2. workload 的 batch、input length 和 output length 如何改变 calls 与 payload 支撑；
3. group size 从 TP2 增加到 TP4/TP8 时，逻辑 PatternDemand 与拓扑折算量如何变化；
4. 第三模型是否为后续三模型 leave-one-model-out 分析提供完整、无泄漏的数据基础。

本阶段采集的是 TP collective 结构，不是 expert-parallel 通信，也不包含完整阶段
wall time 或 Phase 11 的 all-rank 时间标签。

## 2. 模型准入与固定版本

| 项目 | 值 |
|---|---|
| 模型 | Qwen3-30B-A3B |
| architecture | Qwen3MoeForCausalLM |
| revision | ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 |
| dtype | bfloat16 |
| hidden size | 2048 |
| layers | 48 |
| attention heads / KV heads | 32 / 4 |
| experts / experts per token | 128 / 8 |
| 权重 | 16 shards，61,066,575,648 bytes |
| config SHA256 | 2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297 |
| index SHA256 | df0d481ec595c55a0ba58426d517390c6214a566ec4ff1c8fc4bbce9f57b3c24 |

准入 smoke 分别覆盖 TP=2 与 TP=8。两者均完成模型加载、固定输出长度和
histogram-only 通信采集，得到 97 次 Prefill collective calls；TP=4 的正式 runner
要求 TP2 与 TP8 准入同时通过后才允许执行。

## 3. 正式实验网格

| 阶段 | 网格 | 每次重复 |
|---|---|---:|
| Prefill | B={1,2,4,8,16} × L={128,512,2048,8192} × M=8 | 20 |
| Decode | B={1,2,4,8,16} × L={128,2048,8192} × M={32,128,512} | 45 |

并行配置为 TP={2,4,8}，每个配置重复 3 次。正式实验共有：

- 3 TP × 3 repeats × 2 phases = 18 个实验单元；
- 3 TP × 3 repeats × 65 workloads = 585 条原始记录；
- 三次重复先按 (TP, phase, B, L, M) 聚合后，得到 195 个独立 workload 配置。

585 条记录不能作为 585 个独立样本随机拆分。后续任何 train/validation/test
划分必须以完整 workload 配置为单位，三个 repeat 不得跨 split。

## 4. 采集契约

- comm-profile-mode 为 histogram-only；
- 所有 TP ranks 均保存紧凑统计；
- 不保存 raw events，raw_events_saved=false 且 events 为空；
- 同一 workload 的所有 rank 必须具有完全相同的 stats 与 event histograms；
- 每个 workload 先执行同 shape warmup；
- 实际生成长度必须等于配置的 output length；
- telemetry 以 1 秒间隔记录 GPU 状态；
- runner 在 GPU 非空闲、准入缺失或已发现错误时拒绝启动或标记完成。

观测到的原语为：

- all_reduce；
- fused_allreduce_residual_rmsnorm。

正式 Phase 13 直方图键必须保留 (op, payload)，不能只聚合成
payload 到 count。可额外派生 collective_family=all_reduce 做等效
bytes/rounds 折算，但不得丢掉 fused op 的原始身份。

## 5. 完整性审计

2026-08-04 重新执行全目录审计与仓库内 validator：

- 18/18 DONE；
- 18/18 result.jsonl；
- 18/18 run.log；
- 18/18 validate.log；
- 18/18 telemetry.csv；
- 585/585 原始记录；
- 195/195 独立配置，每个配置恰好出现 3 次；
- 所有配置实际生成长度与请求长度一致；
- 所有 workload 的 rank 集合为 0 到 TP-1；
- 所有 TP rank 的 stats 与直方图完全一致；
- 全部为 histogram-only，raw events 保存数为 0；
- 未发现 OOM、Traceback、CPU fallback、NCCL error 或 rank mismatch；
- admission TP2/TP8 与 18 个正式单元均通过重新验证。

机器可读审计结果见 audit_summary.json，正式复验输出见
revalidate_admission.log 与 revalidate_pattern.log。

## 6. 当前可复核的结构事实

所有 TP 和 repeat 上均一致观察到：

| 阶段 | 每条配置 collective calls | payload 支撑点数 |
|---|---|---:|
| Prefill | 97 | 11 |
| Decode M=32/128/512 | 3,007 / 12,319 / 49,567 | 5 |

这些是当前 runner 的完整阶段统计结果。Calls 在 TP2/4/8 上保持不变不代表通信
成本保持不变；后续必须结合 group size 计算 equivalent bytes 与 equivalent
rounds，并保留 backend/op 信息。

## 7. 结论边界

本阶段可以准确声称：

1. Qwen3-30B-A3B 已在单节点 B200 上完成 TP2/4/8 的规则
   Prefill/Decode PatternDemand 采集；
2. 585 条原始记录完整对应 195 个独立配置的三次重复；
3. 当前 TP runtime 同时产生 all_reduce 与
   fused_allreduce_residual_rmsnorm；
4. 数据完整性足以进入三模型 Phase 13 分析。

本阶段不能声称：

- 已测量 Qwen3-30B 的 all-rank collective 时间或端到端推理时间；
- 已验证 expert-parallel All-to-All 或 MoE routing 跨卡通信；
- 已完成 L2/L3 跨节点实验；
- 585 条重复记录是 585 个独立训练样本；
- 仅由本阶段已经证明第三模型或未见模型的预测精度。

## 8. 正式产物与复验

正式产物包括 README、audit_summary.json、manifest.sha256、两份全目录复验日志、
TP2/TP8 admission 结果，以及 TP2/4/8 的三次 Prefill/Decode 正式数据、运行日志、
验证日志和 telemetry。

重新验证不会覆盖已经通过的正式数据：

    bash scripts/run_qwen3_30b_a3b_admission.sh all
    bash scripts/run_qwen3_30b_a3b_pattern_dataset.sh all

## 9. 下一步

Phase 13 首先聚合每个配置的三次重复，再进行 Qwen3-8B、
DeepSeek-V2-Lite 与 Qwen3-30B-A3B 的联合分析。正式实现必须：

1. 使用 (op, payload) 到 count 作为精确直方图；
2. 分别报告 logical calls/payload、equivalent bytes 与 equivalent rounds；
3. 自动寻找近等总 payload、不同 calls/op/TP/模型的对照；
4. 以 workload 组为单位进行 leave-one-model-out，不允许 repeat 泄漏；
5. 将 EP All-to-All 与当前 TP collective 结论严格分开。
