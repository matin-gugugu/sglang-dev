# Phase 1：PatternDemand 契约、通信代价曲线与真实通信时间

## 1. 本轮目标与环境

本轮完成三个任务：

1. 冻结第一阶段 `PatternDemand v1` 的数据契约；
2. 在 B200 单机 NVLink、TP=2 上测量 AllReduce 连续消息尺度代价曲线；
3. 对 Qwen3-8B 的代表性 Prefill/Decode workload 采集 GPU 通信 kernel 时间，校验第一阶段埋点。

远端仓库与环境：

- 仓库：`/sgl-workspace/sglang-src`
- 分支：`experiment/pattern-demand-v0.5.15-clean`
- 基线提交：`95226d5`
- 模型：`/media/ssd1/Qwen3-8B`
- GPU：2 × NVIDIA B200，单节点 NVLink
- 并行配置：TP=2
- 推理模式：eager，关闭 Prefill/Decode CUDA Graph
- 重复次数：3

## 2. PatternDemand v1 数据契约

冻结的核心统计口径：

- `count`：一个通信组内 collective 的 group-level 调用次数；取代表 rank，不跨 rank 求和，也不是 NCCL ring step 数。
- `input_payload_bytes`：代表 rank 单次调用的逻辑输入 tensor 字节数；不是所有 rank 总和，不是等效字节，也不是实际链路流量。
- `group_level_message_histogram`：第一阶段的规范标签。
- `derived`：仅为便于分析而保存的冗余结果，必须能由 histogram 重建。
- `output_lens_per_request`：存在时表示每个请求实际固定生成的 token 数，不能用可能提前遇到 EOS 的 `max_tokens` 代替。

验证结果：

```text
PatternDemand v1 validation PASSED: 635 row(s) across 11 file(s)
```

对应文件：

- `experiment-results/phase0/schemas/pattern_demand_v1.schema.json`
- `experiment-results/phase0/docs/pattern_demand_v1_contract.md`
- `scripts/validate_pattern_demand_schema.py`

## 3. TP=2 单节点连续 AllReduce 代价曲线

### 3.1 NCCL 基线

测量范围为 8 KiB–1 GiB，额外包含实验中出现的 48 KiB；每个尺寸预热 30 次、计时 100 次、独立重复 3 轮。计时值为两个 rank 中较慢者的 CUDA Event call-envelope。

主要现象：

- 8 KiB–8 MiB 处在约 41–48 μs 的启动开销平台；
- 随 payload 增大，有效带宽持续提高；
- 16 MiB 后延迟开始明显随 payload 增长；
- 1 GiB 时算法带宽约 564 GB/s。

因此，链路代价不能使用一个全局固定带宽；小消息由启动开销主导，大消息逐渐由传输带宽主导。

### 3.2 SGLang 实际自定义后端

Qwen3-8B TP=2 的小消息实际走 `CustomAllReduceV2`，不是 NCCL。补测范围为 8 KiB–16 MiB：

- 8 KiB–4 MiB：`ONE_SHOT_PUSH`；
- 8–16 MiB：`ONE_SHOT_PULL`；
- CUDA Event call-envelope 中位数约 48–59 μs。

该 call-envelope 包含 eager 调用过程中的 host dispatch gap，不能与 profiler 中“纯 GPU kernel duration”混为同一标签。预测模型必须显式保留 `communication_backend`、`algorithm`、`execution_mode`、`op`、`TP/group_size`、`payload_bytes` 和 `topology`。

对应结果目录：

- `experiment-results/phase1/b200_tp2_nvlink_allreduce/`
- `experiment-results/phase1/b200_tp2_nvlink_allreduce_sglang_custom/`

## 4. 真实推理通信时间与埋点校验

GPU-only PyTorch profiler trace 用于提取通信 kernel。当前 Qwen3-8B TP=2 的实际 kernel 包括：

- `all_reduce_one_shot_push_kernel`
- `all_reduce_one_shot_kernel`
- 大于自定义后端上限时的 `ncclDevKernel_AllReduce`

所有代表性 workload 中，PatternDemand 的 group-level calls 与匹配到的 GPU collective kernel 数量均精确一致。这证明当前第一阶段没有跨 rank 重复累计，`count` 可以作为预测模型的可信输入。

### 4.1 等总 payload、不同消息形态的 Decode 对照

三种 workload 均满足 B=8、L=512、每批总输出 token 数为 128，Decode 逻辑 payload 总量为 71,761,920 bytes，即 68.44 MiB。

| 形态 | 每请求输出长度 | Group-level calls | `calls × median kernel` | Trace 总通信时间中位数 | Trace CV |
|---|---|---:|---:|---:|---:|
| uniform | `[16,16,16,16,16,16,16,16]` | 1,095 | 4.91 ms | 4.96 ms | 0.001 |
| mixed | `[4,4,8,8,16,16,36,36]` | 2,555 | 11.36 ms | 21.02 ms | 0.353 |
| longtail | `[4,4,4,4,4,4,52,52]` | 3,723 | 16.56 ms | 22.37 ms | 0.204 |

核心结论：

> 总通信字节完全相同，不代表通信代价相同。输出长度分布改变 active batch 的下降过程，进而改变每步 payload、调用次数和消息直方图；调用次数从 1,095 增加到 2,555/3,723 后，通信时间显著上升。

每次 kernel 的中位数在三种形态下都稳定在约 4.45 μs。mixed/longtail 的 trace 总量出现额外长尾，是因为少量同步 kernel 在等待对端或受执行时序影响，最长达到约 0.9–1.5 ms。它说明：

- `calls × 连续代价曲线` 适合作为结构化基础项；
- rank skew、通信/计算重叠、调度时序应作为残差或尾部特征建模；
- 只有 3 次重复时，不应把 noisy trace total 当作无噪声标签；应同时保存中位数、P95/P99、最大值与 CV。

### 4.2 Prefill 真实通信曲线

每个 workload 都有 73 次 AllReduce，改变输入长度得到不同的单次 payload：

| 输入长度 L | 单次 payload | 实际后端 | 73 次 kernel 时间中位数 | CV |
|---:|---:|---|---:|---:|
| 128 | 1 MiB | Custom one-shot push | 0.559 ms | 0.174 |
| 512 | 4 MiB | Custom one-shot push | 1.080 ms | 0.100 |
| 1,024 | 8 MiB | Custom one-shot | 1.718 ms | 0.689 |
| 2,048 | 16 MiB | Custom one-shot | 2.750 ms | 0.038 |
| 4,096 | 32 MiB | NCCL fallback | 6.391 ms | 0.020 |

该结果同时证明：

1. payload 增大时通信时间不是一个全局线性系数；
2. 4 MiB/16 MiB 附近存在算法或后端边界；
3. 相同 `op × TP × topology` 下仍需连续 payload 代价曲线；
4. backend/algorithm 切换属于结构性离散特征，不能完全交给三个硬桶吸收。

8 MiB 的 CV 较高，原因是 r1 出现同步长尾；后续应补到至少 10 次重复，报告 bootstrap 置信区间。

## 5. 对原始论文设计的判断

原始“两阶段”设计可行，但建议把第二阶段从三个固定桶参数升级为：

```text
C(op, payload_bytes, group_size, topology, backend, algorithm, execution_mode)
```

第一阶段仍输出与拓扑无关的消息需求分布；三个桶可以用于论文可视化或粗粒度调度，但预测模型输入应保留连续 payload histogram。最终结构化基础项为：

```text
T_base = Σ count_bin × C(op, payload_bin, TP, topology, backend, ...)
```

神经网络不直接绕过 PatternDemand 预测总时间，而是拟合：

```text
T_comm = T_base + residual(workload, model, overlap, rank_skew, runtime_state)
```

这样既保留机理可解释性，也能吸收同步等待、通信重叠和运行时抖动造成的非线性。

## 6. 结果文件

汇总结果：

- `experiment-results/phase1/summary/phase1_summary.json`
- `experiment-results/phase1/summary/collective_curve_summary.csv`
- `experiment-results/phase1/summary/decode_equal_payload_summary.csv`
- `experiment-results/phase1/summary/prefill_inference_curve_summary.csv`
- `experiment-results/phase1/summary/phase1_communication_results.png`

真实推理结果：

- `experiment-results/phase1/qwen3_8b_tp2_inference_comm/representative/decode_equal_payload/`
- `experiment-results/phase1/qwen3_8b_tp2_inference_comm/representative/prefill_payload_curve/`

每个重复目录包含 `result.jsonl`、`comm_ground_truth.jsonl`、`run.log` 和 GPU-only `*.trace.json.gz`。单个 trace 约 30 KiB–1.7 MiB。

## 7. 当前边界与下一步

当前结论只覆盖 Qwen3-8B、TP=2、B200 单节点 NVLink、eager 模式。下一步按优先级为：

1. 将 Prefill 8 MiB 和 mixed/longtail Decode 增加到至少 10 次重复，稳定尾部统计；
2. 在 TP=4/8 上重复同一组等总 payload 对照，加入 group size 与 rounds；
3. 测量 L2 同机架跨节点真实曲线，再处理 L3 跨机架；
4. 对比 `total bytes only`、三硬桶、连续 histogram 三种时延预测基线；
5. 最后再引入新模型，检验从 Qwen3-8B 学到的 PatternDemand 结构能否迁移。
