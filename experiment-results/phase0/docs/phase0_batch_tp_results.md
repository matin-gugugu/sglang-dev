# Phase 0：Batch Size 与 TP 规模实验结果（r0）

## 1. 实验口径

- 日期：2026-07-22
- 设备：8 × NVIDIA B200 183 GB
- 模型：Qwen3-8B、DeepSeek-V3.2
- 固定工作负载：`input_len=2048`、实际生成 `output_len=32`、`batch_size=4`（TP 主效应）
- Batch 主效应：`batch_size={1,2,4,8,16}`
- TP 主效应：`TP={1,2,4,8}`；DeepSeek-V3.2 受模型容量限制，只能得到 TP4、TP8
- CUDA Graph 关闭，保证 Python 通信埋点能观察到每次 collective
- `count` 为 group-level collective 次数；`payload_bytes` 为代表 rank 的逻辑消息大小
- Decode 实际通信步数为 `output_len-1=31`

所有成功样本均通过：rank 直方图一致、stats 聚合守恒、实际输出长度一致。

## 2. Batch Size 主效应

### 2.1 Qwen3-8B，TP2

| Batch | Prefill 单次 payload | Prefill calls | Decode 单次 payload | Decode calls | Prefill ms | Decode median ms/step |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 MiB | 73 | 8 KiB | 2263 | 25.41 | 23.56 |
| 2 | 32 MiB | 73 | 16 KiB | 2263 | 311.56* | 23.58 |
| 4 | 64 MiB | 73 | 32 KiB | 2263 | 58.72 | 23.95 |
| 8 | 128 MiB | 73 | 64 KiB | 2263 | 112.14 | 23.81 |
| 16 | 256 MiB | 73 | 128 KiB | 2263 | 228.58 | 23.69 |

`*` Batch=2 的 Prefill 延迟是明显离群值，当前只有 r0，不能用于拟合，需补 r1/r2。

结论：calls 不变，Prefill 和 Decode 的单次逻辑 payload 与 batch 严格成比例。Decode 时延几乎不随 batch 增长，说明在 8–128 KiB 区间，增加 payload 后计算/通信吞吐利用率提高，不能用“payload 线性增长 ⇒ 时间线性增长”解释。

### 2.2 DeepSeek-V3.2，TP8

| Batch | Prefill Pattern | Prefill 单次 payload | Decode Pattern/31 steps | Decode 单次 payload | Prefill ms | Decode median ms/step |
|---:|---|---:|---|---:|---:|---:|
| 1 | 2 AR + 121 fused AR | 28 MiB | 62 AR + 3751 fused AR | 14 KiB | 194.52 | 226.96 |
| 2 | 123 AR | 56 MiB | 62 AR + 3751 fused AR | 28 KiB | 286.77 | 228.49 |
| 4 | 123 AR | 112 MiB | 62 AR + 3751 fused AR | 56 KiB | 221.42* | 229.86 |
| 8 | 123 AR | 224 MiB | 62 AR + 3751 fused AR | 112 KiB | 333.92 | 230.39 |
| 16 | 123 AR | 448 MiB | 62 AR + 3751 fused AR | 224 KiB | 731.34 | 232.57 |

`*` Prefill 延迟存在运行次序、首次 kernel/autotune 等噪声，需重复实验后才能使用。

关键发现：Batch=1 到 Batch=2 时，DeepSeek Prefill 从融合 AllReduce 路径切换为普通 AllReduce 路径。这是离散的通信原语迁移，不是人为消息桶制造的非线性。Decode 路径没有发生这一切换。

## 3. TP 主效应

### 3.1 Qwen3-8B，Batch=4

| TP | Pattern | Prefill 单次 payload | Decode 单次 payload | Prefill ms | Decode median ms/step |
|---:|---|---:|---:|---:|---:|
| 1 | 无 TP collective | 0 | 0 | 88.38 | 14.97 |
| 2 | 73 AR / 73 AR per step | 64 MiB | 32 KiB | 56.01 | 23.99 |
| 4 | 73 AR / 73 AR per step | 64 MiB | 32 KiB | 39.52 | 23.38 |
| 8 | 73 AR / 73 AR per step | 64 MiB | 32 KiB | 37.33 | 25.01 |

逻辑 payload 与 calls 在 TP2/4/8 间不变，但 group size 改变。按 ring 近似：

\[
\alpha_{AR}(p)=2(p-1)/p,\qquad \beta_{AR}(p)=2(p-1)
\]

因此 TP2/4/8 的单次等效 bytes 系数分别为 1、1.5、1.75，单次等效 rounds 分别为 2、6、14。Prefill 的计算并行收益逐渐饱和；Decode 的小消息高频通信使 TP2/4/8 均慢于 TP1，而且 TP8 最慢。

### 3.2 DeepSeek-V3.2，Batch=4

| TP | 可行性/Pattern | Prefill 单次 payload | Decode 单次 payload | Prefill ms | Decode median ms/step |
|---:|---|---:|---:|---:|---:|
| 1 | 不可行：权重下界约 637 GB/GPU | — | — | — | — |
| 2 | 不可行：权重下界约 318 GB/GPU | — | — | — | — |
| 4 | 可行，`mem_fraction_static=0.95`；Prefill 123 AR，Decode 62 AR + 3751 fused AR | 112 MiB | 56 KiB | 239.52 | 227.65 |
| 8 | 可行；同上 | 112 MiB | 56 KiB | 221.42 | 229.86 |

TP4 在 `mem_fraction_static=0.85` 时权重加载后仅余约 17 GB，无法分配 KV Cache；提高到 0.95 后成功。TP4 与 TP8 的逻辑 Pattern 相同，但 group size、ring 折算量和实测时间不同：TP8 Prefill 略快，Decode 略慢。

## 4. 对“实验厚度”的评估

当前结果已经从“只有 payload 随 L 线性变化”扩展为三层结构：

1. **可解析的连续部分**：固定模型和执行路径时，payload 对 `batch × token_count × hidden_size × dtype_bytes` 呈线性关系。
2. **离散路径切换**：DeepSeek Prefill 在 Batch=1 与 Batch≥2 之间发生 fused AR → ordinary AR 的原语迁移。
3. **非线性代价交互**：逻辑 Pattern 相同，TP group size 不同，等效 bytes/rounds 和时间不同；Prefill 与 Decode 对增大 TP 的响应方向甚至相反。

因此第一阶段不必宣称所有输出都是非线性的。更严谨的模型是：

\[
D(w,c)=D_{analytic}(w,c;z)+D_{switch}(z),
\]

其中 `z` 是运行时路径（普通/融合原语、chunking、kernel backend 等）。第一阶段预测连续需求和路径类别；第二阶段根据 `op × payload × group_size × topology` 映射为代价。神经网络只学习结构公式未覆盖的切换边界与残差。

## 5. 当前限制与下一轮

- 本轮仅 r0；Pattern 是确定性的，但 latency 不能据此做最终统计结论。
- Batch 实验复制相同长度请求，活动 batch 在整个 Decode 中不变；尚未覆盖混合输出长度导致的 `active_batch(t)` 动态变化。
- DeepSeek TP4 与 TP8 使用不同 `mem_fraction_static`，逻辑 Pattern 可比较，端到端时延需谨慎解释。
- 下一轮优先补 r1/r2，并随机化 batch/TP 点运行顺序；将模型加载、autotune、首轮 kernel 编译与正式测量分离。
- 随后增加混合输出长度 continuous batching，以及 chunked-prefill 边界点，获得更强的阶段内 Pattern 迁移证据。

## 6. 远端产物

```text
/sgl-workspace/results/qwen3_8b_batch_tp2/
/sgl-workspace/results/deepseek_v32_batch_tp8/
/sgl-workspace/results/qwen3_8b_tp_scale/
/sgl-workspace/results/deepseek_v32_tp_scale/
```
