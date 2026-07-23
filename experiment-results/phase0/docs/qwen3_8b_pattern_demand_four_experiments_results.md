# Qwen3-8B PatternDemand 四组实验结果摘要

更新时间：2026-07-23
状态：四组实验均已完成并通过校验

## 1. 统一口径

```text
model: Qwen3-8B
model_path: /media/ssd1/Qwen3-8B
parallel_form: TP
tp_size: 2
CUDA_VISIBLE_DEVICES: 0,1
execution_mode: eager（disable CUDA graph）
count: group-level collective 次数
payload_bytes: 代表 rank 的逻辑输入消息大小
capture_mode: histogram-only（除正确性对照外）
```

主结果位于远端：

```text
/sgl-workspace/sglang-src/experiment-results/phase0/
```

## 2. 实验一：histogram-only 正确性与体积

目录：

```text
qwen3_8b_histogram_only_smoke/
├── full_trace.jsonl
├── full_trace.log
├── histogram_only.jsonl
└── histogram_only.log
```

配置为 `B=1, L=128, M=32`。full-trace 与 histogram-only 的 `stats` 和消息直方图完全相同，均记录 2,336 个通信事件；histogram-only 的 raw events 数为 0。

```text
full_trace.jsonl:       1,262,996 bytes
histogram_only.jsonl:       2,070 bytes
体积缩减:                    610.14 倍
```

作用：证明不保存逐事件 raw JSON 仍能无损保留第一阶段需要的 `phase × op × group × payload × calls` 聚合标签，后续扩大 workload 网格不会再产生约 330 MB 的单轮原始文件。

## 3. 实验二：eager r1/r2 重复性

目录：

```text
qwen3_8b_hist_tp2_eager/
├── hist_main_eager_r0.jsonl
├── hist_main_eager_r1.jsonl
├── hist_main_eager_r2.jsonl
├── pattern_dataset_eager_r0.jsonl
├── pattern_dataset_eager_r1.jsonl
├── pattern_dataset_eager_r2.jsonl
├── pattern_dataset_eager_all.jsonl
├── run_eager_r0.log
├── run_eager_r1.log
├── run_eager_r2.log
└── repeat_validation.json
```

每轮覆盖 60 个 `batch_size × prompt_len × output_len` 组合，共 180 条。r1/r2 使用 histogram-only，单轮结果约 123 KB。

校验结果：

```text
跨 repeat Pattern 完全一致: 60/60
跨 rank 直方图一致:         180/180
stats 与 histogram 守恒:   180/180
实际输出长度一致:           180/180
Prefill latency CV 中位数:  3.13%
Decode latency CV 中位数:   1.05%
```

作用：说明 PatternDemand 是由 workload 和执行配置决定的稳定结构标签；延迟存在运行噪声，因此第一阶段应学习通信需求，时间留给第二阶段和残差模型处理。

## 4. 实验三：混合输出长度的受控 draining batch

目录：

```text
qwen3_8b_mixed_continuous_tp2/
├── main_uniform_r{0,1,2}.jsonl
├── main_mixed_r{0,1,2}.jsonl
├── main_long_tail_r{0,1,2}.jsonl
├── run_uniform_r{0,1,2}.log
├── run_mixed_r{0,1,2}.log
├── run_long_tail_r{0,1,2}.log
├── pattern_dataset_mixed.jsonl
├── validation_summary.json
├── smoke.jsonl
├── smoke.log
└── suite.log
```

三种 profile 均固定 `B=8`、总输出 1,024 token，并在 `L∈{512,2048}` 上重复三次：

| Profile | 每请求输出长度 | Decode calls | Decode payload | L=512 总时延均值 |
|---|---|---:|---:|---:|
| uniform | `[128]×8` | 9,271 | 607,584,256 B | 3,057.15 ms |
| mixed | `[32,32,64,64,128,128,288,288]` | 20,951 | 607,584,256 B | 6,810.64 ms |
| long-tail | `[32,32,32,32,32,32,416,416]` | 30,295 | 607,584,256 B | 9,709.93 ms |

三组总 Decode payload 完全相同，但消息形态分别为：

```text
uniform:
  65,536 B × 9,271 calls

mixed:
  65,536 B × 2,263 calls
  49,152 B × 2,336 calls
  32,768 B × 4,672 calls
  16,384 B × 11,680 calls

long-tail:
  65,536 B × 2,263 calls
  16,384 B × 28,032 calls
```

18/18 条结果通过跨 rank、stats 守恒和实际输出长度校验，Pattern 跨三次重复完全一致。

作用：这是“相同总 payload、不同消息直方图与 calls”的核心匹配实验，直接证明只预测 total bytes 无法唯一确定通信需求形态，第一阶段必须保留消息尺度分布。

## 5. 实验四：chunked-prefill 边界

目录：

```text
qwen3_8b_chunked_prefill_tp2/
├── main_r0.jsonl
├── main_r1.jsonl
├── main_r2.jsonl
├── run_r0.log
├── run_r1.log
├── run_r2.log
├── pattern_dataset_chunked.jsonl
├── validation_summary.json
├── smoke.jsonl
├── smoke.log
└── suite.log
```

配置：

```text
chunk_size = 2048
batch_size ∈ {1,4}
prompt_len ∈ {2047,2048,2049,4095,4096,4097,8191,8192,8193}
output_len = 32
repeat ∈ {0,1,2}
```

54/54 条结果通过跨 rank、stats 守恒、输出长度和 chunk 结构公式校验，Pattern 跨重复完全一致。实测公式为：

```text
Prefill calls = 73 × ceil(L / 2048)
单次消息 payload = batch_size × chunk_tokens × 8192
Prefill total payload = 73 × batch_size × L × 8192
```

边界匹配结果：

| Batch | L 变化 | payload 增量 | calls 变化 | 新增尾块消息 | Prefill 时延变化 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2048→2049 | +0.0488% | 73→146（+100%） | 8 KiB × 73 | +58.66% |
| 1 | 4096→4097 | +0.0244% | 146→219（+50%） | 8 KiB × 73 | +47.54% |
| 1 | 8192→8193 | +0.0122% | 292→365（+25%） | 8 KiB × 73 | +23.16% |
| 4 | 2048→2049 | +0.0488% | 73→146（+100%） | 32 KiB × 73 | +43.41% |
| 4 | 4096→4097 | +0.0244% | 146→219（+50%） | 32 KiB × 73 | +19.82% |
| 4 | 8192→8193 | +0.0122% | 292→365（+25%） | 32 KiB × 73 | +8.96% |

作用：证明第一阶段并非只有平滑的 `L→payload` 线性关系；执行策略会在 chunk 阈值产生离散的 calls 和小尾部消息迁移。`B=4,L=2047` 是该 batch 首个新 shape 的预热离群点，不用于边界匹配结论。

## 6. 代码改动

```text
python/sglang/srt/distributed/comm_profile.py
  - full-trace / histogram-only
  - active_batch_size
  - prefill_chunk_index / prefill_chunk_tokens

python/sglang/benchmark/one_batch.py
  - 每请求固定实际输出长度
  - 受控 draining mixed batch
  - 受控 chunked-prefill
  - 对应 CLI 和结果元数据

scripts/aggregate_pattern_events.py
  - 聚合新直方图维度
  - 同时兼容 raw events 与 histogram-only
  - 校验每请求实际输出长度
```

已通过 Python 编译、Black 格式、`git diff --check`、CPU capture-mode 单测，以及 Qwen3-8B TP=2 实模型烟测与主实验。

## 7. 对论文设计的结论

当前结果已经把第一阶段从“预测总字节”推进为“预测通信事件分布”：

```text
PatternDemand(w,c)
  = phase × op × group_size × payload × calls
    + active_batch / chunk context
```

最有力的两类证据是：

1. Decode：总 payload 完全相同，calls 与消息直方图显著不同；
2. Prefill：边界前后 payload 几乎相同，calls 和小消息数量离散跳变。

这足以支持消息直方图作为中间表征的必要性。当前 `one_batch` 时延包含计算、kernel launch 和通信，不能把时延差全部表述为“纯通信时间”。下一阶段仍需测量 `op × payload × TP × topology → latency` 连续代价曲线，再比较“total bytes 基线”与“PatternDemand 直方图方案”的跨拓扑预测误差。
