# PatternDemand 的 TP / PP / PD Teacher

## 1. 先明确 teacher、Hfull、H0 和 DNN 的关系

PatternDemand 里的 **teacher 不是神经网络**，而是经过代表性 GPU sentinel
验证的确定性标签生成器。它复现冻结的 SGLang 调度与通信语义，把一个请求窗口转换成
拓扑无关的逻辑通信直方图。

最终的学习关系是：

```text
完整、有序请求窗口
        ↓  仅离线使用
确定性 teacher
        ↓
Hfull：训练/评估真值

低维历史画像 + 模型结构 + 固定并行配置 + 固定策略
        ↓
确定性构造 32 个伪请求
        ↓
同一执行语义的 teacher
        ↓
H0：可解释结构基线

H0 + DNN residual ≈ Hfull
```

因此：

- `Hfull` 输入完整、有序的请求列表，是训练和盲测打开后的真值；
- `H0` 不读取完整请求列表，只从低维画像恢复 32 个伪请求，再调用相同的结构规则；
- DNN 是 student，只学习 `Hfull - H0` 中没有被结构先验解释的部分；
- teacher 输出逻辑通信需求，不直接预测通信时间；
- 物理时间由冻结直方图与 L1/L2/L3 实测通信曲线卷积得到。

三种 teacher 的根本区别是它们统计的通信对象不同：

| Teacher | 通信对象 | 一次逻辑 call 的定义 |
|---|---|---|
| TP | TP rank 之间的 collective | 一次逻辑 AllReduce |
| PP | 相邻 pipeline stage 之间的 activation | 一次 proxy tensor 发送 |
| PD | Prefill 实例向 Decode 实例发送 KV | 一次逻辑 KV chunk 发送 |

最终三者都输出每 1000 请求归一化的 12-bin `calls` 和 `logical_bytes`
直方图，但内部调度器、消息公式和配置参数不同。

---

## 2. 三种 teacher 的共同输入

### 2.1 Hfull 输入

Hfull 使用完整、有序的请求窗口：

```text
requests = [
    (input_tokens_0, output_tokens_0),
    (input_tokens_1, output_tokens_1),
    ...
]
```

请求顺序不能丢失，因为 FCFS admission、长 prompt 的位置、chunk 边界、lane
填充和请求完成顺序都可能改变最终直方图。

当前实验冻结的是 `fixed-draining` 语义：处理完整窗口并 drain，不把真实 arrival
timestamp 作为在线调度事件重放。因此结论不能自动外推到 arrival-aware 在线服务。

### 2.2 H0 输入

H0 只使用部署时允许的低维信息，主要包括：

- 4×4 输入/输出长度联合分布；
- 输入和输出长度的均值及分位数；
- 请求数、RPS、interarrival CV、burst、Fano 等画像特征；
- 模型结构；
- 固定的 TP/PP/PD 配置和执行策略。

32 个伪请求的初始代表点为：

```text
input representatives  = [64, 320, 1280, 4096]
output representatives = [8, 24, 48, 96]
```

先按照 4×4 联合分布为 32 个位置分配请求，再缩放代表长度，使伪请求的均值尽量匹配
原画像。该过程不能恢复完整请求顺序、极端请求位置或精确 chunk 边界，这部分偏差正是
residual DNN 的学习目标。

### 2.3 模型结构和执行合同

不同 teacher 使用不同结构字段：

- TP/PP 主要使用 `num_hidden_layers`、`hidden_size` 和 `dtype_bytes`；
- PD 使用完整 KV 结构，包括层数、KV head、head dimension，或 MLA 的
  `kv_lora_rank` 与 `qk_rope_head_dim`；
- TP/PP/PD 都要求并行度、调度策略、chunk/page 规则、cache/overlap 开关等执行合同被冻结。

### 2.4 输出合同

每个 teacher 最终产生：

```text
calls_by_12bin
logical_bytes_by_12bin
total_calls_per_1000
total_logical_bytes_per_1000
```

每个事件先按 payload size 落入一个 bin：

```text
calls[bin]         += logical_call_count
logical_bytes[bin] += payload_bytes × logical_call_count
```

再乘以：

```text
normalization_scale = 1000 / request_count
```

早期 TP structural teacher 使用 4 KiB–512 MiB 的 12 个几何 bin，PP 使用
4 KiB–8 GiB。后续统一实验冻结了共享的 12-bin 合同。阅读旧阶段结果时应以该阶段
contract 中的 `bin_edges_bytes` 为准，不应把“测量支持范围”“实际非空范围”和
“最终统一 bin 合同”混为一谈。

---

## 3. TP teacher

### 3.1 TP teacher 统计什么

Tensor Parallel 把一层计算切到多个 TP rank。一次模型 forward 会产生若干 collective。
PatternDemand 的 canonical TP 口径统计逻辑 AllReduce，不把 ring/tree 内部的每个网络
step 展开成新的 call。

TP teacher 回答：

> 当前 prefill 或 decode forward 有多少 active token？这些 token 会形成多大的逻辑
> collective，以及一共发生多少次 collective？

### 3.2 主要输入参数

| 参数 | 作用 |
|---|---|
| `requests` | 完整、有序的 `(input_tokens, output_tokens)` 请求列表 |
| `max_batch_size` | 一个 fixed-draining batch 最多容纳的请求数 |
| `max_prefill_tokens` | 一个 prefill batch 的 prompt-token 总预算 |
| `phase` | `prefill` 或 `decode` |
| `num_hidden_layers = L` | 决定每次 forward 的逻辑 collective 数 |
| `hidden_size = H` | 决定每个 active token 的 payload 元素数 |
| `dtype_bytes` | 每个元素的字节数，当前 BF16 通常为 2 |
| `tp_size` | TP2/4/8；标识 collective group size，并选择后续物理曲线 |
| `request_count` | 用于归一化到每 1000 请求 |

冻结的三种 TP 策略为：

| 策略 | `max_batch_size` | `max_prefill_tokens` |
|---|---:|---:|
| `latency` | 4 | 8192 |
| `balanced` | 8 | 32768 |
| `throughput` | 16 | 65536 |

模型结构先转换成两个核心常量：

```text
calls_per_forward     = 2 × num_hidden_layers + 1
bytes_per_active_token = hidden_size × dtype_bytes
```

以 Qwen3-8B 为例：

```text
L = 36
H = 4096
dtype_bytes = 2

calls_per_forward      = 73
bytes_per_active_token = 8192 bytes
```

### 3.3 TP teacher 如何工作

#### 第一步：按原请求顺序组成 fixed-draining batch

```text
current_batch = []

for request in requests:
    如果 current_batch 已非空，并且：
        len(current_batch) 已达到 max_batch_size
        或加入该请求后 prompt token 总量超过 max_prefill_tokens
    则先提交 current_batch，再开始新 batch

    将 request 加入 current_batch
```

这里的 token budget 是 batch 边界条件。单个请求本身仍会被加入空 batch，因此不能把它
简单理解成“超过预算的请求被删除”。

#### 第二步：生成 prefill 通信事件

对每个 batch：

```text
active_tokens = Σ input_tokens
payload_per_collective = active_tokens × bytes_per_active_token
logical_calls = calls_per_forward
```

#### 第三步：生成 decode 通信事件

Prefill 已经生成第一个 token，因此 decode 从 `step = 1` 开始：

```text
for step = 1 ... max(output_tokens) - 1:
    active_requests = count(output_tokens > step)
    active_tokens = active_requests
    payload_per_collective = active_tokens × bytes_per_active_token
    logical_calls = calls_per_forward
```

请求输出越长，在更多 decode step 中保持 active；因此 `output_tokens` 直接决定 decode
通信事件数量和 active-lane 形状。

#### 第四步：分桶和归一化

每个 `(payload, logical_calls)` 落入 12-bin calls/bytes 直方图，并归一化到每 1000
请求。

### 3.4 TP structured teacher

Phase30 将 TP teacher 拆成两层：

```text
完整请求窗口
    ↓
与模型无关的 scheduler event teacher
    ↓
模型结构 adapter
    ↓
12-bin calls / logical_bytes
```

中间事件向量有 62 维：

- 23 个 prefill token-sum category 的 batch count；
- 23 个对应 category 的 token mass；
- decode active lanes 1–16 的 step count。

随后才使用 `2L+1` 和 `H × dtype_bytes` 把调度事件映射为具体模型的通信需求。
这样可以把“流量如何被调度”和“该模型每次 forward 产生什么通信”分开。

### 3.5 `tp_size` 为什么没有直接乘进逻辑直方图

在当前 canonical teacher 中，同一模型、流量和策略下，TP2/4/8 的：

- `2L+1` 不变；
- representative-rank logical input payload 不变；
- 逻辑直方图可以相同。

`tp_size` 真正改变的是 collective 参与 rank 数、算法实现及物理代价曲线。因此不能把
逻辑 calls 或 bytes 简单乘以 `tp_size - 1`；TP size 的影响在后续
“逻辑需求 × TP group-size 物理曲线”中体现。

---

## 4. PP teacher

### 4.1 PP teacher 统计什么

Pipeline Parallel 把模型层分到多个 pipeline stage。每次 forward 需要把 activation
从 stage `i` 发送到 stage `i+1`。

PP teacher 回答：

> 真实 pipeline scheduler 在某次 lane visit 上运行了哪些请求、多少 active token，
> 从而在每个相邻 stage boundary 上产生多少 proxy tensor 消息？

PP 不能只靠静态 batch 公式，因为：

- 每个 pipeline lane 都有独立 running batch；
- 请求完成后才会腾出 slot；
- 新请求按照 FCFS 重新填入；
- 长 prompt 会被 chunk；
- 未完成 chunk 可跨 lane 继续；
- scheduler budget 按 page rounding 扣除；
- `prefill` 与“过滤刚完成 decode 请求”的执行顺序会改变下一次 batch。

因此正式 PP teacher 是一个离散状态机。

### 4.2 主要输入参数

| 参数 | 作用 |
|---|---|
| `requests` | 完整、有序的 `(input_tokens, output_tokens)` 请求列表 |
| `pp_size` | pipeline stage 数，同时决定 PP loop 的 lane 数 |
| `max_microbatch` | 每个 lane 最多同时运行的请求数，冻结为 1/4/16 |
| `chunk_tokens` | 每次 prefill scheduler pass 的预算，冻结为 4096 |
| `page_size` | scheduler 扣预算的页粒度，冻结为 64 |
| `hidden_size` | activation 每 token 的元素数 |
| `dtype_bytes` | 每个 activation 元素的字节数 |
| `proxy_tensor_count` | 每次 forward、每个 boundary 的 proxy 消息数，冻结为 2 |
| scheduler 开关 | FCFS、无 radix、无 overlap、无 mixed chunk、async depth 0、ignore EOS |

对 Qwen3-8B：

```text
bytes_per_active_token = hidden_size × dtype_bytes
                       = 4096 × 2
                       = 8192 bytes
```

### 4.3 PP teacher 的状态

每个请求维护：

```text
input_tokens
output_tokens
input_position
generated_tokens
finished
```

每个 lane 维护：

```text
running_request_ids
pending_batch
batch_is_full
```

全局维护：

```text
FCFS waiting queue
chunked_request_id
pp_size 个 lane
```

### 4.4 PP teacher 如何工作

每次访问一个 lane 时，按以下顺序执行。

#### 1. 接收该 lane 上一次 forward 的结果

- final prefill chunk 完成时，为该请求计入第一个输出 token；
- decode 完成时，为每个 active request 增加一个输出 token。

#### 2. 合并完成 prefill 且尚未结束的请求

只有 final-prefill 且未完成的请求会进入该 lane 的 running batch。中间 chunk 和已经
一 token 完成的请求不会被合并。

#### 3. 优先尝试 prefill

```text
优先继续全局 chunked_request_id
然后按 FCFS 从 waiting queue 加新请求
最多使用 max_microbatch - running_batch_size 个 slot
一个 pass 最多消耗 4096 token budget
```

实际 activation payload 使用真实 `active_tokens`：

```text
payload = active_tokens × hidden_size × dtype_bytes
```

但 scheduler budget 按页扣除：

```text
charged_tokens = ceil(tokens / 64) × 64
```

#### 4. 如果没有 prefill，再处理 decode

先过滤已完成请求，再对剩余 running batch 做一次 decode forward，每个活跃请求生成
一个 token。

这里存在一个已经由 Phase25B 修正的重要顺序：

```text
先尝试 prefill
再过滤刚完成的 decode 请求
```

因此某个请求刚结束、slot 刚腾出时，当前 lane visit 可能先产生一个较小 decode
batch；下一次 visit 才重新填入。静态分组 teacher 无法复现这个现象。

### 4.5 PP 事件如何变成通信直方图

每个 forward event 记录：

```text
phase
active_requests
active_tokens
```

单个 boundary 上：

```text
payload_per_proxy = active_tokens × hidden_size × dtype_bytes
calls_per_boundary = forward_event_count × proxy_tensor_count
```

teacher 同时保留 representative boundary 和 pipeline-wide 口径：

```text
pipeline_calls = per_boundary_calls × (pp_size - 1)
pipeline_bytes = per_boundary_bytes × (pp_size - 1)
```

这里乘的是 boundary 数，不是修改单条消息的 payload。

---

## 5. PD teacher

### 5.1 PD teacher 统计什么

Prefill/Decode Disaggregation 把 Prefill 和 Decode 放在不同实例。Prefill 完成 prompt
的一段后，需要把对应 KV cache 从 P 实例发送到 D 实例。

当前合同是纯：

```text
P: TP=1, PP=1
D: TP=1, PP=1
```

因此 PD teacher 只统计 `P → D KV`，不统计 P 或 D 内部的 TP/PP 通信。

PD teacher 回答：

> Prefill scheduler 按 FCFS、wave、chunk 和 page 规则把每个 prompt 切成了哪些 KV
> page 区间？每个区间需要发送多少逻辑 KV bytes？

### 5.2 主要输入参数

| 参数 | 作用 |
|---|---|
| `requests` | 有序 `(prompt_tokens, output_tokens)` 请求列表 |
| `wave_size` | 每个原子 wave 最多 64 个请求 |
| `chunk_tokens` | 每个 scheduler pass 的 prefill budget，冻结为 4096 |
| `page_size_tokens` | KV 页粒度；普通模型为 1，DeepSeek MLA 为 64 |
| `kv_bytes_per_page` | 一个请求的一页 KV 在全模型所有层上的逻辑字节数 |
| `schedule_policy` | FCFS |
| `max_running_requests` | 冻结为 64 |
| batch barrier | 让整个 wave 按原顺序原子放行 |
| backend / transport | Mooncake + RDMA；影响 GPU 语义验证和物理曲线，不改变 teacher 的逻辑输出定义 |

`output_tokens` 在纯 PD teacher 中主要是合同和审计字段。该 teacher 只发送 prompt KV，
最终 decode 生成多少 token 不改变本次 `P → D` KV 传输。

### 5.3 KV 字节公式

普通 MHA/GQA 模型：

```text
kv_bytes_per_token =
    num_hidden_layers
  × 2                         # K 和 V
  × num_key_value_heads
  × head_dim
  × dtype_bytes

kv_bytes_per_page = kv_bytes_per_token × page_size_tokens
```

DeepSeek MLA：

```text
kv_bytes_per_token =
    num_hidden_layers
  × (kv_lora_rank + qk_rope_head_dim)
  × dtype_bytes

kv_bytes_per_page = kv_bytes_per_token × page_size_tokens
```

当前冻结结构中的例子：

```text
Qwen3-8B:
    page_size_tokens = 1
    kv_bytes_per_token = 147456

DeepSeek-V2-Lite:
    page_size_tokens = 64
    kv_bytes_per_token = 31104
    kv_bytes_per_page = 1990656
```

### 5.4 PD teacher 如何工作

#### 第一步：把完整窗口切成有界 wave

保持原始顺序，切成最多 64 请求的连续、不重叠切片：

```text
wave0: requests[0:64]
wave1: requests[64:128]
...
```

wave `k` 必须完全 drain 后才提交 wave `k+1`；4096-token budget 在每个新 wave
重置。请求和 chunk 都不能跨 wave 边界。

#### 第二步：在 wave 内按 FCFS 生成 KV chunk

最终六模型 page-aware teacher 的核心过程是：

```text
for each wave:
    while 仍有请求:
        budget = 4096

        while budget > 0 and 仍有请求:
            remaining_prompt = prompt_tokens - token_offset
            send_tokens = min(remaining_prompt, budget)

            如果不是最终 chunk：
                send_tokens 必须按 page_size_tokens 对齐

            page_count = ceil(send_tokens / page_size_tokens)
            charged_tokens = page_count × page_size_tokens

            如果 charged_tokens 超过剩余 budget：
                缩小到 budget 可容纳的完整页

            logical_bytes = page_count × kv_bytes_per_page
            记录一次逻辑 KV chunk send
            budget -= charged_tokens
```

Phase40 的 Qwen3-8B `page_size=1` 语义与上述规则等价；Phase47 为支持
DeepSeek-V2-Lite 的 `page_size=64`，正式冻结了 page-aligned budget 扣减规则。

#### 第三步：记录逻辑事件

每个 chunk 记录：

```text
rid
wave_index
scheduler_pass
chunk_index
page_start
page_end
kv_page_count
logical_bytes
```

并定义：

```text
logical_calls = 1 per KV chunk
```

最后按 `logical_bytes` 进入 12-bin 直方图，并归一化到每 1000 请求。

### 5.5 为什么 descriptor 数不是 calls 数

一次真实 Mooncake `batch_transfer_sync` 内部可以包含多个 descriptor：

- 普通 MHA/GQA 通常为 `2 × num_hidden_layers` 个 K/V descriptor；
- MLA 通常为 `num_hidden_layers` 个 descriptor。

但 PatternDemand 统计的是 sender 侧的一次逻辑 KV chunk send，因此整个 chunk 仍只记
一个逻辑 call。descriptor 数量会影响物理实现，但不能被重复计成 PatternDemand calls。

### 5.6 PD teacher 明确不统计什么

- bootstrap/control metadata；
- transport header 或 NIC packet；
- receiver 侧副本；
- Mooncake 内部 descriptor 作为额外 calls；
- 网络传输时间；
- Decode 计算和输出 token；
- 拥塞、排队、计算通信重叠；
- P/D 实例内部的 TP 或 PP 通信。

这些边界保证 PD 输出仍是拓扑无关的逻辑通信需求。物理延迟由 Phase51 的六模型
L1/L2/L3 Mooncake/RDMA 曲线单独提供。

---

## 6. TP / PP / PD 对比总结

| 维度 | TP teacher | PP teacher | PD teacher |
|---|---|---|---|
| 核心问题 | 一次 forward 有多少 active token？ | 某次 lane visit 实际运行了多少 token？ | prompt 被切成了哪些 KV page 区间？ |
| 调度形态 | fixed-draining batch | 多 lane 离散状态机 | 有界 wave + FCFS chunk/page 状态机 |
| payload 公式 | `active_tokens × H × dtype` | `active_tokens × H × dtype` | `page_count × KV bytes/page` |
| calls 公式 | `forward_count × (2L+1)` | `forward_count × 2 proxy/boundary` | `1 call / KV chunk` |
| 输入长度作用 | 决定 prefill batch token mass | 决定 prefill chunk、lane 填充 | 决定 KV page/chunk 数量 |
| 输出长度作用 | 决定 decode active steps | 决定请求完成、slot 释放 | 不改变 prompt KV 发送 |
| 并行度作用 | `tp_size` 主要选择物理曲线 | `pp_size` 决定 lane 和 boundary 数 | 当前固定纯 P1→D1 |
| 输出 | 12-bin calls/bytes | 12-bin calls/bytes | 12-bin calls/bytes |

最简洁地说：

```text
TP teacher
    active tokens per forward
    → active_tokens × hidden_size × dtype
    → 每次 forward 产生 2L+1 个逻辑 collective

PP teacher
    scheduler 每次 lane visit 的 active tokens
    → active_tokens × hidden_size × dtype
    → 每个相邻 boundary 发送 2 个 proxy tensor

PD teacher
    Prefill scheduler 产生的 KV page chunks
    → page_count × full-model KV bytes/page
    → 每个 chunk 是一次 P→D 逻辑发送
```

三者都只回答“会产生多少条、什么大小的逻辑消息”，不直接回答“需要多少微秒”。只有
把冻结的 12-bin 直方图代入对应的 TP、PP 或 PD L1/L2/L3 物理曲线后，才能得到
communication-only cost，并进一步进入 placement 或 scheduler 决策。

---

## 7. 对应实现与实验依据

主要实现和冻结合同：

- TP/PP full-window structural teacher：
  [`scripts/build_phase25_full_window_teacher.py`](../../scripts/build_phase25_full_window_teacher.py)
- scheduler-faithful PP teacher：
  [`scripts/build_phase25b_pp_scheduler_teacher.py`](../../scripts/build_phase25b_pp_scheduler_teacher.py)
- compact-profile PP H0 与 32 请求重建：
  [`scripts/build_phase21b_pp_h0.py`](../../scripts/build_phase21b_pp_h0.py)
- TP structured-event contract：
  [`experiment-results/phase30a_tp_structured_event_contract/modeling_contract.json`](../../experiment-results/phase30a_tp_structured_event_contract/modeling_contract.json)
- 纯 PD 基础 teacher：
  [`workflows/patterndemand/phase40_pure_pd_semantics_teacher/contracts.py`](../../workflows/patterndemand/phase40_pure_pd_semantics_teacher/contracts.py)
- 完整窗口 PD 合同：
  [`workflows/patterndemand/phase41_pd_full_window_dataset/experiment.json`](../../workflows/patterndemand/phase41_pd_full_window_dataset/experiment.json)
- 六模型 page-aware PD teacher：
  [`workflows/patterndemand/phase47_pd_five_model_teacher_validation/contracts.py`](../../workflows/patterndemand/phase47_pd_five_model_teacher_validation/contracts.py)

已完成证据链的总览见：

- [`experiment-results/phase53_tp_pp_pd_conclusion_freeze/docs/PatternDemand_experiment_guide_through_Phase52.md`](../../experiment-results/phase53_tp_pp_pd_conclusion_freeze/docs/PatternDemand_experiment_guide_through_Phase52.md)
- [`docs/patterndemand/kaiti_patternDemand实验.md`](./kaiti_patternDemand实验.md)

## 8. 配套图集

- [图集说明与口径](./patterndemand_figures/README.md)
- [全部图总览](./patterndemand_figures/contact_sheet.png)
- [图集压缩包](./patterndemand_figures_bundle.zip)
- [样例选择审计](./patterndemand_figures/audit/sample_selection.csv)
- [图集验证报告](./patterndemand_figures/audit/validation_report.md)
- [可复现绘图脚本](./patterndemand_figures/generate_patterndemand_figures.py)
