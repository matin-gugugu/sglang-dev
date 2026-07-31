# Phase 8：第二模型 DeepSeek-V2-Lite PatternDemand

## 1. 目的

本阶段在 Qwen3-8B 之外引入第二个模型，验证第一阶段通信需求画像不是只对单一
模型成立，并为后续预测器加入 `model` 与模型结构特征提供数据。

第二模型选择
[DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)：

- 约 16B 总参数、2.4B 激活参数，权重约 31.4 GB，能够在当前 8 × B200
  节点上稳定覆盖 TP=2/4/8；
- 使用 MLA、DeepSeekMoE、64 个路由专家和每 token 6 个激活专家，与
  Qwen3-8B 的稠密 GQA 结构形成明显对照；
- 27 层、hidden size 2048、16 个注意力头，隐藏维度、注意力头数和专家数
  均能被 TP=2/4/8 整除；
- SGLang 当前源码包含原生 `DeepseekV2ForCausalLM` 实现。

Qwen3-30B-A3B 暂不作为第二模型。它适合后续作为第三模型，用于同一 Qwen
家族内的“模型规模 + MoE”受控对照；第二模型优先跨模型家族，能以更小的下载
和运行成本获得更强的结构差异。

## 2. 公平对照网格

DeepSeek-V2-Lite 使用与 Qwen3-8B Phase 6 相同的网格：

| 阶段 | batch size | input length | output length | 每个 TP 的点数 |
|---|---|---|---|---:|
| Prefill | 1, 2, 4, 8, 16 | 128, 512, 2048, 8192 | 8 | 20 |
| Decode | 1, 2, 4, 8, 16 | 128, 2048, 8192 | 32, 128, 512 | 45 |

- TP：2、4、8；
- 独立重复：`r0`、`r1`、`r2`；
- 总计：65 workloads/TP × 3 TP × 3 repeats = 585 条结果；
- 每个 workload 先进行同形状预热；
- 每个请求固定实际生成 `output_len` 个 token，不能因 EOS 提前停止；
- 关闭 CUDA Graph，减少图捕获对不同 workload 的混杂影响。

## 3. 第一阶段统计口径

本阶段只保存 `histogram-only` PatternDemand，不保存 raw events：

- `count`：一次 TP group-level collective 调用计一次，不跨 rank 求和；
- `input_payload_bytes`：代表 rank 输入张量的逻辑消息大小；
- `group_size`：该 collective 的 TP rank 数；
- 直方图键保留 `phase × op × group_id × group_size × payload × active_batch`
  以及 Prefill chunk 或 Decode step 范围；
- 每个 TP rank 都输出一个 histogram，验证时要求各 rank 的统计和直方图完全
  一致；
- `generated_output_tokens_per_request` 必须全部等于请求的 `output_len`。

第一阶段结果不直接声明为链路实际流量。等效 bytes 与 rounds 在聚合分析时再由
`op × group_size` 的折算函数计算。

## 4. 需要验证的论文论点

1. 同一 workload 在不同模型结构下会形成不同的 calls、payload 位置和原语
   组合，因此预测器必须包含模型或模型结构特征。
2. `B × L` 或 Decode active batch 对单个张量大小可能近似线性，但模型层数、
   隐藏维度、MoE/注意力实现、阶段和 TP 共同决定完整消息直方图。
3. 同一套连续消息尺度表示能够同时容纳两个模型的 PatternDemand，具备作为
   第二阶段链路代价查询输入的可迁移性。
4. 在总 payload 接近的样本对中，若 calls 或消息尺度分布不同，则仅使用
   total bytes 不能充分表示通信需求。

## 5. 执行与产物

执行脚本：

```bash
bash scripts/run_deepseek_v2_lite_pattern_dataset.sh tp2
bash scripts/run_deepseek_v2_lite_pattern_dataset.sh tp4
bash scripts/run_deepseek_v2_lite_pattern_dataset.sh tp8
```

输出目录：

```text
experiment-results/phase8/deepseek_v2_lite_pattern_demand/
└── tp{2,4,8}/r{0,1,2}/{prefill,decode}/
    ├── result.jsonl
    ├── run.log
    ├── telemetry.csv
    ├── validate.log
    └── DONE
```

`result.jsonl` 是第一阶段正式数据；`run.log` 和 `telemetry.csv` 用于排查运行
异常；`validate.log` 与 `DONE` 表示该目录已通过数据口径校验。

## 6. 当前状态

- [x] 模型选择与配置核验；
- [x] TP=2/4/8 可切分性核验；
- [x] histogram-only 数据集脚本；
- [x] TP=2 dummy-weight 软件链路预检；
- [x] 官方 BF16 完整权重下载与 shard/index 校验；
- [x] TP=2 与 TP=8 官方权重冒烟；
- [x] TP=2/4/8 三重复正式采集；
- [x] Qwen3-8B 与 DeepSeek-V2-Lite 跨模型分析。

正式数据共包含 585 条运行记录和 195 个
`TP × phase × B × L × M` 聚合 workload。18 个阶段目录全部生成 `DONE`，
且通过以下验证：

- 每个 workload 的三个重复完整；
- 每个请求实际生成长度等于 `output_len`；
- 每条记录包含 TP 个 rank profile，rank 编号连续；
- 同一 TP group 内各 rank 的 `stats` 与直方图完全一致；
- 所有记录均为 `histogram-only`，没有保存 raw events。

在 `B=1,L=128,M=8` 下，正式结果为：

| 模型 | Prefill PatternDemand | Decode 每步 PatternDemand |
|---|---|---|
| Qwen3-8B | 73 calls × 1 MiB | 73 calls × 8 KiB |
| DeepSeek-V2-Lite | 55 calls × 512 KiB | 55 calls × 4 KiB |

各 TP group 内所有 rank 的 histogram 完全一致，且每个请求实际生成 8 个
token。55 与 73
分别对应两个模型不同的层数和层内 TP collective 结构；payload 的二倍差异对应
hidden size 2048 与 4096。

跨模型分析获得 390 个 `model × workload` 聚合点和 195 个同 workload
匹配点。Qwen3-8B 相对 DeepSeek-V2-Lite 的 calls 比值中位数为 1.327，总逻辑
payload 比值中位数为 2.655。分析还自动找到 126 组总 payload 相差不超过
3.5%、但消息直方图不同的样本对。其中 `B=1,M=512` 与 `B=16,M=32` 的
Decode 对照在两个模型和 TP=2/4/8 上均复现：总 payload 仅相差 2.94%，calls
却相差 16.48 倍。这支持将连续消息直方图作为核心中间表征，并将 total bytes
仅作为消融基线。

### TP=8 运行说明

DeepSeek-V2-Lite 的 dense MLP intermediate size 为 10944；TP=8 时每卡分片
宽度为 1368 个 BF16 元素，不满足 B200 JIT 激活内核的 16 元素向量对齐要求。
因此在该形状下使用数学等价的原生 PyTorch `SiLU × Mul` 实现，其余满足对齐的
形状仍使用 JIT 内核。该回退只改变本地激活计算实现，不改变 TP collective 的
调用位置、group size、逻辑 payload 或直方图统计口径；TP=8 单点验证及完整
195 条数据均已通过 rank 一致性检查。
