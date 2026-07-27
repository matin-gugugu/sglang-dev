# PatternDemand v1 数据契约

## 1. 适用范围

`pattern-demand-v1` 是第一阶段“拓扑无关通信需求画像”的冻结数据格式。每一行 JSONL 对应一个 workload、模型、并行配置和一次重复运行，供后续 Pattern 预测、拓扑代价映射和消融实验使用。

机器可校验定义位于：

```text
experiment-results/phase0/schemas/pattern_demand_v1.schema.json
```

校验命令：

```bash
python scripts/validate_pattern_demand_schema.py \
  --input 'experiment-results/phase0/**/pattern_dataset*.jsonl'
```

## 2. 冻结的统计口径

### 2.1 count

`count` 表示 **group-level collective 调用次数**。

- 同一个通信组内，各 rank 对同一次 collective 的埋点只统计一次；
- 聚合时选择一个代表 rank 的直方图；
- 不对所有 rank 的 calls 再求和；
- 不等价于 NCCL 内部 ring step、链路包数量或网络轮次。

若需要等效轮次，必须在后续代价阶段根据 `op` 和 `group_size` 计算，不能修改本字段的含义。

### 2.2 input_payload_bytes

`input_payload_bytes` 表示 **代表 rank 调用该 collective 时的逻辑输入 payload**。

- 单位为 byte；
- 是单次调用、单个代表 rank 的逻辑张量大小；
- 不是所有 rank payload 的求和；
- 不是 NCCL 算法折算后的等效 bytes；
- 不是实际链路流量；
- 不包含协议头、分块、重传和网络软件栈开销。

实际链路负担在第二阶段根据 `op × group_size × topology` 映射。

### 2.3 一致性要求

同一个通信组的各 rank 必须产生一致的 group-level 消息直方图。数据行只有在以下条件全部通过时才可进入训练集：

```text
rank_histogram_consistent = true
stats_conservation_passed = true
output_length_consistent = true
```

`raw_events_truncated` 只描述 full-trace 的逐事件列表是否达到保存上限，不影响持续更新的 histogram；因此它是审计字段，不是 PatternDemand 训练数据的淘汰条件。`histogram-only` 主动不保存 raw events 也不属于 truncated。

## 3. 核心结构

```text
features
  model
  parallel_form
  parallel_size
  batch_size
  input_len
  output_len
  output_lens_per_request
  prefill_chunk_size

group_level_message_histogram[]
  phase
  op
  group_id
  group_size
  active_batch_size
  prefill_chunk_index
  prefill_chunk_tokens
  input_payload_bytes
  output_payload_bytes
  dtype
  tensor_shape
  count
  first_decode_step
  last_decode_step
```

第一阶段的规范标签是 `group_level_message_histogram`。`derived` 是便于分析的冗余聚合量，不能替代直方图。

## 4. 输出长度口径

- uniform batch：`output_len` 表示每个请求实际生成的 token 数；
- mixed batch：`output_lens_per_request` 的长度必须等于 `batch_size`，最大值必须等于 `output_len`；
- `generated_output_tokens` 必须与上述配置一致；
- 不能把可能因 EOS 提前结束的 `max_tokens` 当成实际生成长度。

## 5. phase 上下文

- Prefill 事件的 `first_decode_step` 和 `last_decode_step` 必须为 `null`；
- Decode 事件的 `prefill_chunk_index` 和 `prefill_chunk_tokens` 必须为 `null`；
- chunked-prefill 使用 `prefill_chunk_index` 和 `prefill_chunk_tokens` 区分完整 chunk 与尾块；
- continuous/draining batch 使用 `active_batch_size` 区分 Decode 阶段内的消息尺度迁移。

## 6. 版本规则

以下变化需要升级 schema 版本：

- 修改 calls 或 payload 的统计口径；
- 改为所有 rank 求和；
- 将等效链路 bytes 写回 `input_payload_bytes`；
- 删除或重定义核心直方图字段；
- 改变一行数据代表的 workload/run 粒度。

新增可选元数据、增加新的 `op` 类型，或增加新的 `parallel_form` 时，应先扩展 schema 并重新验证旧数据兼容性。
