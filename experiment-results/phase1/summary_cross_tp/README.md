# Qwen3-8B TP2/TP4/TP8 PatternDemand 实验摘要

## 本轮目的

1. 将 TP2 的 Prefill 8 MiB、Decode mixed/longtail 补到 10 次重复，稳定 median、p95 和 CV。
2. 在 TP4、TP8 上重复等总逻辑 payload 的 uniform/mixed/longtail 对照。
3. 显式加入 `group_size`、ring 等效 bytes 和 ring 等效 rounds，验证相同总 payload 下消息调用结构及并行规模仍会改变通信需求与通信代价。

## 实验矩阵与数据完整性

- 模型：Qwen3-8B
- 设备：单节点 8×NVIDIA B200
- 通信原语：AllReduce
- 模式：eager、固定实际输出长度、GPU-only Nsight trace、histogram-only 通信埋点
- TP2：
  - Prefill 8 MiB：`r0-r9`，共 10 次
  - Decode mixed/longtail：各 `r0-r9`，共 10 次
  - Decode uniform：已有 `r0-r2`，共 3 次；其 CV 仅 0.0012，本轮未追加
- TP4、TP8：
  - Decode uniform/mixed/longtail：每种均为 `r0-r9`，各 10 次
- 本轮新增 81 个采集 case；最终 Decode 汇总共 83 条重复级记录。
- TP4、TP8 每个 case 均同时保存 `result.jsonl`、`comm_ground_truth.jsonl`、`run.log` 和压缩 trace。
- 所有 case 均满足预期 group-level calls 与代表 rank 的 GPU AllReduce kernel 数完全一致。

## 等总 payload 对照

三种 Decode 形态在所有 TP 下的代表 rank 逻辑总 payload 均严格相同：

`71,761,920 bytes = 68.4375 MiB`

| TP | group_size | ring bytes 系数 | ring 等效 bytes | uniform rounds | mixed rounds | longtail rounds |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 1.00 | 68.44 MiB | 2,190 | 5,110 | 7,446 |
| 4 | 4 | 1.50 | 102.66 MiB | 6,570 | 15,330 | 22,338 |
| 8 | 8 | 1.75 | 119.77 MiB | 15,330 | 35,770 | 52,122 |

三种形态对应的 group-level calls 恒定为：

- uniform：1,095
- mixed：2,555
- longtail：3,723

计算口径：

- `ring_equivalent_bytes = logical_payload_bytes × 2(p-1)/p`
- `ring_equivalent_rounds = group_level_calls × 2(p-1)`

这些量是 ring-style 结构化需求，不是直接测得的实际链路流量或 NCCL 内部 kernel step 数。

## 通信时间结果

单位为 GPU AllReduce kernel 累计时间，表中为重复实验的 median、p95 和 CV。

| TP | 形态 | 重复数 | median (ms) | p95 (ms) | CV |
|---:|---|---:|---:|---:|---:|
| 2 | uniform | 3 | 4.961 | 4.969 | 0.001 |
| 2 | mixed | 10 | 17.007 | 94.942 | 1.363 |
| 2 | longtail | 10 | 23.790 | 42.373 | 0.373 |
| 4 | uniform | 10 | 16.573 | 74.655 | 1.113 |
| 4 | mixed | 10 | 35.186 | 105.466 | 0.831 |
| 4 | longtail | 10 | 25.150 | 99.944 | 0.815 |
| 8 | uniform | 10 | 16.707 | 35.177 | 0.543 |
| 8 | mixed | 10 | 59.111 | 175.252 | 0.758 |
| 8 | longtail | 10 | 50.645 | 102.774 | 0.481 |

Prefill 8 MiB：

- 每次逻辑 payload：8 MiB
- group-level calls：73
- 重复数：10
- median：1.717 ms
- p95：3.440 ms
- CV：0.504

## 对 PatternDemand 设计的意义

1. **总 payload 不足以唯一决定通信需求。** 在每个 TP 内，uniform、mixed、longtail 的总 payload 完全相同，但 calls 和 rounds 分别显著不同，测得的通信时间分布也不同。
2. **并行规模不能只作为普通类别字段。** TP 从 2 增至 4、8 时，ring 等效 bytes 系数从 1 增至 1.5、1.75，单次 collective 的等效 rounds 从 2 增至 6、14。
3. **第一阶段结构画像比时延更稳定。** calls、payload、group size 和 rounds 在重复间完全一致；GPU kernel 累计时延则存在明显长尾。这支持“第一阶段预测 PatternDemand，第二阶段映射链路代价，再由残差模型校正系统抖动”的设计。
4. **预测标签应采用分布统计。** 不能用单次运行或简单均值作为唯一标签；建议至少同时保存 median、p95、CV，并保留重复级样本。

## 使用限制

- 本轮只覆盖单节点 B200 拓扑，尚不能代表同机架跨节点或跨机架链路。
- ring 等效 bytes/rounds 是结构化近似，不应表述为真实 wire bytes。
- 多个 case 出现少量超长 AllReduce kernel，使 p95 和 CV 较高；这些异常值被完整保留。它们说明系统层存在随机尾部，但不改变 PatternDemand 结构统计。
- TP8 longtail-r5 曾发生一次 NCCL 初始化 barrier 挂起；未完成样本没有进入结果，清理后断点重试成功。

## 文件位置

远端：

- `experiment-results/phase1/qwen3_8b_tp2_inference_comm/`
- `experiment-results/phase1/qwen3_8b_tp4_inference_comm/`
- `experiment-results/phase1/qwen3_8b_tp8_inference_comm/`
- `experiment-results/phase1/summary_cross_tp/`
- `experiment-results/phase1/logs/qwen3_8b_tp_comm_suite_20260728.log`

本地：

- `outputs/phase1_cross_tp/qwen3_8b_cross_tp_results.png`
- `outputs/phase1_cross_tp/decode_cross_tp_summary.csv`
- `outputs/phase1_cross_tp/prefill_8mib_summary.csv`
- `outputs/phase1_cross_tp/cross_tp_summary.json`

Git：

- 主结果提交：`f4610b2`
- 最终图与汇总脚本修正后的分支头：`11267d0`
