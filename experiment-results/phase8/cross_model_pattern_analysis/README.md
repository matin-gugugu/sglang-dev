# Phase 8 跨模型 PatternDemand 分析

## 数据完整性

| 模型 | workloads | TP | phases | repeats/workload |
|---|---:|---|---|---:|
| deepseek-v2-lite | 195 | 2,4,8 | decode,prefill | 3 |
| qwen3-8b | 195 | 2,4,8 | decode,prefill | 3 |

## 同 workload 跨模型结构指纹

| phase | TP | B,L,M | left model | left calls/payload | right model | right calls/payload |
|---|---:|---|---|---|---|---|
| decode | 2 | 1,128,32 | deepseek-v2-lite | 1705 / 6.66 MiB | qwen3-8b | 2263 / 17.7 MiB |
| decode | 4 | 1,128,32 | deepseek-v2-lite | 1705 / 6.66 MiB | qwen3-8b | 2263 / 17.7 MiB |
| decode | 8 | 1,128,32 | deepseek-v2-lite | 1705 / 6.66 MiB | qwen3-8b | 2263 / 17.7 MiB |
| prefill | 2 | 1,128,8 | deepseek-v2-lite | 55 / 27.5 MiB | qwen3-8b | 73 / 73 MiB |
| prefill | 4 | 1,128,8 | deepseek-v2-lite | 55 / 27.5 MiB | qwen3-8b | 73 / 73 MiB |
| prefill | 8 | 1,128,8 | deepseek-v2-lite | 55 / 27.5 MiB | qwen3-8b | 73 / 73 MiB |

## 近等总 payload、不同消息形态

| model | phase | TP | workload A | workload B | payload gap | calls ratio | shape distance |
|---|---|---:|---|---|---:|---:|---:|
| deepseek-v2-lite | decode | 2 | decode-tp2-b1-l128-m512 | decode-tp2-b16-l128-m32 | 2.94% | 16.48× | 1.000 |
| deepseek-v2-lite | decode | 2 | decode-tp2-b1-l2048-m512 | decode-tp2-b16-l2048-m32 | 2.94% | 16.48× | 1.000 |
| deepseek-v2-lite | decode | 4 | decode-tp4-b1-l128-m512 | decode-tp4-b16-l128-m32 | 2.94% | 16.48× | 1.000 |
| deepseek-v2-lite | decode | 4 | decode-tp4-b1-l2048-m512 | decode-tp4-b16-l2048-m32 | 2.94% | 16.48× | 1.000 |
| deepseek-v2-lite | decode | 8 | decode-tp8-b1-l128-m512 | decode-tp8-b16-l128-m32 | 2.94% | 16.48× | 1.000 |
| deepseek-v2-lite | decode | 8 | decode-tp8-b1-l2048-m512 | decode-tp8-b16-l2048-m32 | 2.94% | 16.48× | 1.000 |
| qwen3-8b | decode | 2 | decode-tp2-b1-l128-m512 | decode-tp2-b16-l128-m32 | 2.94% | 16.48× | 1.000 |
| qwen3-8b | decode | 2 | decode-tp2-b1-l2048-m512 | decode-tp2-b16-l2048-m32 | 2.94% | 16.48× | 1.000 |
| qwen3-8b | decode | 4 | decode-tp4-b1-l128-m512 | decode-tp4-b16-l128-m32 | 2.94% | 16.48× | 1.000 |
| qwen3-8b | decode | 4 | decode-tp4-b1-l2048-m512 | decode-tp4-b16-l2048-m32 | 2.94% | 16.48× | 1.000 |
| qwen3-8b | decode | 8 | decode-tp8-b1-l128-m512 | decode-tp8-b16-l128-m32 | 2.94% | 16.48× | 1.000 |
| qwen3-8b | decode | 8 | decode-tp8-b1-l2048-m512 | decode-tp8-b16-l2048-m32 | 2.94% | 16.48× | 1.000 |

## 结论

- 共获得 390 个 model × workload 聚合点；匹配的跨模型 workload 为 195 个。
- 同 workload 的跨模型 calls 比值中位数为 1.327，总逻辑 payload 比值中位数为 2.655。
- 自动找到 126 组总 payload 相差不超过 3.5% 但消息直方图不同的样本对。
- 第一阶段预测器应输入模型或模型结构特征，并输出连续 `op × group_size × payload` 直方图；total bytes 只能作为消融基线。
