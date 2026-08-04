# Phase 13A：三模型多 TP PatternDemand 综合分析

## 1. 数据与口径

本分析聚合 585 个 model × workload 配置：三模型各 195 个独立配置，均以完整 workload 聚合重复，不把 repeat 随机拆分为独立样本。

| 模型 | 结构 | hidden | layers | calls/forward | bytes/active token | raw ops |
|---|---|---:|---:|---:|---:|---|
| qwen3-8b | dense | 4096 | 36 | 73 | 8192 | ["all_reduce"] |
| deepseek-v2-lite | moe | 2048 | 27 | 55 | 4096 | ["all_reduce"] |
| qwen3-30b-a3b | moe | 2048 | 48 | 97 | 4096 | ["all_reduce","fused_allreduce_residual_rmsnorm"] |

精确直方图键为 (raw_op, payload)；拓扑折算时，fused_allreduce_residual_rmsnorm 保留 raw op 身份，但 collective_family 按 AllReduce 计算 equivalent bytes/rounds。

## 2. Workload 与 TP 规律

- qwen3-8b prefill：log2(payload) 对 log2(active tokens) 斜率 1.000000，R²=1.000000。
- qwen3-8b decode：log2(payload) 对 log2(active tokens) 斜率 1.000000，R²=1.000000。
- deepseek-v2-lite prefill：log2(payload) 对 log2(active tokens) 斜率 1.000000，R²=1.000000。
- deepseek-v2-lite decode：log2(payload) 对 log2(active tokens) 斜率 1.000000，R²=1.000000。
- qwen3-30b-a3b prefill：log2(payload) 对 log2(active tokens) 斜率 1.000000，R²=1.000000。
- qwen3-30b-a3b decode：log2(payload) 对 log2(active tokens) 斜率 1.000000，R²=1.000000。
- qwen3-8b：Decode 在固定 TP、B、M 时，45/45 组对 L 完全不变。
- deepseek-v2-lite：Decode 在固定 TP、B、M 时，45/45 组对 L 完全不变。
- qwen3-30b-a3b：Decode 在固定 TP、B、M 时，45/45 组对 L 完全不变。

- TP2 到 TP4/TP8 的所有配置中，logical calls 和 logical payload 保持不变；TP scaling 对照共 390 组。
- 等效 bytes 与 rounds 随 group size 增长，因此正式模型必须显式保留 group_size，不能把逻辑直方图不变误写为通信成本不变。

## 3. 近等总 payload 对照

自动找到 1701 组总逻辑 payload 差不超过 3.5%、但 raw op/payload 直方图不同的对照。
同 workload 的三模型两两匹配对照为 585 组。

最强的近等 payload 对照：

| phase | left | right | payload gap | calls ratio | histogram TV |
|---|---|---|---:|---:|---:|
| decode | qwen3-8b:decode-tp2-b1-l128-m512 | qwen3-8b:decode-tp2-b16-l128-m32 | 2.935% | 16.484× | 1.000 |
| decode | qwen3-8b:decode-tp2-b1-l128-m512 | qwen3-8b:decode-tp2-b16-l2048-m32 | 2.935% | 16.484× | 1.000 |
| decode | qwen3-8b:decode-tp2-b1-l128-m512 | qwen3-8b:decode-tp2-b16-l8192-m32 | 2.935% | 16.484× | 1.000 |
| decode | qwen3-8b:decode-tp2-b1-l128-m512 | qwen3-8b:decode-tp4-b16-l128-m32 | 2.935% | 16.484× | 1.000 |
| decode | qwen3-8b:decode-tp2-b1-l128-m512 | qwen3-8b:decode-tp4-b16-l2048-m32 | 2.935% | 16.484× | 1.000 |
| decode | qwen3-8b:decode-tp2-b1-l128-m512 | qwen3-8b:decode-tp4-b16-l8192-m32 | 2.935% | 16.484× | 1.000 |
| decode | qwen3-8b:decode-tp2-b1-l128-m512 | qwen3-8b:decode-tp8-b16-l128-m32 | 2.935% | 16.484× | 1.000 |
| decode | qwen3-8b:decode-tp2-b1-l128-m512 | qwen3-8b:decode-tp8-b16-l2048-m32 | 2.935% | 16.484× | 1.000 |
| decode | qwen3-8b:decode-tp2-b1-l128-m512 | qwen3-8b:decode-tp8-b16-l8192-m32 | 2.935% | 16.484× | 1.000 |
| decode | qwen3-8b:decode-tp2-b1-l2048-m512 | qwen3-8b:decode-tp2-b16-l128-m32 | 2.935% | 16.484× | 1.000 |

## 4. Leave-one-model-out

每个 held-out 模型使用另外两个模型的同 workload 数据。四种方法依次增加模型类别、结构缩放和解析 PatternDemand。解析方法使用模型/运行时元数据，不读取 held-out 模型的实验直方图。

| held model | method | calls MAPE | payload MAPE | eq bytes MAPE | eq rounds MAPE | hist TV | log-payload EMD |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3-8b | Workload only | 4.110% | 47.945% | 47.945% | 4.110% | 1.0000 | 1.0000 |
| qwen3-8b | Workload + dense/MoE | 4.110% | 47.945% | 47.945% | 4.110% | 1.0000 | 1.0000 |
| qwen3-8b | Model structure scaling | 0.000% | 0.000% | 0.000% | 0.000% | 0.0000 | 0.0000 |
| qwen3-8b | Analytical PatternDemand | 0.000% | 0.000% | 0.000% | 0.000% | 0.0000 | 0.0000 |
| deepseek-v2-lite | Workload only | 54.545% | 120.909% | 120.909% | 54.545% | 0.8937 | 0.4294 |
| deepseek-v2-lite | Workload + dense/MoE | 76.364% | 76.364% | 76.364% | 76.364% | 0.8136 | 0.0000 |
| deepseek-v2-lite | Model structure scaling | 0.000% | 0.000% | 0.000% | 0.000% | 0.0000 | 0.0000 |
| deepseek-v2-lite | Analytical PatternDemand | 0.000% | 0.000% | 0.000% | 0.000% | 0.0000 | 0.0000 |
| qwen3-30b-a3b | Workload only | 34.021% | 3.608% | 3.608% | 34.021% | 0.9102 | 0.5703 |
| qwen3-30b-a3b | Workload + dense/MoE | 43.299% | 43.299% | 43.299% | 43.299% | 0.8136 | 0.0000 |
| qwen3-30b-a3b | Model structure scaling | 0.000% | 0.000% | 0.000% | 0.000% | 0.8136 | 0.0000 |
| qwen3-30b-a3b | Analytical PatternDemand | 0.000% | 0.000% | 0.000% | 0.000% | 0.0000 | 0.0000 |

## 5. 解析公式审计

解析 PatternDemand 使用：payload = active_tokens × hidden_size × dtype_bytes；calls/forward = 2 × layers + 1；Decode 乘 (M - 1)。Qwen3-30B-A3B 在 payload 不超过 8 MiB 时保留 2 次 all_reduce，其余调用表现为 fused op；更大 payload 回到 all_reduce。

该公式对 585 个聚合配置的逐 raw-op 直方图审计失败数为 0。

这说明当前规则网格中的 PatternDemand 可由模型结构、workload 和运行时 lowering 规则精确重建；它不是完整端到端时延模型，也不能外推为 expert-parallel All-to-All 结论。

## 6. 正式产物

- pattern_summary.csv：585 个聚合配置及 raw-op 直方图；
- model_structure_summary.csv：从正式数据重算的结构指纹；
- workload_effects.csv 与 decode_input_length_invariance.csv；
- tp_scaling.csv；
- same_workload_model_pairs.csv 与 near_equal_payload_pairs.csv；
- model_holdout_predictions.csv 与 model_holdout_metrics.csv；
- summary.json、phase13a_three_model_analysis.png 和 analyze.log。

## 7. 结论边界

可以声称：三模型共同网格下，calls、payload、raw op 和 group size 共同构成可复核的 TP PatternDemand；解析结构模型在当前规则网格上能够跨模型重建直方图。

不能声称：已经测量 Qwen3-30B 的真实 collective 时间、端到端推理时间、EP routing All-to-All、L2/L3，或已经证明解析公式能覆盖 mixed Decode、chunked Prefill 和未见 runtime lowering。
