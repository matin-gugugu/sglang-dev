# Phase 14C：三模型 TP×phase 扩展验证

更新时间：2026-08-05

状态：smoke、正式采集、联合分析、确定性复验和归档均已完成

## 1. 目标

Phase 14B 只覆盖两个 Qwen 模型，且 Decode 仅有 balanced、staircase、bimodal
三种形态。Phase 14C 在不把实际 backend 当作模型输入的前提下补充：

1. DeepSeek-V2-Lite 的 TP4/8 时间标签；
2. 三模型、TP2/4/8 上的 `uniform_b4`、`uniform_b16`、`long_tail` Decode；
3. 三模型联合 workload 留出和整模型留出验证。

## 2. 数据规模与完整性

| 项目 | 结果 |
|---|---:|
| 硬件 | 单节点 8×B200 |
| 模型 | Qwen3-8B、Qwen3-30B-A3B、DeepSeek-V2-Lite |
| TP | 2、4、8 |
| phase | Prefill、Decode |
| smoke 单元 | 3/3 通过 |
| 正式新增单元 | 111/111 生成 `DONE` |
| 新增 compact labels | 171/171 |
| 联合原始重复标签 | 486 |
| 联合聚合配置 | 162（每个配置 3 个 repeat） |
| TP-linked workload | 54 |
| Prefill / Decode 配置 | 108 / 54 |
| 被哈希源文件 | 216 |

所有正式标签均通过 all-rank kernel count、backend sequence、profiled/full-phase
PatternDemand 对齐检查；profiled-to-full scale 均为 1.0；同一 workload 的 logical
PatternDemand 在 TP2/4/8 间保持不变；正式目录中 raw profiler trace 已清零。

三重复 post-rendezvous IQR/median 的中位数为 0.704%，P95 为 3.277%，最大值为
12.779%。

## 3. 模型与验证口径

主目标是每个完整配置三次重复的 all-rank post-rendezvous completion time 中位数。
候选模型只使用运行前可得信息：logical PatternDemand、TP、phase 和 TP×phase；不使用：

- 实际 backend 或 kernel name；
- 模型身份；
- Phase 2 实测 cost curve；
- 同一 workload 的其他 TP 标签。

外层验证为 6 折完整 workload 留出，同一 workload 的 TP2/4/8 始终位于同一折；
ridge alpha 只在每个外层训练集内部选择。另执行 leave-one-model-out。backend
signature 只用于预测后的残差诊断。

## 4. Phase 14C 主结果

首选基线 `PatternDemand + TP + phase + TP×phase`：

| 范围 | MAPE | P95 APE | R² |
|---|---:|---:|---:|
| 全部 workload CV | 12.805% | 39.366% | 0.8961 |
| Prefill | 10.245% | 35.530% | 0.9289 |
| Decode | 17.924% | 66.706% | 0.6527 |

分 TP MAPE 为 TP2 13.380%、TP4 11.590%、TP8 13.444%。分模型 workload-CV
MAPE 为 Qwen3-8B 10.518%、Qwen3-30B-A3B 16.112%、DeepSeek-V2-Lite
11.784%。

完整留出模型结果：

| 留出模型 | MAPE | P95 APE | R² |
|---|---:|---:|---:|
| Qwen3-8B | 36.618% | 70.908% | 0.4351 |
| Qwen3-30B-A3B | 15.150% | 36.384% | 0.8829 |
| DeepSeek-V2-Lite | 16.864% | 38.091% | 0.8229 |

因此仅 Prefill MAPE <15% 门槛通过；整体 MAPE <10%、整体 P95 <25%、Decode
MAPE <10% 和全部留出模型 MAPE <15% 均未通过。

## 5. 新 Decode 形态暴露的问题

| Decode profile | MAPE |
|---|---:|
| balanced | 4.76% |
| bimodal | 6.54% |
| long_tail | 12.55% |
| staircase | 17.71% |
| uniform_b16 | 28.50% |
| uniform_b4 | 37.48% |

最差组合为 DeepSeek-V2-Lite `uniform_b4`，MAPE 79.15%；Qwen3-30B-A3B
`uniform_b16` 为 59.46%。这说明“加一个 TP/phase 截距”无法稳定覆盖消息结构变化，
下一阶段应允许 PatternDemand 的斜率随 TP、Prefill/Decode 改变。

## 6. backend 的结论边界

backend 不作为输入。仅做事后分组时，跨 TP 不发生 backend transition 的 99 个配置
MAPE 为 14.825%，发生 transition 的 63 个配置为 9.631%。这与 Phase 14B 的方向
相反，说明 transition 标签不是跨模型稳定的误差解释变量，也不能据此把实际 backend
加入输入。

## 7. 结论与下一步

Phase 14C 成功完成了三模型和更广 Decode 形态的压力测试，但当前 additive
TP×phase 基线未收敛，尤其 Decode 和整模型留出误差明显偏高。下一步 Phase 14D
只使用 TP、Prefill/Decode 与 PatternDemand，比较 TP 条件斜率、phase 条件斜率、
完整 TP×phase 条件斜率和 ring-equivalent 结构特征；仍不读取实际 backend。

## 8. 正式产物

```text
experiment-results/phase14c/
├── README.md
├── audit_summary.json
├── manifest.sha256
├── smoke/
├── deepseek_tp_extension/
├── decode_extension/
├── extended_dataset_analysis/
├── tp_phase_no_backend_analysis/
├── smoke_runner.log
├── formal_runner.log
└── revalidate_phase14c.log
```

根 manifest 覆盖除自身以外的全部 Phase 14C 归档文件，并通过逐项 SHA-256 复验。

## 9. 外推限制

结果只覆盖单节点 B200、三个模型、TP2/4/8 和当前 Prefill/Decode workload。不能外推
到跨节点、PP、PD、expert-parallel All-to-All、其他 GPU，或未覆盖的调度和消息形态。
