# Phase 14E：Decode active-batch 时序特征验证

更新时间：2026-08-06

状态：可行性审计、smoke、正式分析、确定性复验和归档均已完成；时序特征假设未通过

## 1. 目标

Phase 14D 中，TP 条件 PatternDemand 斜率是描述性最优候选，但整体 MAPE 仍为
12.150%，Decode MAPE 为 17.583%。Phase 14E 检验当前直方图是否因丢失 Decode
过程中的 active-batch 下降形态而产生残差。

本阶段不新增 GPU 采集，复用 Phase 14C 的 162 个聚合配置，并且不使用实际
backend、kernel name、模型身份、Phase 2 cost curve 或目标时间派生特征。

## 2. 时序能否从现有数据重建

可以。每条配置都保存了 `output_lens_json`。对 Decode workload，在第 `s` 个 step
上统计 `output_len > s` 的请求数，即可重建 active-batch 序列。

| 审计项 | 结果 |
|---|---:|
| Decode TP-expanded 配置 | 54 |
| Decode workload | 18 |
| Decode profile / 唯一时序 | 6 / 6 |
| TP2/4/8 间 output lengths | 完全一致 |
| 同一 profile 跨模型时序 | 完全一致 |
| 每折完整留出一种 Decode profile | 通过 |

六折依次完整留出 balanced、bimodal、long_tail、staircase、uniform_b16、
uniform_b4，并且该 profile 在三个模型上的全部 TP 变体同时留出。因此验证的是
“能否泛化到未见过的 Decode 形态”，不是随机拆分已见形态。

## 3. 新增特征

从 active-batch 序列提取 14 个紧凑特征，包括：初始 batch、Decode step 数、总 token、
平均/标准差/最终 active-batch 比例、active level 数、变化频率、最长平台期、输出长度
四分位数、变异系数和前后段 active-batch 比。

这些特征在当前 ignore-EOS 合成实验中由预先配置的每请求输出长度得到。真实线上请求
通常不知道最终输出长度，因此生产使用时只能使用请求上限或单独的长度预测，不能读取
执行后观察到的完成长度。

TP 条件 PatternDemand 的特征矩阵秩为 80；14 个时序特征自身秩仅为 6，合并后秩为
83。也就是说，新特征只增加 3 个独立方向，大部分信息已被现有 PatternDemand 表达。

## 4. 正式结果

| 方法 | 特征数 | MAPE | P95 APE | R² |
|---|---:|---:|---:|---:|
| additive TP×phase | 34 | 12.805% | 39.366% | 0.8961 |
| TP-conditioned PatternDemand | 92 | **12.150%** | **39.013%** | 0.8953 |
| additive + schedule | 48 | 27.187% | 126.302% | 0.1945 |
| TP-conditioned + schedule | 106 | 25.255% | 122.103% | 0.2419 |
| TP-conditioned + schedule×TP | 134 | 26.526% | 132.523% | 0.1802 |

预先设定的 `TP-conditioned + schedule` 相比 Phase 14D 最优基线：

- MAPE 从 12.150% 恶化到 25.255%，相对增加 107.9%；
- P95 从 39.013% 恶化到 122.103%，相对增加 213.0%；
- Prefill MAPE 为 10.633%；
- Decode MAPE 为 54.498%，R² 为 -1.7059。

描述性最优方法仍然是完全不加 schedule 的 Phase 14D TP-conditioned PatternDemand。

## 5. 为什么恶化

当前只有 6 种唯一 Decode 时序。每个外层折用 5 种形态训练，再预测第 6 种完全未见
形态。14 个相关特征在这么少的形态上无法学到稳定关系，反而驱动线性模型外推。

最明显的是 staircase：完整留出后，预设 schedule 模型 MAPE 为 163.838%，而
Phase 14D TP-conditioned 基线为 16.907%。其他 Decode profile 的 schedule MAPE：

| profile | MAPE |
|---|---:|
| bimodal | 10.839% |
| balanced | 25.411% |
| uniform_b4 | 37.512% |
| uniform_b16 | 43.700% |
| long_tail | 45.689% |
| staircase | 163.838% |

这不是后台或采集失败，而是严格 profile 留出验证暴露了时序特征的外推不稳定性。

## 6. 整模型留出与收敛门槛

预设 schedule 模型的整模型留出 MAPE：

| 留出模型 | MAPE | P95 APE |
|---|---:|---:|
| Qwen3-8B | 37.312% | 78.459% |
| Qwen3-30B-A3B | 33.044% | 51.526% |
| DeepSeek-V2-Lite | 68.481% | 152.372% |

该模型只通过 Prefill MAPE <15%；整体 MAPE <10%、P95 <25%、Decode MAPE
<10%、全部留出模型 MAPE <15%，以及优于 Phase 14D 基线等门槛均未通过。

## 7. 结论与下一步

Phase 14E 可以明确排除：在只有 6 种 Decode 形态的情况下，把 output-length
时序摘要直接加入 ridge 模型不能改善泛化。当前应继续保留 Phase 14D 的 TP-conditioned
PatternDemand 作为描述性基线，不能将 schedule 模型升级为默认模型。

不建议继续从相同数据制造更多时序特征。后续只有两个合理选择：

1. 接受当前 backend-free 模型约 12.15% MAPE 的边界，结束 TP 分支；
2. 若业务必须追求 <10%，新增更多 Decode 形态，而不是新增模型参数。应使用
   space-filling 设计覆盖不同 batch、step 数、drain 速度和平台期，并继续按完整
   未见 profile 留出。

Kimi-K2.5 仍应留作模型稳定后的整模型外部验证，不应加入当前六种时序的训练集来掩盖
形态不足。

## 8. 正式产物

```text
experiment-results/phase14e/
├── README.md
├── audit_summary.json
├── manifest.sha256
├── smoke/smoke_summary.json
├── decode_schedule_analysis/
│   ├── README.md
│   ├── schedule_features.csv
│   ├── fold_assignments.csv
│   ├── alpha_selection.csv
│   ├── predictions.csv
│   ├── metrics.csv
│   ├── posthoc_backend_diagnostics.csv
│   ├── summary.json
│   └── decode_schedule_analysis.png
├── smoke.log
├── runner.log
├── revalidate_smoke.log
└── revalidate_phase14e.log
```

首次后台正式分析耗时约 78.7 秒，只使用 CPU。smoke 和正式分析的独立重算均逐字节
一致；根 manifest 覆盖除自身以外的全部 Phase 14E 文件。

## 9. 外推限制

结果只覆盖单节点 B200、三个模型、TP2/4/8、六种 Decode profile 和当前合成
ignore-EOS 设置。它不能证明时序永远无用，只能证明当前 6 种形态不足以支持这组
时序特征泛化，也不能直接迁移到未知线上输出长度分布。
