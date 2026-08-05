# Phase 14B：无实际 backend 的 TP×phase 条件模型

更新时间：2026-08-05

状态：分析、嵌套交叉验证、信息泄漏审计和确定性复跑均已完成

## 1. 目标与特征边界

本分析只复用 Phase 14 的 90 个聚合配置，不重新占用 GPU。目标是检验：在不读取
实际 runtime backend 的前提下，PatternDemand 加入 TP、Prefill/Decode 和
TP×phase 交互后，能否形成可泛化的通信时间模型。

预测特征包括 logical calls/bytes、连续 log2 payload 软直方图、logical raw-op
直方图、TP 和 phase。fused_allreduce_residual_rmsnorm 是运行前可知的逻辑
raw op，不是实际 backend。明确排除：

- backend_signature 和 kernel name；
- 模型身份；
- Phase 2 production/backend cost curve；
- 任何运行后才能观测的 backend 信息。

backend signature 只在预测完成后用于残差分组诊断，不参与训练或预测。

## 2. 防泄漏验证

采用 6 折完整 workload 留出。每折包含 5 个 workload group、15 个 TP 展开配置；
同一个 workload 的 TP2/4/8 必须一起进入同一测试折。每个外层训练折内部再进行
ridge alpha 选择，测试折不参与超参数选择。

此外执行 leave-one-model-out：完整留出 Qwen3-8B 或 Qwen3-30B-A3B。输入数据
已经先对每个配置的 3 次重复取中位数，不存在 repeat 级切分。

正式复验确认 720 条预测、30 个 fold assignment、32 条超参数选择记录和所有指标
均完整，预测特征名不包含 backend/kernel/model；确定性复跑与正式输出逐文件一致。

## 3. workload 留出消融

| 方法 | MAPE | P95 APE | R² |
|---|---:|---:|---:|
| PatternDemand | 18.495% | 41.554% | 0.9408 |
| PatternDemand + TP | 13.795% | 46.939% | 0.9100 |
| PatternDemand + phase | 19.048% | 42.820% | 0.9318 |
| PatternDemand + TP + phase + TP×phase | 11.819% | 38.261% | 0.9597 |

TP×phase 条件模型把平均 MAPE 相对 PatternDemand-only 降低约 36.1%，并改善
P95 和 R²。单独加入 phase 没有改善，说明当前数据中 TP 条件和 TP×phase 交互比
phase 截距本身更关键。

TP×phase 条件模型分 phase：

| scope | 样本 | MAPE | P95 APE | R² |
|---|---:|---:|---:|---:|
| Prefill | 72 | 11.858% | 34.285% | 0.9624 |
| Decode | 18 | 11.664% | 43.128% | 0.6054 |

Prefill 通过了预设的 MAPE <15% 门槛；整体 MAPE <10%、整体 P95 <25% 和
Decode MAPE <10% 均未通过。Decode 样本只有 18 个聚合点，尾部和 R² 仍不稳定。

## 4. 整模型留出

| 留出模型 | MAPE | P95 APE | R² |
|---|---:|---:|---:|
| Qwen3-8B | 33.169% | 64.915% | 0.7827 |
| Qwen3-30B-A3B | 29.258% | 116.545% | 0.8522 |

因此当前结果只支持“在已见模型家族和代表 workload 内，TP×phase 条件有效”，
不支持稳定跨模型泛化。

## 5. backend 事后诊断

backend 不进入预测，但预测后按观测 backend 分组：

| workload backend 状态 | 样本 | MAPE | P95 APE |
|---|---:|---:|---:|
| 跨 TP 不切换 | 54 | 8.947% | 31.698% |
| 跨 TP 发生切换 | 36 | 16.127% | 44.080% |

误差在 backend transition workload 上明显更高。该结果说明残差与 runtime
lowering 有关，但不能反向证明应该把运行后 backend 直接作为输入。更合理的后续
方案是补充 Decode/模型多样性，或仅测试运行前可确定的结构特征和 deterministic
dispatch proxy。

## 6. 结论

不使用实际 backend 是可行的方向：TP×phase 将 workload 留出的平均误差显著降低，
Prefill 平均误差已进入约 12%。但当前尾部、Decode 和整模型留出均未达到验收门槛，
所以应将该模型标记为“有效基线，尚非生产默认模型”。

正式产物包括 cv_predictions.csv、metrics.csv、fold_assignments.csv、
alpha_selection.csv、posthoc_backend_diagnostics.csv、summary.json 和分析图。
