# Phase 9：双模型第一阶段 PatternDemand 预测与留出评测

## 1. 实验目的

本阶段使用 Qwen3-8B 与 DeepSeek-V2-Lite 的正式 `histogram-only` 数据，评估
从

```text
(B, L, M, phase, TP, model/config)
```

到第一阶段通信需求

```text
(op, group_size, calls, payload_bytes, total_payload_bytes, equivalent_rounds)
```

的预测能力。这里预测的是拓扑无关 `PatternDemand`，不是通信时间。DeepSeek
本轮没有 nsys 通信时延标签，因此不能用于重训 Phase 6 的第二阶段通信时间
四模型。

输入数据为 Phase 8 的 390 个 `model × workload` 聚合点。每个点均由三个正式
重复验证，重复间 PatternDemand 完全一致。

## 2. 当前数据实际呈现的结构

本轮正式规则网格中的 390 行全部满足：

- 仅包含 `AllReduce`；
- 每个 workload 的直方图只有一个 payload 支撑点；
- TP 改变 `group_size` 和等效 rounds，但不改变代表 rank 的逻辑 payload
  与 group-level calls；
- Qwen3-8B 每次 forward 有 73 次 collective，DeepSeek-V2-Lite 有 55 次；
- 每 token payload 分别为 8192 B 和 4096 B。

因此，对当前两个模型、当前 TP 实现和当前非混合请求网格，PatternDemand 可
精确写为：

```text
k(model) = 2 × num_hidden_layers + 1

Prefill:
  calls   = k(model)
  payload = B × L × hidden_size × dtype_bytes

Decode:
  calls   = k(model) × (M - 1)
  payload = B × hidden_size × dtype_bytes

AllReduce ring-equivalent rounds:
  rounds  = calls × 2 × (TP - 1)
```

这是一项实验发现，不应外推为所有模型、并行形态和调度策略的通用定律。

## 3. 四种预测器

| 预测器 | 输入与作用 |
|---|---|
| Categorical ridge | workload、phase、TP 和模型类别；用于检验类别记忆能否迁移 |
| Structure-aware ridge | 加入 hidden size、层数、MoE 标记和结构驱动量 |
| Analytic PatternDemand | 根据模型执行图中的 TP collective 模板直接计算 |
| Analytic + residual MLP | 只学习 `log(actual / analytic)`，不绕过结构公式 |

解析模型所使用的 `hidden_size`、`num_hidden_layers`、每层 collective 数及固定
collective 数属于模型配置/执行图元数据。因而“未见模型”结果表示：
在已知新模型 TP operator template 的条件下迁移，而不是仅凭模型名称对任意
未知架构进行零样本推断。

## 4. 无泄漏留出协议

| 协议 | fold | train | validation | test |
|---|---|---:|---:|---:|
| 未见 workload | grouped | 234 | 78 | 78 |
| 未见 TP | TP=2 | 214 | 46 | 130 |
|  | TP=4 | 214 | 46 | 130 |
|  | TP=8 | 214 | 46 | 130 |
| 未见模型 | DeepSeek-V2-Lite | 159 | 36 | 195 |
|  | Qwen3-8B | 159 | 36 | 195 |

- workload 留出以完整 `(phase,B,L,M)` 为单位，同时从所有模型和 TP 中移除，
  不允许同一个 workload 从其他模型或 TP 泄漏到训练集；
- TP 留出采用 leave-one-TP-out，汇总时每行恰好作为一次测试样本；
- 模型留出采用 leave-one-model-out，训练集中完全没有被测模型；
- 三个重复先聚合再划分，重复数据不会跨 split。

## 5. 主要结果

### 总逻辑 payload MAPE

| 预测器 | 未见 workload | 未见 TP | 未见模型 |
|---|---:|---:|---:|
| Categorical ridge | 0.0582% | 0.0733% | 113.8832% |
| Structure-aware ridge | 0.0001% | 0.0001% | 52.3260% |
| Analytic PatternDemand | 0% | 0% | 0% |
| Analytic + residual MLP | 0% | 0% | 0% |

未见模型的两个方向并不对称：

| 被留出模型 | Categorical ridge | Structure-aware ridge |
|---|---:|---:|
| DeepSeek-V2-Lite | 165.4376% | 65.1437% |
| Qwen3-8B | 62.3288% | 39.5082% |

解析模型在全部测试行上的 calls 与 payload 最大误差均为 0。残差 MLP 的最优
目标也是零残差，因此没有产生精度增益。

## 6. 结论

1. 当前规则 workload 对未见 workload 和 TP 的预测几乎是确定性的，证明
   第一阶段口径正确、模型结构差异可解释，但不能证明需要复杂神经网络。
2. 模型名称类别只能记忆已有模型；leave-one-model-out 时明显失效。
3. 只有两个模型时，每个 model-holdout fold 的训练集实际上只有一个模型结构
   点，普通结构回归无法可靠学习跨模型缩放关系。
4. 已知执行图 operator template 后，结构公式可精确重建当前直方图。该结果
   支持“结构公式为主、神经网络只校正残差”的设计。
5. 当前 390 行全部是单支撑点直方图，说明实验仍然偏薄。此时把 DNN 写成主要
   创新会受到“用网络拟合确定性线性规律”的合理质疑。

## 7. 下一轮必须增加的厚度

优先级从高到低：

1. 混合输出长度 continuous batching：让 `active_batch(t)` 在 Decode 内下降，
   同一 workload 产生多个 payload 支撑点；
2. chunked prefill：跨 chunk 边界时验证 calls 与 payload 的离散变化；
3. 动态调度条件：改变 chunk size、max running requests 和请求到达间隔；
4. 扩展模型与并行形态：增加第三个模型，并加入 PP、PD 分离或会引入
   AllGather/ReduceScatter 的执行配置；
5. 在上述数据出现非零结构残差后，再比较解析公式、纯 DNN 和
   `解析公式 + residual DNN`。如果 residual DNN 仍无收益，就不应强行使用。

当前推荐供调度器使用的第一阶段实现是解析 PatternDemand，并保留 residual
接口；待多支撑点、动态调度和更多并行原语数据补齐后再决定是否启用神经网络
校正。

## 8. 结果文件

```text
experiment-results/phase9/cross_model_pattern_prediction/
├── cross_model_pattern_prediction.png
├── metrics.csv
├── predictions.csv
├── residual_mlp_models.pt
├── split_assignments.csv
├── summary.json
└── train.log
```

执行脚本：

```bash
python scripts/evaluate_cross_model_pattern_predictor.py
```
