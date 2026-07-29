# Phase 4：Qwen3-8B 扩展数据集与四模型留出评测

## 1. 实验目标

本阶段把前述“PatternDemand 机理验证”扩展为可训练、可留出评测的预测数据集，并回答三个问题：

1. 只使用总通信字节能否预测通信时间；
2. 三个硬消息桶与连续消息直方图相比，是否存在量化误差；
3. 在结构化通信公式基础上，DNN 残差校正是否还能降低预测误差。

实验固定模型为 Qwen3-8B，拓扑为单节点 B200 NVLink（L1），并行规模为 TP=2/4/8。

## 2. 扩展数据集

### 2.1 Workload 网格

Prefill：

- `batch_size ∈ {1,2,4,8,16}`
- `input_len ∈ {128,512,2048,8192}`
- `output_len = 8`
- TP=2/4/8

Decode：

- `batch_size ∈ {1,2,4,8,16}`
- `input_len ∈ {128,2048,8192}`
- `output_len ∈ {32,128,512}`
- TP=2/4/8

共得到：

- 60 个 Prefill workload；
- 135 个 Decode workload；
- 195 个唯一 workload；
- 每个 workload 独立重复 3 次；
- 总计 585 个阶段级测量。

Prefill 单次逻辑消息覆盖 1 MiB–1 GiB，Decode 单次逻辑消息覆盖 8–128 KiB，因而同时覆盖小消息启动区、大消息带宽区及通信算法/后端切换区。

### 2.2 测量口径

- `calls`：group-level collective 次数；
- `payload_bytes`：代表 rank 的逻辑输入大小；
- PatternDemand 使用 rank 0 的 lossless histogram；
- Prefill 对完整阶段进行 GPU profiling；
- Decode 对固定 8-step 窗口 profiling，再以
  `窗口平均 collective kernel cost × 完整阶段 calls`
  估计完整 Decode 阶段的代表 rank kernel envelope；
- 标签取 3 次独立重复的中位数。

所有 workload 都通过以下校验：

- 结果数量符合预期；
- workload 不重复；
- 三次重复的 PatternDemand 完全一致；
- profile 窗口中 group-level calls 与识别到的 GPU collective kernel 数量一致。

原始 profiler trace 保存在远端实验目录中，不提交 Git；compact result、ground truth、日志、划分、模型和图表均提交。

## 3. 数据划分

先按完整 workload 聚合三次重复，再进行划分：

- 训练集：135；
- 验证集：30；
- 测试集：30。

同一 workload 的 repeat 不会跨集合，避免数据泄漏。

测试集固定保留以下近等总 payload 对照：

- `B=1, M=512`
- `B=4, M=128`
- `B=16, M=32`

三者总逻辑 payload 约为 291.4、289.7 和 282.9 MiB，但 calls 相差约 16 倍，用于直接检验消息形态表征的必要性。

## 4. 四种预测模型

1. **Total bytes only**
   - 输入仅包含阶段、TP 和总逻辑通信字节；
   - 使用 log-target ridge 回归。
2. **Three hard bins**
   - 输入包含阶段、TP，以及 small/medium/large 三桶中的 calls 和 bytes；
   - 使用 log-target ridge 回归。
3. **Continuous histogram**
   - 对每个精确 payload 使用实测的 `payload × TP × backend → latency` 连续代价曲线；
   - 将 `calls(payload) × cost(payload)` 求和；
   - 只在训练集上拟合 phase×TP 的乘法校准系数。
4. **Continuous histogram + DNN residual**
   - 以连续直方图结构公式为基线；
   - MLP 只学习
     `log(measured / structured)`
     残差；
   - 输入包含 workload、TP、histogram 统计量、rounds 和结构化预测值；
   - 使用验证集早停，不直接黑盒回归总时间。

## 5. Workload 留出结果

### 5.1 全部测试 workload

| 模型 | MAPE | Median APE | P95 APE | R² |
|---|---:|---:|---:|---:|
| Total bytes only | 63.72% | 39.72% | 206.70% | 0.3918 |
| Three hard bins | 11.24% | 4.07% | 43.89% | 0.9692 |
| Continuous histogram | 8.65% | 1.85% | 47.61% | 0.9599 |
| Continuous + DNN residual | **7.19%** | 2.25% | **29.15%** | 0.9677 |

### 5.2 重复稳定的测试 workload

将三次重复的 IQR/median 不超过 20% 定义为稳定标签。主指标仍保留全部样本；该口径仅用于区分可预测结构误差与 profiler 状态长尾。

| 模型 | MAPE | P95 APE |
|---|---:|---:|
| Total bytes only | 65.30% | 211.78% |
| Three hard bins | 8.62% | 25.22% |
| Continuous histogram | 5.36% | 35.70% |
| Continuous + DNN residual | **3.75%** | **12.95%** |

30 个测试 workload 中有 27 个属于稳定标签，3 个存在明显重复间状态长尾。

### 5.3 分阶段结果

| 阶段 | Total bytes | Three bins | Continuous | Continuous + DNN |
|---|---:|---:|---:|---:|
| Prefill MAPE | 17.72% | 18.24% | 13.90% | **8.17%** |
| Decode MAPE | 83.44% | 8.24% | **6.40%** | 6.76% |

DNN 残差主要改善 Prefill 中算法切换和大消息区的系统偏差；在 Decode 小消息区，连续结构公式已经较准确，DNN 没有进一步改善平均误差。该结果不支持“DNN 在所有阶段都必然更好”的过度结论。

### 5.4 近等总 payload 留出对照

| 模型 | MAPE | R² |
|---|---:|---:|
| Total bytes only | 94.26% | -0.2231 |
| Three hard bins | 7.04% | 0.9862 |
| Continuous histogram | **4.18%** | **0.9913** |
| Continuous + DNN residual | 4.41% | 0.9908 |

以 TP=2 为例，三种近等总 payload workload 的实测通信时间分别约为：

- `B=1, M=512`：165.6 ms，37,303 calls；
- `B=4, M=128`：41.4 ms，9,271 calls；
- `B=16, M=32`：15.4 ms，2,263 calls。

总字节模型给出的预测几乎相同，而 histogram 模型能够恢复由 calls/消息尺度造成的约 10.7 倍时间差。这是本阶段对 PatternDemand 核心论点最直接的验证。

## 6. 结论

1. 总通信字节不足以作为调度预测的核心中间表征；
2. 保留 calls 与消息尺度后，MAPE 从 63.72% 降至 11.24%；
3. 连续代价曲线进一步把 MAPE 降至 8.65%，说明硬桶量化在跨算法、跨消息尺度时会损失信息；
4. DNN 只校正结构残差时，全部测试 MAPE 降至 7.19%，稳定测试集降至 3.75%；
5. 当前结果支持“PatternDemand 结构公式为主体、DNN 为残差校正”的建模路线，而不支持纯黑盒绕过 PatternDemand。

## 7. 限制

- 当前仅覆盖 Qwen3-8B 和单节点 L1 拓扑；
- 评测是 workload 留出，不是完全未见 TP 的跨并行规模外推；
- Decode 标签由固定窗口外推完整阶段，适用于本阶段的 uniform Decode 网格；
- 少数 workload 存在 profiler/rank 状态长尾，最终标签使用三重复中位数；正式论文可对高 IQR 点追加到 5–10 次重复；
- 后续还需加入真实 mixed/longtail trace、其他模型，以及 L2/L3 拓扑。

## 8. 文件

- `qwen3_8b_expanded/`：分 TP、repeat、phase 的 compact 原始结果、ground truth 和日志；
- `qwen3_8b_prediction_eval/aggregated_workloads.csv`：195 个聚合 workload；
- `qwen3_8b_prediction_eval/split_assignments.csv`：固定划分；
- `qwen3_8b_prediction_eval/predictions.csv`：四模型逐 workload 预测；
- `qwen3_8b_prediction_eval/metrics.csv`：全部评测指标；
- `qwen3_8b_prediction_eval/summary.json`：数据、模型参数、训练过程和指标；
- `qwen3_8b_prediction_eval/dnn_residual_model.pt`：DNN 残差模型；
- `qwen3_8b_prediction_eval/qwen3_8b_expanded_prediction_eval.png`：总体评测图；
- `qwen3_8b_prediction_eval/qwen3_8b_equal_payload_holdout.png`：近等总 payload 核心对照图。
