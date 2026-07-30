# Phase 6：Qwen3-8B 通信标签定稿与完整预测数据集

## 1. 阶段目标

Phase 6 将 Phase 5 的标签复核扩展到完整数据集：

1. 用 10 次重复控制实验验证同形状 warmup；
2. 将 all-rank 时间拆成 intrinsic、post-rendezvous 和
   synchronization-inclusive 三种口径；
3. 在完整 195 点上比较三种标签的稳定性和物理含义；
4. 在 TP=2/4/8、Prefill/Decode 网格上采集三重复数据；
5. 分别使用 intrinsic 和 post-rendezvous 训练四类模型；
6. 用标签稳定性、预测误差和跨拓扑可实现性共同选择最终口径。

## 2. 三种时间口径

对按调用顺序严格对齐的第 \(e\) 个 group-level collective，设 rank \(r\)
上的 kernel 起止时间为 \(s_{e,r}\)、\(f_{e,r}\)，duration 为
\(d_{e,r}=f_{e,r}-s_{e,r}\)。

### 2.1 Post-rendezvous completion：当前同节点主标签

\[
T_{\mathrm{post}}=
\sum_e\left(\max_r f_{e,r}-\max_r s_{e,r}\right).
\]

它从最后一个 rank 进入 collective 的时刻开始计时，到所有 rank 完成为止，
排除 pre-entry wait，同时保留真正的 collective completion。完整数据中该
标签重复稳定性最好，因此作为当前 L1 同节点预测的主标签。

限制：该定义要求跨 rank trace timestamp 可比较。单节点 B200 的 profiler
时间域已通过稳定性和时序检查验证；未来 L2/L3 跨节点实验必须增加显式时钟
同步或直接的 rendezvous timing，不能默认不同节点 trace 时钟天然一致。

### 2.2 Skew-free intrinsic：可移植下包络

\[
T_{\mathrm{intrinsic}}=\sum_e\min_r d_{e,r}.
\]

它只使用各 rank 自身 duration，不依赖跨 rank 时钟同步，是可移植的通信本体
下包络。但完整 TP=8 Decode 中，最快退出角色会随重复变化，导致 5 个点的
IQR 超过 20%，因此不再作为当前同节点唯一真值。

### 2.3 Synchronization-inclusive：仅诊断

\[
T_{\mathrm{sync}}=\sum_e\max_r d_{e,r}.
\]

它会把提前进入 collective 的等待写入 duration，并跨 rank 重复累计相互
重叠的等待区间。该量只用于诊断 rank 到达偏斜，不用于训练。

## 3. 实验 A：同形状 warmup 控制组

控制点：

- TP=2：B1/L128、B1/L2048、B1/L8192、B8/L128；
- TP=4：B1/L128、B1/L2048、B1/L8192；
- TP=8：B1/L128、B1/L2048、B1/L8192；
- 每点 10 次重复，共 100 条正式记录。

协议：

- 启用 `--warmup-each-workload`；
- 每个正式点前执行相同 B、L 和执行开关的非 profile 热身；
- Decode 热身最多 32 token；
- 热身关闭 `comm_profile` 和 profiler，不进入正式 histogram；
- 启用新协议时关闭旧的“只热身第一个网格点”逻辑。

已验证：

- 10/10 workload 的 PatternDemand 在历史三重复和新十重复中完全一致；
- intrinsic 历史/十重复 IQR 中位数为 0.41%/1.10%；
- intrinsic 热身后/历史目标中位比为 1.000，范围 0.978–1.005；
- post-rendezvous 历史/十重复 IQR 中位数为 0.46%/0.73%；
- post-rendezvous 热身后/历史目标中位比为 1.001，范围 0.968–1.006；
- synchronization-inclusive IQR 中位数由 24.19% 降至 18.51%。

结论：warmup 能部分降低 rank 到达等待，但不会系统性改变通信本体。完整网格
继续保留同形状 warmup，以统一执行协议。

输出：

- `qwen3_8b_warmup_control/`；
- `qwen3_8b_warmup_control_summary/`。

## 4. 实验 B：完整 195 点 corrected dataset

网格：

- TP \(\in\{2,4,8\}\)；
- Prefill：B \(\in\{1,2,4,8,16\}\)，
  L \(\in\{128,512,2048,8192\}\)，M=8，共 60 点；
- Decode：B \(\in\{1,2,4,8,16\}\)，
  L \(\in\{128,2048,8192\}\)，
  M \(\in\{32,128,512\}\)，共 135 点；
- 每点三重复，总计 585 条正式记录。

完整性：

- 195/195 workload 的 PatternDemand 在三重复中不变；
- 每条记录 all-rank kernel count 等于 group-level calls；
- backend sequence 与 PatternDemand 跨 rank 完全一致；
- 数据按 TP 均分为 65/65/65 点；
- Prefill/Decode 为 60/135 点。

标签稳定性：

| 标签 | IQR/median 中位数 | P95 | >20% workload |
|---|---:|---:|---:|
| Rank 0 | 14.25% | 555.53% | 82 |
| Intrinsic | 4.26% | 18.62% | 5 |
| Post-rendezvous | 1.59% | 6.24% | 0 |
| Sync-inclusive | 5.62% | 33.73% | 37 |

Post-rendezvous / intrinsic 的中位比为 1.246，P95 为 1.686；它不是简单复制
intrinsic，而是补回所有 rank 到齐后的 completion。Sync-inclusive /
intrinsic 中位数为 49.81×，进一步证明同步等待不能混入网络本体。

近等总 payload 的 B1/M512 与 B16/M32 对照：

| TP | payload 比 | Intrinsic 时间比 | Post-rendezvous 时间比 |
|---:|---:|---:|---:|
| 2 | 1.030 | 10.64× | 11.63× |
| 4 | 1.030 | 11.22× | 12.06× |
| 8 | 1.030 | 12.60× | 12.59× |

因此，无论采用下包络还是 completion 标签，“总 bytes 相近但消息形态不同时
代价显著不同”的核心论点都成立。

输出：

- `qwen3_8b_corrected_all_rank/`：585 条 compact records、日志和遥测；
- `qwen3_8b_corrected_summary/corrected_dataset_summary.csv`；
- `qwen3_8b_corrected_summary/qwen3_8b_corrected_dataset_audit.png`；
- `qwen3_8b_corrected_summary/summary.json`。

## 5. 实验 C：四模型 grouped holdout

195 个 workload 先聚合三重复，再按完整 workload 分为
train/validation/test=135/30/30，repeat 不跨集合泄漏。

对比模型：

1. total bytes only；
2. three hard bins；
3. continuous message histogram + matching cost curve；
4. continuous histogram + DNN residual。

### 5.1 Post-rendezvous 主结果

结构曲线使用 CustomAllReduce 的 `completion_median_latency_us` 和 NCCL 的
`median_latency_us`。DNN 只拟合
`log(measured_post / structural_prediction)`。

| 模型 | Test MAPE | P95 APE | R² | 等总 payload MAPE |
|---|---:|---:|---:|---:|
| Total bytes only | 59.52% | 203.20% | 0.4780 | 96.73% |
| Three hard bins | 7.04% | 22.64% | 0.9966 | 3.66% |
| Continuous histogram | 10.93% | 41.57% | 0.9874 | 6.78% |
| Continuous + DNN residual | **3.71%** | **9.39%** | **0.9977** | **2.39%** |

分阶段 DNN MAPE：

- Prefill：5.51%；
- Decode：2.94%。

分 TP DNN MAPE：

- TP=2：5.54%；
- TP=4：3.19%；
- TP=8：2.39%。

### 5.2 Intrinsic 对照

| 模型 | Test MAPE | P95 APE | R² | 等总 payload MAPE |
|---|---:|---:|---:|---:|
| Total bytes only | 58.45% | 180.63% | 0.4581 | 95.13% |
| Three hard bins | 8.74% | 23.63% | 0.9966 | 5.20% |
| Continuous histogram | 6.87% | 36.73% | 0.9882 | 5.13% |
| Continuous + DNN residual | **3.63%** | **10.39%** | **0.9976** | **3.07%** |

Intrinsic DNN 的平均误差略低 0.08 个百分点，但 post-rendezvous 无
`IQR>20%` 点、P95 更低、等总 payload 误差更低，并直接表示最后一个 rank
到齐后的完成时间。因此当前同节点正式结果选择 post-rendezvous；intrinsic
作为跨节点时钟未同步时的可移植下包络和消融对照。

输出：

- `qwen3_8b_prediction_eval_post_rendezvous/`：主模型和主评测；
- `qwen3_8b_prediction_eval_intrinsic/`：下包络消融。

## 6. 对论文设计的含义

1. PatternDemand 继续保持 topology-independent：
   `op × payload histogram × calls × group size × rounds`；
2. 第二阶段曲线必须与目标时间语义一致：
   post-rendezvous 使用同步进入后的 completion curve；
3. DNN 只校正结构化公式残差，不绕过 PatternDemand；
4. rank 到达偏斜、计算/通信 overlap 应作为独立调度特征，不能伪装成 RTT；
5. 三硬桶在本数据上已很强，但连续曲线提供可解释的跨尺度结构基线；其
   Prefill 大消息误差由小型残差网络显著修正，体现“结构公式 + 残差”的价值。

## 7. 数据保存

原始 profiler traces 仅保留在远端：

`/sgl-workspace/sglang-src/experiment-results/phase6/**/traces/`

当前完整网格共有 2,730 个 trace 文件。compact JSONL、CSV、日志、遥测、
图片、摘要和模型文件提交 Git。论文引用时必须注明：

- PatternDemand 是 group-level；
- payload 是代表 rank 的逻辑输入；
- L1 主时间标签是 all-rank post-rendezvous completion；
- intrinsic 是 duration-only 下包络；
- sync-inclusive 仅是 rank 到达偏斜诊断。
