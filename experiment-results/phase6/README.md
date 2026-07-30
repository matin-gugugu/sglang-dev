# Phase 6：Qwen3-8B intrinsic 标签定稿与完整预测数据集

## 1. 阶段目标

Phase 6 将 Phase 5 的标签复核结论落实到完整数据集：

1. 用 10 次重复控制实验验证同形状 warmup 与标签稳定性；
2. 将 all-rank 标签拆成 intrinsic、post-rendezvous 和
   synchronization-inclusive 三种口径；
3. 以 skew-free intrinsic 作为结构化通信代价的唯一训练标签；
4. 在 TP=2/4/8、Prefill/Decode 的完整 195 点网格上采集三重复数据；
5. 重新训练并留出评测四类预测模型。

## 2. 标签定义

对按调用顺序严格对齐的第 \(e\) 个 group-level collective，设 rank \(r\)
上的 kernel 起止时间为 \(s_{e,r}\)、\(f_{e,r}\)，duration 为
\(d_{e,r}=f_{e,r}-s_{e,r}\)。

训练标签：

\[
T_{\mathrm{intrinsic}}=\sum_e\min_r d_{e,r}.
\]

该口径去除 rank 提前到达 collective 所支付的等待，能够与隔离通信曲线中
`skew-free-minimum-duration-across-ranks` 的统计口径一致。它只依赖各 rank
自身的 duration，不要求跨节点 trace 时钟严格同步。

同一记录另外保留：

- `post_rendezvous_completion`：
  \(\sum_e(\max_r f_{e,r}-\max_r s_{e,r})\)，同节点时钟诊断；
- `synchronization_inclusive_max_duration_sum`：
  \(\sum_e\max_r d_{e,r}\)，rank 到达偏斜诊断。

后两者不作为链路本体训练标签。

## 3. 实验 A：同形状 warmup 控制组

控制点：

- TP=2：B1/L128、B1/L2048、B1/L8192、B8/L128；
- TP=4：B1/L128、B1/L2048、B1/L8192；
- TP=8：B1/L128、B1/L2048、B1/L8192；
- 每点 10 次重复，共 100 条正式记录。

协议：

- 启用 `--warmup-each-workload`；
- 每个正式点之前执行相同 B、L 和执行开关的非 profile 热身；
- Decode 热身最多 32 token；
- 热身关闭 `comm_profile` 和 profiler，不进入正式 histogram；
- 启用该协议时关闭旧的“只热身网格第一个点”逻辑，避免首点双重热身。

结果：

- 10/10 workload 的 PatternDemand 在历史三重复和新十重复中完全一致；
- intrinsic `IQR / median` 中位数：历史 0.41%，十重复 1.10%；
- intrinsic 热身后/历史目标中位比：1.000；
- 各 workload 比值范围：0.978–1.005；
- synchronization-inclusive 诊断 IQR 中位数由 24.19% 降至 18.51%。

结论：同形状 warmup 能部分降低 rank 到达等待，但 intrinsic 通信本体本来就
稳定，且热身不会系统性改变目标。完整网格继续保留 warmup 以统一执行协议。

输出：

- `qwen3_8b_warmup_control/`：100 条 compact all-rank 记录、日志、遥测；
- `qwen3_8b_warmup_control_summary/warmup_control.csv`：逐 workload 对照；
- `qwen3_8b_warmup_control_summary/qwen3_8b_warmup_control.png`；
- `qwen3_8b_warmup_control_summary/summary.json`。

## 4. 实验 B：完整 195 点 corrected dataset

网格：

- TP \(\in\{2,4,8\}\)；
- Prefill：B \(\in\{1,2,4,8,16\}\)，
  L \(\in\{128,512,2048,8192\}\)，M=8，共 60 点；
- Decode：B \(\in\{1,2,4,8,16\}\)，
  L \(\in\{128,2048,8192\}\)，
  M \(\in\{32,128,512\}\)，共 135 点；
- 每点三重复，总计 585 条正式记录。

所有记录必须通过：

- all-rank kernel count 与 group-level calls 完全相等；
- backend sequence 跨 rank 完全一致；
- PatternDemand 跨 rank 完全一致且不重复累计；
- intrinsic 不大于 synchronization-inclusive；
- Decode window 为连续 8 steps，并按 full-phase calls 扩展。

结果目录：`qwen3_8b_corrected_all_rank/`。

## 5. 实验 C：corrected-target 四模型评测

完整 workload 在聚合重复后再做 train/validation/test 分组划分，严禁同一
workload 的不同 repeat 跨集合泄漏。

对比模型：

1. total bytes only；
2. three hard bins；
3. continuous message histogram + intrinsic cost curve；
4. continuous histogram + DNN residual。

连续结构模型的小消息部分读取 CustomAllReduce 曲线的
`intrinsic_median_latency_us`，大消息 NCCL 部分读取从既有 raw rank samples
重算的 `intrinsic_min_median_latency_us`。DNN 只拟合
`log(measured_intrinsic / structural_prediction)`。

正式指标包括：

- test-all MAPE、P95 APE、R²；
- Prefill/Decode 分层；
- TP=2/4/8 分层；
- near-equal-payload Decode holdout；
- repeat IQR 与稳定子集。

模型和评测输出目录：`qwen3_8b_prediction_eval_intrinsic/`。

## 6. 数据保存

原始 profiler traces 仅保留在远端：

`/sgl-workspace/sglang-src/experiment-results/phase6/**/traces/`

compact JSONL、CSV、日志、遥测、图片、摘要和模型文件提交 Git。正式论文
引用时必须同时注明：

- PatternDemand 是 group-level；
- payload 是代表 rank 的逻辑输入；
- 时间标签是 all-rank skew-free intrinsic；
- synchronization-inclusive 仅是 rank 到达偏斜诊断。
