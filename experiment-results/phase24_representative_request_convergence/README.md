# Phase 24：代表请求规模 H32/H64/H128/Hfull 收敛

本实验复用 Phase 16 固定的24个BurstGPT/Mooncake medoid历史窗口，在相同
fixed-draining语义下用GPU验证过的CPU结构公式生成Qwen3-8B TP和PP消息直方图。
Hfull是唯一参考；完整请求列表只用于离线teacher label，不是部署时预测器输入。

## 输入与策略

- 完整窗口：24个，合计18,285条请求；
- H128：Phase 16的固定4×4联合长度分层代表池；
- H32/H64：从同一H128池再次确定性分层选择，H32逐项匹配既有Phase 16 replay plan；
- compact32：仅由低维画像重建的32个伪请求，用于分离画像重建误差；
- TP：TP2/4/8 × latency/balanced/throughput；
- PP：PP2/4/8 × `pp_max_micro_batch_size=1/4/16`，每条sender boundary计数一次；
- 所有结果归一化到每1000请求，Prefill/Decode另存并提供total汇总。

## 主要结果（phase=total，Hfull为真值）

| 并行 | 样本 | calls MAPE | calls WAPE | bytes MAPE | bytes WAPE | hist TV | norm EMD | common cost MAPE | P95 calls APE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TP | H32 | 11.48% | 7.98% | 5.82% | 4.78% | 0.2751 | 0.0121 | 6.96% | 39.35% |
| TP | H64 | 6.50% | 4.65% | 2.78% | 2.27% | 0.2324 | 0.0096 | 3.88% | 16.44% |
| TP | H128 | 7.57% | 5.27% | 2.01% | 1.37% | 0.2056 | 0.0084 | 4.89% | 27.69% |
| PP | H32 | 8.38% | 4.28% | 5.82% | 4.78% | 0.0922 | 0.0052 | 6.63% | 27.14% |
| PP | H64 | 4.62% | 2.76% | 2.78% | 2.27% | 0.0834 | 0.0041 | 3.27% | 15.40% |
| PP | H128 | 4.70% | 2.67% | 2.01% | 1.37% | 0.0779 | 0.0039 | 3.56% | 22.81% |

预注册门槛下最小充分规模：TP=`None`，
PP=`None`。门槛和逐配置结果见
`summary.json`与`analysis/aggregate_metrics.csv`，不能只按平均calls MAPE挑选样本规模。

## PP误差分解

- exact H32→Hfull calls MAPE：8.38%；
- exact H64→Hfull calls MAPE：4.62%；
- exact H128→Hfull calls MAPE：4.70%；
- compact32→exact H32 calls MAPE：10.39%；
- compact32→Hfull calls MAPE：8.14%；
- 诊断：both finite-sample and compact-profile reconstruction errors are material; compact32 differs from exact H32 more than exact H32 differs from Hfull, but partial cancellation against Hfull makes the two components non-additive。

因没有任一规模同时满足均值、尾部与直方图门槛，首版teacher label应使用
full-window fixed-draining H0；H64/H128可继续作为低成本近似，但不应直接替代teacher。

## Cost口径与结论边界

跨TP/PP的`common_reference_cost`统一使用显式参考曲线
`5 us + payload / 100 GB/s`，只评价同一曲线下的直方图收敛，不是PP物理时延。
TP另在逐样本与聚合文件中报告B200 L1 AllReduce实测连续曲线传播误差。PP P2P物理曲线
尚未测量，因此不能从本实验报告PP真实通信时间MAPE。

本实验可以判断代表请求规模是否逼近full-window fixed-draining teacher，以及画像重建
相对精确代表样本造成多少额外误差；不能证明online arrival期望直方图，也不能替代未来
的PP P2P曲线、多模型PP或预测器留出评测。

## 正式资产

- `input_windows/selected_requests.jsonl.gz`：H32/H64/H128/Hfull与compact32精确长度列表；
- `labels/histogram_labels.jsonl.gz`：phase×配置×payload的每1000请求精确标签；
- `analysis/per_case_metrics.csv.gz`：逐窗口、配置、phase/total误差；
- `analysis/decomposition_metrics.csv.gz`：compact32相对exact H32的画像重建误差；
- `analysis/aggregate_metrics.csv`：MAPE/WAPE/P95/L1/TV/EMD/cost汇总；
- `figures/convergence.svg`：TP/PP收敛曲线；
- `summary.json`、`run.log`、`DONE`、`PIPELINE_DONE`与`manifest.sha256`：审计和完整性。
