# Phase33资产索引

分支：`experiment/pattern-demand-v0.5.15-clean`。

|阶段|目录|提交|
|---|---|---|
|33A|`experiment-results/phase33a_fresh_data_contract/`|`6aa760a3`|
|33B|`experiment-results/phase33b_expanded_development_dataset/`|`e8a9fc22`|
|33C|`experiment-results/phase33c_target_free_model_selection/`|`284dcaef`|
|33D|`experiment-results/phase33d_blind_confirmation_evaluation/`|`3222f972`|

关键资产：

- TP三seed checkpoint：Phase33C的`checkpoints/tp_top2_seed*.pt`。
- PP bytes calibration：Phase33C的`checkpoints/pp_bytes_calibration.json`。
- 冻结预测：Phase33C的`analysis/frozen_predictions.csv.gz`。
- 新盲测紧凑标签：本目录`labels/phase33_blind_hfull_targets.csv.gz`。
- 聚合与逐case指标：本目录`analysis/aggregate_metrics.csv`、`analysis/per_case_metrics.csv.gz`。
- 中文README、summary、logs、图表、DONE和manifest均已归档。

node55仓库为`/sgl-workspace/sglang-src`；本地仓库为`/Users/liyafei06/Documents/Codex/2026-07-21/login-klingai-wlf2-ge151-node55-idchb2az2/work/sglang-phase2-curve`。

继续保护本地`data/`、远端Phase16/19/23旧目录、raw trace、缓存、大模型权重和PID；禁止`git add .`。
