# 实验资产与保存流程全量索引：截至Phase34

## 目录和提交

|阶段|目录|数据量|主要提交|
|---|---|---:|---|
|34A|`experiment-results/phase34a_six_model_contract/`|156 KiB/14文件|`3480fe68`|
|34B|`experiment-results/phase34b_six_model_hfull_dataset/`|7.3 MiB/17文件|`3480fe68`|
|34C|`experiment-results/phase34c_six_model_target_free_training/`|16 MiB/52文件|`6bdc0208`|
|34D|`experiment-results/phase34d_six_model_blind_evaluation/`|约1.7 MiB|`0c4058f0`起|

远端根目录：`/sgl-workspace/sglang-src`。本地根目录：`/Users/liyafei06/Documents/Codex/2026-07-21/login-klingai-wlf2-ge151-node55-idchb2az2/work/sglang-phase2-curve`。

关键资产包括：34B 六模型 TP/PP dataset；34C 的 TP/PP top3×3seed checkpoint 和 `analysis/frozen_predictions_all_versions.csv.gz`；34D 的 Hfull 紧凑标签、聚合与逐 case 指标、同窗 Phase33 比较和图表。冻结预测 SHA 为 `faffe08800e6336fa9272b765ca1965d4aee806a6162255f7bd9a50d5d5b5bda`。

Phase34A/B/C（含 TP/PP 子目录）/D manifest 均须两端校验。只选择性添加正式文件；禁止 `git add .`；不提交本地 `data/`、远端 Phase16/19/23 保护目录、raw trace、缓存、大模型权重和 PID。
