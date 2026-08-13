# 实验资产与保存流程全量索引：截至Phase35

- 目录：`experiment-results/phase35_six_model_inference_cost_integration/`
- 首次正式结果提交：`d7e74bea2b2f9de4f9d1cb169e25a7e487a85f7d`
- 脚本：`scripts/run_phase35_six_model_inference_cost_integration.py`
- 直方图：`predictions/unified_six_model_histograms.csv.gz`
- 预测冻结：`predictions/PREDICTION_FREEZE.json`
- 曲线契约：`contracts/topology_curve_registry.json`
- 全量 cost：`costs/placement_costs.csv.gz`
- 指标：`analysis/cost_metrics.csv`
- 通信排名：`analysis/communication_only_rankings.csv.gz`
- 图表：`figures/topology_cost_wape.svg`
- 审计入口：`README.md`、`summary.json`、`audit_summary.json`、`logs/runtime.log`、`DONE`、`manifest.sha256`

node55 根目录为 `/sgl-workspace/sglang-src`；本地根目录为 `/Users/liyafei06/Documents/Codex/2026-07-21/login-klingai-wlf2-ge151-node55-idchb2az2/work/sglang-phase2-curve`。

正式结果只选择性添加，禁止 `git add .`；不得提交本地 `data/`、远端 Phase16/19/23 保护目录、raw trace、缓存、大模型权重或 PID。同步只用 ff-only，并在两端校验 manifest。
