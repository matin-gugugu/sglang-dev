# 实验资产与保存流程全量索引（截至Phase31G）

## Phase31正式资产

| 阶段 | 远端/本地相对目录 | 提交 | manifest状态 |
|---|---|---|---|
| PHASE31A | `experiment-results/phase31a_known_model_convergence_contract` | `217ad83bcc8340941538a41ccd09d472726d6d6c` | PASS |
| PHASE31B | `experiment-results/phase31b_known_model_hfull_dataset` | `6352f2a8b8c877bfdf505ce779c1e881fef6643d` | PASS |
| PHASE31C | `experiment-results/phase31c_known_model_residual_training` | `eb9b8b7373ec41c3f9d123aeb152905414d491b3` | PASS |
| PHASE31D | `experiment-results/phase31d_known_model_fixed_evaluation` | `1d182b0457f3ec8b1cc29ccc2da08e5d2f24de4e` | PASS |
| PHASE31E | `experiment-results/phase31e_tp_weighted_residual_round2` | `0e0dbd90846a02e9cdb6da8bde9899078e57793c` | PASS |
| PHASE31F | `experiment-results/phase31f_tp_round2_fixed_evaluation` | `fde778a8bcc4d1e5b9b3930fcd4ed2158241d4cc` | PASS |
| PHASE31G | `experiment-results/phase31g_tp_pp_convergence_final` | 本目录归档提交 | PASS |

远端根目录为`/sgl-workspace/sglang-src`，本地根目录为`/Users/liyafei06/Documents/Codex/2026-07-21/login-klingai-wlf2-ge151-node55-idchb2az2/work/sglang-phase2-curve`。每个目录均以自身`manifest.sha256`为校验入口。

## 数据量

- 画像：59（39/10/10）；开发完整请求21,058，固定完整请求2,786；
- 开发Hfull标签5,292条phase rows；固定Hfull标签1,080条phase rows；
- Phase31C checkpoint 12个，Phase31E候选checkpoint 6个；
- 冻结预测：Phase31C共2,160条phase-method rows；Phase31E TP共1,080条。

## 保存流程

1. 先冻结选择与SHA；2. target-free训练并冻结预测；3. 选择性`git add`/`git add -f`正式目录；4. commit/push；5. 另一端`git pull --ff-only`；6. 在目录内校验manifest；7. 只在预测归档后打开固定target。

## 排除资产

不提交本地`data/`、raw profiler trace、完整请求列表、模型权重缓存、旧Phase16 GPU目录、Phase19 formal/smoke、Phase23 PID/tmp及任何PID文件。Phase31中的`.pt`是本次正式小型DNN checkpoint，不是模型权重缓存。


# Phase32A-F资产补充

- `experiment-results/phase32a_expanded_search_contract/`：扩容合同、新互斥窗口选择、无target特征与清单；84 KiB级。
- `experiment-results/phase32b_expanded_residual_search/`：42个新增候选初筛、18个正式checkpoint、4,104条冻结预测；约13 MiB。
- `experiment-results/phase32c_frozen_prediction_evaluation/`：新确认Hfull标签、一次性评测、逐切片指标；约652 KiB。
- `experiment-results/phase32d_tp_gate_rescue/`：TP六组定向gate、gate checkpoint、2,052条冻结预测；约724 KiB。
- `experiment-results/phase32e_tp_rescue_repeated_evaluation/`：TP救援后重复工程评测；约292 KiB。
- `experiment-results/phase32f_tp_pp_expanded_convergence_final/`：最终报告、摘要、逐模型/逐policy/逐并行规模指标、搜索清单、checkpoint/预测索引、图表、日志、DONE与manifest。

正式提交链：`b476e093`（32A）、`89b91d9b`（32B）、`0a8b5d74`（32C）、`842479a7`（32D结果）、`c1f73c6f`（32E结果）；32F以最终HEAD为准。所有大体积raw、完整请求列表、缓存和PID均未提交。
