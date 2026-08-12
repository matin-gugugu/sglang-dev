# Phase31A-G TP/PP收敛实验里程碑报告

本报告补充今晚执行参考，不替代实验总导引。

| 里程碑 | 实际状态与证据 | Git提交 | 结果目录 | 文件数 | 数据量 |
|---|---|---|---|---:|---:|
| Phase 31A | 冻结59个请求级互斥画像与三模型/TP/PP合同 | `217ad83bcc83` | `phase31a_known_model_convergence_contract` | 9 | 0.02 MiB |
| Phase 31B | 生成开发Hfull teacher与target-free固定特征 | `6352f2a8b8c8` | `phase31b_known_model_hfull_dataset` | 16 | 1.78 MiB |
| Phase 31C | TP/PP各12组有限H0+DNN residual训练并冻结预测 | `eb9b8b7373ec` | `phase31c_known_model_residual_training` | 23 | 2.27 MiB |
| Phase 31D | 首次打开固定Hfull target；TP fail、PP conditional pass | `1d182b0457f3` | `phase31d_known_model_fixed_evaluation` | 11 | 0.38 MiB |
| Phase 31E | TP追加6组加权/多头方案至18组上限，保留incumbent | `0e0dbd90846a` | `phase31e_tp_weighted_residual_round2` | 17 | 2.33 MiB |
| Phase 31F | 同一固定集一致性复评，TP指标逐值不变 | `fde778a8bcc4` | `phase31f_tp_round2_fixed_evaluation` | 9 | 0.17 MiB |
| Phase 31G | 汇总最终裁定、逐模型/逐policy指标、结论边界与下一步 | 本目录归档提交 | `phase31g_tp_pp_convergence_final` | 见manifest | 见manifest |

最终裁定：TP未收口，PP有条件收口，整体第一阶段尚未完全收口。TP固定集calls/cost WAPE为14.33%/8.99%，相对H0改善26.47%/28.64%；PP固定集calls/bytes/cost WAPE为6.91%/4.05%/5.42%。

所有里程碑均包含中文README、summary、logs、DONE与manifest；正式训练阶段保存checkpoint和冻结预测，评测阶段保存逐case/聚合指标与图表。raw trace、完整请求列表、缓存和PID未提交。
