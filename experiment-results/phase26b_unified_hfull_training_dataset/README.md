# Phase 26B：统一 TP/PP Hfull 训练数据集

状态：**PASS**。

本阶段把 Phase 26A 晋升后的 TP Hfull teacher 与 Phase 25B scheduler-faithful PP
Hfull teacher 合并为同一套训练数据契约。共 1728
条 phase-level 样本，其中 TP 1296 条、PP
432 条；每条 Hfull target 都有且仅有一条由低维画像生成的
compact32 H0 baseline 和一条部署可用输入特征记录。

## 输入与输出口径

- 预测输入只含低维常态流量画像、模型结构、已确定的 TP/PP size、固定策略和 phase；
- 完整请求列表只参与上游离线 Hfull teacher 生成，没有复制进本数据集；
- target 是每 1000 请求归一化的 calls、logical bytes、原生 12 桶及 exact payload histogram；
- TP 原生桶范围是 4 KiB–512 MiB，PP 是 4 KiB–8 GiB。两者不能暗中共用同一桶语义，
  因此每条样本显式携带 `bin_schema_id` 和 `bin_edges_bytes_json`；
- `profile_splits.csv` 固化 Phase 16 的 5 train、5 validation、5 temporal test、
  8 external test、1 external synthetic 画像划分。后续任何早停和测试都必须按完整画像分组。

## compact32 H0 相对 Hfull 的未训练基线

| 并行 | cases | calls MAPE/WAPE | bytes MAPE/WAPE | histogram TV | log-payload EMD | common cost MAPE |
|---|---:|---:|---:|---:|---:|---:|
| TP | 648 | 13.14% / 11.67% | 3.29% / 1.33% | 0.3542 | 0.0157 | 8.40% |
| PP | 216 | 62.25% / 21.07% | 3.29% / 1.33% | 0.4483 | 0.0489 | 13.85% |

这里的 common cost 使用统一的 5 μs 启动项和 100 GB/s 参数曲线，只用于比较消息
直方图误差传播，不是 PP 物理链路实测。

## 完整性检查

- 1,728 个 target ID、baseline ID、feature ID 一一对应且无重复；
- 1,296 条 TP target 保持 Phase 26A 的 GPU sentinel 晋升状态；
- 432 条 PP target 保持 Phase 25B/25C 验证过的 scheduler contract；
- Qwen3-8B TP compact32 与 Phase 24 的 432 条记录 exact 回归一致；
- PP Hfull 与 Phase 25D 的 432 条记录 exact 回归一致；
- teacher exact histogram、原生 12 桶和 total calls/bytes 互相复算一致；
- 结果中没有完整请求列表、raw profiler events、模型权重、缓存或 PID。

## 文件

- `labels/hfull_targets.csv.gz`：统一后的 Hfull teacher；
- `baselines/compact32_h0.csv.gz`：一一对应的 compact32 H0；
- `features/low_dimensional_inputs.csv.gz`：部署可用低维输入；
- `training_examples.csv.gz`：可直接用于 Phase 26C 的紧凑 join；
- `splits/profile_splits.csv`：画像级划分；
- `analysis/h0_vs_hfull_per_row.csv.gz`、`h0_vs_hfull_total.csv.gz` 与
  `h0_vs_hfull_aggregate.csv`：phase-level、配置 total 和聚合后的未训练基线；
- `analysis/dataset_inventory.csv`：配置与 split 库存；
- `contract.json`、`summary.json`、`audit_summary.json`、`logs/build.log`、`DONE`
  和 `manifest.sha256`：契约、审计与归档证据。

## 可以与不可以得出的结论

可以确认训练数据标签已经从 Phase 16 的 H32 GPU label 改成 Hfull teacher，且 TP/PP
输入、baseline、target 和 split 在同一契约内可追溯。不能据此宣称模型精度已经改善，
因为 Phase 26C 尚未重训，Phase 26D 尚未做画像级留出评测；也不能把 common cost 当作
PP 实际通信时间。

下一步：在此数据集上分别训练 direct、H0 和 H0+bounded residual，优先报告 TP/PP
分项与 phase 分项，不跨 `bin_schema_id` 混淆桶含义。
