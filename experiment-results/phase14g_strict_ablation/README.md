# Phase 14G：同口径消融与未见 payload 支撑留出

所有 workload 标签和微基准曲线均使用 all-rank post-rendezvous 时间契约。

## Workload-CV

| 方法 | MAPE | P95 APE |
|---|---:|---:|
| total bytes + phase scale | 46.800% | 154.374% |
| payload histogram × TP + phase scale | 7.202% | 21.654% |
| exact raw op，不校准 | 4.389% | 11.060% |
| exact raw op + phase scale | 4.425% | 10.762% |
| raw op，逐支撑点 LOO 插值 | 4.538% | 12.447% |
| backend-proxy 分段，逐支撑点 LOO 插值 | 4.684% | 12.687% |

在当前 105 个精确支撑点上，`raw_op + payload + TP` 唯一决定 pre-run backend
proxy，所以 exact raw-op 与 exact backend-proxy 查表不能得到两个独立分数。
backend proxy 的增量价值只在未见 payload 插值中评估。

## 未见 payload 支撑点

逐个隐藏 105 个曲线支撑点、只使用其余点插值时：

- 不区分 backend 的曲线级 MAPE 为 2.349%，
  P95 为 16.071%；
- 按运行前 backend proxy 分段后，曲线级 MAPE 为
  2.101%，P95 为
  12.200%；
- 在 12 个 proxy 边界点上，MAPE 从
  2.980% 降至
  0.813%；
- 将全部 LOO 插值曲线重新卷积到 162 个 workload 后，raw-op 插值和 backend
  分段插值 MAPE 分别为 4.538% 与
  4.684%。

因此，backend 分段对算法边界的单点插值明显有帮助，但没有改善当前 workload
总体 MAPE；主要增益顺序是 total bytes → payload histogram → raw op 精确曲线。
phase calibration 略微改善 P95/RMSE，但不改善平均 MAPE。

## 产物

- `workload_metrics.csv` / `workload_predictions.csv`；
- `support_holdout_metrics.csv` / `support_holdout_predictions.csv`；
- `summary.json`。
