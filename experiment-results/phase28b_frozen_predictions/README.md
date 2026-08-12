# Phase 28B：第二确认集预测冻结

本阶段从Phase 28A冻结的18个窗口聚合108列低维画像，使用Phase 27C已经冻结的增强
bounded-residual checkpoint，并与无参数H0一起生成648行预测。按Phase 28A
方法映射实际选中的预测为324行：MB1使用H0，MB4/MB16使用增强residual。

脚本没有Hfull target参数，没有生成或读取Hfull标签。完整请求列表只在内存中聚合画像，
没有进入`profiles/`、`dataset/`或Git。`analysis/frozen_predictions.csv.gz`的SHA-256已写入
summary和audit；下一阶段必须先核验该hash，随后才可生成Hfull真值并评测。
