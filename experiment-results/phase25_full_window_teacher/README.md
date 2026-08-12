# Phase 25A：完整窗口 fixed-draining 结构 teacher

最终状态：**PROVISIONAL_PP_SCHEDULER_MISMATCH（旧PP结构teacher存在scheduler语义不一致）**。
TP full-window smoke精确通过；旧静态PP标签只保留为失败基线，不能称为GPU真值，也不得用于正式训练。
PP teacher的后续修正与验证见Phase 25B和Phase 25C。

## 实验契约

- 离线teacher输入：按原始顺序保留完整窗口内每个截断后请求长度；调度时不使用arrival timestamp。
- 最终预测器输入不变：紧凑历史画像（compact profile）+ 模型结构 + 固定TP/PP配置 + 固定策略 + phase。
- 输出：按每1,000请求归一化的精确直方图，以及12桶calls/logical-bytes直方图。
- PP另外为每条sender boundary保存一条显式标签。

## 暂定资产

- TP标签：1,296条。
- PP phase标签：432条。
- PP boundary标签：1,584条。
- 完整窗口请求总数：18,285条。
- Phase 24 Qwen Hfull回归：864/864条精确直方图一致，864/864条标量记录一致。

`gpu_audit/sentinel_profiles.csv`记录确定性的流量来源与尾部覆盖；
`gpu_audit/plans/`保存TP trace-replay计划和PP完整draining请求列表。
只有GPU calls、logical bytes、精确直方图、12桶守恒和PP sender boundary均通过，
相应标签才能晋升为正式teacher数据集。

本目录不包含大体积raw trace、模型权重、缓存或PID文件。

## GPU smoke结果

42请求完整窗口smoke完成了1个TP cell和全部9个PP cell。TP与结构teacher完全一致。
PP的9/9 cell均通过GPU执行完整性检查，但旧静态teacher只与3个MB1 cell精确一致；
MB4/MB16的6个cell均出现scheduler拆分/合并不一致。logical bytes仍完全一致，
但calls、直方图形状和代入曲线后的通信代价发生变化，因此旧静态PP标签不能作为正式训练真值。

后续Phase 25B已恢复scheduler-faithful PP teacher；本目录保留Phase 25A结果作为发现旧公式缺口的正式证据。
详细smoke结果见`analysis/gpu_smoke-v1/README.md`和`gpu_audit/smoke_summary.json`。
