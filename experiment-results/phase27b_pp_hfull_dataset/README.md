# Phase 27B：新窗口 PP Hfull 数据集

本阶段按照 Phase 27A 已提交的事前合同，处理 60 个此前未使用的 300 秒历史窗口：
30 个开发训练、12 个开发验证、18 个独立确认。完整请求列表只在本脚本内存中用于两件事：
聚合低维画像，以及通过 Phase 25B/25C 已经 GPU 验证的 SGLang PP fixed-draining
调度公式生成 Hfull teacher；结果中不保存任何请求列表。

## 规模与隔离

- 完整历史请求：50,274 条；
- 低维输入特征：108 列；
- Hfull 目标：1,080 个 phase rows（PP2/4/8 × MB1/4/16 × prefill/decode）；
- 开发目标：756 rows；独立确认目标：324 rows；
- `dataset/independent_confirmation_features.csv.gz` 明确不含 target 列；训练阶段不得读取
  `labels/independent_confirmation_hfull_targets.csv.gz`。

## 证据与口径

六个公开 trace 文件的大小和 SHA-256 全部匹配 Phase 15 manifest。共执行
1,080 次 Hfull/compact32 调度模拟；每次都验证请求完成、prefill
token mass 和 decode token mass 精确守恒。teacher 是每 1000 请求归一化的、单 PP boundary
拓扑无关消息直方图；`pipeline_*` 仅通过 `pp_size-1` 给出链路边界总量审计。

本阶段没有计算独立确认集的 H0 或学习器误差，因此仍不能声称新增特征改善了 PP。下一步应
只读取开发集训练并冻结 checkpoint，随后由独立评测脚本加载确认集真值。
