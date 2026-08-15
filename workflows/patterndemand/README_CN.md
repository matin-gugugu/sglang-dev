# PatternDemand跨环境实验workflow

这里的workflow不是GitHub Actions，而是给GPU或控制环境中的Agent执行的、可审计的实验合同。

每个workflow分为两层：

1. 不可改变的合同：研究语义、冻结输入、指标口径、最小重复次数、结果格式和Git边界。
2. 可自主判断的执行手册：Agent可以诊断环境、选择同类GPU对、有限重试、增加warmup或重复次数，但必须写入`decision_log.jsonl`，且不得改变实验问题。

统一执行顺序：

```text
workflow commit W
  -> 对应执行环境从W创建run分支
  -> preflight
  -> run（一条命令）
  -> verify
  -> 只添加允许的结果目录
  -> 单个result commit R
  -> push run分支
  -> 原环境验证R的父提交、路径和manifest
```

当前三个workflow：

- `phase36_cross_environment_replay`：一张GPU即可，不训练、不读teacher，复播Phase34冻结的六模型TP/PP直方图并演练commit回传。
- `phase37_pp_single_node_p2p_curve`：至少两张GPU，实测单机PP GPU P2P连续曲线；raw逐次样本保存在仓库外，只提交紧凑曲线与审计。
- `phase38_pp_physical_curve_cost_recompute`：Phase37结果验收并合入后，在CPU上将Phase34冻结PP直方图与Hfull target确定性代入物理P2P曲线；不加载checkpoint、不重训。本workflow只有在R37合入后才能提交形成W38，以保持固定W36/W37的ff-only结果链。

完整交接见`PatternDemand跨环境GPU执行交接_Phase36_Phase37.md`。
