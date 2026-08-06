# Phase 14F 初始结果无效

状态：仅保留审计，禁止用于论文、模型比较或后续训练。

本目录最初使用每次 collective 的 `max(rank kernel duration)` 构造微基准曲线，
而 Phase 14C 的主目标是：

```text
max(rank kernel end) - max(rank kernel start)
```

两者在 rank 到达存在偏斜时不等价。初始曲线因此与目标标签的时间契约不一致，得到的
26.086% overall MAPE、67.283% P95 APE 和 35.141% Decode MAPE 均为无效诊断结果。

正式结果位于：

```text
experiment-results/phase14f_post_rendezvous/
```

代码修正提交为 `95940aa9`。旧目录保留是为了说明错误来源和保证实验可审计，不代表
存在两套可供选择的 Phase 14F 结论。

