# Phase 25B：scheduler-faithful PP完整窗口teacher

状态：在本阶段限定的fixed-draining scheduler契约下为 **PASS**。
从SGLang源码恢复的模拟器与已保存的9/9个GPU smoke cell完全一致；
核对范围包括calls、logical bytes、payload直方图、phase标签、active batch size和active tokens。

## 相比Phase 25A修正了什么

Phase 25A的PP公式采用静态prefill/decode分组，而真实SGLang会为每条PP loop lane维护独立running batch，
允许全局chunked request跨lane继续执行，并且在过滤已经完成的decode请求之前先尝试prefill admission。
因此，一个刚释放的slot会先产生一次缩小后的decode forward，到下一次访问该lane时才补入新请求。
此外，只有按page向上取整计算chunk budget，才能精确恢复prefill payload。

## 数据资产与校验

- 完整窗口：24个画像、18,285条请求，保持原始顺序并采用fixed-draining。
- PP phase标签：432条。
- 显式sender-boundary标签：1,584条。
- CPU不变量审计：216个配置全部完成请求，且token mass全部守恒。
- GPU smoke：42请求BurstGPT sentinel上的`PP2/4/8 × MB1/4/16`，9/9 cell完全一致。

## Phase 25A旧静态公式的偏差

以新的scheduler-faithful标签为reference，旧静态公式的overall calls WAPE为121.59%，
平均直方图TV为0.5557，normalized log-payload EMD为0.0941，reference-cost MAPE为44.58%。
logical bytes仍然守恒。

- MB1：calls WAPE为31.93%；它在不发生跨chunk的smoke窗口上精确，但在包含长prompt的完整窗口上并不普遍精确。
- MB4：calls WAPE为208.36%。
- MB16：calls WAPE为603.21%。

## 科学结论边界

这些标签只适用于本阶段记录的fixed-draining契约。9/9完全一致说明一个异构完整窗口上的全部
PP size/microbatch组合通过验证，并不代表已经覆盖所有流量分布。online arrival、preemption、
radix cache、mixed chunk、非零async PP depth、speculative decoding和其他策略都需要独立teacher或审计。

最终预测器仍只读取紧凑历史画像、模型结构、固定PP配置、固定策略和phase；
完整请求列表仅用于离线生成训练标签。本目录不包含raw profiler trace、模型权重、缓存或PID文件。
