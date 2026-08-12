# Phase 26A：TP Hfull teacher跨模型GPU审计

状态：**PASS**。本阶段审计TP完整窗口结构teacher是否可以从暂定标签晋升为正式训练真值。

## GPU覆盖

- 计入Phase 25A已有Qwen3-8B/TP2 smoke，并新增3个GPU cell；
- 覆盖Qwen3-8B、Qwen3-30B-A3B、DeepSeek-V2-Lite三个正式模型；
- 覆盖TP2、TP4、TP8；每个cell都同时覆盖latency、balanced、throughput；
- 流量包含42请求最小窗口、312请求中等窗口和6,216-token长prompt尾部。

## 结果

- 4/4 cell通过；
- 24/24个`cell × strategy × phase`比较完全一致；
- 精确payload直方图的calls L1与logical-bytes L1均为0；
- 由直方图重新求和得到的标量只有浮点累加顺序残差：calls最大绝对误差为1.16e-10，logical bytes为3.05e-05 bytes/千请求，均在预设浮点容差内；
- 原Phase 25A的1,296条TP Hfull标签以`GPU_VALIDATED_STRUCTURAL_FORMULA_SENTINELS_4_CELLS`状态晋升并保存在`labels/`。

## 结论边界

全量标签由已经GPU验证的结构公式离线生成，不是1,296次GPU逐条实测。结果只适用于当前fixed-draining、固定TP和固定策略契约，不能外推到online arrival-aware或其他执行语义。

由于监督真值从Phase 16的exact H32改成Hfull，direct DNN与H0+bounded residual必须重新训练；H0没有学习参数，只需在Hfull口径下重新评测。
