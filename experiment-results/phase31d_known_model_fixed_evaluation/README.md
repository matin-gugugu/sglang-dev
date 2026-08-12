# Phase 31D：三模型固定预测集最终评测

本阶段在Phase31C预测文件和SHA冻结之后，才读取10个固定预测窗口的完整请求并生成Hfull target。训练、候选选择、alpha和checkpoint均未读取这些target，固定预测窗口也未因结果而更换。

## 范围与规模

- 模型：DeepSeek-V2-Lite、Qwen3-8B、Qwen3-30B-A3B；
- 固定画像：10个，与训练/验证窗口请求级不重叠；
- TP：TP2/4/8 × latency/balanced/throughput；
- PP：PP2/4/8 × MB1/4/16；
- Hfull target：1,080条phase rows；冻结预测：2,160条phase rows（H0与H0+DNN residual）；
- 逐case评测：3,240条，包含prefill、decode和total。

## 裁定

- TP：`fail`；
- PP：`conditional_pass`。

完整整体、单模型、policy、并行规模和来源指标见`analysis/aggregate_metrics.csv`；核心裁定见`summary.json`。结论仅限当前三个已知模型与正常历史流量范围，不代表未见模型、极端流量或所有生产环境的零样本泛化。
