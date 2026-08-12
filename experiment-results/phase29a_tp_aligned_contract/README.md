# Phase 29A：TP与PP历史窗口对齐合同

本阶段在生成任何Phase 29 TP Hfull标签或预测前，冻结TP重新训练合同。TP最终方法仍是
`H0 + bounded residual DNN`；Phase 26D的H0胜出只说明5个训练画像和55列旧特征下的
residual没有跨域泛化，H0在本合同中是结构先验、baseline和失效回退，不替代DNN。

历史流量骨架与PP严格对齐：复用Phase 27的60个窗口及30/12/18角色，再复用Phase 28的18个
第二独立确认窗口。TP保持自己的机制语义：三个模型、TP2/4/8、latency/balanced/throughput、
prefill/decode和TP原生12桶。相同的是历史窗口、split、每1000请求归一化和评测指标；不同的
是TP teacher、TP batching特征和输出桶，不能为了形式统一而混用PP语义。

Phase 29B预计生成Phase 27骨架上的3,240条TP Hfull phase labels，其中1,620/648/972分别用于
训练、验证和第一确认；Phase 28骨架再生成972条第二确认标签。完整请求列表只用于离线teacher，
最终预测器仍只读取低维历史画像、模型结构、固定TP配置、策略、phase和H0。
