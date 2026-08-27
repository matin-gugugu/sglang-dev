# Phase73：Direct-GBDT独立baseline

本阶段回答一个简单问题：完全不用H0和32个伪请求，直接让树模型从低维画像预测24个直方图bin，效果如何？

候选容量只在Phase48开发/验证集选择；选完并refit后，才在固定Phase50六模型300画像基准上机械比较`H0`、`H0+DNN`和`Direct-GBDT`。Phase50标签早已公开，因此结果严格属于`target-open fixed benchmark`，不是新盲测。

Direct-GBDT输入只允许`feature_*`；合同和验收明确禁止`h0_*`、伪请求、teacher、raw、完整请求、GPU和网络。

```bash
python3 preflight.py --expected-workflow-commit "$W73"
python3 run.py --expected-workflow-commit "$W73"
python3 verify.py
```
