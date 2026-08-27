# Phase71：PD多流通信代价与placement重新集成

Phase52只完成了`P1D1直方图 × Phase51单链路曲线`。Phase71把后续已经GPU验证的R61两流修正和R69四流高-page修正接回同一条CPU主链：冻结的H0/H0+DNN与Hfull直方图先按总calls守恒、各边均衡分流组成固定饱和wave，再计算L1/L2/L3 communication-only代价和placement。

直方图没有原始请求顺序，也没有消息之间的联合并发关系。因此正式口径固定为`bin_aligned`边际配对；`cyclic_staggered`与`opposed_extremes`只报告敏感性，禁止看结果后挑策略。该过程不是恢复真实wave，也不是完整scheduler。

覆盖范围严格分层：P1D1与R61的P1D2/P2D1覆盖六模型；R69的P1D4/P4D1/P2D2 matching/all-to-all只覆盖Qwen3-8B和DeepSeek-V2-Lite。所有输入和公式冻结，不训练、不重算teacher、不用GPU或网络。

运行：

```bash
python3 workflows/patterndemand/phase71_pd_multiflow_cost_placement_integration/preflight.py --expected-workflow-commit <W71>
python3 workflows/patterndemand/phase71_pd_multiflow_cost_placement_integration/run.py --expected-workflow-commit <W71>
python3 workflows/patterndemand/phase71_pd_multiflow_cost_placement_integration/verify.py
```
