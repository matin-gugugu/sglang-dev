# Phase50：六模型纯PD fresh blind 评估

Phase50 只在 R49 已正式合入后运行。它先从受保护 raw 重建 R49 的 300 个低维画像并展开六模型，要求 1800 条 feature/H0 与冻结文件逐字段一致；全部通过后才首次生成 1800 条 Hfull 标签，并只和 R49 已提交的 3600 条预测对比。

科学硬门包括：overall 四指标都优于 H0；六个模型逐个四指标都优于 H0；三个 BurstGPT 段均保护。另报告 18 个模型×流量段单元、十个请求量 strata，以及以画像为 cluster、六模型不拆散的 20000 次 bootstrap。所有这些都不能用来事后调模型、alpha 或删样本。

```bash
python3 workflows/patterndemand/phase50_pd_six_model_blind_evaluation/preflight.py --expected-workflow-commit "$W50" --raw-dir data/phase15_traces/raw
python3 workflows/patterndemand/phase50_pd_six_model_blind_evaluation/run.py --expected-workflow-commit "$W50" --raw-dir data/phase15_traces/raw
python3 workflows/patterndemand/phase50_pd_six_model_blind_evaluation/verify.py
```

这只能回答六个已知模型结构在 pure P1-D1、BurstGPT fresh blind 上是否优于 H0；不回答未见模型、Mooncake、物理通信时间、placement、延迟或在线调度。
