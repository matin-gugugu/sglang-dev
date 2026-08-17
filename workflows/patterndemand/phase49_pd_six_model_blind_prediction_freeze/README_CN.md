# Phase49：六模型 fresh blind 预测冻结

Phase49 在本地 CPU 上运行，不训练、不用 GPU、不联网。它从 Phase15 的 history-only 窗口表冻结 300 个新窗口；每个窗口与全部历史训练、验证和旧 blind 窗口至少相隔 300 秒。选择按三个 BurstGPT 段、十个请求量分层，每格 10 个。

对每个低维画像展开六个模型结构，加载 R48 原 checkpoint，输出 1800 条 target-free 特征和 3600 条 H0/H0+DNN 预测。此阶段严禁生成或访问 Hfull/target/residual；完整请求只用于重建历史画像，不能进入 Git。

```bash
python3 workflows/patterndemand/phase49_pd_six_model_blind_prediction_freeze/preflight.py --expected-workflow-commit "$W49" --raw-dir data/phase15_traces/raw
python3 workflows/patterndemand/phase49_pd_six_model_blind_prediction_freeze/run.py --expected-workflow-commit "$W49" --raw-dir data/phase15_traces/raw
python3 workflows/patterndemand/phase49_pd_six_model_blind_prediction_freeze/verify.py
```

R49 通过并 ff-only 合入前不得编写或运行会生成这 300 个窗口 Hfull 的 Phase50 结果流程。
