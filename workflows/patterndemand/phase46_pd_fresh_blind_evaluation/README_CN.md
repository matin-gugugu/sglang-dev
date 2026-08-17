# Phase46 workflow：纯PD fresh blind一次性评估

Phase46只能在R45正式合入后于本地CPU执行。它首先从受保护raw重建同一批300个窗口的低维画像，逐字段核对R45冻结值；全部低于`1e-10`误差后，才允许teacher首次生成Hfull。

评估直接使用R45已经提交的600行预测，不训练、不加载checkpoint、不重算预测。主要成功门预先固定为：整体四项指标严格优于H0，并且三个BurstGPT segment分别通过composite和calls/bytes保护门。另报告20,000次paired bootstrap和10个请求量层，不用于事后选择。

```bash
python3 workflows/patterndemand/phase46_pd_fresh_blind_evaluation/preflight.py --expected-workflow-commit W46 --raw-dir /ABSOLUTE/protected/raw
python3 workflows/patterndemand/phase46_pd_fresh_blind_evaluation/run.py --expected-workflow-commit W46 --raw-dir /ABSOLUTE/protected/raw
python3 workflows/patterndemand/phase46_pd_fresh_blind_evaluation/verify.py
```

无论科学结论正负都必须完整保存，不能删异常窗口、改变门槛或重新训练。
