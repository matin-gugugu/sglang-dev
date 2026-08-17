# Phase45 workflow：纯PD fresh blind预测冻结

本阶段在本地CPU执行，只冻结全新的blind输入和预测，不生成Hfull、不训练、不评分。

300个窗口分别来自三个BurstGPT segment，每段100个，并按请求数分成10层。所有窗口彼此至少间隔300秒，且避开Phase27–44使用过的所有区间。运行时用受保护raw重建低维画像，完整请求只短暂存在内存，不进入Git。

```bash
python3 workflows/patterndemand/phase45_pd_fresh_blind_prediction_freeze/preflight.py --expected-workflow-commit W45 --raw-dir /ABSOLUTE/protected/raw
python3 workflows/patterndemand/phase45_pd_fresh_blind_prediction_freeze/run.py --expected-workflow-commit W45 --raw-dir /ABSOLUTE/protected/raw
python3 workflows/patterndemand/phase45_pd_fresh_blind_prediction_freeze/verify.py
```

必须原样加载R44 checkpoint：`pd44_causal_w32_d1`、alpha=0.5、epochs=533、三个固定seed。R45正式合入前禁止制作或读取这300个窗口的Hfull标签。
