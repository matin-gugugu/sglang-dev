# Phase44 workflow：扩展PD数据与H0保护残差训练

Phase44在本地CPU执行。它从三个BurstGPT segment各冻结400个全新、不重叠且避开历次实验窗口的300秒画像，共1200个画像和486,242个完整请求。完整请求只在内存中供已验证teacher生成标签，永不进入Git。

960个训练画像内部做五折OOF候选和alpha选择；240个验证画像只在候选冻结后评估一次。DNN必须在OOF和验证整体的calls WAPE、bytes WAPE、TV、EMD四项全部严格优于H0，并通过三个segment保护门，才设置`new_blind_permitted=true`。

```bash
python3 workflows/patterndemand/phase44_pd_expanded_protected_training/preflight.py --expected-workflow-commit W44 --raw-dir /ABSOLUTE/protected/raw
python3 workflows/patterndemand/phase44_pd_expanded_protected_training/run.py --expected-workflow-commit W44 --raw-dir /ABSOLUTE/protected/raw
python3 workflows/patterndemand/phase44_pd_expanded_protected_training/verify.py
```

Phase43的12个标签禁止用于本阶段训练或选择。即使模型门失败，扩展数据与负结果也应提交；不得降低门槛或直接进入新blind。
