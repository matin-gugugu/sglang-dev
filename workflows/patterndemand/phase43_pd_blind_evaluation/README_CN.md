# Phase43 workflow：纯PD一次性blind评估

Phase43只能在R42正式合入以后创建和执行。它不训练、不加载checkpoint、不重新生成预测，而是读取R42已经提交的24行冻结预测，再从Git外六个受保护raw源重建12个blind完整窗口并生成Hfull标签。

推荐在新的本地worktree执行，将主仓库受保护raw作为显式只读输入：

```bash
python3 workflows/patterndemand/phase43_pd_blind_evaluation/preflight.py --expected-workflow-commit W43 --raw-dir /ABSOLUTE/protected/raw
python3 workflows/patterndemand/phase43_pd_blind_evaluation/run.py --expected-workflow-commit W43 --raw-dir /ABSOLUTE/protected/raw
python3 workflows/patterndemand/phase43_pd_blind_evaluation/verify.py
```

允许进入Git的是12行Hfull直方图标签、冻结预测的评价指标、bootstrap和审计。禁止提交原始trace行、时间戳流、2,887个请求的长度对或任何模型/缓存/PID。无论结果正负都必须提交；负结果不能通过换模型、换窗口或重新训练来消失。
