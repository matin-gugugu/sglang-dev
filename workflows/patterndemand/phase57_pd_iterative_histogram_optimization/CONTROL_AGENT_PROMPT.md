# Phase57 控制端交接

远端 Agent 只需按 README 运行 W57-fix；这不是 GPU 实验，不要登录 node55，不要申请 GPU，不要下载模型。若旧 W57 进程仍在运行，先按其 PID 文件精确停止旧进程，再从 W57-fix HEAD 重新开始，不能让两个版本同时写同一结果目录。

控制端在运行前必须确认 HEAD 等于 W57、Phase48 manifest/compact examples 和 Phase54 三个源码 SHA 一致，工作树的受保护 `data/` 与旧 Phase54–56 结果之外没有未经授权的改动。运行后先执行 `verify.py`，再阅读 `summary.json`、`analysis/round_trace.csv`、`analysis/oof_candidate_metrics.csv`、`analysis/oof_bin_bias.csv`、model/segment validation 和 `logs/runtime.log`。

只有以下条件全部为真才可报告达到合同，并允许下一次 blind-freeze workflow：OOF protection、OOF target、一次性 validation overall/model/segment guards、total WAPE 和四项严格优于 H0。若 `target_met=false`，这是有效的 development negative result；不得把 validation 再用于调参，不得打开 Phase50 blind。W57-fix 的候选上限和 MLP scope 是硬约束，不能为追求精度恢复旧版预算。

结果目录只能选择性提交，正式结果 commit 必须以 W57 为唯一父提交；禁止 `git add .`、raw、JSONL、完整请求、权重、cache、PID 和 `.pt/.pth/.safetensors`。
