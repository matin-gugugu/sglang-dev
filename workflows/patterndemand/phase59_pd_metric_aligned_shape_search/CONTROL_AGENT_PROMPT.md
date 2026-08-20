# Phase59 控制端/本地执行交接

必须从精确 W59-fix HEAD 运行。先执行 `preflight.py`，再以 `PYTHONUNBUFFERED=1` 运行 `run.py`；stdout/stderr 和 `--runtime-state` 都写到 Git 外路径，并使用独立 detached session，便于长时间监控和异常后的同 commit 断点恢复。不得使用 GPU、联网、raw、Phase50 blind、完整请求或模型权重。

搜索在 OOF 达标时提前结束，否则持续到约 9 小时搜索预算，并为最终训练与 validation 预留约 1 小时。每个候选都会输出一行进度；不要因为短期未达标而人工停止。只有脚本异常、数据安全风险或合同定义的 BLOCKED 才中断。

完成后运行 `verify.py`。正式结果只能选择性提交 `experiment-results/phase59_pd_metric_aligned_shape_search/`；禁止 `git add .`。结果 commit 必须以实际 W59-fix 为唯一父提交，且不能包含本地 `data/`、历史未跟踪结果、用户修改中的开题文档、PID、raw、JSONL、权重、runtime checkpoint 或 cache。
