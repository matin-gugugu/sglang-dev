# Phase73本地控制端提示

从W73创建隔离run分支/worktree，使用带NumPy的本地CPU Python执行。只提交allowlist中的紧凑结果；禁止`git add .`，禁止访问`data/`、raw、完整请求、teacher、GPU、网络或现有未跟踪Phase54–56目录。

候选只按Phase48 validation选择。运行代码必须在选择完成并记录摘要后，才加载Phase50 target做固定基准评分；无论结果好坏都如实保存。
