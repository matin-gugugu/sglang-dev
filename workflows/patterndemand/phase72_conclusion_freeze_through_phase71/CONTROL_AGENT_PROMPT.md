# Phase72控制端执行提示

从W72创建隔离的本地run分支和worktree，在`CUDA_VISIBLE_DEVICES=-1`下顺序执行preflight、run、verify。只允许添加`commit_allowlist.txt`中的Phase72结果文件，禁止`git add .`，禁止读取或提交`data/`、本地未跟踪Phase54–56结果、raw、权重、缓存或PID。

结果commit必须只有W72一个父提交。控制端使用统一`verify_result_commit.py --phase phase72`验收后，才可ff-only合入正式分支。
