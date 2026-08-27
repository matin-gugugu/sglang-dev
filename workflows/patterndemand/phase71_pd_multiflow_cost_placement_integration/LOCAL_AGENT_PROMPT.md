# Phase71本地执行合同

从控制端指定的精确W71创建唯一run分支，在干净隔离worktree中完整阅读本目录。先preflight，再run和verify。不得使用GPU、网络、模型权重、受保护data或raw；不得修改Phase49/50/51、R61/R67/R69或wave policy。只按`commit_allowlist.txt`选择性提交结果，禁止`git add .`。回传W71/R71、run分支、结果路径、数据量、cost WAPE/MAPE、placement agreement/regret、wave敏感性、科学结论及边界。
