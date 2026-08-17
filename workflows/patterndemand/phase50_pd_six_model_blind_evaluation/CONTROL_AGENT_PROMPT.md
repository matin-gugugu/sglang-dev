# Phase50 控制端合同

只在 exact W50 且 R49 已正式合入时执行。必须先通过 preflight 和 1800 行 feature/H0 reconstruction gate，之后才允许 target access。不得训练、加载 checkpoint、重算或改动冻结预测、调门、删异常或降低样本。

结果只选择性添加 Phase50 allowlist；忽略文件逐个 `git add -f`，禁止 `git add .`。R50 唯一父提交必须是 W50，经 `verify.py` 与 `verify_result_commit.py --phase phase50` 双验收后才能 ff-only 合入。
