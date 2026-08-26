# Phase69控制端验收提示词

仅接受父提交唯一且等于W69的R69。先运行`verify_result_commit.py --phase phase69 --workflow-commit W69 --result-commit R69`，再在R69树上执行本目录`verify.py`。检查只有allowlist结果、manifest完整、Phase70 targets为零、GPU/network/new measurement均为false、R67在page<=32严格保持、候选按固定顺序选择。全部通过才允许ff-only合入正式分支；Phase69 PASS不能写成Phase70 fresh-blind PASS。
