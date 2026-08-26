# Phase67控制端验收提示词

仅接受父提交唯一且等于W67的R67。先运行`verify_result_commit.py --phase phase67 --workflow-commit W67 --result-commit R67`，再在R67树上执行本目录`verify.py`。检查只有allowlist结果、manifest完整、Phase68 targets为零、GPU/network/new measurement均为false、候选按固定顺序选择。全部通过才允许ff-only合入正式分支；Phase67 PASS不能写成Phase68 fresh-blind PASS。
