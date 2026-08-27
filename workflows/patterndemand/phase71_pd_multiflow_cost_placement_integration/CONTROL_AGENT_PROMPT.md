# Phase71控制端验收

先运行`verify_result_commit.py --phase phase71 --workflow-commit <W71> --result-commit <R71>`，检查唯一父提交、路径、禁止资产和commit内manifest；再在R71树上运行本目录`verify.py`独立复算。确认正式policy始终为`bin_aligned`、总call mass守恒、R69只用于两个代表模型、46800行cost和15600个placement decision均可重建，并且没有训练、GPU、网络、teacher重算或原始顺序恢复声明。
