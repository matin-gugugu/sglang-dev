# Phase52控制端任务

在exact W52的隔离本地run分支执行本目录README中的preflight、run和verify。不得使用GPU/网络/模型权重/受保护raw，不得训练、加载checkpoint、重算预测、重建teacher或修改Phase49–51输入。

正式使用Phase51未经平滑的official曲线；lower replica和累计最大单调曲线只做敏感性诊断。候选严格为同一固定P1/D1配置下的L1/L2/L3，禁止把TP/PP/P/D size或任何计算、内存、可用性、排队因素加入决策。

PASS后只选择性暂存Phase52结果，运行`verify_staging.py --phase phase52`，创建唯一父提交为W52的R52。回传W52/R52、输入pins、10800 cost行、3600 placement决策、H0与H0+DNN的cost WAPE/MAPE、agreement/regret、区间robust与单调敏感性、允许和不允许得出的结论。禁止`git add .`。
