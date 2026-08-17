# Phase46控制端执行提示

在精确W46的干净本地CPU worktree中运行preflight、run、verify。先完整复现并核对R45的300行target-free特征，之后才允许一次性生成Hfull。禁止GPU、联网、训练、加载checkpoint、重算预测、删样本或改变预注册指标/门槛。结果只提交300行直方图标签、指标和审计；完整请求与raw不得进入Git。
