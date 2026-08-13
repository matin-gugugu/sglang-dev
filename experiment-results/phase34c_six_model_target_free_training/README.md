# Phase 34C：六模型TP/PP训练与新确认预测冻结

本阶段在同一94个开发画像、35,524个唯一完整teacher请求上，分别重新训练六模型TP与PP `H0 + DNN residual`。两个方向均完成18组常规初筛和前三名3-seed × 5-fold确认；同一画像派生的六模型、并行配置、policy和phase从未跨折。

TP开发OOF的calls/bytes/TV/EMD/cost WAPE为`7.94%`、`0.00%`、`0.1877`、`0.0220`、`4.26%`。PP对应为`4.90%`、`0.00%`、`0.1432`、`0.0190`、`3.46%`。两个选中模型都保留非零DNN residual。

在Phase34新确认target不存在时，已冻结六模型TP/PP预测；同时把Phase33三模型incumbent对同一批新窗口的预测作为可比基线冻结。合并冻结文件SHA-256为`faffe08800e6336fa9272b765ca1965d4aee806a6162255f7bd9a50d5d5b5bda`。只有本目录完成Git归档后，才允许一次性打开12个新确认窗口的Hfull target。
