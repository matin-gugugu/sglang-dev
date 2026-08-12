# Phase 31C：三模型 TP/PP H0+DNN residual 有限训练

本阶段只读取Phase31B的39个训练画像和10个验证画像。10个固定预测画像只有低维特征与H0，不存在Hfull target；预测文件已在target生成前冻结，SHA-256为`84c24637db54e0033f9ed4a0308dd12f92691d8c14d09c935685f56665e472f2`。

## 有限搜索

TP和PP各筛选12组配置：完整/去arrival特征、共享/按policy小头、三档学习率。每个方向只取验证最好的2组做3-seed训练，最终仍由验证集选择一组。没有整模型留出，三个已知模型同时参与训练、验证和固定预测。

最终模型始终是`H0 + DNN residual`。DNN输出经过H0空间的有界残差和验证集校准alpha；H0同时保留为对照。固定预测集没有参与网络、特征、alpha或checkpoint选择。

## 验证结果入口

TP与PP的最终验证指标见`summary.json`中的`selected`；全部24组初筛见`analysis/candidate_grid.csv`。下一步必须先归档本阶段与冻结预测，然后另行生成固定预测Hfull target并评测。
