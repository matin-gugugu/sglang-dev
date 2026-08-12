# Phase 32B：TP/PP扩容有限搜索与预测冻结

本阶段只使用Phase31的39训练、10验证及五折profile-grouped CV；原10固定target和Phase32新确认target均未读取。TP新增24组、累计42组；PP新增18组、累计30组。每个新组初筛1个seed，开发侧前三名做5-fold × 3-seed确认。

TP探索总量/形状分头、共享主干加model/policy小头、sample residual gate和低维顺序/形状摘要；PP探索bytes/cost保护loss和MB独立gate。两个方向均保持`H0 + DNN residual`且residual非零。

选中模型已经同时对不变的原10固定窗口和9个新BurstGPT确认窗口冻结预测，SHA-256为`13147bc92de6b70586d330e0e2ebddcf744aab8a6087f982b7abc4f79160c144`。下一阶段必须先归档本目录，之后才能生成新确认Hfull target；原固定集后续结果只能称为重复工程复评。
