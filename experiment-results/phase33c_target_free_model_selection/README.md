# Phase 33C：TP继续收敛与PP保守改进（打开新确认真值前）

本阶段只在开发数据上选模型。TP使用Phase31与Phase33合并后的94个开发画像、35524个完整teacher请求，比较18组`H0 + DNN residual`；每组先1个seed，开发前三名再做3-seed、5折profile分组确认。PP不重训calls/形状网络，保留Phase32 incumbent，只在45个全新开发画像上比较8种独立bytes校准。

bytes总量锚点来自部署时允许的低维均值、模型bytes/token先验和已验证结构通信倍数，不读取完整请求列表或确认target；开发集审计与Hfull teacher最大相对误差为TP `1.225e-15`、PP `1.130e-15`。bytes的12-bin形状仍保留H0分配。

TP开发五折calls/bytes/TV/EMD/cost WAPE为`8.20%`、`0.00%`、`0.1803`、`0.0204`、`4.90%`。PP新验证结果为`3.99%`、`0.00%`、`0.1491`、`0.0187`、`3.16%`。

9个Phase33全新确认窗口的Hfull target仍不存在。三套预测冻结SHA-256为`f634a6f0c82c0109132e752e4f52f266b9aa8904d947874ce98b32e9dd80f7d0`。Phase31固定集和Phase32确认集后续只能作为重复工程证据。原训练运行在写summary时遇到NumPy布尔序列化错误；模型、九个checkpoint和冻结预测均已完成，本元数据由九个checkpoint重新推断验证后恢复，没有重训或改变候选。
