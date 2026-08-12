# Phase 32A：扩容有限搜索合同与新确认特征

本阶段落实搜索上限扩容补充：TP累计常规/绝对上限42/48，PP累计常规/绝对上限30/36；Phase31已有TP 18组、PP 12组计入累计。初筛每组1个seed，每个方向开发侧前三名做3-seed确认。

在任何新确认Hfull target生成前，本阶段冻结9个新的BurstGPT正常窗口（BurstGPT三段各3个）及其低维特征、compact32 H0。它们与Phase27/28/30/31所有角色都满足300秒请求区间互斥，TP/PP各486条feature rows，不含任何target。Mooncake在累计embargo下没有剩余完整容量，因此新确认的证据范围明确限定为BurstGPT；原10个固定窗口不变。

完整请求列表未保存；以后只允许在预测文件与SHA归档后用于一次性生成Hfull teacher。
