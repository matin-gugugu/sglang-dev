# Phase 25C：PP scheduler teacher的GPU尾部审计

状态：**PASS**。Phase 25B scheduler-faithful teacher与3/3个实测GPU cell完全一致，
并且12/12个`profile × phase`比较全部精确通过。

## 审计设计

本阶段使用两个完整fixed-draining窗口：

- BurstGPT：48个请求，包含6,216-token长prompt；
- Mooncake conversation：930个请求，包含8,192-token长prompt。

GPU审计选择`PP2/MB1`、`PP4/MB4`和`PP8/MB16`三组对角cell，覆盖跨chunk继续执行、
不同lane数量以及小/大microbatch限制，同时避免运行成本较高的完整笛卡尔矩阵。

## 校验结果与保存内容

每个cell均通过以下检查：GPU执行完整性、sender-boundary一致性、total calls、logical bytes、
12桶calls/bytes守恒以及精确payload直方图。目录保留紧凑GPU直方图和日志；
模型权重、缓存、大体积raw profiler trace和PID文件均不进入归档。

## 科学结论边界

结果支持Phase 25B公式在本次审计的BurstGPT和Mooncake尾部窗口上晋升为正式teacher。
它不能证明online-arrival语义，也不能代替对每个尾部窗口运行全部9种PP/MB组合。

本阶段完成后，下一步是在scheduler-faithful PP teacher下重新计算H32/H64/H128/Hfull收敛；
该工作已经在Phase 25D完成。
