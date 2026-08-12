# Phase 30A：TP结构化batch事件与新窗口合同

状态：**PASS**。Phase 29第二确认否决了当前直方图residual checkpoint，但不取消DNN路线。
本阶段在任何Phase 30 target或预测生成前，冻结新的结构化目标和全新窗口。

TP teacher可拆为两类scheduler事件：prefill每个batch的输入token总和，以及decode每一步的
活跃lane数。对当前4096/8192 bytes-per-token两类模型，1–65,536 token可划为
23个联合区间，无损映射到两类模型的TP原生12桶。保存每个区间的batch count和
token mass，再加1–16活跃lane的decode step count，共62个非负目标。
模型的collectives-per-forward与bytes-per-token由确定性结构适配器加入，不再把模型、TP size
和phase展开行误当独立流量样本。

窗口选择排除Phase 16的24个、Phase 27的60个和Phase 28的18个窗口。从BurstGPT三段及
Mooncake conversation/toolagent各选18个history-only medoid，共90个；冻结为45 train、
15 validation、15 first confirmation和15 second confirmation。Mooncake synthetic的12个
合格窗口已被前三轮冻结实验全部使用，因此本轮明确记录为0个可用，而不复用旧确认窗口。

Phase 29的30个train和12个validation画像允许作为开发数据复用；Phase 29两批确认画像永久
关闭，不得调参。Phase 30B训练单位是“独立画像×固定策略”，primary仍是compact32 H0事件
先验加bounded residual DNN，direct仅为控制。两批新确认预测必须在读取第一真值前同时冻结。
