# Phase 29B：三模型TP对齐Hfull数据集

本阶段把Phase 27/28已经冻结的同一批历史窗口扩展到三个TP模型。60个Phase 27窗口生成
3,240条Hfull phase labels；其中30/12个开发训练与验证画像形成
2,268条带target样本，18个第一确认画像的972
条feature与同数量target物理隔离。Phase 28的18个第二确认画像只生成972
条无target feature和compact32 H0，尚未生成第二确认Hfull真值，供后续预测先冻结。

覆盖DeepSeek-V2-Lite、Qwen3-8B、Qwen3-30B-A3B，TP2/4/8，latency/balanced/throughput和
prefill/decode。Hfull teacher沿用Phase 26A四个跨模型/TP/策略GPU sentinel精确验证的结构公式；
无需为每个新窗口重跑GPU。每条样本按1000请求归一化，输出保持TP原生4 KiB–512 MiB 12桶。

保存118列特征并冻结两个视图：Phase 26旧55列legacy和Phase 29增强113列。完整请求数组只在
构建器内存中生成Hfull，没有写入profiles、dataset、labels或Git。最终预测器输入仍是低维
历史画像、模型结构、固定TP size、固定策略、phase和H0。
