# Phase 16F：ProfileDemand GPU 正式标签集

GPU 矩阵已完成 3 个模型 × TP2/4/8 × 24 个服务画像 × 3 种策略。底层共执行
4905 个 histogram-only microbatch workloads，聚合为
1296 条 `model×TP×profile×strategy×phase` 正式标签。

每条标签包含每 1000 请求的 total calls、logical bytes、12 桶 calls、12 桶 logical
bytes、canonical 精确直方图和 raw-op 精确直方图。9/9 组 all-rank、固定实际输出、
group size、H0 canonical 映射及 histogram-only 契约全部通过；Qwen3-8B TP2 smoke 的
三次重复完全一致；canonical labels 在同一 model/profile/strategy/phase 下跨 TP 完全
不变。

约 89 MB 的 `result.jsonl` 只包含各 rank 紧凑直方图而非 raw events，继续保存在远端
运行目录。当前目录归档可训练的紧凑标签、文件哈希和运行清单，避免把中间结果重复
写入 Git。
