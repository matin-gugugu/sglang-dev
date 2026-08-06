# Phase 15：Qwen3-8B 真实 trace 窗口 PatternDemand

本阶段在单节点 B200 上，对 20 个 BurstGPT/Mooncake 窗口分别以 TP=2/4/8
执行 histogram-only 通信采集，共得到 120 条“窗口 × TP × 阶段”标签。

审计结果：

- 20/20 个窗口在三个 TP 下均成功；所有 rank 的通信直方图一致；
- 实际生成长度与逐请求 `output_lens_per_request` 完全一致，没有 EOS 提前退出；
- 120/120 条 GPU 直方图均与 Qwen3-8B 的解析事件公式一致；
- logical calls、logical bytes 和 `(raw_op,payload)` 直方图跨 TP 完全不变；
- TP 只通过 ring 折算改变 equivalent bytes 和 equivalent rounds；
- 仅保存 histogram，不保存 raw events。

这证明当前第一阶段表征能够从异构输入长度和 draining Decode 的 active batch 变化中，
稳定抽取消息尺度结构。边界是：所有请求仍在同一时刻进入，arrival offset 仅用于审计，
尚不能声称已经验证真实在线交错到达和 continuous batching。

主要文件：

- `pattern_labels_all_tp.csv`：正式 PatternDemand 标签；
- `analytic_checks.csv`：GPU 直方图与解析公式逐条比对；
- `tp_invariance.csv`：logical pattern 跨 TP 不变及等效量随 TP 变化；
- `close_payload_pairs.csv`：近等总 payload、不同消息形态候选对照；
- `tp*/r0/result.jsonl`：20 个 compact histogram-only 原始结果；
- `summary.json`：机器可读审计结论。
