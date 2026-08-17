# Phase40：纯PD语义与Hfull teacher基础闭环

最终状态：`PASS`。本阶段只运行纯`P1→D1`，P和D内部均为`TP=1、PP=1`；固定FlashInfer attention、page size 1、Mooncake/RDMA、FCFS、4096-token chunk并关闭cache与overlap。正式raw前的独立P→D smoke已完成1个真实transport sender chunk及两次精确的多请求原子放行探针，且没有传输错误。

共执行`45`个请求、`75`个真实sender-side逻辑KV chunk；CPU teacher生成`75`个chunk，逐请求精确匹配`45/45`。calls、logical bytes和12-bin直方图误差均为0，五个场景的三次重复直方图完全一致。

运行时KV字节/page为`[147456]`，与模型结构公式精确一致；没有Mamba/SWA等额外state payload。raw profiler JSONL和完整P/D/router日志保存在Git外，Git只归档其SHA、数量和聚合结果。

该结果建立了冻结Qwen3纯PD语义的GPU证据与离线teacher基础，不代表低维画像预测器已经训练、六模型已经泛化、Mooncake物理时间曲线已经完成，也不包含placement或线上scheduler结论。
