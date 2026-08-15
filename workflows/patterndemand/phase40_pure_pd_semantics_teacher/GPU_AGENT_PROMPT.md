# 给纯PD GPU环境Agent的Phase40提示词

从控制端明确给出的W40-fix4创建全新的retry run分支，不要从`05501bb2`、`f8a4c283`、`c6994b97`、`376dfb97`、`e55a913b`或其他旧`BLOCKED`提交继续。完整阅读本目录`README_CN.md`和`experiment.json`。本阶段需要同一台机器两张GPU。允许在preflight前从官方Hugging Face仓库下载固定的`Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`；已经通过SHA核验的同一模型目录可复用，不得换repo、revision、镜像模型或量化版本。下载目录必须在Git和受保护资产之外并对计算节点可见。下载完成后设置`HF_HUB_OFFLINE=1`和`TRANSFORMERS_OFFLINE=1`，正式preflight/run阶段禁止外网。

严格按README设置`REPO_ROOT`和`PYTHONPATH=$REPO_ROOT/python`，使用全新的Git外raw、smoke和preflight路径，再依次执行离线preflight、run和verify。preflight会对官方config、weight index、5个safetensors分片、FlashInfer可用性及17项冻结源码/输入做精确核验，并记录repo Python、容器router、Mooncake和FlashInfer的实际来源。run固定`--attention-backend flashinfer --page-size 1 --optimistic-prefill-retries 0`、`WITH_NVIDIA_PEERMEM=0`及实验专用原子bootstrap barrier；先在独立smoke目录和端口验证1个真实P→D transport chunk，再连续两次验证`[1000,1000,1000,2000]`原子wave均精确产生5个chunk。只有合计11个page-size-1 sender chunks、两次签名一致、请求全部返回且日志没有transfer/session错误后，才允许创建正式profile/log子目录。health与内置warmup不等价于这道smoke门。

smoke失败必须原样保留Git外证据并使用新的smoke目录重试；允许在正式raw为空时换空闲端口。换同机GPU对或HCA前必须重新执行preflight生成匹配audit。每个正式wave保持一个批量`input_ids`请求，只把`rid`改为标量wave前缀，由SGLang确定性展开为`<prefix>_<batch_index>`；不得拆成串行单请求，也不得关闭`SGLANG_PD_BOOTSTRAP_BATCH_BARRIER`。P与D各只使用一张GPU，必须显式保持`TP=1、PP=1`；FlashInfer/page-size-1、Mooncake/RDMA dma-buf注册、FCFS、4096-token chunk、关闭cache/overlap等均为实验语义，不是可调参数。一旦产生首条正式raw，不得换backend、transport、模型、策略、GPU对、HCA、请求顺序或重复数。

raw profiler JSONL和P/D/router完整日志必须放在`--raw-dir`的Git外目录。不要读取或移动用户已有raw、`data/`、旧GPU实验、PID、权重或缓存。失败时保留Git外证据，使用`record_blocked.py --phase phase40`生成紧凑BLOCKED记录；不得用NIXL/TCP/fake替代，不得降低重复数，也不得删除不一致记录后继续。

PASS后只选择性添加`experiment-results/phase40_pure_pd_semantics_teacher/`，运行`verify_staging.py --phase phase40`，提交一个唯一父提交为W40-fix4的R40并push run分支。回传必须报告：W40-fix4/R40、run分支、容器与SGLang版本、模型config SHA、两张GPU及拓扑、attention backend与实际page size、HCA/Mooncake transport、Git外smoke/raw目录和数量、11-chunk smoke门全部检查、45个请求与真实chunk数、逐请求/直方图是否exact、README/summary/logs/DONE/manifest、本阶段可以和不可以得出的结论。
