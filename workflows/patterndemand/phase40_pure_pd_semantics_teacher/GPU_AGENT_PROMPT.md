# 给纯PD GPU环境Agent的Phase40提示词

从控制端明确给出的W40-fix创建独立retry run分支，完整阅读本目录`README_CN.md`和`experiment.json`。本阶段需要同一台机器两张GPU。允许在preflight前从官方Hugging Face仓库下载固定的`Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`；不得换repo、revision、镜像模型或量化版本。下载目录必须在Git和受保护资产之外并对计算节点可见。下载完成后设置`HF_HUB_OFFLINE=1`和`TRANSFORMERS_OFFLINE=1`，正式preflight/run阶段禁止外网。

严格按README依次执行模型下载、离线preflight、run和verify。preflight会对官方config、weight index和5个safetensors分片做SHA-256精确核验。P与D各只使用一张GPU，必须显式保持`TP=1、PP=1`；Mooncake/RDMA、FCFS、4096-token chunk、关闭cache/overlap等均为实验语义，不是可调参数。允许在任何正式raw产生前诊断端口、HCA、GPU可见性、Mooncake初始化和模型路径，允许换同机GPU对或空闲端口；一旦产生首条正式raw，不得换backend、transport、模型、策略、GPU对、HCA、请求顺序或重复数。

raw profiler JSONL和P/D/router完整日志必须放在`--raw-dir`的Git外目录。不要读取或移动用户已有raw、`data/`、旧GPU实验、PID、权重或缓存。失败时保留Git外证据，使用`record_blocked.py --phase phase40`生成紧凑BLOCKED记录；不得用NIXL/TCP/fake替代，不得降低重复数，也不得删除不一致记录后继续。

PASS后只选择性添加`experiment-results/phase40_pure_pd_semantics_teacher/`，运行`verify_staging.py --phase phase40`，提交一个唯一父提交为W40的R40并push run分支。回传必须报告：W40/R40、run分支、容器与SGLang版本、模型config SHA、两张GPU及拓扑、HCA/Mooncake transport、Git外raw目录和数量、45个请求与真实chunk数、逐请求/直方图是否exact、README/summary/logs/DONE/manifest、本阶段可以和不可以得出的结论。
