# Phase40：纯PD语义与Hfull teacher基础闭环

Phase40只做纯PD：一个单GPU Prefill实例向一个单GPU Decode实例传KV，P和D内部均为`TP=1、PP=1`。它不是把现有TP/PP模型换个名字，也不开始训练六模型PD预测器。

本阶段回答三个基础问题：SGLang在固定纯PD策略下到底把哪些KV块从P发到D；一个拓扑无关“消息”和“逻辑字节”应如何计数；完整有序请求窗口能否由独立CPU teacher精确恢复真实GPU记录。若这三件事没有精确闭环，后续低维画像训练会把采集或teacher错误学进residual。

## 冻结语义

- P和D各使用一张GPU，`TP=PP=1`；不允许在任何一侧引入TP/PP。
- 固定SGLang `mooncake` backend和RDMA transport，不允许AUTO换成NIXL、TCP或fake。
- 固定FCFS、fixed-draining、4096-token chunk、4096 max-prefill budget、page size 1、关闭radix cache、decode radix cache、dynamic chunking和overlap schedule。
- 只计Prefill sender在TP/CP过滤后提交的KV chunk。每次`CommonKVSender.send`是一条逻辑消息；逻辑字节与SGLang `KVTransferMetric`的per-page K/V字节口径一致。
- 不计bootstrap、metadata、receiver副本、Mooncake内部逐层descriptor、网卡packet、header和时间。

采集钩子默认完全关闭。只有设置`SGLANG_PD_COMM_PROFILE_DIR`后才把无tensor内容的request/chunk元数据写到Git外JSONL。正式Git结果只保存SHA、数量、聚合12-bin直方图和精确对齐表。

## 工作量为何是这个大小

五个请求wave覆盖：小包、恰好一个chunk、跨界一个token、批内剩余budget造成的拆分、短中长混合及连续多chunk。每个wave独立重复3次，共45个请求。它足以检验字节公式、FCFS切分、边界和重复确定性，又没有提前扩成六模型训练、物理曲线或线上调度器阶段。

## 运行前提与命令

必须从交接方给出的W40-fix2创建独立retry run分支，不能从任何旧`BLOCKED`提交继续。模型固定为官方`Qwen/Qwen3-8B`的revision `b968826d9c46dd6066d109eabc6255188de91218`。允许在preflight前联网下载一次，已核验通过的同revision模型目录可以直接复用；下载完成后必须切换为离线模式，正式preflight和GPU运行不得继续联网。下载目录必须位于计算节点可见的持久化存储且在Git、raw目录、其他用户缓存和受保护实验资产之外。`--gpu-pair`使用物理GPU编号，`--ib-device`使用已验证的RDMA HCA。

推荐命令如下；`/PERSISTENT/MODELS`应替换为本环境自己的持久化模型目录：

```bash
MODEL_DIR=/PERSISTENT/MODELS/Qwen3-8B-b968826d
mkdir -p "$MODEL_DIR"
hf download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir "$MODEL_DIR"

# 从这里开始正式实验必须离线；preflight会核对config、index和5个权重分片的官方SHA-256。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 固定Python侧使用当前W40-fix2仓库源码；router和Mooncake仍可来自容器包，实际来源会写入审计。
REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="$REPO_ROOT/python"
```

如果环境只有旧版CLI，可用等价的`huggingface-cli download`，但repo、revision和落盘目录不得改变。不要把token写进命令、日志或Git；该公开模型正常情况下不需要token。

```bash
unset CUDA_VISIBLE_DEVICES
python3 workflows/patterndemand/phase40_pure_pd_semantics_teacher/preflight.py \
  --expected-workflow-commit W40_FIX2 \
  --model-path "$MODEL_DIR" \
  --gpu-pair 0,1 \
  --ib-device mlx5_X \
  --raw-dir /EXTERNAL/phase40_raw_fix2 \
  --audit-output /EXTERNAL/phase40_preflight_fix2.json

python3 workflows/patterndemand/phase40_pure_pd_semantics_teacher/run.py \
  --expected-workflow-commit W40_FIX2 \
  --model-path "$MODEL_DIR" \
  --gpu-pair 0,1 \
  --ib-device mlx5_X \
  --raw-dir /EXTERNAL/phase40_raw_fix2 \
  --preflight-audit /EXTERNAL/phase40_preflight_fix2.json

python3 workflows/patterndemand/phase40_pure_pd_semantics_teacher/verify.py
```

端口冲突时可以在第一次正式请求前用`run.py`的端口参数整体替换；GPU对也可在preflight前换为同一台机器上的另一对。首条正式raw出现后，不得换backend、transport、模型、策略、GPU对、HCA、chunk、请求顺序或删除异常；失败应保存Git外证据并按`BLOCKED`回传。

每个wave仍通过一个批量`input_ids`请求提交。为兼容router，请求只携带一个标量`rid`前缀；仓库版SGLang按批内顺序将其展开为`<prefix>_0`、`<prefix>_1`等，teacher使用完全相同的确定性ID。不得把wave拆成串行单请求，这个修复不改变fixed-draining、批内顺序或`packed_remainder`语义。

## Git边界

成功后只允许：

```bash
git add -- experiment-results/phase40_pure_pd_semantics_teacher/
python3 workflows/patterndemand/verify_staging.py --phase phase40
```

禁止`git add .`，禁止添加raw JSONL、完整server log、权重、缓存、PID或`data/`。R40必须是W40-fix2的单一父提交，并只改允许的正式结果目录。

## PASS能说明什么

PASS说明：在这个冻结的P1-D1 Qwen3/Mooncake语义上，真实GPU发送记录与完整请求列表teacher逐请求、逐chunk、逐bin精确一致，可作为后续大规模Hfull标签生成的基础。

PASS不说明：低维画像PD预测器已经训练完成、六模型已泛化、Mooncake物理时间已建曲线、placement或线上scheduler已解决。那些必须在后续Phase单独建立合同。
