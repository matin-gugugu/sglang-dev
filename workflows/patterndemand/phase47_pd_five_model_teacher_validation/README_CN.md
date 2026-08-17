# Phase47 workflow：纯PD其余五模型teacher语义验证

Phase40/41已经用Qwen3-8B证明纯`P1→D1`的fixed-draining GPU发送记录与scheduler-faithful teacher精确一致；Phase46又证明Qwen3-8B的低维`H0+DNN residual`在300个fresh blind画像上优于H0。Phase47不继续训练，而是先补齐冻结六模型阵容中另外五个模型的GPU语义证据，防止Phase48直接批量生成六模型Hfull时把错误的KV公式、MLA page或backend带入标签。

本阶段总共运行5个模型、每模型45个请求、合计225个请求。五个模型顺序复用同一对GPU，不同时加载。每个模型正式raw前先做独立的真实Mooncake传输和两次原子wave smoke；随后运行5个边界场景×3次重复，并逐请求核对GPU chunk、teacher chunk、逻辑字节和12-bin直方图。

- `DeepSeek-V2-Lite`：`TRTLLM MLA + page64`，使用B200/SM100原生支持的MLA执行点；
- 其余四模型：`FlashInfer + page1`，标准K/V公式；
- 全部固定`P TP=PP=1`、`D TP=PP=1`、BF16 KV cache（SGLang CLI固定使用其接受的`bf16`拼写）、Mooncake/RDMA、dma-buf、无staging、FCFS、4096-token chunk、整wave原子放行、关闭radix cache和overlap；
- 不训练、不测物理时间、不做placement或scheduler资源决策。

## 远程执行

先在正式实验之外联网获取固定revision。Llama需要远程操作者已有合法HF授权；token只通过HF环境/登录状态读取，禁止放进命令、日志或Git。

```bash
P47=workflows/patterndemand/phase47_pd_five_model_teacher_validation
python3 "$P47/prepare_models.py" \
  --model-root /PERSISTENT/phase47_models \
  --download missing \
  --model-map-output /EXTERNAL/phase47/model_map.json \
  --inventory-output /EXTERNAL/phase47/acquisition_inventory.json
```

然后关闭网络语义并执行一次preflight和一次run：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

python3 "$P47/preflight.py" \
  --expected-workflow-commit W47 \
  --model-map /EXTERNAL/phase47/model_map.json \
  --gpu-pair 0,1 --ib-device mlx5_0 \
  --raw-root /EXTERNAL/phase47/raw_attempt1 \
  --smoke-root /EXTERNAL/phase47/smoke_attempt1 \
  --audit-output /EXTERNAL/phase47/preflight_attempt1.json

python3 "$P47/run.py" \
  --expected-workflow-commit W47 \
  --model-map /EXTERNAL/phase47/model_map.json \
  --gpu-pair 0,1 --ib-device mlx5_0 \
  --raw-root /EXTERNAL/phase47/raw_attempt1 \
  --smoke-root /EXTERNAL/phase47/smoke_attempt1 \
  --preflight-audit /EXTERNAL/phase47/preflight_attempt1.json

python3 "$P47/verify.py"
```

失败的smoke/raw必须保留，重试使用新的外置目录；不得删除异常、换backend/page/transport、降低重复数或选择性报告模型。只有五模型全部PASS才生成正式结果目录。确属环境阻塞时使用`record_blocked.py --phase phase47`保存紧凑证据。

提交时只能选择性添加`experiment-results/phase47_pd_five_model_teacher_validation/`，先运行`verify_staging.py --phase phase47`。模型、缓存、raw JSONL、完整服务日志、HF token和PID均不得进入Git。
