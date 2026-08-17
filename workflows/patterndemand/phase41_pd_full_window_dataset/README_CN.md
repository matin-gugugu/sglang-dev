# Phase41：纯PD完整窗口teacher与开发数据

Phase40已经证明：在冻结的Qwen3-8B纯`P1→D1`环境里，真实SGLang sender记录与离线teacher可以逐请求精确一致。但Phase40只有45个请求，最大单wave为8，不能直接推出一个含几百至几千请求的完整画像窗口也能按同一口径安全生成标签。

Phase41解决这个扩展边界，同时准备下一阶段训练数据。它不训练模型。

## 三道门

1. `GateA_CONTROL_BUNDLE`：控制端从受保护raw只读重建94个既有开发画像；另外复现12个全新盲测窗口的history-only选择。Git外bundle包含开发画像的完整有序`(input_tokens, output_tokens)`，但盲测只含低维画像，绝不含完整请求。
2. `GateB_GPU_SENTINEL`：GPU端固定Qwen3-8B、FlashInfer page-size 1、Mooncake/RDMA和原子放行屏障，运行63/64/65/129请求边界及三个真实完整窗口。每个请求的sender chunk序列必须与teacher完全相同。
3. `GateC_CPU_DATASET`：只有GateB通过后，才为94个开发画像生成Hfull、32请求H0和逐bin residual；盲测12行只生成feature与H0。

任何GateB误差都会在写正式数据集前停止。GPU raw、完整日志、模型权重和transfer bundle始终在Git外。

## 为什么是最多64请求一wave

完整300秒窗口最多有2,959个请求，而冻结服务器的`max_running_requests=64`。因此本workflow不把整窗伪装成一个无限大batch，而是：

- 保持原始请求顺序；
- 连续切成最多64请求的wave；
- wave内通过`SGLANG_PD_BOOTSTRAP_BATCH_BARRIER=1`原子放行；
- wave `k`的router响应完全返回后才提交wave `k+1`；
- scheduler的4096-token预算在每个新wave开始时重置，chunk不能跨wave。

这一定义既能扩展到大窗口，也把实验口径写成了可复现的协议，而不是依赖请求到达线程的偶然时序。

## 数据隔离

开发集沿用Phase34的94个画像：75个`development_train`、19个`development_validation`，合计35,524个完整请求。Phase42只能用这两部分训练和选择。

Phase41另选12个新`blind_confirmation`画像，来自BurstGPT三个segment、每段4个，并对Phase27/28/30/31/32/33/34用过的窗口施加300秒embargo。选择只用Phase15的历史低维字段。Phase41不会把这些盲测窗口的完整请求放入bundle，也不会生成其Hfull标签；正式盲测目标留给后续封闭评估。

## 执行顺序

控制端先执行（`W41`替换成实际workflow commit）：

控制端Python需能导入`numpy`和`pandas`，与此前Phase34数据构建环境一致；这一步不需要CUDA。

```bash
python workflows/patterndemand/phase41_pd_full_window_dataset/prepare_bundle.py \
  --expected-workflow-commit W41 \
  --raw-dir /ABSOLUTE/PROTECTED/phase15/raw \
  --bundle-dir /ABSOLUTE/EXTERNAL/phase41-bundle-W41
```

核验`bundle_manifest.json`中的SHA后，通过受控文件传输把整个bundle目录交给GPU环境。不要将bundle提交到Git。

GPU Agent严格按`GPU_AGENT_PROMPT.md`执行preflight和run。成功后运行：

```bash
python workflows/patterndemand/phase41_pd_full_window_dataset/verify.py
```

只选择性暂存`experiment-results/phase41_pd_full_window_dataset/`，再用统一`verify_staging.py --phase phase41`检查后提交一个父提交严格等于W41的R41。

## PASS能说明什么

PASS说明：有界64-request wave协议在边界和三个真实完整窗口上仍与GPU精确一致；94个Qwen3纯PD开发标签及H0 residual训练表已经按该协议确定性生成；12个新盲测画像仍保持target-free。

PASS不说明：DNN已经训练、盲测已经通过、其他五模型可泛化、Mooncake物理耗时已测、placement或在线scheduler已经完成。
