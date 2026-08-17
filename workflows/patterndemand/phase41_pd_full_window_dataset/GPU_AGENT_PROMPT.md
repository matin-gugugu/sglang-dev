# Phase41 GPU Agent执行提示词

你是另一个有GPU环境中的执行Agent。用户只需告诉你正式`W41`和几个环境绝对路径；实验语义、数据规格、验收门和Git边界全部来自这个commit，不得靠聊天消息改写。

## 0. 获取准确commit并建run分支

```bash
git fetch github experiment/pattern-demand-v0.5.15-clean
git cat-file -e W41^{commit}
git switch --detach W41
git switch -c run/phase41-pd-full-window-W41
git rev-parse HEAD
```

完整阅读：

- `workflows/patterndemand/phase41_pd_full_window_dataset/experiment.json`
- `feature_contract.json`
- `README_CN.md`
- 本文件

确认bundle目录中的`bundle_manifest.json`声明同一个W41。不得从其他commit拼接脚本或结果。

## 1. 环境变量与preflight

正式preflight和run必须离线。若本机尚无Phase40核验过的官方Qwen3-8B，可在preflight之前按Phase40 `official_model_download`允许的repo、revision和五个shard hash下载到Git外持久目录；下载完成后关闭网络语义并设置：

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset CUDA_VISIBLE_DEVICES
```

选择两张空闲B200和一个ACTIVE IB device。不得占用用户指定的其他任务GPU，不得杀死未知PID。然后执行：

```bash
python workflows/patterndemand/phase41_pd_full_window_dataset/preflight.py \
  --expected-workflow-commit W41 \
  --model-path /ABSOLUTE/Qwen3-8B \
  --gpu-pair P_GPU,D_GPU \
  --ib-device IB_DEVICE \
  --bundle-dir /ABSOLUTE/phase41-bundle-W41 \
  --raw-dir /ABSOLUTE/NEW/phase41-raw-W41-attempt1 \
  --audit-output /ABSOLUTE/NEW/phase41-preflight-W41-attempt1.json
```

preflight必须PASS；raw目录会被创建为空目录。不得绕过官方模型hash、repo Python优先级、FlashInfer、Mooncake/RDMA、B200、bundle SHA或blind隔离检查。

## 2. 一条命令运行三道门

```bash
python workflows/patterndemand/phase41_pd_full_window_dataset/run.py \
  --expected-workflow-commit W41 \
  --model-path /ABSOLUTE/Qwen3-8B \
  --gpu-pair P_GPU,D_GPU \
  --ib-device IB_DEVICE \
  --bundle-dir /ABSOLUTE/phase41-bundle-W41 \
  --raw-dir /ABSOLUTE/NEW/phase41-raw-W41-attempt1 \
  --preflight-audit /ABSOLUTE/NEW/phase41-preflight-W41-attempt1.json
```

脚本先运行4个合成边界和3个真实完整窗口，共4,853请求、82个wave。只有全部GPU记录逐请求精确匹配teacher，才会在CPU上生成94个开发标签和12个target-free盲测feature。

## 3. 允许的自主诊断

`AUTO`范围：读取GPU/IB/端口/磁盘状态；选择同一台机器上另一对空闲B200；换未占用端口；增加启动等待时间；对明确的瞬时环境失败进行有限重试。

`RECORD_AND_CONTINUE`范围：若换GPU pair、IB device、端口或重试，必须使用新的Git外raw目录和新的preflight audit，保存失败attempt证据，并在最终报告说明原因。可以保留更多外部诊断日志，但不得放入Git。

绝对禁止：改变64-request wave、删除或重排请求、跨wave并发、关掉原子屏障、增加optimistic retry、换attention/transfer backend、改page/chunk/cache/overlap、降低场景或重复数、删除异常、选择“更快方向”、联网正式执行、生成blind target、训练DNN、把raw/bundle/权重/cache/PID放进Git。

若需要改变实验语义，或两次干净环境重试仍不能通过，停止并用：

```bash
python workflows/patterndemand/record_blocked.py \
  --phase phase41 --reason '...' --evidence-json '{"compact":"evidence"}'
```

## 4. 验收与单一result commit

```bash
python workflows/patterndemand/phase41_pd_full_window_dataset/verify.py
git status --short
git add experiment-results/phase41_pd_full_window_dataset/
python workflows/patterndemand/verify_staging.py --phase phase41
git commit -m 'experiment: add Phase41 pure-PD full-window dataset'
git show -s --format='%H %P' HEAD
git push github HEAD:run/phase41-pd-full-window-W41
```

禁止`git add .`。R41必须只有一个父提交且该父提交严格等于W41；commit路径只能在Phase41正式结果目录。

回传必须报告：实际状态和证据、W41和R41、run分支、容器/镜像/Python/SGLang来源、GPU pair/型号/拓扑、IB和RDMA、外部bundle/raw/preflight路径及SHA/bytes、4,853请求/82wave与GPU chunk指标、94开发画像/35,524请求/12盲测feature/0盲测target、README/summary/logs/DONE/manifest、可以和不可以得出的结论及下一步Phase42训练。
