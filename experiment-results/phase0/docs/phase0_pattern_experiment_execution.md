# Phase 0 第一阶段 Pattern 实验执行方案

## 1. 今日目标与最终模型

今天先做 **Qwen3-8B、TP=2**，验证事件级埋点和宽范围 workload；通过后再用 **DeepSeek-V3.2、TP=8** 跑代表点。

第一阶段训练通信需求预测器：

\[
\hat D(w,c)=G_\psi(w,c),\quad w=(B,L,M),\quad c=(model,parallel\_form,p)
\]

目标向量为：

\[
D(w,c)=\{N_{\phi,o,j},S_{\phi,o,j}\}
\]

- `N`：阶段×原语×消息尺度下的 group-level 通信次数。
- `S`：相同维度下的单 rank 逻辑 payload 总量。
- 底层保留精确 payload 或细粒度直方图；small/medium/large 只用于展示或由第二阶段实测拐点合并。

最终推荐混合预测模型：

\[
T_{base}=\sum_{\phi,o,j}\hat N_{\phi,o,j}\,t_{o,p,\pi}(\bar m_j)
\]

\[
\hat T_{comm}=T_{base}+H_\theta(F_w,F_c,\hat D,F_\pi,T_{base})
\]

神经网络适用的原因是 workload、模型、并行规模、消息大小、原语、拓扑和通信重叠共同形成非线性映射；但网络只校正结构化公式的残差，避免纯黑盒绕过 PatternDemand。论文通过“总bytes → bytes+calls → phase → op → message size → parallel size → topology → residual DNN”的消融证明增益。

## 2. 已核实环境

```text
节点：klingai-wlf2-ge151-node55.idchb2az2.hb2.kwaidc.com
容器：sglang-codex
源码：/sgl-workspace/sglang-src
Qwen：/media/ssd1/Qwen3-8B，context上限40960
DeepSeek：/media/ssd1/DeepSeek-V3.2，context上限163840
```

`one_batch.py`的Prefill产生第一个输出token，随后固定执行`output_len-1`次Decode，因此总输出token数等于`output_len`，不受EOS提前停止影响。

### 当前埋点缺口

现有`comm_profile.py`只保存每个rank的`phase × op → calls总数、bytes总数`，尚未记录单次payload、`event_seq`、`decode_step`、稳定`group_id`、shape/dtype和group-level去重结果。

因此当前代码只能跑smoke test和基线总量；完成事件级埋点前，不运行全部正式矩阵。

正式事件至少包含：

```json
{"run_id":"qwen_tp2_b1_l128_m4_r0","rank":0,"phase":"decode","decode_step":2,"event_seq_in_step":3,"op":"all_reduce","group_id":"tp:[0,1]","group_size":2,"input_payload_bytes":8192,"dtype":"bfloat16","tensor_shape":[1,4096]}
```

跨rank聚合键：

```text
run_id + phase + decode_step + group_id + event_seq_in_step
```

对称TP必须满足：所有rank的op/payload/序列一致，`rank_event_count==TP_SIZE`，`group_collective_count==1`。

## 3. 登录与进入容器

```bash
login=klingai-wlf2-ge151-node55.idchb2az2.hb2.kwaidc.com \
  ssh -o SendEnv=login relay
sudo nerdctl -n k8s.io exec -it sglang-codex bash
cd /sgl-workspace/sglang-src
export PYTHONPATH=/sgl-workspace/sglang-src/python:${PYTHONPATH:-}
```

## 4. Qwen3-8B TP2 Smoke Test

```bash
mkdir -p /sgl-workspace/results/qwen3_8b_tp2
CUDA_VISIBLE_DEVICES=0,1 \
python -m sglang.benchmark.one_batch \
  --model-path /media/ssd1/Qwen3-8B \
  --tp 2 --trust-remote-code --mem-fraction-static 0.85 \
  --batch-size 1 --input-len 128 --output-len 4 \
  --comm-profile --run-name qwen3-8b-tp2-smoke \
  --result-filename /sgl-workspace/results/qwen3_8b_tp2/smoke.jsonl
tail -1 /sgl-workspace/results/qwen3_8b_tp2/smoke.jsonl
```

通过条件：模型和TP2成功运行，Prefill/Decode统计存在，总输出token数为4，两个rank均有记录，无OOM/NCCL/CUDA错误；升级埋点后还要验证每个collective有2条rank-local记录但group-level count为1。

## 5. Qwen正式Workload

固定`batch=1、TP=2、repeat=3`。

### Prefill主效应

```text
L={128,512,2048,8192,16384,32768}, M=32
```

### Decode主效应

```text
L=128, M={32,128,512,2048,4096}
```

### 稀疏交互训练点

```text
(2048,128) (2048,512)
(8192,128) (8192,512) (8192,2048)
(32768,128) (32768,512) (32768,2048)
```

### 独立测试点（不进入训练）

```text
(1024,64) (4096,256) (16384,1024) (24576,1536)
```

以上均满足Qwen的40960 context上限。

## 6. Qwen主效应执行命令

### Prefill

```bash
for repeat_id in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0,1 python -m sglang.benchmark.one_batch \
    --model-path /media/ssd1/Qwen3-8B --tp 2 --trust-remote-code \
    --mem-fraction-static 0.85 --batch-size 1 \
    --input-len 128 512 2048 8192 16384 32768 --output-len 32 \
    --comm-profile --run-name qwen-tp2-prefill-r${repeat_id} \
    --result-filename /sgl-workspace/results/qwen3_8b_tp2/prefill_r${repeat_id}.jsonl
done
```

### Decode

```bash
for repeat_id in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0,1 python -m sglang.benchmark.one_batch \
    --model-path /media/ssd1/Qwen3-8B --tp 2 --trust-remote-code \
    --mem-fraction-static 0.85 --batch-size 1 --input-len 128 \
    --output-len 32 128 512 2048 4096 --comm-profile \
    --run-name qwen-tp2-decode-r${repeat_id} \
    --result-filename /sgl-workspace/results/qwen3_8b_tp2/decode_r${repeat_id}.jsonl
done
```

稀疏坐标需逐点运行，因为`one_batch.py`会对输入和输出列表做笛卡尔积。正式驱动应增加“坐标列表”参数，让多个稀疏点共享一次模型加载。

## 7. Qwen即时验收

同一`(B,L,M,p)`三次重复应满足：

```text
group-level calls完全一致
logical payload完全一致
op和消息大小分布一致
聚合前后calls/bytes严格守恒
延迟记录mean/std/CV
```

分析：

```text
L → Prefill各op的单次payload、总payload和calls
M → Decode总calls、总payload
M → Decode每token calls/bytes
L → Decode每token Pattern
workload → 消息大小分布变化
```

同一workload的三个repeat必须在同一train/validation/test split，禁止按repeat随机拆分。

## 8. DeepSeek-V3.2 TP8代表点

只有事件级埋点、group-level去重、重复稳定性和聚合守恒全部通过后才运行DeepSeek。

第一轮每点只跑1次：

```text
(128,32) (2048,128) (8192,128) (32768,32)
(128,512) (8192,512) (32768,512)
```

```bash
mkdir -p /sgl-workspace/results/deepseek_v32_tp8
while read -r input_len output_len; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m sglang.benchmark.one_batch \
    --model-path /media/ssd1/DeepSeek-V3.2 --tp 8 \
    --trust-remote-code --mem-fraction-static 0.85 --batch-size 1 \
    --input-len "${input_len}" --output-len "${output_len}" \
    --comm-profile --run-name deepseek-tp8-l${input_len}-m${output_len}-r0 \
    --result-filename /sgl-workspace/results/deepseek_v32_tp8/l${input_len}_m${output_len}_r0.jsonl
done <<'EOF'
128 32
2048 128
8192 128
32768 32
128 512
8192 512
32768 512
EOF
```

确认耗时和资源可接受后补到3次。DeepSeek很大，不先尝试TP2/TP4；并行规模TP2/4/8优先由Qwen等可容纳模型完成。

## 9. 第一阶段输出

```text
pattern_events.jsonl        # 所有rank的原始事件
pattern_group_events.jsonl  # 对齐并去重后的group-level事件
pattern_dataset.jsonl       # phase×op×细粒度消息尺度训练样本
```

训练样本特征至少包含：

```text
model, parallel_form, parallel_size, batch_size, input_len, output_len
```

标签至少包含：

```text
每个phase×op×message interval的count、payload_bytes
Prefill每输入token派生指标
Decode每输出token派生指标
rank一致性、输出长度匹配和聚合守恒标志
```

## 10. 停止条件与今日完成标准

出现输出长度不匹配、跨rank无法对齐、calls被TP size重复累计、payload语义不一致、只能得到总bytes无法恢复消息分布、逻辑Pattern无解释变化、OOM或NCCL错误时，停止扩大矩阵并先修复。

今日最低完成标准：

1. Qwen TP2 smoke test通过。
2. 事件级字段与group-level口径确认。
3. Qwen Prefill/Decode主效应至少完成一轮。
4. 生成`pattern_dataset.jsonl`样例。
5. Qwen稳定后完成DeepSeek TP8代表点至少一轮。

正式完成标准：Qwen训练点、交互点和独立测试点完成3次重复；逻辑Pattern稳定；可训练并验证`D_hat=G(w,c)`；精确消息大小可直接接入第二阶段连续代价曲线`t(o,p,pi,m)`。
