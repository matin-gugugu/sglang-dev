# PatternDemand跨环境GPU执行交接：Phase36与Phase37

## 1. 交接目的

当前环境没有可用GPU。后续采用“控制环境编写并冻结workflow → GPU环境拉取并执行 → 以单个result commit回传 → 控制环境审计并ff-only同步”的方式推进。

本交接只覆盖：

1. Phase36：跨环境冻结推理与commit回传演练；
2. Phase37：PP单机P2P真实连续曲线测量。

它不改变实验总导引、Hfull teacher、fixed-draining语义、六模型H0+DNN residual checkpoint、原固定窗口或指标阈值。

## 2. 当前实验状态

- 正式分支：`experiment/pattern-demand-v0.5.15-clean`。
- 编写workflow前的冻结HEAD：`ef70a56a1b81d393a8146b97f65a9e2071352b3f`。
- Phase34六模型TP/PP正式通过；Phase35统一推理和连续代价接口PASS。
- Phase35的2,592条phase预测与Phase34冻结预测零差异。
- PP的L1/L2/L3当前仍是参数化proxy；Phase37要补的是单机物理P2P曲线。
- Phase37不重新训练，也不读取模型权重或启动模型服务。

GPU Agent开始前应阅读：

1. `experiment-results/phase32f_tp_pp_expanded_convergence_final/docs/截至目前实验结构总导引_含Phase32F状态补充.md`；
2. `experiment-results/phase34d_six_model_blind_evaluation/docs/Phase34_六模型扩展最终报告.md`；
3. `experiment-results/phase35_six_model_inference_cost_integration/docs/Phase35_六模型统一推理与拓扑代价曲线集成最终报告.md`；
4. `experiment-results/phase35_six_model_inference_cost_integration/docs/截至目前实验结构总导引_Phase35状态补充.md`；
5. `experiment-results/phase35_six_model_inference_cost_integration/docs/新会话完整交接_截至Phase35.md`；
6. 本文件；
7. 将要执行的workflow目录中的`README_CN.md`、`experiment.json`和`GPU_AGENT_PROMPT.md`。

## 3. Git交付协议

每次GPU执行都由控制环境给出一个精确base commit，记为`W`。GPU Agent必须：

1. `git fetch`并确认`W`存在；
2. 从`W`创建独立分支，例如`run/phase36-<环境>-<日期>`；
3. 运行时传入`--expected-workflow-commit W`；
4. 不修改workflow、冻结输入或已有结果；
5. 只产生一个Phase结果目录；
6. 运行`verify.py`；
7. 只选择性添加该结果目录，不得使用`git add .`；
8. 运行`verify_staging.py`；
9. 创建一个result commit `R`，使`R`以`W`为唯一父提交；
10. push run分支，回传`W`、`R`、分支、状态、目录大小、文件数和manifest结果。

控制环境收到`R`后运行：

```bash
python3 workflows/patterndemand/verify_result_commit.py \
  --phase phase36 \
  --workflow-commit W \
  --result-commit R
```

校验通过后才允许以`ff-only`把run分支推进到正式分支。Phase36和Phase37顺序执行：Phase36回传并ff-only合入后，控制环境将新的正式HEAD作为Phase37的`W`，因此不会产生两个无法ff-only合并的兄弟result commit。

## 4. Agent可以正常思考，但有三类边界

### AUTO

无需等待确认即可做：环境诊断、采集版本和拓扑、单次失败的有限重试、按合同追加高方差重复、选择每类拓扑的默认GPU对。

### RECORD_AND_CONTINUE

可以继续，但必须写`logs/decision_log.jsonl`：例如默认GPU被占用，换成同一拓扑类别的空闲GPU对；实际机器只存在部分拓扑类别；运行时波动较大但测量完整。

### BLOCKED

必须停止，不得自行绕过：冻结输入或生产源SHA不一致、GPU数量不足、NCCL/P2P不可用、必须换后端或改变测量语义、需要下载模型权重/启动服务/重新训练、raw只能放进Git仓库、需要降低合同要求。

阻塞时使用：

```bash
python3 workflows/patterndemand/record_blocked.py \
  --phase phase37 \
  --reason '简明原因' \
  --evidence-json '{"关键证据":"值"}'
```

`BLOCKED.json`可以作为result commit回传，但不能写`DONE=PASS`。

## 5. Phase36执行

资源：一张CUDA GPU；不训练；不读取teacher/target。

```bash
git checkout -b run/phase36-<env>-<date> W
python3 workflows/patterndemand/phase36_cross_environment_replay/run.py \
  --expected-workflow-commit W
python3 workflows/patterndemand/phase36_cross_environment_replay/verify.py
git add experiment-results/phase36_cross_environment_replay/
python3 workflows/patterndemand/verify_staging.py --phase phase36
git commit -m 'experiment: run Phase36 cross-environment replay on <env>'
git push <remote> HEAD
```

PASS条件：2,592条key完全一致、无target字段、TP和PP都加载三seed五折最终候选、最大相对差不超过`1e-6`、输入manifest和冻结SHA全部通过。

Phase36只证明跨环境复播和Git回传链路成立，不是新盲测，也不评价新的模型精度。

## 6. Phase37执行

资源：至少两张CUDA GPU；运行总控前应`unset CUDA_VISIBLE_DEVICES`；raw目录必须位于Git仓库外。

```bash
git checkout -b run/phase37-<env>-<date> W
unset CUDA_VISIBLE_DEVICES
RAW_DIR=/local_nvme/patterndemand_raw/phase37_<W前12位>
python3 workflows/patterndemand/phase37_pp_single_node_p2p_curve/run.py \
  --expected-workflow-commit W \
  --raw-dir "$RAW_DIR"
python3 workflows/patterndemand/phase37_pp_single_node_p2p_curve/verify.py
git add experiment-results/phase37_pp_single_node_p2p_curve/
python3 workflows/patterndemand/verify_staging.py --phase phase37
git commit -m 'experiment: measure Phase37 PP single-node P2P curves on <env>'
git push <remote> HEAD
```

如默认GPU对被占用，可在仓库外创建JSON，例如`/tmp/phase37_pairs.json`：

```json
{
  "NVLINK_NV18": [2, 3],
  "PIX": [4, 5]
}
```

然后增加：

```bash
--pair-overrides /tmp/phase37_pairs.json \
--override-reason '默认同类GPU对有其他任务，占用证据见运行前nvidia-smi'
```

只允许替换为实际拓扑矩阵中同类别的GPU对。

正式曲线语义：sender侧一个GPU tensor logical message，使用与SGLang异步`send_tensor_dict`一致的NCCL `isend/irecv` device-group原语；计时取sender和receiver完成时间的最大值。每个GPU对双向分别测量，正式拓扑类别曲线对每次repeat取双向中位数的较大值，再跨repeat取中位数，禁止挑选较快方向。CPU metadata、分配、scheduler和通信计算重叠不计入这条曲线。

payload共21点，4KiB至64MiB；30次warmup、100次计时、初始5次独立进程重复。repeat中位数CV超过15%时每轮自动增加2次，最多9次。异常不删除、不平滑。

可接受状态：

- `PASS`：所有发现并选择的拓扑类别完整，且无高方差点；
- `PASS_WITH_RUNTIME_VARIANCE`：测量完整，但到重复上限仍有CV超过15%的点；
- `PASS_WITH_LIMITED_TOPOLOGY`：机器只提供一个可测拓扑类别；
- 两种条件可同时出现。

这些状态评价测量合同，不评价最终cost。Phase38会冻结本曲线后，确定性代入Phase34消息直方图重算。

## 7. 必须提交和不得提交

每个正式结果必须包含：中文README、summary、环境、拓扑、合同副本、日志、decision log、紧凑数据、图、DONE和manifest。

不得提交：

- `data/`；
- Phase16旧GPU目录；
- Phase19 formal-v1/v2/smoke和PID；
- Phase23保护目录；
- raw逐次样本、raw profiler trace；
- 模型权重、缓存、core dump和PID；
- 任何密钥或认证信息。

Phase37的raw必须保存在仓库外；Git中的`RAW_ASSET_MANIFEST.json`只记录bundle id、文件名、大小、记录数和SHA，不记录逐次样本。

## 8. 回传消息模板

```text
Phase：Phase36/Phase37
workflow/base commit W：
result commit R：
run分支：
状态：
实际环境与GPU：
实际拓扑类别/GPU对：
结果目录：
目录大小与文件数：
核心计数/指标：
decision log摘要：
raw bundle id（Phase37）：
manifest校验：
可以得出的结论：
不可以得出的结论：
```
