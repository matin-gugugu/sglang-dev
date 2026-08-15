# 给多节点GPU环境Agent的Phase39提示词

从交接方指定的W39创建独立run分支，完整阅读本目录`README_CN.md`与`experiment.json`。不要从旧W36/W37运行。Phase39需要GPU通信microbenchmark，但不需要模型权重、checkpoint、模型服务、推理或训练。

先在仓库外准备：

```bash
cp workflows/patterndemand/phase39_tp_pp_l1_l3_physical_placement_validation/topology_inventory.example.json /EXTERNAL/phase39_topology_inventory.json
# 将所有示例host/rack/NIC/GPU/证据替换为真实分配，并在任何benchmark前保存。
python3 workflows/patterndemand/phase39_tp_pp_l1_l3_physical_placement_validation/make_topology_plan.py \
  --inventory /EXTERNAL/phase39_topology_inventory.json \
  --output /EXTERNAL/phase39_topology_plan.json
python3 workflows/patterndemand/phase39_tp_pp_l1_l3_physical_placement_validation/preflight.py \
  --expected-workflow-commit W39 \
  --topology-plan /EXTERNAL/phase39_topology_plan.json \
  --raw-dir /EXTERNAL/phase39_raw \
  --audit-output /EXTERNAL/phase39_preflight.json
```

逐measurement完成repeat 0–4。对每次分布式启动，先生成逐节点命令：

```bash
python3 workflows/patterndemand/phase39_tp_pp_l1_l3_physical_placement_validation/render_launch_commands.py \
  --topology-plan /EXTERNAL/phase39_topology_plan.json \
  --measurement-id MEASUREMENT_ID --repeat-id REPEAT_ID \
  --raw-dir /EXTERNAL/phase39_raw --master-addr MASTER_ADDR --master-port MASTER_PORT
```

输出是每个host一条argv数组。必须通过目标环境的既有多节点启动机制同时执行全部节点命令；不得把同一多节点case拆成互不相干的单节点测试。环境诊断、有限重试、端口替换和不改变placement的重新启动属于`AUTO`。若必须换backend、改拓扑标签、删payload或改变rank mapping，必须`BLOCKED`。

完成最低重复后：

```bash
python3 workflows/patterndemand/phase39_tp_pp_l1_l3_physical_placement_validation/raw_status.py \
  --topology-plan /EXTERNAL/phase39_topology_plan.json --raw-dir /EXTERNAL/phase39_raw
```

只对`needs_extra_repeats`追加2个repeat一轮，最多9次。禁止删除异常或挑选较快方向/placement。若到9次仍高方差，保留全部数据并让正式状态携带variance标记。

全部raw合同满足后运行：

```bash
python3 workflows/patterndemand/phase39_tp_pp_l1_l3_physical_placement_validation/run.py \
  --expected-workflow-commit W39 \
  --topology-plan /EXTERNAL/phase39_topology_plan.json \
  --raw-dir /EXTERNAL/phase39_raw \
  --preflight-audit /EXTERNAL/phase39_preflight.json
python3 workflows/patterndemand/phase39_tp_pp_l1_l3_physical_placement_validation/verify.py
```

若必测矩阵、拓扑证据或生产primitive不可满足，使用`record_blocked.py --phase phase39`记录紧凑证据。成功后按`README_CN.md`的两条选择性`git add`命令添加结果：普通结果目录加一次，并只对README列出的4个紧凑CSV和`logs/runtime.log`明确使用`-f`。禁止`git add .`，也禁止对整个结果目录使用`git add -f`。运行`verify_staging.py --phase phase39`，确认结果树和暂存区文件集合完全一致后，提交一个唯一父提交为W39的R39并push run分支。

回传必须包含W39、R39、run分支、24个measurement及其host/rack/GPU/NIC映射、raw外部目录与文件/记录数、12条曲线、方差状态、TP/PP L1/L2/L3 cost、placement agreement/regret、README/summary/logs/DONE/manifest，以及可得与不可得的结论。
