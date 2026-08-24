# Phase62：P1D2/P2D1 contention fresh-blind

Phase62把R61公式完全冻结，只测Phase60预留但从未测过的20个payload pair。它不训练、不调倍率、不改阈值；GPU结果第一次出现后只能机械验收。

正式矩阵为2模型×2配置×L1/L2/L3×2套fresh placement，共24个三rank shard、120个official点和240个replica点。每个shard仍只启动3个GPU进程：L1一台node，L2/L3两台node；全部shard允许顺序运行，不要求4台node同时分配。

Fresh placement有两层约束：

1. Phase62全部host/GPU/HCA endpoint tuple不得与Phase60任何endpoint重复；
2. 每种L1/L2/L3至少一套placement的host signature也必须是Phase60未见的。

先复制topology_inventory.example.json到Git外，用资产和rack/fabric元数据填写，禁止根据测速挑placement：

~~~bash
P62=workflows/patterndemand/phase62_pd_contention_fresh_blind
EXT=/EXTERNAL/phase62_attempt1
mkdir -p "$EXT"
cp "$P62/topology_inventory.example.json" "$EXT/topology_inventory.json"
python3 "$P62/make_topology_plan.py" --inventory "$EXT/topology_inventory.json" --output "$EXT/topology_plan.json"
~~~

在能够看到目标GPU和RDMA设备的`lmsysorg/sglang:v0.5.15`容器内，以仓库源码优先并冻结离线RDMA环境，然后在exact W62运行preflight：

~~~bash
W62=<控制端公布的完整W62>
RAW=/EXTERNAL/phase62_raw_attempt1
mkdir -p "$RAW"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MOONCAKE_PROTOCOL=rdma WITH_NVIDIA_PEERMEM=0 SGLANG_DISAGG_STAGING_BUFFER=0
unset MC_FORCE_TCP MC_FORCE_MNNVL MC_INTRANODE_NVLINK SGLANG_MOONCAKE_CUSTOM_MEM_POOL

python3 "$P62/preflight.py" \
  --expected-workflow-commit "$W62" \
  --container-image lmsysorg/sglang:v0.5.15 \
  --topology-plan "$EXT/topology_plan.json" \
  --raw-dir "$RAW" \
  --audit-output "$EXT/preflight_audit.json"
~~~

对24个measurement分别执行repeat 0–4。下例先渲染一个measurement、一个repeat的3条rank命令；必须同时启动三条，等待三条全部退出后再进入下一repeat：

~~~bash
python3 "$P62/render_launch_commands.py" \
  --topology-plan "$EXT/topology_plan.json" \
  --raw-dir "$RAW" \
  --measurement-id <plan中的measurement_id> \
  --repeat-id 0 \
  --master-addr <三rank均可访问的rank0地址> \
  --master-port <本shard未占用的端口>
~~~

所有measurement完成5次后先做机械状态审计。`raw_status.py`只按合同中的方差规则允许追加到7次或9次，不能人工挑异常、删慢样本或少跑：

~~~bash
python3 "$P62/raw_status.py" \
  --topology-plan "$EXT/topology_plan.json" \
  --raw-dir "$RAW" > "$EXT/raw_status.json"
~~~

若状态为`NEEDS_MORE_REPEATS`，只补`required_repeat_count`尚缺的repeat，再重跑状态审计。全部为`READY`后才生成紧凑正式结果并验收：

~~~bash
RESULT=experiment-results/phase62_pd_contention_fresh_blind
python3 "$P62/run.py" \
  --expected-workflow-commit "$W62" \
  --container "$CONTAINER" \
  --topology-plan "$EXT/topology_plan.json" \
  --preflight-audit "$EXT/preflight_audit.json" \
  --raw-dir "$RAW" \
  --output-dir "$RESULT"

python3 "$P62/verify.py" \
  --output-dir "$RESULT"
~~~

只有`verify.py`通过后才能按`commit_allowlist.txt`逐项`git add`，禁止`git add .`。结果commit必须是W62的唯一子提交并push run分支；raw目录继续留在Git外。

如果fresh环境无法满足GPU/HCA tuple不重复或每种拓扑的新host signature约束，记录`BLOCKED`并换资源；不得退回Phase60旧placement冒充fresh blind。Git只允许提交紧凑结果目录；raw JSONL、模型权重、缓存、PID和密钥禁止入Git。

主要比较：

~~~text
未修正：max(C0,C1)
冻结修正：max(1, -109.8318 + 0.976478×max(C0,C1) + 0.850102×min(C0,C1))
真实值：fresh-blind concurrent wave
~~~

冻结修正必须整体WAPE≤10%、每个配置×拓扑≤15%，对应bias也过门，并严格优于未修正baseline。失败时保留真实失败，不得用blind标签重训。
