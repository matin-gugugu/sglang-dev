# Phase63：四模型两流并发外部验证

Phase63不再学习公式。它把R61的三个系数和全部门槛逐字节冻结，只测从未参与Phase60/R61拟合的四个模型KV布局：Qwen3-30B-A3B、Llama-3.2-3B-Instruct、Qwen2.5-14B-Instruct和Mixtral-8x7B-Instruct-v0.1。四模型通过且已冻结的R62仍为PASS，才形成六模型证据。

正式矩阵为4模型×P1D2/P2D1×L1/L2/L3×2套placement，共48个三rank shard、240个official点、480个replica点。每个shard只启动3个GPU进程：L1一台node，L2/L3两台node；所有shard可顺序运行，不要求4台node同时分配，也不加载四个模型权重。

## 冻结输入与比较

每个模型的10组payload全部来自Phase51曲线的现有knots：6组等大消息和4组一大一小消息。`payload_pair_grid.json`随W63冻结；GPU raw出现后禁止改pair、公式或阈值。

~~~text
未修正：max(C0,C1)
冻结修正：max(1, -109.8318 + 0.976478×max(C0,C1) + 0.850102×min(C0,C1))
真实值：Phase63生产Mooncake并发wave物理测量
~~~

四模型整体WAPE≤10%，每个模型整体WAPE≤10%，每个“模型×配置×拓扑”WAPE≤15%；对应signed bias也必须过门，并且整体和每个模型都严格优于未修正baseline。

## Placement与资源

优先复用Phase62的六套placement，以便只改变模型布局；若原endpoint无法重新分配，允许按调度器资产、rack和RDMA fabric元数据预先冻结同类替代endpoint。禁止先测速再挑快机器。两套replica的endpoint signature必须不同。

**全局资源红线：任意时刻只允许运行一个measurement shard，整个Phase63的峰值始终是2台node、3个GPU进程。禁止申请或保留一个4-node并发allocation。**

将inventory模板复制到Git外并填写。`A0/A1/B0/B1`是同一套placement中的四个GPU slot，不是四台node：

~~~text
L1：A0/A1/B0/B1都在同一台node；单次只启动其中3个slot。
L2/L3：A0/A1在node A，B0/B1在node B；单次仍只启动3个slot。
P1D2：A0 + B0 + B1
P2D1：A0 + A1 + B0
~~~

replica0和replica1必须顺序运行，可以在同一node/node pair上换GPU tuple。L1、L2和L3也允许作为三个独立的scheduler allocation顺序执行。为了同时覆盖“L2同rack”和“L3跨rack”，完整inventory可能先后出现3–4个不同host名字，但这些host不需要同时分配：先完成并释放L2的两台，再申请L3的两台即可。完整资源解释见`RESOURCE_ALLOCATION_CN.md`。

~~~bash
P63=workflows/patterndemand/phase63_pd_contention_four_model_external
EXT=/EXTERNAL/phase63_attempt1
mkdir -p "$EXT"
cp "$P63/topology_inventory.example.json" "$EXT/topology_inventory.json"
# 按资产元数据填写真实host/rack/fabric/GPU/HCA；优先照抄Phase62 endpoint。
python3 "$P63/make_topology_plan.py" \
  --inventory "$EXT/topology_inventory.json" \
  --output "$EXT/topology_plan.json"
~~~

## GPU执行

在能够看到目标GPU和RDMA设备的`lmsysorg/sglang:v0.5.15`容器内，以仓库源码优先并冻结离线RDMA环境：

~~~bash
W63=<控制端公布的完整W63>
RAW=/EXTERNAL/phase63_raw_attempt1
mkdir -p "$RAW"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MOONCAKE_PROTOCOL=rdma WITH_NVIDIA_PEERMEM=0 SGLANG_DISAGG_STAGING_BUFFER=0
unset MC_FORCE_TCP MC_FORCE_MNNVL MC_INTRANODE_NVLINK SGLANG_MOONCAKE_CUSTOM_MEM_POOL

python3 "$P63/preflight.py" \
  --expected-workflow-commit "$W63" \
  --container-image lmsysorg/sglang:v0.5.15 \
  --topology-plan "$EXT/topology_plan.json" \
  --raw-dir "$RAW" \
  --audit-output "$EXT/preflight_audit.json"
~~~

对48个measurement分别执行repeat 0–4。渲染器每次输出3条rank命令，这三条是同一个shard内部必须并发的rank；三条全部退出后才能启动任何下一repeat、replica、配置、模型或拓扑shard：

~~~bash
python3 "$P63/render_launch_commands.py" \
  --topology-plan "$EXT/topology_plan.json" \
  --raw-dir "$RAW" \
  --measurement-id <plan中的measurement_id> \
  --repeat-id 0 \
  --master-addr <三rank均可访问的rank0地址> \
  --master-port <本shard未占用端口>
~~~

先做5次，再机械检查：

~~~bash
python3 "$P63/raw_status.py" \
  --topology-plan "$EXT/topology_plan.json" \
  --raw-dir "$RAW" > "$EXT/raw_status.json"
~~~

若`complete=false`，只按`missing`和`needs_extra`补合同要求的repeat到7或9；不得删异常、挑快replica或少跑。直到`complete=true`后才生成紧凑结果并验收：

~~~bash
RESULT=experiment-results/phase63_pd_contention_four_model_external
python3 "$P63/run.py" \
  --expected-workflow-commit "$W63" \
  --topology-plan "$EXT/topology_plan.json" \
  --preflight-audit "$EXT/preflight_audit.json" \
  --raw-dir "$RAW" \
  --output-dir "$RESULT"

python3 "$P63/verify.py" --output-dir "$RESULT"
~~~

只有`verify.py`通过后才能按`commit_allowlist.txt`逐项`git add`，禁止`git add .`。结果commit必须是W63的唯一子提交并push run分支；raw JSONL、模型权重、缓存、PID和密钥继续留在Git外。

若四模型门失败，必须保留真实失败。Phase63标签不得用于重拟合；后续若要改公式，必须重新建立development/freeze/new-blind链。
